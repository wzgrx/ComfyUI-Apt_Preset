# 标准库
import math
import os
import sys
from dataclasses import dataclass
import enum
from enum import Enum

# 第三方库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d
from torch.nn.modules.utils import _pair
import numpy as np
from tqdm import tqdm
import comfy
from comfy import samplers
import comfy.model_management as mm
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy_extras import nodes_custom_sampler
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked
from comfy.model_patcher import ModelPatcher
from comfy.cli_args import args
import folder_paths
import node_helpers
import latent_preview
import nodes
from nodes import CLIPTextEncode, common_ksampler, VAEDecode, VAEEncode, ImageScale, KSampler, SetLatentNoiseMask, KSamplerAdvanced,InpaintModelConditioning
from einops import rearrange
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
import matplotlib
from PIL import Image, ImageFilter
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing
from typing import Any, Dict, Optional, Tuple, Union, cast
from comfy.samplers import KSAMPLER
import torchvision.transforms as transforms
import types
from comfy.utils import load_torch_file
from comfy import lora
import functools
import comfy.model_management
import comfy.utils
from functools import partial
from comfy.model_base import Flux

from ..main_unit import *

try:
    import cv2
except ImportError:
    cv2 = None


#matplotlib.use('Agg')

#region-----------总------------------------------

SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]




class basic_Ksampler_batch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.1,}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"


    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        latent_samples = latent["samples"]
        # Convert to float32 if not already
        if latent_samples.dtype != torch.float32:
            latent_samples = latent_samples.to(torch.float32)
        num_samples = latent_samples.shape[0]  #latent的帧数

        denoise = adapt_to_batch(denoise, num_samples)
        steps = adapt_to_batch(steps, num_samples)
        cfg = adapt_to_batch(cfg, num_samples)

        denoised_samples = []
        for i in range(latent_samples.shape[0]):
            current_latent = {"samples": latent_samples[i].unsqueeze(0)}
            current_denoise = np.clip(denoise[i], 0.0, 1.0)
            result = common_ksampler(model, seed, steps[i], cfg[i], sampler_name, scheduler, positive, negative, current_latent, denoise=current_denoise)
            denoised_samples.append(result[0]["samples"])
        final_denoised_samples = torch.cat(denoised_samples, dim=0)
        final_denoised_latent = {"samples": final_denoised_samples}

        # Convert to float32 before decoding
        final_denoised_latent["samples"] = final_denoised_latent["samples"].to(torch.float32)


        return (final_denoised_latent,)



#region------------------------ksampler_dual双区域采样------------------------




def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith('AYS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith('GITS[coeff='):
        sigmas = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']().get_sigmas(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    elif scheduler == 'LTXV[default]':
        sigmas = nodes.NODE_CLASS_MAPPINGS['LTXVScheduler']().get_sigmas(20, 2.05, 0.95, True, 0.1)[0]
    else:
        sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_3m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_3m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    else:
        return comfy.samplers.sampler_object(sampler_name)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


# modified version of SamplerCustom.sample
def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None):
    latent = latent_image
    latent_image = latent["samples"]

    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out['samples'] = latent_image

    if noise is None:
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(out)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    preview_callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    if callback is not None:
        def touched_callback(step, x0, x, total_steps):
            callback(step, x0, x, total_steps)
            preview_callback(step, x0, x, total_steps)
    else:
        touched_callback = preview_callback

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    device = mm.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != 'NegativePlaceholder':
        # This way is incompatible with Advanced ControlNet, yet.
        # guider = comfy.samplers.CFGGuider(model)
        # guider.set_conds(positive, negative)
        # guider.set_cfg(cfg)
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask, callback=touched_callback,
                                             disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = nodes_custom_sampler.Guider_Basic(model)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(comfy.model_management.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised


# When sampling one step at a time, it mitigates the problem. (especially for _sde series samplers)
def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None, noise=None, callback=None, scheduler_func=None):

    if scheduler_func is not None:
        total_sigmas = scheduler_func(model, sampler_name, steps)
    else:
        if sampler_opt is None:
            total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
        else:
            total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            if latent_image is not None:
                return latent_image
            else:
                return {'samples': torch.zeros_like(noise)}

    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image
    
    res = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, sigma_ratio=1.0, sampler_opt=None, noise=None, scheduler_func=None):
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                            start_at_step, end_at_step, False, scheduler_func=scheduler_func)


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor=1.0, noise=None, scheduler_func=None):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        # Use separated_sample instead of KSampler for `AYS scheduler`
        # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]

        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + steps

        refined_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step, False,
                                          sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True,
                                       sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False,
                                          sigma_ratio=sigma_factor, scheduler_func=scheduler_func)

    return refined_latent



class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=None):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.scheduler_func = scheduler_func

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=self.scheduler_func)


class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed to use for generating CPU noise for sampling."}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "total sampling steps"}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "classifier free guidance value"}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned."}),
                                "basic_pipe": ("BASIC_PIPE", {"tooltip": "basic_pipe input for sampling"})
                            },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC", {"tooltip": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored."}),
                    }
                }

    OUTPUT_TOOLTIPS = ("sampler wrapper. (Can be used when generating a regional_prompt.)",)

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=scheduler_func_opt)
        return (sampler, )


class TwoSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "latent_image": ("LATENT", {"tooltip": "input latent image"}),
                     "base_sampler": ("KSAMPLER", {"tooltip": "Sampler to apply to the region outside the mask."}),
                     "mask_sampler": ("KSAMPLER", {"tooltip": "Sampler to apply to the masked region."}),
                     "mask": ("MASK", {"tooltip": "region mask"})
                     },
                }

    OUTPUT_TOOLTIPS = ("result latent", )

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(latent_image, base_sampler, mask_sampler, mask):
        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample(latent_image)

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)

        del new_latent_image['noise_mask']

        return (new_latent_image, )

#--------------------------------chx_Ksampler_dual_area----------------------


class chx_Ksampler_dual_area:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "image_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, }),
                "mask_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, }),
                "image_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            
            "optional": {

            "model_img": ("MODEL",),
            "model_mask": ("MODEL",),
            "prompt_img": ("CONDITIONING",),
            "prompt_mask": ("CONDITIONING",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "smoothness":("INT", {"default": 1,  "min":0, "max": 150, "step": 1,"display": "slider"}),
            "image_pos": ("STRING", {"default": "","multiline": True}),
            "mask_pos": ("STRING", {"default": "","multiline": True}),

            },
            
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE","MASK")
    RETURN_NAMES = ("context", "image","mask")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"


    def sample(self, seed, image_denoise, mask_denoise, image_cfg, mask_cfg, image=None, mask=None,  model_img=None,model_mask=None, prompt_img=None,prompt_mask=None,
                context=None,scheduler_func_opt=None, mask_pos="",image_pos="",smoothness=1 ):

        vae = context.get("vae")
        steps = context.get("steps")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        negative= context.get("negative")

        clip = context.get("clip")
        
        if image_pos != None and image_pos != '':       
            positive1, =CLIPTextEncode().encode(clip, image_pos)  
        
        if mask_pos != None and mask_pos != '':       
            positive2, =CLIPTextEncode().encode(clip, mask_pos)  


        if model_img!= None:
            model1=model_img
        else:
            model1=context.get("model")
        
        if model_mask!= None:
            model2=model_mask
        else:
            model2=context.get("model")

        if prompt_img== None:
            prompt_img =positive1
        else:
            prompt_img=context.get("positive")

        if prompt_mask== None:
            prompt_mask = positive2
        else:
            prompt_mask=context.get("positive")

        latent = VAEEncode().encode(vae, image)[0]

        image_sampler = KSamplerWrapper(model1, seed, steps, image_cfg, sampler, scheduler, prompt_img, negative, image_denoise, scheduler_func=scheduler_func_opt)
        mask_sampler  = KSamplerWrapper(model2, seed, steps, mask_cfg, sampler, scheduler, prompt_mask, negative, mask_denoise, scheduler_func=scheduler_func_opt)
        
        
        #----------------------add smooth
        mask=tensor2pil(mask)
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)
        
        latent = TwoSamplersForMask().doit(latent, image_sampler, mask_sampler, mask)[0]
        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context,  latent=latent, images=output_image,)
        
        
        return  (context, output_image, mask)


#endregion------------------------ksampler_dual双区域采样------------------------



#region------------------------deforum_ksampler------------------------

class chx_ksampler_Deforum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "frame": ("INT", {"default": 16}),
                "x": ("INT", {"default": 15, "step": 1, "min": -4096, "max": 4096}),
                "y": ("INT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 0.98, "min": 0.001, "step": 0.01}),
                "angle": ("INT", {"default": -1, "step": 1, "min": -360, "max": 360}),
                "denoise_min": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step":0.01}),
                "denoise_max": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step":0.01}),
                "easing_type": (list(easing_functions.keys()), ),
            },
            "optional": {
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
                "end_frame_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/chx_ksample"

    def apply(self, image, frame, seed, x, y, zoom, angle, denoise_min, denoise_max, easing_type, model=None, context=None, end_frame_image=None):
        if model is None:
            model = context.get("model")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler_name = context.get("sampler")
        scheduler = context.get("scheduler")
        vae = context.get("vae")
        positive = context.get("positive")
        negative = context.get("negative")
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()

        res = [image]
        pbar = comfy.utils.ProgressBar(frame)

        if end_frame_image is not None:
            end_frame_latent = vaeencode.encode(vae, end_frame_image)[0]

            for i in tqdm(range(frame)):
                ratio = (i + 1) / frame
                denoise = (denoise_max - denoise_min) * apply_easing(ratio, easing_type)  + denoise_min
                image = image_2dtransform(image, x[i], y[i], zoom[i], angle[i], 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                blended_latent = latent["samples"] * (1 - ratio) + end_frame_latent["samples"] * ratio
                latent["samples"] = blended_latent
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)
        else:
            for i in tqdm(range(frame)):
                denoise = (denoise_max - denoise_min) * apply_easing((i+1)/frame, easing_type)  + denoise_min
                image = image_2dtransform(image, x[i], y[i], zoom[i], angle[i], 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)
        if res[0].size() != res[-1].size():
            res = res[1:]

        res = torch.cat(res, dim=0)
        return (res, )


class chx_ksampler_Deforum_math:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "frame": ("INT", {"default": 16}),
                "x": ("STRING", {"multiline": False, "default": "15", "tooltip": "可以用当前帧编号i计算"} ), 
                "y": ("STRING", {"multiline": False, "default": "0", }),
                "zoom": ("STRING", {"multiline": False,"default": "0.98", }), 
                "angle": ("STRING", {"multiline": False, "default": "-1",}),
                "denoise_min": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step":0.01}),
                "denoise_max": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step":0.01}),
                "easing_type": (list(easing_functions.keys()), ),
            },
            "optional": {
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
                "end_frame_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/chx_ksample"

    def apply(self, image, frame, seed, x, y, zoom, angle, denoise_min, denoise_max, easing_type, model=None, context=None, end_frame_image=None):
        if model is None:
            model = context.get("model")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler_name = context.get("sampler")
        scheduler = context.get("scheduler")
        vae = context.get("vae")
        positive = context.get("positive")
        negative = context.get("negative")
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()

        res = [image]
        pbar = comfy.utils.ProgressBar(frame)

        if end_frame_image is not None:
            end_frame_latent = vaeencode.encode(vae, end_frame_image)[0]

            for i in tqdm(range(frame)):
                ratio = (i + 1) / frame
                x_str = str(x)
                y_str = str(y)
                zoom_str = str(zoom)
                angle_str = str(angle)

                # 动态计算 x 和 y 的值
                X = int(eval(x_str, {'i': i}))
                Y = int(eval(y_str, {'i': i}))
                angle = int(eval(angle_str, {'i': i}))
                zoom = float(eval(zoom_str, {'i': i}))

                # 检查 zoom 值，确保其为正数
                if zoom <= 0:
                    zoom = 0.001  # 设置一个最小的正数

                denoise = (denoise_max - denoise_min) * apply_easing(ratio, easing_type)  + denoise_min
                image = image_2dtransform(image, X, Y, zoom, angle, 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                blended_latent = latent["samples"] * (1 - ratio) + end_frame_latent["samples"] * ratio
                latent["samples"] = blended_latent
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)
        else:
            for i in tqdm(range(frame)):
                denoise = (denoise_max - denoise_min) * apply_easing((i+1)/frame, easing_type)  + denoise_min
                x_str = str(x)
                y_str = str(y)
                zoom_str = str(zoom)
                angle_str = str(angle)

                # 动态计算 x 和 y 的值
                X = int(eval(x_str, {'i': i}))
                Y = int(eval(y_str, {'i': i}))
                angle = int(eval(angle_str, {'i': i}))
                zoom = float(eval(zoom_str, {'i': i}))

                # 检查 zoom 值，确保其为正数
                if zoom <= 0:
                    zoom = 0.001  # 设置一个最小的正数

                image = image_2dtransform(image, X, Y, zoom, angle, 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)


        if res[0].size() != res[-1].size():
            res = res[1:]

        res = torch.cat(res, dim=0)
        return (res, )


class chx_ksampler_Deforum_sch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "frame": ("INT", {"default": 16}),
                "x": ("INT", {"default": 15, "step": 1, "min": -4096, "max": 4096}),
                "y": ("INT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 0.98, "min": 0.001, "step": 0.01}),
                "angle": ("INT", {"default": -1, "step": 1, "min": -360, "max": 360}),
                "denoise_min": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step":0.01}),
                "denoise_max": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step":0.01}),
                "easing_type": (list(easing_functions.keys()), ),
            },
            "optional": {
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
                "end_frame_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/chx_ksample"

    def apply(self, image, frame, seed, x, y, zoom, angle, denoise_min, denoise_max, easing_type, model=None, context=None, end_frame_image=None):
        if model is None:
            model = context.get("model")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler_name = context.get("sampler")
        scheduler = context.get("scheduler")
        vae = context.get("vae")
        positive = context.get("positive")
        negative = context.get("negative")
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()

        num_samples = frame  # 假设每帧对应一个样本
        x = adapt_to_batch(x, num_samples)
        y = adapt_to_batch(y, num_samples)
        zoom = adapt_to_batch(zoom, num_samples)
        angle = adapt_to_batch(angle, num_samples)

        res = [image]
        pbar = comfy.utils.ProgressBar(frame)

        if end_frame_image is not None:
            end_frame_latent = vaeencode.encode(vae, end_frame_image)[0]

            for i in tqdm(range(frame)):
                ratio = (i + 1) / frame
                denoise = (denoise_max - denoise_min) * apply_easing(ratio, easing_type)  + denoise_min
                image = image_2dtransform(image, x[i], y[i], zoom[i], angle[i], 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                blended_latent = latent["samples"] * (1 - ratio) + end_frame_latent["samples"] * ratio
                latent["samples"] = blended_latent
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)
        else:
            for i in tqdm(range(frame)):
                denoise = (denoise_max - denoise_min) * apply_easing((i+1)/frame, easing_type)  + denoise_min
                image = image_2dtransform(image, x[i], y[i], zoom[i], angle[i], 0, "reflect")
                latent = vaeencode.encode(vae, image)[0]
                noise = comfy.sample.prepare_noise(latent["samples"], i, None)
                samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                            force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)

                image = vaedecode.decode(vae, {"samples": samples})[0]
                pbar.update_absolute(i + 1, frame, None)
                res.append(image)
        if res[0].size() != res[-1].size():
            res = res[1:]

        res = torch.cat(res, dim=0)
        return (res, )


#endregion------------------------deforum_ksampler------------------------



#region------------------------ksampler-tile------------------------

def split_image(img, tile_size=1024):
    if isinstance(img, list):
        print("Warning: img is a list, selecting the first element.")
        img = img[0]
    if not hasattr(img, 'width') or not hasattr(img, 'height'):
        raise TypeError("The input 'img' must be an image object (e.g., PIL Image or torch tensor).")

    tile_width, tile_height = tile_size, tile_size
    width, height = img.width, img.height

    num_tiles_x = ceil(width / tile_width)
    num_tiles_y = ceil(height / tile_height)

    if num_tiles_x < 2:
        num_tiles_x = 2
    if num_tiles_y < 2:
        num_tiles_y = 2

    if width % tile_width == 0:
        num_tiles_x += 1
    if height % tile_height == 0:
        num_tiles_y += 1

    if num_tiles_x > 1:
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    else:
        overlap_x = 0
    if num_tiles_y > 1:
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)
    else:
        overlap_y = 0

    if overlap_x < 256:
        num_tiles_x += 1
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    if overlap_y < 256:
        num_tiles_y += 1
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)

    tiles = []

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_width - j * overlap_x
            y_start = i * tile_height - i * overlap_y

            x_start = round(x_start)
            y_start = round(y_start)

            tile_img = img.crop((x_start, y_start, x_start + tile_width, y_start + tile_height))
            tiles.append(((x_start, y_start, x_start + tile_width, y_start + tile_height), tile_img))

    return tiles

def stitch_images(upscaled_size, tiles):
    if isinstance(upscaled_size, tuple):
        width, height = upscaled_size
    elif hasattr(upscaled_size, 'size'):
        width, height = upscaled_size.size
    elif hasattr(upscaled_size, 'shape'):
        _, height, width = upscaled_size.shape
    else:
        raise TypeError("upscaled_size should be a tuple, PIL.Image, or torch.Tensor.")
    
    result = torch.zeros((3, height, width))
    sorted_tiles = sorted(tiles, key=lambda x: (x[0][1], x[0][0]))
    current_row_upper = None

    for (left, upper, right, lower), tile in sorted_tiles:
        if current_row_upper != upper:
            current_row_upper = upper
            first_tile_in_row = True
        else:
            first_tile_in_row = False

        tile_width = right - left
        tile_height = lower - upper
        feather = tile_width // 8

        mask = torch.ones(tile.shape[0], tile.shape[1], tile.shape[2])

        if not first_tile_in_row:
            for t in range(feather):
                mask[:, :, t:t+1] *= (1.0 / feather) * (t + 1)

        if upper != 0:
            for t in range(feather):
                mask[:, t:t+1, :] *= (1.0 / feather) * (t + 1)

        tile = tile.squeeze(0).squeeze(0)
        tile_to_add = tile.permute(2, 0, 1)
        combined_area = tile_to_add * mask.unsqueeze(0) + result[:, upper:lower, left:right] * (1.0 - mask.unsqueeze(0))
        result[:, upper:lower, left:right] = combined_area

    tensor_expanded = result.unsqueeze(0)
    tensor_final = tensor_expanded.permute(0, 2, 3, 1)
    return tensor_final

def ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, start_step=11, end_step=20):
    vaedecoder = VAEDecode()
    vaeencoder = VAEEncode()
    tile = pil2tensor(tile)
    encoded_tile = vaeencoder.encode(vae, tile)[0]
    tile = common_ksampler(base_model, seed, end_step, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, encoded_tile,
                        start_step=start_step, force_full_denoise=True)[0]
    tile = vaedecoder.decode(vae, tile)[0]
    return tile

def run_tiler_for_steps(enlarged_img, base_model, vae, seed, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, steps=20, denoise=0.25, tile_size=1024):
    if isinstance(enlarged_img, list):
        print("Warning: enlarged_img is a list, selecting the first element.")
        enlarged_img = enlarged_img[0]
    if not hasattr(enlarged_img, 'size') and not hasattr(enlarged_img, 'shape'):
        raise TypeError("enlarged_img should be a valid image object (e.g., PIL.Image or torch tensor).")

    tiles = split_image(enlarged_img, tile_size=tile_size)

    start_step = int(steps - (steps * denoise))
    end_step = steps
    resampled_tiles = [(coords, ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler,
                                            positive_cond_base, negative_cond_base, start_step, end_step)) for coords, tile in tiles]

    result = stitch_images(enlarged_img.size, resampled_tiles)

    return result


class chx_ksampler_tile:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("RUN_CONTEXT",),
                    "model_name": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                    "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise_image": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                    },
                "optional": {"image_optional": ("IMAGE",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            }
            
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = "Apt_Preset/chx_ksample"

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image,
                    steps, cfg, sampler_name, scheduler):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        start_step = int(steps - (steps * denoise_image))
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                start_step=start_step, last_step=steps, force_full_denoise=False)[0]
        pixels = vaedecoder.decode(vae, sample1)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def run(self, seed, model_name, upscale_by=2.0, tile_denoise=0.3, tile_size=512,prompt=None, image_output=None, extra_pnginfo=None,
            upscale_method='normal', denoise_image=1.0, image_optional=None, context=None):
        if image_output == "None":
            output_image = context.get("images",None)
            return (output_image,)
        
        vae = context.get("vae", None)
        steps = context.get("steps", 8)
        cfg = context.get("cfg", 7)
        sampler_name = context.get("sampler", "dpmpp_sde_gpu")
        scheduler = context.get("scheduler", "karras")
        positive_cond_base = context.get("positive", "")
        negative_cond_base = context.get("negative", "")
        base_model = context.get("model", None)
        samples = context.get("latent", None)

        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        
        img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                                            upscale_by, model_name, seed, vae, denoise_image,
                                                            steps, cfg, sampler_name, scheduler)
        img= tensor2pil(img)

        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tile_denoise, tile_size)

        results = easySave(tiled_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": ( tiled_image,)}
            
        return {"ui": {"images": results},
                "result": ( tiled_image,)}

#endregion------------------------_ksampler-tile------------------------



#region------------------------visualstyle ksampler-------------------------------------------------


T = torch.Tensor

class VisualStyleProcessor(object):
    def __init__(self, module_self, keys_scale: float = 1.0, enabled: bool = True, adain_queries: bool = True, adain_keys: bool = True, adain_values: bool = False):
        self.module_self = module_self
        self.keys_scale = keys_scale
        self.enabled = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    def visual_style_forward(self, x, context, value, mask=None):
        q = self.module_self.to_q(x)
        context = default(context, x)
        k = self.module_self.to_k(context)
        if value is not None:
            v = self.module_self.to_v(value)
            del value
        else:
            v = self.module_self.to_v(context)

        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)
            k = concat_first(k, -2, self.keys_scale)
            v = concat_first(v, -2)

        if mask is None:
            out = optimized_attention(q, k, v, self.module_self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.module_self.heads, mask)
        return self.module_self.to_out(out)

@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    keys_scale: float = 1.
    only_self_level: float = 0.

def expand_first(feat: T, scale=1., ) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)

def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)

def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def swapping_attention(key, value, chunk_size=2):
    chunk_length = key.size()[0] // chunk_size
    reference_image_index = [0] * chunk_length
    key = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key = key[:, reference_image_index]
    key = rearrange(key, "b f d c -> (b f) d c")
    value = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value = value[:, reference_image_index]
    value = rearrange(value, "b f d c -> (b f) d c")
    return key, value

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = "k" in share_attn
        self.adain_values = "v" in share_attn
        self.adain_queries = "q" in share_attn
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True

def expand_first(feat: T, scale=1.0,) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)

def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)

def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

class SharedAttentionProcessor:
    def __init__(self, args: StyleAlignedArgs, scale: float):
        self.args = args
        self.scale = scale

    def __call__(self, q, k, v, extra_options):
        if self.args.adain_queries:
            q = adain(q)
        if self.args.adain_keys:
            k = adain(k)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)
        return q, k, v

def get_norm_layers(layer: nn.Module, norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]", share_layer_norm: bool, share_group_norm: bool):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(child_layer, norm_layers_, share_layer_norm, share_group_norm)

def register_norm_forward(norm_layer: Union[nn.GroupNorm, nn.LayerNorm]) -> Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, "orig_forward"):
        setattr(norm_layer, "orig_forward", norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: T) -> T:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)
        return hidden_states[..., :n, :]

    norm_layer.forward = forward_
    return norm_layer

def register_shared_norm(model: ModelPatcher, share_group_norm: bool = True, share_layer_norm: bool = True):
    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model.model, norm_layers, share_layer_norm, share_group_norm)
    print(f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms.")
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [register_norm_forward(layer) for layer in norm_layers["layer"]]

class model_Style_Align:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Apt_Preset/model"

    def patch(self, model: ModelPatcher, share_norm: str, share_attn: str, scale: float):
        m = model.clone()
        share_group_norm = share_norm in ["group", "both"]
        share_layer_norm = share_norm in ["layer", "both"]
        register_shared_norm(model, share_group_norm, share_layer_norm)
        args = StyleAlignedArgs(share_attn)
        m.set_model_attn1_patch(SharedAttentionProcessor(args, scale))
        return (m,)

class chx_Ksampler_VisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "reference_image": ("IMAGE",),
                "reference_image_text": ("STRING", {"multiline": True}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "enabled": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 2}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE")
    RETURN_NAMES = ("context", "images")
    CATEGORY = "Apt_Preset/chx_ksample"
    FUNCTION = "run"

    def run(self, reference_image, reference_image_text, positive_prompt, seed, denoise, enabled, share_norm: str, share_attn: str, scale: float, batch_size=1, context=None):
        vae = context.get("vae")
        negative = context.get("negative")
        clip = context.get("clip")
        model2: comfy.model_patcher.ModelPatcher = context.get("model")
        if not isinstance(model2, comfy.model_patcher.ModelPatcher):
            raise TypeError(f"Expected model2 to be of type ModelPatcher, got {type(model2)}")
        tokens = clip.tokenize(reference_image_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        reference_image_prompt = [[cond, {"pooled_output": pooled}]]
        reference_image = reference_image.repeat(((batch_size + 1) // 2, 1, 1, 1))
        self.model = model2
        reference_latent = vae.encode(reference_image[:, :, :, :3])
        for n, m in model2.model.diffusion_model.named_modules():
            if m.__class__.__name__ == "CrossAttention":
                processor = VisualStyleProcessor(m, enabled=enabled)
                setattr(m, 'forward', processor.visual_style_forward)
        positive, = CLIPTextEncode().encode(clip, positive_prompt)
        positive = reference_image_prompt + positive
        negative = negative * 2
        latents = torch.zeros_like(reference_latent)
        latents = torch.cat([latents] * 2)
        latents[::2] = reference_latent
        denoise_mask = torch.ones_like(latents)[:, :1, ...]
        denoise_mask[0] = 0.
        model_patched = model_Style_Align().patch(model2, share_norm, share_attn, scale)
        model2 = model_patched[0]
        latent = {"samples": latents, "noise_mask": denoise_mask}
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler = context.get("sampler", None)
        scheduler = context.get("scheduler", None)
        latent = common_ksampler(model2, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise)[0]
        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, positive=positive, negative=negative, model=model2, latent=latent)
        return (context, output_image)

#endregion-----------visualstyle ksampler-------------------------------------------------



class chx_Ksampler_dual_paint:    #双区采样 ksampler
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "smoothness":("INT", {"default": 1,  "min":0, "max": 150, "step": 1,"display": "slider"}),
                "mask_area_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_area_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine": ("BOOLEAN", {"default": True}),
                "refine_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/chx_ksample"

    def execute(self,context, image, mask, smoothness, mask_area_denoise, image_area_denoise,refine,refine_denoise, seed,):
        
        vae = context.get("vae",None)
        steps = context.get("steps",None)
        cfg = context.get("cfg",None)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)

        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None) 


        phase_steps = math.ceil(steps / 2)
        device = model.model.device if hasattr(model, 'model') else model.device
        
        vae_encoder = VAEEncode()
        latent_dict = vae_encoder.encode(vae, image)[0]
        input_latent = latent_dict["samples"].to(device)


        if mask is not None :
            mask=tensor2pil(mask)
            if not isinstance(mask, Image.Image):
                raise TypeError("mask is not a valid PIL Image object")
            
            feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
            mask=pil2tensor(feathered_image)


        
        mask = 1-mask.float().to(device)
        
        mask_resized = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(input_latent.shape[2], input_latent.shape[3]), 
            mode='bilinear'
        )
        
        mask_strength = mask_resized * (image_area_denoise - mask_area_denoise) + mask_area_denoise
        
        noise_mask = SetLatentNoiseMask()
        latent_with_mask = noise_mask.set_mask({"samples": input_latent}, mask_strength)[0]
        
        advanced_sampler = KSamplerAdvanced()
        
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_with_mask,
            start_at_step=0,
            end_at_step=phase_steps,
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        binary_mask = (mask_resized >= 0.5).float()
        phase2_mask = binary_mask * 1.0 + (1 - binary_mask) * mask_area_denoise
        
        latent_phase2 = noise_mask.set_mask(
            {"samples": samples},
            phase2_mask
        )[0]
    
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed + 1,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_phase2,
            start_at_step=phase_steps,
            end_at_step=steps,
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        
        if refine:
            sampler = KSampler()
            result = sampler.sample(
                model,
                seed + 1,
                steps,
                cfg,
                sampler,
                scheduler,
                positive,
                negative,
                {"samples": samples},
                refine_denoise
            )[0]
            samples = result["samples"].to(device)
        
        latent= {"samples": samples}
        images = VAEDecode().decode(vae, latent)[0]
        
        context = new_context(context,  latent=latent, images=images, )

        return (context,images,)


#region-----------------------basic node-------------------------------




#region-------------def--------------------------------------------------------------------#

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
matplotlib.use('Agg')



def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item()
    # If we're here, that means the sigma is in between two sigmas in the
    # list.
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        # Shouldn't be possible, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow]
    # Ratio of how close we are to the high neighbor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    # Mix the DD schedule high/low items according to the ratio.
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
) -> torch.Tensor:
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


def generate_tiles(
    image_width, image_height, tile_width, tile_height, overlap, offset=0
):
    tiles = []

    y = 0
    while y < image_height:
        if y == 0:
            next_y = y + tile_height - overlap + offset
        else:
            next_y = y + tile_height - overlap

        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - overlap + offset
            else:
                next_x = x + tile_width - overlap
            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            tiles.append((x, y))

            if next_x > image_width:
                break
            x = next_x

        if next_y > image_height:
            break
        y = next_y

    return tiles


def resize(image, size):
    if image.size()[-2:] == size:
        return image
    return torch.nn.functional.interpolate(image, size)


def make_circular_asymm(model, tileX: bool, tileY: bool):
    for layer in [
        layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
    ]:
        layer.padding_modeX = 'circular' if tileX else 'constant'
        layer.padding_modeY = 'circular' if tileY else 'constant'
        layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
        layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
        layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)
    return model


def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

#endregion---------def------------------------------------------------------------------------#


class lay_texture_Offset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/imgEffect"

    def run(self, pixels, x_percent, y_percent):
        n, y, x, c = pixels.size()
        y = round(y * y_percent / 100)
        x = round(x * x_percent / 100)
        return (pixels.roll((y, x), (1, 2)),)


class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "overlap": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_ksample"

    def process(self, image, tile_width, tile_height, overlap, offset):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_coordinates = generate_tiles(
            image_width, image_height, tile_width, tile_height, overlap, offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))

        iteration = 1

        image_tiles = []
        for tile_coordinate in tile_coordinates:
            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))
            iteration += 1

            image_tile = image[
                :,
                tile_coordinate[1] : tile_coordinate[1] + tile_height,
                tile_coordinate[0] : tile_coordinate[0] + tile_width,
                :,
            ]

            image_tiles.append(image_tile)

        tiles_tensor = torch.stack(image_tiles).squeeze(1)
        tile_calc = (overlap, image_height, image_width, offset)

        return (tiles_tensor, tile_calc)


class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend": ("INT", {"default": 64, "min": 0, "max": 4096}),
                "tile_calc": ("TILE_CALC",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_ksample"

    def process(self, images, blend, tile_calc):
        overlap, final_height, final_width, offset = tile_calc
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        print("Tile height: {}".format(tile_height))
        print("Tile width: {}".format(tile_width))
        print("Final height: {}".format(final_height))
        print("Final width: {}".format(final_width))
        print("Overlap: {}".format(overlap))

        tile_coordinates = generate_tiles(
            final_width, final_height, tile_width, tile_height, overlap, offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))
        original_shape = (1, final_height, final_width, 3)
        count = torch.zeros(original_shape, dtype=images.dtype)
        output = torch.zeros(original_shape, dtype=images.dtype)

        index = 0
        iteration = 1
        for tile_coordinate in tile_coordinates:
            image_tile = images[index]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))
            iteration += 1

            channels = images.shape[3]
            weight_matrix = torch.ones((tile_height, tile_width, channels))

            # blend border
            for i in range(blend):
                weight = float(i) / blend
                weight_matrix[i, :, :] *= weight  # Top rows
                weight_matrix[-(i + 1), :, :] *= weight  # Bottom rows
                weight_matrix[:, i, :] *= weight  # Left columns
                weight_matrix[:, -(i + 1), :] *= weight  # Right columns

            old_tile = output[:, y : y + tile_height, x : x + tile_width, :]
            old_tile_count = count[:, y : y + tile_height, x : x + tile_width, :]

            weight_matrix = (
                weight_matrix * (old_tile_count != 0).float()
                + (old_tile_count == 0).float()
            )

            image_tile = image_tile * weight_matrix + old_tile * (1 - weight_matrix)

            output[:, y : y + tile_height, x : x + tile_width, :] = image_tile
            count[:, y : y + tile_height, x : x + tile_width, :] = 1

            index += 1
        return [output]


class sampler_enhance:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
            },
        }
    CATEGORY = "Apt_Preset/chx_ksample"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    
    @classmethod
    def go(
        cls,
        sampler: object,
        *,
        detail_amount=0.1,
        start=0.2,
        end=0.8,
        bias=0.5,
        exponent=1,
        start_offset=0,
        end_offset=0,
        fade=0,
        smooth="true",
        cfg_scale_override=0,
    ) -> tuple:
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                start,
                end,
                bias,
                detail_amount,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            )

        return (
            KSAMPLER(
                detail_daemon_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                    "dds_cfg_scale_override": cfg_scale_override,
                },
            ),
        )



#endregion-----------------------basic node---------------------------------



#region-----------------------------pre-ic light------------------------

UNET_MAP_ATTENTIONS = {"proj_in.weight","proj_in.bias","proj_out.weight","proj_out.bias","norm.weight","norm.bias"}
TRANSFORMER_BLOCKS = {"norm1.weight","norm1.bias","norm2.weight","norm2.bias","norm3.weight","norm3.bias","attn1.to_q.weight","attn1.to_k.weight","attn1.to_v.weight","attn1.to_out.0.weight","attn1.to_out.0.bias","attn2.to_q.weight","attn2.to_k.weight","attn2.to_v.weight","attn2.to_out.0.weight","attn2.to_out.0.bias","ff.net.0.proj.weight","ff.net.0.proj.bias","ff.net.2.weight","ff.net.2.bias"}
UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight","in_layers.2.bias": "conv1.bias","emb_layers.1.weight": "time_emb_proj.weight","emb_layers.1.bias": "time_emb_proj.bias","out_layers.3.weight": "conv2.weight","out_layers.3.bias": "conv2.bias","skip_connection.weight": "conv_shortcut.weight","skip_connection.bias": "conv_shortcut.bias","in_layers.0.weight": "norm1.weight","in_layers.0.bias": "norm1.bias","out_layers.0.weight": "norm2.weight","out_layers.0.bias": "norm2.bias"}
UNET_MAP_BASIC = {("label_emb.0.0.weight", "class_embedding.linear_1.weight"),("label_emb.0.0.bias", "class_embedding.linear_1.bias"),("label_emb.0.2.weight", "class_embedding.linear_2.weight"),("label_emb.0.2.bias", "class_embedding.linear_2.bias"),("label_emb.0.0.weight", "add_embedding.linear_1.weight"),("label_emb.0.0.bias", "add_embedding.linear_1.bias"),("label_emb.0.2.weight", "add_embedding.linear_2.weight"),("label_emb.0.2.bias", "add_embedding.linear_2.bias"),("input_blocks.0.0.weight", "conv_in.weight"),("input_blocks.0.0.bias", "conv_in.bias"),("out.0.weight", "conv_norm_out.weight"),("out.0.bias", "conv_norm_out.bias"),("out.2.weight", "conv_out.weight"),("out.2.bias", "conv_out.bias"),("time_embed.0.weight", "time_embedding.linear_1.weight"),("time_embed.0.bias", "time_embedding.linear_1.bias"),("time_embed.2.weight", "time_embedding.linear_2.weight"),("time_embed.2.bias", "time_embedding.linear_2.bias")}
TEMPORAL_TRANSFORMER_BLOCKS = {"norm_in.weight","norm_in.bias","ff_in.net.0.proj.weight","ff_in.net.0.proj.bias","ff_in.net.2.weight","ff_in.net.2.bias"}
TEMPORAL_TRANSFORMER_BLOCKS.update(TRANSFORMER_BLOCKS)
TEMPORAL_UNET_MAP_ATTENTIONS = {"time_mixer.mix_factor"}
TEMPORAL_UNET_MAP_ATTENTIONS.update(UNET_MAP_ATTENTIONS)
TEMPORAL_TRANSFORMER_MAP = {"time_pos_embed.0.weight": "time_pos_embed.linear_1.weight","time_pos_embed.0.bias": "time_pos_embed.linear_1.bias","time_pos_embed.2.weight": "time_pos_embed.linear_2.weight","time_pos_embed.2.bias": "time_pos_embed.linear_2.bias"}
TEMPORAL_RESNET = {"time_mixer.mix_factor"}
unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,'in_channels': 8, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],'use_temporal_attention': False, 'use_temporal_resblock': False}

def convert_iclight_unet(state_dict):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)
    transformers_mid = unet_config.get("transformer_depth_middle", None)
    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, b)] = "input_blocks.{}.0.{}".format(n, b)
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.time_stack.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)
    i = 0
    for b in TEMPORAL_UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for b in TEMPORAL_TRANSFORMER_MAP:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, TEMPORAL_TRANSFORMER_MAP[b])] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
        for b in TEMPORAL_TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.temporal_transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.time_stack.{}.{}".format(t, b)
    for i, n in enumerate([0, 2]):
        for b in TEMPORAL_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, b)] = "middle_block.{}.{}".format(n, b)
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.spatial_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.temporal_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.time_stack.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, b)] = "output_blocks.{}.0.{}".format(n, b)
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.time_stack.{}".format(n, b)
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}".format(i, b)] = "output_blocks.{}.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1
    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]
    unet_state_dict = state_dict
    diffusers_keys = diffusers_unet_map
    new_sd = {}
    for k in diffusers_keys:
        if k in unet_state_dict:
            new_sd[diffusers_keys[k]] = unet_state_dict.pop(k)
    leftover_keys = unet_state_dict.keys()
    if len(leftover_keys) > 0:
        spatial_leftover_keys = []
        temporal_leftover_keys = []
        other_leftover_keys = []
        for key in leftover_keys:
            if "spatial" in key:
                spatial_leftover_keys.append(key)
            elif "temporal" in key:
                temporal_leftover_keys.append(key)
            else:
                other_leftover_keys.append(key)
        print("spatial_leftover_keys:")
        for key in spatial_leftover_keys:
            print(key)
        print("temporal_leftover_keys:")
        for key in temporal_leftover_keys:
            print(key)
        print("other_leftover_keys:")
        for key in other_leftover_keys:
            print(key)
    new_sd = {"diffusion_model." + k: v for k, v in new_sd.items()}
    return new_sd

class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"

def generate_gradient_image(width, height, start_color, end_color, multiplier, lightPosition):
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img

class LoadAndApplyICLightUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),"model_path": (folder_paths.get_filename_list("unet"), )}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "IC-Light"
    def load(self, model, model_path):
        type_str = str(type(model.model.model_config).__name__)
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if "SD15" not in type_str:
            raise Exception(f"Attempted to load {type_str} model, IC-Light is only compatible with SD 1.5 models.")
        print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
            model_clone = model.clone()
            iclight_state_dict = load_torch_file(model_full_path)
            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            try:
                if 'conv_in.weight' in iclight_state_dict:
                    iclight_state_dict = convert_iclight_unet(iclight_state_dict)
                    in_channels = iclight_state_dict["diffusion_model.input_blocks.0.0.weight"].shape[1]
                    prefix = ""
                else:
                    prefix = "diffusion_model."
                    in_channels = iclight_state_dict["input_blocks.0.0.weight"].shape[1]
                model_clone.model.model_config.unet_config["in_channels"] = in_channels
                patches={(prefix + key): ("diff",[value.to(dtype=dtype, device=device),{"pad_weight": key == "diffusion_model.input_blocks.0.0.weight" or key == "input_blocks.0.0.weight"},])for key, value in iclight_state_dict.items()}
                model_clone.add_patches(patches)
            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")
            def bound_extra_conds(self, **kwargs):
                 return ICLight.extra_conds(self, **kwargs)
            new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
            model_clone.add_object_patch("extra_conds", new_extra_conds)
            return (model_clone, )

class ICLight:
    def extra_conds(self, **kwargs):
        out = {}
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]
        model_in_channels = self.model_config.unet_config['in_channels']
        input_channels = image.shape[1] + 4
        if model_in_channels != input_channels:
            raise Exception(f"Input channels {input_channels} does not match model in_channels {model_in_channels}, 'opt_background' latent input should be used with the IC-Light 'fbc' model, and only with it")
        if image is None:
            image = torch.zeros_like(noise)
        if image.shape[1:] != noise.shape[1:]:
            image = comfy.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])
        process_image_in = lambda image: image
        out['c_concat'] = comfy.conds.CONDNoiseShape(process_image_in(image))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        return out

class ICLightConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),"negative": ("CONDITIONING", ),"vae": ("VAE", ),"foreground": ("LATENT", ),"multiplier": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.001}),},"optional": {"opt_background": ("LATENT", ),}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "empty_latent")
    FUNCTION = "encode"
    CATEGORY = "IC-Light"
    def encode(self, positive, negative, vae, foreground, multiplier, opt_background=None):
        samples_1 = foreground["samples"]
        if opt_background is not None:
            samples_2 = opt_background["samples"]
            repeats_1 = samples_2.size(0) // samples_1.size(0)
            repeats_2 = samples_1.size(0) // samples_2.size(0)
            if samples_1.shape[1:] != samples_2.shape[1:]:
                samples_2 = comfy.utils.common_upscale(samples_2, samples_1.shape[-1], samples_1.shape[-2], "bilinear", "disabled")
            if repeats_1 > 1:
                samples_1 = samples_1.repeat(repeats_1, 1, 1, 1)
            if repeats_2 > 1:
                samples_2 = samples_2.repeat(repeats_2, 1, 1, 1)
            concat_latent = torch.cat((samples_1, samples_2), dim=1)
        else:
            concat_latent = samples_1
        out_latent = torch.zeros_like(samples_1)
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent * multiplier
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], {"samples": out_latent})

class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                },
            "optional": {
                "batch_size": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                "prev_image": ("IMAGE",),
                } 
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    def execute(self, light_position, multiplier, start_color, end_color, width, height, batch_size=1, prev_image=None):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:
                color_rgb =tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            else:
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb
        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        image = image.repeat(batch_size, 1, 1, 1)
        if prev_image is not None:
            image = torch.cat((prev_image, image), dim=0)
        return (image,)




#endregion-----------------------------pre-ic light------------------------




class pre_ic_light_sd15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "bg_unet": (folder_paths.get_filename_list("unet"), {"default": "iclight_sd15_fbc_unet_ldm.safetensors"} ),
                "fo_unet": (folder_paths.get_filename_list("unet"), {"default": "iclight_sd15_fc_unet_ldm.safetensors"} ),
                "multiplier": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            
            "optional": {
                "fore_img":("IMAGE",),
                "bg_img":("IMAGE",),
                "light_img":("IMAGE",),
                
            },
            
        }


    RETURN_TYPES = ("RUN_CONTEXT","IMAGE" )
    RETURN_NAMES = ("context", "light_img" )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_tool"

    def run(self, context, bg_unet, fo_unet, multiplier, fore_img=None, light_img=None, bg_img=None ):


        vae = context.get("vae",None)
        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None)
        images = context.get("images",None)

        if light_img is None:
            outimg = decode(vae, latent)[0]
            return (context,outimg)

        if fore_img is None:    
            fore_img = images

        foreground = encode(vae, fore_img)[0]

        opt_background = None
        if bg_img is not None:
            bg_img = get_image_resize(bg_img,fore_img)   #尺寸一致性
            opt_background = encode(vae, bg_img)[0]

        if bg_img is not None:
            unet = bg_unet
        else:
            unet = fo_unet

        model = LoadAndApplyICLightUnet().load(model, unet)[0]
        positive, negative, empty_latent = ICLightConditioning().encode(
            positive=positive,
            negative=negative,
            vae=vae,
            foreground=foreground,
            multiplier=multiplier,
            opt_background=opt_background
        )

        light_img = get_image_resize(light_img,fore_img)   #尺寸一致性
        context = new_context(context, positive=positive, negative=negative, model=model, images=light_img,)

        return(context, light_img )



class pre_latent_light:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "weigh": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.7, "step": 0.01}),

            },
            "optional": {
                "img_targe": ("IMAGE",),
                "img_light": ("IMAGE",),
            },

        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE","LATENT" )
    RETURN_NAMES = ("context", "image", "latent")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_tool"

    def run(self,context, weigh, img_targe=None, img_light=None, ):

        latent = context.get("latent", None)
        vae = context.get("vae", None)

        if img_light is None:
            outimg = decode(vae, latent)[0]
            return (context,outimg)


        if img_targe is None:
            img_targe = context.get("images", None)

        img_light = get_image_resize(img_light,img_targe)   

        latent2 = encode(vae, img_targe)[0]
        latent1 = encode(vae, img_light)[0]
        latent = latent_inter_polate(latent1, latent2, weigh)

        output_image = decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image, )
        
        return  (context, output_image, latent)


#region-----------------------flex2---------------------------------

def flex2_concat_cond(self: Flux, **kwargs):
    return None

def flex2_extra_conds(self, **kwargs):
    out = self._flex2_orig_extra_conds(**kwargs)
    noise = kwargs.get("noise", None)
    device = kwargs["device"]
    flex2_concat_latent = kwargs.get("flex2_concat_latent", None)
    flex2_concat_latent_no_control = kwargs.get(
        "flex2_concat_latent_no_control", None)
    control_strength = kwargs.get("flex2_control_strength", 1.0)
    control_start_percent = kwargs.get("flex2_control_start_percent", 0.0)
    control_end_percent = kwargs.get("flex2_control_end_percent", 0.1)
    if flex2_concat_latent is not None:
        flex2_concat_latent = comfy.utils.resize_to_batch_size(
            flex2_concat_latent, noise.shape[0])
        flex2_concat_latent = self.process_latent_in(flex2_concat_latent)
        flex2_concat_latent = flex2_concat_latent.to(device)
        out['flex2_concat_latent'] = comfy.conds.CONDNoiseShape(
            flex2_concat_latent)
    if flex2_concat_latent_no_control is not None:
        flex2_concat_latent_no_control = comfy.utils.resize_to_batch_size(
            flex2_concat_latent_no_control, noise.shape[0])
        flex2_concat_latent_no_control = self.process_latent_in(
            flex2_concat_latent_no_control)
        flex2_concat_latent_no_control = flex2_concat_latent_no_control.to(
            device)
        out['flex2_concat_latent_no_control'] = comfy.conds.CONDNoiseShape(
            flex2_concat_latent_no_control)

    out['flex2_control_start_percent'] = comfy.conds.CONDConstant(
        control_start_percent)
    out['flex2_control_end_percent'] = comfy.conds.CONDConstant(
        control_end_percent)

    return out


def flex_apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    sigma = t
    xc = self.model_sampling.calculate_input(sigma, x)
    if c_concat is not None:
        xc = torch.cat([xc] + [c_concat], dim=1)

    flex2_control_start_sigma = 1.0 - \
        kwargs.get("flex2_control_start_percent", 0.0)
    flex2_control_end_sigma = 1.0 - \
        kwargs.get("flex2_control_end_percent", 1.0)

    flex2_concat_latent_active = kwargs.get("flex2_concat_latent", None)
    flex2_concat_latent_inactive = kwargs.get(
        "flex2_concat_latent_no_control", None)

    sigma_float = sigma.mean().cpu().item()
    sigma_int = int(sigma_float * 1000)

    # simple, but doesnt work right because of shift
    is_being_controlled = sigma_float <= flex2_control_start_sigma and sigma_float >= flex2_control_end_sigma

    sigmas = transformer_options.get("sample_sigmas", None)

    if sigmas is not None:
        # we have all the timesteps here, determine what percent we are through the
        # timesteps we are doing. This way is more intuitive to user.
        all_timesteps = [int(sigma.cpu().item() * 1000) for sigma in sigmas]
        current_idx = all_timesteps.index(sigma_int)
        current_percent = current_idx / len(all_timesteps)
        current_percent_sigma = 1.0 - current_percent
        is_being_controlled = current_percent_sigma <= flex2_control_start_sigma and current_percent_sigma >= flex2_control_end_sigma

    if is_being_controlled:
        # it is active
        xc = torch.cat([xc] + [flex2_concat_latent_active], dim=1)
    else:
        # it is inactive
        xc = torch.cat([xc] + [flex2_concat_latent_inactive], dim=1)

    context = c_crossattn
    dtype = self.get_dtype()

    if self.manual_cast_dtype is not None:
        dtype = self.manual_cast_dtype

    xc = xc.to(dtype)
    t = self.model_sampling.timestep(t).float()
    if context is not None:
        context = context.to(dtype)

    extra_conds = {}
    for o in kwargs:
        extra = kwargs[o]
        if hasattr(extra, "dtype"):
            if extra.dtype != torch.int and extra.dtype != torch.long:
                extra = extra.to(dtype)
        extra_conds[o] = extra

    t = self.process_timestep(t, x=x, **extra_conds)
    model_output = self.diffusion_model(
        xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
    return self.model_sampling.calculate_denoised(sigma, model_output, x)


class Flex2Conditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "vae": ("VAE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "latent": ("LATENT", ),
                "inpaint_image": ("IMAGE", ),
                "inpaint_mask": ("MASK", ),
                "control_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "do_it"
    CATEGORY = "advanced/conditioning/flex"

    def do_it(self, model, vae, positive, negative, control_strength, latent=None, inpaint_image=None, inpaint_mask=None, control_image=None):
        control_start_percent=0
        control_end_percent=1

        if not hasattr(model.model, "_flex2_orig_concat_cond"):
            model.model._flex2_orig_concat_cond = model.model.concat_cond
            model.model.concat_cond = partial(flex2_concat_cond, model.model)
        if not hasattr(model.model, "_flex2_orig_extra_conds"):
            model.model._flex2_orig_extra_conds = model.model.extra_conds
            model.model.extra_conds = partial(flex2_extra_conds, model.model)
        if not hasattr(model.model, "_flex2_orig_apply_model"):
            model.model._flex2_orig_apply_model = model.model._apply_model
            model.model._apply_model = partial(flex_apply_model, model.model)

        batch_size = 1
        latent_height: int = None
        latent_width: int = None

        if latent is not None:
            latent_height = latent['samples'].shape[2]
            latent_width = latent['samples'].shape[3]
            if latent['samples'].shape[1] == 4:
                latent['samples'] = torch.cat([latent['samples'] for _ in range(4)], dim=1)
            batch_size = latent['samples'].shape[0]
        elif inpaint_image is not None:
            batch_size = inpaint_image.shape[0]
            latent_height = inpaint_image.shape[1] // 8
            latent_width = inpaint_image.shape[2] // 8
        elif control_image is not None:
            batch_size = control_image.shape[0]
            latent_height = control_image.shape[1] // 8
            latent_width = control_image.shape[2] // 8
        else:
            raise ValueError("No latent, inpaint or control image provided")

        img_width = latent_width * 8
        img_height = latent_height * 8

        model = model.clone()

        concat_latent = torch.zeros((batch_size, 33, latent_height, latent_width), device='cpu', dtype=torch.float32)
        concat_latent[:, 16:17, :, :] = torch.ones((batch_size, 1, latent_height, latent_width), device='cpu', dtype=torch.float32)

        if latent is not None:
            out_latent = latent
        else:
            out_latent = {"samples": torch.zeros((batch_size, 16, latent_height, latent_width), device='cpu', dtype=torch.float32)}

        if inpaint_image is not None:
            if inpaint_image.shape[1] != img_height or inpaint_image.shape[2] != img_width:
                inpaint_image = torch.nn.functional.interpolate(inpaint_image.permute(0, 3, 1, 2), size=(img_height, img_width), mode="bilinear").permute(0, 2, 3, 1)

            if inpaint_mask is not None:
                inpaint_mask_latent = torch.nn.functional.interpolate(inpaint_mask.reshape((-1, 1, inpaint_mask.shape[-2], inpaint_mask.shape[-1])), size=(latent_height, latent_width), mode="bilinear")
            else:
                inpaint_mask_latent = torch.ones((batch_size, 1, latent_height, latent_width), device='cpu', dtype=torch.float32)

            inpaint_latent_orig = vae.encode(inpaint_image)
            out_latent["samples"] = inpaint_latent_orig.clone()
            inpaint_latent_masked = inpaint_latent_orig * (1 - inpaint_mask_latent)
            concat_latent[:, 0:16, :, :] = inpaint_latent_masked
            concat_latent[:, 16:17, :, :] = inpaint_mask_latent

        concat_latent_no_control = concat_latent.clone()

        if control_image is not None:
            if control_image.shape[1] != img_height or control_image.shape[2] != img_width:
                control_image = torch.nn.functional.interpolate(control_image.permute(0, 3, 1, 2), size=(img_height, img_width), mode="bilinear").permute(0, 2, 3, 1)

            control_latent = vae.encode(control_image)
            concat_latent[:, 17:33, :, :] = control_latent * control_strength

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {
                "flex2_concat_latent": concat_latent,
                "flex2_concat_latent_no_control": concat_latent_no_control,
                "flex2_control_strength": control_strength,
                "flex2_control_start_percent": control_start_percent,
                "flex2_control_end_percent": control_end_percent,
            })
            out.append(c)
        positive = out[0]
        negative = out[1]

        return (model, positive, negative, out_latent)


class pre_Flex2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "control_image": ("IMAGE", ),
                "inpaint_image": ("IMAGE", ),
                "inpaint_mask": ("MASK", ),
            }
        }


    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("context", "positive", "latent")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_tool"

    def run(self, context,control_strength, inpaint_image=None,inpaint_mask=None,control_image=None ):

        vae = context.get("vae",None)
        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None)

        model, positive, negative, latent = Flex2Conditioner().do_it(
            model=model, 
            vae=vae, 
            positive=positive, 
            negative=negative, 
            control_strength=control_strength, 
            latent=latent, 
            inpaint_image=inpaint_image, 
            inpaint_mask=inpaint_mask, 
            control_image=control_image
        )

        context = new_context(context, positive=positive, negative=negative, model=model, latent=latent,)
        return(context,positive,latent )




#endregion-----------------------pre_sample--------------------------------



#endregion-----------总------------------------------



class chx_Ksampler_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps": ("INT", {"default": -1, "min": -1, "max": 10000,  "tooltip": "-1 means no change"}),
            "repaint_mode": (["latent_image", "latent_blank"],),
            "mask_extend": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
            "feather":("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
            
            "crop_mask_as_inpaint_area": ("BOOLEAN", {"default": False,  "tooltip": "Below options are only valid when it is true."}),
            "crop_area_scale": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "crop_area_extend": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),


        },
                
            "optional": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "pos":("STRING",{"multiline": True, "default": ""}),
            }
                     
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "IMAGE",  "MASK")
    RETURN_NAMES = ('context', 'result_img', 'sample_img', 'mask')
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"

    def mask_crop(self, image, mask, crop_area_extend, crop_area_scale=0):
        

        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)
        mask_array = np.array(mask_pil) > 0
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            return (image, None, mask)
        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
        x0 -= crop_area_extend
        y0 -= crop_area_extend
        x1 += crop_area_extend
        y1 += crop_area_extend
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        if crop_area_scale > 0:
            min_size = min(cropped_image_pil.size)
            if min_size < crop_area_scale or min_size > crop_area_scale:
                scale_ratio = crop_area_scale / min_size
                new_size = (int(cropped_image_pil.width * scale_ratio), int(cropped_image_pil.height * scale_ratio))
                cropped_image_pil = cropped_image_pil.resize(new_size, Image.LANCZOS)
                cropped_mask_pil = cropped_mask_pil.resize(new_size, Image.LANCZOS)

        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        qtch = image
        qtzz = mask
        return (cropped_image_tensor, cropped_mask_tensor, (y0, y1, x0, x1) ,qtch ,qtzz )

    def encode(self, vae, image, mask, mask_extend=6, repaint_mode="latent_blank"):
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                            size=(image.shape[1], image.shape[2]), mode="bilinear")
        if repaint_mode == "latent_blank":
            image = image.clone()
            if image.shape[1] != x or image.shape[2] != y:
                x_offset = (image.shape[1] % 8) // 2
                y_offset = (image.shape[2] % 8) // 2
                image = image[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
        if mask_extend == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, mask_extend, mask_extend))
            padding = math.ceil((mask_extend - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        if repaint_mode == "latent_blank":
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        t = vae.encode(image)
        return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}, None
    
    def paste_cropped_image_with_mask(self, original_image, cropped_image, crop_coords, mask, MHmask, feather):
        y0, y1, x0, x1 = crop_coords
        original_image_pil = tensor2pil(original_image)
        cropped_image_pil = tensor2pil(cropped_image)
        mask_pil = tensor2pil(mask)
        crop_width = x1 - x0
        crop_height = y1 - y0
        crop_size = (crop_width, crop_height)

        cropped_image_pil = cropped_image_pil.resize(crop_size, Image.LANCZOS)
        mask_pil = mask_pil.resize(crop_size, Image.LANCZOS)

        mask_binary = mask_pil.convert('L')
        mask_rgba = mask_binary.convert('RGBA')
        blurred_mask = mask_rgba
        transparent_mask = mask_binary
        blurred_mask = mask_binary
        cropped_image_pil = cropped_image_pil.convert('RGBA')
        original_image_pil = original_image_pil.convert('RGBA')
        original_image_pil.paste(cropped_image_pil, (x0, y0), mask=blurred_mask)
        ZT_image_pil=original_image_pil.convert('RGB')
        IMAGEEE = pil2tensor(ZT_image_pil)        
        mask_ecmhpil= tensor2pil(MHmask)   
        mask_ecmh = mask_ecmhpil.convert('L')
        mask_ecrgba = tensor2pil(MHmask)   
        maskecmh = None
        if feather is not None:
            if feather > -1:
                maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather))
        dyzz = pil2tensor(maskecmh)
        maskeccmh = pil2tensor(maskecmh)
        destination = original_image
        source = IMAGEEE
        dyyt = source
        multiplier = 8
        resize_source = True
        mask = dyzz
        destination = destination.clone().movedim(-1, 1)
        source=source.clone().movedim(-1, 1)
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
        x=0
        y=0
        x = int(x)
        y = int(y)  
        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        zztx = destination.movedim(1, -1)
        return zztx,dyzz,dyyt


    def sample(self, context, seed, image=None,steps=1, mask=None,pos=None, mask_extend=6, repaint_mode="latent_blank", denoise=1.0, crop_mask_as_inpaint_area=False, crop_area_extend=0, crop_area_scale=0, feather=1, ):

        guidance = context.get("guidance",3.5)
        if steps == 0 or steps==-1: steps = context.get("steps")
        model = context.get("model", None)
        model = DifferentialDiffusion().apply(model)[0]      
        vae = context.get ("vae", None)
        
        clip = context.get("clip", None)
        positive = None
        if pos and pos.strip():
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        

        negative = context.get("negative", None)
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler_name = context.get("sampler", None)
        scheduler = context.get("scheduler", None)

        if image is not None :
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent", None)
        positive, negative, latent=InpaintModelConditioning().encode(positive, negative, image, vae, mask, noise_mask=True)   

        if mask is None or torch.all(mask == 0):

            latent = common_ksampler(model,seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=denoise)[0]
            output_image = VAEDecode().decode(vae, latent)[0]
            
            context = new_context(context, latent=latent,positive=positive,negative=negative, images=output_image, )
            zztx = output_image
            decoded_image= output_image
            return (context, zztx, decoded_image, None)   


        if mask is not None:
            original_image = image
            hqccimage = tensor2pil(image)
            sfmask = tensor2pil(mask)
            sfhmask = sfmask.resize(hqccimage.size, Image.LANCZOS)
            mask = pil2tensor(sfhmask)
            
            MHmask = mask
            
            if crop_mask_as_inpaint_area:
                image, mask, crop_coords,bytx, byzz = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                final_image,dyzz,dyyt = self.paste_cropped_image_with_mask(original_image, decoded_image, crop_coords, mask, MHmask, feather)

             
                latent = VAEEncode().encode(vae, final_image)[0]
                
                context = new_context(context, latent = latent, positive=positive,negative=negative, images = final_image)
                return (context, final_image, decoded_image,dyzz)     

            else:
                bytx, byzz, crop_coords,image, mask = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                
                mask_ecrgba = tensor2pil(mask)   
                
                maskecmh = None
                if feather is not None:
                    if feather > -1:
                        maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather))
                dyzz = pil2tensor(maskecmh)
                mask = dyzz
                #maskeccmh = pil2tensor(maskecmh)
                #mask = maskeccmh
                destination = original_image
                source = decoded_image       
                multiplier = 8
                resize_source = True

                destination = destination.clone().movedim(-1, 1)
                source=source.clone().movedim(-1, 1)
                source = source.to(destination.device)
                if resize_source:
                    source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

                source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
                x=0
                y=0
                x = int(x)
                y = int(y)  
                x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
                y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

                left, top = (x // multiplier, y // multiplier)
                right, bottom = (left + source.shape[3], top + source.shape[2],)

                if mask is None:
                    mask = torch.ones_like(source)
                else:
                    mask = mask.to(destination.device, copy=True)
                    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
                    mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
                visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)
                mask = mask[:, :, :visible_height, :visible_width]
                inverse_mask = torch.ones_like(mask) - mask
                source_portion = mask * source[:, :, :visible_height, :visible_width]
                destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]
                destination[:, :, top:bottom, left:right] = source_portion + destination_portion
                zztx = destination.movedim(1, -1)
            
                latent = VAEEncode().encode(vae, zztx)[0]

                context = new_context(context, latent = latent, positive=positive, pos=pos,negative=negative, images = zztx)
                return (context, zztx, decoded_image, dyzz)     


class chx_Ksampler_Kontext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 10000,  "tooltip": "-1 means no change"}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "LATENT")
    RETURN_NAMES = ("context", "image", "latent")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"
    OUTPUT_NODE = True

    def run(self, context, seed, image_output="Preview", image=None, mask=None, steps=0, denoise=1, prompt_weight=0.5, auto_adjust_image=False, pos="", prompt=None, extra_pnginfo=None):
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")
        if steps == 0 or steps==-1: steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        guidance = context.get("guidance", 3.5)

        if image is None: 
            image = context.get("images", None)
        assert image is not None, "Image must be provided or exist in the context."

        if image.dim() == 3: image = image.unsqueeze(0)

        all_output_images = []
        all_results = []

        for i in range(image.shape[0]):
            img = image[i:i+1]
            pixels = img
            if auto_adjust_image:
                width = img.shape[2]
                height = img.shape[1]
                aspect_ratio = width / height
                _, target_width, target_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
                scaled_image = comfy.utils.common_upscale(img.movedim(-1, 1), target_width, target_height, "lanczos", "center").movedim(1, -1)
                pixels = scaled_image[:, :, :, :3].clamp(0, 1)
            
            encoded_latent = vae.encode(pixels)[0]
            if encoded_latent.dim() == 3: encoded_latent = encoded_latent.unsqueeze(0)
            latent = {"samples": encoded_latent}

            positive = None
            if pos and pos.strip(): positive, = CLIPTextEncode().encode(clip, pos)
            else: positive = context.get("positive", None)

            if positive is not None and prompt_weight > 0:
                influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                scaled_latent = latent["samples"] * influence
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            if mask is not None: latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

            context = new_context(context, positive=positive, latent=latent)
            negative = context.get("negative", None)

            result_tuple = common_ksampler(model, seed + i, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise)
            
            if isinstance(result_tuple, dict):
                result_latent = result_tuple["samples"] if "samples" in result_tuple else result_tuple
            elif isinstance(result_tuple, (list, tuple)):
                result_latent = result_tuple[0] if len(result_tuple) > 0 else None
            else: result_latent = result_tuple
            
            assert result_latent is not None, "Failed to get valid latent from common_ksampler"

            if isinstance(result_latent, dict):
                if "samples" in result_latent:
                    samples = result_latent["samples"]
                    if isinstance(samples, torch.Tensor):
                        if samples.dim() == 3: samples = samples.unsqueeze(0)
                        result_latent = samples
                    else: raise TypeError(f"Expected tensor but got {type(samples).__name__} in 'samples' key")
                else: raise KeyError("Result dictionary does not contain 'samples' key")
            elif isinstance(result_latent, torch.Tensor):
                if result_latent.dim() == 3: result_latent = result_latent.unsqueeze(0)
            else: raise TypeError(f"Unsupported result type: {type(result_latent).__name__}")

            samples_dict = {"samples": result_latent} if isinstance(result_latent, torch.Tensor) else result_latent
            output_image = VAEDecode().decode(vae, samples_dict)[0]
            all_output_images.append(output_image)
            results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
            all_results.extend(results)

        final_output = torch.cat(all_output_images, dim=0) if all_output_images else None
        context = new_context(context, latent=samples_dict,pos=pos,positive=positive, negative=negative,images=final_output)

        if image_output == "None": return (context, None, samples_dict)
        elif image_output in ("Hide", "Hide/Save"): return {"ui": {}, "result": (context, final_output, samples_dict)}
        else: return {"ui": {"images": all_results}, "result": (context, final_output, samples_dict)}
    

class chx_Ksampler_Kontext_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "add_noise": (["enable", "disable"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "-1 means no change"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 1000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "LATENT")
    RETURN_NAMES = ("context", "image", "latent")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"
    OUTPUT_NODE = True

    def run(self, context, seed, steps=0, image_output="Preview", image=None, denoise=1, mask=None, prompt_weight=0.5, pos="", auto_adjust_image=False, prompt=None, extra_pnginfo=None, add_noise="enable", start_at_step=0, end_at_step=0, return_with_leftover_noise="disable"):
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")
        if steps == 0 or steps==-1: steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        guidance = context.get("guidance", 3.5)

        force_full_denoise = False if return_with_leftover_noise == "enable" else True
        disable_noise = True if add_noise == "disable" else False

        if image is None: image = context.get("images", None)
        assert image is not None, "Image must be provided or exist in the context."

        if image.dim() == 3: image = image.unsqueeze(0)

        all_output_images = []
        all_results = []

        for i in range(image.shape[0]):
            img = image[i:i+1]
            pixels = img
            if auto_adjust_image:
                width = img.shape[2]
                height = img.shape[1]
                aspect_ratio = width / height
                _, target_width, target_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
                scaled_image = comfy.utils.common_upscale(img.movedim(-1, 1), target_width, target_height, "lanczos", "center").movedim(1, -1)
                pixels = scaled_image[:, :, :, :3].clamp(0, 1)
            
            encoded_latent = vae.encode(pixels)[0]
            if encoded_latent.dim() == 3: encoded_latent = encoded_latent.unsqueeze(0)
            latent = {"samples": encoded_latent}

            positive = None
            if pos and pos.strip(): positive, = CLIPTextEncode().encode(clip, pos)
            else: positive = context.get("positive", None)

            if positive is not None and prompt_weight > 0:
                influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                scaled_latent = latent["samples"] * influence
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            if mask is not None: latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

            context = new_context(context, positive=positive, latent=latent)
            negative = context.get("negative", None)

            result_tuple = common_ksampler(model, seed + i, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]

            if isinstance(result_tuple, dict):
                result_latent = result_tuple["samples"] if "samples" in result_tuple else result_tuple
            elif isinstance(result_tuple, (list, tuple)):
                result_latent = result_tuple[0] if len(result_tuple) > 0 else None
            else: result_latent = result_tuple
            
            assert result_latent is not None, "Failed to get valid latent from common_ksampler"

            if isinstance(result_latent, dict):
                if "samples" in result_latent:
                    samples = result_latent["samples"]
                    if isinstance(samples, torch.Tensor):
                        if samples.dim() == 3: samples = samples.unsqueeze(0)
                        result_latent = samples
                    else: raise TypeError(f"Expected tensor but got {type(samples).__name__} in 'samples' key")
                else: raise KeyError("Result dictionary does not contain 'samples' key")
            elif isinstance(result_latent, torch.Tensor):
                if result_latent.dim() == 3: result_latent = result_latent.unsqueeze(0)
            else: raise TypeError(f"Unsupported result type: {type(result_latent).__name__}")

            samples_dict = {"samples": result_latent} if isinstance(result_latent, torch.Tensor) else result_latent
            output_image = VAEDecode().decode(vae, samples_dict)[0]
            all_output_images.append(output_image)
            results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
            all_results.extend(results)

        final_output = torch.cat(all_output_images, dim=0) if all_output_images else None
        context = new_context(context, latent=samples_dict,pos=pos,positive=positive, negative=negative, images=final_output)

        if image_output == "None": return (context, None, samples_dict)
        elif image_output in ("Hide", "Hide/Save"): return {"ui": {}, "result": (context, final_output, samples_dict)}
        else: return {"ui": {"images": all_results}, "result": (context, final_output, samples_dict)}


class pre_Kontext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK",),
                "prompt_weight":("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "smoothness":("INT", {"default": 0,  "min":0, "max": 10, "step": 0.1,}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),  # 新增的输入开关
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","LATENT" )
    RETURN_NAMES = ("context","latent" )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool"

    def process(self, context=None, image=None, mask=None, prompt_weight=0.5, pos="", smoothness=0, auto_adjust_image=True):  # 添加参数


        vae = context.get("vae", None)
        clip = context.get("clip", None)

        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)



        if image is None:
            image = context.get("images", None)
            if  image is None:
                return (context,None)


        image=kontext_adjust_image_resolution(image, auto_adjust_image)[0]

        encoded_latent = vae.encode(image)  #
        latent = {"samples": encoded_latent}

        if positive is not None:
            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
            scaled_latent = latent["samples"] * influence
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [scaled_latent]},  append=True)

        if mask is not None:
            mask = tensor2pil(mask)
            if not isinstance(mask, Image.Image):
                raise TypeError("mask is not a valid PIL Image object")
            feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
            mask = pil2tensor(feathered_image)
            latent = {"samples": encoded_latent,"noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])) }

        context = new_context(context, positive=positive, latent=latent)

        return (context,latent)
    


class XXXchx_Ksampler_Kontext_inpaint:# 遮罩选项，可参与采样或不参与采样
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "denoise": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps": ("INT", {"default": -1, "min": -1, "max": 10000,"tooltip": "-1 means no change"}),
            #"repaint_mode": (["latent_image", "latent_blank"],),
            "mask_extend": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            "feather":("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "prompt_weight":("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "mask_in_sampling": ("BOOLEAN", {"default": True,}),
            "crop_mask_as_inpaint_area": ("BOOLEAN", {"default": False,  "tooltip": "Below options are only valid when it is true."}),
            "crop_area_scale": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 8}),
            "crop_area_extend": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),


        },
                
            "optional": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "pos":("STRING",{"multiline": True, "default": ""}),
            }
                     
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "IMAGE", )
    RETURN_NAMES = ('context', 'result_img', 'sample_img',)
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"

    def mask_crop(self, image, mask, crop_area_extend, crop_area_scale=0):
        

        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)
        mask_array = np.array(mask_pil) > 0
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            return (image, None, mask)
        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
        x0 -= crop_area_extend
        y0 -= crop_area_extend
        x1 += crop_area_extend
        y1 += crop_area_extend
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        if crop_area_scale > 0:
            min_size = min(cropped_image_pil.size)
            if min_size < crop_area_scale or min_size > crop_area_scale:
                scale_ratio = crop_area_scale / min_size
                new_size = (int(cropped_image_pil.width * scale_ratio), int(cropped_image_pil.height * scale_ratio))
                cropped_image_pil = cropped_image_pil.resize(new_size, Image.LANCZOS)
                cropped_mask_pil = cropped_mask_pil.resize(new_size, Image.LANCZOS)

        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        qtch = image
        qtzz = mask
        return (cropped_image_tensor, cropped_mask_tensor, (y0, y1, x0, x1) ,qtch ,qtzz )

    def encode(self, vae, image, mask, mask_extend=6, repaint_mode="latent_blank"):
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                            size=(image.shape[1], image.shape[2]), mode="bilinear")
        if repaint_mode == "latent_blank":
            image = image.clone()
            if image.shape[1] != x or image.shape[2] != y:
                x_offset = (image.shape[1] % 8) // 2
                y_offset = (image.shape[2] % 8) // 2
                image = image[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
        if mask_extend == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, mask_extend, mask_extend))
            padding = math.ceil((mask_extend - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        if repaint_mode == "latent_blank":
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        t = vae.encode(image)
        return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}, None
    
    def paste_cropped_image_with_mask(self, original_image, cropped_image, crop_coords, mask, MHmask, feather):
        y0, y1, x0, x1 = crop_coords
        original_image_pil = tensor2pil(original_image)
        cropped_image_pil = tensor2pil(cropped_image)
        mask_pil = tensor2pil(mask)
        crop_width = x1 - x0
        crop_height = y1 - y0
        crop_size = (crop_width, crop_height)

        cropped_image_pil = cropped_image_pil.resize(crop_size, Image.LANCZOS)
        mask_pil = mask_pil.resize(crop_size, Image.LANCZOS)

        mask_binary = mask_pil.convert('L')
        mask_rgba = mask_binary.convert('RGBA')
        blurred_mask = mask_rgba
        transparent_mask = mask_binary
        blurred_mask = mask_binary
        cropped_image_pil = cropped_image_pil.convert('RGBA')
        original_image_pil = original_image_pil.convert('RGBA')
        original_image_pil.paste(cropped_image_pil, (x0, y0), mask=blurred_mask)
        ZT_image_pil=original_image_pil.convert('RGB')
        IMAGEEE = pil2tensor(ZT_image_pil)        
        mask_ecmhpil= tensor2pil(MHmask)   
        mask_ecmh = mask_ecmhpil.convert('L')
        mask_ecrgba = tensor2pil(MHmask)   
        maskecmh = None
        if feather is not None:
            if feather > -1:
                maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather))
        dyzz = pil2tensor(maskecmh)
        maskeccmh = pil2tensor(maskecmh)
        destination = original_image
        source = IMAGEEE
        dyyt = source
        multiplier = 8
        resize_source = True
        mask = dyzz
        destination = destination.clone().movedim(-1, 1)
        source=source.clone().movedim(-1, 1)
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
        x=0
        y=0
        x = int(x)
        y = int(y)  
        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        zztx = destination.movedim(1, -1)
        return zztx,dyzz,dyyt


    def sample(self, context, seed, steps, image=None, mask=None, pos=None, prompt_weight=0.5, mask_extend=6, 
            denoise=1.0, crop_mask_as_inpaint_area=False, crop_area_extend=0, crop_area_scale=0, 
            feather=1, mask_in_sampling=True):
        """
        采样方法，添加mask_in_sampling参数控制遮罩是否参与实际采样计算
        """
        repaint_mode="latent_image"
        guidance = context.get("guidance", 3.5)
        if steps == 0 or steps==-1: steps = context.get("steps")
        model = context.get("model", None)
        model = DifferentialDiffusion().apply(model)[0]      
        vae = context.get("vae", None)
        
        clip = context.get("clip", None)
        positive = None
        if pos and pos.strip():
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        negative = context.get("negative", None)
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler_name = context.get("sampler", None)
        scheduler = context.get("scheduler", None)

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent", None)

        if mask is None or torch.all(mask == 0):
            # 无遮罩的情况保持不变
            if positive is not None and prompt_weight > 0:
                influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                scaled_latent = latent["samples"] * influence
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=denoise)[0]
            output_image = decode(vae, latent)[0]
            
            context = new_context(context, latent=latent, positive=positive, negative=negative, images=output_image)
            zztx = output_image
            decoded_image = output_image
            return (context, zztx, decoded_image, None)

        if mask is not None:
            latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            original_image = image
            hqccimage = tensor2pil(image)
            sfmask = tensor2pil(mask)
            sfhmask = sfmask.resize(hqccimage.size, Image.LANCZOS)
            mask = pil2tensor(sfhmask)
            MHmask = mask
            
            if crop_mask_as_inpaint_area:
                # 执行图像裁剪，获取裁剪区域和坐标
                image, mask, crop_coords, bytx, byzz = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                
                # 对裁剪后的图像进行编码
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                
                # 为裁剪区域创建专门的positive条件
                if positive is not None and prompt_weight > 0:
                    influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                    scaled_latent = latent_image["samples"] * influence
                    positive_crop = node_helpers.conditioning_set_values(positive.copy(), {"reference_latents": [scaled_latent]}, append=True)
                    positive_crop = node_helpers.conditioning_set_values(positive_crop, {"guidance": guidance})
                else:
                    positive_crop = positive
                
                # 根据mask_in_sampling决定是否传递noise_mask给ksampler
                if not mask_in_sampling and "noise_mask" in latent_image:
                    # 创建一个不包含noise_mask的副本，确保遮罩不参与采样
                    latent_image_no_mask = latent_image.copy()
                    latent_image_no_mask.pop("noise_mask")
                    
                    # 使用不包含遮罩的latent进行采样
                    samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            positive_crop, negative, latent_image_no_mask, denoise=denoise)
                else:
                    # 使用原始latent_image（包含noise_mask），遮罩参与采样
                    samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                            positive_crop, negative, latent_image, denoise=denoise)
                
                decoded_image = vae.decode(samples[0]["samples"])
                
                # 确保采样结果正确贴回原图
                if not mask_in_sampling:
                    # 当遮罩不参与采样时，使用全1遮罩确保完全替换裁剪区域
                    full_mask = torch.ones_like(mask)
                    final_image, dyzz, dyyt = self.paste_cropped_image_with_mask(
                        original_image, decoded_image, crop_coords, full_mask, full_mask, feather
                    )
                else:
                    # 当遮罩参与采样时，使用原始裁剪遮罩进行粘贴
                    final_image, dyzz, dyyt = self.paste_cropped_image_with_mask(
                        original_image, decoded_image, crop_coords, mask, MHmask, feather
                    )
                
                # 更新latent和context
                latent = VAEEncode().encode(vae, final_image)[0]
                context = new_context(context, latent=latent, positive=positive, negative=negative, images=final_image)
                return (context, final_image, decoded_image)     

            else:
                # 非裁剪模式保持原有逻辑不变
                bytx, byzz, crop_coords, image, mask = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                
                # 对完整图像应用positive条件
                if positive is not None and prompt_weight > 0:
                    influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                    scaled_latent = latent_image["samples"] * influence
                    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                    positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
                
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                
                # 后处理逻辑保持不变
                mask_ecrgba = tensor2pil(mask)   
                maskecmh = None
                if feather is not None:
                    if feather > -1:
                        maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather))
                dyzz = pil2tensor(maskecmh)
                mask = dyzz
                
                destination = original_image
                source = decoded_image       
                multiplier = 8
                resize_source = True

                destination = destination.clone().movedim(-1, 1)
                source = source.clone().movedim(-1, 1)
                source = source.to(destination.device)
                if resize_source:
                    source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

                source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
                x = 0
                y = 0
                x = int(x)
                y = int(y)  
                x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
                y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

                left, top = (x // multiplier, y // multiplier)
                right, bottom = (left + source.shape[3], top + source.shape[2])

                if mask is None:
                    mask = torch.ones_like(source)
                else:
                    mask = mask.to(destination.device, copy=True)
                    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
                    mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
                
                visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y))
                mask = mask[:, :, :visible_height, :visible_width]
                inverse_mask = torch.ones_like(mask) - mask
                source_portion = mask * source[:, :, :visible_height, :visible_width]
                destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]
                destination[:, :, top:bottom, left:right] = source_portion + destination_portion
                zztx = destination.movedim(1, -1)
            
                latent = VAEEncode().encode(vae, zztx)[0]
                context = new_context(context, latent=latent, positive=positive, negative=negative, pos=pos, images=zztx)
                return (context, zztx, decoded_image)


class chx_Ksampler_Kontext_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "denoise": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps": ("INT", {"default": -1, "min": -1, "max": 10000,"tooltip": "-1 means no change"}),
            "mask_extend": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            "feather_mask":("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "羽化遮罩边缘"}),
            "prompt_weight":("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "crop_mask_as_inpaint_area": ("BOOLEAN", {"default": False,  "tooltip": "Below options are only valid when it is true."}),
            "crop_area_scale": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 8}),
            "crop_area_extend": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),
            "feather_image":("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "羽化贴图边缘"}),
        },
                
            "optional": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "pos":("STRING",{"multiline": True, "default": ""}),
            }
                     
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "IMAGE", )
    RETURN_NAMES = ('context', 'result_img', 'sample_img',)
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"

    def mask_crop(self, image, mask, crop_area_extend, crop_area_scale=0):
        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)
        mask_array = np.array(mask_pil) > 0
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            return (image, None, mask)
        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
        x0 -= crop_area_extend
        y0 -= crop_area_extend
        x1 += crop_area_extend
        y1 += crop_area_extend
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        if crop_area_scale > 0:
            min_size = min(cropped_image_pil.size)
            if min_size < crop_area_scale or min_size > crop_area_scale:
                scale_ratio = crop_area_scale / min_size
                new_size = (int(cropped_image_pil.width * scale_ratio), int(cropped_image_pil.height * scale_ratio))
                cropped_image_pil = cropped_image_pil.resize(new_size, Image.LANCZOS)
                cropped_mask_pil = cropped_mask_pil.resize(new_size, Image.LANCZOS)

        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        qtch = image
        qtzz = mask
        return (cropped_image_tensor, cropped_mask_tensor, (y0, y1, x0, x1) ,qtch ,qtzz )

    def encode(self, vae, image, mask, mask_extend=6, repaint_mode="latent_blank"):
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                            size=(image.shape[1], image.shape[2]), mode="bilinear")
        if repaint_mode == "latent_blank":
            image = image.clone()
            if image.shape[1] != x or image.shape[2] != y:
                x_offset = (image.shape[1] % 8) // 2
                y_offset = (image.shape[2] % 8) // 2
                image = image[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
        if mask_extend == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, mask_extend, mask_extend))
            padding = math.ceil((mask_extend - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        if repaint_mode == "latent_blank":
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        t = vae.encode(image)
        return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}, None
    
    def paste_cropped_image_with_mask(self, original_image, cropped_image, crop_coords, mask, MHmask, feather_mask, feather_image, crop_mask_as_inpaint_area):
        y0, y1, x0, x1 = crop_coords
        original_image_pil = tensor2pil(original_image)
        cropped_image_pil = tensor2pil(cropped_image)
        mask_pil = tensor2pil(mask)
        crop_width = x1 - x0
        crop_height = y1 - y0
        crop_size = (crop_width, crop_height)

        cropped_image_pil = cropped_image_pil.resize(crop_size, Image.LANCZOS)
        mask_pil = mask_pil.resize(crop_size, Image.LANCZOS)

        # 处理遮罩羽化
        mask_binary = mask_pil.convert('L')
        mask_rgba = mask_binary.convert('RGBA')
        if feather_mask > 0:
            blurred_mask = mask_rgba.filter(ImageFilter.GaussianBlur(feather_mask))
        else:
            blurred_mask = mask_rgba
            
        cropped_image_pil = cropped_image_pil.convert('RGBA')
        original_image_pil = original_image_pil.convert('RGBA')
        original_image_pil.paste(cropped_image_pil, (x0, y0), mask=blurred_mask)
        ZT_image_pil=original_image_pil.convert('RGB')
        IMAGEEE = pil2tensor(ZT_image_pil)        
        mask_ecmhpil= tensor2pil(MHmask)   
        mask_ecmh = mask_ecmhpil.convert('L')
        mask_ecrgba = mask_ecmhpil
        
        # 处理合成图羽化（仅在裁剪模式下应用）
        if crop_mask_as_inpaint_area and feather_image > 0:
            maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather_image))
        else:
            maskecmh = mask_ecrgba
            
        dyzz = pil2tensor(maskecmh)
        maskeccmh = pil2tensor(maskecmh)
        destination = original_image
        source = IMAGEEE
        dyyt = source
        multiplier = 8
        resize_source = True
        mask = dyzz
        destination = destination.clone().movedim(-1, 1)
        source=source.clone().movedim(-1, 1)
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
        x=0
        y=0
        x = int(x)
        y = int(y)  
        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        zztx = destination.movedim(1, -1)
        return zztx,dyzz,dyyt


    def sample(self, context, seed, steps,image=None, mask=None, pos=None, prompt_weight=0.5, mask_extend=6,  denoise=1.0, crop_mask_as_inpaint_area=False, crop_area_extend=0, crop_area_scale=0, feather_mask=1, feather_image=1):
        
        repaint_mode="latent_image"
        guidance = context.get("guidance", 3.5)
        if steps == 0 or steps==-1: steps = context.get("steps")
        model = context.get("model", None)
        model = DifferentialDiffusion().apply(model)[0]      
        vae = context.get("vae", None)
        
        clip = context.get("clip", None)
        positive = None
        if pos and pos.strip():
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        

        negative = context.get("negative", None)
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler_name = context.get("sampler", None)
        scheduler = context.get("scheduler", None)

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent", None)

        if mask is None or torch.all(mask == 0):
            # 无遮罩的情况保持不变
            if positive is not None and prompt_weight > 0:
                influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                scaled_latent = latent["samples"] * influence
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=denoise)[0]
            output_image = decode(vae, latent)[0]
            
            context = new_context(context, latent=latent, positive=positive, negative=negative, images=output_image)
            zztx = output_image
            decoded_image = output_image
            return (context, zztx, decoded_image, None)

        if mask is not None:
            latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            original_image = image
            hqccimage = tensor2pil(image)
            sfmask = tensor2pil(mask)
            sfhmask = sfmask.resize(hqccimage.size, Image.LANCZOS)
            mask = pil2tensor(sfhmask)
            MHmask = mask
            
            if crop_mask_as_inpaint_area:
                # 执行图像裁剪，获取裁剪区域和坐标
                image, mask, crop_coords, bytx, byzz = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                
                # 对裁剪后的图像进行编码
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                
                # 为裁剪区域创建专门的positive条件
                if positive is not None and prompt_weight > 0:
                    influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                    # 只对裁剪区域的latent应用positive条件
                    scaled_latent = latent_image["samples"] * influence
                    positive_crop = node_helpers.conditioning_set_values(positive.copy(), {"reference_latents": [scaled_latent]}, append=True)
                    positive_crop = node_helpers.conditioning_set_values(positive_crop, {"guidance": guidance})
                else:
                    positive_crop = positive
                
                # 使用裁剪区域的positive条件进行采样
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                        positive_crop, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                
                # 将生成的结果粘贴回原图（应用羽化参数）
                final_image, dyzz, dyyt = self.paste_cropped_image_with_mask(
                    original_image, decoded_image, crop_coords, mask, MHmask, feather_mask, feather_image, crop_mask_as_inpaint_area
                )
                
                # 更新latent和context
                latent = VAEEncode().encode(vae, final_image)[0]
                context = new_context(context, latent=latent, positive=positive, negative=negative, images=final_image)
                return (context, final_image, decoded_image)     

            else:
                # 非裁剪模式保持原有逻辑不变
                bytx, byzz, crop_coords, image, mask = self.mask_crop(image, mask, crop_area_extend, crop_area_scale)
                latent_image, _ = self.encode(vae, image, mask, mask_extend, repaint_mode)
                
                # 对完整图像应用positive条件
                if positive is not None and prompt_weight > 0:
                    influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                    scaled_latent = latent_image["samples"] * influence
                    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                    positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
                
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                
                # 非裁剪模式下只应用feather_mask
                mask_ecrgba = tensor2pil(mask)   
                if feather_mask > 0:
                    maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather_mask))
                else:
                    maskecmh = mask_ecrgba
                    
                dyzz = pil2tensor(maskecmh)
                mask = dyzz
                
                destination = original_image
                source = decoded_image       
                multiplier = 8
                resize_source = True

                destination = destination.clone().movedim(-1, 1)
                source = source.clone().movedim(-1, 1)
                source = source.to(destination.device)
                if resize_source:
                    source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

                source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
                x = 0
                y = 0
                x = int(x)
                y = int(y)  
                x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
                y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

                left, top = (x // multiplier, y // multiplier)
                right, bottom = (left + source.shape[3], top + source.shape[2])

                if mask is None:
                    mask = torch.ones_like(source)
                else:
                    mask = mask.to(destination.device, copy=True)
                    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
                    mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
                
                visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y))
                mask = mask[:, :, :visible_height, :visible_width]
                inverse_mask = torch.ones_like(mask) - mask
                source_portion = mask * source[:, :, :visible_height, :visible_width]
                destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]
                destination[:, :, top:bottom, left:right] = source_portion + destination_portion
                zztx = destination.movedim(1, -1)
            
                latent = VAEEncode().encode(vae, zztx)[0]
                context = new_context(context, latent=latent, positive=positive, negative=negative, pos=pos, images=zztx)
                return (context, zztx, decoded_image)
            








class XXXInpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1})
            }
        }

    CATEGORY = "Apt_Preset/chx_ksample"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitch")
    FUNCTION = "inpaint_crop"

    def inpaint_crop(self, image, mask, expand, rescale_factor, smoothness):
        import cv2
        import numpy as np
        import torch
        from PIL import Image, ImageFilter  # 添加缺失的导入

        # Convert tensors to numpy
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)

        # 保存原始图像的深拷贝
        original_image = image_np.copy()

        # Apply smoothing
        if smoothness > 0:
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(smoothness))
            mask_np = np.array(mask_pil).astype(np.uint8)

        # Apply expand
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if expand > 0:
            mask_np = cv2.dilate(mask_np, kernel, iterations=expand)
        elif expand < 0:
            mask_np = cv2.erode(mask_np, kernel, iterations=-expand)

        # Find bounding box
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty after processing")

        x, y, w, h = cv2.boundingRect(coords)
        original_h, original_w = image_np.shape[0], image_np.shape[1]

        # Scale image and mask
        new_w = int(w * rescale_factor)
        new_h = int(h * rescale_factor)
        x_center = x + w // 2
        y_center = y + h // 2
        x_new = max(0, x_center - new_w // 2)
        y_new = max(0, y_center - new_h // 2)
        x_end = min(original_w, x_new + new_w)
        y_end = min(original_h, y_new + new_h)

        # Crop image and mask
        cropped_image = image_np[y_new:y_end, x_new:x_end]
        cropped_mask = mask_np[y_new:y_end, x_new:x_end]

        # Convert back to tensor
        cropped_image_tensor = torch.from_numpy(cropped_image / 255.0).float()
        cropped_mask_tensor = torch.from_numpy(cropped_mask / 255.0).float().unsqueeze(0)

        # Stitch info - 添加原始图像数据
        stitch = {
            "original_shape": (original_h, original_w),
            "crop_position": (x_new, y_new),
            "crop_size": (x_end - x_new, y_end - y_new),
            "original_mask": mask_np,
            "original_image": original_image  # 添加原始图像到stitch数据中
        }

        return (cropped_image_tensor.unsqueeze(0), cropped_mask_tensor, stitch)


class sampler_InpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),
            }
        }

    CATEGORY = "Apt_Preset/chx_ksample"
    RETURN_TYPES = ("IMAGE", "IMAGE",)  
    RETURN_NAMES = ("image","cropped_iamge", )
    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, inpainted_image, mask, stitch):


        # 提取 stitch 信息
        original_h, original_w = stitch["original_shape"]
        x, y = stitch["crop_position"]
        w, h = stitch["crop_size"]
        original_mask = stitch["original_mask"]
        original_image = stitch["original_image"]  # 获取原始图像

        # 转换为 numpy
        inpainted_np = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)

        # Resize 到裁剪区域大小
        inpainted_np = cv2.resize(inpainted_np, (w, h))
        mask_np = cv2.resize(mask_np, (w, h))

        # 创建裁剪区域的合成图（带透明通道）
        cropped_merged = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
        cropped_merged[:, :, :3] = inpainted_np  # RGB
        cropped_merged[:, :, 3] = mask_np  # Alpha

        # 创建最终图像（带透明度混合）
        result = np.zeros((original_h, original_w, 4), dtype=np.uint8)  # RGBA
        
        # 使用原始图像作为背景
        result[:, :, :3] = original_image.copy()
        result[:, :, 3] = 255  # 初始透明度为完全不透明

        # 获取原始图像中对应区域
        original_region = result[y:y+h, x:x+w, :3].copy()
        
        # 准备裁切图和遮罩
        inpainted_region = cropped_merged[:, :, :3]
        alpha = cropped_merged[:, :, 3:4] / 255.0
        
        # 正确的alpha混合：将修复区域与原图区域根据遮罩进行混合
        blended_region = inpainted_region * alpha + original_region * (1 - alpha)
        blended_region = blended_region.astype(np.uint8)
        
        # 将混合后的区域放回结果图像
        result[y:y+h, x:x+w, :3] = blended_region
        result[y:y+h, x:x+w, 3] = 255  # 该区域已混合，设置为不透明

        # 转为 tensor 格式
        final_image_tensor = torch.from_numpy(result[:, :, :3] / 255.0).float().unsqueeze(0)
        cropped_merged_tensor = torch.from_numpy(cropped_merged / 255.0).float().unsqueeze(0)

        return ( final_image_tensor,cropped_merged_tensor,)



class sampler_InpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1})
            }
        }

    CATEGORY = "Apt_Preset/chx_ksample"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitch")
    FUNCTION = "inpaint_crop"

    def inpaint_crop(self, image, mask, expand_width, expand_height, rescale_factor, smoothness):
        import cv2
        import numpy as np
        import torch
        from PIL import Image, ImageFilter

        # Convert tensors to numpy
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)

        # 保存原始图像的深拷贝
        original_image = image_np.copy()
        original_mask = mask_np.copy()

        # Find bounding box of the mask
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty after processing")

        x, y, w, h = cv2.boundingRect(coords)
        original_h, original_w = image_np.shape[0], image_np.shape[1]
        
        # 计算遮罩中心
        mask_center_x = x + w // 2
        mask_center_y = y + h // 2

        # 应用扩展（对称地围绕遮罩中心扩展裁剪区域）
        new_half_width = (w // 2) + expand_width
        new_half_height = (h // 2) + expand_height
        
        # 计算新的裁剪区域坐标，确保不超出原始图像边界
        x_new = max(0, mask_center_x - new_half_width)
        y_new = max(0, mask_center_y - new_half_height)
        x_end = min(original_w, mask_center_x + new_half_width)
        y_end = min(original_h, mask_center_y + new_half_height)
        
        # 调整后的裁剪尺寸
        new_w = x_end - x_new
        new_h = y_end - y_new

        # 裁剪图像
        cropped_image = image_np[y_new:y_end, x_new:x_end]

        # 裁剪遮罩区域（可能超出实际裁剪区域）
        mask_x_start = max(0, x - x_new)
        mask_y_start = max(0, y - y_new)
        mask_x_end = min(new_w, (x + w) - x_new)
        mask_y_end = min(new_h, (y + h) - y_new)
        
        # 创建与裁剪图像相同大小的新遮罩，初始化为全0
        new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # 如果遮罩区域在裁剪区域内，则复制相应部分
        if mask_x_start < mask_x_end and mask_y_start < mask_y_end:
            new_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = original_mask[
                y + mask_y_start - mask_y_start:y + mask_y_end - mask_y_start,
                x + mask_x_start - mask_x_start:x + mask_x_end - mask_x_start
            ]

        # 应用缩放因子
        if rescale_factor != 1.0:
            # 计算缩放后的尺寸
            scaled_w = int(new_w * rescale_factor)
            scaled_h = int(new_h * rescale_factor)
            
            # 缩放图像
            cropped_image = cv2.resize(
                cropped_image, 
                (scaled_w, scaled_h), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # 缩放遮罩
            new_mask = cv2.resize(
                new_mask, 
                (scaled_w, scaled_h), 
                interpolation=cv2.INTER_LINEAR
            )

        # 仅对最终的遮罩应用平滑处理
        if smoothness > 0:
            mask_pil = Image.fromarray(new_mask)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(smoothness))
            new_mask = np.array(mask_pil).astype(np.uint8)

        # Convert back to tensor
        cropped_image_tensor = torch.from_numpy(cropped_image / 255.0).float()
        cropped_mask_tensor = torch.from_numpy(new_mask / 255.0).float().unsqueeze(0)

        # Stitch info
        stitch = {
            "original_shape": (original_h, original_w),
            "crop_position": (x_new, y_new),
            "crop_size": (new_w, new_h),
            "scaled_size": (scaled_w, scaled_h) if rescale_factor != 1.0 else (new_w, new_h),
            "original_mask": original_mask,
            "original_image": original_image,
            "mask_info": {
                "position": (x, y),
                "size": (w, h),
                "relative_position": (mask_x_start, mask_y_start),
                "relative_size": (mask_x_end - mask_x_start, mask_y_end - mask_y_start)
            }
        }

        return (cropped_image_tensor.unsqueeze(0), cropped_mask_tensor, stitch)    
