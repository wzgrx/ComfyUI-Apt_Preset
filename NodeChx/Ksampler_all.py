# 标准库
import math
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from telnetlib import OUTMRK
import enum

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
from comfy.model_base import BaseModel
from comfy.cli_args import args
import folder_paths
import node_helpers
import latent_preview
import nodes
from nodes import CLIPTextEncode, common_ksampler, VAEDecode, VAEEncode, ImageScale, KSampler, SetLatentNoiseMask, KSamplerAdvanced
from einops import rearrange
import kornia.filters
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
import matplotlib
import matplotlib.scale
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing
from typing import Any, Dict, Optional, Tuple, Union, cast
import seam_carving
from comfy.samplers import KSAMPLER


matplotlib.use('Agg')



# 本地模块
from ..main_unit import *



SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]




class basic_Ksampler_batch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
    CATEGORY = "Apt_Preset/ksampler"


    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        latent_samples = latent["samples"]
        # Convert to float32 if not already
        if latent_samples.dtype != torch.float32:
            latent_samples = latent_samples.to(torch.float32)
        num_samples = latent_samples.shape[0]

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
    CATEGORY = "Apt_Preset/ksampler"


    def sample(self, seed, image, image_denoise, mask, mask_denoise, image_cfg, mask_cfg, model_img=None,model_mask=None, prompt_img=None,prompt_mask=None,
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
    CATEGORY = "Apt_Preset/ksampler"

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
    CATEGORY = "Apt_Preset/ksampler"

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
    CATEGORY = "Apt_Preset/ksampler"

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
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Hide"}),
                    },
                "optional": {"image_optional": ("IMAGE",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            }
            
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = "Apt_Preset/ksampler"

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

class StyleAlignedBatchAlign:
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
    CATEGORY = "Style"

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
    CATEGORY = "Apt_Preset/ksampler"
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
        model_patched = StyleAlignedBatchAlign().patch(model2, share_norm, share_attn, scale)
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



#region-----------------------pre inpaint ksampler-------------------------------
def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def to_torch(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        mask = mask_unsqueeze(mask)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask


def to_comfy(image: Tensor):
    return image.permute(0, 2, 3, 1)  # BCHW -> BHWC


def mask_floor(mask: Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


# torch pad does not support padding greater than image size with "reflect" mode
def pad_reflect_once(x: Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = F.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(additional_padding), mode="constant")
    return x


def resize_square(image: Tensor, mask: Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma))


def binary_erosion(mask: Tensor, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def binary_dilation(mask: Tensor, radius: int):
    kernel = torch.ones(1, radius * 2 + 1, device=mask.device)
    mask = kornia.filters.filter2d_separable(mask, kernel, kernel, border_type="constant")
    mask = (mask > 0).to(mask.dtype)
    return mask


def make_odd(x):
    if x > 0 and x % 2 == 0:
        return x + 1
    return x







class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    for key in to_load.values():
        if value := lora.get(key, None):
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for x in lora if x not in loaded_keys)
    if not_loaded > 0:
        print(
            f"[ApplyFooocusInpaint] {len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model."
        )
    return patch_dict


if not hasattr(comfy.lora, "calculate_weight") and hasattr(ModelPatcher, "calculate_weight"):
    too_old_msg = "comfyui-inpaint-nodes requires a newer version of ComfyUI (v0.1.1 or later), please update!"
    raise RuntimeError(too_old_msg)


original_calculate_weight = comfy.lora.calculate_weight
injected_model_patcher_calculate_weight = False


def calculate_weight_patched(patches, weight, key, intermediate_dtype=torch.float32):
    remaining = []

    for p in patches:
        alpha = p[0]
        v = p[1]

        is_fooocus_patch = isinstance(v, tuple) and len(v) == 2 and v[0] == "fooocus"
        if not is_fooocus_patch:
            remaining.append(p)
            continue

        if alpha != 0.0:
            v = v[1]
            w1 = cast_to_device(v[0], weight.device, torch.float32)
            if w1.shape == weight.shape:
                w_min = cast_to_device(v[1], weight.device, torch.float32)
                w_max = cast_to_device(v[2], weight.device, torch.float32)
                w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
            else:
                print(
                    f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged ({w1.shape} != {weight.shape})"
                )

    if len(remaining) > 0:
        return original_calculate_weight(remaining, weight, key, intermediate_dtype)
    return weight


def inject_patched_calculate_weight():
    global injected_model_patcher_calculate_weight
    if not injected_model_patcher_calculate_weight:
        print(
            "[comfyui-inpaint-nodes] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight"
        )
        comfy.lora.calculate_weight = calculate_weight_patched
        injected_model_patcher_calculate_weight = True



MODELS_DIR = os.path.join(folder_paths.models_dir, "inpaint")
if "inpaint" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["inpaint"]
folder_paths.folder_names_and_paths["inpaint"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


class pre_inpaint_xl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "context": ("RUN_CONTEXT",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
                "head": (folder_paths.get_filename_list("inpaint"), {"default": "fooocus_inpaint_head.pth"}),
                "patch": (folder_paths.get_filename_list("inpaint"), {"default": "inpaint_v26.fooocus.patch"}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    CATEGORY = "Apt_Preset/ksampler"
    FUNCTION = "patch"

    _inpaint_head_feature: Tensor | None = None
    _inpaint_block: Tensor | None = None


    def _input_block_patch(self, h: Tensor, transformer_options: dict):
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self._inpaint_head_feature is not None
                batch = h.shape[0] // self._inpaint_head_feature.shape[0]
                self._inpaint_block = self._inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h


    def patch(self, head: str, patch: str,  pixels, mask,context):

        model: ModelPatcher=context.get("model")
        positive=context.get("positive")
        negative=context.get("negative")
        vae=context.get("vae")



        if isinstance(head, list):
            if len(head) > 0:
                head = head[0]  # 选择列表中的第一个元素
            else:
                raise ValueError("The 'head' list is empty.")
        if isinstance(patch, list):
            if len(patch) > 0:
                patch = patch[0]  # 选择列表中的第一个元素
            else:
                raise ValueError("The 'patch' list is empty.")

        # 加载文件的逻辑
        head_file = folder_paths.get_full_path("inpaint", head)
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location="cpu", weights_only=True)
        inpaint_head_model.load_state_dict(sd)

        patch_file = folder_paths.get_full_path("inpaint", patch)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        positive, negative, latent = nodes.InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        latent0=latent
        
        latent = dict(samples=positive[0][1]["concat_latent_image"], noise_mask=latent["noise_mask"].round())

        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()

        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        self._inpaint_head_feature = inpaint_head_model(feed)
        self._inpaint_block = None

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        lora_keys.update({x: x for x in base_model.state_dict().keys()})
        loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(self._input_block_patch)
        patched = m.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in patched)
        if not_patched_count > 0:
            print(f"[ApplyFooocusInpaint] Failed to patch {not_patched_count} keys")

        inject_patched_calculate_weight()
        
        model = DifferentialDiffusion().apply(m)
        
        context = new_context(context, latent=latent0, positive=positive,negative=negative,vae=vae,model=m,)
        return (context,  )       

#endregion-----------------------pre inpaint ksampler-------------------------------



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
    CATEGORY = "Apt_Preset/ksampler"

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


class chx_Ksampler_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            "repaint_mode": (["basic", "fill"],),
            "extended_mask": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            "local_repaint": ("BOOLEAN", {"default": True}),
            "repaint_area": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "extended_repaint_area": ("INT", {"default": 50, "min": 0}),
            "feather":("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
        },
                
            "optional": {
            "model":("MODEL",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            }
            
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "IMAGE",  "MASK")
    RETURN_NAMES = ('context', 'result_img', 'sample_img', 'mask')
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"

    def mask_crop(self, image, mask, extended_repaint_area, repaint_area=0):
        

        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)
        mask_array = np.array(mask_pil) > 0
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            return (image, None, mask)
        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
        x0 -= extended_repaint_area
        y0 -= extended_repaint_area
        x1 += extended_repaint_area
        y1 += extended_repaint_area
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        if repaint_area > 0:
            min_size = min(cropped_image_pil.size)
            if min_size < repaint_area or min_size > repaint_area:
                scale_ratio = repaint_area / min_size
                new_size = (int(cropped_image_pil.width * scale_ratio), int(cropped_image_pil.height * scale_ratio))
                cropped_image_pil = cropped_image_pil.resize(new_size, Image.LANCZOS)
                cropped_mask_pil = cropped_mask_pil.resize(new_size, Image.LANCZOS)

        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        qtch = image
        qtzz = mask
        return (cropped_image_tensor, cropped_mask_tensor, (y0, y1, x0, x1) ,qtch ,qtzz )

    def encode(self, vae, image, mask, extended_mask=6, repaint_mode="fill"):
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                            size=(image.shape[1], image.shape[2]), mode="bilinear")
        if repaint_mode == "fill":
            image = image.clone()
            if image.shape[1] != x or image.shape[2] != y:
                x_offset = (image.shape[1] % 8) // 2
                y_offset = (image.shape[2] % 8) // 2
                image = image[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
        if extended_mask == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, extended_mask, extended_mask))
            padding = math.ceil((extended_mask - 1) / 2)
            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        if repaint_mode == "fill":
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


    def sample(self, context, seed, image=None, model=None,  mask=None, extended_mask=6, repaint_mode="fill", denoise=1.0, local_repaint=False, extended_repaint_area=0, repaint_area=0, feather=1, ):
        

        if model is None :
            model = context.get("model", None)
        vae = context.get ("vae", None)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler_name = context.get("sampler", None)
        scheduler = context.get("scheduler", None)

        if image is not None :
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent", None)


        if mask is None :

            latent = common_ksampler(model,seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=denoise)[0]
            output_image = VAEDecode().decode(vae, latent)[0]
            
            context = new_context(context, latent=latent, images=output_image, )
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
            
            if local_repaint:
                image, mask, crop_coords,bytx, byzz = self.mask_crop(image, mask, extended_repaint_area, repaint_area)
                latent_image, _ = self.encode(vae, image, mask, extended_mask, repaint_mode)
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                final_image,dyzz,dyyt = self.paste_cropped_image_with_mask(original_image, decoded_image, crop_coords, mask, MHmask, feather)
                #return (samples[0], final_image,decoded_image,dyzz)
                
                latent = VAEEncode().encode(vae, final_image)[0]
                
                context = new_context(context, latent = latent, image = final_image)
                return (context, final_image, decoded_image,dyzz)     

            else:
                bytx, byzz, crop_coords,image, mask = self.mask_crop(image, mask, extended_repaint_area, repaint_area)
                latent_image, _ = self.encode(vae, image, mask, extended_mask, repaint_mode)
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                decoded_image = vae.decode(samples[0]["samples"])
                
                mask_ecrgba = tensor2pil(mask)   
                
                maskecmh = None
                if feather is not None:
                    if feather > -1:
                        maskecmh = mask_ecrgba.filter(ImageFilter.GaussianBlur(feather))
                dyzz = pil2tensor(maskecmh)
                maskeccmh = pil2tensor(maskecmh)
                mask = maskeccmh
                destination = original_image
                source = decoded_image       
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
                #return (samples[0], zztx, decoded_image, dyzz)
                
                latent = VAEEncode().encode(vae, zztx)[0]
                
                context = new_context(context, latent = latent, image = zztx)
                return (context, zztx, decoded_image, dyzz)     


#region-----------------------basic node-------------------------------


#region-------------def--------------------------------------------------------------------#

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
matplotlib.use('Agg')


class GraphScale(enum.StrEnum):
    linear = "linear"
    log = "log"




def tensor_to_graph_image(tensor, color="blue", scale: GraphScale=GraphScale.linear):
    SCALE_FUNCTIONS: dict[str, matplotlib.scale.ScaleBase] = {
        GraphScale.linear: matplotlib.scale.LinearScale,
        GraphScale.log: matplotlib.scale.LogScale,
    }
    plt.figure()
    plt.plot(tensor.numpy(), marker='o', linestyle='-', color=color)
    plt.title("Graph from Tensor")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.yscale(scale)
    with BytesIO() as buf:
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).copy()
    plt.close()
    return image


def fibonacci_normalized_descending(n):
    fib_sequence = [0, 1]
    for _ in range(n):
        if n > 1:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    max_value = fib_sequence[-1]
    normalized_sequence = [x / max_value for x in fib_sequence]
    descending_sequence = normalized_sequence[::-1]
    return descending_sequence




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


class img_texture_Offset:
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


class img_Seam_adjust_size:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512},
                ),
                "height": (
                    "INT",
                    {"default": 512},
                ),
            },
            "optional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/imgEffect"

    def run(self, pixels, width, height, keep_mask=None, drop_mask=None):
        results = []

        if keep_mask is not None:
            while keep_mask.dim() < 4:
                keep_mask = keep_mask[None]
            keep_mask = resize(keep_mask, pixels.size()[1:3])

        if drop_mask is not None:
            while drop_mask.dim() < 4:
                drop_mask = drop_mask[None]
            drop_mask = resize(drop_mask, pixels.size()[1:3])

        for i in range(pixels.size()[0]):
            image = pixels[i]
            if keep_mask is not None:
                k_mask = keep_mask[np.clip(i, 0, keep_mask.size()[0] - 1)][0]
            else:
                k_mask = None
            if drop_mask is not None:
                d_mask = drop_mask[np.clip(i, 0, drop_mask.size()[0] - 1)][0]
                if d_mask.any(0).all() or d_mask.any(1).all():
                    print("SeamCarving: Drop mask would delete entire image, ignoring")
                    d_mask = None
            else:
                d_mask = None
            src = (255 * image.cpu().clamp(min=0, max=1).numpy()).astype(np.uint8)
            dst = seam_carving.resize(
                image,
                size=(width, height),
                energy_mode="forward",  # choose from {backward, forward}
                order="width-first",  # choose from {width-first, height-first}
                keep_mask=k_mask,  # object mask to protect from removal
                drop_mask=d_mask,
                step_ratio=0.1
            )
            results.append(torch.from_numpy(dst))
        return (torch.stack(results),)


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
    CATEGORY = "Apt_Preset/ksampler"

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
    CATEGORY = "Apt_Preset/ksampler"

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
    CATEGORY = "Apt_Preset/ksampler"
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


class sampler_sigmas:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        scale_options = [option.value for option in GraphScale]
        return {
            "required": {
                "model": ("MODEL",),
                "schedule": ("STRING", {"default": "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"}),
                "steps": ("INT", {"default": 20, "min": 0, "max": 100000, "step": 1}),
                "sgm": ("BOOLEAN", {"default": False}),
                "color": (["black", "red", "green", "blue"], {"default": "blue"}),
                "scale": (scale_options, {"default": GraphScale.linear})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS","IMAGE",)
    RETURN_NAMES = ("sigmas","sch_image",)
    CATEGORY = "Apt_Preset/ksampler"
    
    
    def simple_output(self, model, schedule, steps, sgm, color, scale: GraphScale):
        if sgm:
            steps += 1
        s = model.get_model_object("model_sampling")
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))
        phi = (1 + 5 ** 0.5) / 2
        sigmas = []
        s = steps
        fibo = fibonacci_normalized_descending(s)
        for j in range(steps):
            y = j / (s - 1)
            x = 1 - y
            f = fibo[j]
            try:
                f = eval(schedule)
            except:
                print(f"could not evaluate {schedule}")
                f = 0
            sigmas.append(f)
        if sgm:
            sigmas = sigmas[:-1]
        sigmas = torch.tensor(sigmas + [0])

        sigmas_graph = tensor_to_graph_image(sigmas.cpu(), color=color, scale=scale)
        numpy_image = np.array(sigmas_graph)
        numpy_image = numpy_image / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        tensor_image = tensor_image.unsqueeze(0)
        images_tensor = torch.cat([tensor_image], 0)
        
        return (sigmas, images_tensor,)



# Statement: Source code from comfyUI-InpaintCropAndStitch  https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

class InpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["free size", "forced size"], {"default": "free size"}),
                "force_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "force_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": {}
        }

    CATEGORY = "Apt_Preset/ksampler"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH", )
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitch", )
    FUNCTION = "inpaint_crop"

    def adjust_to_aspect_ratio(self, x_min, x_max, y_min, y_max, width, height, target_width, target_height):
        x_min_key, x_max_key, y_min_key, y_max_key = x_min, x_max, y_min, y_max
        current_width = x_max - x_min + 1
        current_height = y_max - y_min + 1
        aspect_ratio = target_width / target_height
        current_aspect_ratio = current_width / current_height

        if current_aspect_ratio < aspect_ratio:
            new_width = int(current_height * aspect_ratio)
            extend_x = (new_width - current_width)
            x_min = max(x_min - extend_x//2, 0)
            x_max = min(x_max + extend_x//2, width - 1)
        else:
            new_height = int(current_width / aspect_ratio)
            extend_y = (new_height - current_height)
            y_min = max(y_min - extend_y//2, 0)
            y_max = min(y_max + extend_y//2, height - 1)

        return int(x_min), int(x_max), int(y_min), int(y_max)

    def adjust_to_preferred(self, x_min, x_max, y_min, y_max, width, height, preferred_x_start, preferred_x_end, preferred_y_start, preferred_y_end):
        if preferred_x_start <= x_min and preferred_x_end >= x_max and preferred_y_start <= y_min and preferred_y_end >= y_max:
            return x_min, x_max, y_min, y_max

        if x_max - x_min + 1 <= preferred_x_end - preferred_x_start + 1:
            if x_min < preferred_x_start:
                x_shift = preferred_x_start - x_min
                x_min += x_shift
                x_max += x_shift
            elif x_max > preferred_x_end:
                x_shift = x_max - preferred_x_end
                x_min -= x_shift
                x_max -= x_shift

        if y_max - y_min + 1 <= preferred_y_end - preferred_y_start + 1:
            if y_min < preferred_y_start:
                y_shift = preferred_y_start - y_min
                y_min += y_shift
                y_max += y_shift
            elif y_max > preferred_y_end:
                y_shift = y_max - preferred_y_end
                y_min -= y_shift
                y_max -= y_shift

        return int(x_min), int(x_max), int(y_min), int(y_max)

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    def inpaint_crop(self, image, mask, mode,  force_width, force_height, rescale_factor,  optional_context_mask=None):
        invert_mask = False
        fill_mask_holes = True
        context_expand_factor = 1
        rescale_algorithm = "bicubic"
        context_expand_pixels = 0
        padding = 8
        min_width, min_height, max_width, max_height = 258, 258, 1024, 1024

        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"

        result_stitch = {'x': [], 'y': [], 'original_image': [], 'cropped_mask': [], 'rescale_x': [], 'rescale_y': [], 'start_x': [], 'start_y': [], 'initial_width': [], 'initial_height': []}
        results_image = []
        results_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = None
            if optional_context_mask is not None:
                one_optional_context_mask = optional_context_mask[b].unsqueeze(0)

            stitch, cropped_image, cropped_mask = self.inpaint_crop_single_image(
                one_image, one_mask, context_expand_pixels, context_expand_factor, invert_mask,
                fill_mask_holes, mode, rescale_algorithm, force_width, force_height, rescale_factor,
                padding, min_width, min_height, max_width, max_height, one_optional_context_mask
            )

            for key in result_stitch:
                result_stitch[key].append(stitch[key])
            cropped_image = cropped_image.squeeze(0)
            results_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            results_mask.append(cropped_mask)

        result_image = torch.stack(results_image, dim=0)
        result_mask = torch.stack(results_mask, dim=0)

        return result_image, result_mask, result_stitch


    def inpaint_crop_single_image(self, image, mask, context_expand_pixels, context_expand_factor, invert_mask, fill_mask_holes, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, optional_context_mask=None):
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        if invert_mask:
            mask = 1.0 - mask

        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        assert initial_height == mask_height and initial_width == mask_width, "Image and mask dimensions must match"
        assert initial_height == context_mask_height and initial_width == context_mask_width, "Image and context mask dimensions must match"

        extend_y = (initial_width + 1) // 2
        extend_x = (initial_height + 1) // 2
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x

        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype)
        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)

        start_y = extend_y
        start_x = extend_x

        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask

        image = new_image
        mask = new_mask
        context_mask = new_context_mask

        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            stitch = {'x': 0, 'y': 0, 'original_image': original_image, 'cropped_mask': mask, 'rescale_x': 1.0, 'rescale_y': 1.0, 'start_x': start_x, 'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}
            return (stitch, original_image, original_mask)

        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]

        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size*(context_expand_factor-1), context_expand_pixels))
        x_grow = round(max(x_size*(context_expand_factor-1), context_expand_pixels))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0

        if mode == 'forced size':
            mode = 'ranged size'
            min_width = max_width = force_width
            min_height = max_height = force_height
            rescale_factor = 100

        if mode == 'ranged size':
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            current_aspect_ratio = current_width / current_height
            min_aspect_ratio = min_width / max_height
            max_aspect_ratio = max_width / min_height

            if current_aspect_ratio < min_aspect_ratio:
                target_width = min(current_width, min_width)
                target_height = int(target_width / min_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            elif current_aspect_ratio > max_aspect_ratio:
                target_height = min(current_height, max_height)
                target_width = int(target_height * max_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            else:
                target_width = current_width
                target_height = current_height

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1

            min_rescale_width = min_width / x_size
            min_rescale_height = min_height / y_size
            min_rescale_factor = min(min_rescale_width, min_rescale_height)
            rescale_factor = max(min_rescale_factor, rescale_factor)
            max_rescale_width = max_width / x_size
            max_rescale_height = max_height / y_size
            max_rescale_factor = min(max_rescale_width, max_rescale_height)
            rescale_factor = min(max_rescale_factor, rescale_factor)

        if rescale_factor < 0.999 or rescale_factor > 1.001:
            samples = image            
            samples = samples.movedim(-1, 1)
            width = math.floor(samples.shape[3] * rescale_factor)
            height = math.floor(samples.shape[2] * rescale_factor)
            samples = rescale(samples, width, height, rescale_algorithm)
            effective_upscale_factor_x = float(width)/float(original_width)
            effective_upscale_factor_y = float(height)/float(original_height)
            samples = samples.movedim(1, -1)
            image = samples

            samples = mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, rescale_algorithm)
            samples = samples.squeeze(1)
            mask = samples

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            target_x_size = int(x_size * effective_upscale_factor_x)
            target_y_size = int(y_size * effective_upscale_factor_y)

            x_min = math.floor(x_min * effective_upscale_factor_x)
            x_max = x_min + target_x_size
            y_min = math.floor(y_min * effective_upscale_factor_y)
            y_max = y_min + target_y_size

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1

        if mode == 'free size' and padding > 1:
            x_min, x_max = self.apply_padding(x_min, x_max, width, padding)
            y_min, y_max = self.apply_padding(y_min, y_max, height, padding)

        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]

        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask': cropped_mask, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y, 'start_x': start_x, 'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}

        return (stitch, cropped_image, cropped_mask)


class InpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "stitch": ("STITCH",),
            }
        }

    CATEGORY = "Apt_Preset/ksampler"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"

    def composite(self, destination, source, x, y, mask=None, multiplier=8, resize_source=False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

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
        return destination

    def inpaint_stitch(self, stitch, inpainted_image):
        rescale_algorithm = "bicubic"
        results = []
        batch_size = inpainted_image.shape[0]
        assert len(stitch['x']) == batch_size, "Stitch size doesn't match image batch size"
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitch = {}
            for key in stitch:
                one_stitch[key] = stitch[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitch, one_image, rescale_algorithm)
            one_image = one_image.squeeze(0)
            results.append(one_image)

        result_batch = torch.stack(results, dim=0)
        return (result_batch,)

    def inpaint_stitch_single_image(self, stitch, inpainted_image, rescale_algorithm):
        original_image = stitch['original_image']
        cropped_mask = stitch['cropped_mask']
        x = stitch['x']
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)
        start_x = stitch['start_x']
        start_y = stitch['start_y']
        initial_width = stitch['initial_width']
        initial_height = stitch['initial_height']

        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]

        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)
            width = round(float(inpaint_width)/stitch['rescale_x'])
            height = round(float(inpaint_height)/stitch['rescale_y'])
            x = round(float(x)/stitch['rescale_x'])
            y = round(float(y)/stitch['rescale_y'])
            samples = rescale(samples, width, height, rescale_algorithm)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = rescale(samples, width, height, rescale_algorithm)
            samples = samples.squeeze(0)
            cropped_mask = samples.movedim(1, -1)

        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask, 1).movedim(1, -1)
        cropped_output = output[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :]
        output = cropped_output
        return (output,)



#endregion-----------------------basic node---------------------------------
