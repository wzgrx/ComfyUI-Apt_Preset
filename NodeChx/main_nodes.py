

#region-----------导入与全局函数-------------------------------------------------
# 内置库
import os
import sys
import glob
import csv
import json
import math
from telnetlib import OUTMRK
import re

# 第三方库
import numpy as np
import torch
import toml
import aiohttp
from aiohttp import web
from PIL import Image, ImageFilter
from tqdm import tqdm
#import torchvision.transforms.functional as TF
#import torch.nn.functional as F
import cv2
import nodes
# 本地库

import comfy
import folder_paths
import node_helpers
import latent_preview
from server import PromptServer
from nodes import common_ksampler, CLIPTextEncode, ControlNetLoader, LoadImage, ControlNetApplyAdvanced, VAEDecode, VAEEncode, DualCLIPLoader, CLIPLoader,ConditioningConcat, ConditioningAverage, InpaintModelConditioning, LoraLoader, CheckpointLoaderSimple, ImageScale,VAEDecodeTiled
from comfy.cli_args import args
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
from typing import Optional, Tuple, Dict, Any, Union, cast
from comfy.comfy_types.node_typing import IO
from comfy_extras.nodes_custom_sampler import RandomNoise,  BasicScheduler, KSamplerSelect, SamplerCustomAdvanced, BasicGuider
from comfy_extras.nodes_sd3 import TripleCLIPLoader

from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_controlnet import SetUnionControlNetType
from math import ceil
from nodes import CLIPSetLastLayer, CheckpointLoaderSimple, UNETLoader

# 相对导入
from ..main_unit import *
from .main_stack import  Apply_LoRAStack



#region------------------------preset---------------------------------#


# 获取当前文件所在目录的上一级目录
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 插入上一级目录下的 comfy 到 sys.path
sys.path.insert(0, os.path.join(parent_directory, "comfy"))

routes = PromptServer.instance.routes
@routes.post('/Apt_Preset_path')
async def my_function(request):
    the_data = await request.post()
    sum_load.handle_my_message(the_data)
    return web.json_response({})

# 使用上一级目录作为基础路径
my_directory_path = parent_directory
presets_directory_path = os.path.join(my_directory_path, "presets")


preset_list = []
tmp_list = []
tmp_list += glob.glob(f"{presets_directory_path}/**/*.toml", recursive=True)
for l in tmp_list:
    preset_list.append(os.path.relpath(l, presets_directory_path))

if len(preset_list) > 1: preset_list.sort()


available_ckpt = folder_paths.get_filename_list("checkpoints")
available_unets = folder_paths.get_filename_list("unet")
available_clips = folder_paths.get_filename_list("text_encoders")+folder_paths.get_filename_list("clip")
available_loras = folder_paths.get_filename_list("loras")
available_vaes = folder_paths.get_filename_list("vae")
available_controlnets = folder_paths.get_filename_list("controlnet")
available_embeddings = folder_paths.get_filename_list("embeddings")




CLIP_TYPE = ["sdxl", "sd3", "flux", "hunyuan_video", "stable_diffusion", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan"]





def getNewTomlnameExt(tomlname, folderpath, savetype):

    tomlnameExt = tomlname + ".toml"
    
    if savetype == "new save":

        filename_list = []
        tmp_list = []
        tmp_list += glob.glob(f"{folderpath}/**/*.toml", recursive=True)
        for l in tmp_list:
            filename_list.append(os.path.relpath(l, folderpath))
        
        duplication_flag = False
        for f in filename_list:
            if tomlnameExt == f:
                duplication_flag = True
                
        if duplication_flag:
            count = 1
            while duplication_flag:
                new_tomlnameExt = f"{tomlname}_{count}.toml"
                if not new_tomlnameExt in filename_list:
                    tomlnameExt = new_tomlnameExt
                    duplication_flag = False
                count += 1
                
    return tomlnameExt


#endregion------------------------preset---------------------------------#

#endregion


#region-----------基础节点context------------------------------------------------------------------------------#






class Data_chx_Merge:
    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "required": {  },
            "optional": {
            "context": ("RUN_CONTEXT",),  
                "chx1": ("RUN_CONTEXT",),
                "chx2": ("RUN_CONTEXT",),
            },
        }

    RETURN_TYPES = "RUN_CONTEXT",
    RETURN_NAMES = "context",
    CATEGORY = "Apt_Preset/chx_load"
    FUNCTION = "merge"
    

    def get_return_tuple(self, ctx):
        return get_orig_context_return_tuple(ctx)

    def merge(self, context=None, chx1=None, chx2=None):
        ctxs = [context, chx1, chx2]  
        ctx = merge_new_context(*ctxs)
        return self.get_return_tuple(ctx)




class Data_basic:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "required": {  "context": ("RUN_CONTEXT",),   
                
                    },
            "optional": {

                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "images": ("IMAGE",),
                "mask": ("MASK",), 
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","CONDITIONING","LATENT","VAE","CLIP","IMAGE","MASK",)
    RETURN_NAMES = ("context", "model","positive","negative","latent","vae","clip","images","mask",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context=None,model=None,positive=None,negative=None,latent=None,vae =None,clip =None,images =None, mask =None,):
        

        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")
        if vae is None:
            vae = context.get("vae")
        if clip is None:
            clip = context.get("clip")
        
        if latent is None:
            latent = context.get("latent")
        if images is None:
            images = context.get("images")
        if mask is None:
            mask = context.get("mask")
        
        context = new_context(context,model=model,positive=positive,negative=negative,latent=latent,vae=vae,clip=clip,images=images,mask=mask,)
        return (context, model, positive, negative, latent, vae, clip, images, mask,)


class Data_basic_easy:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "required": { 
                
                    },
            "optional": {
                "context": ("RUN_CONTEXT",),   
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CLIP", "CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("context", "model","clip","positive","negative","latent",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context=None,model=None,positive=None,negative=None,clip =None, latent=None):
        

        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")
        if clip is None:
            clip = context.get("clip")
        if latent is None:
            latent = context.get("latent")

        
        context = new_context(context,model=model,positive=positive,negative=negative,clip=clip,latent=latent,)
        return (context, model,clip, positive, negative, latent,  )


class Data_sample:  
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "optional": {  "context": ("RUN_CONTEXT",),   },
        }

    RETURN_TYPES = ("RUN_CONTEXT","INT","FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("context","steps","cfg","sampler","scheduler" )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context, ):
        
        steps=context.get("steps",None)
        cfg=context.get("cfg",None) 
        sampler=context.get("sampler",None) 
        scheduler=context.get("scheduler",None) 
        
        return (context,steps,cfg,sampler,scheduler )


class load_create_basic_chx:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":  {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                
            },
        }
    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "pipein"

    CATEGORY = "Apt_Preset/chx_load"

    def pipein(self, model = None, clip = None, positive = None, negative = None, vae = None, latent = None, steps = None, cfg = None, sampler = None, scheduler = None):

        context = {"model":model, "clip":clip, "positive":positive, "negative":negative, "vae":vae, "latent":latent, "steps":steps, "cfg":cfg, "sampler":sampler, "scheduler":scheduler, }

        return (context,)



class Data_select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "context": ("RUN_CONTEXT",),
                "type": (["model", "clip", "positive", "negative", "vae", "latent", "images", "mask",
                        "clip1", 
                        "clip2", 
                        "clip3", 
                        "unet_name", 
                        "ckpt_name",
                        "pos", 
                        "neg",
                        "width",
                        "height",
                        "batch"
                        ], {}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    FUNCTION = "pipeout"

    CATEGORY = "Apt_Preset/chx_load"

    def pipeout(self, type, context=None):

        out = context[type]
        return (out,)





class Data_presetData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "context": ("RUN_CONTEXT",),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT",
                    available_clips,
                    available_clips,
                    available_clips,  
                    folder_paths.get_filename_list("unet"), 
                    folder_paths.get_filename_list("checkpoints"),
                    "STRING", 
                    "STRING", 
                    "INT",
                    "INT",
                    "INT",
                    )

    RETURN_NAMES = (
            "context",
            "clip1", 
            "clip2", 
            "clip3", 
            "unet_name", 
            "ckpt_name",
            "pos", 
            "neg",
            "width",
            "height",
            "batch",
                    )

    CATEGORY = "Apt_Preset/chx_load"
    FUNCTION = "convert"

    def convert(self, context=None):
        ckpt_name = context.get("ckpt_name", None)
        unet_name = context.get("unet_name", None)
        clip1 = context.get("clip1", None)
        clip2 = context.get("clip2", None)
        clip3 = context.get("clip3", None)
        pos = context.get("pos", None)
        neg = context.get("neg", None)
        width = context.get("width", None)
        height = context.get("height", None)
        batch = context.get("batch", None)

        return (
            context,
            clip1, 
            clip2, 
            clip3, 
            unet_name, 
            ckpt_name,
            pos, 
            neg,
            width,
            height,
            batch
        )



#endregion


#region-----------加载器load-----------------------------------------------------------------------------------#


class sum_load:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),
                "run_Mode": (["basic","clip", "FLUX", "SD3.5", "only_clip",],),
                "ckpt_name": (["None"] + available_ckpt,),  
                "clipnum": ("INT", {"default": -1, "min": -24, "max": 1, "step": 1}),
                "unet_name": (["None"] + available_unets, ), # Flux &SD3.5
                "unet_Weight_Dtype": (["None"]+ ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),

                "clip_type": (["None"]+ CLIP_TYPE, ),  # Flux
                "clip1": (["None"] + available_clips, ),  # SD3.5 和 Flux
                "clip2": (["None"] + available_clips, ),  # SD3.5 和 Flux
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "clip3": (["None"] + available_clips,),  # SD3.5

                "vae": (["None"] + available_vaes, ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),  
                "lora_stack": ("LORASTACK",),
                        },

            "hidden": { "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id, run_Mode, lora_strength, clipnum, 
                        width, height, batch, steps, cfg, sampler, scheduler, unet_Weight_Dtype, guidance, over_clip=None,  clip_type=None,device="default", 
                        vae=None, lora=None, unet_name=None, ckpt_name=None, clip1=None, lora_stack=None, over_model=None,
                        clip2=None, clip3=None, pos="default", neg="default", preset=[]):
        
        # 非编码后的数据
        clip = None

        parameters_data = []
        parameters_data.append({
            "run_Mode": run_Mode,
            "ckpt_name": ckpt_name,
            "clipnum": clipnum,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": clip3,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        
        
        model_options = {}
        if unet_Weight_Dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif unet_Weight_Dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_Weight_Dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2


        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  


        # Case 1: basic mode
        if run_Mode == "basic":

            if ckpt_name!= "None":
                model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=available_embeddings)
                model = out[0]
                vae = out[2]
                if over_clip is None:
                    # 检查 out[1] 是否为 None
                    if out[1] is not None:
                        clip = out[1].clone()
                    else:
                        clip = None
                        print("Warning: clip object is None after loading checkpoint.")
                else:
                    clip = over_clip

            if over_clip is not None:        #同步clip和model
                clip = over_clip
            if over_model is not None:
                model = over_model
            if clip is not None:
                clip.clip_layer(clipnum)

        # Case 2: clip mode
        if run_Mode == "clip":
            if  ckpt_name!= "None":
                model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=available_embeddings)
                model = out[0]
                vae = out[2]
                # 检查 out[1] 是否为 None
                if out[1] is not None:
                    clip = out[1].clone()
                else:
                    clip = None
                    print("Warning: clip object is None after loading checkpoint.")

            if  unet_name!= "None":
                unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

            if  clip is None and clip1 != "None":
                clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]

            if over_model is not None:
                model = over_model
            if over_clip is not None:
                clip = over_clip

            # 修正 else 语句块的位置，确保和 if run_Mode == "clip": 配对
            if ckpt_name == "None" and unet_name == "None":
                raise ValueError("ckpt_name or unet_name must be entered. Please enter a valid ckpt_name or unet_name.")


        # Case 3: FLUX mode
        if run_Mode == "FLUX":
            if over_model is not None:
                model = over_model
            elif over_model is None and unet_name!= "None":
                unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            else:
                raise ValueError("unet cannot be None.")

            if over_clip is not None:
                clip = over_clip
            elif over_clip is None and clip1 != "None" and clip2 != "None":
                clip = DualCLIPLoader().load_clip( clip1, clip2, clip_type, device)[0]
            else:
                raise ValueError("clip1 and clip2  cannot be None.")


        # Case 4: SD3.5 mode
        if run_Mode == "SD3.5":
            if over_model is not None:
                model = over_model
            elif over_model is None and unet_name!= "None":
                unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            else:
                raise ValueError("unet cannot be None.")
            
            if over_clip is not None:
                clip = over_clip
            elif over_clip is None and clip1 != "None" and clip2 != "None" and clip3!= "None":
                clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3)[0]
            else:
                raise ValueError("clip1 and clip2 and clip3  cannot be None.")


        # Case 5: only_clip mode
        if run_Mode == "only_clip":
            if over_model is not None:
                model = over_model
            elif over_model is None:
                model = None
            if over_clip is not None:
                clip = over_clip
            elif over_clip is None :
                if clip1 != "None":
                    clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
                    if clip2 != "None":
                        clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
                        if clip3 != "None":
                            clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3, clip_type, device)[0]
            else:
                raise ValueError("At least one clip file (clip1) must be provided in 'only_clip' mode.")
            if clip is None:
                raise ValueError("Clip object is None. Cannot encode text without a valid clip.")
            

            (positive,) = CLIPTextEncode().encode(clip, pos)
            (negative,) = CLIPTextEncode().encode(clip, neg)
            if isinstance(vae, str) and vae != "None":
                vae_path = folder_paths.get_full_path("vae", vae)
                vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

            context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": clip3, 
            "unet_name": unet_name, 
            "ckpt_name": ckpt_name,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
            }
            return (context, model, parameters_data, )


        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)
        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)


        if run_Mode == "FLUX":
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})


        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))


        positive = positive
        negative = negative
        
        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,

            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": clip3, 
            "unet_name": unet_name, 
            "ckpt_name": ckpt_name,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,

        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



class load_clip:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),

                "ckpt_name": (["None"] + available_ckpt,),  
                "unet_name": (["None"] + available_unets, ), # Flux &SD3.5
                "unet_Weight_Dtype": (["None"]+ ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),
                "clip_type": (["None"]+ CLIP_TYPE, ),  # Flux
                "clip1": (["None"] + available_clips, ),  # SD3.5 和 Flux


                "vae": (["None"] + available_vaes, ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "lora_stack": ("LORASTACK",),
                        },

            "hidden": { "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id, lora_strength, 
                        width, height, batch, steps, cfg, sampler, scheduler, unet_Weight_Dtype, clip_type=None,device="default", 
                        vae=None, lora=None, unet_name=None, ckpt_name=None, clip1=None, lora_stack=None, 
                        pos="default", neg="default", preset=[]):
        
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "clip",
            "ckpt_name": ckpt_name,
            "clipnum": -1,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": None,
            "guidance": 3.5,
            "clip3": None,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        
        
        model_options = {}
        if unet_Weight_Dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif unet_Weight_Dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_Weight_Dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2


        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  


        if ckpt_name!= "None":
            model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=available_embeddings)
            model = out[0]
            vae = out[2]
            clip = out[1].clone()

        if  unet_name!= "None":
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            if clip1 != "None":
                clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
            else:
                raise ValueError("In clip mode, clip1 cannot be None. Please enter a valid clip1.")
            
        else:
            raise ValueError("Please enter a valid ckpt_name or unet_name.")



        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))


        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)
        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  


        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)
        

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": 3.5,

            "clip1": clip1, 
            "clip2": None, 
            "clip3": None, 
            "unet_name": unet_name, 
            "ckpt_name": ckpt_name,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,

        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



class load_only_clip:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),

                "clip_type": (["None"]+ CLIP_TYPE, ),  # Flux
                "clip1": (["None"] + available_clips, ),  # SD3.5 和 Flux
                "clip2": (["None"] + available_clips, ),  # SD3.5 和 Flux
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "clip3": (["None"] + available_clips,),  # SD3.5
                "vae": (["None"] + available_vaes, ),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "over_model": ("MODEL",),
                        },

            "hidden": { "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id, 
                        width, height, batch, steps, cfg, sampler, scheduler,  guidance,  clip_type=None,device="default", 
                        vae=None, clip1=None, over_model=None,
                        clip2=None, clip3=None, pos="default", neg="default", preset=[]):
        
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "only_clip",
            "ckpt_name": None,
            "clipnum": -1,
            "unet_name": None,
            "unet_Weight_Dtype": None,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": clip3,
            "vae": vae,  
            "lora": None,
            "lora_strength": None,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        
        

        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  


        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))


        if over_model is not None:
            model = over_model
        elif over_model is None:
            model = None

        if clip1 != "None":
            clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
            if clip2 != "None":
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
                if clip3 != "None":
                    clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3, clip_type, device)[0]


        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)


        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,

            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": clip3, 
            "unet_name": None, 
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,

        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



class load_basic:
    @classmethod
    def INPUT_TYPES(cls):


        return {
            "required": {
                "preset": (preset_list, ),
                "ckpt_name": (["None"] + available_ckpt,),  
                "clipnum": ("INT", {"default": 0, "min": -24, "max": 1}),
                "vae": (["None"] + available_vaes, ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                        },

            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id,  lora_strength, clipnum, 
                        width, height, batch, steps, cfg, sampler, scheduler,  device="default", 
                        vae=None, lora=None, ckpt_name=None,  pos="default", neg="default", preset=[]):
        
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "basic",
            "ckpt_name": ckpt_name,
            "clipnum": clipnum,
            "unet_name": None,
            "unet_Weight_Dtype": None,
            "clip_type": None,
            "clip1": None,
            "clip2": None,
            "guidance": None,
            "clip3": None,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        

        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  


        model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        vae = out[2]
        clip = out[1].clone()
        if clip is not None:
            clip.clip_layer(clipnum)

        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": None, 
            "clip1": None,  
            "clip2": None,  
            "clip3": None,  
            "unet_name": None, 
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



class load_FLUX:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),
                "unet_name": (available_unets, ), 
                "unet_Weight_Dtype": ( ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),

                "clip_type": (["flux", "sdxl", "sd3", "hunyuan_video"], ),  # Flux
                "clip1": (available_clips,{"default": "clip_l.safetensors"} ),  
                "clip2": (available_clips, {"default": "t5xxl_fp8_e4m3fn.safetensors"} ), 
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),

                "vae": (available_vaes,{"default": "ae.safetensors"} ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                #"neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                        },
            
            
            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id, lora_strength, 
                        width, height, batch, steps, cfg, sampler, scheduler, unet_Weight_Dtype, guidance, clip_type=None,device="default", 
                        vae=None, lora=None, unet_name=None, clip1=None, 
                        clip2=None, pos="default", neg="", preset=[]):
        
        neg="worst quality, low quality"
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "FLUX",
            "ckpt_name": None,
            "clipnum": None,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": None,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        
        
        model_options = {}
        if unet_Weight_Dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif unet_Weight_Dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_Weight_Dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2


        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  

        if isinstance(vae, str) and vae!= "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        if unet_name!= "None":
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        if clip1 != None and clip2 != None:
            clip =DualCLIPLoader().load_clip( clip1, clip2, clip_type, device)[0]

        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)
        
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})


        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,

            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": None, 
            "unet_name": unet_name, 
            "ckpt_name": None, 
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



class load_SD35:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),
                "unet_name": (["None"] + available_unets, ), # Flux &SD3.5
                "unet_Weight_Dtype": ( ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),
                "clip1": (available_clips, {"default": "clip_l.safetensors"} ),  
                "clip2": (available_clips, {"default": "clip_g.safetensors"} ), 
                "clip3": (available_clips, {"default": "t5xxl_fp8_e4m3fn.safetensors"} ), 

                "vae": (available_vaes,{"default": "SD3.5vae.safetensors"} ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                #"neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                        },
            
            
            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, node_id, lora_strength, 
                        width, height, batch, steps, cfg, sampler, scheduler, unet_Weight_Dtype, device="default", 
                        vae=None, lora=None, clip1=None, unet_name=None,
                        clip2=None, clip3=None, pos="default", neg="", preset=[]):
        
        neg="worst quality, low quality"
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "SD3.5",
            "ckpt_name": None,
            "clipnum": None,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": None,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": None,
            "clip3": clip3,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            
        })
        
        
        model_options = {}
        if unet_Weight_Dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif unet_Weight_Dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_Weight_Dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  

        if vae!= "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        if unet_name!= "None":
            unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        if clip1 != None and clip2 != None and clip3 != None:
            clip=TripleCLIPLoader().load_clip(clip1, clip2, clip3,)[0]

        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)
        

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": None,
            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": clip3, 
            "unet_name": unet_name, 
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})




class load_create_chx:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {

                "vae": (["None"] + available_vaes, ),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),  
                "over_vae": ("VAE",),
                "lora_stack": ("LORASTACK",),
                        },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", )
    RETURN_NAMES = ("context","model", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self, 
                        width, height, batch, steps, cfg, sampler, scheduler, 
                        vae=None, over_vae=None, over_clip=None, over_model=None, lora_stack=None, pos="default", neg="default", ):


        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  

        if over_vae is not None:
            vae = over_vae
        elif over_vae is None and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))


        if over_model is not None:
            model = over_model
        else:
            model = None

        if over_clip is not None:
            clip = over_clip
        else:
            raise ValueError(" Please enter a valid clip.")


        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": None,
            "clip1": None,
            "clip2": None,
            "clip3": None,
            "unet_name": None,
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, )





class Data_preset_save:
    @classmethod
    def INPUT_TYPES(s):
        savetype_list = ["new save", "overwrite save"]
        return {
            "required": {
                "param": ("PDATA", ),
                "tomlname": ("STRING", {"default": "new_preset"}),
                "savetype": (savetype_list,),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveparam"
    CATEGORY = "Apt_Preset/chx_load"

    def saveparam(self, param, tomlname, savetype):
        # 初始化 tomltext 为一个空字符串
        tomltext = ""

        def fix_path(path):
            if isinstance(path, str):
                return path.replace("\\", "\\\\")
            return path

        tomltext += f"run_Mode = \"{param[0]['run_Mode']}\"\n"
        tomltext += f"ckpt_name = \"{fix_path(param[0]['ckpt_name'])}\"\n"
        tomltext += f"clipnum = \"{param[0]['clipnum']}\"\n"
        
        tomltext += f"unet_name = \"{fix_path(param[0]['unet_name'])}\"\n"
        tomltext += f"unet_Weight_Dtype = \"{param[0]['unet_Weight_Dtype']}\"\n"

        tomltext += f"clip_type = \"{param[0]['clip_type']}\"\n"
        tomltext += f"clip1 = \"{fix_path(param[0]['clip1'])}\"\n"
        tomltext += f"clip2 = \"{fix_path(param[0]['clip2'])}\"\n"
        tomltext += f"guidance = \"{param[0]['guidance']}\"\n"

        tomltext += f"clip3 = \"{fix_path(param[0]['clip3'])}\"\n"

        tomltext += f"vae = \"{fix_path(param[0]['vae'])}\"\n"
        tomltext += f"lora = \"{fix_path(param[0]['lora'])}\"\n"
        tomltext += f"lora_strength = \"{param[0]['lora_strength']}\"\n"
        
        tomltext += f"width = \"{param[0]['width']}\"\n"
        tomltext += f"height = \"{param[0]['height']}\"\n"
        tomltext += f"batch = \"{param[0]['batch']}\"\n"

        tomltext += f"steps = \"{param[0]['steps']}\"\n"
        tomltext += f"cfg = \"{param[0]['cfg']}\"\n"
        tomltext += f"sampler = \"{param[0]['sampler']}\"\n"
        tomltext += f"scheduler = \"{param[0]['scheduler']}\"\n"

        tomltext += f"positive = \"{param[0]['positive']}\"\n"
        tomltext += f"negative= \"{param[0]['negative']}\"\n"



        tomlnameExt = getNewTomlnameExt(tomlname, presets_directory_path, savetype)

        check_folder_path = os.path.dirname(f"{presets_directory_path}/{tomlnameExt}")
        os.makedirs(check_folder_path, exist_ok=True)

        with open(f"{presets_directory_path}/{tomlnameExt}", mode='w', encoding='utf-8') as f:
            f.write(tomltext)

        return ()



class sum_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                "preset": (preset_list,),
                "switch": (["true", "false"],),
                
                "image1": ("IMAGE",),
                "controlnet1": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),

                
                "image2": ("IMAGE",),
                "controlnet2": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),

                
                "image3": ("IMAGE",),
                "controlnet3": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),

                
                "image4": ("IMAGE",),
                "controlnet4": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),

                
                
                
            },
            "hidden": {"node_id": "UNIQUE_ID",}
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("context","positive", "negative",)
    CATEGORY = "Apt_Preset/chx_load"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, node_id, switch,
                        strength1, 
                        strength2,
                        strength3, 
                        strength4, 
                        context=None, 
                        controlnet1=None, controlnet2=None, controlnet3=None, controlnet4=None,
                        image1=None, image2=None, image3=None, image4=None, vae=None, preset=[], extra_concat=[]):


        if switch == "false":
            return (context,context.get("positive"),context.get("negative"))

        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)

        if controlnet1 != "None" and image1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet1)
            controlnet1 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet( positive, negative, controlnet1, image1, strength1, 0, 1, vae, extra_concat=[])
            positive = conditioning[0]
            negative = conditioning[1]

        if controlnet2 != "None" and image2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet2)
            controlnet2 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet( positive, negative, controlnet2, image2, strength2, 0, 1, vae, extra_concat=[])
            positive = conditioning[0]
            negative = conditioning[1]

        if controlnet3 != "None" and image3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet3)
            controlnet3 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet( positive, negative, controlnet3, image3, strength3, 0, 1, vae, extra_concat=[])
            positive = conditioning[0]
            negative = conditioning[1]

        if controlnet4 != "None" and image4 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet4)
            controlnet4 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet( positive, negative, controlnet4, image4, strength4, 0, 1, vae, extra_concat=[])
            positive = conditioning[0]
            negative = conditioning[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative, )



class sum_lora:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_01":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_02":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_03":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "model": ("MODEL",),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "style": (none2list(style_list()[0]),{"default": "None"}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL","CLIP", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("context", "model","clip", "positive", "negative",)
    CATEGORY = "Apt_Preset/chx_load"
    FUNCTION = "load_lora"

    def load_lora(self, style, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, context, model=None,  pos="", neg=""):

        if model is None:
            model=  context.get("model")
        clip = context.get("clip")
        positive = context.get("positive")
        negative = context.get("negative") 
        guidance = context.get("guidance",3.5)

        if style != "None":
            pos += f"{pos}, {style_list()[1][style_list()[0].index(style)][1]}"
            neg += f"{neg}, {style_list()[1][style_list()[0].index(style)][2]}" if len(style_list()[1][style_list()[0].index(style)]) > 2 else ""

        if lora_01!= "None" and strength_01!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_01, strength_01, strength_01)
        if lora_02!= "None" and strength_02!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_02, strength_02, strength_02)
        if lora_03!= "None" and strength_03!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_03, strength_03, strength_03)

        if pos != None and pos != '':          
            positive, = CLIPTextEncode().encode(clip, pos)
        if neg!= None and pos != '':  
            negative, = CLIPTextEncode().encode(clip, neg)
            
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        context = new_context(context, model=model, clip=clip, positive=positive, negative=negative, )
        return (context, model, clip, positive, negative,)


class pre_sample_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == no change"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == no change"}),
                "sampler": (['None'] + comfy.samplers.KSampler.SAMPLERS, {"default": "None"}),  
                "scheduler": (['None'] + comfy.samplers.KSampler.SCHEDULERS, {"default": "None"}), 
                
                
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"

    def sample(self, context, steps, cfg, sampler, scheduler):
        
        if cfg == 0.0:
            cfg = context.get("cfg")
        if steps == 0:
            steps = context.get("steps")
        if sampler == "None":
            sampler = context.get("sampler")
        if scheduler == "None":
            scheduler = context.get("scheduler")
        
        context = new_context(context, steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler)
        return (context, )


class sum_editor:

    ratio_sizes, ratio_dict = read_ratios()
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",), 
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == None"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == None"}),
                "sampler": (['None'] + comfy.samplers.KSampler.SAMPLERS, {"default": "None"}),  
                "scheduler": (['None'] + comfy.samplers.KSampler.SCHEDULERS, {"default": "None"}), 
                
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "style": (["None"] + style_list()[0], {"default": "None"}),
                "ratio_selected": (['None'] + s.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, })
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE","CLIP",  "IMAGE",)
    RETURN_NAMES = ("context", "model","positive", "negative", "latent", "vae","clip", "image", )
    FUNCTION = "text"
    CATEGORY = "Apt_Preset/chx_load"

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )

    def text(self, context=None, model=None, clip=None, positive=None, negative=None, pos="", neg="", image=None, vae=None, latent=None, steps=None, cfg=None, sampler=None, scheduler=None, style=None, batch_size=1, ratio_selected=None, ):
        
        if model is None:
            model = context.get("model")
        if clip is None:
            clip = context.get("clip")
        if vae is None:
            vae= context.get("vae")
        if steps == 0:
            steps = context.get("steps")
        if cfg == 0.0:
            cfg = context.get("cfg")
        if sampler == "None":
            sampler = context.get("sampler")
        if scheduler == "None":
            scheduler = context.get("scheduler")

        guidance = context.get("guidance",3.5)
        
        #latent选项-------------------
        if latent is None:
            latent = context.get("latent")
            
        if image is None:
            image = context.get("images")

        if image is not None:
            image = image
            latent = VAEEncode().encode(vae, image)[0]
            
        latent = latentrepeat(latent, batch_size)[0]   # latent批次
        if ratio_selected != "None":
            latent = self.generate(ratio_selected, batch_size)[0]    
        
        positive, negative = add_style_to_subject(style, pos, neg)  # 风格

        if pos is not None and pos!= "":
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive")

        if neg is not None and neg!= "":
            negative, = CLIPTextEncode().encode(clip, neg)
        else:
            negative = context.get("negative")
        
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        context = new_context(context, model=model , latent=latent , clip=clip, vae=vae, positive=positive, negative=negative, images=image,steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,) 
        
        return (context, model, positive, negative, latent, vae, clip,  image, )


class sum_latent:
    
    ratio_sizes, ratio_dict = read_ratios()
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "required": {  "context": ("RUN_CONTEXT",),   },
            
            "optional":  {
                "latent": ("LATENT", ),
                "pixels": ("IMAGE", ),
                "mask": ("MASK", ),
                "noise_mask": ("BOOLEAN", {"default": True, }),
                "smoothness":("INT", {"default": 1,  "min":0, "max": 150, "step": 1,"display": "slider"}),
                "ratio_selected": (['None'] + s.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, })
                
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","LATENT","MASK"  )
    RETURN_NAMES = ("context","latent","mask"  )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_load"

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )


    def process(self,  noise_mask, ratio_selected,smoothness=1, batch_size=1, context=None, latent=None, pixels=None, mask=None,):
        
        vae = context.get("vae")
        positive = context.get("positive")
        negative = context.get("negative")
        model = context.get("model")
        
                
        if ratio_selected != "None":
            latent = self.generate(ratio_selected, batch_size)[0]    
            context = new_context(context, latent=latent)
            return (context,latent, None)

        if latent is None:
            latent=context.get("latent",None)
        
        if pixels is not None :
            latent = VAEEncode().encode(vae, pixels)[0]
        
        if mask is not None :
            mask=tensor2pil(mask)
            if not isinstance(mask, Image.Image):
                raise TypeError("mask is not a valid PIL Image object")
            
            feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
            mask=pil2tensor(feathered_image)
            
            positive, negative, latent=InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask)   #out[0], out[1], out_latent
            model = DifferentialDiffusion().apply(model)[0] 
            
        latent = latentrepeat(latent, batch_size)[0]   # latent批次
        context = new_context(context, model = model, positive =positive, negative =negative , latent=latent)
        return (context,latent,mask)




class sum_text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "preset": (preset_list, ),
            },

            "optional": {
                "positive": ("STRING", {"default": "", "multiline": True, }),
                "negative": ("STRING", {"default": "", "multiline": False,}),

                "target": ("STRING", {"multiline": False,"default": "object, prompt, Texture"}),
                "replace": ("STRING", {"multiline": False,"default": "a girl, simle, fire "}),

                "pos1": ("STRING", {"default": "", "multiline": False, }),
                "neg1":  ("STRING", {"default": "", "multiline": False, }),
                "style": (["None"] + style_list()[0], {"default": "None"}),
            },

            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }


    RETURN_TYPES = ("RUN_CONTEXT","PDATA", )
    RETURN_NAMES = ("context", "preset_save", )

    FUNCTION = 'process_settings'
    CATEGORY = "Apt_Preset/chx_load"

    def process_settings(self, node_id, context,  style="default", positive="", negative="",target="", replace="",pos1="", neg1="" , preset=[]):
        
        parameters_data = []
        parameters_data.append({
            "run_Mode": None,
            "ckpt_name": None,
            "clipnum": None,
            "unet_name": None,
            "unet_Weight_Dtype": None,
            "clip_type": None,
            "clip1": None,
            "clip2": None,
            "guidance": None,
            "clip3": None,
            "vae":  None, 
            "lora": None,
            "lora_strength": None,
            "width": None,
            "height": None,
            "batch": None,
            "steps": None,
            "cfg": None,
            "sampler": None,
            "scheduler": None,
            "positive": positive,
            "negative": negative,
            
        })
        


        if pos1 is not None and pos1 != "":
            positive = positive + "," + pos1
        if neg1 is not None and neg1 != "":
            negative = negative + "," + neg1
            
        if target is not None and target!= "":
            positive = replace_text(positive, target, replace)

        if isinstance(positive, tuple):
            if len(positive) > 0:
                positive = str(positive[0])
            else:
                positive = ""

        positive, negative = add_style_to_subject(style, positive, negative)




        pos=positive  #更新纯文本
        neg = negative  #更新纯文本


        clip = context.get("clip")

        if positive != None and positive != "":       
            (positive,) = CLIPTextEncode().encode(clip, positive)   
        else:
            positive = context.get("positive")
        
        if negative != None and negative != "":
            (negative,) = CLIPTextEncode().encode(clip, negative)
        else:
            negative = context.get("negative") 

        context = new_context(context, positive=positive, negative=negative, pos=pos, neg=neg, )

        return (context, parameters_data)

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})



#endregion---------加载器-----------------------------------------------------------------------------------#


#region-----------采样器---------------------------------------------------------------------------------------#

class basic_Ksampler_full:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == None"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == None"}),
                "sampler": (['None'] + comfy.samplers.KSampler.SAMPLERS, {"default": "None"}),  
                "scheduler": (['None'] + comfy.samplers.KSampler.SCHEDULERS, {"default": "None"}), 

                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},

        }
        
        
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT","VAE","CLIP", )
    RETURN_NAMES = ("context", "image", "model","positive", "negative",  "latent", "vae", "clip", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"


    def sample(self,  seed, denoise, context=None, clip=None, model=None,vae=None, positive=None, negative=None, latent=None,steps=None, cfg=None, sampler=None, scheduler=None, image=None, prompt=None, image_output=None, extra_pnginfo=None, ):


        if steps == 0:
            steps = context.get("steps")
        if cfg == 0.0:
            cfg = context.get("cfg")
        if sampler == "None":
            sampler = context.get("sampler")
        if scheduler == "None":
            scheduler = context.get("scheduler")



        if positive is None:
            positive = context.get("positive" )
        if negative is None:
            negative = context.get("negative" )
        if vae is None:
            vae= context.get("vae")
        if model is None:
            model= context.get("model")
        if clip is None:
            clip= context.get("clip")


        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = latent or context.get("latent")
            
        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]
        
        
        if image_output == "None":
            context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=None, vae=vae,steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )
            return(context, None, model, positive, negative, latent, vae, clip)


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=output_image, vae=vae,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )

        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}
            
        return {"ui": {"images": results},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}



class basic_Ksampler_mid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},

        }
        
        
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT","VAE","CLIP", )
    RETURN_NAMES = ("context", "image", "model","positive", "negative",  "latent", "vae", "clip", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"


    def sample(self,  seed, denoise, context=None, clip=None, model=None,vae=None, positive=None, negative=None, latent=None, image=None, prompt=None, image_output=None, extra_pnginfo=None, ):


        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")


        if positive is None:
            positive = context.get("positive" )
        if negative is None:
            negative = context.get("negative" )
        if vae is None:
            vae= context.get("vae")
        if model is None:
            model= context.get("model")
        if clip is None:
            clip= context.get("clip")


        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = latent or context.get("latent")
            
        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]
        
        
        if image_output == "None":
            context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=None, vae=vae,steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )
            return(context, None, model, positive, negative, latent, vae, clip)


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=output_image, vae=vae,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )

        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}
            
        return {"ui": {"images": results},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}


class basic_Ksampler_simple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                
                
            },
            
            "optional": {
                "image": ("IMAGE",),
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},

        }


    RETURN_TYPES = ("RUN_CONTEXT",  "IMAGE", )
    RETURN_NAMES = ("context",  "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/ksampler"

    def run(self,context, seed, denoise, image=None,  prompt=None, image_output=None, extra_pnginfo=None, ):


        vae = context.get("vae",None)
        steps = context.get("steps",None)
        cfg = context.get("cfg",None)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)

        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None) 

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]

        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]

        if image_output == "None":
            context = new_context(context, latent=latent, images=None,  )
            return(context, None)


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image,  )
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}


class basic_Ksampler_custom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            
            "optional":
                    {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent": ("LATENT", ),
                    "image": ("IMAGE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "image_output": (["None","Hide", "Preview", "Save", "Hide/Save"], {"default": "None", "tooltip": "  output_image will take up CPU resources "}),
                    
                    },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            
                }
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL","CONDITIONING","CONDITIONING","LATENT", "VAE","IMAGE", )
    RETURN_NAMES = ("context", "model","positive","negative","latent", "vae", "image", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"

    def sample(self, context,model=None,image=None,positive=None, negative=None, latent=None, noise=None, sampler=None, guider=None, sigmas=None, seed=1, denoise=1, prompt=None, image_output=None, extra_pnginfo=None,):
        
        vae=context.get("vae")
        steps = context.get("steps")
        cfg = context.get("cfg")
        scheduler = context.get("scheduler")
        
        if model is None:
            model=context.get("model")
            
        if positive is None:
            positive = context.get("positive",None)
            
        if negative is None:    
            negative = context.get("negative",None)
            
        if sampler is None:
            sampler_name = context.get("sampler",None)
            sampler = KSamplerSelect().get_sampler(sampler_name)[0]  
            
        if noise is None:
            noise = RandomNoise().get_noise(seed)[0] 
            
        if guider is None:
            guider = BasicGuider().get_guider(model, positive)[0] 
            
        if sigmas is None:
            sigmas = BasicScheduler().get_sigmas(model, scheduler, steps, denoise)[0]
            
        if  latent is None:
            latent = context.get("latent")
        
        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        
        out= SamplerCustomAdvanced().sample( noise, guider, sampler, sigmas, latent)
        latent= out[0]
        
        if image_output == "None":
            context = new_context(context, images=None, latent=latent, model=model, positive=positive, negative=negative,  )
            return(context, model, positive, negative, latent, vae, None, ) 
            
        output_image = VAEDecode().decode(vae, latent)[0]  
        latent = VAEEncode().encode(vae, output_image)[0]
        context = new_context(context, images=output_image, latent=latent, model=model, positive=positive, negative=negative,  )   #不能直接更新latent，改成二次编码，占用显存过大
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, model, positive, negative, latent, vae, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, model, positive, negative, latent, vae, output_image,)}


class basic_Ksampler_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Hide"}),
                    },
                "optional": {
                    "latent": ("LATENT", ),
                    },
                
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
                
                }

    RETURN_TYPES = ("RUN_CONTEXT","IMAGE")
    RETURN_NAMES = ("context ", "image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"

    def sample(self, context, add_noise, noise_seed, steps,  start_at_step, return_with_leftover_noise, denoise=1.0, 
                pos="", neg="", latent=None, prompt=None, image_output=None, extra_pnginfo=None, ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        model = context.get("model", None)  
        vae = context.get("vae", None)
        cfg = context.get("cfg", 8.0)  
        sampler_name = context.get("sampler", None) 
        scheduler = context.get("scheduler", None) 
        latent = latent or context.get("latent", None)
        
        positive = context.get("positive", None)
        negative = context.get("negative", None)

        latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=steps, force_full_denoise=force_full_denoise)[0]
        
        
        
        
        if image_output =="None":
            context = new_context(context, latent=latent,images=None)

            return (context, None,)
        
        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image)  
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}


class chx_Ksampler_mix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff}),
                    
                    "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "mid_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "percentage of denoising at mid step"}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    
                    },
                    
                "optional": {
                    "latent": ("LATENT", ),
                    "pos1": ("STRING", {"default": "a dog ", "multiline": True}),
                    "pos2": ("STRING", {"default": "a girl ", "multiline": True}),
                    "neg": ("STRING", {"default": "blur, blurry,", "multiline": False}),
                    "mix_method": (["None","combine", "concat", "average"],),

                    },
                
                }

    RETURN_TYPES = ("RUN_CONTEXT","IMAGE")
    RETURN_NAMES = ("context ", "image")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"

    def sample(self, context, add_noise, noise_seed,  start_step,  steps,  return_with_leftover_noise, denoise=1.0, mid_denoise =0.3,
                pos1="", pos2="", neg="", latent=None, mix_method=None ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        model = context.get("model", None)  
        vae = context.get("vae", None)
        cfg = context.get("cfg", 8.0)  
        sampler_name = context.get("sampler", None) 
        scheduler = context.get("scheduler", None) 
        clip = context.get("clip", None)
        latent = latent or context.get("latent", None)
        
        mix_step = math.ceil(steps * mid_denoise) 

        positive1 = CLIPTextEncode().encode(clip, pos1)[0]
        positive2 = CLIPTextEncode().encode(clip, pos2)[0]
        negative = CLIPTextEncode().encode(clip, neg)[0]

        latent1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive1, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=mix_step, force_full_denoise=force_full_denoise)[0]


        if mix_method == "combine":
            if isinstance(positive1, torch.Tensor) and isinstance(positive2, torch.Tensor):
                if positive1.shape != positive2.shape:
                    positive2 = torch.nn.functional.interpolate(positive2, size=positive1.shape[2:])
                positive2 = positive2 + positive1

        elif mix_method == "concat":
            positive2 = ConditioningConcat().concat(positive1, positive2)[0]

        elif mix_method == "average":
            positive2 = ConditioningAverage().addWeighted(positive1, positive2, 0.5,)[0]
        
        
        latent2 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive2, negative, latent1, denoise=denoise, disable_noise=disable_noise, start_step=mix_step, last_step=steps, force_full_denoise=force_full_denoise)[0]

        output_image = VAEDecode().decode(vae, latent2)[0]
        
        context = new_context(context, latent=latent2, images=output_image, positive=positive2,negative=negative,)  
        
        return (context ,output_image )


class chx_Ksampler_texture:
    def __init__(self):
        pass

    @classmethod

    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tileX": ("INT", {"default": 1, "min": 0, "max": 2}),
                "tileY": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"


    def apply_asymmetric_tiling(self, model, tileX, tileY):
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            print(layer.paddingX, layer.paddingY)

    def __hijackConv2DMethods(self, model, tileX: bool, tileY: bool):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            
            def make_bound_method(method, current_layer):
                def bound_method(self, *args, **kwargs):  # Add 'self' here
                    return method(current_layer, *args, **kwargs)
                return bound_method
                
            bound_method = make_bound_method(self.__replacementConv2DConvForward, layer)
            layer._conv_forward = bound_method.__get__(layer, type(layer))

    def __replacementConv2DConvForward(self, layer, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        working = torch.nn.functional.pad(input, layer.paddingX, mode=layer.padding_modeX)
        working = torch.nn.functional.pad(working, layer.paddingY, mode=layer.padding_modeY)
        return torch.nn.functional.conv2d(working, weight, bias, layer.stride, (0, 0), layer.dilation, layer.groups)

    def __restoreConv2DMethods(self, model):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)
    
    
    def sample(self, context,  seed, tileX, tileY,  denoise=1.0):

        
        vae = context.get("vae")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler_name = context.get("sampler")
        scheduler = context.get("scheduler")

        positive = context.get("positive")
        negative = context.get("negative")
        model = context.get("model")
        latent_image = context.get("latent") 
        
        self.__hijackConv2DMethods(model.model, tileX == 1, tileY == 1)
        result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        self.__restoreConv2DMethods(model.model)

        
        for layer in [layer for layer in vae.first_stage_model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = 'circular'

        out_image = vae.decode(result["samples"])

        return (out_image,)  



class chx_Ksampler_refine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                "upscale_method": (["lanczos", "nearest-exact", "bicubic"],),
                "Add_img_scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                
                
            },
            
            "optional": {
                "image": ("IMAGE",),
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }


    RETURN_TYPES = ("RUN_CONTEXT",  "IMAGE", )
    RETURN_NAMES = ("context",  "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/ksampler"

    def run(self,context, seed, denoise, upscale_model,upscale_method, image=None,  prompt=None, image_output=None, extra_pnginfo=None,Add_img_scale=1):


        vae = context.get("vae",None)
        steps = context.get("steps",None)
        cfg = context.get("cfg",None)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)

        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None) 

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]

        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]

        if image_output == "None":
            context = new_context(context, latent=latent, images=None, )
            return(context, None)


        output_image = VAEDecode().decode(vae, latent)[0]
        
        
        upimage = image_upscale(output_image, upscale_method, Add_img_scale)[0]
        up_model = load_upscale_model(upscale_model)
        output_image = upscale_with_model(up_model, upimage )
        
        context = new_context(context, latent=latent, images=output_image,  )
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}



#endregion-----------采样器--------------------------------------------------------------------------------#


#region-----------tool--------------------------------------------------------------------------------------#--



class chx_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "context": ("RUN_CONTEXT",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "controlNet": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("context","positive", "negative",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_controlnet"

    def load_controlnet(self,  
                        strength,control_net=None,
                        context=None, controlNet=None, 
                        image=None,  extra_concat=[]):

        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)

        if control_net is not None:
                out=ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image, strength, 0, 1, vae, extra_concat)
                positive=out[0]
                negative=out[1]

        if control_net is None:
            if controlNet == "None" or image is None:
                return (context,)
            if controlNet != "None" and strength != 0 and image is not None:
                control_net = ControlNetLoader().load_controlnet(controlNet)[0]
                out=ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image, strength, 0, 1, vae, extra_concat)
                positive=out[0]
                negative=out[1]


        context = new_context(context, positive=positive, negative=negative)
        return (context, )



class chx_controlnet_union:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "context": ("RUN_CONTEXT",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "controlNet": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "type1": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength2": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "type2": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "type3": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),)
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","CONDITIONING", )
    RETURN_NAMES = ("context", "positive", "negative",  )
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_controlnet"

    def load_controlnet(self,  strength1, strength2, strength3, context=None, image1=None, image2=None, image3=None,
                        controlNet=None, type1=None,type2=None, type3=None,
                        extra_concat=[]):

        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)
        if controlNet == "None" :
            return (context, )
        control_net = ControlNetLoader().load_controlnet(controlNet)[0]


        if type1!= "None" and strength1 != 0 and image1 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type( control_net, type1)[0]
            out = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image1, strength1, 0, 1, vae, extra_concat)


        if type2!= "None" and strength2 != 0 and image2 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type( control_net, type2)[0]
            out = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image2, strength2, 0, 1, vae, extra_concat)


        if type3!= "None" and strength3 != 0 and image3 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type( control_net, type3)[0]
            out = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image3, strength3, 0, 1, vae, extra_concat)

        positive = out[0]
        negative = out[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative, )



class chx_condi_hook:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            },
            "optional": {
                "mask": ("MASK", ),
                "hooks": ("HOOKS",),
                "timesteps": ("TIMESTEPS_RANGE",),
            }
        }

    EXPERIMENTAL = True
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ( "context","positive",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "set_properties"


    def set_properties(self, context,
                    strength: float, set_cond_area: str,
                    mask: torch.Tensor=None, hooks: comfy.hooks.HookGroup=None, timesteps: tuple=None):
        
        cond_NEW = context.get("positive")

        (final_cond,) = comfy.hooks.set_conds_props(conds=[cond_NEW],
                                                strength=strength, set_cond_area=set_cond_area,
                                                mask=mask, hooks=hooks, timesteps_range=timesteps)
        
        context=new_context(context,positive=final_cond,)
        
        return (context , final_cond,)



class mask_Mulcondition:
    @classmethod
    def INPUT_TYPES(s):
        return {

            "optional": {
                
                "context": ("RUN_CONTEXT",),
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos4": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos5": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_5": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_5_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),

                "neg": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "Poor quality" }),
            }
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("context","positive", "negative",)


    FUNCTION = "Mutil_Clip"
    CATEGORY = "Apt_Preset/chx_tool"


    def Mutil_Clip (self, pos1, pos2, pos3, pos4, pos5, neg,  set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_5_strength,mask_1=None, mask_2=None, mask_3=None, mask_4=None, mask_5=None,context=None,):

        clip = context.get("clip")
        positive_1, = CLIPTextEncode().encode(clip, pos1)
        positive_2, = CLIPTextEncode().encode(clip, pos2)
        positive_3, = CLIPTextEncode().encode(clip, pos3)
        positive_4, = CLIPTextEncode().encode(clip, pos4)
        positive_5, = CLIPTextEncode().encode(clip, pos5)
        negative, = CLIPTextEncode().encode(clip, neg)

        c = []
        set_area_to_bounds = False
        if set_cond_area!= "default":
            set_area_to_bounds = True

        if mask_1 is not None and len(mask_1.shape) < 3:  
            mask_1 = mask_1.unsqueeze(0)
        if mask_2 is not None and len(mask_2.shape) < 3:  
            mask_2 = mask_2.unsqueeze(0)
        if mask_3 is not None and len(mask_3.shape) < 3:  
            mask_3 = mask_3.unsqueeze(0)
        if mask_4 is not None and len(mask_4.shape) < 3:  
            mask_4 = mask_4.unsqueeze(0)
        if mask_5 is not None and len(mask_5.shape) < 3:  
            mask_5 = mask_5.unsqueeze(0)

        if mask_1 is not None:  # 新增判断，如果 mask_1 不为 None 才处理 positive_1
            for t in positive_1:
                append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        if mask_2 is not None:  # 新增判断，如果 mask_2 不为 None 才处理 positive_2
            for t in positive_2:
                append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        if mask_3 is not None:  # 新增判断，如果 mask_3 不为 None 才处理 positive_3
            for t in positive_3:
                append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        if mask_4 is not None:  # 新增判断，如果 mask_4 不为 None 才处理 positive_4
            for t in positive_4:
                append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
        if mask_5 is not None:  # 新增判断，如果 mask_5 不为 None 才处理 positive_5
            for t in positive_5:
                append_helper(t, mask_5, c, set_area_to_bounds, mask_5_strength)


        context = new_context(context, positive=c, negative=negative,clip=clip, )

        return (context, c, negative)



class chx_Upscale_simple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                "upscale_method": (["lanczos", "nearest-exact", "bicubic"],),
                "Add_img_scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "upscale"

    def upscale(self, upscale_model,upscale_method, image=None, context=None, Add_img_scale=1):
        
        vae = context.get("vae", None)
        if image is None :
            image = context.get("images", None)  

        upimage = image_upscale(image, upscale_method, Add_img_scale)[0]
        up_model = load_upscale_model(upscale_model)
        images_out = upscale_with_model(up_model, upimage)

        latent = VAEEncode().encode(vae, images_out)[0]
        
        context = new_context(context, images=images_out, latent=latent)
        
        return (context, images_out)


class chx_vae_encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            "optional": {
                "context": ("RUN_CONTEXT",),
                "latent": ("LATENT",),
                "vae": ("VAE",),

            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "upscale"

    def decode(self, vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )

    def upscale(self, context, latent=None, vae=None,):
        
        if latent is None :
            latent = context.get("latent", None)

        if vae is None :
            vae = context.get("vae", None)

        # 修改此处，通过 self 调用 decode 方法
        images_out = self.decode(vae, latent)

        context = new_context(context, latent=latent , vae=vae, images=images_out,) 
        return (context, images_out)


class chx_vae_encode_tile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                    "tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    },
            "optional": {
                    "context": ("RUN_CONTEXT",),
                    "samples": ("LATENT", ), 
            }

                }
    

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "encode_tile"


    def encode_tile(self, context, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):

        vae = context.get("vae", None)
        if samples is None :
            samples = context.get("latent", None)

        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        context = new_context(context, latent=samples , vae=vae, images=images,) 
        return (context, images)





#endregion-----------tool--------------------------------------------------------------------------------------#--


#region-----------风格组--------------------------------------------------------------------------------------#--



class chx_YC_LG_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            "context": ("RUN_CONTEXT",),
            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
            "image": ("IMAGE",),
            "crop": (["center", "mask_area", "none"], {
                "default": "none",
                "tooltip": "裁剪模式：center-中心裁剪, mask_area-遮罩区域裁剪, none-不裁剪"
            }),
            "sharpen": ("FLOAT", {
                "default": 0.0,
                "min": -5.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "锐化强度：负值为模糊，正值为锐化，0为不处理"
            }),
            "patch_res": ("INT", {
                "default": 16,
                "min": 1,
                "max": 64,
                "step": 1,
                "tooltip": "patch分辨率，数值越大分块越细致"
            }),
            "style_strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "风格强度，越高越偏向参考图片"
            }),
            "prompt_strength": ("FLOAT", { 
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "文本提示词强度，越高文本特征越强"
            }),
            "blend_mode": (["lerp", "feature_boost", "frequency"], {
                "default": "lerp",
                "tooltip": "风格强度的计算方式：\n" +
                        "lerp - 线性混合 - 高度参考原图\n" +
                        "feature_boost - 特征增强 - 增强真实感\n" +
                        "frequency - 频率增强 - 增强高频细节"
            }),
            "noise_level": ("FLOAT", { 
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "添加随机噪声的强度，可用于修复错误细节"
            }),
        },
        "optional": { 
            "mask": ("MASK", ), 
            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
        
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/IPA"

    def crop_to_mask_area(self, image, mask):
        if len(image.shape) == 4:
            B, H, W, C = image.shape
            image = image.squeeze(0)
        else:
            H, W, C = image.shape
        
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
        
        nonzero_coords = torch.nonzero(mask)
        if len(nonzero_coords) == 0:
            return image, mask
        
        top = nonzero_coords[:, 0].min().item()
        bottom = nonzero_coords[:, 0].max().item()
        left = nonzero_coords[:, 1].min().item()
        right = nonzero_coords[:, 1].max().item()
        
        width = right - left
        height = bottom - top
        size = max(width, height)
        
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        
        half_size = size // 2
        new_top = max(0, center_y - half_size)
        new_bottom = min(H, center_y + half_size)
        new_left = max(0, center_x - half_size)
        new_right = min(W, center_x + half_size)
        
        cropped_image = image[new_top:new_bottom, new_left:new_right]
        cropped_mask = mask[new_top:new_bottom, new_left:new_right]
        
        cropped_image = cropped_image.unsqueeze(0)
        cropped_mask = cropped_mask.unsqueeze(0)
        
        return cropped_image, cropped_mask
    
    def apply_image_preprocess(self, image, strength):
        original_shape = image.shape
        original_device = image.device
        
        if torch.is_tensor(image):
            if len(image.shape) == 4:
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        if strength < 0:
            abs_strength = abs(strength)
            kernel_size = int(3 + abs_strength * 12) // 2 * 2 + 1
            sigma = 0.3 + abs_strength * 2.7
            processed = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
        elif strength > 0:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * strength + np.array([[0,0,0],
                                                               [0,1,0],
                                                               [0,0,0]]) * (1 - strength)
            processed = cv2.filter2D(image_np, -1, kernel)
            processed = np.clip(processed, 0, 255)
        else:
            processed = image_np
        
        processed_tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).to(original_device)
        if len(original_shape) == 4:
            processed_tensor = processed_tensor.unsqueeze(0)
        
        return processed_tensor
    
    def apply_style_strength(self, cond, txt, strength, mode="lerp"):
        if mode == "lerp":
            if txt.shape[1] != cond.shape[1]:
                txt_mean = txt.mean(dim=1, keepdim=True)
                txt_expanded = txt_mean.expand(-1, cond.shape[1], -1)
                return torch.lerp(txt_expanded, cond, strength)
            return torch.lerp(txt, cond, strength)
        
        elif mode == "feature_boost":
            mean = torch.mean(cond, dim=-1, keepdim=True)
            std = torch.std(cond, dim=-1, keepdim=True)
            normalized = (cond - mean) / (std + 1e-6)
            boost = torch.tanh(normalized * (strength * 2.0))
            return cond * (1 + boost * 2.0)
    
        elif mode == "frequency":
            try:
                B, N, C = cond.shape
                x = cond.float()
                fft = torch.fft.rfft(x, dim=-1)
                magnitudes = torch.abs(fft)
                phases = torch.angle(fft)
                freq_dim = fft.shape[-1]
                freq_range = torch.linspace(0, 1, freq_dim, device=cond.device)
                alpha = 2.0 * strength
                beta = 0.5
                filter_response = 1.0 + alpha * torch.pow(freq_range, beta)
                filter_response = filter_response.view(1, 1, -1)
                enhanced_magnitudes = magnitudes * filter_response
                enhanced_fft = enhanced_magnitudes * torch.exp(1j * phases)
                enhanced = torch.fft.irfft(enhanced_fft, n=C, dim=-1)
                mean = enhanced.mean(dim=-1, keepdim=True)
                std = enhanced.std(dim=-1, keepdim=True)
                enhanced_norm = (enhanced - mean) / (std + 1e-6)
                mix_ratio = torch.sigmoid(torch.tensor(strength * 2 - 1))
                result = torch.lerp(cond, enhanced_norm.to(cond.dtype), mix_ratio)
                residual = (result - cond) * strength
                final = cond + residual
                return final
            except Exception as e:
                print(f"频率处理出错: {e}")
                print(f"输入张量形状: {cond.shape}")
                return cond
                
        return cond
    
    def apply_stylemodel(self, style_model, clip_vision, image, 
                        patch_res=16, style_strength=1.0, prompt_strength=1.0, 
                        noise_level=0.0, crop="none", sharpen=0.0, guidance=30,
                        blend_mode="lerp", mask=None, context=None):
        
        
        conditioning = context.get("positive", None)  

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)


        
        processed_image = image.clone()
        if sharpen != 0:
            processed_image = self.apply_image_preprocess(processed_image, sharpen)
        if crop == "mask_area" and mask is not None:
            processed_image, mask = self.crop_to_mask_area(processed_image, mask)
            clip_vision_output = clip_vision.encode_image(processed_image, crop=False)
        else:
            crop_image = True if crop == "center" else False
            clip_vision_output = clip_vision.encode_image(processed_image, crop=crop_image)
        
        cond = style_model.get_cond(clip_vision_output)
        
        B = cond.shape[0]
        H = W = int(math.sqrt(cond.shape[1]))
        C = cond.shape[2]
        cond = cond.reshape(B, H, W, C)
        
        new_H = H * patch_res // 16
        new_W = W * patch_res // 16
        
        cond = torch.nn.functional.interpolate(
            cond.permute(0, 3, 1, 2),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        cond = cond.permute(0, 2, 3, 1)
        cond = cond.reshape(B, -1, C)
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        c_out = []
        for t in conditioning:
            txt, keys = t
            keys = keys.copy()
            
            if prompt_strength != 1.0:
                txt_enhanced = txt * (prompt_strength ** 3)
                txt_repeated = txt_enhanced.repeat(1, 2, 1)
                txt = txt_repeated
            
            if style_strength != 1.0:
                processed_cond = self.apply_style_strength(
                    cond, txt, style_strength, blend_mode
                )
            else:
                processed_cond = cond
    
            if mask is not None:
                feature_size = int(math.sqrt(processed_cond.shape[1]))
                processed_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1) if mask.dim() == 3 else mask,
                    size=(feature_size, feature_size),
                    mode='bilinear',
                    align_corners=False
                ).flatten(1).unsqueeze(-1)
                
                if txt.shape[1] != processed_cond.shape[1]:
                    txt_mean = txt.mean(dim=1, keepdim=True)
                    txt_expanded = txt_mean.expand(-1, processed_cond.shape[1], -1)
                else:
                    txt_expanded = txt
                
                processed_cond = processed_cond * processed_mask + \
                               txt_expanded * (1 - processed_mask)
    
            if noise_level > 0:
                noise = torch.randn_like(processed_cond)
                noise = (noise - noise.mean()) / (noise.std() + 1e-8)
                processed_cond = torch.lerp(processed_cond, noise, noise_level)
                processed_cond = processed_cond * (1.0 + noise_level)
                
            c_out.append([torch.cat((txt, processed_cond), dim=1), keys])
        
        
        
        positive = node_helpers.conditioning_set_values(c_out, {"guidance": guidance})

        context = new_context(context, positive=positive,)
        
        return (context,positive,)



class chx_StyleModelApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "context": ("RUN_CONTEXT",),
                            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
                            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
                            "image": ("IMAGE",),
                            
                            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                            "strength_type": (["multiply", "attn_bias"], ),
                            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
                            }}

    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/IPA"

    def apply_stylemodel(self, style_model, clip_vision, strength, strength_type, guidance=30, context=None, image=None):
        
        conditioning = context.get("positive", None)  

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)
        clip_vision_output = clip_vision.encode_image(image, crop="center") 
    
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if strength_type == "attn_bias" and strength != 1.0:
                attn_bias = torch.log(torch.Tensor([strength]))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]

                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                if mask.dtype == torch.bool:

                    mask = torch.log(mask.to(dtype=torch.float16))
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)

                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]

                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])
        
        positive = node_helpers.conditioning_set_values(c_out, {"guidance": guidance})
        context = new_context(context, positive=positive,)
        return (context,positive,)  


class chx_re_fluxguide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            "optional": {
                "context": ("RUN_CONTEXT",),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "fluxguide"
    CATEGORY = "Apt_Preset/chx_tool"

    def fluxguide(self,context, guidance, ):  

        positive = context.get("positive")
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        

        context = new_context(context, positive=positive, guidance=guidance )
        return (context, )



class chx_Style_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
            "image": ("IMAGE",),
            "style_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制整体艺术风格的权重"
            }),
            "color_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制颜色特征的权重"
            }),
            "content_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制内容语义的权重"
            }),
            "structure_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制结构布局的权重"
            }),
            "texture_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制纹理细节的权重"
            }),
            "similarity_threshold": ("FLOAT", {
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "特征相似度阈值，超过此值的区域将被替换"
            }),
            "enhancement_base": ("FLOAT", {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "tooltip": "文本特征替换的基础增强系数"
            })
        },
        
        "optional": { 
            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
        
    
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    
    FUNCTION = "apply_style"
    CATEGORY = "Apt_Preset/IPA"

    def __init__(self):
        
        import comfy.ops
        ops = comfy.ops.manual_cast

        self.text_projector = ops.Linear(4096, 4096)  # 保持维度一致
        # 为不同类型特征设置增强系数
        self.enhancement_factors = {
            'style': 1.2,    # 风格特征增强系数
            'color': 1.0,    # 颜色特征增强系数
            'content': 1.1,  # 内容特征增强系数
            'structure': 1.3, # 结构特征增强系数
            'texture': 1.0   # 纹理特征增强系数
        }

    def compute_similarity(self, text_feat, image_feat):
        """计算多种相似度的组合"""
        # 1. 余弦相似度
        cos_sim = torch.cosine_similarity(text_feat, image_feat, dim=-1)
        
        l2_dist = torch.norm(text_feat - image_feat, p=2, dim=-1)
        l2_sim = 1 / (1 + l2_dist)  # 转换为相似度
        
        dot_sim = torch.sum(text_feat * image_feat, dim=-1)
        dot_sim = torch.tanh(dot_sim)  # 归一化到[-1,1]
        
        attn_weights = torch.softmax(torch.matmul(text_feat, image_feat.transpose(-2, -1)) / math.sqrt(text_feat.size(-1)), dim=-1)
        attn_sim = torch.mean(attn_weights, dim=-1)
        
        combined_sim = (
            0.4 * cos_sim +
            0.2 * l2_sim +
            0.2 * dot_sim +
            0.2 * attn_sim
        )
        
        return combined_sim.mean()

    def apply_style(self, style_weight=1.0, color_weight=1.0, content_weight=1.0,guidance=30,
                structure_weight=1.0, texture_weight=1.0, image=None, style_model=None, clip_vision=None,
                similarity_threshold=0.7, enhancement_base=1.5,context=None,):
        
        conditioning = context.get("positive", None)  

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)
        clip_vision_output = clip_vision.encode_image(image, crop="center") 
    
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        
        text_features = conditioning[0][0]  # [batch_size, seq_len, 4096]
        text_features = text_features.mean(dim=1)  # [batch_size, 4096]
        
        text_features = self.text_projector(text_features)  # [batch_size, 4096]
        
        if text_features.shape[0] != image_cond.shape[0]:
            text_features = text_features.expand(image_cond.shape[0], -1)

        feature_size = image_cond.shape[-1]  # 4096
        splits = feature_size // 5  # 每部分约819维

        image_features = {
            'style': image_cond[..., :splits],
            'color': image_cond[..., splits:splits*2],
            'content': image_cond[..., splits*2:splits*3],
            'structure': image_cond[..., splits*3:splits*4],
            'texture': image_cond[..., splits*4:]
        }
        
        similarities = {}
        for key, region_features in image_features.items():
            region_text_features = text_features[..., :region_features.shape[-1]]
            similarities[key] = self.compute_similarity(region_text_features, region_features)
        final_features = {}
        weights = {
            'style': style_weight,
            'color': color_weight,
            'content': content_weight,
            'structure': structure_weight,
            'texture': texture_weight
        }
        
        for key in image_features:
            if similarities[key] > similarity_threshold:
                region_size = image_features[key].shape[-1]
                dynamic_factor = enhancement_base * self.enhancement_factors[key]
                final_features[key] = text_features[..., :region_size] * weights[key] * dynamic_factor
            else:
                final_features[key] = image_features[key] * weights[key]
        
        # 合并所有特征
        combined_cond = torch.cat([
            final_features['style'],
            final_features['color'],
            final_features['content'],
            final_features['structure'],
            final_features['texture']
        ], dim=-1).unsqueeze(dim=0)
        
        # 构建新的条件
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], combined_cond), dim=1), t[1].copy()]
            c.append(n)
            
        positive = node_helpers.conditioning_set_values(c, {"guidance": guidance})
        
        context = new_context(context, positive=positive,)
        return (context,positive,)  



#endregion-----------风格组--------------------------------------------------------------------------------------#--


#------------------


