


#region-----------导入与全局函数-------------------------------------------------
# 内置库
import os
import sys
import glob
import csv
import json
import math
import re
#from turtle import width



# 第三方库
import numpy as np
import torch
import toml
import aiohttp
from aiohttp import web
from PIL import Image, ImageFilter
from tqdm import tqdm

import nodes
# 本地库
import comfy
import folder_paths
import node_helpers
import latent_preview
from server import PromptServer
from nodes import common_ksampler, CLIPTextEncode, ControlNetLoader, LoadImage, ControlNetApplyAdvanced, VAEDecode, VAEEncode, DualCLIPLoader, CLIPLoader,ConditioningConcat, ConditioningAverage, InpaintModelConditioning, LoraLoader, CheckpointLoaderSimple, ImageScale,VAEDecodeTiled,VAELoader
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
from comfy_extras.nodes_hidream import QuadrupleCLIPLoader
from comfy_extras.nodes_hooks import PairConditioningSetProperties
from comfy.utils import load_torch_file as comfy_load_torch_file

from .load_GGUF.nodes import  UnetLoaderGGUF2
from ..main_unit import *




#---------------------安全导入------

try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  

try:
    from comfy_extras.nodes_model_patch import ModelPatchLoader, QwenImageDiffsynthControlnet
    REMOVER_AVAILABLE = True  
except ImportError:
    ModelPatchLoader = None
    QwenImageDiffsynthControlnet = None
    REMOVER_AVAILABLE = False  



#region---------------注册和检测gguf----------------------------------------

def update_folder_names_and_paths(key, targets=[]):
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])




def check_UnetLoaderGGUF2_installed():
    if UnetLoaderGGUF2 is None:
        raise RuntimeError(" Please install the plugin ComfyUI-GGUF first")





#endregion---------------注册gguf----------------------------------------




#region------------------------preset---------------------------------#


# 获取当前文件所在目录的上一级目录
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 插入上一级目录下的 comfy 到 sys.path
sys.path.insert(0, os.path.join(parent_directory, "comfy"))

routes = PromptServer.instance.routes
@routes.post('/Apt_Preset_path')
async def my_function(request):
    the_data = await request.post()
    sum_load_adv.handle_my_message(the_data)
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
available_unets = list(set(folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf")))
available_clips = list(set(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")))

CLIP_TYPE = ["sdxl", "sd3", "flux", "hunyuan_video", "stable_diffusion", "stable_audio", "mochi", 
             "ltxv", "pixart", "cosmos","lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image"]



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


class Data_sampleData:  
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



class Data_select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "context": ("RUN_CONTEXT",),           
                        },
        
            "optional": {
                "type": (["model", "clip", "positive", "negative", "vae", "latent", "images", "mask",
                        "clip1", 
                        "clip2", 
                        "clip3", 
                        "clip4",
                        "unet_name", 
                        "ckpt_name",
                        "pos", 
                        "neg",
                        "width",
                        "height",
                        "batch",
                        "data",
                        "data1",
                        "data2",
                        "data3",
                        "data4",
                        "data5",
                        "data6",
                        "data7",
                        "data8",
                        "data9",
                        ], {}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
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
                    available_ckpt,
                    available_unets, 
                    available_clips,
                    available_clips,
                    available_clips,  
                    available_clips,  
                    "STRING", 
                    "STRING", 
                    "INT",
                    "INT",
                    "INT",
                    )

    RETURN_NAMES = (
            "context",
            "ckpt_name",
            "unet_name", 
            "clip1_name", 
            "clip2_name", 
            "clip3_name", 
            "clip4_name", 
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
        clip4 = context.get("clip3", None)
        pos = context.get("pos", None)
        neg = context.get("neg", None)
        width = context.get("width", None)
        height = context.get("height", None)
        batch = context.get("batch", None)

        return (
            context,
            ckpt_name,
            unet_name, 
            clip1, 
            clip2, 
            clip3, 
            clip4, 
            pos, 
            neg,
            width,
            height,
            batch
        )


class Data_bus_chx:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "optional": {
                "context": ("RUN_CONTEXT",),   
                "data1": ( ANY_TYPE, ),
                "data2": ( ANY_TYPE, ),
                "data3": ( ANY_TYPE, ),
                "data4": ( ANY_TYPE, ),
                "data5": ( ANY_TYPE, ),
                "data6": ( ANY_TYPE, ),
                "data7": ( ANY_TYPE, ),
                "data8": ( ANY_TYPE, ),
            },
        }


    RETURN_TYPES = ("RUN_CONTEXT",ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,)
    RETURN_NAMES = ("context", "data1","data2","data3","data4","data5","data6","data7","data8",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context=None, data1=None, data2=None, data3=None, data4=None, data5=None, data6=None, data7=None, data8=None):
        # 先检查 context 是否为 None
        if context is None:
            # 如果 context 为 None，可以创建一个新的上下文或者根据需求处理
            # 这里假设 new_context 可以在没有输入的情况下创建一个默认的上下文
            context = {}  # 或者使用其他方式初始化一个新的 context

        if data1 is None:
            data1 = context.get("data1")
        if data2 is None:
            data2 = context.get("data2")
        if data3 is None:
            data3 = context.get("data3")
        if data4 is None:
            data4 = context.get("data4")
        if data5 is None:
            data5 = context.get("data5")
        if data6 is None:
            data6 = context.get("data6")
        if data7 is None:
            data7 = context.get("data7")
        if data8 is None:
            data8 = context.get("data8")

        context = new_context(context, data1=data1, data2=data2, data3=data3, data4=data4, data5=data5, data6=data6, data7=data7, data8=data8)

        return (context, data1,data2,data3,data4,data5,data6,data7,data8,)

#endregion


#region-----------加载器load-----------------------------------------------------------------------------------#



def safe_load_torch_file(path, device="cpu"):
    """兼容 PyTorch 2.6 的加载方式，先检测再降低安全等级"""
    try:
        return comfy_load_torch_file(path, device=device)
    except Exception as e:
        logger.warning(f"Failed with weights_only=True: {e}")
        logger.info("Retrying with weights_only=False (unsafe)")
        return torch.load(path, map_location=device, weights_only=False)



class sum_load_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional":{
                "preset": (["None"] + preset_list, {"default": "None"}),
                "ckpt_name": (["None"] + available_ckpt,),
                "unet_name": (["None"] + available_unets,),
                "unet_Weight_Dtype": (["None", "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_type": (["None"] + CLIP_TYPE,),
                "clip1": (["None"] + available_clips,),
                "clip2": (["None"] + available_clips,),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "clip3": (["None"] + available_clips,),
                "clip4": (["None"] + available_clips,),
                "vae": (["None"] + available_vaes,),
                "lora": (["None"] + available_loras,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "pos": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "a girl"}),
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "worst quality, low quality"}),
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),
                "lora_stack": ("LORASTACK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "PDATA")
    RETURN_NAMES = ("context", "model", "preset_save")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"

    def process_settings(self,
                        node_id,
                        lora_strength,
                        width, height, steps, cfg, sampler, scheduler,
                        unet_Weight_Dtype, guidance, clip_type=None, device="default",
                        vae=None, lora=None, unet_name=None, ckpt_name=None,
                        clip1=None, clip2=None, clip3=None, clip4=None,
                        pos="default", neg="default", over_model=None, over_clip=None, lora_stack=None, preset=[]):

        if preset != "None":
            pass

        # 构建参数记录
        parameters_data = [{
            "run_Mode": None,
            "ckpt_name": ckpt_name,
            "clipnum": -2,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": clip3,
            "clip4": clip4,
            "vae": vae,
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": 1,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
        }]

        # 分辨率修正
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([1, 4, height // 8, width // 8])
        if latent.shape[1] != 16:
            latent = latent.repeat(1, 16 // 4, 1, 1)

        # 加载模型
        model = None
        clip = over_clip
        vae2 = None

        if over_model is not None:
            model = over_model
        elif ckpt_name != "None" and unet_name == "None":
            model, clip, vae2 = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        elif unet_name != "None" and ckpt_name == "None":
            if unet_name.endswith(".gguf"):
                from .load_GGUF.nodes import UnetLoaderGGUF2
                if UnetLoaderGGUF2 is None:
                    raise RuntimeError("Please install ComfyUI-GGUF plugin.")
                result = UnetLoaderGGUF2().load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model = UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]
        elif ckpt_name != "None" and unet_name != "None":
            raise ValueError("ckpt_name and unet_name cannot be entered at the same time. Please enter only one of them.")

        # 加载 CLIP
        if over_clip is not None:
            clip = over_clip
        elif clip1 == "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            pass
        elif clip1 != "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import CLIPLoaderGGUF2
                clip = CLIPLoaderGGUF2().load_clip(clip1, clip_type)[0]
            else:
                clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import DualCLIPLoaderGGUF2
                clip = DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
            else:
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import TripleCLIPLoaderGGUF2
                clip = TripleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip_type="sd3")[0]
            else:
                clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 != "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import QuadrupleCLIPLoaderGGUF2
                clip = QuadrupleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip4, clip_type="stable_diffusion")[0]
            else:
                clip = QuadrupleCLIPLoader().load_clip(clip1, clip2, clip3, clip4)[0]

        # 应用 LoRA
        if lora_stack is not None:
            model, clip = apply_lora_stack(model, clip, lora_stack)
        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)

        # 编码提示词
        if clip is not None:
            (positive,) = CLIPTextEncode().encode(clip, pos)
            (negative,) = CLIPTextEncode().encode(clip, neg)
        else:
            positive = None
            negative = None

        # 处理 guidance
        if clip1 != "None" and clip2 != "None":
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        # 处理 VAE
        if isinstance(vae, str) and vae != "None":
            vae = VAELoader().load_vae(vae)[0]
        elif vae2 is not None:
            vae = vae2

        # 构造 context
        context = new_context(None, **{
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
            "clip4": clip4,
            "unet_name": unet_name,
            "ckpt_name": ckpt_name,
            "pos": pos,
            "neg": neg,
            "width": width,
            "height": height,
            "batch": 1,
        })

        return (context, model, parameters_data)

    @classmethod
    def handle_my_message(cls, d):
        """从 Web 接收 preset 文件并发送到前端"""
        preset_path = os.path.join(presets_directory_path, d['message'])
        if not os.path.exists(preset_path):
            logger.error(f"Preset file not found: {preset_path}")
            return
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message": preset_data, "node": d['node_id']})




class sum_load_only_model():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional":{
                "ckpt_name": (["None"] + available_ckpt,),
                "unet_name": (["None"] + available_unets,),
                "unet_Weight_Dtype": (["None", "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_type": (["None"] + CLIP_TYPE,),
                "clip1": (["None"] + available_clips,),
                "clip2": (["None"] + available_clips,),
                "clip3": (["None"] + available_clips,),
                "clip4": (["None"] + available_clips,),
                "vae": (["None"] + available_vaes,),
                "lora": (["None"] + available_loras,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),
                "over_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL","CLIP","VAE" )
    RETURN_NAMES = ("model","clip","vae")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"


    def process_settings(self,
                        lora_strength,
                        unet_Weight_Dtype, clip_type=None, device="default",
                        vae=None, lora=None, unet_name=None, ckpt_name=None,
                        clip1=None, clip2=None, clip3=None, clip4=None,
                        pos="default", neg="default", over_model=None, over_clip=None, over_vae=None, ):


        # 加载模型
        model = None
        clip = over_clip
        vae2 = None

        if over_model is not None:
            model = over_model
        elif ckpt_name != "None" and unet_name == "None":
            model, clip, vae2 = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        elif unet_name != "None" and ckpt_name == "None":
            if unet_name.endswith(".gguf"):
                from .load_GGUF.nodes import UnetLoaderGGUF2
                if UnetLoaderGGUF2 is None:
                    raise RuntimeError("Please install ComfyUI-GGUF plugin.")
                result = UnetLoaderGGUF2().load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model = UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]
        elif ckpt_name != "None" and unet_name != "None":
            raise ValueError("ckpt_name and unet_name cannot be entered at the same time. Please enter only one of them.")

        # 加载 CLIP
        if over_clip is not None:
            clip = over_clip
        elif clip1 == "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            pass
        elif clip1 != "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import CLIPLoaderGGUF2
                clip = CLIPLoaderGGUF2().load_clip(clip1, clip_type)[0]
            else:
                clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import DualCLIPLoaderGGUF2
                clip = DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
            else:
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import TripleCLIPLoaderGGUF2
                clip = TripleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip_type="sd3")[0]
            else:
                clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 != "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import QuadrupleCLIPLoaderGGUF2
                clip = QuadrupleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip4, clip_type="stable_diffusion")[0]
            else:
                clip = QuadrupleCLIPLoader().load_clip(clip1, clip2, clip3, clip4)[0]

        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)

        if isinstance(vae, str) and vae != "None":
            vae = VAELoader().load_vae(vae)[0]
        elif vae2 is not None:
            vae = vae2

        if over_vae is not None:
            vae = over_vae

        return (model,clip,vae )








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
            "clip4": None,
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
            "clip4": None, 
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
            "clip4": None,
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
            if unet_name.endswith(".gguf"):
                check_UnetLoaderGGUF2_installed()
                loader = UnetLoaderGGUF2()  # 创建实例
                result = loader.load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model=UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]



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
            "clip4": None,
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
            "clip4": None,
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
            if unet_name.endswith(".gguf"):
                check_UnetLoaderGGUF2_installed()
                loader = UnetLoaderGGUF2()  # 创建实例
                result = loader.load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model=UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]

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
            "clip4": None,
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
        tomltext += f"clip4 = \"{fix_path(param[0]['clip4'])}\"\n"
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



class pre_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                # 第一个ControlNet相关参数
                "image1": ("IMAGE",),
                "controlnet1": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 第二个ControlNet相关参数
                "image2": ("IMAGE",),
                "controlnet2": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 第三个ControlNet相关参数
                "image3": ("IMAGE",),
                "controlnet3": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("context","positive", "negative",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, 
                        strength1,strength2, strength3, start_percent1=0.0, end_percent1=1.0,
                         start_percent2=0.0, end_percent2=1.0,
                        start_percent3=0.0, end_percent3=1.0,
                        context=None, 
                        controlnet1=None, controlnet2=None, controlnet3=None,
                        image1=None, image2=None, image3=None, vae=None,):


        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)

        # 处理第一个ControlNet
        if controlnet1 != "None" and image1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet1)
            controlnet1 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet1, image1, 
                strength1, start_percent1, end_percent1, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        # 处理第二个ControlNet
        if controlnet2 != "None" and image2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet2)
            controlnet2 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet2, image2, 
                strength2, start_percent2, end_percent2, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        # 处理第三个ControlNet
        if controlnet3 != "None" and image3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet3)
            controlnet3 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet3, image3, 
                strength3, start_percent3, end_percent3, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative, )





class sum_lora:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_01":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_02":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_03":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "context": ("RUN_CONTEXT",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "style": (["None"] + style_list()[0],{"default": "None"}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("context",  "positive", "negative",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_lora"

    def load_lora(self, style, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, context=None, clip=None, model=None,  pos="", neg=""):

        if model is None:
            model=  context.get("model",None)
        
        if clip is None:
            clip=  context.get("clip",None)

        positive = context.get("positive")
        negative = context.get("negative") 

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
            
        context = new_context(context, model=model, clip=clip, positive=positive, negative=negative, )
        return (context, positive, negative,)



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
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "style": (["None"] + style_list()[0], {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, }),
                "ratio_selected": (['None'] + s.ratio_sizes, {"default": "None"}),

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

    def text(self, context=None, model=None, clip=None, positive=None, negative=None, pos="", neg="", image=None, vae=None, latent=None, steps=None, cfg=None, sampler=None, scheduler=None, style=None, batch_size=1, ratio_selected=None, guidance=0 ):

        width = context.get("width")
        height = context.get("height")       
        if ratio_selected and ratio_selected != "None" and ratio_selected in self.ratio_dict:
            try:
                width = self.ratio_dict[ratio_selected]["width"]
                height = self.ratio_dict[ratio_selected]["height"]
            except KeyError as e:
                print(f"[ERROR] Invalid ratio selected: {e}")
                width = context.get("width", 512)
                height = context.get("height", 512)
        else:
            if width is None or height is None:
                width = 512
                height = 512       

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

        if guidance == 0.0:
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
        
        pos, neg = add_style_to_subject(style, pos, neg)  # 风格

        if pos is not None and pos!= "":
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive")

        if neg is not None and neg!= "":
            negative, = CLIPTextEncode().encode(clip, neg)
        else:
            negative = context.get("negative")
        
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        context = new_context(context, model=model , latent=latent , clip=clip, vae=vae, positive=positive, negative=negative, images=image,steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,guidance=guidance, pos=pos, neg=neg, width=width, height=height, batch=batch_size )
        
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
                "diff_difusion": ("BOOLEAN", {"default": True}), 
                "smoothness":("INT", {"default": 1,  "min":0, "max": 150, "step": 1,"display": "slider"}),
                "ratio_selected": (['None'] + s.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, })
                
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","LATENT","MASK"  )
    RETURN_NAMES = ("context","latent","mask"  )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool"

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )


    def process(self, noise_mask, ratio_selected, smoothness=1, batch_size=1, context=None, latent=None, pixels=None, mask=None, diff_difusion=True):

        model = context.get("model")
        if diff_difusion:
            model = DifferentialDiffusion().apply(model)[0]

        if latent is not None and pixels is not None:
            raise ValueError("Only one of 'latent', 'pixels' should be provided.")
        if latent is not None:
            latent = latentrepeat(latent, batch_size)[0]
            context = new_context(context, model=model,latent=latent)
            return (context, latent, None)
        if latent is None and pixels is None and ratio_selected == "None":
            latent = context.get("latent", None)
            latent = latentrepeat(latent, batch_size)[0]
            context = new_context(context, model=model,latent=latent)
            return (context, latent, None)

        vae = context.get("vae")
        positive = context.get("positive", None)
        negative = context.get("negative", None)


        if ratio_selected != "None":
           latent = self.generate(ratio_selected, batch_size)[0]

        if pixels is not None:
            if mask is not None:
                if torch.all(mask == 0):
                    latent = VAEEncode().encode(vae, pixels)[0]
                else:
                    mask = tensor2pil(mask)
                    if not isinstance(mask, Image.Image):
                        raise TypeError("mask is not a valid PIL Image object")
                    feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
                    mask = pil2tensor(feathered_image)
                    
                    positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask)
            else:
                latent = VAEEncode().encode(vae, pixels)[0]
            latent = latentrepeat(latent, batch_size)[0]
        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent)

        return (context, latent, mask)




class sum_create_chx:
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
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "model": ("MODEL",),
                "clip": ("CLIP",),  
                "over_vae": ("VAE",),
                "over_positive": ("CONDITIONING",),
                "over_negative": ("CONDITIONING",),
                "over_latent": ("LATENT",),
                "lora_stack": ("LORASTACK",),
                "data":(ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "ANY_TYPE")
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent", "vae", "clip", "data")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"

    def process_settings(self, 
                        width, height, batch, steps, cfg, sampler, scheduler, data=None, guidance=3.5, lora_stack=None,over_latent=None,
                        vae=None, over_vae=None, clip=None, model=None, over_positive=None, over_negative=None, pos="default", neg="default"):

        # 分辨率修正
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([1, 4, height // 8, width // 8])
        if latent.shape[1] != 16:
            latent = latent.repeat(1, 16 // 4, 1, 1)

        if over_latent is not None:
            latent = over_latent
        # 处理VAE
        if over_vae is not None:
            vae = over_vae
        elif over_vae is None and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        # 初始化条件为None
        positive = None
        negative = None
        
        # 处理LoRA和文本编码
        if clip is not None:
            # 如果提供了clip，可以处理文本条件
            # 只有当model和clip都提供时，才应用LoRA
            if model is not None and lora_stack is not None:
                model, clip = apply_lora_stack(model, clip, lora_stack)
                
            # 处理条件
            positive, = CLIPTextEncode().encode(clip, pos)
            negative, = CLIPTextEncode().encode(clip, neg)



        # 覆盖条件（如果提供）
        if over_positive:
            positive = over_positive
            if negative is None:
                negative = condi_zero_out(over_positive)[0]
                
        if over_negative:
            negative = over_negative


        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        # 确保latent格式正确
        latent_dict = {"samples": latent}

        # 创建上下文
        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": latent_dict,  # 使用正确格式的latent
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "clip1": None,
            "clip2": None,
            "clip3": None,
            "clip4": None,
            "unet_name": None,
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
            "data": data,
        }

        return (context, model, positive, negative, latent_dict, vae, clip, data)  # 返回正确格式的latent

#endregion---------加载器-----------------------------------------------------------------------------------#


#region-----------采样器---------------------------------------------------------------------------------------#

class basic_Ksampler_full:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                "steps": ("INT", {"default": -1, "min": -1, "max": 10000,"tooltip": "  -1  == None"}),
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
    CATEGORY = "Apt_Preset/chx_ksample"


    def sample(self,  seed, denoise, context=None, clip=None, model=None,vae=None, positive=None, negative=None, latent=None,steps=None, cfg=None, sampler=None, scheduler=None, image=None, prompt=None, image_output=None, extra_pnginfo=None, ):


        if steps == -1:
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
        if latent is None:
           latent = context.get("latent")
        if image is not None:
           latent = encode(vae, image)[0]

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
    CATEGORY = "Apt_Preset/chx_ksample"


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

        if latent is None:
           latent = context.get("latent")
        if image is not None:
           latent = encode(vae, image)[0]

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
    CATEGORY = "Apt_Preset/chx_ksample"


    def run(self,context, seed, denoise, image=None,  prompt=None, image_output=None, extra_pnginfo=None,):
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
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL","CONDITIONING","CONDITIONING","LATENT", "VAE", )
    RETURN_NAMES = ("context","image", "model","positive","negative","latent", "vae",  )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"

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
            latent = encode(vae, image)[0]
        
        out= SamplerCustomAdvanced().sample( noise, guider, sampler, sigmas, latent)
        latent= out[0]
        
        if image_output == "None":
            context = new_context(context, images=None, latent=latent, model=model, positive=positive, negative=negative,  )
            return(context, model, positive, negative, latent, vae, None, ) 
            
        output_image = VAEDecode().decode(vae, latent)[0]  
        context = new_context(context, images=output_image, latent=latent, model=model, positive=positive, negative=negative,  )   
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context,output_image, model, positive, negative, latent, vae, )}
            
        return {"ui": {"images": results},
                "result": (context,output_image, model, positive, negative, latent, vae,)}


class basic_Ksampler_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 10000,"tooltip": "  0  == None"}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 1000, "min": 0, "max": 10000}),
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
    CATEGORY = "Apt_Preset/chx_ksample"

    def sample(self, context, add_noise, steps, noise_seed, start_at_step,end_at_step, return_with_leftover_noise, denoise=1.0, 
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

        """guidance = context.get("guidance",None)
        if guidance is None:
            guidance = 3.5  # 默认值
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})"""


        latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, 
                                positive, negative, latent, denoise=denoise, 
                                disable_noise=disable_noise, 
                                start_step=start_at_step, 
                                last_step=end_at_step,
                                force_full_denoise=force_full_denoise)[0]      
        
        
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
    CATEGORY = "Apt_Preset/chx_ksample"

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
    CATEGORY = "Apt_Preset/chx_ksample"


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
    CATEGORY = "Apt_Preset/chx_ksample"

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
            latent = encode(vae, image)[0]

        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]

        if image_output == "None":
            context = new_context(context, latent=latent, images=None, )
            return(context, None)


        output_image = decode(vae, latent)[0]
        
        
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
            "required": { "context": ("RUN_CONTEXT",),},
            "optional": {
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





class pre_controlnet_union:
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
                "type1": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "type2": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength2": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "type3": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","CONDITIONING", )
    RETURN_NAMES = ("context", "positive", "negative",  )
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, strength1, strength2, strength3, 
                       start_percent1=0.0, end_percent1=1.0,
                       start_percent2=0.0, end_percent2=1.0,
                       start_percent3=0.0, end_percent3=1.0,
                       context=None, image1=None, image2=None, image3=None,
                       controlNet=None, type1=None, type2=None, type3=None,
                       extra_concat=[]):

        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)
        
        if controlNet == "None":
            return (context, positive, negative)
            
        control_net = ControlNetLoader().load_controlnet(controlNet)[0]

        # 处理第一个ControlNet
        if type1 != "None" and strength1 != 0 and image1 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type(control_net, type1)[0]
            out =  ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image1, 
                                  strength1, start_percent1, end_percent1, 
                                  vae, extra_concat)
            positive, negative = out[0], out[1]

        # 处理第二个ControlNet
        if type2 != "None" and strength2 != 0 and image2 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type(control_net, type2)[0]
            out =  ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image2, 
                                  strength2, start_percent2, end_percent2, 
                                  vae, extra_concat)
            positive, negative = out[0], out[1]

        # 处理第三个ControlNet
        if type3 != "None" and strength3 != 0 and image3 is not None:
            control_net = SetUnionControlNetType().set_controlnet_type(control_net, type3)[0]
            out =  ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image3, 
                                  strength3, start_percent3, end_percent3, 
                                  vae, extra_concat)
            positive, negative = out[0], out[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative)






class pre_mul_Mulcondi:
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
                "background": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "background is sea"}),
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True,"default": "Poor quality" }),
            }
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("context","positive", "negative",)

    FUNCTION = "Mutil_Clip"
    CATEGORY = "Apt_Preset/chx_tool"

    def Mutil_Clip (self, pos1, pos2, pos3, pos4, pos5, background, neg,  mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_5_strength, mask_1=None, mask_2=None, mask_3=None, mask_4=None, mask_5=None, context=None):
        set_cond_area = "default"
        clip = context.get("clip")
        positive_1, = CLIPTextEncode().encode(clip, pos1)
        positive_2, = CLIPTextEncode().encode(clip, pos2)
        positive_3, = CLIPTextEncode().encode(clip, pos3)
        positive_4, = CLIPTextEncode().encode(clip, pos4)
        positive_5, = CLIPTextEncode().encode(clip, pos5)
        negative, = CLIPTextEncode().encode(clip, neg)

        c = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        valid_masks = []
        
        # 处理遮罩维度
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

        # 应用各个遮罩并收集有效的遮罩
        if mask_1 is not None:
            for t in positive_1:
                append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
            valid_masks.append(mask_1)
        if mask_2 is not None:
            for t in positive_2:
                append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
            valid_masks.append(mask_2)
        if mask_3 is not None:
            for t in positive_3:
                append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
            valid_masks.append(mask_3)
        if mask_4 is not None:
            for t in positive_4:
                append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
            valid_masks.append(mask_4)
        if mask_5 is not None:
            for t in positive_5:
                append_helper(t, mask_5, c, set_area_to_bounds, mask_5_strength)
            valid_masks.append(mask_5)

        # 计算背景遮罩
        if valid_masks:
            total_mask = sum(valid_masks)
            # 确保总遮罩不超过1
            total_mask = torch.clamp(total_mask, 0, 1)
            mask_6 = 1 - total_mask
        else:
            mask_6 = torch.ones_like(mask_1) if mask_1 is not None else None

        # 应用背景条件
        if mask_6 is not None:
            background_cond, = CLIPTextEncode().encode(clip, background)
            for t in background_cond:
                append_helper(t, mask_6, c, set_area_to_bounds, 1)

        context = new_context(context, positive=c, negative=negative, clip=clip)

        return (context, c, negative)








#endregion-----------tool--------------------------------------------------------------------------------------#--


#region-----------风格组--------------------------------------------------------------------------------------#--



class chx_YC_LG_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            "context": ("RUN_CONTEXT",),
            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),

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
            "image": ("IMAGE",),
            "mask": ("MASK", ), 
            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
        
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/chx_IPA"

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
    
    def apply_stylemodel(self, style_model, clip_vision,
                        patch_res=16, style_strength=1.0, prompt_strength=1.0, 
                        noise_level=0.0, crop="none", sharpen=0.0, guidance=30,
                        blend_mode="lerp", image=None,  mask=None, context=None):
        
        
        conditioning = context.get("positive", None)  
        if image is None:
            return (context,positive,)

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
    CATEGORY = "Apt_Preset/chx_IPA"

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
    CATEGORY = "Apt_Preset/chx_IPA"

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



class pre_sample_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == no change"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == no change"}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),  
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}), 
                
                
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_tool"

    def sample(self, context, steps, cfg, sampler, scheduler):
        
        if cfg == 0.0:
            cfg = context.get("cfg")
        if steps == 0:
            steps = context.get("steps")
        sampler = context.get("sampler","euler")
        scheduler = context.get("scheduler","normal")
        
        context = new_context(context, steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler)
        return (context, )

#------------------


    

class pre_Kontext_mul:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE",),
                "mask": ("MASK", ),
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos4": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos5": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "mask1": ("MASK", ),
                "mask2": ("MASK", ),
                "mask3": ("MASK", ),
                "mask4": ("MASK", ),
                "mask5": ("MASK", ),
                "prompt_weight1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 修改范围为 0~1，默认 0.5
                "prompt_weight2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight5": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                #"set_cond_area": (["default", "mask bounds"],),
            }
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","positive","latent")

    FUNCTION = "Mutil_Clip"
    CATEGORY = "Apt_Preset/chx_tool"

    def Mutil_Clip(self, pos1, pos2, pos3, pos4, pos5, image, mask,  prompt_weight1, prompt_weight2, prompt_weight3, prompt_weight4,prompt_weight5,
                    mask1=None, mask2=None, mask3=None, mask4=None, mask5=None, context=None):
        
        set_cond_area = "default" 
        if mask is not None and image is not None:
            vae = context.get("vae", None)
            latent = encode(vae, image)[0]
            # 确保 latent 是张量
            if isinstance(latent, dict):
                latent_tensor = latent["samples"]
            else:
                latent_tensor = latent
            result = set_latent_mask2(latent_tensor, mask)
            Flatent = result  # 如果 Flatent 后续只用于 new_context，则保留整个字典
        else:
            raise Exception("pls input image and mask")

        clip = context.get("clip")

        positive_1, = CLIPTextEncode().encode(clip, pos1)
        positive_2, = CLIPTextEncode().encode(clip, pos2)
        positive_3, = CLIPTextEncode().encode(clip, pos3)
        positive_4, = CLIPTextEncode().encode(clip, pos4)
        positive_5, = CLIPTextEncode().encode(clip, pos5)

        c = []
        set_area_to_bounds = False
        if set_cond_area!= "default":
            set_area_to_bounds = True


        # 处理 mask 维度
        if mask1 is not None and len(mask1.shape) < 3:
            mask1 = mask1.unsqueeze(0)
        if mask2 is not None and len(mask2.shape) < 3:
            mask2 = mask2.unsqueeze(0)
        if mask3 is not None and len(mask3.shape) < 3:
            mask3 = mask3.unsqueeze(0)
        if mask4 is not None and len(mask4.shape) < 3:
            mask4 = mask4.unsqueeze(0)
        if mask5 is not None and len(mask5.shape) < 3:
            mask5 = mask5.unsqueeze(0)

        # 添加条件权重
        if mask1 is not None:
            for t in positive_1:
                append_helper(t, mask1, c, set_area_to_bounds, 1)
        if mask2 is not None:
            for t in positive_2:
                append_helper(t, mask2, c, set_area_to_bounds, 1)
        if mask3 is not None:
            for t in positive_3:
                append_helper(t, mask3, c, set_area_to_bounds, 1)
        if mask4 is not None:
            for t in positive_4:
                append_helper(t, mask4, c, set_area_to_bounds, 1)
        if mask5 is not None:
            for t in positive_5:
                append_helper(t, mask5, c, set_area_to_bounds, 1)
        
        b = c
        # 创建一个原始 latent 的副本，避免重复修改
        original_latent = latent_tensor  # 使用确保的张量

        if mask1 is not None:
            influence = 8 * prompt_weight1 * (prompt_weight1 - 1) - 6 * prompt_weight1 + 6
            result = set_latent_mask2(original_latent, mask1)
            masked_latent = result["samples"]  # 提取 samples 部分进行计算
            latent_samples = masked_latent * influence
            b1 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b1
        if mask2 is not None:
            influence = 8 * prompt_weight2 * (prompt_weight2 - 1) - 6 * prompt_weight2 + 6
            result = set_latent_mask2(original_latent, mask2)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b2 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b2
        if mask3 is not None:
            influence = 8 * prompt_weight3 * (prompt_weight3 - 1) - 6 * prompt_weight3 + 6
            result = set_latent_mask2(original_latent, mask3)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b3 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b3
        if mask4 is not None:
            influence = 8 * prompt_weight4 * (prompt_weight4 - 1) - 6 * prompt_weight4 + 6
            result = set_latent_mask2(original_latent, mask4)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b4 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b4

        if mask5 is not None:
            influence = 8 * prompt_weight5 * (prompt_weight5 - 1) - 6 * prompt_weight5 + 6
            result = set_latent_mask2(original_latent, mask5)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b5 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b5  

        # 返回张量而不是字典
        context = new_context(context, positive=b, latent=Flatent)
        return (context, b, Flatent)



class pre_qwen_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "controlnet1": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
                "controlnet2": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength2": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                
                "image3": ("IMAGE",),
                "mask3": ("MASK",),
                "controlnet3": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                
            },

        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", )
    RETURN_NAMES = ("context","model", )
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, 
                        strength1, 
                        strength2,
                        strength3,  
                        context=None, 
                        controlnet1=None, controlnet2=None, controlnet3=None,
                        image1=None, image2=None, image3=None, vae=None,mask1=None, mask2=None, mask3=None):



        vae = context.get("vae", None)
        model = context.get("model", None)

        if controlnet1 != "None" and image1 is not None:
            cn1=ModelPatchLoader().load_model_patch(controlnet1)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn1, vae, image1, strength1, mask1)[0]


        if controlnet2 != "None" and image2 is not None:
            cn2=ModelPatchLoader().load_model_patch(controlnet2)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn2, vae, image2, strength2, mask2)[0]


        if controlnet3 != "None" and image3 is not None:
            cn3=ModelPatchLoader().load_model_patch(controlnet3)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn3, vae, image3, strength3, mask3)[0]


        context = new_context(context, model=model)
        return (context, model)







class pre_guide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "fluxguide"
    CATEGORY = "Apt_Preset/chx_tool"

    def fluxguide(self,context, guidance, ):  

        positive = context.get("positive",2.5)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        
        context = new_context(context, positive=positive, guidance=guidance )
        return (context, )




from comfy_extras.nodes_flux import FluxKontextMultiReferenceLatentMethod
from comfy_extras.nodes_model_patch import ModelPatchLoader, USOStyleReference
from nodes import CLIPVisionLoader, CLIPVisionEncode


class pre_USO:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "context": ("RUN_CONTEXT",),

            "image": ("IMAGE", ),
            "reference_latents_method": (("uxo/uno","offset", "index" ), ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "smoothness":("INT", {"default": 0,  "min":0, "max": 10, "step": 0.1,}),

            "crop": (["center", "none"],),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"),{"default": "sigclip_vision_patch14_384.safetensors"} ),
            "model_patch": (folder_paths.get_filename_list("model_patches"), {"default": "uso-flux1-projector-v1.safetensors"}),


                    },

        "optional": {
            "mask": ("MASK",),
            "ref_image": ("IMAGE", ),                 
                    }
               }


    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","model","positive","latent")
    FUNCTION = "append"
    CATEGORY = "Apt_Preset/chx_tool"


    def append(self,context, guidance, crop, clip_vision=None, model_patch=None, image=None, mask=None,smoothness=0, ref_image=None, reference_latents_method="uxo/uno"):
        vae = context.get("vae", None)
        conditioning = context.get("positive", None)
        negative = context.get("negative", None)



        if image is None:
           raise Exception("Please provide an input image.")


        latent = encode(vae, image)[0]

        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        conditioning = FluxKontextMultiReferenceLatentMethod().append(conditioning, reference_latents_method)[0]
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})

        positive=conditioning
        

        if mask is not None:
            mask =smoothness_mask(mask, smoothness)
            positive, negative, latent = InpaintModelConditioning().encode(positive, negative, image, vae, mask, True)
        else:
            latent = encode(vae, image)[0]


        if ref_image is not None:
            model=context.get("model", None)
            model_patch= ModelPatchLoader().load_model_patch(model_patch)[0]
            clip_vision = CLIPVisionLoader().load_clip(clip_vision)[0]
            clip_vision_out = CLIPVisionEncode().encode(clip_vision, ref_image, crop)[0]

            model= USOStyleReference().apply_patch(model, model_patch, clip_vision_out)[0]
        else:
            model=context.get("model", None)

        context = new_context(context, positive=positive, latent=latent, model=model )

        return (context, model, positive, latent )



























































