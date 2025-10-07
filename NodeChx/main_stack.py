
#region-------------------------------import-----------------------


import folder_paths
from random import random, uniform
import torch
import numpy as np
from nodes import common_ksampler, CLIPTextEncode, ControlNetApplyAdvanced, VAEDecode, VAEEncode, InpaintModelConditioning, ControlNetLoader
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import node_helpers
from PIL import Image, ImageFilter
from comfy_extras.nodes_controlnet import SetUnionControlNetType
from dataclasses import dataclass
#import torch.nn.functional as F
import numpy as np

import logging
from .IPAdapterPlus import ipadapter_execute, IPAdapterUnifiedLoader
from .AdvancedCN import *
from .main_nodes import Data_chx_Merge


from ..main_unit import *
from ..office_unit import DifferentialDiffusion


#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False



WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']


#endregion-----------------------------import----------------------------


#region---------------------收纳----------------------------


#region--------latent总控-----------------

class Stack_latent:
    ratio_sizes, ratio_dict = read_ratios()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "latent": ("LATENT",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
                "noise_mask": ("BOOLEAN", {"default": True}),
                "diff_difusion": ("BOOLEAN", {"default": True}),  # 新增参数
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1, "display": "slider"}),
                "ratio_selected": (['None'] + cls.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300})
            }
        }

    RETURN_TYPES = ("LATENT_STACK",)
    RETURN_NAMES = ("latent_stack",)
    FUNCTION = "stack_latent"
    CATEGORY = "Apt_Preset/stack/stack"

    def stack_latent(self, latent=None, pixels=None, mask=None, noise_mask=True, diff_difusion=True,  # 新增参数
                    smoothness=1, ratio_selected="None", batch_size=1):
        # 将diff_difusion加入存储的信息中
        latent_info = (latent, pixels, mask, noise_mask, diff_difusion, smoothness, ratio_selected, batch_size)
        latent_stack = [latent_info]
        return (latent_stack,)


class Apply_latent:
    ratio_sizes, ratio_dict = read_ratios()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE",),
                "latent_stack": ("LATENT_STACK",),
            }
        }

    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("model","positive","negative","latent",)
    FUNCTION = "apply_latent_stack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def apply_latent_stack(self, model, positive, negative, vae, latent_stack):
        default_width = 512
        default_height = 512
        batch_size = 1

        for latent_info in latent_stack:
            # 从存储的信息中解包diff_difusion参数
            latent, pixels, mask, noise_mask, diff_difusion, smoothness, ratio_selected, batch_size = latent_info

            if ratio_selected == "None" and latent is None and pixels is None:
                raise ValueError("pls input latent, or pixels, or ratio_selected.")

            if ratio_selected != "None":
                width = self.ratio_dict[ratio_selected]["width"]
                height = self.ratio_dict[ratio_selected]["height"]
                latent = {"samples": torch.zeros([batch_size, 4, height // 8, width // 8])}
                # 应用diff_difusion
                if diff_difusion:
                    model = DifferentialDiffusion().apply(model)[0]
                return model, positive, negative, latent

            if latent is None :
                latent = {"samples": torch.zeros([batch_size, 4, default_height // 8, default_width // 8])}
                
            if pixels is not None:
                latent = VAEEncode().encode(vae, pixels)[0]

            if pixels is None and mask is None:
                raise TypeError("No input pixels")
            
            if mask is not None:
                mask = tensor2pil(mask)
                feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
                mask = pil2tensor(feathered_image)
                positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask)
            
            # 应用diff_difusion
            if diff_difusion:
                model = DifferentialDiffusion().apply(model)[0]

        latent = latentrepeat(latent, batch_size)[0]
        return model, positive, negative, latent

#endregion--------latent总控-----------------


#region--------mold串联-----------------


class Stack_LoRA:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "lora_name_1": (loras,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name_2": (loras,),
                "weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_name_3": (loras,),
                "weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "lora_stack": ("LORASTACK",)
            },
        }

    RETURN_TYPES = ("LORASTACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "lora_stacker"
    CATEGORY = "Apt_Preset/stack/stack"


    def lora_stacker(self, lora_name_1, weight_1, lora_name_2, weight_2, lora_name_3, weight_3, lora_stack=None):
        """
        将多个 LoRA 配置添加到堆中。
        """
        lora_list = []

        # 如果传入了已有的 lora_stack，将其内容合并到 lora_list 中
        if lora_stack is not None:
            lora_list.extend([lora for lora in lora_stack if lora[0] != "None"])

        # 如果 LoRA 配置有效，则将其添加到列表中
        if lora_name_1 != "None":
            lora_list.append((lora_name_1, weight_1))
        if lora_name_2 != "None":
            lora_list.append((lora_name_2, weight_2))
        if lora_name_3 != "None":
            lora_list.append((lora_name_3, weight_3))

        return (lora_list,)


class Apply_LoRAStack:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORASTACK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("MODEL", "CLIP",)
    FUNCTION = "apply_lora_stack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def apply_lora_stack(self, model, clip, lora_stack=None):
        if not lora_stack:
            return (model, clip,)

        model_lora = model
        clip_lora = clip

        for tup in lora_stack:
            lora_name, weight = tup  
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, weight, weight )

        return (model_lora, clip_lora,)




class Stack_IPA:

    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "preset": ([ 'STANDARD (medium strength)','LIGHT - SD1.5 only (low strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image": ("IMAGE",),
                "attn_mask": ("MASK",),
                "ipa_stack": ("IPA_STACK",),
                #"image_negative": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IPA_STACK",)
    RETURN_NAMES = ("ipa_stack",)
    FUNCTION = "ipa_stack"
    CATEGORY = "Apt_Preset/stack/stack"

    def ipa_stack(self,   preset, weight, weight_type, combine_embeds, start_at, end_at, embeds_scaling,image=None, attn_mask=None, image_negative=None, ipa_stack=None):
        
        if image is None:
            return (None,)
        
        # 初始化ipa_list
        ipa_list = []

        # 如果传入了ipa_stack，将其中的内容添加到ipa_list中
        if ipa_stack is not None:
            ipa_list.extend([ipa for ipa in ipa_stack if ipa[0] != "None"])

        # 将当前IPA的相关信息打包成一个元组，并添加到ipa_list中
        ipa_info = (
            image,
            preset,
            weight,
            weight_type,
            combine_embeds,
            start_at,
            end_at,
            embeds_scaling,
            attn_mask,
            image_negative,
        )
        ipa_list.append(ipa_info)

        return (ipa_list,)


class Apply_IPA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipa_stack": ("IPA_STACK",),
            }
        }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "apply_ipa_stack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def apply_ipa_stack(self, model, ipa_stack):

        if not ipa_stack:
            raise ValueError("ipa_stack 不能为空")

        # 初始化变量
        image0 = None
        mask0 = None
        work_model = model.clone()

        # 遍历 ipa_stack 中的每个 IPA 配置
        for ipa_info in ipa_stack:
            (
                image,
                preset,
                weight,
                weight_type,
                combine_embeds,
                start_at,
                end_at,
                embeds_scaling,
                attn_mask,
                image_negative,
            ) = ipa_info

            # 记录第一个 image 和 mask
            if image0 is None:
                image0 = image
            if mask0 is None:
                mask0 = attn_mask

            # 加载 IPAdapter 模型
            model, ipadapter = IPAdapterUnifiedLoader().load_models(
                work_model, preset, lora_strength=0.0, provider="CPU", ipadapter=None
            )

            if 'ipadapter' in ipadapter:
                ipadapter_model = ipadapter['ipadapter']['model']
                clip_vision = ipadapter['clipvision']['model']
            else:
                ipadapter_model = ipadapter


            ipa_args = {
                "image": image,
                "image_negative": image_negative,
                "weight": weight,
                "weight_type": weight_type,
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": attn_mask,
                "embeds_scaling": embeds_scaling,
                "insightface": None,  # 如果需要 insightface，可以从 ipa_stack 中传递
                "layer_weights": None,  # 如果需要 layer_weights，可以从 ipa_stack 中传递
                "encode_batch_size": 0,  # 默认值
                "style_boost": None,  # 如果需要 style_boost，可以从 ipa_stack 中传递
                "composition_boost": None,  # 如果需要 composition_boost，可以从 ipa_stack 中传递
                "enhance_tiles": 1,  # 默认值
                "enhance_ratio": 1.0,  # 默认值
                "weight_kolors": 1.0,  # 默认值
            }

            # 应用 IPA 配置
            model, _ = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        return (model,)       #model在下面运行正确，但是这里会报错，要统一元祖或统一模型对象


class AD_sch_IPA:

    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": ([ 'STANDARD (medium strength)','LIGHT - SD1.5 only (low strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                #"weight_type": (["none", "style", "content"], ),
                #"combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                #"start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                #"end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                "invert": ("BOOLEAN", {"default": False}),
                "frames": ("INT", {"default": 16,"min": 2, "max": 255, "step": 1}),
                #"width": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                #"height": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "ipa_stack": ("IPA_STACK",),

            }
        }

    RETURN_TYPES = ("IPA_STACK",)
    RETURN_NAMES = ("ipa_stack",)
    FUNCTION = "ipa_stack"
    CATEGORY = "Apt_Preset/AD"

    def createfademask(self, frames, width, height, invert, points_string, interpolation):
        
        def ease_in(t):
            return t * t
        
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the input string into a list of tuples
        points = []
        points_string = points_string.rstrip(',\n')
        for point_str in points_string.split(','):
            frame_str, color_str = point_str.split(':')
            frame = int(frame_str.strip())
            color = float(color_str.strip()[1:-1])  # Remove parentheses around color
            points.append((frame, color))

        # Check if the last frame is already in the points
        if len(points) == 0 or points[-1][0] != frames - 1:
            # If not, add it with the color of the last specified frame
            points.append((frames - 1, points[-1][1] if points else 0))

        # Sort the points by frame number
        points.sort(key=lambda x: x[0])

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        # Index of the next point to interpolate towards
        next_point = 1

        for i in range(batch_size):
            while next_point < len(points) and i > points[next_point][0]:
                next_point += 1

            # Interpolate between the previous point and the next point
            prev_point = next_point - 1
            t = (i - points[prev_point][0]) / (points[next_point][0] - points[prev_point][0])
            if interpolation == "ease_in":
                t = ease_in(t)
            elif interpolation == "ease_out":
                t = ease_out(t)
            elif interpolation == "ease_in_out":
                t = ease_in_out(t)
            elif interpolation == "linear":
                pass  # No need to modify `t` for linear interpolation

            color = points[prev_point][1] - t * (points[prev_point][1] - points[next_point][1])
            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return 1.0 - torch.cat(out, dim=0)
        return torch.cat(out, dim=0)

    def ipa_stack(self, image, preset, weight, embeds_scaling, points_string, invert, frames, interpolation, ipa_stack=None):
        
        start_at=0
        end_at=1
        weight_type = "style"
        combine_embeds = "add"
        
        
        attn_mask = self.createfademask(frames, 512, 512, invert, points_string, interpolation)
        
        image_negative = None
        ipa_list = []
        if ipa_stack is not None:
            ipa_list.extend([ipa for ipa in ipa_stack if ipa[0] != "None"])

        ipa_info = (
            image,
            preset,
            weight,
            weight_type,
            combine_embeds,
            start_at,
            end_at,
            embeds_scaling,
            attn_mask,
            image_negative,
        )
        ipa_list.append(ipa_info)

        return (ipa_list,)


#endregion--------mold串联-----------------





#region--------条件----分区串联-----------------




#-----------------



class XXXpre_condi_Mul_strength:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
                "mode": (["combine", "average", "concat"], ),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})  
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"

    DESCRIPTION = "strength1-strength4调节各条件强度：\n" \
    "- combine：合并，混合特征（如红色+圆形→融合为红色圆形，无法拆分。文本串联）\n" \
    "- average：平衡融合（归一化权重，避免某特征过强）\n" \
    "- concat：连接，保留独立特征（如红色+圆形→同时保留红色和圆形通道，可分别处理）"

    def merge(self, conditioning_1, conditioning_2, mode, 
              conditioning_3=None, conditioning_4=None,
              strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0):
        
        conditionings = [conditioning_1, conditioning_2]
        if conditioning_3 is not None:
            conditionings.append(conditioning_3)
        if conditioning_4 is not None:
            conditionings.append(conditioning_4)
            
        strengths = [strength1, strength2, strength3, strength4]
        valid_strengths = strengths[:len(conditionings)]
        
        if mode == "combine":
            result = []
            for i, conditioning in enumerate(conditionings):
                for cond_item in conditioning:
                    cond_tensor = cond_item[0]
                    cond_dict = cond_item[1].copy()
                    cond_tensor = torch.mul(cond_tensor, valid_strengths[i])
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[i])
                    result.append([cond_tensor, cond_dict])
            return (result, )
        
        elif mode == "average":
            if len(conditionings) == 1:
                result = []
                for cond_item in conditionings[0]:
                    cond_tensor = torch.mul(cond_item[0], valid_strengths[0])
                    cond_dict = cond_item[1].copy()
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[0])
                    result.append([cond_tensor, cond_dict])
                return (result,)
            
            if len(conditionings) == 2:
                out = []
                conditioning_to = conditionings[0]
                conditioning_from = conditionings[1]
                
                if len(conditioning_from) > 1:
                    logging.warning("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied.")
                cond_from = conditioning_from[0][0]
                pooled_output_from = conditioning_from[0][1].get("pooled_output", None)
                
                conditioning_to_strength = valid_strengths[0]
                
                for i in range(len(conditioning_to)):
                    t1 = conditioning_to[i][0]
                    pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
                    
                    t0 = cond_from[:,:t1.shape[1]]
                    if t0.shape[1] < t1.shape[1]:
                        t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]),
                                            dtype=t0.dtype, device=t0.device)], dim=1)
                    
                    tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
                    t_to = conditioning_to[i][1].copy()
                    
                    if pooled_output_from is not None and pooled_output_to is not None:
                        t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
                    elif pooled_output_from is not None:
                        t_to["pooled_output"] = pooled_output_from
                    
                    out.append([tw, t_to])
                
                return (out, )
            
            total_strength = sum(valid_strengths)
            if total_strength > 0:
                normalized_strengths = [s/total_strength for s in valid_strengths]
            else:
                normalized_strengths = [1.0/len(conditionings) for _ in valid_strengths]
            
            base_conditioning = conditionings[0]
            out = []
            
            for i in range(len(base_conditioning)):
                t_base = torch.mul(base_conditioning[i][0], normalized_strengths[0])
                t_result = base_conditioning[i][1].copy()
                
                pooled_output = None
                if "pooled_output" in t_result and t_result["pooled_output"] is not None:
                    pooled_output = torch.mul(t_result["pooled_output"], normalized_strengths[0])
                
                for j in range(1, len(conditionings)):
                    cond_tensor = conditionings[j][i][0]
                    if cond_tensor.shape[1] != t_base.shape[1]:
                        if cond_tensor.shape[1] > t_base.shape[1]:
                            cond_tensor = cond_tensor[:, :t_base.shape[1], :]
                        else:
                            pad_size = t_base.shape[1] - cond_tensor.shape[1]
                            padding = torch.zeros((cond_tensor.shape[0], pad_size, cond_tensor.shape[2]), 
                                                dtype=cond_tensor.dtype, device=cond_tensor.device)
                            cond_tensor = torch.cat([cond_tensor, padding], dim=1)
                    
                    t_base = t_base + torch.mul(cond_tensor, normalized_strengths[j])
                    
                    if pooled_output is not None and "pooled_output" in conditionings[j][i][1] and conditionings[j][i][1]["pooled_output"] is not None:
                        pooled2 = conditionings[j][i][1]["pooled_output"]
                        if pooled2.shape[0] != pooled_output.shape[0]:
                            if pooled2.shape[0] > pooled_output.shape[0]:
                                pooled2 = pooled2[:pooled_output.shape[0]]
                            else:
                                pad_size = pooled_output.shape[0] - pooled2.shape[0]
                                padding = torch.zeros((pad_size, pooled2.shape[1]), 
                                                    dtype=pooled2.dtype, device=pooled2.device)
                                pooled2 = torch.cat([pooled2, padding], dim=0)
                        pooled_output = pooled_output + torch.mul(pooled2, normalized_strengths[j])
                
                if pooled_output is not None:
                    t_result["pooled_output"] = pooled_output
                
                out.append([t_base, t_result])
                
            return (out, )
        
        elif mode == "concat":
            if len(conditionings) == 1:
                result = []
                for cond_item in conditionings[0]:
                    cond_tensor = torch.mul(cond_item[0], valid_strengths[0])
                    cond_dict = cond_item[1].copy()
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[0])
                    result.append([cond_tensor, cond_dict])
                return (result,)
            
            base_conditioning = conditionings[0]
            out = []
            
            for i in range(len(base_conditioning)):
                t_result = torch.mul(base_conditioning[i][0], valid_strengths[0])
                t_dict = base_conditioning[i][1].copy()
                
                if "pooled_output" in t_dict and t_dict["pooled_output"] is not None:
                    t_dict["pooled_output"] = torch.mul(t_dict["pooled_output"], valid_strengths[0])
                
                for j in range(1, len(conditionings)):
                    cond_tensor = torch.mul(conditionings[j][i][0], valid_strengths[j])
                    t_result = torch.cat((t_result, cond_tensor), 1)
                    
                    if "pooled_output" in t_dict and t_dict["pooled_output"] is not None and \
                       "pooled_output" in conditionings[j][i][1] and conditionings[j][i][1]["pooled_output"] is not None:
                        pooled1 = t_dict["pooled_output"]
                        pooled2 = torch.mul(conditionings[j][i][1]["pooled_output"], valid_strengths[j])
                        t_dict["pooled_output"] = torch.cat((pooled1, pooled2), dim=0)
                
                out.append([t_result, t_dict])
            
            return (out, )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")






#---------------------







class Stack_condition_mode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
                "mode": (["combine", "average", "concat"], ),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})  
            }
        }
    
    RETURN_TYPES = ("CONDITION_MODE_STACK",)
    RETURN_NAMES = ("condi_stack",)
    FUNCTION = "pack"
    CATEGORY = "Apt_Preset/chx_tool"


    def pack(self, conditioning_1, conditioning_2, mode, 
             conditioning_3=None, conditioning_4=None,
             strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0):

        condition_mode_pack = conditioning_1, conditioning_2, mode,conditioning_3, conditioning_4,strength1, strength2, strength3, strength4
        
        return condition_mode_pack



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


class Stack_condi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos4": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos5": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),  # 新增pos5
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_5": ("MASK", ),  # 新增mask_5输入
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_5_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),  # 新增mask_5强度
                "background": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "background is sea"}),
                "background_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),  # 新增背景强度
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "Poor quality"}),
            }
        }
    
    RETURN_TYPES = ("STACK_CONDI",)
    RETURN_NAMES = ("condi_stack", )
    FUNCTION = "stack_condi"
    CATEGORY = "Apt_Preset/stack/stack"

    def stack_condi(self, pos1, pos2, pos3, pos4, pos5, background, background_strength, neg, 
                    mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_5_strength,
                    mask_1=None, mask_2=None, mask_3=None, mask_4=None, mask_5=None):
        condi_stack = list()
        set_cond_area ="default"
        
        # 打包逻辑：每组 pos、mask 和 mask_strength 是配套的
        def pack_group(pos, mask, mask_strength):
            if mask is None or mask_strength <= 0:  # 新增mask_strength检查
                return None
            return {
                "pos": pos,
                "mask": mask,
                "mask_strength": mask_strength,
            }
        
        valid_masks = []
        if mask_1 is not None:
            valid_masks.append(mask_1)
        if mask_2 is not None:
            valid_masks.append(mask_2)
        if mask_3 is not None:
            valid_masks.append(mask_3)
        if mask_4 is not None:
            valid_masks.append(mask_4)
        if mask_5 is not None:
            valid_masks.append(mask_5)

        # 计算背景遮罩，确保范围在0-1之间
        if valid_masks:
            total_mask = sum(valid_masks)
            # 确保总遮罩不超过1
            total_mask = torch.clamp(total_mask, 0, 1)
            mask_bg = 1 - total_mask
        else:
            # 如果没有有效遮罩，背景遮罩应该是全1
            mask_bg = None  # 注意：这里改为None，在Apply_condiStack中处理全1的情况
        
        # 打包每组信息
        group1 = pack_group(pos1, mask_1, mask_1_strength)
        group2 = pack_group(pos2, mask_2, mask_2_strength)
        group3 = pack_group(pos3, mask_3, mask_3_strength)
        group4 = pack_group(pos4, mask_4, mask_4_strength)
        group5 = pack_group(pos5, mask_5, mask_5_strength)
        group_bg = pack_group(background, mask_bg, background_strength)  # 使用背景强度参数
        
        # 将打包的组添加到 condi_stack
        if group1 is not None:
            condi_stack.append(group1)
        if group2 is not None:
            condi_stack.append(group2)
        if group3 is not None:
            condi_stack.append(group3)
        if group4 is not None:
            condi_stack.append(group4)
        if group5 is not None:
            condi_stack.append(group5)
        if group_bg is not None:  # 添加背景组
            condi_stack.append(group_bg)
        
        # 打包负面提示和 set_cond_area
        condi_stack.append({
            "neg": neg,
            "set_cond_area": set_cond_area,
        })
        
        return (condi_stack,)


class Apply_condiStack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "stack_condi": ("STACK_CONDI",),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "condiStack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def condiStack(self, clip, stack_condi):
        neg_data = stack_condi[-1]
        neg = neg_data["neg"]
        set_cond_area = neg_data["set_cond_area"]

        negative, = CLIPTextEncode().encode(clip, neg)
        positive = []
        set_area_to_bounds = (set_cond_area != "default")

        for group in stack_condi[:-1]:
            pos = group["pos"]
            mask = group["mask"]
            mask_strength = group["mask_strength"]

            encoded_pos, = CLIPTextEncode().encode(clip, pos)

            if mask is None:
                # 处理背景遮罩为None的情况（即全1遮罩）
                for t in encoded_pos:
                    # 创建一个全1的遮罩
                    full_mask = torch.ones_like(t[0][0]) if t is not None and len(t) > 0 and len(t[0]) > 0 else None
                    if full_mask is not None:
                        append_helper(t, full_mask, positive, set_area_to_bounds, mask_strength)
            else:
                if len(mask.shape) < 3:
                    mask = mask.unsqueeze(0)
                for t in encoded_pos:
                    append_helper(t, mask, positive, set_area_to_bounds, mask_strength)
                    
        return (positive, negative)

#endregion--------条件串联-----------------


#region--------条件---CN串联-----------------

class pre_inpaint_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "control_net": (["None"] + folder_paths.get_filename_list("controlnet"), {"default": None}),
                "mask_mode": (["Ailmama", "mask_black", "mask_white", "mask_gray"], {"default": "Ailmama"}),
                "image": ("IMAGE", ),  
                "mask": ("MASK", ),    
                "smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                        
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("context","positive", "negative","latent")
    FUNCTION = "apply_controlnet"
    CATEGORY = "Apt_Preset/chx_tool/controlnet"
    DESCRIPTION = """
    - mask_mode：针对遮罩区图片的预处理方式，mask_black处理成黑色块，mask_white处理成白色块，mask_gray处理成灰色块

    - 模式一：没有选择controlnet，采样内部重绘模式InpaintModelConditioning
    - （1）低噪重绘：适合所有模型，适合低噪重绘，细节修复，微改变
    - （2）高噪重绘：适合图像编辑模型，如qwenEdit,kontext

    - 模式二：选择controlnet，采用CN控制模式，controlnet inpainting的模型
    - SD重绘，用mask_black
    - XL重绘，用mask_white
    - flux重绘，用Alimama
    - qwen_image重绘，用Alimama
    """



    def apply_controlnet(self, context, control_net, image, strength, smoothness, start_percent, end_percent, mask_mode="mask_black", mask=None):


        vae = context.get("vae", None)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        latent = encode(vae, image)[0]
        mask =smoothness_mask(mask, smoothness)


        if control_net == "None":
            positive, negative, latent = InpaintModelConditioning().encode(positive, negative, image, vae, mask, True)
            context = new_context(context, positive=positive, negative=negative, latent=latent)
            return (context, positive, negative, latent)

        
        control_net = ControlNetLoader().load_controlnet(control_net)[0]
        latent = set_mask(latent, mask)

        # 处理预处理
        if mask_mode != "Ailmama":
            processed_image = self.InpaintPreprocessor(image, mask, mask_mode)[0]
        else:
            processed_image = image

        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, processed_image.shape[2], processed_image.shape[1], "bilinear", "center").round()
            processed_image = processed_image * mask_apply.movedim(1, -1).repeat(1, 1, 1, processed_image.shape[3])
            extra_concat = [mask]

        if strength == 0:
            return (context, positive, negative, latent)

        control_hint = processed_image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)

        context = new_context(context, positive=out[0], negative=out[1], latent=latent)
        return (context, out[0], out[1], latent)

    def InpaintPreprocessor(self, image, mask, mask_color="mask_black"):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
        mask = mask.movedim(1, -1).expand((-1, -1, -1, 3))
        image = image.clone()

        if mask_color == "mask_black":
            masked_pixel = -1.0  
        elif mask_color == "mask_white":
            masked_pixel = 1.0  
        elif mask_color == "mask_gray":
            masked_pixel = 0.0  
        else:
            masked_pixel = -1.0  # 默认黑色
        
        image[mask > 0.5] = masked_pixel
        return (image,)



class pack_inpaint: #隐藏
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "control_net": (["None"] + folder_paths.get_filename_list("controlnet"), {"default": None}),
                "mask_mode": (["Ailmama", "mask_black", "mask_white", "mask_gray"], {"default": "Ailmama"}),
                "image": ("IMAGE", ),  
                "mask": ("MASK", ),    
                "smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                        
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("positive", "negative","latent")
    FUNCTION = "apply_controlnet"
    CATEGORY = "Apt_Preset/chx_tool/controlnet"


    def apply_controlnet(self, vae, positive, negative, control_net, image, strength, smoothness, start_percent, end_percent, mask_mode="mask_black", mask=None):

        mask =smoothness_mask(mask, smoothness)

        latent = None

        if control_net == "None":
            positive, negative, latent = InpaintModelConditioning().encode(positive, negative, image, vae, mask, True)

            return ( positive, negative, latent)


        encoded_result = encode(vae, image)
        if encoded_result:
            latent = encoded_result[0]

        control_net = ControlNetLoader().load_controlnet(control_net)[0]
        if latent is not None:
           latent = set_mask(latent, mask)

        # 处理预处理
        if mask_mode != "Ailmama":
            processed_image = self.InpaintPreprocessor(image, mask, mask_mode)[0]
        else:
            processed_image = image

        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, processed_image.shape[2], processed_image.shape[1], "bilinear", "center").round()
            processed_image = processed_image * mask_apply.movedim(1, -1).repeat(1, 1, 1, processed_image.shape[3])
            extra_concat = [mask]

        if strength == 0:
            return ( positive, negative, latent)

        control_hint = processed_image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
            
        if latent is None:
            latent = {"samples": torch.zeros([1, 4, 512 // 8, 512 // 8])}

        return (out[0], out[1], latent)

    def InpaintPreprocessor(self, image, mask, mask_color="mask_black"):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
        mask = mask.movedim(1, -1).expand((-1, -1, -1, 3))
        image = image.clone()

        if mask_color == "mask_black":
            masked_pixel = -1.0  
        elif mask_color == "mask_white":
            masked_pixel = 1.0  
        elif mask_color == "mask_gray":
            masked_pixel = 0.0  
        else:
            masked_pixel = -1.0  # 默认黑色
        
        image[mask > 0.5] = masked_pixel
        return (image,)




class Stack_inpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            "optional": {
                "control_net": (["None"] + folder_paths.get_filename_list("controlnet"), {"default": None}),
                "mask_mode": (["Ailmama", "mask_black", "mask_white", "mask_gray"], {"default": "Ailmama"}),
                "image": ("IMAGE", ),  
                "mask": ("MASK", ),    
                "smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }

    RETURN_TYPES = ("INPAINT_STACK",)
    RETURN_NAMES = ("inpaint",)
    FUNCTION = "pack_inpaint_params"
    CATEGORY = "Apt_Preset/stack/stack"
    DESCRIPTION = """
    - mask_mode：针对遮罩区图片的预处理方式，mask_black处理成黑色块，mask_white处理成白色块，mask_gray处理成灰色块

    - 模式一：没有选择controlnet，采样内部重绘模式InpaintModelConditioning
    - （1）低噪重绘：适合所有模型，适合低噪重绘，细节修复，微改变
    - （2）高噪重绘：适合图像编辑模型，如qwenEdit,kontext

    - 模式二：选择controlnet，采用CN控制模式，controlnet inpainting的模型
    - SD重绘，用mask_black
    - XL重绘，用mask_white
    - flux重绘，用Alimama
    - qwen_image重绘，用Alimama
    """


    def pack_inpaint_params(self, control_net, mask_mode="Ailmama", image=None, mask=None, smoothness=0.0, strength=1.0, start_percent=0.0, end_percent=1.0):
        inpaint_sum_pack = ( control_net, mask_mode, image, mask,smoothness, strength, start_percent, end_percent)
        return (inpaint_sum_pack,)




class pre_controlnet_union:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "context": ("RUN_CONTEXT",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "dual_image": ("IMAGE",),

                "controlNet": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "type1": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "type2": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength2": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "dual_type_1": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "dual_type_2": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","CONDITIONING", )
    RETURN_NAMES = ("context", "positive", "negative",  )
    CATEGORY = "Apt_Preset/chx_tool/controlnet"
    FUNCTION = "load_controlnet"
    DESCRIPTION = """
    - dual_image: 双控图，常见的是线稿+深度, pose+深度
    - dual_type_1: 只选一个，也可以当单控图，
    - dual_type_2: 只选一个，也可以当单控图
    """



    def load_controlnet(self, strength1, strength2, strength3, 
                       start_percent1=0.0, end_percent1=1.0,
                       start_percent2=0.0, end_percent2=1.0,
                       start_percent3=0.0, end_percent3=1.0,
                       context=None, image1=None, image2=None, dual_image=None,
                       controlNet=None, type1=None, type2=None, dual_type_1=None,dual_type_2=None,
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
        if strength3 != 0 and dual_image is not None:
           if dual_type_1 == "None":
               control_net = SetUnionControlNetType().set_controlnet_type(control_net, dual_type_1)[0]
           if dual_type_2 != "None":
               control_net = SetUnionControlNetType().set_controlnet_type(control_net, dual_type_2)[0]
           
           out =  ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, dual_image, 
                                    strength3, start_percent3, end_percent3, 
                                    vae, extra_concat)
           
        positive, negative = out[0], out[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative)



class Stack_CN_union:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "image": ("IMAGE",),
                "controlNet": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "type1": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "type2": (["None"] + list(UNION_CONTROLNET_TYPES.keys()),),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "union_stack": ("UNION_STACK",),
            }
        }

    RETURN_TYPES = ("UNION_STACK", )
    RETURN_NAMES = ("union_stack", )
    CATEGORY = "Apt_Preset/stack/stack"
    FUNCTION = "load_controlnet"
    DESCRIPTION = """
    - 同时选type1和type2: 用双控图，常见的是线稿+深度, pose+深度
    - dual_type_1: 只选一个，单控图
    - dual_type_2: 只选一个，单控图
    """

    def load_controlnet(self,  
                        image="None",
                        controlNet="None",
                        type1="None",
                        type2="None",
                        strength=0.8,
                        start_percent=0.0,
                        end_percent=1.0,
                        union_stack=None,
                        extra_concat=[]):
        
        stack_list = []
        if union_stack is not None:
            stack_list.extend([item for item in union_stack if item[0] is not None])

        if controlNet != "None" and strength != 0 and image is not None:
            # 先加载control_net
            control_net = ControlNetLoader().load_controlnet(controlNet)[0]
            
            # 根据type1和type2设置控制网络类型
            if type1 != "None":
                control_net = SetUnionControlNetType().set_controlnet_type(control_net, type1)[0]  
            if type2 != "None":
                control_net = SetUnionControlNetType().set_controlnet_type(control_net, type2)[0]

            stack_list.append((control_net, image, strength, start_percent, end_percent))

        return (tuple(stack_list), )


class Apply_CN_union:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

            },
            "optional": {
                "vae": ("VAE",),
                "union_stack": ("UNION_STACK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING", )
    RETURN_NAMES = ("positive", "negative",  )
    CATEGORY = "Apt_Preset/stack/uppack"
    FUNCTION = "apply_union_stack"

    
    def apply_union_stack(self, positive, negative, vae=None, union_stack=None, extra_concat=[]):
        # 检查union_stack是否存在且不为空
        if union_stack is not None and len(union_stack) > 0:
            # 遍历栈中的每个控制网络，包含新增的start_percent和end_percent参数
            for control_net, image, strength, start_percent, end_percent in union_stack:
                # 应用控制网络时使用从栈中获取的开始和结束百分比
                out = ControlNetApplyAdvanced().apply_controlnet(
                    positive, negative, control_net, image, 
                    strength, start_percent, end_percent, 
                    vae, extra_concat
                )
                positive = out[0]
                negative = out[1]
        return (positive, negative, )
    

class Stack_ControlNet:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

            },
            "optional": {
                "controlnet": (["None"] + folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 0.8, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cn_stack": ("CN_STACK",),

            }
        }

    RETURN_TYPES = ("CN_STACK",)
    RETURN_NAMES = ("cn_stack",)
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack/stack"

    def controlnet_stacker(self, controlnet, strength, image=None, start_percent=0.0, end_percent=1.0, cn_stack=None):

        controlnet_list = []
        if cn_stack is not None:
            controlnet_list.extend([cn for cn in cn_stack if cn[0] != "None"])

        if controlnet != "None" and image is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            # 将start_percent和end_percent添加到元组中
            controlnet_list.append((controlnet, image, strength, start_percent, end_percent))

        return (controlnet_list,)


class Apply_ControlNetStack:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "switch": (["Off","On"],),
                            "positive": ("CONDITIONING", ),
                            "negative": ("CONDITIONING",),
                            "controlnet_stack": ("CN_STACK", ),
                            },
            "optional": {
                "vae": ("VAE",),
            }
        }                    

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("positive", "negative", )
    FUNCTION = "apply_controlnet_stack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def apply_controlnet_stack(self, positive, negative, switch, vae=None, controlnet_stack=None):

        if switch == "Off":
            return (positive, negative, )
    
        if controlnet_stack is not None:
            for controlnet_tuple in controlnet_stack:
                # 从元组中获取start_percent和end_percent
                controlnet, image, strength, start_percent, end_percent = controlnet_tuple
                
                # 使用获取到的start_percent和end_percent参数
                conditioning = ControlNetApplyAdvanced().apply_controlnet(
                    positive, negative, controlnet, image, strength, 
                    start_percent, end_percent, vae, extra_concat=[]
                )
                positive, negative = conditioning[0], conditioning[1]

        return (positive, negative, )





#endregion--------CN串联-----------------


#region--------条件--文本生成-----------------

class Stack_text:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "text_stack": ("TEXT_STACK",),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "style": (["None"] + style_list()[0], {"default": "None"}),
            }
        }

    RETURN_TYPES = ("TEXT_STACK", )
    RETURN_NAMES = ("text_stack", )
    FUNCTION = "lora_stacker"
    CATEGORY = "Apt_Preset/stack/stack"
    def lora_stacker(self, style,  pos, neg, text_stack=None,):  # 添加 text_stack 参数
        stack = list()
        if text_stack:
            stack.extend(text_stack)
            stack.append(',')  # 添加逗号隔开
        pos, neg = add_style_to_subject(style, pos, neg) 
        
        stack.extend([(pos, neg)])
        return (stack,)


class Apply_textStack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "clip": ("CLIP",),
                "text_stack": ("TEXT_STACK",),
                },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "textStack"
    CATEGORY = "Apt_Preset/stack/uppack"
    def textStack(self, clip, text_stack):
        positive_list = []
        negative_list = []
        for item in text_stack:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pos, neg = item
                if pos is not None and pos != '':
                    (positive,) = CLIPTextEncode().encode(clip, pos)
                    (negative,) = CLIPTextEncode().encode(clip, neg)
                    positive_list.append(positive)
                    negative_list.append(negative)

        if positive_list and negative_list:
            positive = sum(positive_list, [])
            negative = sum(negative_list, [])
        else:
            positive = []
            negative = []

        return (positive, negative)



class AD_sch_prompt_stack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": DefaultPromp}),
                "easing_type": (list(easing_functions.keys()), {"default": "Linear"}),
            },
            "optional": {
                "max_length": ("INT", {"default": 120, "min": 0, "max": 100000}),
                "f_text": ("STRING", {"default": "", "multiline": False}),
                "b_text": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE_STACK",)
    RETURN_NAMES = ("prompt_stack",)
    FUNCTION = "create_schedule"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = """
    - 插入缓动函数举例Examples functions：
    - 0:0.5 @Sine_In@
    - 30:1 @Linear@
    - 60:0.5
    - 90:1
    - 支持的缓动函数Supported easing functions:
    - Linear,
    - Sine_In,Sine_Out,Sine_InOut,Sin_Squared,
    - Quart_In,Quart_Out,Quart_InOut,
    - Cubic_In,Cubic_Out,Cubic_InOut,
    - Circ_In,Circ_Out,Circ_InOut,
    - Back_In,Back_Out,Back_InOut,
    - Elastic_In,Elastic_Out,Elastic_InOut,
    - Bounce_In,Bounce_Out,Bounce_InOut"
    """






    def create_schedule(self, prompts: str, max_length=0, easing_type="Linear", f_text="", b_text="",):

        PROMPT_STACK = (prompts, easing_type, max_length, f_text, b_text)
        return ( PROMPT_STACK,)


class AD_sch_prompt_apply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_stack": ("PROMPT_SCHEDULE_STACK",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("CONDITIONING","IMAGE")
    RETURN_NAMES = ("positive","graph")
    FUNCTION = "create_schedule"
    CATEGORY = "Apt_Preset/AD"

    def create_schedule(self,clip, prompt_stack=None):
        (prompts, easing_type, max_length, f_text, b_text)= prompt_stack 
        frames = parse_prompt_schedule(prompts.strip(), easing_type)

        curve_img = generate_frame_weight_curve_image(frames, max_length)
        positive = build_conditioning(frames, clip, max_length, f_text=f_text, b_text=b_text)

        return ( positive, curve_img)


#endregion--------条件生成-重置-----------------


#region--------条件----Redux stack----------------------


class YC_LG_Redux:   #作为函数调用
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            "positive": ("CONDITIONING",),
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
        
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/chx_tool"

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
    
    def apply_stylemodel(self, style_model, clip_vision, image, positive, 
                        patch_res=16, style_strength=1.0, prompt_strength=1.0, 
                        noise_level=0.0, crop="none", sharpen=0.0, guidance=30,
                        blend_mode="lerp", mask=None, ):
        
        
        conditioning = positive

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

        
        return (positive,)


class Stack_Redux:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

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
                "redux_stack": ("REDUX_STACK",),  # 新增输入
            }
        }

    RETURN_TYPES = ("REDUX_STACK",)
    RETURN_NAMES = ("redux_stack",)
    FUNCTION = "redux_stack"
    CATEGORY = "Apt_Preset/stack/stack"

    def redux_stack(self,style_model, clip_vision,  crop, sharpen, patch_res, style_strength, prompt_strength, blend_mode, noise_level, image=None,mask=None, guidance=30, redux_stack=None):

        if image is None:
            return (None,)
        

        # 初始化redux_list
        redux_list = []

        # 如果传入了redux_stack，将其中的内容添加到redux_list中
        if redux_stack is not None:
            redux_list.extend([redux for redux in redux_stack if redux[0] != "None"])

        # 将当前Redux的相关信息打包成一个元组，并添加到redux_list中
        redux_info = (
            style_model,
            clip_vision,
            image,
            crop,
            sharpen,
            patch_res,
            style_strength,
            prompt_strength,
            blend_mode,
            noise_level,
            mask,
            guidance
        )
        redux_list.append(redux_info)

        return (redux_list,)


class Apply_Redux:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "positive": ("CONDITIONING",),
            "redux_stack": ("REDUX_STACK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply_redux_stack"
    CATEGORY = "Apt_Preset/stack/uppack"

    def apply_redux_stack(self, positive, redux_stack):
        if not redux_stack:
            raise ValueError("redux_stack 不能为空")

        chx_yc_lg_redux = YC_LG_Redux()

        # 遍历 redux_stack 中的每个 Redux 配置
        for redux_info in redux_stack:
            (
                style_model,
                clip_vision,
                image,
                crop,
                sharpen,
                patch_res,
                style_strength,
                prompt_strength,
                blend_mode,
                noise_level,
                mask,
                guidance
            ) = redux_info

            # 直接调用 chx_YC_LG_Redux 类中的 apply_stylemodel 方法
            positive = chx_yc_lg_redux.apply_stylemodel(
                style_model, clip_vision, image, positive, 
                patch_res=patch_res, style_strength=style_strength, prompt_strength=prompt_strength, 
                noise_level=noise_level, crop=crop, sharpen=sharpen, guidance=guidance,
                blend_mode=blend_mode, mask=mask
            )[0]

        return (positive,)




#endregion------------------Redux stack----------------------





class Stack_pre_Mark2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "fill"}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
              
            }
        }

    RETURN_TYPES = ("MASK_STACK2", )
    RETURN_NAMES = ("mask_stack", )
    FUNCTION = "visualize"
    CATEGORY = "Apt_Preset/stack/stack"

    def visualize(self, mask_mode,smoothness, mask_expand, mask_min, mask_max):  
        
        mask_stack2 = (
            mask_mode,
            smoothness, 
            mask_expand,            
            mask_min, 
            mask_max, 

        )
        return (mask_stack2,)




#endregion---------------------收纳----------------------------




class stack_sum_pack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "lora_stack": ("LORASTACK",),
                "ipa_stack": ("IPA_STACK",),
                "redux_stack": ("REDUX_STACK",),
                "condi_stack": ("STACK_CONDI", ),
                "cn_stack": ("CN_STACK",),
                "union_stack": ("UNION_STACK",),
                "latent_stack": ("LATENT_STACK",),
            }
        }

    RETURN_TYPES = ("STACK_PACK",)
    RETURN_NAMES = ("stack_pack",)
    FUNCTION = "stackpack"
    CATEGORY = "Apt_Preset/Deprecated"

    def stackpack(self, ipa_stack=None, redux_stack=None, lora_stack=None,  condi_stack=None,union_stack=None, cn_stack=None, latent_stack=None):
        stack_pack=  ipa_stack, redux_stack, lora_stack, condi_stack,union_stack, cn_stack, latent_stack
        return (stack_pack,)


class sum_stack_all:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "stack_pack": ("STACK_PACK",),

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT" )
    RETURN_NAMES = ("context", "model", "positive", "negative","latent" )
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/Deprecated"

    def merge(self, context=None, model=None, positive=None, stack_pack=None,):
        
        if model is None:
            model = context.get("model")

        clip = context.get("clip")
        latent = context.get("latent", None)
        image_orc = context.get("images", None)
        vae = context.get("vae", None)

        ipa_stack = None
        redux_stack = None
        lora_stack = None
        condi_stack = None
        union_stack = None
        cn_stack = None
        latent_stack = None

        if stack_pack is not None:
            ipa_stack, redux_stack, lora_stack, condi_stack, union_stack, cn_stack, latent_stack = stack_pack


        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)




#----------------------------------------------------------------------------
        pos = context.get("pos","a girl")
        neg = context.get("neg","bad quality")  
        positive, = CLIPTextEncode().encode(clip, pos)
        negative, = CLIPTextEncode().encode(clip, neg)



        if redux_stack is not None:
            positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)


        if condi_stack is not None:
            positive, negative = Apply_condiStack().condiStack(clip, condi_stack)

        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])

        if cn_stack is not None and len(cn_stack) > 0:
            first_element = cn_stack[0]
            
            if len(first_element) == 5:
                positive, negative = Apply_ControlNetStack().apply_controlnet_stack(
                    positive=positive, 
                    negative=negative, 
                    switch="On", 
                    vae=vae,
                    controlnet_stack=cn_stack
                )
            elif len(first_element) == 8:
                positive, = Apply_adv_CN().apply_controlnet(positive, cn_stack)
            else:
                print(f"警告: 未知的控制网络堆栈类型，元素长度为 {len(first_element)}")



        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)
        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, vae=vae,  latent = latent)
        return (context, model, positive, negative, latent, image_orc)
    




class sum_stack_AD:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "chx_Merge": ("RUN_CONTEXT",),
                "lora_stack": ("LORASTACK",),
                "ipa_stack": ("IPA_STACK",),
                "pos_sch_stack": ("PROMPT_SCHEDULE_STACK",),
                "cn_stack": ("CN_STACK",),
                "latent_stack": ("LATENT_STACK",),
            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE" )
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent", "graph")
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"

    def merge(self, model=None,chx_Merge=None, ipa_stack=None, lora_stack=None, pos_sch_stack=None, cn_stack=None, context=None,latent_stack=None,):

        if chx_Merge is not None :
            context = Data_chx_Merge().merge(context, chx_Merge, chx_Merge)[0] 

        clip = context.get("clip", None)
        vae = context.get("vae", None)
        latent = context.get("latent",None)


        if model is None:
            model = context.get("model")
        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)
        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)

#--------------------------------------------------------------------   
        pos = context.get("pos","a girl")
        neg = context.get("neg","bad quality")  
        positive, = CLIPTextEncode().encode(clip, pos)
        negative, = CLIPTextEncode().encode(clip, neg)


        if pos_sch_stack is None:
            graph = None
        if pos_sch_stack is not None:
            positive, graph = AD_sch_prompt_apply().create_schedule(clip, pos_sch_stack)
        
        if cn_stack is not None:
            positive, = Apply_adv_CN().apply_controlnet(positive, cn_stack)

#-------------------------------------------------------------------- 
        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)

        context = new_context(context, clip=clip, positive=positive, latent=latent, negative=negative, model=model)
        return (context, model, positive, negative, latent,graph) 



class sum_stack_image:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "lora_stack": ("LORASTACK",),
                "ipa_stack": ("IPA_STACK",),
                "redux_stack": ("REDUX_STACK",),
                "condi_stack": ("STACK_CONDI", ),
                "union_stack": ("UNION_STACK",),
                "cn_stack": ("CN_STACK",),
                "inpaint": ("INPAINT_STACK",),
                "latent_stack": ("LATENT_STACK",),
            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","CONDITIONING","LATENT","VAE","CLIP")
    RETURN_NAMES = ("context", "model","positive","negative","latent","vae","clip")
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"

    def merge(self, context=None, model=None, ipa_stack=None, 
              redux_stack=None, lora_stack=None,condi_stack=None,
              union_stack=None, cn_stack=None,inpaint=None, latent_stack=None):
        

        clip = context.get("clip")
        latent = context.get("latent", None)
        vae = context.get("vae", None)
        
        if model is None:
            model = context.get("model", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)

#------------条件防止被重置-----------------------------
        pos = context.get("pos","a girl")
        neg = context.get("neg","bad quality")  
        positive, = CLIPTextEncode().encode(clip, pos)
        negative, = CLIPTextEncode().encode(clip, neg)
#-------------------------------------------

        if redux_stack is not None:
            positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)

        if condi_stack is not None:
            positive, negative = Apply_condiStack().condiStack(clip, condi_stack)



        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])


        if cn_stack is not None and len(cn_stack) > 0:
            first_element = cn_stack[0]
            
            if len(first_element) == 5:
                positive, negative = Apply_ControlNetStack().apply_controlnet_stack(
                    positive=positive, 
                    negative=negative, 
                    switch="On", 
                    vae=vae,
                    controlnet_stack=cn_stack
                )
            elif len(first_element) == 8:
                positive, = Apply_adv_CN().apply_controlnet(positive, cn_stack)
            else:
                print(f"警告: 未知的控制网络堆栈类型，元素长度为 {len(first_element)}")


        if inpaint is not None:
            control_net, mask_mode, image, mask, smoothness, strength, start_percent, end_percent = inpaint
            positive, negative, latent = pack_inpaint().apply_controlnet(
                vae, positive, negative, control_net, image, 
                strength, smoothness, start_percent, end_percent, 
                mask_mode, mask
            )


        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)

        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent=latent, vae=vae)
        return (context, model, positive, negative, latent,vae,clip )







#endregion-------stack_pack------------------------------------------------------------------------------#

  


class pre_condi_combine_switch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),

            "detail1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "detail2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),

            "structure_switch": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,"display": "slider"}),

            }
            }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("condi_combine",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_tool"

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2, )


    def set_range(self, conditioning, start, end):
        c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                "end_percent": end})
        return (c, )

    def append(self, conditioning, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"strength": strength})
        return (c, )


    def run(self, conditioning_to, conditioning_from, detail1, detail2, structure_switch):
        # 确保 structure_switch 在合理范围内
        structure_switch = max(0.0, min(1.0, structure_switch))
        
        # 第一个条件应用在 0 到 structure_switch 范围
        conditioning_to = self.set_range(conditioning_to, 0.0, structure_switch)[0]
        conditioning_to = self.append(conditioning_to, detail1)[0]
        
        # 第二个条件应用在 structure_switch 到 1 范围
        conditioning_from = self.set_range(conditioning_from, structure_switch, 1.0)[0]
        conditioning_from = self.append(conditioning_from, detail2)[0]
        
        # 合并两个条件
        c = self.combine(conditioning_to, conditioning_from)[0]
        
        return (c, )












































































