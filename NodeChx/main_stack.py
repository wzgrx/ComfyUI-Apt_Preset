
#region-------------------------------import-----------------------

from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
import comfy.controlnet
import comfy.sd
import folder_paths
import comfy.utils
import hashlib
from random import random, uniform
import torch
import numpy as np
from nodes import common_ksampler, CLIPTextEncode, ControlNetApplyAdvanced, VAEDecode, VAEEncode, InpaintModelConditioning, ControlNetLoader
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import node_helpers
import cv2
from PIL import Image, ImageFilter
from comfy_extras.nodes_controlnet import SetUnionControlNetType



from .AdvancedCN import *
from .AnimateDiff import *
from .IPAdapterSD3 import Apply_IPA_SD3
from .IPAdapterPlus import ipadapter_execute, IPAdapterUnifiedLoader
import numexpr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import json
import comfy.samplers
from ..main_unit import *




WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']


#endregion-----------------------------import----------------------------


#region---------------------收纳----------------------------


class Stack_ControlNet:
    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {
                },
                "optional": {
                    "controlnet_1": (["None"] + folder_paths.get_filename_list("controlnet"),),
                    "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "controlnet_2": (["None"] + folder_paths.get_filename_list("controlnet"),),
                    "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "controlnet_3": (["None"] + folder_paths.get_filename_list("controlnet"),),
                    "controlnet_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "image_1": ("IMAGE",),
                    "image_2": ("IMAGE",),
                    "image_3": ("IMAGE",),
                },
        }

    RETURN_TYPES = ("CN_STACK", )
    RETURN_NAMES = ("CONTROLNET_STACK", )
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack"

    def controlnet_stacker(self, controlnet_1, controlnet_strength_1, 
                        controlnet_2, controlnet_strength_2, 
                        controlnet_3, controlnet_strength_3, 
                        image_1=None, image_2=None, image_3=None, ):

        controlnet_list= []

        if controlnet_1 != "None" and image_1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_1)
            controlnet_1 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_1, image_1, controlnet_strength_1)]),

        if controlnet_2 != "None"  and image_2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_2)
            controlnet_2 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_2, image_2, controlnet_strength_2)]),

        if controlnet_3 != "None"  and image_3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_3)
            controlnet_3 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_3, image_3, controlnet_strength_3)]),

        return (controlnet_list, )


class Stack_ControlNet1:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

            },
            "optional": {
                "controlnet": (["None"] + folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 0.8, "min": -10.0, "max": 10.0, "step": 0.01}),
                "cn_stack": ("CN_STACK",),

            }
        }

    RETURN_TYPES = ("CN_STACK",)
    RETURN_NAMES = ("cn_stack",)
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack"

    def controlnet_stacker(self, controlnet, strength, image=None, cn_stack=None):

        controlnet_list = []
        if cn_stack is not None:
            controlnet_list.extend([cn for cn in cn_stack if cn[0] != "None"])

        if controlnet != "None" and image is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.append((controlnet, image, strength,))

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
    CATEGORY = "Apt_Preset/stack/apply"

    def apply_controlnet_stack(self, positive, negative, switch, vae=None, controlnet_stack=None,):

        if switch == "Off":
            return (positive, negative, )
    
        if controlnet_stack is not None:
            for controlnet_tuple in controlnet_stack:
                controlnet, image, strength = controlnet_tuple
                
                conditioning = ControlNetApplyAdvanced().apply_controlnet( positive, negative, controlnet, image, strength, 0, 1, vae, extra_concat=[])
                positive, negative = conditioning[0], conditioning[1]

        return (positive, negative, )



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
    CATEGORY = "Apt_Preset/stack"
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
    CATEGORY = "Apt_Preset/stack/apply"
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
    CATEGORY = "Apt_Preset/stack"


    def lora_stacker(self, lora_name_1, weight_1, lora_name_2, weight_2, lora_name_3, weight_3, lora_stack=None):
        """
        将多个 LoRA 配置添加到堆栈中。
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
    CATEGORY = "Apt_Preset/stack/apply"

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


class Stack_condi:
    @classmethod
    def INPUT_TYPES(s):



        
        return {

            "required": {
            },

            "optional": {
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "pos4": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                # 将 background 重命名为 background
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                # 移除原有的 mask_5 输入
                # "mask_5": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),

                "background": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "background is sea"}),
                "neg": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Poor quality"}),
            }
        }
        
    RETURN_TYPES = ("STACK_CONDI",)
    RETURN_NAMES = ("condi_stack", )
    FUNCTION = "stack_condi"
    CATEGORY = "Apt_Preset/stack"

    def stack_condi(self, pos1, pos2, pos3, pos4, background, neg, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_1=None, mask_2=None, mask_3=None, mask_4=None):
        condi_stack = list()
        
        # 打包逻辑：每组 pos、mask 和 mask_strength 是配套的
        def pack_group(pos, mask, mask_strength):
            if mask is not None:  # 如果 mask 存在，则打包整组
                return {
                    "pos": pos,
                    "mask": mask,
                    "mask_strength": mask_strength,
                }
            return None  # 如果 mask 不存在，则忽略整组
        
        valid_masks = []
        if mask_1 is not None:
            valid_masks.append(mask_1)
        if mask_2 is not None:
            valid_masks.append(mask_2)
        if mask_3 is not None:
            valid_masks.append(mask_3)
        if mask_4 is not None:
            valid_masks.append(mask_4)

        if valid_masks:
            total_mask = sum(valid_masks)
            mask_5 = 1 - total_mask
        else:
            mask_5 = None

        # 打包每组信息
        group1 = pack_group(pos1, mask_1, mask_1_strength)
        group2 = pack_group(pos2, mask_2, mask_2_strength)
        group3 = pack_group(pos3, mask_3, mask_3_strength)
        group4 = pack_group(pos4, mask_4, mask_4_strength)
        group5 = pack_group(background, mask_5, 1)
        
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
    CATEGORY = "Apt_Preset/stack/apply"

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

            if mask is not None:
                if len(mask.shape) < 3:
                    mask = mask.unsqueeze(0)
                for t in encoded_pos:
                    append_helper(t, mask, positive, set_area_to_bounds, mask_strength)
                    
        positive = positive

        return (positive, negative)



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
    CATEGORY = "Apt_Preset/stack"

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
    CATEGORY = "Apt_Preset/stack/apply"

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
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1, "display": "slider"}),
                "ratio_selected": (['None'] + cls.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300})
            }
        }

    RETURN_TYPES = ("LATENT_STACK",)
    RETURN_NAMES = ("latent_stack",)
    FUNCTION = "stack_latent"
    CATEGORY = "Apt_Preset/stack"

    def stack_latent(self, latent=None, pixels=None, mask=None, noise_mask=True, smoothness=1, ratio_selected="None", batch_size=1):
        latent_info = (latent, pixels, mask, noise_mask, smoothness, ratio_selected, batch_size)
        # 这里直接将单个 latent_info 作为一个只含一个元素的列表，代表 stack
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
    CATEGORY = "Apt_Preset/stack/apply"

    def apply_latent_stack(self, model, positive, negative, vae, latent_stack):
        # 先初始化一个默认的 latent ，避免后续操作中 latent 为 None 的情况
        default_width = 512
        default_height = 512
        batch_size = 1

        for latent_info in latent_stack:
            latent, pixels, mask, noise_mask, smoothness, ratio_selected, batch_size = latent_info

            if ratio_selected == "None" and latent is None and pixels is None:
                raise ValueError("No valid input provided for latent, pixels, or ratio_selected.")


            if ratio_selected != "None":
                width = self.ratio_dict[ratio_selected]["width"]
                height = self.ratio_dict[ratio_selected]["height"]
                latent = {"samples": torch.zeros([batch_size, 4, height // 8, width // 8])}
                return model, positive, negative, latent

            if latent is None :
                latent = {"samples": torch.zeros([batch_size, 4, default_height // 8, default_width // 8])}
                
            if pixels is not None:
                latent = VAEEncode().encode(vae, pixels)[0]

            if mask is not None:
                mask = tensor2pil(mask)
                feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
                mask = pil2tensor(feathered_image)
                positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask)
                model = DifferentialDiffusion().apply(model)[0]

        latent = latentrepeat(latent, batch_size)[0]
        return model, positive, negative, latent





class Stack_CN_union:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},

            "optional": {
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

    RETURN_TYPES = ("UNION_STACK", )
    RETURN_NAMES = ("union_stack", )
    CATEGORY = "Apt_Preset/stack"
    FUNCTION = "load_controlnet"

    def load_controlnet(self,  strength1, strength2, strength3, image1=None, image2=None, image3=None,
                        controlNet=None, type1=None, type2=None, type3=None, extra_concat=[]):
        if controlNet == "None":
            # 假设 union_stack 初始为空，这里简单返回空元组
            return ((), )
        control_net = ControlNetLoader().load_controlnet(controlNet)[0]
        stack = []
        if type1 != "None" and strength1 != 0 and image1 is not None:
            control_net_type1 = SetUnionControlNetType().set_controlnet_type(control_net, type1)[0]
            stack.append((control_net_type1, image1, strength1))
        if type2 != "None" and strength2 != 0 and image2 is not None:
            control_net_type2 = SetUnionControlNetType().set_controlnet_type(control_net, type2)[0]
            stack.append((control_net_type2, image2, strength2))
        if type3 != "None" and strength3 != 0 and image3 is not None:
            control_net_type3 = SetUnionControlNetType().set_controlnet_type(control_net, type3)[0]
            stack.append((control_net_type3, image3, strength3))
        return (tuple(stack), )


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
    CATEGORY = "Apt_Preset/stack/apply"
    FUNCTION = "apply_union_stack"

    
    def apply_union_stack(self, positive, negative, vae=None, union_stack=None, extra_concat=[]):
        for control_net, image, strength in union_stack:
            out = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image, strength, 0, 1, vae, extra_concat)
            positive = out[0]
            negative = out[1]
        return (positive, negative, )







#region------------------AD prompt schedule------------------------

#region------------------AD def----------------------


class ScheduleSettings:
    def __init__(
            self,
            text_g: str,
            pre_text_G: str,
            app_text_G: str,
            text_L: str,
            pre_text_L: str,
            app_text_L: str,
            max_frames: int,
            current_frame: int,
            print_output: bool,
            pw_a: float,
            pw_b: float,
            pw_c: float,
            pw_d: float,
            start_frame: int,
            end_frame:int,
            width: int,
            height: int,
            crop_w: int,
            crop_h: int,
            target_width: int,
            target_height: int,
    ):
        self.text_g=text_g
        self.pre_text_G=pre_text_G
        self.app_text_G=app_text_G
        self.text_l=text_L
        self.pre_text_L=pre_text_L
        self.app_text_L=app_text_L
        self.max_frames=max_frames
        self.current_frame=current_frame
        self.print_output=print_output
        self.pw_a=pw_a
        self.pw_b=pw_b
        self.pw_c=pw_c
        self.pw_d=pw_d
        self.start_frame=start_frame
        self.end_frame=end_frame
        self.width=width
        self.height=height
        self.crop_w=crop_w
        self.crop_h=crop_h
        self.target_width=target_width
        self.target_height=target_height

    def set_sync_option(self, sync_option: bool):
        self.sync_context_to_pe = sync_option


defaultPrompt = """"0" :"",
"30" :"",
"60" :"",
"90" :"",
"120" :""
"""


defaultValue = """0:(0),
30:(0),
60:(0),
90:(0),
120:(0)
"""


def batch_parse_key_frames(string, max_frames):
    # because math functions (i.e. sin(t)) can utilize brackets
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    string = re.sub(r',\s*$', '', string)
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        frame = int(sanitize_value(frameParam[0])) if check_is_number(
            sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
            frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",1)[::-1]))
        frames[frame] = frameParam[1].strip()
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def sanitize_value(value):
    # Remove single quotes, double quotes, and parentheses
    value = value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")
    return value


def batch_get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
    value_is_number = False
    for i in range(0, max_frames):
        if i in key_frames:
            value = str(key_frames[i])  # Convert to string to ensure it's treated as an expression
            value_is_number = check_is_number(sanitize_value(value))
            if value_is_number:
                key_frame_series[i] = sanitize_value(value)
        if not value_is_number:
            t = i
            # workaround for values formatted like 0:("I am test") //used for sampler schedules
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitize_value(value)
        elif is_single_string:  # take previous string value and replicate it
            key_frame_series[i] = key_frame_series[i - 1]
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

#--------------------------------------------------------------------------------------
def batch_prompt_schedule(settings:ScheduleSettings,clip):
    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # Add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    p = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, settings)
    n = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, settings)

    # return positive and negative conditioning as well as the current and next prompts for each
    return (p, n,)


def process_input_text(text: str) -> dict:
    input_text = text.replace('\n', '')
    input_text = "{" + input_text + "}"
    input_text = re.sub(r',\s*}', '}', input_text)
    animation_prompts = json.loads(input_text.strip())
    return animation_prompts


def BatchPoolAnimConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip, settings:ScheduleSettings):
    pooled_out = []
    cond_out = []
    max_size = 0

    if settings.end_frame == 0:
        settings.end_frame = settings.max_frames
        print("end_frame at 0, using max_frames instead!")

    if settings.start_frame >= settings.end_frame:
        settings.start_frame = 0
        print("start_frame larger than or equal to end_frame, using max_frames instead!")

    if max_size == 0:
        for i in range(0, settings.end_frame):
            tokens = clip.tokenize(str(cur_prompt_series[i]))
            cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
            max_size = max(max_size, cond_to.shape[1])
    for i in range(settings.start_frame, settings.end_frame):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        if i < len(nxt_prompt_series):
            tokens = clip.tokenize(str(nxt_prompt_series[i]))
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            cond_from, pooled_from = torch.zeros_like(cond_to), torch.zeros_like(pooled_to)

        interpolated_conditioning = addWeighted([[cond_to, {"pooled_output": pooled_to}]],
                                                [[cond_from, {"pooled_output": pooled_from}]],
                                                weight_series[i],max_size)

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from)

        cond_out.append(interpolated_cond)
        pooled_out.append(interpolated_pooled)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return [[final_conditioning, {"pooled_output": final_pooled_output}]]


def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength, max_size=0):
    out = []

    if len(conditioning_from) > 1:
        print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        if max_size == 0:
            max_size = max(t1.shape[1], cond_from.shape[1])
        t0, max_size = pad_with_zeros(cond_from, max_size)
        t1, max_size = pad_with_zeros(t1, t0.shape[1])  # Padding t1 to match max_size
        t0, max_size = pad_with_zeros(t0, t1.shape[1])

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        t_to = conditioning_to[i][1].copy()

        t_to["pooled_output"] = pooled_output_from
        n = [tw, t_to]
        out.append(n)

    return out


def pad_with_zeros(tensor, target_length):
    current_length = tensor.shape[1]

    if current_length < target_length:
        # Calculate the required padding length
        pad_length = target_length - current_length

        # Calculate padding on both sides to maintain the tensor's original shape
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        # Pad the tensor along the second dimension
        tensor = F.pad(tensor, (0, 0, left_pad, right_pad))

    return tensor, target_length


def batch_split_weighted_subprompts(text, pre_text, app_text):
    pos = {}
    neg = {}
    pre_text = str(pre_text)
    app_text = str(app_text)

    if "--neg" in pre_text:
        pre_pos, pre_neg = pre_text.split("--neg")
    else:
        pre_pos, pre_neg = pre_text, ""

    if "--neg" in app_text:
        app_pos, app_neg = app_text.split("--neg")
    else:
        app_pos, app_neg = app_text, ""

    for frame, prompt in text.items():
        negative_prompts = ""
        positive_prompts = ""
        prompt_split = prompt.split("--neg")

        if len(prompt_split) > 1:
            positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
        else:
            positive_prompts = prompt_split[0]

        pos[frame] = ""
        neg[frame] = ""
        pos[frame] += (str(pre_pos) + " " + positive_prompts + " " + str(app_pos))
        neg[frame] += (str(pre_neg) + " " + negative_prompts + " " + str(app_neg))
        if pos[frame].endswith('0'):
            pos[frame] = pos[frame][:-1]
        if neg[frame].endswith('0'):
            neg[frame] = neg[frame][:-1]
    return pos, neg


def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)


def convert_pw_to_tuples(settings):
    if isinstance(settings.pw_a, (int, float, np.float64)):
        settings.pw_a = tuple([settings.pw_a] * settings.max_frames)
    if isinstance(settings.pw_b, (int, float, np.float64)):
        settings.pw_b = tuple([settings.pw_b] * settings.max_frames)
    if isinstance(settings.pw_c, (int, float, np.float64)):
        settings.pw_c = tuple([settings.pw_c] * settings.max_frames)
    if isinstance(settings.pw_d, (int, float, np.float64)):
        settings.pw_d = tuple([settings.pw_d] * settings.max_frames)


def interpolate_prompt_seriesA(animation_prompts, settings:ScheduleSettings):

    max_f = settings.max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}


    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(settings.max_frames):
        sorted_prompts.append((str(settings.max_frames), sorted_prompts[-1][1]))

    # Setup containers for interpolated prompts
    nan_list = [np.nan for a in range(settings.max_frames)]
    cur_prompt_series = pd.Series(nan_list,dtype=object)
    nxt_prompt_series = pd.Series(nan_list,dtype=object)

    # simple array for strength values
    weight_series = [np.nan] * settings.max_frames

    # in case there is only one keyed prompt, set all prompts to that prompt
    if settings.max_frames == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

    #make sure prompt weights are tuples and convert them if not
    convert_pw_to_tuples(settings)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0

    index_offset = 0

    # Evaluate the current and next prompt's expressions
    for i in range(settings.start_frame, min(settings.end_frame,len(cur_prompt_series))):
        cur_prompt_series[i] = prepare_batch_promptA(cur_prompt_series[i], settings, i)
        nxt_prompt_series[i] = prepare_batch_promptA(nxt_prompt_series[i], settings, i)
        if settings.print_output == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            if(settings.start_frame >= i):
                if(settings.end_frame > 0):
                    if(settings.end_frame > i):
                        print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i),
                              "\n", "Current Prompt: ",
                              cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                              weight_series[i], "\n")
                else:
                    print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i), "\n", "Current Prompt: ",
                          cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                          weight_series[i], "\n")
        index_offset = index_offset + 1

    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)


def prepare_batch_promptA(prompt, settings:ScheduleSettings, index):
    max_f = settings.max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace(
            't',
            f'{index}').replace("pw_a",
            f"{settings.pw_a[index]}").replace("pw_b",
            f"{settings.pw_b[index]}").replace("pw_c",
            f"{settings.pw_c[index]}").replace("pw_d",
            f"{settings.pw_d[index]}").replace("max_f",
            f"{max_f}").replace('`', '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()


#endregion--------------------------------def-------------------------------------------------


class Apply_prompt_Schedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "pos_sch_stack": ("PROMPT_SCHEDULE_STACK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_schedules"
    CATEGORY = "Apt_Preset/stack/apply"

    def apply_schedules(self, clip, pos_sch_stack):
        if not pos_sch_stack:
            return (None, None,)
        
        positive = None
        negative = None

        # Apply each schedule in stack
        for schedule_info in pos_sch_stack:
            text, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d = schedule_info
            
            # Generate conditioning from schedule parameters
            settings = ScheduleSettings(
                text_g=text,
                pre_text_G=pre_text,
                app_text_G=app_text,
                text_L=None,
                pre_text_L=None,
                app_text_L=None,
                max_frames=max_frames,
                current_frame=None,
                print_output=None,
                pw_a=pw_a,
                pw_b=pw_b,
                pw_c=pw_c,
                pw_d=pw_d,
                start_frame=0,
                end_frame=max_frames,
                width=None,
                height=None,
                crop_w=None,
                crop_h=None,
                target_width=None,
                target_height=None,
            )
            
        positive, negative = batch_prompt_schedule(settings, clip)

        return (positive, negative,)


#endregion------------------prompt schedule------------------------


#endregion---------------------收纳----------------------------



#region------------------Redux stack----------------------


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
    CATEGORY = "Apt_Preset/stack"

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
    CATEGORY = "Apt_Preset/stack/apply"

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



class AD_stack_prompt:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),

            },
            "optional": {

                "pre_text": ("STRING", {"multiline": False}),
                "app_text": ("STRING", {"multiline": False}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE_STACK",)
    RETURN_NAMES = ("pos_sch_stack",)
    FUNCTION = "stack_schedule"
    CATEGORY = "Apt_Preset/AD"

    def stack_schedule(self, text, max_frames, pre_text=None, app_text=None, pw_a=None, pw_b=None, pw_c=None, pw_d=None, ):
        schedule_list = []
        schedule_info = (text, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d,)
        schedule_list.append(schedule_info)

        return (schedule_list,)




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
                "ipa3_stack": ("IPA3_STACK",),
                "ipa_stack": ("IPA_STACK",),
                "text_stack": ("TEXT_STACK",),
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
    CATEGORY = "Apt_Preset/stack"

    def stackpack(self, ipa3_stack=None, ipa_stack=None, redux_stack=None, lora_stack=None, text_stack=None, condi_stack=None,union_stack=None, cn_stack=None, latent_stack=None):
        stack_pack= ipa3_stack, ipa_stack, redux_stack, lora_stack, text_stack, condi_stack,union_stack, cn_stack, latent_stack
        return (stack_pack,)




class sum_stack_image:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "positive": ("CONDITIONING",),
                "lora_stack": ("LORASTACK",),
                "ipa3_stack": ("IPA3_STACK",),
                "ipa_stack": ("IPA_STACK",),
                "text_stack": ("TEXT_STACK",),
                "redux_stack": ("REDUX_STACK",),
                "condi_stack": ("STACK_CONDI", ),
                "union_stack": ("UNION_STACK",),
                "cn_stack": ("CN_STACK",),
                "latent_stack": ("LATENT_STACK",),



            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT" ,"IMAGE" )
    RETURN_NAMES = ("context", "model", "positive", "negative","latent", "image" )
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_load"

    def merge(self, context=None, model=None, positive=None, ipa3_stack=None, ipa_stack=None, redux_stack=None, lora_stack=None, text_stack=None, condi_stack=None,union_stack=None, cn_stack=None, latent_stack=None):
        
        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")

        clip = context.get("clip")
        negative = context.get("negative")
        latent = context.get("latent", None)
        image_orc = context.get("images", None)
        vae = context.get("vae", None)



        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)


        if ipa3_stack is not None:
            model, =  Apply_IPA_SD3().apply_ipa_stack(model, ipa3_stack)


        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)

        if text_stack and text_stack is not None:
            if text_stack is not None:
                positive, negative = Apply_textStack().textStack(clip,text_stack)


        if redux_stack is not None:
            positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)


        if condi_stack is not None:
            positive, negative = Apply_condiStack().condiStack(clip, condi_stack)


        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])


        if cn_stack is not None:  # 常规cn
            positive, negative = Apply_ControlNetStack().apply_controlnet_stack(
                positive=positive, 
                negative=negative, 
                switch="On", 
                vae=vae,
                controlnet_stack=cn_stack
            )


        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)



        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent = latent)
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
                "positive": ("CONDITIONING",),
                "lora_stack": ("LORASTACK",),
                "ad_stack": ("AD_STACK",),

                "ipa_stack": ("IPA_STACK",),
                "pos_sch_stack": ("PROMPT_SCHEDULE_STACK",),
                "cn_stack": ("ADV_CN_STACK",),
                "latent_stack": ("LATENT_STACK",),
            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT",)
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_load"


    def merge(self, model=None,ad_stack=None, positive=None, ipa_stack=None, lora_stack=None, pos_sch_stack=None, cn_stack=None, context=None,latent_stack=None,):
        
        clip = context.get("clip", None)
        negative = context.get("negative", None)
        vae = context.get("vae", None)
        latent = context.get("latent",None)

        if model is None:
            model = context.get("model")
        
        if positive is None:
            positive = context.get("positive")
        
        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)
            # 确保 model 是模型对象，而不是元组，非model[-1]，已经处理了
            if isinstance(model, tuple):
                model = model[0]

        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)

        if ad_stack is not None:
            model = Apply_AD_diff().apply_ad_params(model, ad_stack)
            # 确保 model 是模型对象，而不是元组
            if isinstance(model, tuple):
                model = model[0]


        if pos_sch_stack is not None:
            positive, negative = Apply_prompt_Schedule().apply_schedules(clip, pos_sch_stack)


        if cn_stack is not None:
            positive, = Apply_adv_CN().apply_controlnet(positive, cn_stack)


        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)



        context = new_context(context, clip=clip, positive=positive, latent=latent, negative=negative, model=model)
        return (context, model, positive, negative, latent )



class sum_stack_all:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "positive": ("CONDITIONING",),
                "stack_pack": ("STACK_PACK",),

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT" ,"IMAGE" )
    RETURN_NAMES = ("context", "model", "positive", "negative","latent", "image" )
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_load"

    def merge(self, context=None, model=None, positive=None, stack_pack=None,):
        
        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")

        clip = context.get("clip")
        negative = context.get("negative")
        latent = context.get("latent", None)
        image_orc = context.get("images", None)
        vae = context.get("vae", None)



        # 初始化所有变量为 None
        ipa3_stack = None
        ipa_stack = None
        redux_stack = None
        lora_stack = None
        text_stack = None
        condi_stack = None
        union_stack = None
        cn_stack = None
        latent_stack = None



        if stack_pack is not None:
            ipa3_stack, ipa_stack, redux_stack, lora_stack, text_stack, condi_stack, union_stack, cn_stack, latent_stack = stack_pack


        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)


        if ipa3_stack is not None:
            model, =  Apply_IPA_SD3().apply_ipa_stack(model, ipa3_stack)


        if ipa_stack is not None:
            model, = Apply_IPA().apply_ipa_stack(model, ipa_stack)

        if text_stack and text_stack is not None:
            if text_stack is not None:
                positive, negative = Apply_textStack().textStack(clip,text_stack)


        if redux_stack is not None:
            positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)


        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])


        if condi_stack is not None:
            positive, negative = Apply_condiStack().condiStack(clip, condi_stack)


        if cn_stack is not None:  # 常规cn
            positive, negative = Apply_ControlNetStack().apply_controlnet_stack(
                positive=positive, 
                negative=negative, 
                switch="On", 
                vae=vae,
                controlnet_stack=cn_stack
            )


        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)


        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent = latent)
        return (context, model, positive, negative, latent, image_orc)



