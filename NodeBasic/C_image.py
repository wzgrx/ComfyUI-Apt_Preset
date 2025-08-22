
from nodes import MAX_RESOLUTION
import torch

import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from typing import Literal, Any
import math
from comfy_extras.nodes_mask import composite
from comfy.utils import common_upscale
import re
from math import ceil, sqrt
from typing import cast
from rembg import remove
from PIL import Image, ImageDraw,  ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps
import logging
from tqdm import tqdm
import onnxruntime as ort
from enum import Enum
import random
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import copy
from pymatting import estimate_alpha_cf, estimate_foreground_ml, fix_trimap
import ast
from nodes import CLIPTextEncode, common_ksampler

from ..main_unit import *







if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#---------------------安全导入

try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False






#--------------------------------------------------------------------------------------#





#region --------batch-------------------------


class str_edit:
    def __init__(self):
        pass
    @classmethod
    def convert_list(cls, string_input,arrangement=True):
        if string_input == "":
            return ([],)
        if arrangement:
            string_input = cls.tolist_v1(string_input)
        if string_input[0] != "[":
            string_input = "[" + string_input + "]"
            return (ast.literal_eval(string_input),)
        else:
            return (ast.literal_eval(string_input),)
        
    def tolist_v1(cls,user_input):#转换为简单的带负数多维数组格式
        user_input = user_input.replace('{', '[').replace('}', ']')# 替换大括号
        user_input = user_input.replace('(', '[').replace(')', ']')# 替换小括号
        user_input = user_input.replace('，', ',')# 替换中文逗号
        user_input = re.sub(r'\s+', '', user_input)#去除空格和换行符
        user_input = re.sub(r'[^\d,.\-[\]]', '', user_input)#去除非数字字符，但不包括,.-[]
        return user_input
    @classmethod
    def tolist_v2(cls,str_input,to_list=True,to_oneDim=False,to_int=False,positive=False):#转换为数组格式
        if str_input == "":
            if to_list:return ([],)
            else:return ""
        else:
            str_input = str_input.replace('，', ',')# 替换中文逗号
            if to_oneDim:
                str_input = re.sub(r'[\(\)\[\]\{\}（）【】｛｝]', "" , str_input)
                str_input = "[" + str_input + "]"
            else:
                text=re.sub(r'[\(\[\{（【｛]', '[', text)#替换括号
                text=re.sub(r'[\)\]\}）】｝]', ']', text)#替换反括号
                if str_input[0] != "[":str_input = "[" + str_input + "]"
            str_input = re.sub(r'[^\d,.\-[\]]', '', str_input)#去除非数字字符，但不包括,.-[]
            str_input = re.sub(r'(?<![0-9])[,]', '', str_input)#如果,前面不是数字则去除
            #str_input = re.sub(r'(-{2,}|\.{2,})', '', str_input)#去除多余的.和-
            str_input = re.sub(r'\.{2,}', '.', str_input)#去除多余的.
            if positive:
                str_input = re.sub(r'-','', str_input)#移除-
            else:
                str_input = re.sub(r'-{2,}', '-', str_input)#去除多余的-
            list1=np.array(ast.literal_eval(str_input))
            if to_int:
                list1=list1.astype(int)
            if to_list:
                return list1.tolist()
            else:
                return str_input
            
    def repair_brackets(cls,str_input):#括号补全(待开发)
        pass



class Image_batch_select:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2,"}),
                "canvas_operations": (["None", "Horizontal Flip", "Vertical Flip", "90 Degree Rotation", "180 Degree Rotation", "Horizontal Flip + 90 Degree Rotation", "Horizontal Flip + 180 Degree Rotation"], {"default": "None"}),
            },
        }
    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"

    def SelectImages(self, images, indexes, canvas_operations):
        select_list = np.array(str_edit.tolist_v2(
            indexes, to_oneDim=True, to_int=True, positive=True))
        select_list1 = select_list[(select_list >= 1) & (
            select_list <= len(images))]-1
        if len(select_list1) < 1:  # 若输入的编号全部不在范围内则返回原输入
            print(
                "Warning:The input value is out of range, return to the original input.")
            return (images, [])
        else:
            exclude_list = np.arange(1, len(images) + 1)-1
            exclude_list = np.setdiff1d(exclude_list, select_list1)  # 排除的图像
            if len(select_list1) < len(select_list):  # 若输入的编号超出范围则仅输出符合编号的图像
                n = abs(len(select_list)-len(select_list1))
                print(
                    f"Warning:The maximum value entered is greater than the batch number range, {n} maximum values have been removed.")
            print(f"Selected the first {select_list1} image")

            selected_images = images[torch.tensor(select_list1, dtype=torch.long)]
            excluded_images = images[torch.tensor(exclude_list, dtype=torch.long)]

            def apply_operation(images):
                if canvas_operations == "Horizontal Flip":
                    return torch.flip(images, [2])
                elif canvas_operations == "Vertical Flip":
                    return torch.flip(images, [1])
                elif canvas_operations == "90 Degree Rotation":
                    return torch.rot90(images, 1, [1, 2])
                elif canvas_operations == "180 Degree Rotation":
                    return torch.rot90(images, 2, [1, 2])
                elif canvas_operations == "Horizontal Flip + 90 Degree Rotation":
                    flipped = torch.flip(images, [2])
                    return torch.rot90(flipped, 1, [1, 2])
                elif canvas_operations == "Horizontal Flip + 180 Degree Rotation":
                    flipped = torch.flip(images, [2])
                    return torch.rot90(flipped, 2, [1, 2])
                return images

            selected_images = apply_operation(selected_images)
            excluded_images = apply_operation(excluded_images)

            return (selected_images, excluded_images)



class Image_batch_composite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bg_image": ("IMAGE",),
                "batch_image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "Invert": ("BOOLEAN", {"default": False}),

            },
            "optional": {
                "batch_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Apt_Preset/image"

    def composite(self, batch_image, bg_image, x, y, resize_source, Invert, batch_mask = None):
        if Invert and batch_mask is not None:
            batch_mask = 1 - batch_mask
        batch_image = batch_image.clone().movedim(-1, 1)


        output = composite(batch_image, bg_image.movedim(-1, 1), x, y, batch_mask, 1, resize_source).movedim(1, -1)
        return (output,)


#endregion--------batch-------------------------



class Image_pad_outfill:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "bottom": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "left": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "right": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "color": ("STRING", {"default": "#000000"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scale_mode": (["width", "height", "out_image"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    def add_padding(self, image, left, top, right, bottom, color):  # 移除 transparent 参数
        padded_images = []
        image = [tensor2pil(img) for img in image]
        for img in image:
            # 固定使用 RGB 模式，移除 transparent 相关逻辑
            padded_image = Image.new("RGB", 
                (img.width + left + right, img.height + top + bottom), 
                hex_to_rgb_tuple(color))  # 假设 COLOR 输入需要转换为 RGB 元组
            padded_image.paste(img, (left, top))
            padded_images.append(pil2tensor(padded_image))
        return torch.cat(padded_images, dim=0)
    
    def create_mask(self, image, left, top, right, bottom, mask=None):
        masks = []
        image = [tensor2pil(img) for img in image]
        if mask is not None:
            mask = [tensor2pil(mask) for mask in mask]
        for i, img in enumerate(image):
            shape = (left, top, img.width + left, img.height + top)
            mask_image = Image.new("L", (img.width + left + right, img.height + top + bottom), 255)
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle(shape, fill=0)
            if mask is not None:
                # 确保输入的 mask 与图片位置保持固定
                mask_image.paste(mask[i], (left, top))
            masks.append(pil2tensor(mask_image))
        return torch.cat(masks, dim=0)

    def scale_image_and_mask(self, image, mask, scale, scale_mode, original_width, original_height, output_width, output_height):
        scaled_images = []
        scaled_masks = []
        image = [tensor2pil(img) for img in image]
        if mask is not None:
            mask = [tensor2pil(mask) for mask in mask]

        # 计算 out_image 输出后的宽高比
        aspect_ratio = output_width / output_height

        for i, img in enumerate(image):
            if scale_mode == "width":
                # 输出图像宽度等于原图宽度，高度按 out_image 比例计算
                new_width = original_width
                new_height = int(new_width / aspect_ratio)
            elif scale_mode == "height":
                # 输出图像高度等于原图高度，宽度按 out_image 比例计算
                new_height = original_height
                new_width = int(new_height * aspect_ratio)
            elif scale_mode == "out_image":
                # 按缩放比例调整宽高
                new_width = output_width 
                new_height = output_height

            new_width = int(new_width * scale)
            new_height = int(new_height * scale)


            scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
            scaled_images.append(pil2tensor(scaled_img))

            if mask is not None:
                scaled_mask = mask[i].resize((new_width, new_height), Image.LANCZOS)
                scaled_masks.append(pil2tensor(scaled_mask))

        scaled_images = torch.cat(scaled_images, dim=0)
        if mask is not None:
            scaled_masks = torch.cat(scaled_masks, dim=0)
        else:
            scaled_masks = None

        return scaled_images, scaled_masks

    def resize(self, image, left, top, right, bottom, color, scale, scale_mode, mask=None):  # 移除 transparent 参数
        original_width = image.shape[2]
        original_height = image.shape[1]

        padded_image = self.add_padding(image, left, top, right, bottom, color)
        output_width = padded_image.shape[2]
        output_height = padded_image.shape[1]

        created_mask = self.create_mask(image, left, top, right, bottom, mask)

        scaled_image, scaled_mask = self.scale_image_and_mask(padded_image, created_mask, scale, scale_mode, original_width, original_height, output_width, output_height)

        return (scaled_image, scaled_mask)


class Image_Resize_sum:  #图像与遮罩同步裁切
    def __init__(self):
        pass


    ACTION_TYPE_RESIZE = "resize only"
    ACTION_TYPE_CROP = "crop to ratio"
    ACTION_TYPE_PAD = "pad to ratio"
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "action": ([s.ACTION_TYPE_RESIZE, s.ACTION_TYPE_CROP, s.ACTION_TYPE_PAD],),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "side_ratio": ("STRING", {"default": "4:3"}),
                "crop_pad_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_feathering": ("INT", {"default": 20, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            },
        }


    @classmethod
    def VALIDATE_INPUTS(s, action, smaller_side, larger_side, scale_factor, side_ratio, **_):
        if side_ratio is not None:
            if action != s.ACTION_TYPE_RESIZE and s.parse_side_ratio(side_ratio) is None:
                return f"Invalid side ratio: {side_ratio}"

        if smaller_side is not None and larger_side is not None and scale_factor is not None:

            if int(smaller_side > 0) + int(larger_side > 0) + int(scale_factor > 0) > 1:
                return f"At most one scaling rule (smaller_side, larger_side, scale_factor) should be enabled by setting a non-zero value"

        return True


    @classmethod
    def parse_side_ratio(s, side_ratio):
        try:
            x, y = map(int, side_ratio.split(":", 1))
            if x < 1 or y < 1:
                raise Exception("Ratio factors have to be positive numbers")
            return float(x) / float(y)
        except:
            return None


    def resize(self, pixels, action, smaller_side, larger_side, scale_factor, side_ratio, crop_pad_position, pad_feathering, mask_optional=None):
        validity = self.VALIDATE_INPUTS(action, smaller_side, larger_side, scale_factor, side_ratio)
        if validity is not True:
            raise Exception(validity)

        height, width = pixels.shape[1:3]
        if mask_optional is None:
            mask = torch.zeros(1, height, width, dtype=torch.float32)
        else:
            mask = mask_optional
            if mask.shape[1] != height or mask.shape[2] != width:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(height, width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        crop_x, crop_y, pad_x, pad_y = (0.0, 0.0, 0.0, 0.0)
        if action == self.ACTION_TYPE_CROP:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio < width:
                crop_x = width - height * target_ratio
            else:
                crop_y = height - width / target_ratio
        elif action == self.ACTION_TYPE_PAD:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio > width:
                pad_x = height * target_ratio - width
            else:
                pad_y = width / target_ratio - height

        if smaller_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(smaller_side) / (height + pad_y - crop_y)
            else:
                scale_factor = float(smaller_side) / (width + pad_x - crop_x)
        if larger_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(larger_side) / (width + pad_x - crop_x)
            else:
                scale_factor = float(larger_side) / (height + pad_y - crop_y)

        if scale_factor > 0.0:
            pixels = torch.nn.functional.interpolate(pixels.movedim(-1, 1), scale_factor=scale_factor, mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=scale_factor, mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)
            height, width = pixels.shape[1:3]

            crop_x *= scale_factor
            crop_y *= scale_factor
            pad_x *= scale_factor
            pad_y *= scale_factor

        if crop_x > 0.0 or crop_y > 0.0:
            remove_x = (round(crop_x * crop_pad_position), round(crop_x * (1 - crop_pad_position))) if crop_x > 0.0 else (0, 0)
            remove_y = (round(crop_y * crop_pad_position), round(crop_y * (1 - crop_pad_position))) if crop_y > 0.0 else (0, 0)
            pixels = pixels[:, remove_y[0]:height - remove_y[1], remove_x[0]:width - remove_x[1], :]
            mask = mask[:, remove_y[0]:height - remove_y[1], remove_x[0]:width - remove_x[1]]
        elif pad_x > 0.0 or pad_y > 0.0:
            add_x = (round(pad_x * crop_pad_position), round(pad_x * (1 - crop_pad_position))) if pad_x > 0.0 else (0, 0)
            add_y = (round(pad_y * crop_pad_position), round(pad_y * (1 - crop_pad_position))) if pad_y > 0.0 else (0, 0)

            new_pixels = torch.zeros(pixels.shape[0], height + add_y[0] + add_y[1], width + add_x[0] + add_x[1], pixels.shape[3], dtype=torch.float32)
            new_pixels[:, add_y[0]:height + add_y[0], add_x[0]:width + add_x[0], :] = pixels
            pixels = new_pixels

            new_mask = torch.ones(mask.shape[0], height + add_y[0] + add_y[1], width + add_x[0] + add_x[1], dtype=torch.float32)
            new_mask[:, add_y[0]:height + add_y[0], add_x[0]:width + add_x[0]] = mask
            mask = new_mask

            if pad_feathering > 0:
                for i in range(mask.shape[0]):
                    for j in range(pad_feathering):
                        feather_strength = (1 - j / pad_feathering) * (1 - j / pad_feathering)
                        if add_x[0] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, add_x[0] + j] = max(mask[i, k, add_x[0] + j], feather_strength)
                        if add_x[1] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, width + add_x[0] - j - 1] = max(mask[i, k, width + add_x[0] - j - 1], feather_strength)
                        if add_y[0] > 0 and j < height:
                            for k in range(width):
                                mask[i, add_y[0] + j, k] = max(mask[i, add_y[0] + j, k], feather_strength)
                        if add_y[1] > 0 and j < height:
                            for k in range(width):
                                mask[i, height + add_y[0] - j - 1, k] = max(mask[i, height + add_y[0] - j - 1, k], feather_strength)

        return (pixels, mask)



class Image_Resize2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": 99999, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": 99999, "step": 1, }),

                "keep_ratio": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 8, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "get_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    def resize(self, image, width, height, keep_ratio, divisible_by, get_image_size=None,):
        B, H, W, C = image.shape

        upscale_method ="lanczos"
        crop = "center"

        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape
        

        if keep_ratio and get_image_size is None:

                if width == 0 and height != 0:
                    ratio = height / H
                    width = round(W * ratio)
                elif height == 0 and width != 0:
                    ratio = width / W
                    height = round(H * ratio)
                elif width != 0 and height != 0:

                    ratio = min(width / W, height / H)
                    width = round(W * ratio)
                    height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H
    
        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)
        
        image = image.movedim(-1,1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image,image.shape[2], image.shape[1])



class Image_transform_solo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "y": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "shear": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "border_handling": (["edge", "constant", "reflect", "symmetric"], {"default": "edge"}),
            },
            "optional": {
                "filter_type": (["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"], {"default": "bilinear"}),
                "constant_color": ("STRING", {"default": "#000000"}),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAME = ("image",)
    CATEGORY = "Apt_Preset/image"



    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling="edge",
        constant_color=None,
        filter_type="nearest",
    ):
        filter_map = {
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "bilinear": Image.BILINEAR,
            "hamming": Image.HAMMING,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resampling_filter = filter_map[filter_type]

        x = int(x)
        y = int(y)
        angle = int(angle)


        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        frames_count, frame_height, frame_width, frame_channel_count = (
            image.size()
        )

        new_height, new_width = (
            int(frame_height * zoom),
            int(frame_width * zoom),
        )


        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [
            max(0, pw + x),
            max(0, ph + y),
            max(0, pw - x),
            max(0, ph - y),
        ]

        constant_color = hex_to_rgb(constant_color)

        for img in list_tensor2pil(image):
            img = TF.pad(
                img,  # transformed_frame,
                padding=padding,
                padding_mode=border_handling,
                fill=constant_color or 0,
            )

            img = cast(
                Image.Image,
                TF.affine(
                    img,
                    angle=angle,
                    scale=zoom,
                    translate=[x, y],
                    shear=shear,
                    interpolation=resampling_filter,
                ),
            )

            left = abs(padding[0])
            upper = abs(padding[1])
            right = img.width - abs(padding[2])
            bottom = img.height - abs(padding[3])
            img = img.crop((left, upper, right, bottom))

            transformed_images.append(img)

        return (list_pil2tensor(transformed_images),)



class Image_Upscaletile:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_image"
    CATEGORY = "Apt_Preset/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (folder_paths.get_filename_list("upscale_models"),),  # Model selection
                "upscale_by": (
                    "FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01},  # Input for scaling factor
                ),
                "image": ("IMAGE",),  # Input restored image
                "tile_size": (
                    "INT", {"default": 512, "min": 128, "max": 8192, "step": 8},  # Control for tile size
                )
            }
        }

    def upscale_image(self, model_name, upscale_by, image, tile_size):
        # Load the selected model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        upscale_model = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(upscale_model, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0  # Memory estimate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        overlap = 32  # Keep the original overlap value

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile_size, tile_y=tile_size, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile_size, tile_y=tile_size, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                
                # Adjust according to the upscale_by value
                size_diff = upscale_by / upscale_model.scale
                if size_diff != 1:
                    s = comfy.utils.common_upscale(
                        s,
                        width=round(s.shape[3] * size_diff),
                        height=round(s.shape[2] * size_diff),
                        upscale_method="lanczos",
                        crop="disabled",
                    )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile_size //= 2
                if tile_size < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)



#region --------------------color--------------------



class color_adjust_light:  #法向光源图

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_img": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "specular_map": ("IMAGE",),
                "light_yaw": ("FLOAT", {"default": 45, "min": -180, "max": 180, "step": 1}),
                "light_pitch": ("FLOAT", {"default": 30, "min": -90, "max": 90, "step": 1}),
                "specular_power": ("FLOAT", {"default": 32, "min": 1, "max": 200, "step": 1}),
                "ambient_light": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.01}),
                "NormalDiffuseStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "SpecularHighlightsStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "TotalGain": ("FLOAT", {"default": 1.00, "min": 0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image"

    def execute(self, target_img, normal_map, specular_map, light_yaw, light_pitch, specular_power, ambient_light,NormalDiffuseStrength,SpecularHighlightsStrength,TotalGain):

        diffuse_tensor = target_img.permute(0, 3, 1, 2)  
        normal_tensor = normal_map.permute(0, 3, 1, 2) * 2.0 - 1.0  
        specular_tensor = specular_map.permute(0, 3, 1, 2)  

        normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=1)
        light_direction = self.euler_to_vector(light_yaw, light_pitch, 0 )
        light_direction = light_direction.view(1, 3, 1, 1)  
        camera_direction = self.euler_to_vector(0,0,0)
        camera_direction = camera_direction.view(1, 3, 1, 1) 


        diffuse = torch.sum(normal_tensor * light_direction, dim=1, keepdim=True)
        diffuse = torch.clamp(diffuse, 0, 1)

        half_vector = torch.nn.functional.normalize(light_direction + camera_direction, dim=1)
        specular = torch.sum(normal_tensor * half_vector, dim=1, keepdim=True)
        specular = torch.pow(torch.clamp(specular, 0, 1), specular_power)

        output_tensor = ( diffuse_tensor * (ambient_light + diffuse * NormalDiffuseStrength ) + specular_tensor * specular * SpecularHighlightsStrength) * TotalGain

        output_tensor = output_tensor.permute(0, 2, 3, 1)  

        return (output_tensor,)


    def euler_to_vector(self, yaw, pitch, roll):
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)

        direction = np.array([
            sin_yaw * cos_pitch,
            sin_pitch,
            cos_pitch * cos_yaw
        ])


        return torch.from_numpy(direction).float()

    def convert_tensor_to_image(self, tensor):
        tensor = tensor.squeeze(0)  
        tensor = tensor.clamp(0, 1)  
        image = Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))
        return image


class color_adjust_WB_balance:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha_trimap": ("IMAGE",),
                "preblur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 256,
                    "step": 1
                }),
                "blackpoint": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01
                }),
                "whitepoint": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_iterations": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
                "estimate_fg": (["true", "false"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("alpha", "fg", "bg",)
    FUNCTION = "alpha_matte"

    CATEGORY = "Apt_Preset/image"

    def alpha_matte(self, images, alpha_trimap, preblur, blackpoint, whitepoint, max_iterations, estimate_fg):
        
        d = preblur * 2 + 1
        
        i_dup = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        a_dup = copy.deepcopy(alpha_trimap.cpu().numpy().astype(np.float64))
        fg = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        bg = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        
        
        for index, image in enumerate(i_dup):
            trimap = a_dup[index][:,:,0] # convert to single channel
            if preblur > 0:
                trimap = cv2.GaussianBlur(trimap, (d, d), 0)
            trimap = fix_trimap(trimap, blackpoint, whitepoint)
            
            alpha = estimate_alpha_cf(image, trimap, laplacian_kwargs={"epsilon": 1e-6}, cg_kwargs={"maxiter":max_iterations})
            
            if estimate_fg == "true":
                fg[index], bg[index] = estimate_foreground_ml(image, alpha, return_background=True)
            
            a_dup[index] = np.stack([alpha, alpha, alpha], axis = -1) # convert back to rgb
        
        return (
            torch.from_numpy(a_dup.astype(np.float32)), # alpha
            torch.from_numpy(fg.astype(np.float32)), # fg
            torch.from_numpy(bg.astype(np.float32)), # bg
            )


class color_adjust_HDR:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                }),
                "HDR_intensity": ("FLOAT", {
                    "default": 1,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.01,
                }),
                "underexposure_factor": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "overexposure_factor": ("FLOAT", {
                    "default": 1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "gamma": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                }),
                "highlight_detail": ("FLOAT", {
                    "default": 1/30.0,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "midtone_detail": ("FLOAT", {
                    "default": 0.25,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "shadow_detail": ("FLOAT", {
                    "default": 2,
                    "min": 1/1000.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "overall_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image"

    def execute(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, highlight_detail, midtone_detail, shadow_detail, overall_intensity):
        try:
            image = self.ensure_image_format(image)

            processed_image = self.apply_hdr(image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, [highlight_detail, midtone_detail, shadow_detail])

            blended_image = cv2.addWeighted(processed_image, overall_intensity, image, 1 - overall_intensity, 0)

            if isinstance(blended_image, np.ndarray):
                blended_image = np.expand_dims(blended_image, axis=0)

            blended_image = torch.from_numpy(blended_image).float()
            blended_image = blended_image / 255.0
            blended_image = blended_image.to(torch.device('cpu'))

            return [blended_image]
        except Exception as e:
            if image is not None and hasattr(image, 'shape'):
                black_image = torch.zeros((1, 3, image.shape[0], image.shape[1]), dtype=torch.float32)
            else:
                black_image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            return [black_image.to(torch.device('cpu'))]

    def ensure_image_format(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
        return image

    def apply_hdr(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, exposure_times):
        hdr = cv2.createMergeDebevec()

        times = np.array(exposure_times, dtype=np.float32)

        exposure_images = [
            np.clip(image * underexposure_factor, 0, 255).astype(np.uint8),  # Underexposed
            image,  # Normal exposure
            np.clip(image * overexposure_factor, 0, 255).astype(np.uint8)   # Overexposed
        ]

        hdr_image = hdr.process(exposure_images, times=times.copy())

        tonemap = cv2.createTonemapReinhard(gamma=gamma)
        ldr_image = tonemap.process(hdr_image)

        ldr_image = ldr_image * HDR_intensity
        ldr_image = np.clip(ldr_image, 0, 1)
        ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

        return ldr_image


class color_adjust_HSL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "gaussian_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "edge_enhance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_enhance": (["false", "true"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_adjust_HSL"

    CATEGORY = "Apt_Preset/image"

    def color_adjust_HSL(self, image, brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):


        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None

                # Apply NP Adjustments
                if brightness > 0.0 or brightness < 0.0:
                    # Apply brightness
                    img = np.clip(img + brightness, 0.0, 1.0)

                if contrast > 1.0 or contrast < 1.0:
                    # Apply contrast
                    img = np.clip(img * contrast, 0.0, 1.0)

                # Apply PIL Adjustments
                if saturation > 1.0 or saturation < 1.0:
                    # PIL Image
                    pil_image = tensor2pil(img)
                    # Apply saturation
                    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

                if sharpness > 1.0 or sharpness < 1.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply sharpness
                    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

                if blur > 0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply blur
                    for _ in range(blur):
                        pil_image = pil_image.filter(ImageFilter.BLUR)

                if gaussian_blur > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply Gaussian blur
                    pil_image = pil_image.filter(
                        ImageFilter.GaussianBlur(radius=gaussian_blur))

                if edge_enhance > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Edge Enhancement
                    edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    # Blend Mask
                    blend_mask = Image.new(
                        mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                    # Composite Original and Enhanced Version
                    pil_image = Image.composite(
                        edge_enhanced_img, pil_image, blend_mask)
                    # Clean-up
                    del blend_mask, edge_enhanced_img

                if detail_enhance == "true":
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = pil_image.filter(ImageFilter.DETAIL)

                # Output image
                out_image = (pil2tensor(pil_image) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:

            pil_image = None
            img = image

            # Apply NP Adjustments
            if brightness > 0.0 or brightness < 0.0:
                # Apply brightness
                img = np.clip(img + brightness, 0.0, 1.0)

            if contrast > 1.0 or contrast < 1.0:
                # Apply contrast
                img = np.clip(img * contrast, 0.0, 1.0)

            # Apply PIL Adjustments
            if saturation > 1.0 or saturation < 1.0:
                # PIL Image
                pil_image = tensor2pil(img)
                # Apply saturation
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness > 1.0 or sharpness < 1.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply sharpness
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if blur > 0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply blur
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)

            if gaussian_blur > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply Gaussian blur
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Edge Enhancement
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                # Blend Mask
                blend_mask = Image.new(
                    mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                # Composite Original and Enhanced Version
                pil_image = Image.composite(
                    edge_enhanced_img, pil_image, blend_mask)
                # Clean-up
                del blend_mask, edge_enhanced_img

            if detail_enhance == "true":
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = pil_image.filter(ImageFilter.DETAIL)

            # Output image
            out_image = (pil2tensor(pil_image) if pil_image else img)

            tensors = out_image


        return (tensors, )


class color_tool:
    """Returns to inverse of a color"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("STRING", {"default": "#000000"}),  # 改为STRING类型，默认黑色
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hex_str": ("STRING",  {"forceInput": True}, ),
                "r": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "g": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "b": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "a": ("FLOAT", {"forceInput": True, "min": 0.0, "max": 1.0,} ,),
            }
        }

    CATEGORY = "Apt_Preset/image"
    # 修改返回类型，添加 alpha 通道
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT",)
    # 修改返回名称，添加 alpha 通道
    RETURN_NAMES = ("hex_str", "R", "G", "B", "A",)

    FUNCTION = "execute"

    def execute(self, color, alpha, hex_str=None, r=None, g=None, b=None, a=None):
        if hex_str:
            hex_color = hex_str
        else:
            hex_color = color

        hex_color = hex_color.lstrip("#")
        original_r, original_g, original_b = hex_to_rgb_tuple(hex_color)

        # 若有 r, g, b, a 输入则替换对应值
        final_r = r if r is not None else original_r
        final_g = g if g is not None else original_g
        final_b = b if b is not None else original_b
        final_a = a if a is not None else alpha

        final_hex_color = "#{:02x}{:02x}{:02x}".format(final_r, final_g, final_b)
        return (final_hex_color, final_r, final_g, final_b, final_a)
    

class color_OneColor_replace:
    """Replace Color in an Image"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {"default": "#09FF00"}),
                "replace_color": ("STRING", {"default": "#FF0000"}),
                "clip_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_remove_color"

    CATEGORY = "Apt_Preset/image"

    def image_remove_color(self, image, clip_threshold=10, target_color='#ffffff',replace_color='#ffffff'):
        return (pil2tensor(self.apply_remove_color(tensor2pil(image), clip_threshold, hex_to_rgb_tuple(target_color), hex_to_rgb_tuple(replace_color))), )

    def apply_remove_color(self, image, threshold=10, color=(255, 255, 255), rep_color=(0, 0, 0)):
        # Create a color image with the same size as the input image
        color_image = Image.new('RGB', image.size, color)

        # Calculate the difference between the input image and the color image
        diff_image = ImageChops.difference(image, color_image)

        # Convert the difference image to grayscale
        gray_image = diff_image.convert('L')

        # Apply a threshold to the grayscale difference image
        mask_image = gray_image.point(lambda x: 255 if x > threshold else 0)

        # Invert the mask image
        mask_image = ImageOps.invert(mask_image)

        # Apply the mask to the original image
        result_image = Image.composite(
            Image.new('RGB', image.size, rep_color), image, mask_image)

        return result_image


class color_OneColor_keep:  #保留一色
    NAME = "Color Stylizer"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                # Modified to directly input color
                "color": ("STRING", {"default": "#000000"}),
                "falloff": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "gain": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stylize"
    OUTPUT_NODE = False
    CATEGORY = "Apt_Preset/image"

    def stylize(self, image, color, falloff, gain):
        print(f"Type of color: {type(color)}, Value of color: {color}")  # Add debugging information
        try:
            # Extract BGR values from the color tuple
            target_b, target_g, target_r = [int(c * 255) if isinstance(c, (int, float)) else 0 for c in color]
            target_color = (target_b, target_g, target_r)
        except Exception as e:
            print(f"Error converting color: {e}")
            target_color = (0, 0, 0)  # Set default color if conversion fails

        image = image.squeeze(0)
        image = image.mul(255).byte().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        falloff_mask = self.create_falloff_mask(image, target_color, falloff)
        image_amplified = image.copy()
        image_amplified[:, :, 2] = np.clip(image_amplified[:, :, 2] * gain, 0, 255).astype(np.uint8)
        stylized_img = (image_amplified * falloff_mask + gray_img * (1 - falloff_mask)).astype(np.uint8)
        stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB)
        stylized_img_tensor = to_tensor(stylized_img).float()
        stylized_img_tensor = stylized_img_tensor.permute(1, 2, 0).unsqueeze(0)
        return (stylized_img_tensor,)

    def create_falloff_mask(self, img, target_color, falloff):
        target_color = np.array(target_color, dtype=np.uint8)

        target_color = np.full_like(img, target_color)

        diff = cv2.absdiff(img, target_color)

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(diff, falloff, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (0, 0), falloff / 2)
        mask = mask / 255.0
        mask = mask.reshape(*mask.shape, 1)
        return mask


class color_Local_Gray:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_gray": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "bj_gray": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mask_img", "bj_img", "mask")
    FUNCTION = "apply"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, mask_gray, bj_gray, smoothness):
        # 处理mask区域的灰度化
        image_input = image.clone()
        mask_input = mask.clone()

        if image_input.ndim != 4 or mask_input.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        # 确保mask是4D张量 (B, H, W, 1)
        if mask_input.ndim == 3:
            mask_input = mask_input.unsqueeze(-1)  # (B, H, W) -> (B, H, W, 1)

        results = []
        bj_results = []
        masks = []
        
        # 批量处理每一张图像
        for i in range(image_input.shape[0]):
            # 获取当前图像和对应的mask
            current_image = image_input[i]  # (H, W, C)
            current_mask = mask_input[i] if mask_input.shape[0] > 1 else mask_input[0]  # (H, W, 1)
            
            # 转换为PIL图像
            target = Image.fromarray((255. * current_image).cpu().numpy().astype(np.uint8))
            subjectmask = Image.fromarray((255. * current_mask.squeeze(-1)).cpu().numpy().astype(np.uint8)).convert('RGB')
            
            # 对mask区域进行灰度化处理
            grayall_target = target.convert('L').convert('RGB')
            grayall_target = ImageEnhance.Brightness(grayall_target)
            grayall_target = grayall_target.enhance(mask_gray)

            graysubject = ImageChops.darker(grayall_target, subjectmask)
            colorbackground = ImageChops.darker(target, ImageChops.invert(subjectmask))
            mask_img_result = ImageChops.lighter(colorbackground, graysubject)
            
            # 应用羽化效果到输出图像
            if smoothness > 0:
                # 创建羽化遮罩
                feathered_mask_pil = Image.fromarray((255. * current_mask.squeeze(-1)).cpu().numpy().astype(np.uint8))
                feathered_mask = feathered_mask_pil.filter(ImageFilter.GaussianBlur(smoothness))
                
                # 将羽化遮罩应用到处理后的图像上，实现图像边缘的柔和过渡
                mask_img_result = Image.composite(mask_img_result, target, feathered_mask)
            
            # 转换回tensor
            image_result = np.array(mask_img_result).astype(np.float32) / 255.0
            image_result = torch.from_numpy(image_result).unsqueeze(0)  # (1, H, W, C)
            results.append(image_result)
            
            # 处理反向遮罩区域(1-mask)的灰度化
            inverted_mask = Image.fromarray((255. * (1.0 - current_mask.squeeze(-1))).cpu().numpy().astype(np.uint8)).convert('RGB')
            
            # 对反向遮罩区域进行灰度化处理
            inverted_grayall_target = target.convert('L').convert('RGB')
            inverted_grayall_target = ImageEnhance.Brightness(inverted_grayall_target)
            inverted_grayall_target = inverted_grayall_target.enhance(bj_gray)
            
            inverted_graysubject = ImageChops.darker(inverted_grayall_target, inverted_mask)
            inverted_colorbackground = ImageChops.darker(target, ImageChops.invert(inverted_mask))
            bj_img_result = ImageChops.lighter(inverted_colorbackground, inverted_graysubject)
            
            # 应用羽化效果到背景图像
            if smoothness > 0:
                # 创建羽化遮罩（反向遮罩的羽化）
                inverted_feathered_mask_pil = Image.fromarray((255. * (1.0 - current_mask.squeeze(-1))).cpu().numpy().astype(np.uint8))
                inverted_feathered_mask = inverted_feathered_mask_pil.filter(ImageFilter.GaussianBlur(smoothness))
                
                # 将羽化遮罩应用到处理后的背景图像上
                bj_img_result = Image.composite(bj_img_result, target, inverted_feathered_mask)
            
            bj_image_result = np.array(bj_img_result).astype(np.float32) / 255.0
            bj_image_result = torch.from_numpy(bj_image_result).unsqueeze(0)  # (1, H, W, C)
            bj_results.append(bj_image_result)
            
            # 处理当前mask的羽化
            current_mask_pil = Image.fromarray((255. * current_mask.squeeze(-1)).cpu().numpy().astype(np.uint8))
            feathered_image = current_mask_pil.filter(ImageFilter.GaussianBlur(smoothness))
            mask_result = np.array(feathered_image).astype(np.float32) / 255.0
            mask_result = torch.from_numpy(mask_result).unsqueeze(0)  # (1, H, W)
            masks.append(mask_result)
        
        # 合并所有结果
        image_result = torch.cat(results, dim=0)  # (B, H, W, C)
        bj_image_result = torch.cat(bj_results, dim=0)  # (B, H, W, C)
        mask_result = torch.cat(masks, dim=0)  # (B, H, W)

        return (image_result, bj_image_result, mask_result)


class color_selector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HEX",)
    FUNCTION = 'selector'
    CATEGORY = "Apt_Preset/image"
    
    def selector(self, color):
        return (color,)




#region----color_match adv----------


def image_stats(image):
    return np.mean(image[:, :, 1:], axis=(0, 1)), np.std(image[:, :, 1:], axis=(0, 1))


def is_skin_or_lips(lab_image):
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]
    skin = (l > 20) & (l < 250) & (a > 120) & (a < 180) & (b > 120) & (b < 190)
    lips = (l > 20) & (l < 200) & (a > 150) & (b > 140)
    return (skin | lips).astype(np.float32)


def adjust_brightness(image, factor, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        v = np.where(mask > 0, np.clip(v * factor, 0, 255), v)
    else:
        v = np.clip(v * factor, 0, 255)
    hsv[:, :, 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_saturation(image, factor, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        s = np.where(mask > 0, np.clip(s * factor, 0, 255), s)
    else:
        s = np.clip(s * factor, 0, 255)
    hsv[:, :, 1] = s.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor, mask=None):
    mean = np.mean(image)
    adjusted = image.astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        adjusted = np.where(mask > 0, np.clip((adjusted - mean) * factor + mean, 0, 255), adjusted)
    else:
        adjusted = np.clip((adjusted - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_tone(source, target, tone_strength=0.7, mask=None):
    h, w = target.shape[:2]
    source = cv2.resize(source, (w, h))
    lab_image = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_image = lab_image[:,:,0]
    l_source = lab_source[:,:,0]

    if mask is not None:
        mask = cv2.resize(mask, (w, h))
        mask = mask.astype(np.float32) / 255.0
        l_adjusted = np.copy(l_image)
        mean_source = np.mean(l_source[mask > 0])
        std_source = np.std(l_source[mask > 0])
        mean_target = np.mean(l_image[mask > 0])
        std_target = np.std(l_image[mask > 0])
        l_adjusted[mask > 0] = (l_image[mask > 0] - mean_target) * (std_source / (std_target + 1e-6)) * 0.7 + mean_source
        l_adjusted[mask > 0] = np.clip(l_adjusted[mask > 0], 0, 255)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_adjusted.astype(np.uint8))
        l_final = cv2.addWeighted(l_adjusted, 0.7, l_enhanced.astype(np.float32), 0.3, 0)
        l_final = np.clip(l_final, 0, 255)
        l_contrast = cv2.addWeighted(l_final, 1.3, l_final, 0, -20)
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image[mask > 0] = l_image[mask > 0] * (1 - tone_strength) + l_contrast[mask > 0] * tone_strength
    else:
        mean_source = np.mean(l_source)
        std_source = np.std(l_source)
        l_mean = np.mean(l_image)
        l_std = np.std(l_image)
        l_adjusted = (l_image - l_mean) * (std_source / (l_std + 1e-6)) * 0.7 + mean_source
        l_adjusted = np.clip(l_adjusted, 0, 255)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_adjusted.astype(np.uint8))
        l_final = cv2.addWeighted(l_adjusted, 0.7, l_enhanced.astype(np.float32), 0.3, 0)
        l_final = np.clip(l_final, 0, 255)
        l_contrast = cv2.addWeighted(l_final, 1.3, l_final, 0, -20)
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image = l_image * (1 - tone_strength) + l_contrast * tone_strength

    lab_image[:,:,0] = l_image
    return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)


def tensor2cv2(image: torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)


def color_transfer(source, target, mask=None, strength=1.0, skin_protection=0.2, auto_brightness=True,
                   brightness_range=0.5, auto_contrast=False, contrast_range=0.5,
                   auto_saturation=False, saturation_range=0.5, auto_tone=False, tone_strength=0.7):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_means, src_stds = image_stats(source_lab)
    tar_means, tar_stds = image_stats(target_lab)

    skin_lips_mask = is_skin_or_lips(target_lab.astype(np.uint8))
    skin_lips_mask = cv2.GaussianBlur(skin_lips_mask, (5, 5), 0)

    if mask is not None:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0

    result_lab = target_lab.copy()
    for i in range(1, 3):
        adjusted_channel = (target_lab[:, :, i] - tar_means[i - 1]) * (src_stds[i - 1] / (tar_stds[i - 1] + 1e-6)) + \
                           src_means[i - 1]
        adjusted_channel = np.clip(adjusted_channel, 0, 255)

        if mask is not None:
            result_lab[:, :, i] = target_lab[:, :, i] * (1 - mask) + \
                                  (target_lab[:, :, i] * skin_lips_mask * skin_protection + \
                                   adjusted_channel * skin_lips_mask * (1 - skin_protection) + \
                                   adjusted_channel * (1 - skin_lips_mask)) * mask
        else:
            result_lab[:, :, i] = target_lab[:, :, i] * skin_lips_mask * skin_protection + \
                                  adjusted_channel * skin_lips_mask * (1 - skin_protection) + \
                                  adjusted_channel * (1 - skin_lips_mask)

    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    final_result = cv2.addWeighted(target, 1 - strength, result_bgr, strength, 0)

    if mask is not None:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0
        if auto_brightness:
            source_brightness = np.mean(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor, mask)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor, mask)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor, mask)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength, mask)
    else:
        if auto_brightness:
            source_brightness = np.mean(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength)

    return final_result


class color_match_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "ref_img": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_protection": ("FLOAT", {"default": 0.2, "min": 0, "max": 1.0, "step": 0.1}),
                "brightness_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "contrast_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "saturation_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "tone_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "ref_mask": ("MASK", {"default": None}),
            },
        }

    CATEGORY = "Apt_Preset/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match_hue"

    def match_hue(self, ref_img, target_image, strength, skin_protection,  brightness_range,
                contrast_range, saturation_range, tone_strength, ref_mask=None):
        
        auto_brightness =True
        auto_contrast =True
        auto_tone =True
        auto_saturation =True


        for img in ref_img:
            img_cv1 = tensor2cv2(img)

        for img in target_image:
            img_cv2 = tensor2cv2(img)

        img_cv3 = None
        if ref_mask is not None:
            for img3 in ref_mask:
                img_cv3 = img3.cpu().numpy()
                img_cv3 = (img_cv3 * 255).astype(np.uint8)

        result_img = color_transfer(img_cv1, img_cv2, img_cv3, strength, skin_protection, auto_brightness,
                                    brightness_range,auto_contrast, contrast_range, auto_saturation,
                                    saturation_range, auto_tone, tone_strength)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)

        return (rst,)





#endregion-----------------------color_transfer----------------


class Image_Channel_RemoveAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)


class Image_Channel_Extract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": ([ "A","R", "G", "B"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("channel_data",)
    CATEGORY = "Apt_Preset/image"
    FUNCTION = "image_extract_alpha"

    def image_extract_alpha(self, images: torch.Tensor, channel):
        # images in shape (N, H, W, C)

        if len(images.shape) < 4:
            images = images.unsqueeze(3).repeat(1, 1, 1, 3)

        if channel == "A" and images.shape[3] < 4:
            raise Exception("Image does not have an alpha channel")

        channel_index = ["R", "G", "B", "A"].index(channel)
        mask = images[:, :, :, channel_index].cpu().clone()

        return (mask,)


class Image_Channel_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel_data": ("MASK",),
                "channel": (["A", "R", "G", "B"],),
                "invert_channel": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "background_color": (
                    ["none", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"],
                    {"default": "none"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Preset/image"
    FUNCTION = "Image_Channel_Apply"

    def Image_Channel_Apply(self, images: torch.Tensor, channel_data: torch.Tensor, channel, 
                           invert_channel=False, background_color="none"):
        # 定义颜色字典，将颜色名称映射到RGB值（已归一化到0-1范围）
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5)  
        }
        
        merged_images = []

        # 如果需要反向通道数据，则进行反向处理
        if invert_channel:
            channel_data = 1.0 - channel_data

        for image in images:
            image = image.cpu().clone()

            # 应用通道数据到指定通道
            if channel == "A":
                # 如果图像没有Alpha通道，则添加一个
                if image.shape[2] < 4:
                    image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)
                # 设置Alpha通道
                image[:, :, 3] = channel_data
            elif channel == "R":
                image[:, :, 0] = channel_data
            elif channel == "G":
                image[:, :, 1] = channel_data
            else:  # channel == "B"
                image[:, :, 2] = channel_data

            # 如果选择了背景颜色，则创建纯色背景并合成
            if background_color != "none" and background_color in colors:
                # 获取颜色值
                r, g, b = colors[background_color]
                
                # 创建与图像相同大小的纯色背景（RGB）
                h, w, _ = image.shape
                background = torch.zeros((h, w, 3))
                background[:, :, 0] = r  # R通道
                background[:, :, 1] = g  # G通道
                background[:, :, 2] = b  # B通道
                
                # 如果图像有Alpha通道，则使用Alpha通道进行混合
                if image.shape[2] >= 4:
                    alpha = image[:, :, 3].unsqueeze(2)
                    # 合成公式：result = image_rgb * alpha + background * (1 - alpha)
                    image_rgb = image[:, :, :3]
                    image = image_rgb * alpha + background * (1 - alpha)
                else:
                    # 如果没有Alpha通道，直接使用背景（实际上这种情况不会发生，因为原图像已有RGB通道）
                    image = background

            merged_images.append(image)

        return (torch.stack(merged_images),)



#endregion----------------------------------color---------------------------------------------------------------------





#region------图像-双图合并---总控制---------

class Image_Pair_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "box2": ("BOX2",),
                "crop_image": ("BOOLEAN", {"default": False,"label_on": "img2", "label_off": "img1"}),

            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("裁剪图像","裁剪遮罩")
    FUNCTION = "cropbox"

    def cropbox(self, mask=None, image=None, box2=None, crop_image=False):
        if box2 is None:
            return (None, None)
            
        box1_w, box1_h, box1_x, box1_y, box2_w, box2_h, box2_x, box2_y = box2
        
        if crop_image:
            region_width, region_height, center_x, center_y = box2_w, box2_h, box2_x, box2_y
        else:
            region_width, region_height, center_x, center_y = box1_w, box1_h, box1_x, box1_y
        
        x_start = max(0, int(center_x - region_width // 2))
        y_start = max(0, int(center_y - region_height // 2))
        
        x_end = x_start + region_width
        y_end = y_start + region_height
        
        cropped_image = None
        if image is not None:
            img_h, img_w = image.shape[1], image.shape[2]
            x_start = min(x_start, img_w)
            y_start = min(y_start, img_h)
            x_end = min(x_end, img_w)
            y_end = min(y_end, img_h)
            cropped_image = image[:, y_start:y_end, x_start:x_end, :]

        cropped_mask = None
        if mask is not None:
            mask_h, mask_w = mask.shape[1], mask.shape[2]
            x_start = min(x_start, mask_w)
            y_start = min(y_start, mask_h)
            x_end = min(x_end, mask_w)
            y_end = min(y_end, mask_h)
            cropped_mask = mask[:, y_start:y_end, x_start:x_end]

        return (cropped_image, cropped_mask,)



class Image_Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout_mode": ([
                                 "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                 "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐", 
                                 "居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐"],), 
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "size_mode": (["auto", "输出宽度", "输出高度"],),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "divider_thickness": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "image1": ("IMAGE",), 
                "mask1": ("MASK",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
                "mask1_stack": ("MASK_STACK",),
                "mask2_stack": ("MASK_STACK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2", "IMAGE", "MASK", )
    RETURN_NAMES = ("composite_image", "composite_mask", "box2", "new_img2", "new_mask2", )
    FUNCTION = "composite2"
    CATEGORY = "Apt_Preset/image"


    def composite2(self, layout_mode, bg_mode, size_mode, target_size, divider_thickness, 
                image1=None, image2=None, mask1=None, mask2=None, mask1_stack=None, mask2_stack=None, layer_stack=None):

        # 处理 mask1
        if mask1_stack and mask1 is not None:
            if hasattr(mask1, 'convert'):
                mask1_tensor = pil2tensor(mask1.convert('L'))
            else:
                if isinstance(mask1, torch.Tensor):
                    mask1_tensor = mask1 if len(mask1.shape) <= 3 else mask1.squeeze(-1) if mask1.shape[-1] == 1 else mask1
                else:
                    mask1_tensor = mask1
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = mask1_stack            
            
            separated_result = Mask_transform_sum().separate(
                bg_mode=bg_mode, 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=expand_width, 
                expand_height=expand_height,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image1.clone() if image1 is not None else image1,  # 使用clone避免修改原始图像
                mask=mask1_tensor, 
                crop_to_mask=False, 
                divisible_by=divisible_by
            )
            image1_processed = separated_result[0]  # 保存处理后的图像
            mask1 = separated_result[1]
        # 如果没有 mask1_stack 或 mask1 为 None，则直接使用 mask1（可能为 None）
        
        # 处理 mask2
        if mask2_stack and mask2 is not None: 
            if hasattr(mask2, 'convert'):
                mask2_tensor = pil2tensor(mask2.convert('L'))
            else:  
                if isinstance(mask2, torch.Tensor):
                    mask2_tensor = mask2 if len(mask2.shape) <= 3 else mask2.squeeze(-1) if mask2.shape[-1] == 1 else mask2
                else:
                    mask2_tensor = mask2
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = mask2_stack            
            
            separated_result = Mask_transform_sum().separate(  
                bg_mode=bg_mode, 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=expand_width, 
                expand_height=expand_height,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image2.clone() if image2 is not None else image2,  # 使用clone避免修改原始图像
                mask=mask2_tensor, 
                crop_to_mask=False, 
                divisible_by=divisible_by
            )
            image2_processed = separated_result[0]  # 保存处理后的图像
            mask2 = separated_result[1]
    
        # 如果经过处理，使用处理后的图像；否则使用原始图像
        if 'image1_processed' in locals():
            image1 = image1_processed
        if 'image2_processed' in locals():
            image2 = image2_processed
        

        if image1 is not None and not isinstance(image1, torch.Tensor):
            if hasattr(image1, 'numpy'):
                image1 = torch.from_numpy(image1.numpy())
            else:
                image1 = torch.from_numpy(np.array(image1))
        
        if image2 is not None and not isinstance(image2, torch.Tensor):
            if hasattr(image2, 'numpy'):
                image2 = torch.from_numpy(image2.numpy())
            else:
                image2 = torch.from_numpy(np.array(image2))

        if image1 is not None:
            if len(image1.shape) == 3:
                image1 = image1.unsqueeze(0)
            if image1.shape[-1] != 3:
                raise ValueError(f"image1 应该是3通道图像，实际形状: {image1.shape}")
        
        if image2 is not None:
            if len(image2.shape) == 3:
                image2 = image2.unsqueeze(0)
            if image2.shape[-1] != 3:
                raise ValueError(f"image2 应该是3通道图像，实际形状: {image2.shape}")

        if image1 is None and image2 is not None:
            image1 = image2.clone()
            
        if image1 is None and image2 is None:
            image1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image2 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        if mask1 is None:
            mask1 = torch.zeros((image1.shape[0], image1.shape[1], image1.shape[2]), dtype=torch.float32)
        else:
            if not isinstance(mask1, torch.Tensor):
                mask1 = torch.from_numpy(np.array(mask1)) if hasattr(mask1, 'numpy') else torch.from_numpy(mask1)
            
            if len(mask1.shape) == 2:
                mask1 = mask1.unsqueeze(0)
            if mask1.shape[1:] != image1.shape[1:3]:
                mask1 = torch.nn.functional.interpolate(
                    mask1.unsqueeze(1) if len(mask1.shape) == 3 else mask1, 
                    size=(image1.shape[1], image1.shape[2]), 
                    mode='nearest'
                )
                if len(mask1.shape) == 4:
                    mask1 = mask1.squeeze(1)

        if mask2 is None:
            mask2 = torch.ones((image2.shape[0], image2.shape[1], image2.shape[2]), dtype=torch.float32)
        else:
            if not isinstance(mask2, torch.Tensor):
                mask2 = torch.from_numpy(np.array(mask2)) if hasattr(mask2, 'numpy') else torch.from_numpy(mask2)
            
            if len(mask2.shape) == 2:
                mask2 = mask2.unsqueeze(0)
            if mask2.shape[1:] != image2.shape[1:3]:
                mask2 = torch.nn.functional.interpolate(
                    mask2.unsqueeze(1) if len(mask2.shape) == 3 else mask2, 
                    size=(image2.shape[1], image2.shape[2]), 
                    mode='nearest'
                )
                if len(mask2.shape) == 4:
                    mask2 = mask2.squeeze(1)

        final_img_tensor, final_mask_tensor, box2, image2, mask2 = Pair_Merge().composite(
            layout_mode=layout_mode, bg_mode=bg_mode, size_mode=size_mode, 
            target_size=target_size, divider_thickness=divider_thickness, 
            image1=image1, image2=image2, mask1=mask1, mask2=mask2
        )

        return (final_img_tensor, final_mask_tensor, box2, image2, mask2, )



class Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout_mode": ([
                                "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐"],
                                "居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐", ),  
                "bg_mode": (BJ_MODE,),  
                "size_mode": (["auto", "输出宽度", "输出高度"],),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "divider_thickness": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "image1": ("IMAGE",), 
                "mask1": ("MASK",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2", "IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "composite_mask", "box2", "new_img2", "new_mask2")
    FUNCTION = "composite"
    CATEGORY = "Apt_Preset/image"

    def composite(self, layout_mode, bg_mode, size_mode, target_size, divider_thickness, image1=None, image2=None, mask1=None, mask2=None):
        composite_img = None
        composite_mask = None
        box1_w, box1_h, box1_x, box1_y = 0, 0, 0, 0
        box2_w, box2_h, box2_x, box2_y = 0, 0, 0, 0

        adjusted_img2_np = np.zeros((512, 512, 3), dtype=np.float32)
        adjusted_mask2_np = np.ones((512, 512), dtype=np.float32)

        if image1 is None and image2 is not None:
            image1 = image2
        elif image1 is None and image2 is None:
            default_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image1 = default_img
            image2 = default_img

        def get_img_size(img):
            if isinstance(img, torch.Tensor):
                return img.shape[1], img.shape[2]
            else:
                return img.shape[0], img.shape[1] if len(img.shape) >= 3 else (img.shape[0], img.shape[1])

        h1, w1 = get_img_size(image1)
        h2, w2 = get_img_size(image2)

        def tensor2numpy(img_tensor):
            if isinstance(img_tensor, torch.Tensor):
                return img_tensor.cpu().numpy()[0]
            else:
                return img_tensor[0] if len(img_tensor.shape) == 4 else img_tensor

        img1_np = tensor2numpy(image1)
        img2_np = tensor2numpy(image2)

        def process_mask(mask, img_h, img_w):
            if mask is None:
                return np.ones((img_h, img_w), dtype=np.float32)
            mask_np = tensor2numpy(mask)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            if mask_np.shape != (img_h, img_w):
                mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return mask_np

        mask1_np = process_mask(mask1, h1, w1)
        mask2_np = process_mask(mask2, h2, w2)

        if layout_mode == "居中-自动对齐":
            if h2 > h1:
                layout_mode = "居中-高度对齐"
            elif w2 > w1:
                layout_mode = "居中-宽度对齐"
            else:
                layout_mode = "居中-中心对齐"

        if layout_mode == "左右-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + w1
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 + (w1 // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[:h1, w1:w1+w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:h1, w1:w1+w1] = mask2_resized_np

        elif layout_mode == "左右-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + box2_w
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_x = w1 + (box2_w // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[:box2_h, w1:w1+box2_w] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:box2_h, w1:w1+box2_w] = mask2_resized_np

        elif layout_mode == "左右-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + box2_w
            new_h = max(h1, box2_h)
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - box2_h) // 2
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = y1_offset + (h1 // 2)
            box2_x = w1 + (box2_w // 2)
            box2_y = y2_offset + (box2_h // 2)
            effective_bg_mode = "black" if bg_mode in ["image", "transparent"] else bg_mode
            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
            composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[y1_offset:y1_offset+h1, :w1] = mask1_np
            composite_mask[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = mask2_resized_np

        elif layout_mode == "上下-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1 + h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 // 2
            box2_y = h1 + (h1 // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+h1, :w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+h1, :w1] = mask2_resized_np

        elif layout_mode == "上下-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1 + box2_h
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_x = w1 // 2
            box2_y = h1 + (box2_h // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+box2_h, :w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+box2_h, :w1] = mask2_resized_np

        elif layout_mode == "上下-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = max(w1, box2_w)
            new_h = h1 + box2_h
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - box2_w) // 2
            box1_w, box1_h = w1, h1
            box1_x = x1_offset + (w1 // 2)
            box1_y = h1 // 2
            box2_x = x2_offset + (box2_w // 2)
            box2_y = h1 + (box2_h // 2)
            effective_bg_mode = "black" if bg_mode in ["image", "transparent"] else bg_mode
            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
            composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, x1_offset:x1_offset+w1] = mask1_np
            composite_mask[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = mask2_resized_np

        elif layout_mode == "居中-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 // 2
            box2_y = h1 // 2
            composite_img = img1_np.copy()
            alpha = mask2_resized_np[..., np.newaxis]
            composite_img = composite_img * (1 - alpha) + img2_resized_np * alpha
            composite_mask = mask2_resized_np.copy()

        elif layout_mode == "居中-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = max(w1, box2_w)
            new_h = h1
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - box2_w) // 2
            box1_w, box1_h = w1, h1
            box1_x = x1_offset + (w1 // 2)
            box1_y = h1 // 2
            box2_x = x2_offset + (box2_w // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
            overlap_start_x = max(x1_offset, x2_offset)
            overlap_end_x = min(x1_offset + w1, x2_offset + box2_w)
            if overlap_start_x < overlap_end_x:
                overlap_width = overlap_end_x - overlap_start_x
                img1_overlap_idx = overlap_start_x - x1_offset
                img2_overlap_idx = overlap_start_x - x2_offset
                alpha = mask2_resized_np[:, img2_overlap_idx:img2_overlap_idx+overlap_width][..., np.newaxis]
                composite_img[:h1, overlap_start_x:overlap_end_x] = composite_img[:h1, overlap_start_x:overlap_end_x] * (1 - alpha) + img2_resized_np[:h1, img2_overlap_idx:img2_overlap_idx+overlap_width] * alpha
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, x1_offset:x1_offset+w1] = mask1_np
            if overlap_start_x < overlap_end_x:
                composite_mask[:h1, overlap_start_x:overlap_end_x] = mask2_resized_np[:h1, img2_overlap_idx:img2_overlap_idx+overlap_width]

        elif layout_mode == "居中-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=cv2.INTER_LANCZOS4)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=cv2.INTER_NEAREST)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = max(h1, box2_h)
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - box2_h) // 2
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = y1_offset + (h1 // 2)
            box2_x = w1 // 2
            box2_y = y2_offset + (box2_h // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
            overlap_start_y = max(y1_offset, y2_offset)
            overlap_end_y = min(y1_offset + h1, y2_offset + box2_h)
            if overlap_start_y < overlap_end_y:
                overlap_height = overlap_end_y - overlap_start_y
                img1_overlap_idx = overlap_start_y - y1_offset
                img2_overlap_idx = overlap_start_y - y2_offset
                alpha = mask2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :][..., np.newaxis]
                composite_img[overlap_start_y:overlap_end_y, :w1] = composite_img[overlap_start_y:overlap_end_y, :w1] * (1 - alpha) + img2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :w1] * alpha
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[y1_offset:y1_offset+h1, :w1] = mask1_np
            if overlap_start_y < overlap_end_y:
                composite_mask[overlap_start_y:overlap_end_y, :w1] = mask2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :w1]

        scale_ratio = 1.0
        if size_mode != "auto":
            current_h, current_w = composite_img.shape[:2]
            if size_mode == "输出宽度":
                scale_ratio = target_size / current_w
            else:
                scale_ratio = target_size / current_h
            new_w = int(current_w * scale_ratio)
            new_h = int(current_h * scale_ratio)
            composite_img = cv2.resize(composite_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            composite_mask = cv2.resize(composite_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            adjusted_img2_np = cv2.resize(adjusted_img2_np, (int(adjusted_img2_np.shape[1]*scale_ratio), int(adjusted_img2_np.shape[0]*scale_ratio)), interpolation=cv2.INTER_LANCZOS4)
            adjusted_mask2_np = cv2.resize(adjusted_mask2_np, (int(adjusted_mask2_np.shape[1]*scale_ratio), int(adjusted_mask2_np.shape[0]*scale_ratio)), interpolation=cv2.INTER_NEAREST)
            box1_w = int(box1_w * scale_ratio)
            box1_h = int(box1_h * scale_ratio)
            box1_x = int(box1_x * scale_ratio)
            box1_y = int(box1_y * scale_ratio)
            box2_w = int(box2_w * scale_ratio)
            box2_h = int(box2_h * scale_ratio)
            box2_x = int(box2_x * scale_ratio)
            box2_y = int(box2_y * scale_ratio)

        if divider_thickness > 0 and layout_mode not in ["居中-高度对齐", "居中-宽度对齐", "居中-中心对齐"]:
            if layout_mode.startswith("左右"):
                divider_x = box1_x + (box1_w // 2)
                start_x = max(0, divider_x - divider_thickness)
                end_x = min(composite_img.shape[1], divider_x)
                if start_x < end_x:
                    composite_img[:, start_x:end_x, :] = 0
            else:
                divider_y = box1_y + (box1_h // 2)
                start_y = max(0, divider_y - divider_thickness)
                end_y = min(composite_img.shape[0], divider_y)
                if start_y < end_y:
                    composite_img[start_y:end_y, :, :] = 0

        def np2tensor(np_arr, is_mask=False):
            if is_mask:
                return torch.from_numpy(np_arr).float().unsqueeze(0)
            else:
                if np_arr.shape[-1] == 3 and bg_mode == "transparent":
                    alpha = composite_mask[..., np.newaxis]
                    np_arr = np.concatenate([np_arr, alpha], axis=-1)
                return torch.from_numpy(np_arr).float().unsqueeze(0)

        final_img_tensor = np2tensor(composite_img)
        final_mask_tensor = np2tensor(composite_mask, is_mask=True)
        adjusted_img2_tensor = torch.from_numpy(adjusted_img2_np).float().unsqueeze(0)
        adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_np).float().unsqueeze(0)

        if not isinstance(final_img_tensor, torch.Tensor):
            final_img_tensor = torch.from_numpy(final_img_tensor).float() if isinstance(final_img_tensor, np.ndarray) else torch.tensor(final_img_tensor, dtype=torch.float32)
        if not isinstance(final_mask_tensor, torch.Tensor):
            final_mask_tensor = torch.from_numpy(final_mask_tensor).float() if isinstance(final_mask_tensor, np.ndarray) else torch.tensor(final_mask_tensor, dtype=torch.float32)
        if not isinstance(adjusted_img2_tensor, torch.Tensor):
            adjusted_img2_tensor = torch.from_numpy(adjusted_img2_tensor).float() if isinstance(adjusted_img2_tensor, np.ndarray) else torch.tensor(adjusted_img2_tensor, dtype=torch.float32)
        if not isinstance(adjusted_mask2_tensor, torch.Tensor):
            adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_tensor).float() if isinstance(adjusted_mask2_tensor, np.ndarray) else torch.tensor(adjusted_mask2_tensor, dtype=torch.float32)

        box2 = (box1_w, box1_h, box1_x, box1_y, box2_w, box2_h, box2_x, box2_y)
        return (final_img_tensor, final_mask_tensor, box2, adjusted_img2_tensor, adjusted_mask2_tensor)



#endregion----图像-双图合并---总控制---------





#region----------------------------------------------------------


class Image_solo_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bj_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),

            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "IMAGE",)  
    RETURN_NAMES = ("image","cropped_image", )
    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, inpainted_image, mask, stitch, bj_image):
        original_h, original_w = stitch["original_shape"]
        x, y = stitch["crop_position"]
        w, h = stitch["crop_size"]

        inpainted_np = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        background_np = (bj_image[0].cpu().numpy() * 255).astype(np.uint8)

        inpainted_np = cv2.resize(inpainted_np, (w, h))
        mask_np = cv2.resize(mask_np, (w, h))
        background_np = cv2.resize(background_np, (original_w, original_h))

        cropped_merged = np.zeros((h, w, 4), dtype=np.uint8)
        cropped_merged[:, :, :3] = inpainted_np
        cropped_merged[:, :, 3] = mask_np

        result = np.zeros((original_h, original_w, 4), dtype=np.uint8)
        result[:, :, :3] = background_np.copy()
        result[:, :, 3] = 255

        original_region = result[y:y+h, x:x+w, :3].copy()
        inpainted_region = cropped_merged[:, :, :3]
        alpha = cropped_merged[:, :, 3:4] / 255.0
        
        blended_region = inpainted_region * alpha + original_region * (1 - alpha)
        blended_region = blended_region.astype(np.uint8)
        
        result[y:y+h, x:x+w, :3] = blended_region
        result[y:y+h, x:x+w, 3] = 255

        final_image_tensor = torch.from_numpy(result[:, :, :3] / 255.0).float().unsqueeze(0)
        cropped_merged_tensor = torch.from_numpy(cropped_merged / 255.0).float().unsqueeze(0)

        return (final_image_tensor, cropped_merged_tensor,)


 
class Stack_sample_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == no change"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == no change"}),
                "sampler": ([None] + list(comfy.samplers.KSampler.SAMPLERS), {"default": "euler"}),  
                "scheduler": ([None] + list(comfy.samplers.KSampler.SCHEDULERS), {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("SAMPLE_STACK", )
    RETURN_NAMES = ("sample_stack", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/stack"

    def sample(self, steps, cfg, sampler, scheduler):
        sample_stack = (steps, cfg, sampler, scheduler)     
        return (sample_stack, )
    



class Mask_transform_sum:
    def __init__(self):
        self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "original"}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 400, "step": 1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "crop_to_mask": ("BOOLEAN", {"default": False}),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "rescale_crop": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),  # 新增参数
            },
            "optional": {"base_image": ("IMAGE",), "mask": ("MASK",)}
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"
    
    def separate(self, bg_mode, mask_mode="fill", 
                 ignore_threshold=100, opacity=1.0, outline_thickness=1, 
                 smoothness=1, mask_expand=0,
                 expand_width=0, expand_height=0, rescale_crop=1.0,
                 tapered_corners=True, mask_min=0.0, mask_max=1.0,
                 base_image=None, mask=None, crop_to_mask=False, divisible_by=8):  # 添加 divisible_by 参数
        
        if mask is None:
            if base_image is not None:
                combined_image_tensor = base_image
                empty_mask = torch.zeros_like(base_image[:, :, :, 0])
            else:
                empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
                combined_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (combined_image_tensor, empty_mask)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= ignore_threshold:
                filtered_contours.append(contour)
        
        contours_with_positions = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append((x, y, contour))
        contours_with_positions.sort(key=lambda item: (item[1], item[0]))
        sorted_contours = [item[2] for item in contours_with_positions]
        
        final_mask = np.zeros_like(binary_mask)
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=np.uint8)
        
        for contour in sorted_contours[:8]:
            temp_mask = np.zeros_like(binary_mask)
            
            if mask_mode == "original":
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_image, temp_mask)
            elif mask_mode == "fill":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "fill_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "outline_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline_circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=outline_thickness)
            
            if mask_expand != 0:
                expand_amount = abs(mask_expand)
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=expand_amount)
                else:
                    temp_mask = cv2.erode(temp_mask, kernel, iterations=expand_amount)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_pil = final_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(final_mask_pil)
        
        original_h, original_w = final_mask.shape[:2]
        
        coords = cv2.findNonZero(final_mask)
        crop_params = None
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            mask_center_x = x + w // 2
            mask_center_y = y + h // 2

            new_half_width = (w // 2) + expand_width
            new_half_height = (h // 2) + expand_height
            
            x_new = max(0, mask_center_x - new_half_width)
            y_new = max(0, mask_center_y - new_half_height)
            x_end = min(original_w, mask_center_x + new_half_width)
            y_end = min(original_h, mask_center_y + new_half_height)
            
            crop_params = (x_new, y_new, x_end, y_end)
        else:
            crop_params = (0, 0, original_w, original_h)
        
        if base_image is None:
            base_image_np = np.zeros((original_h, original_w, 3), dtype=np.float32)
        else:
            base_image_np = base_image[0].cpu().numpy() * 255.0
            base_image_np = base_image_np.astype(np.float32)
        
        if crop_to_mask and crop_params is not None:
            x_new, y_new, x_end, y_end = crop_params[:4]
            cropped_final_mask = final_mask[y_new:y_end, x_new:x_end]
            cropped_base_image = base_image_np[y_new:y_end, x_new:x_end].copy()
            
            if rescale_crop != 1.0:
                scaled_w = int(cropped_final_mask.shape[1] * rescale_crop)
                scaled_h = int(cropped_final_mask.shape[0] * rescale_crop)
                
                cropped_final_mask = cv2.resize(
                    cropped_final_mask, 
                    (scaled_w, scaled_h), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                cropped_base_image = cv2.resize(
                    cropped_base_image, 
                    (scaled_w, scaled_h), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            final_mask = cropped_final_mask
            base_image_np = cropped_base_image
        else:
            if base_image_np.shape[:2] != (original_h, original_w):
                base_image_np = cv2.resize(base_image_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        h, w = base_image_np.shape[:2]
        background = np.zeros((h, w, 3), dtype=np.float32)
        if bg_mode in self.colors:
            background[:] = self.colors[bg_mode]
        elif bg_mode == "image" and base_image is not None:
            background = base_image_np.copy()
        
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if bg_mode == "crop_image":
            combined_image = base_image_np.copy()
        else:
            combined_image = background.copy()
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            
            color = np.array(self.colors["white"], dtype=np.float32)
            
            for c in range(3):
                combined_image[:, :, c] = (mask_float * (opacity * color[c] + (1 - opacity) * combined_image[:, :, c]) + 
                                         (1 - mask_float) * combined_image[:, :, c])
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)
        
        # 新增功能：调整图像和遮罩尺寸以满足整除倍数要求
        if divisible_by > 1:
            h, w = combined_image.shape[:2]
            new_h = ((h + divisible_by - 1) // divisible_by) * divisible_by
            new_w = ((w + divisible_by - 1) // divisible_by) * divisible_by
            
            # 如果需要调整尺寸，则进行填充
            if new_h != h or new_w != w:
                # 填充图像
                padded_image = np.zeros((new_h, new_w, 3), dtype=combined_image.dtype)
                padded_image[:h, :w, :] = combined_image
                
                # 填充遮罩
                padded_mask = np.zeros((new_h, new_w), dtype=final_mask.dtype)
                padded_mask[:h, :w] = final_mask
                
                combined_image = padded_image
                final_mask = padded_mask
        
        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)
        
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        
        return (combined_image_tensor, final_mask_tensor)



class XXImage_solo_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["按遮罩裁切", "不裁切", "设置裁切图_宽度", "设置裁切图_高度", "设置背景图_宽度", "设置背景图_高度"],),
                "crop_value": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_stack": ("MASK_STACK",),
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("bj_image", "bj_mask", "cropped_image", "cropped_mask", "stitch")
    FUNCTION = "inpaint_crop"

    def get_mask_bounding_box(self, mask):
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty")
        x, y, w, h = cv2.boundingRect(coords)
        return w, h

    def process_resize(self, image, mask, crop_mode, crop_value, expand_width, expand_height, divisible_by):
        batch_size, img_height, img_width, channels = image.shape
        image_ratio = img_width / img_height
        mask_w, mask_h = self.get_mask_bounding_box(mask)
        mask_ratio = mask_w / mask_h
        new_width, new_height = img_width, img_height
        
        # 先应用扩展值调整遮罩尺寸
        adjusted_mask_w = mask_w + expand_width
        adjusted_mask_h = mask_h + expand_height
        
        if crop_mode == "设置背景图_宽度":
            new_width = crop_value
            new_height = int(new_width / image_ratio)
        elif crop_mode == "设置背景图_高度":
            new_height = crop_value
            new_width = int(new_height * image_ratio)
        elif crop_mode == "设置裁切图_宽度":
            new_mask_width = crop_value
            new_mask_height = int(new_mask_width / mask_ratio)
            mask_scale = new_mask_width / adjusted_mask_w
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "设置裁切图_高度":
            new_mask_height = crop_value
            new_mask_width = int(new_mask_height * mask_ratio)
            mask_scale = new_mask_height / adjusted_mask_h
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "不裁切":
            new_width, new_height = img_width, img_height
            if adjusted_mask_w > adjusted_mask_h:
                new_mask_width = crop_value
                new_mask_height = int(new_mask_width / mask_ratio)
            else:
                new_mask_height = crop_value
                new_mask_width = int(new_mask_height * mask_ratio)
        
        # 确保尺寸可被divisible_by整除
        if divisible_by > 1:
            new_width = new_width - (new_width % divisible_by)
            new_height = new_height - (new_height % divisible_by)
            new_width = max(new_width, divisible_by)
            new_height = max(new_height, divisible_by)
        
        # 调整图像大小
        resized_images = []
        for img in image:
            pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
            resized_pil = pil_img.resize((new_width, new_height), Image.LANCZOS)
            resized_tensor = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0)
            resized_images.append(resized_tensor)
        crop_image = torch.stack(resized_images)
        
        # 调整遮罩大小
        resized_masks = []
        for m in mask:
            pil_mask = Image.fromarray((m.numpy() * 255).astype(np.uint8))
            resized_pil_mask = pil_mask.resize((new_width, new_height), Image.LANCZOS)
            resized_tensor_mask = torch.from_numpy(np.array(resized_pil_mask).astype(np.float32) / 255.0).unsqueeze(0)
            resized_masks.append(resized_tensor_mask)
        crop_mask = torch.cat(resized_masks, dim=0)
        
        return (crop_image, crop_mask)

    def inpaint_crop(self, image, crop_mode, crop_value, mask=None, mask_stack=None):  
        if mask is None:
            batch_size, height, width, _ = image.shape
            mask = torch.ones((batch_size, height, width), dtype=torch.float32)

        if mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = mask_stack            
        else:
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = "original", 1, 0, 0, 1, 20, 20, 1

        crop_image, original_crop_mask = self.process_resize(
            image, mask, crop_mode, crop_value, expand_width, expand_height, divisible_by)
        
        # 保存原始缩放后的遮罩作为bj_mask
        bj_mask_tensor = original_crop_mask
        
        # 处理遮罩堆叠（如果有）
        crop_mask = original_crop_mask
        if mask_stack and crop_mask is not None:
            if hasattr(crop_mask, 'convert'):
                mask_tensor = pil2tensor(crop_mask.convert('L'))
            else:
                if isinstance(crop_mask, torch.Tensor):
                    mask_tensor = crop_mask if len(crop_mask.shape) <= 3 else crop_mask.squeeze(-1) if crop_mask.shape[-1] == 1 else crop_mask
                else:
                    mask_tensor = crop_mask
          
            separated_result = Mask_transform_sum().separate(  
                bg_mode="crop_image", 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=expand_width, 
                expand_height=expand_height,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=crop_image, 
                mask=mask_tensor, 
                crop_to_mask=False,
                divisible_by=divisible_by
            )
            crop_image = separated_result[0]
            crop_mask = separated_result[1]
        
        # 处理图像裁切
        image_np = (crop_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (crop_mask[0].cpu().numpy() * 255).astype(np.uint8)
        original_h, original_w = image_np.shape[0], image_np.shape[1]
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty after processing")
        x, y, w, h = cv2.boundingRect(coords)
        
        # 使用已处理过的扩展值计算新的裁切区域
        mask_center_x = x + w // 2
        mask_center_y = y + h // 2
        new_half_width = (w // 2) + expand_width
        new_half_height = (h // 2) + expand_height
        x_new = max(0, mask_center_x - new_half_width)
        y_new = max(0, mask_center_y - new_half_height)
        x_end = min(original_w, mask_center_x + new_half_width)
        y_end = min(original_h, mask_center_y + new_half_height)
        new_w = x_end - x_new
        new_h = y_end - y_new
        
        # 根据裁切模式处理
        if crop_mode == "不裁切":
            cropped_image = image_np.copy()
            new_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            new_mask[y:y+h, x:x+w] = mask_np[y:y+h, x:x+w]
            current_crop_position = (x, y)
            current_crop_size = (w, h)
        else:
            cropped_image = image_np[y_new:y_end, x_new:x_end]
            mask_x_start = max(0, x - x_new)
            mask_y_start = max(0, y - y_new)
            mask_x_end = min(new_w, (x + w) - x_new)
            mask_y_end = min(new_h, (y + h) - y_new)
            new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            if mask_x_start < mask_x_end and mask_y_start < mask_y_end:
                new_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = mask_np[
                    y + mask_y_start - mask_y_start:y + mask_y_end - mask_y_start,
                    x + mask_x_start - mask_x_start:x + mask_x_end - mask_x_start
                ]
            current_crop_position = (x_new, y_new)
            current_crop_size = (new_w, new_h)
        
        # 转换为张量
        cropped_image_tensor = torch.from_numpy(cropped_image / 255.0).float()
        cropped_mask_tensor = torch.from_numpy(new_mask / 255.0).float().unsqueeze(0)
        
        # 准备拼接信息
        stitch = {
            "original_shape": (original_h, original_w),
            "crop_position": current_crop_position,
            "crop_size": current_crop_size,
        }
        
        return (crop_image, bj_mask_tensor, cropped_image_tensor.unsqueeze(0), cropped_mask_tensor, stitch)



class chx_Ksampler_inpaint:   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE", ),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "work_pattern": (["普通采样", "kontext采样", "仅调整遮罩"], {"default": "普通采样"}),
                "mask_sampling": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                # Image_solo_crop所需参数
                "crop_mode": (["按遮罩裁切", "不裁切", "设置裁切图_宽度", "设置裁切图_高度", "设置背景图_宽度", "设置背景图_高度"], {"default": "设置裁切图_宽度"}),
                "crop_value": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
             
            },

            "optional": {
                "mask": ("MASK", ),
                "pos": ("STRING", {"multiline": True, "default": ""}),
                "mask_stack": ("MASK_STACK",),  
                "sample_stack": ("SAMPLE_STACK",),
            },
        }


    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "IMAGE", "MASK", "STITCH2", "IMAGE")
    RETURN_NAMES = ("context", "bj_image", "image",  "cropped_mask", "stitch", "cropped_image")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"


    def run(self, context, seed, image=None, mask=None, denoise=1, prompt_weight=0.5, pos="",
            work_pattern="普通采样",sample_stack=None, mask_sampling=False,
             mask_stack=None, crop_mode="none", crop_value=512,):

   #------------------------------确保输入有效的mask和image----------------------------------------        
        if mask is None:
            batch_size, height, width, _ = image.shape
            mask = torch.ones((batch_size, height, width), dtype=torch.float32)
  #-------------------------------初始化-------------------------------------
 
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")

        if sample_stack is not None:
            steps, cfg, sampler, scheduler = sample_stack   
            if steps == 0: 
                steps = context.get("steps")
            if cfg == 0: 
                cfg = context.get("cfg")
            if scheduler == None: 
                scheduler = context.get("scheduler")
            if sampler == None: 
                sampler = context.get("sampler")    
                
        steps = context.get("steps")       
        cfg = context.get("cfg")
        scheduler = context.get("scheduler")
        sampler = context.get("sampler")  

        guidance = context.get("guidance", 3.5)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)

        # 初始化变量
        background_tensor = None
        background_mask_tensor = None
        cropped_image_tensor = None
        cropped_mask_tensor = None
        stitch = None

    #--------------------------------接入裁切处理iamge和mask-------------------------------------------------
        if image is not None and mask is not None :
            background_tensor, background_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch = Image_solo_crop().inpaint_crop(
                    image=image,
                    mask=mask,
                    crop_mode=crop_mode,
                    crop_value=crop_value,
                    mask_stack=mask_stack)
            processed_image = cropped_image_tensor
            processed_mask = cropped_mask_tensor
            
            if work_pattern == "仅调整遮罩": 
                return (context, background_tensor, cropped_image_tensor, cropped_mask_tensor, stitch, cropped_image_tensor)

    #----------------------------latent处理-------------------------------------------------------------------------------------
            encoded_result = encode(vae, processed_image)[0]
            if isinstance(encoded_result, dict):
                if "samples" in encoded_result:
                    encoded_latent = encoded_result["samples"]
                else:
                    raise ValueError(f"Encoded result dict doesn't contain 'samples' key. Keys: {list(encoded_result.keys())}")
            elif torch.is_tensor(encoded_result):
                encoded_latent = encoded_result
            else:
                try:
                    encoded_latent = torch.tensor(encoded_result)
                except Exception as e:
                    raise TypeError(f"Cannot convert encoded result to tensor. Type: {type(encoded_result)}, Error: {e}")
                
            if encoded_latent.dim() == 5:
                if encoded_latent.shape[2] == 1:
                    encoded_latent = encoded_latent.squeeze(2)  # 移除大小为1的维度
                else:
                     encoded_latent = encoded_latent.view(encoded_latent.shape[0], 
                                                    encoded_latent.shape[1], 
                                                    encoded_latent.shape[3], 
                                                    encoded_latent.shape[4])
            elif encoded_latent.dim() == 3:
                encoded_latent = encoded_latent.unsqueeze(0)
            elif encoded_latent.dim() != 4:
                raise ValueError(f"Unexpected latent dimensions: {encoded_latent.dim()}. Expected 4D tensor (B,C,H,W). Shape: {encoded_latent.shape}")

            if encoded_latent.size(0) > 1:
                encoded_latent = encoded_latent[:1]

    # ---------------------------------------遮罩是否参与采样--------------------------------------------------------------------------------------------------
    
            latent2 = encoded_latent              
            if not isinstance(latent2, dict):
                if torch.is_tensor(latent2):
                    latent2 = {"samples": latent2}
                else:
                    raise ValueError(f"Unexpected latent format: {type(latent2)}")
            if "samples" not in latent2:
                raise ValueError("Latent dictionary must contain 'samples' key")
                
            if mask_sampling == False:
                latent3 = latent2
            else:
                if processed_mask is not None:
                    if not torch.is_tensor(processed_mask):
                        processed_mask = torch.tensor(processed_mask)
                    if processed_mask.dim() == 3:
                        processed_mask = processed_mask.unsqueeze(0)
                    import copy
                    latent3 = copy.deepcopy(latent2)
                    if processed_mask.shape[1] == 1:
                        processed_mask = processed_mask.repeat(1, 4, 1, 1)  # 修复通道数                    
                    latent3["noise_mask"] = processed_mask
                else:
                    latent3 = latent2

    # ---------------------------------条件处理， kontext 采样模式--------------------------------------------------------------------------------------------------
            if work_pattern == "kontext采样":
                if positive is not None and prompt_weight > 0:
                    latent_samples = None
                    if isinstance(latent3, dict) and "samples" in latent3:
                        latent_samples = latent3["samples"]
                    elif torch.is_tensor(latent3):
                        latent_samples = latent3                     
                    if latent_samples is not None and latent_samples.numel() > 0:
                        try:
                            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                            scaled_latent = latent_samples * influence
                            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                        except Exception as e:
                            print(f"Warning: Failed to process kontext sampling: {e}")


    #----------------------------------------------------------------------------------------------------
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent3, denoise=denoise)
            latent_result = result[0]
            output_image = decode(vae, latent_result)[0]
            context = new_context(context, latent=latent_result, images=output_image)
        
            return (context, background_tensor, output_image, cropped_mask_tensor, stitch, cropped_image_tensor)



class XXXImage_transform_layer:  #坐标会越界
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS , {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bj_img": ("IMAGE",),  
                "fj_img": ("IMAGE",),  
                "mask": ("MASK",),      
                "stitch": ("STITCH2",),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", )
    RETURN_NAMES = ("composite", "mask", "line_mask",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"
    
    def process( self, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, mask_expand, smoothness,
                opacity, blending_mode, blend_strength, bj_img=None, fj_img=None, stitch=None, mask=None ):
        
        color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }
        if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
        
        if bj_img is None:
            bj_img = fj_img.clone()
        
        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        
        canvas_width, canvas_height = bj_pil.size
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2
        
        original_image_info = {
            "width": canvas_width,
            "height": canvas_height,
            "center_x": canvas_center_x,
            "center_y": canvas_center_y
        }
        
        stitch_original_position = None
        if stitch is not None:
            original_shape = stitch.get("original_shape", (canvas_height, canvas_width))
            crop_position = stitch.get("crop_position", (0, 0))
            crop_size = stitch.get("crop_size", (canvas_width, canvas_height))
            
            original_center_x = crop_position[0] + (crop_size[0] // 2)
            original_center_y = crop_position[1] + (crop_size[1] // 2)
            
            stitch_original_position = {
                "x": original_center_x,
                "y": original_center_y,
                "width": crop_size[0],
                "height": crop_size[1],
            }

        if mask is not None: 
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:  
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask

        separated_result = Mask_transform_sum().separate(  
            bg_mode="crop_image", 
            mask_mode="original",
            ignore_threshold=0, 
            opacity=1, 
            outline_thickness=1, 
            smoothness=smoothness,
            mask_expand=mask_expand,
            expand_width=0, 
            expand_height=0,
            rescale_crop=1.0,
            tapered_corners=True,
            mask_min=0, 
            mask_max=1,
            base_image=fj_img, 
            mask=mask_tensor, 
            crop_to_mask=False,
            divisible_by=1
        )

        fj_img = separated_result[0]
        mask = separated_result[1]
        
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
            
            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)
            
            bbox = mask_pil.getbbox()
            if bbox:
                fj_cropped = fj_with_mask.crop(bbox)
                mask_cropped = mask_pil.crop(bbox)
            else:
                fj_cropped = fj_with_mask
                mask_cropped = mask_pil
        else:
            mask_cropped = Image.new("L", fj_pil.size, 255)
            fj_cropped = fj_pil.copy()
            fj_cropped.putalpha(mask_cropped)
        
        cropped_width, cropped_height = fj_cropped.size
        mask_center_x, mask_center_y = cropped_width // 2, cropped_height // 2
        scale_x, scale_y = 1.0, 1.0
        adjusted_fj = fj_cropped
        adjusted_mask = mask_cropped

 #----------------------------------------------------------     
 
   
        adjusted_width, adjusted_height = adjusted_fj.size
        rotation = float(rotation)
        
        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            
            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                scale_x *= scale
                scale_y *= scale
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)
            
            mask_center_x, mask_center_y = adjusted_fj.size[0] // 2, adjusted_fj.size[1] // 2
        
        if stitch is not None and stitch_original_position is not None:
            x_position = stitch_original_position["x"] - mask_center_x + x_offset
            y_position = stitch_original_position["y"] - mask_center_y + y_offset
        else:
            x_position = canvas_center_x - mask_center_x + x_offset
            y_position = canvas_center_y - mask_center_y + y_offset
        
        paste_x = max(0, x_position)
        paste_y = max(0, y_position)
        
        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))
        
        cropped_fj = adjusted_fj.crop((
            max(0, -x_position),
            max(0, -y_position),
            min(adjusted_fj.size[0], canvas_width - x_position),
            min(adjusted_fj.size[1], canvas_height - y_position)
        ))
        
        if blending_mode != "normal":
            temp_img = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            temp_img.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
            composite_pil = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            
            for x in range(canvas_width):
                for y in range(canvas_height):
                    if temp_img.getpixel((x, y))[3] > 0:
                        bg_pixel = bj_pil.getpixel((x, y))
                        fg_pixel = temp_img.getpixel((x, y))
                        bg_pixel_img = Image.new('RGBA', (1, 1), bg_pixel)
                        fg_pixel_img = Image.new('RGBA', (1, 1), fg_pixel)
                        blended_pixel_img = apply_blending_mode(
                            bg_pixel_img, fg_pixel_img, blending_mode, blend_strength
                        )
                        composite_pil.putpixel((x, y), blended_pixel_img.getpixel((0, 0)))
                    else:
                        composite_pil.putpixel((x, y), bj_pil.getpixel((x, y)))
        else:
            composite_pil = bj_pil.copy()
            composite_pil.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
        
        if edge_detection:
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0
            
            threshold = 128
            full_size_mask = Image.new("L", composite_pil.size, 0)
            full_size_mask.paste(adjusted_mask.crop((
                max(0, -x_position),
                max(0, -y_position),
                min(adjusted_mask.size[0], canvas_width - x_position),
                min(adjusted_mask.size[1], canvas_height - y_position)
            )), (paste_x, paste_y))
            
            mask_array = np.array(full_size_mask)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)
            
            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
            
            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            full_size_mask = Image.new("L", composite_pil.size, 0)
            full_size_mask.paste(adjusted_mask.crop((
                max(0, -x_position),
                max(0, -y_position),
                min(adjusted_mask.size[0], canvas_width - x_position),
                min(adjusted_mask.size[1], canvas_height - y_position)
            )), (paste_x, paste_y))
            line_mask_pil = Image.new("L", composite_pil.size, 0)
        
        composite_pil = Image.alpha_composite(bj_pil.convert("RGBA"), composite_pil)
        composite_pil = composite_pil.convert("RGB")
        
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
        
        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)
        
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
        
        return (composite_tensor, mask_tensor, line_mask_tensor, )



class Image_transform_layer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS , {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bj_img": ("IMAGE",),  
                "fj_img": ("IMAGE",),  
                "mask": ("MASK",),      
                "stitch": ("STITCH2",),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", )
    RETURN_NAMES = ("composite", "mask", "line_mask",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"
    
    def process( self, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, mask_expand, smoothness,
                opacity, blending_mode, blend_strength, bj_img=None, fj_img=None, stitch=None, mask=None ):
        
        color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }
        if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
        
        if bj_img is None:
            bj_img = fj_img.clone()
        
        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        
        canvas_width, canvas_height = bj_pil.size
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2
        
        original_image_info = {
            "width": canvas_width,
            "height": canvas_height,
            "center_x": canvas_center_x,
            "center_y": canvas_center_y
        }
        
        stitch_original_position = None
        if stitch is not None:
            original_shape = stitch.get("original_shape", (canvas_height, canvas_width))
            crop_position = stitch.get("crop_position", (0, 0))
            crop_size = stitch.get("crop_size", (canvas_width, canvas_height))
            
            original_center_x = crop_position[0] + (crop_size[0] // 2)
            original_center_y = crop_position[1] + (crop_size[1] // 2)
            
            stitch_original_position = {
                "x": original_center_x,
                "y": original_center_y,
                "width": crop_size[0],
                "height": crop_size[1],
            }

        if mask is not None: 
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:  
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask

        separated_result = Mask_transform_sum().separate(  
            bg_mode="crop_image", 
            mask_mode="original",
            ignore_threshold=0, 
            opacity=1, 
            outline_thickness=1, 
            smoothness=smoothness,
            mask_expand=mask_expand,
            expand_width=0, 
            expand_height=0,
            rescale_crop=1.0,
            tapered_corners=True,
            mask_min=0, 
            mask_max=1,
            base_image=fj_img, 
            mask=mask_tensor, 
            crop_to_mask=False,
            divisible_by=1
        )

        fj_img = separated_result[0]
        mask = separated_result[1]
        
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
            
            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)
            
            bbox = mask_pil.getbbox()
            if bbox:
                fj_cropped = fj_with_mask.crop(bbox)
                mask_cropped = mask_pil.crop(bbox)
            else:
                fj_cropped = fj_with_mask
                mask_cropped = mask_pil
        else:
            mask_cropped = Image.new("L", fj_pil.size, 255)
            fj_cropped = fj_pil.copy()
            fj_cropped.putalpha(mask_cropped)
        
        cropped_width, cropped_height = fj_cropped.size
        mask_center_x, mask_center_y = cropped_width // 2, cropped_height // 2
        scale_x, scale_y = 1.0, 1.0
        adjusted_fj = fj_cropped
        adjusted_mask = mask_cropped

 #----------------------------------------------------------     
 
   
        adjusted_width, adjusted_height = adjusted_fj.size
        rotation = float(rotation)
        
        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            
            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                scale_x *= scale
                scale_y *= scale
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)
            
            mask_center_x, mask_center_y = adjusted_fj.size[0] // 2, adjusted_fj.size[1] // 2
        
        if stitch is not None and stitch_original_position is not None:
            x_position = stitch_original_position["x"] - mask_center_x + x_offset
            y_position = stitch_original_position["y"] - mask_center_y + y_offset
        else:
            x_position = canvas_center_x - mask_center_x + x_offset
            y_position = canvas_center_y - mask_center_y + y_offset
        
        paste_x = max(0, x_position)
        paste_y = max(0, y_position)
        
        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))
        
        # 修复前景图像裁剪坐标
        # 计算裁剪坐标
        left = max(0, -x_position)
        top = max(0, -y_position)
        right = min(adjusted_fj.size[0], canvas_width - x_position)
        bottom = min(adjusted_fj.size[1], canvas_height - y_position)
        
        # 确保坐标有效（左 <= 右，上 <= 下）
        left = min(left, right)
        top = min(top, bottom)
        right = max(left, right)
        bottom = max(top, bottom)
        
        # 执行裁剪
        cropped_fj = adjusted_fj.crop((left, top, right, bottom))
        
        if blending_mode != "normal":
            temp_img = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            temp_img.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
            composite_pil = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            
            for x in range(canvas_width):
                for y in range(canvas_height):
                    if temp_img.getpixel((x, y))[3] > 0:
                        bg_pixel = bj_pil.getpixel((x, y))
                        fg_pixel = temp_img.getpixel((x, y))
                        bg_pixel_img = Image.new('RGBA', (1, 1), bg_pixel)
                        fg_pixel_img = Image.new('RGBA', (1, 1), fg_pixel)
                        blended_pixel_img = apply_blending_mode(
                            bg_pixel_img, fg_pixel_img, blending_mode, blend_strength
                        )
                        composite_pil.putpixel((x, y), blended_pixel_img.getpixel((0, 0)))
                    else:
                        composite_pil.putpixel((x, y), bj_pil.getpixel((x, y)))
        else:
            composite_pil = bj_pil.copy()
            composite_pil.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
        
        if edge_detection:
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0
            
            threshold = 128
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 修复掩码裁剪坐标
            # 计算掩码裁剪坐标
            mask_left = max(0, -x_position)
            mask_top = max(0, -y_position)
            mask_right = min(adjusted_mask.size[0], canvas_width - x_position)
            mask_bottom = min(adjusted_mask.size[1], canvas_height - y_position)
            
            # 确保坐标有效
            mask_left = min(mask_left, mask_right)
            mask_top = min(mask_top, mask_bottom)
            mask_right = max(mask_left, mask_right)
            mask_bottom = max(mask_top, mask_bottom)
            
            # 执行掩码裁剪和粘贴
            full_size_mask.paste(adjusted_mask.crop((
                mask_left,
                mask_top,
                mask_right,
                mask_bottom
            )), (paste_x, paste_y))
            
            mask_array = np.array(full_size_mask)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)
            
            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
            
            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 修复掩码裁剪坐标（else分支）
            # 计算掩码裁剪坐标
            mask_left = max(0, -x_position)
            mask_top = max(0, -y_position)
            mask_right = min(adjusted_mask.size[0], canvas_width - x_position)
            mask_bottom = min(adjusted_mask.size[1], canvas_height - y_position)
            
            # 确保坐标有效
            mask_left = min(mask_left, mask_right)
            mask_top = min(mask_top, mask_bottom)
            mask_right = max(mask_left, mask_right)
            mask_bottom = max(mask_top, mask_bottom)
            
            # 执行掩码裁剪和粘贴
            full_size_mask.paste(adjusted_mask.crop((
                mask_left,
                mask_top,
                mask_right,
                mask_bottom
            )), (paste_x, paste_y))
            line_mask_pil = Image.new("L", composite_pil.size, 0)
        
        composite_pil = Image.alpha_composite(bj_pil.convert("RGBA"), composite_pil)
        composite_pil = composite_pil.convert("RGB")
        
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
        
        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)
        
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
        
        return (composite_tensor, mask_tensor, line_mask_tensor, )




class Image_solo_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["按遮罩裁切", "不裁切", "设置裁切图_宽度", "设置裁切图_高度", "设置背景图_宽度", "设置背景图_高度"],),
                "crop_value": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_stack": ("MASK_STACK",),
                "background_color": (
                    ["none", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"],
                    {"default": "none"}
                ),
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("bj_image", "bj_mask", "cropped_image", "cropped_mask", "stitch")
    FUNCTION = "inpaint_crop"

    def get_mask_bounding_box(self, mask):
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty")
        x, y, w, h = cv2.boundingRect(coords)
        return w, h

    def process_resize(self, image, mask, crop_mode, crop_value, expand_width, expand_height, divisible_by):
        batch_size, img_height, img_width, channels = image.shape
        image_ratio = img_width / img_height
        mask_w, mask_h = self.get_mask_bounding_box(mask)
        mask_ratio = mask_w / mask_h
        new_width, new_height = img_width, img_height
        
        adjusted_mask_w = mask_w + expand_width
        adjusted_mask_h = mask_h + expand_height
        
        if crop_mode == "设置背景图_宽度":
            new_width = crop_value
            new_height = int(new_width / image_ratio)
        elif crop_mode == "设置背景图_高度":
            new_height = crop_value
            new_width = int(new_height * image_ratio)
        elif crop_mode == "设置裁切图_宽度":
            new_mask_width = crop_value
            new_mask_height = int(new_mask_width / mask_ratio)
            mask_scale = new_mask_width / adjusted_mask_w
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "设置裁切图_高度":
            new_mask_height = crop_value
            new_mask_width = int(new_mask_height * mask_ratio)
            mask_scale = new_mask_height / adjusted_mask_h
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "不裁切":
            new_width, new_height = img_width, img_height
            if adjusted_mask_w > adjusted_mask_h:
                new_mask_width = crop_value
                new_mask_height = int(new_mask_width / mask_ratio)
            else:
                new_mask_height = crop_value
                new_mask_width = int(new_mask_height * mask_ratio)
        
        if divisible_by > 1:
            new_width = new_width - (new_width % divisible_by)
            new_height = new_height - (new_height % divisible_by)
            new_width = max(new_width, divisible_by)
            new_height = max(new_height, divisible_by)
        
        resized_images = []
        for img in image:
            pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
            resized_pil = pil_img.resize((new_width, new_height), Image.LANCZOS)
            resized_tensor = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0)
            resized_images.append(resized_tensor)
        crop_image = torch.stack(resized_images)
        
        resized_masks = []
        for m in mask:
            pil_mask = Image.fromarray((m.numpy() * 255).astype(np.uint8))
            resized_pil_mask = pil_mask.resize((new_width, new_height), Image.LANCZOS)
            resized_tensor_mask = torch.from_numpy(np.array(resized_pil_mask).astype(np.float32) / 255.0).unsqueeze(0)
            resized_masks.append(resized_tensor_mask)
        crop_mask = torch.cat(resized_masks, dim=0)
        
        return (crop_image, crop_mask)

    def inpaint_crop(self, image, crop_mode, crop_value, mask=None, mask_stack=None, background_color="none"):
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5)
        }
        
        if mask is None:
            batch_size, height, width, _ = image.shape
            mask = torch.ones((batch_size, height, width), dtype=torch.float32)

        if mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = mask_stack            
        else:
            mask_mode, smoothness, mask_expand, mask_min, mask_max, expand_width, expand_height, divisible_by = "original", 1, 0, 0, 1, 20, 20, 1

        crop_image, original_crop_mask = self.process_resize(
            image, mask, crop_mode, crop_value, expand_width, expand_height, divisible_by)
        
        bj_mask_tensor = original_crop_mask
        
        crop_mask = original_crop_mask
        if mask_stack and crop_mask is not None:
            if hasattr(crop_mask, 'convert'):
                mask_tensor = pil2tensor(crop_mask.convert('L'))
            else:
                if isinstance(crop_mask, torch.Tensor):
                    mask_tensor = crop_mask if len(crop_mask.shape) <= 3 else crop_mask.squeeze(-1) if crop_mask.shape[-1] == 1 else crop_mask
                else:
                    mask_tensor = crop_mask
          
            separated_result = Mask_transform_sum().separate(  
                bg_mode="crop_image", 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=expand_width, 
                expand_height=expand_height,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=crop_image, 
                mask=mask_tensor, 
                crop_to_mask=False,
                divisible_by=divisible_by
            )
            crop_image = separated_result[0]
            crop_mask = separated_result[1]
        
        if background_color != "none" and background_color in colors:
            r, g, b = colors[background_color]
            h, w, _ = crop_image.shape[1:]
            background = torch.zeros((crop_image.shape[0], h, w, 3))
            background[:, :, :, 0] = r
            background[:, :, :, 1] = g
            background[:, :, :, 2] = b
            
            if crop_image.shape[3] >= 4:
                alpha = crop_image[:, :, :, 3].unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)
            else:
                alpha = crop_mask.unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)
        
        image_np = (crop_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (crop_mask[0].cpu().numpy() * 255).astype(np.uint8)
        original_h, original_w = image_np.shape[0], image_np.shape[1]
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty after processing")
        x, y, w, h = cv2.boundingRect(coords)
        
        mask_center_x = x + w // 2
        mask_center_y = y + h // 2
        new_half_width = (w // 2) + expand_width
        new_half_height = (h // 2) + expand_height
        x_new = max(0, mask_center_x - new_half_width)
        y_new = max(0, mask_center_y - new_half_height)
        x_end = min(original_w, mask_center_x + new_half_width)
        y_end = min(original_h, mask_center_y + new_half_height)
        new_w = x_end - x_new
        new_h = y_end - y_new
        
        if crop_mode == "不裁切":
            cropped_image = image_np.copy()
            new_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            new_mask[y:y+h, x:x+w] = mask_np[y:y+h, x:x+w]
            current_crop_position = (x, y)
            current_crop_size = (w, h)
        else:
            cropped_image = image_np[y_new:y_end, x_new:x_end]
            mask_x_start = max(0, x - x_new)
            mask_y_start = max(0, y - y_new)
            mask_x_end = min(new_w, (x + w) - x_new)
            mask_y_end = min(new_h, (y + h) - y_new)
            new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            if mask_x_start < mask_x_end and mask_y_start < mask_y_end:
                new_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = mask_np[
                    y + mask_y_start - mask_y_start:y + mask_y_end - mask_y_start,
                    x + mask_x_start - mask_x_start:x + mask_x_end - mask_x_start
                ]
            current_crop_position = (x_new, y_new)
            current_crop_size = (new_w, new_h)
        
        cropped_image_tensor = torch.from_numpy(cropped_image / 255.0).float()
        cropped_mask_tensor = torch.from_numpy(new_mask / 255.0).float().unsqueeze(0)
        
        stitch = {
            "original_shape": (original_h, original_w),
            "crop_position": current_crop_position,
            "crop_size": current_crop_size,
        }
        
        return (crop_image, bj_mask_tensor, cropped_image_tensor.unsqueeze(0), cropped_mask_tensor, stitch)






#endregion----------------------------合并----------
























































