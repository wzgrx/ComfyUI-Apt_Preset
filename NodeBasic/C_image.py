
from nodes import MAX_RESOLUTION
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from typing import Literal, Any
import math
from PIL import Image, ImageDraw, ImageFilter
from comfy_extras.nodes_mask import composite
from comfy.utils import common_upscale
import re
from math import ceil, sqrt
from typing import cast
from rembg import remove
from PIL import Image, ImageDraw,  ImageFilter, ImageEnhance, ImageDraw, ImageFont
import logging
from tqdm import tqdm
import onnxruntime as ort
from enum import Enum
import random
import folder_paths
from PIL import Image, ImageOps, ImageEnhance, Image, ImageOps, ImageChops, ImageFilter
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import copy
from pymatting import estimate_alpha_cf, estimate_foreground_ml, fix_trimap

import ast
from ..main_unit import *

#logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



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


class Image_batch_selct:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2,"}),
            },
        }
    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"

    def SelectImages(self, images, indexes):
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
            return (images[torch.tensor(select_list1, dtype=torch.long)], images[torch.tensor(exclude_list, dtype=torch.long)])



class Image_batch_selct:

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
                "color": ("COLOR",{"default": "#000000"}),
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



class Image_pad_overlay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "color": ("COLOR",{"default": "#000000"}),
                "width": ("INT", {"default": 512, "min": 0, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "generate_masked_image"
    CATEGORY = "Apt_Preset/image"

    def generate_masked_image(self, image, x_offset, y_offset, scale, width, height, color, mask=None,):
        if mask is not None:
            image, crop_mask = mask_crop(image, mask)
            if width == 0 or height == 0: 
                return (image, crop_mask,)

        new_image = image.clone()
        batch_size, img_height, img_width, _ = new_image.shape

        # 应用缩放
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # 计算位置
        x1 = max(0, x_offset)
        y1 = max(0, y_offset)
        x2 = min(x_offset + scaled_width, img_width)
        y2 = min(y_offset + scaled_height, img_height)

        if isinstance(color, str):
            if color.startswith('rgb'):
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
                if match:
                    r, g, b = map(int, match.groups())
                    color = [r / 255.0, g / 255.0, b / 255.0]
                else:
                    color = [0.0, 0.0, 0.0]
            elif color.startswith('#'):
                color = color.lstrip('#')
                lv = len(color)
                try:
                    r, g, b = tuple(int(color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))
                    color = [r, g, b]
                except ValueError:
                    color = [0.0, 0.0, 0.0]
            else:
                color = [0.0, 0.0, 0.0]
        elif not isinstance(color, (list, tuple)):
            color = [0.0, 0.0, 0.0]

        color_tensor = torch.tensor(color, dtype=torch.float32, device=image.device)

        block_shape = (y2 - y1, x2 - x1, 3)
        colored_block = color_tensor.expand(block_shape)

        final_mask = torch.zeros((img_height, img_width), dtype=torch.float32, device=image.device)
        final_mask[y1:y2, x1:x2] = 1.0

        for i in range(batch_size):
            new_image[i, y1:y2, x1:x2, :] = colored_block

        if mask is not None:
            final_mask = torch.zeros((img_height, img_width), dtype=torch.float32, device=image.device)
            final_mask[y1:y2, x1:x2] = mask[i, :y2 - y1, :x2 - x1]

        return (new_image, final_mask.unsqueeze(0),)



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
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),

                "keep_ratio": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "cut_by_mask": ("MASK",),
                "get_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    def resize(self, image, width, height, keep_ratio, divisible_by, get_image_size=None,cut_by_mask=None):
        B, H, W, C = image.shape

        upscale_method ="lanczos"
        crop = "center"

        if cut_by_mask is not None:
            crop_image,corp_mask = mask_crop(image, cut_by_mask)
            _, height, width, _ = crop_image.shape
            return (crop_image, corp_mask)

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



class Image_transform_batch:
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
                "constant_color": ("COLOR", {"default": "#000000"}),
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
                "color": ("COLOR",),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
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

        # 重新生成 hex 字符串
        final_hex_color = "#{:02x}{:02x}{:02x}".format(final_r, final_g, final_b)

        return (final_hex_color, final_r, final_g, final_b, final_a)


class color_OneColor_replace:
    """Replace Color in an Image"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("COLOR",),
                "replace_color": ("COLOR",),
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
            "required": {
                "image": ("IMAGE",),
                # Modified to directly input color
                "color": ("COLOR",),
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
                "channel": (["R", "G", "B", "A"],),
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
                "channel": (["R", "G", "B", "A"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Preset/image"
    FUNCTION = "Image_Channel_Apply"

    def Image_Channel_Apply(self, images: torch.Tensor, channel_data: torch.Tensor, channel):
        merged_images = []

        for image in images:
            image = image.cpu().clone()

            if channel == "A":
                if image.shape[2] < 4:
                    image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)

                image[:, :, 3] = channel_data
            elif channel == "R":
                image[:, :, 0] = channel_data
            elif channel == "G":
                image[:, :, 1] = channel_data
            else:
                image[:, :, 2] = channel_data

            merged_images.append(image)

        return (torch.stack(merged_images),)



class Image_crop_box2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "box2": ("BOX2",), 
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("image","mask",)
    FUNCTION = "cropbox"

    def cropbox(self, mask=None, image=None, box2=None):
        if mask is not None and box2 is not None:
            w, h, x, y = box2
            mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
            mask = mask[:, y:y + h, x:x + w]
        else:
            mask = None

        if image is not None and box2 is not None:
            w, h, x, y = box2  # 确保 x 和 y 有定义
            x = min(x, image.shape[2] - 1)
            y = min(y, image.shape[1] - 1)
            to_x = w + x
            to_y = h + y
            image = image[:, y:to_y, x:to_x, :]
        else:
            image = None

        return (image, mask,)



#endregion----------------------------------color---------------------------------------------------------------------


#region -----Image_transform_sum----------------------------------#

# ---- Image and Mask Conversion Utils ----
def pil2tensor(image):
    """Converts a PIL image to a PyTorch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def pil2mask(image):
    """Converts a PIL image to a standard mask tensor (1, H, W), normalized to [0,1]."""
    image = image.convert("L")  # 确保是灰度图
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)[None,]
    return tensor

class BlendingMode(Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    VIVID_LIGHT = "vivid_light"
    LINEAR_LIGHT = "linear_light"
    PIN_LIGHT = "pin_light"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    SUBTRACT = "subtract"


class Image_transform_sum:
    def __init__(self):
        self.session = None
        self.bria_pipeline = None
        self.birefnet = None
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        self.custom_model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["transparent", "color", "image",],),
                "bg_color": ("COLOR", {"default": "#000000"}),
                "x_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "front_img_sacle": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),

                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": ("COLOR", {"default": "#FFFFFF"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": ([mode.value for mode in BlendingMode],),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bg_img": ("IMAGE",),
                "front_img": ("IMAGE",),
                "front_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "invert_mask")
    FUNCTION = "process_image"
    CATEGORY = "Apt_Preset/image"


    def process_mask(self, mask, mask_blur, mask_expansion):
        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)
        return mask
    
    def apply_blending_mode(self, bg, fg, mode, strength):
        bg_np = np.array(bg).astype(np.float32) / 255.0
        fg_np = np.array(fg).astype(np.float32) / 255.0

        if bg_np.shape[-1] == 3:
            bg_np = np.dstack([bg_np, np.ones(bg_np.shape[:2], dtype=np.float32)])
        if fg_np.shape[-1] == 3:
            fg_np = np.dstack([fg_np, np.ones(fg_np.shape[:2], dtype=np.float32)])

        bg_rgb, bg_a = bg_np[..., :3], bg_np[..., 3]
        fg_rgb, fg_a = fg_np[..., :3], fg_np[..., 3]

        if mode == BlendingMode.NORMAL.value:
            blended_rgb = fg_rgb
        elif mode == BlendingMode.MULTIPLY.value:
            blended_rgb = bg_rgb * fg_rgb
        elif mode == BlendingMode.SCREEN.value:
            blended_rgb = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif mode == BlendingMode.OVERLAY.value:
            blended_rgb = np.where(bg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.DARKEN.value:
            blended_rgb = np.minimum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.LIGHTEN.value:
            blended_rgb = np.maximum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.COLOR_DODGE.value:
            blended_rgb = np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (1 - fg_rgb)))
        elif mode == BlendingMode.COLOR_BURN.value:
            blended_rgb = np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / fg_rgb))
        elif mode == BlendingMode.LINEAR_DODGE.value:
            blended_rgb = np.minimum(1, bg_rgb + fg_rgb)
        elif mode == BlendingMode.LINEAR_BURN.value:
            blended_rgb = np.maximum(0, bg_rgb + fg_rgb - 1)
        elif mode == BlendingMode.HARD_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.SOFT_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   bg_rgb - (1 - 2 * fg_rgb) * bg_rgb * (1 - bg_rgb),
                                   bg_rgb + (2 * fg_rgb - 1) * (np.sqrt(bg_rgb) - bg_rgb))
        elif mode == BlendingMode.VIVID_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / (2 * fg_rgb))),
                                   np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (2 * (1 - fg_rgb)))))
        elif mode == BlendingMode.LINEAR_LIGHT.value:
            blended_rgb = np.clip(bg_rgb + 2 * fg_rgb - 1, 0, 1)
        elif mode == BlendingMode.PIN_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, np.minimum(bg_rgb, 2 * fg_rgb), np.maximum(bg_rgb, 2 * fg_rgb - 1))
        elif mode == BlendingMode.DIFFERENCE.value:
            blended_rgb = np.abs(bg_rgb - fg_rgb)
        elif mode == BlendingMode.EXCLUSION.value:
            blended_rgb = bg_rgb + fg_rgb - 2 * bg_rgb * fg_rgb
        elif mode == BlendingMode.SUBTRACT.value:
            blended_rgb = np.maximum(0, bg_rgb - fg_rgb)
        else:
            blended_rgb = fg_rgb

        blended_rgb = blended_rgb * strength + bg_rgb * (1 - strength)
        blended_a = fg_a * strength + bg_a * (1 - strength)
        blended = np.dstack([blended_rgb, blended_a])

        return Image.fromarray((blended * 255.0).astype(np.uint8))



    def process_image(self, bg_mode, bg_color, blending_mode, blend_strength, front_img_sacle, x_position, y_position, rotation, opacity, edge_detection, edge_thickness, edge_color, mask_blur, mask_expansion, front_img=None, bg_img=None, front_mask=None):
        error_img = Image.new("RGB", (64, 64), color="red")
        error_tensor = pil2tensor(error_img)
        error_mask = torch.zeros((1, 64, 64), dtype=torch.float32)

        if front_img is None and bg_img is None:
            return (error_tensor, error_mask, error_mask)

        if front_img is None:
            front_img = bg_img
            bg_img = None

        fg_pil = tensor2pil(front_img)
        bg_pil = tensor2pil(bg_img) if bg_img is not None else None

        input_mask_np = None
        if front_mask is not None:
            mask_pil = tensor2pil(front_mask).convert('L')
            input_mask_np = np.array(mask_pil).astype(np.uint8)

        # 创建与前景图像相同尺寸的mask
        if input_mask_np is not None:
            if input_mask_np.shape != fg_pil.size[::-1]:
                mask_pil = Image.fromarray(input_mask_np).resize(fg_pil.size, Image.LANCZOS)
                input_mask_np = np.array(mask_pil).astype(np.uint8)
            final_mask = input_mask_np
        else:
            final_mask = np.full(fg_pil.size[::-1], 255, dtype=np.uint8)
        
        final_mask = self.process_mask(final_mask, mask_blur, mask_expansion)

        try:
            if bg_mode == "transparent":
                bg_pil = Image.new("RGBA", fg_pil.size, (0, 0, 0, 0))
            elif bg_mode == "color":
                bg_color = bg_color.lstrip('#')
                r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
                bg_pil = Image.new("RGBA", fg_pil.size, (r, g, b, 255))
            elif bg_mode == "image":
                if bg_img is not None:
                    bg_pil = tensor2pil(bg_img).convert("RGBA")
                else:
                    return (error_tensor, error_mask, error_mask)

            if fg_pil.mode != "RGBA":
                fg_pil = fg_pil.convert("RGBA")

            mask_image = Image.fromarray(final_mask)
            fg_with_mask = fg_pil.copy()
            fg_data = np.array(fg_with_mask)
            mask_data = np.array(mask_image.resize(fg_pil.size, Image.LANCZOS))
            
            # 应用mask到alpha通道
            fg_data[..., 3] = (fg_data[..., 3].astype(np.float32) * (mask_data.astype(np.float32)/255.0)).astype(np.uint8)
            fg_with_mask = Image.fromarray(fg_data)

            if edge_detection:
                edge_color_hex = edge_color.lstrip('#')
                r, g, b = tuple(int(edge_color_hex[i:i+2], 16) for i in (0, 2, 4))
                edge_image = Image.new("RGBA", fg_pil.size, (0, 0, 0, 0))
                edge_draw = ImageDraw.Draw(edge_image)

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    for i in range(edge_thickness):
                        points = [tuple(point[0]) for point in contour]
                        edge_draw.line(points, fill=(r, g, b, 255), width=edge_thickness-i+1)
                fg_with_mask = Image.alpha_composite(fg_with_mask, edge_image)

            # 创建单独的mask副本用于变换
            mask_for_transform = mask_image.copy()
            
            # 先应用缩放变换
            if front_img_sacle != 1.0:
                new_size = (int(fg_with_mask.width * front_img_sacle), int(fg_with_mask.height * front_img_sacle))
                fg_with_mask = fg_with_mask.resize(new_size, Image.LANCZOS)
                mask_for_transform = mask_for_transform.resize(new_size, Image.NEAREST)
            
            # 然后应用旋转变换
            if abs(rotation) > 0.1:
                # 旋转前景图像
                rotated_fg = fg_with_mask.rotate(rotation, expand=True, resample=Image.BILINEAR)
                
                # 旋转mask（使用最近邻插值保持二值性）
                rotated_mask = mask_for_transform.rotate(rotation, expand=True, resample=Image.NEAREST)
                
                # 计算旋转后的尺寸并更新
                fg_with_mask = rotated_fg
                mask_for_transform = rotated_mask
            
            # 更新final_mask以供后续使用
            final_mask = np.array(mask_for_transform).astype(np.uint8)

            result = bg_pil.copy()
            bg_width, bg_height = bg_pil.size
            fg_width, fg_height = fg_with_mask.size

            # 计算放置位置 - 这里添加了x,y偏移量
            pos_x = bg_width // 2 - fg_width // 2 + x_position
            pos_y = bg_height // 2 - fg_height // 2 + y_position
            
            # 确保位置不会导致图像超出范围
            #pos_x = max(0, min(pos_x, bg_width - fg_width))
            #pos_y = max(0, min(pos_y, bg_height - fg_height))

            if blending_mode != BlendingMode.NORMAL.value:
                bg_region = result.crop((pos_x, pos_y, pos_x+fg_width, pos_y+fg_height))
                blended_region = self.apply_blending_mode(bg_region, fg_with_mask, blending_mode, blend_strength)

                if opacity < 1.0:
                    r, g, b, a = blended_region.split()
                    a = a.point(lambda x: int(x * opacity))
                    blended_region = Image.merge("RGBA", (r, g, b, a))

                result.paste(blended_region, (pos_x, pos_y), blended_region.split()[3] if blended_region.mode=="RGBA" else None)
            else:
                if opacity < 1.0:
                    r, g, b, a = fg_with_mask.split()
                    a = a.point(lambda x: int(x * opacity))
                    fg_with_mask = Image.merge("RGBA", (r, g, b, a))

                result.paste(fg_with_mask, (pos_x, pos_y), fg_with_mask.split()[3] if fg_with_mask.mode=="RGBA" else None)

            result = result.convert("RGB")
            
            # 创建最终mask图像，尺寸与输出图像一致
            full_mask = Image.new("L", result.size, 0)
            full_mask.paste(mask_for_transform, (pos_x, pos_y), mask_for_transform)

            # 创建invert_mask
            invert_mask = ImageOps.invert(full_mask)
            
            # 转换为tensor格式
            result_tensor = pil2tensor(result)
            mask_tensor = pil2mask(full_mask)
            invert_mask_tensor = pil2mask(invert_mask)
            
            if result_tensor is None or mask_tensor is None or invert_mask_tensor is None:
                return (error_tensor, error_mask, error_mask)
                
            return (result_tensor, mask_tensor, invert_mask_tensor)
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            return (error_tensor, error_mask, error_mask)




#endregion ----------------Image_transform_sum---------------------------------------#



