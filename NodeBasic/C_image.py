
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




from ..main_unit import *

#logging.basicConfig(level=logging.INFO)

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Preset/image"
    FUNCTION = "Image_Channel_Apply"

    def Image_Channel_Apply(self, images: torch.Tensor, channel_data: torch.Tensor, channel, invert_channel=False):
        merged_images = []

        # 如果需要反向通道数据，则进行反向处理
        if invert_channel:
            channel_data = 1.0 - channel_data

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
            else:  # channel == "B"
                image[:, :, 2] = channel_data

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
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("裁剪图像","裁剪遮罩")
    FUNCTION = "cropbox"

    def cropbox(self, mask=None, image=None, box2=None):
        # 检查输入参数
        if box2 is None:
            return (None, None)
            
        # 从box2解包参数: 宽度, 高度, 中心点X坐标, 中心点Y坐标
        region_width, region_height, center_x, center_y = box2
        
        # 计算裁剪区域的左上角坐标
        x_start = max(0, int(center_x - region_width // 2))
        y_start = max(0, int(center_y - region_height // 2))
        
        # 计算裁剪区域的右下角坐标
        x_end = x_start + region_width
        y_end = y_start + region_height
        
        # 裁切图像
        cropped_image = None
        if image is not None:
            img_h, img_w = image.shape[1], image.shape[2]
            
            # 确保坐标不超出图像边界
            x_start = min(x_start, img_w)
            y_start = min(y_start, img_h)
            x_end = min(x_end, img_w)
            y_end = min(y_end, img_h)
            
            # 执行裁切
            cropped_image = image[:, y_start:y_end, x_start:x_end, :]

        # 裁切遮罩
        cropped_mask = None
        if mask is not None:
            mask_h, mask_w = mask.shape[1], mask.shape[2]
            
            # 确保坐标不超出遮罩边界
            x_start = min(x_start, mask_w)
            y_start = min(y_start, mask_h)
            x_end = min(x_end, mask_w)
            y_end = min(y_end, mask_h)
            
            # 执行裁切
            cropped_mask = mask[:, y_start:y_end, x_start:x_end]

        return (cropped_image, cropped_mask,)


class Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "layout_mode": ( ["居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐", 
                                                 "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                                 "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐"],),  
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
    RETURN_NAMES = ("composite_image", "composite_mask", "box2", "new_img2", "new_mask2")  #box2是图2变化后的WH和左上角坐标
    FUNCTION = "composite"
    CATEGORY = "Apt_Preset/image"

    def composite(self, layout_mode, bg_mode, size_mode, target_size, divider_thickness, image1=None, image2=None, mask1=None, mask2=None):

        composite_img = None
        composite_mask = None
        box2_w, box2_h, box2_x, box2_y = 0, 0, 0, 0

        adjusted_img2_np = np.zeros((h1, w1, 3), dtype=np.float32) if 'h1' in locals() and 'w1' in locals() else np.zeros((512, 512, 3), dtype=np.float32)
        adjusted_mask2_np = np.ones((h1, w1), dtype=np.float32) if 'h1' in locals() and 'w1' in locals() else np.ones((512, 512), dtype=np.float32)


        # 处理image1为空的情况
        if image1 is None and image2 is not None:
            image1 = image2
        elif image1 is None and image2 is None:
            # 创建默认图像
            default_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image1 = default_img
            image2 = default_img

        # 先获取图像尺寸，再处理numpy转换
        if isinstance(image1, torch.Tensor):
            h1, w1 = image1.shape[1], image1.shape[2]
        else:
            h1, w1 = image1.shape[0], image1.shape[1] if len(image1.shape) >= 3 else (image1.shape[0], image1.shape[1])
            
        if isinstance(image2, torch.Tensor):
            h2, w2 = image2.shape[1], image2.shape[2]
        else:
            h2, w2 = image2.shape[0], image2.shape[1] if len(image2.shape) >= 3 else (image2.shape[0], image2.shape[1])

        # 确保输入图像为numpy数组格式
        if isinstance(image1, torch.Tensor):
            img1_np = image1.cpu().numpy()[0]
        else:
            img1_np = image1[0] if len(image1.shape) == 4 else image1
            
        if isinstance(image2, torch.Tensor):
            img2_np = image2.cpu().numpy()[0]
        else:
            img2_np = image2[0] if len(image2.shape) == 4 else image2
        
        # 处理mask
        if mask1 is not None:
            if isinstance(mask1, torch.Tensor):
                mask1_np = mask1.cpu().numpy()[0]
            else:
                mask1_np = mask1[0] if len(mask1.shape) == 4 else mask1
                
            if len(mask1_np.shape) == 3:
                mask1_np = mask1_np[:, :, 0]
        else:
            mask1_np = np.ones((h1, w1))
            
        if mask2 is not None:
            if isinstance(mask2, torch.Tensor):
                mask2_np = mask2.cpu().numpy()[0]
            else:
                mask2_np = mask2[0] if len(mask2.shape) == 4 else mask2
                
            if len(mask2_np.shape) == 3:
                mask2_np = mask2_np[:, :, 0]
        else:
            mask2_np = np.ones((h2, w2))
    
        
        box2_w, box2_h, box2_x, box2_y = 0, 0, 0, 0
            
        # 初始化调整后的image2和mask2
        adjusted_img2_np = None
        adjusted_mask2_np = None

        if layout_mode == "居中-自动对齐":
            if h2 > h1:
                layout_mode = "居中-高度对齐"
            elif w2 > w1:
                layout_mode = "居中-宽度对齐"
            else:
                layout_mode = "居中-中心对齐"


        if layout_mode == "左右-中心对齐":
            img2_tensor = torch.from_numpy(img2_np).unsqueeze(0)
            img2_resized = get_image_resize(img2_tensor, torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                crop_h = h1
                crop_w = w1
                start_y = max(0, (h2 - crop_h) // 2)
                start_x = max(0, (w2 - crop_w) // 2)
                
                mask2_cropped = mask2_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
                
                # 使用OpenCV重写遮罩处理，避免PIL的类型错误
                if mask2_cropped.size > 0:
                    # 转换为OpenCV可处理的格式
                    mask2_cv = (mask2_cropped * 255).astype(np.uint8)
                    
                    # 使用OpenCV调整大小，避免PIL的类型错误
                    mask2_resized_cv = cv2.resize(
                        mask2_cv, 
                        (w1, h1), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # 转回ComfyUI使用的格式
                    mask2_resized_np = mask2_resized_cv / 255.0
                else:
                    mask2_resized_np = np.ones((h1, w1))
            else:
                mask2_resized_np = np.ones((h1, w1))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            new_w = w1 + w1
            new_h = h1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_img[:h1, :w1] = img1_np
            composite_img[:h1, w1:w1+w1] = img2_resized_np[:h1, :w1]
            
            composite_mask = np.zeros((new_h, new_w))
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:h1, w1:w1+w1] = mask2_resized_np[:h1, :w1]
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = w1, h1, w1 + w1//2, h1//2
            
        elif layout_mode == "上下-中心对齐":
            img2_tensor = torch.from_numpy(img2_np).unsqueeze(0)
            img2_resized = get_image_resize(img2_tensor, torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                crop_h = h1
                crop_w = w1
                start_y = max(0, (h2 - crop_h) // 2)
                start_x = max(0, (w2 - crop_w) // 2)
                
                mask2_cropped = mask2_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
                
                if mask2_cropped.size > 0:
                    mask2_cv = (mask2_cropped * 255).astype(np.uint8)
                    mask2_resized_cv = cv2.resize(
                        mask2_cv, 
                        (w1, h1), 
                        interpolation=cv2.INTER_NEAREST
                    )                   
                    mask2_resized_np = mask2_resized_cv / 255.0
                else:
                    mask2_resized_np = np.ones((h1, w1))
            else:
                mask2_resized_np = np.ones((h1, w1))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            new_w = w1
            new_h = h1 + h1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+h1, :w1] = img2_resized_np[:h1, :w1]
            
            composite_mask = np.zeros((new_h, new_w))
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+h1, :w1] = mask2_resized_np[:h1, :w1]
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = w1, h1, w1//2, h1 + h1//2
            
        elif layout_mode == "左右-高度对齐":
            ratio = h1 / h2
            new_w2 = int(w2 * ratio)
            new_h2 = h1
            
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            new_w = w1 + new_w2
            new_h = h1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_img[:h1, :w1] = img1_np
            composite_img[:new_h2, w1:w1+new_w2] = img2_resized_np[:new_h2, :new_w2]
            
            composite_mask = np.zeros((new_h, new_w))
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:new_h2, w1:w1+new_w2] = mask2_resized_np[:new_h2, :new_w2]
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, w1 + new_w2//2, new_h2//2
            
        elif layout_mode == "上下-宽度对齐":
            ratio = w1 / w2
            new_h2 = int(h2 * ratio)
            new_w2 = w1
            
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            new_w = w1
            new_h = h1 + new_h2
            composite_img = np.zeros((new_h, new_w, 3))
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+new_h2, :new_w2] = img2_resized_np[:new_h2, :new_w2]
            
            composite_mask = np.zeros((new_h, new_w))
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+new_h2, :new_w2] = mask2_resized_np[:new_h2, :new_w2]
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, new_w2//2, h1 + new_h2//2
            
        elif layout_mode == "居中-中心对齐":
            # image2调整到与image1相同尺寸，按中心对齐
            img2_tensor = torch.from_numpy(img2_np).unsqueeze(0)
            img2_resized = get_image_resize(img2_tensor, torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                crop_h = h1
                crop_w = w1
                start_y = max(0, (h2 - crop_h) // 2)
                start_x = max(0, (w2 - crop_w) // 2)
                
                mask2_cropped = mask2_np[start_y:start_y+crop_h, start_x:start_x+crop_w]
                
                if mask2_cropped.size > 0:
                    mask2_cv = (mask2_cropped * 255).astype(np.uint8)
                    mask2_resized_cv = cv2.resize(
                        mask2_cv, 
                        (w1, h1), 
                        interpolation=cv2.INTER_NEAREST
                    )                   
                    mask2_resized_np = mask2_resized_cv / 255.0
                else:
                    mask2_resized_np = np.ones((h1, w1))
            else:
                mask2_resized_np = np.ones((h1, w1))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            # 确定输出图像尺寸（取两个图像的最大尺寸）
            new_w = w1
            new_h = h1
            
            # 创建合成图像 - 先放置image1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_img[:h1, :w1] = img1_np
            
            # 创建合成遮罩 - 注意：在透明模式下，我们不使用mask1影响图像显示
            composite_mask = np.zeros((new_h, new_w))
            
            # 只在mask2区域覆盖image2，mask1不影响图像显示
            if mask2 is not None:
                # 使用alpha混合，只在mask2区域显示image2
                alpha = mask2_resized_np[..., np.newaxis]
                composite_img[:h1, :w1] = img1_np * (1 - alpha) + img2_resized_np[:h1, :w1] * alpha
                # 遮罩只使用mask2，忽略mask1
                composite_mask[:h1, :w1] = mask2_resized_np
            else:
                # 如果没有mask2，直接覆盖
                composite_img[:h1, :w1] = img2_resized_np[:h1, :w1]
                composite_mask[:h1, :w1] = np.ones((h1, w1))
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = w1, h1, w1//2, h1//2

        elif layout_mode == "居中-高度对齐":
            # image2调整到与image1相同高度，按中心对齐
            ratio = h1 / h2
            new_w2 = int(w2 * ratio)
            new_h2 = h1
            
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            # 确定输出图像尺寸
            new_w = max(w1, new_w2)
            new_h = h1
            
            # 创建合成图像 - 先放置image1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_mask = np.zeros((new_h, new_w))
            
            # 计算居中位置
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - new_w2) // 2
            
            # 放置image1
            composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
            
            # 只在mask2区域覆盖image2，mask1不影响图像显示
            if mask2 is not None:
                # 获取重叠区域
                overlap_start_x = max(x1_offset, x2_offset)
                overlap_end_x = min(x1_offset + w1, x2_offset + new_w2)
                
                if overlap_start_x < overlap_end_x:
                    # 在重叠区域进行alpha混合
                    overlap_width = overlap_end_x - overlap_start_x
                    img1_overlap_start = overlap_start_x - x1_offset
                    img2_overlap_start = overlap_start_x - x2_offset
                    
                    # Alpha混合只在mask2区域
                    alpha = mask2_resized_np[:, img2_overlap_start:img2_overlap_start+overlap_width][..., np.newaxis]
                    img1_part = composite_img[:h1, overlap_start_x:overlap_end_x]
                    img2_part = img2_resized_np[:h1, img2_overlap_start:img2_overlap_start+overlap_width]
                    
                    composite_img[:h1, overlap_start_x:overlap_end_x] = img1_part * (1 - alpha) + img2_part * alpha
                    
                    # 遮罩只使用mask2，忽略mask1
                    composite_mask[:h1, overlap_start_x:overlap_end_x] = mask2_resized_np[:h1, img2_overlap_start:img2_overlap_start+overlap_width]
            else:
                # 没有mask2时直接覆盖
                composite_img[:new_h2, x2_offset:x2_offset+new_w2] = img2_resized_np[:new_h2, :new_w2]
                composite_mask[:new_h2, x2_offset:x2_offset+new_w2] = np.ones((new_h2, new_w2))
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, x2_offset + new_w2//2, new_h2//2

        elif layout_mode == "居中-宽度对齐":
            # image2调整到与image1相同宽度，按中心对齐
            ratio = w1 / w2
            new_h2 = int(h2 * ratio)
            new_w2 = w1
            
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()  # 保存调整后的image2
            
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()  # 保存调整后的mask2
            
            # 确定输出图像尺寸
            new_w = w1
            new_h = max(h1, new_h2)
            
            # 创建合成图像 - 先放置image1
            composite_img = np.zeros((new_h, new_w, 3))
            composite_mask = np.zeros((new_h, new_w))
            
            # 计算居中位置
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - new_h2) // 2
            
            # 放置image1
            composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
            
            # 只在mask2区域覆盖image2，mask1不影响图像显示
            if mask2 is not None:
                # 获取重叠区域
                overlap_start_y = max(y1_offset, y2_offset)
                overlap_end_y = min(y1_offset + h1, y2_offset + new_h2)
                
                if overlap_start_y < overlap_end_y:
                    # 在重叠区域进行alpha混合
                    overlap_height = overlap_end_y - overlap_start_y
                    img1_overlap_start = overlap_start_y - y1_offset
                    img2_overlap_start = overlap_start_y - y2_offset
                    
                    # Alpha混合只在mask2区域
                    alpha = mask2_resized_np[img2_overlap_start:img2_overlap_start+overlap_height, :][..., np.newaxis]
                    img1_part = composite_img[overlap_start_y:overlap_end_y, :w1]
                    img2_part = img2_resized_np[img2_overlap_start:img2_overlap_start+overlap_height, :w1]
                    
                    composite_img[overlap_start_y:overlap_end_y, :w1] = img1_part * (1 - alpha) + img2_part * alpha
                    
                    # 遮罩只使用mask2，忽略mask1
                    composite_mask[overlap_start_y:overlap_end_y, :w1] = mask2_resized_np[img2_overlap_start:img2_overlap_start+overlap_height, :w1]
            else:
                # 没有mask2时直接覆盖
                composite_img[y2_offset:y2_offset+new_h2, :new_w2] = img2_resized_np[:new_h2, :new_w2]
                composite_mask[y2_offset:y2_offset+new_h2, :new_w2] = np.ones((new_h2, new_w2))
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, new_w2//2, y2_offset + new_h2//2

        elif layout_mode == "左右-宽度对齐":
            # 左右排版，image2参考image1的宽度按比例生成
            ratio = w1 / w2
            new_h2 = int(h2 * ratio)
            new_w2 = w1
            
            # 调整image2尺寸
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()
            
            # 调整mask2尺寸
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()
            
            # 计算输出尺寸
            new_w = w1 + new_w2
            new_h = max(h1, new_h2)
            
            # 当bg_mode为"image"或"transparent"时，转换为"black"
            effective_bg_mode = bg_mode
            if bg_mode == "image":
                effective_bg_mode = "black"
            if bg_mode == "transparent":
                effective_bg_mode = "black"        
            # 创建背景
            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            composite_mask = np.zeros((new_h, new_w))

            # 计算垂直居中偏移
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - new_h2) // 2
            
            # 放置图像 - 修复类型和通道数不匹配问题
            if isinstance(composite_img, torch.Tensor):
                # 如果是torch tensor，需要检查通道数并相应处理
                if composite_img.shape[-1] == 4:  # RGBA背景
                    # 转换RGB numpy数组为RGBA tensor
                    img1_rgba = np.dstack([img1_np, np.ones((h1, w1))])  # 添加alpha通道
                    img2_rgba = np.dstack([img2_resized_np, np.ones((new_h2, new_w2))])  # 添加alpha通道
                    
                    img1_tensor = torch.from_numpy(img1_rgba).to(composite_img.device, composite_img.dtype)
                    img2_tensor = torch.from_numpy(img2_rgba).to(composite_img.device, composite_img.dtype)
                else:  # RGB背景
                    img1_tensor = torch.from_numpy(img1_np).to(composite_img.device, composite_img.dtype)
                    img2_tensor = torch.from_numpy(img2_resized_np).to(composite_img.device, composite_img.dtype)
                    
                composite_img[y1_offset:y1_offset+h1, :w1] = img1_tensor
                composite_img[y2_offset:y2_offset+new_h2, w1:w1+new_w2] = img2_tensor
            else:
                # 如果是numpy数组，直接赋值
                composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
                composite_img[y2_offset:y2_offset+new_h2, w1:w1+new_w2] = img2_resized_np
            
            # 放置遮罩 - 保持numpy数组操作
            composite_mask[y1_offset:y1_offset+h1, :w1] = mask1_np
            composite_mask[y2_offset:y2_offset+new_h2, w1:w1+new_w2] = mask2_resized_np
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, w1 + new_w2//2, y2_offset + new_h2//2

        elif layout_mode == "上下-高度对齐":
            # 上下排版，image2参考image1的高度按比例生成
            ratio = h1 / h2
            new_w2 = int(w2 * ratio)
            new_h2 = h1
            
            # 调整image2尺寸
            img2_pil = Image.fromarray(np.clip(img2_np * 255, 0, 255).astype(np.uint8))
            img2_pil = img2_pil.resize((new_w2, new_h2), Image.LANCZOS)
            img2_resized_np = np.array(img2_pil) / 255.0
            adjusted_img2_np = img2_resized_np.copy()
            
            # 调整mask2尺寸
            if mask2 is not None:
                mask2_pil = Image.fromarray(np.clip(mask2_np * 255, 0, 255).astype(np.uint8))
                mask2_pil = mask2_pil.resize((new_w2, new_h2), Image.LANCZOS)
                mask2_resized_np = np.array(mask2_pil) / 255.0
            else:
                mask2_resized_np = np.ones((new_h2, new_w2))
            
            adjusted_mask2_np = mask2_resized_np.copy()
            
            # 计算输出尺寸
            new_w = max(w1, new_w2)
            new_h = h1 + new_h2
            
            # 当bg_mode为"image"或"transparent"时，转换为"black"

            effective_bg_mode = bg_mode
            if bg_mode == "image":
                effective_bg_mode = "black"
            if bg_mode == "transparent":
                effective_bg_mode = "black"        

            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            composite_mask = np.zeros((new_h, new_w))
            
            # 计算水平居中偏移
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - new_w2) // 2
            
            # 放置图像 - 修复类型和通道数不匹配问题
            if isinstance(composite_img, torch.Tensor):
                # 如果是torch tensor，需要检查通道数并相应处理
                if composite_img.shape[-1] == 4:  # RGBA背景
                    # 转换RGB numpy数组为RGBA tensor
                    img1_rgba = np.dstack([img1_np, np.ones((h1, w1))])  # 添加alpha通道
                    img2_rgba = np.dstack([img2_resized_np, np.ones((new_h2, new_w2))])  # 添加alpha通道
                    
                    img1_tensor = torch.from_numpy(img1_rgba).to(composite_img.device, composite_img.dtype)
                    img2_tensor = torch.from_numpy(img2_rgba).to(composite_img.device, composite_img.dtype)
                else:  # RGB背景
                    img1_tensor = torch.from_numpy(img1_np).to(composite_img.device, composite_img.dtype)
                    img2_tensor = torch.from_numpy(img2_resized_np).to(composite_img.device, composite_img.dtype)
                    
                composite_img[:h1, x1_offset:x1_offset+w1] = img1_tensor
                composite_img[h1:h1+new_h2, x2_offset:x2_offset+new_w2] = img2_tensor
            else:
                # 如果是numpy数组，直接赋值
                composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
                composite_img[h1:h1+new_h2, x2_offset:x2_offset+new_w2] = img2_resized_np
            
            # 放置遮罩 - 保持numpy数组操作
            composite_mask[:h1, x1_offset:x1_offset+w1] = mask1_np
            composite_mask[h1:h1+new_h2, x2_offset:x2_offset+new_w2] = mask2_resized_np
            
            # 打包为box2元组
            box2_w, box2_h, box2_x, box2_y = new_w2, new_h2, x2_offset + new_w2//2, h1 + new_h2//2



        # 确保所有必需的变量都已初始化，防止UnboundLocalError
        if 'composite_img' not in locals() or composite_img is None:
            # 如果image1存在，使用其尺寸作为默认尺寸
            if image1 is not None:
                if isinstance(image1, torch.Tensor):
                    h1, w1 = image1.shape[1], image1.shape[2]
                else:
                    h1, w1 = image1.shape[0], image1.shape[1] if len(image1.shape) >= 3 else (image1.shape[0], image1.shape[1])
            else:
                # 否则使用默认尺寸
                h1, w1 = 512, 512
                
            composite_img = np.zeros((h1, w1, 3), dtype=np.float32)
            
        if 'composite_mask' not in locals() or composite_mask is None:
            if image1 is not None:
                if isinstance(image1, torch.Tensor):
                    h1, w1 = image1.shape[1], image1.shape[2]
                else:
                    h1, w1 = image1.shape[0], image1.shape[1] if len(image1.shape) >= 3 else (image1.shape[0], image1.shape[1])
            else:
                h1, w1 = 512, 512
                
            composite_mask = np.zeros((h1, w1), dtype=np.float32)
            
        # 确保box2变量已初始化
        if 'box2_w' not in locals(): box2_w = 0
        if 'box2_h' not in locals(): box2_h = 0
        if 'box2_x' not in locals(): box2_x = 0
        if 'box2_y' not in locals(): box2_y = 0

        if box2_w == 0 and box2_h == 0:
            if image1 is not None:
                if isinstance(image1, torch.Tensor):
                    h1, w1 = image1.shape[1], image1.shape[2]
                else:
                    h1, w1 = image1.shape[0], image1.shape[1] if len(image1.shape) >= 3 else (image1.shape[0], image1.shape[1])
            else:
                h1, w1 = 512, 512
            box2_w, box2_h, box2_x, box2_y = w1, h1, w1//2, h1//2

        # 确保adjusted_img2_np和adjusted_mask2_np已初始化
        if 'adjusted_img2_np' not in locals():
            adjusted_img2_np = None
        if 'adjusted_mask2_np' not in locals():
            adjusted_mask2_np = None


        final_img = composite_img
        final_mask = composite_mask

        # 添加分割线到合成图像，确保只覆盖图1区域（新增模式不添加分割线）
        if divider_thickness > 0 and layout_mode not in ["居中-高度对齐", "居中-宽度对齐", "居中-中心对齐"]:
            if layout_mode in ["左右-中心对齐", "左右-高度对齐", "左右-宽度对齐", "上下-高度对齐"]:
                # 垂直分割线，只覆盖图1的右侧边缘
                divider_x = w1  # 图1和图2的交界处
                start_x = max(0, divider_x - divider_thickness)
                end_x = divider_x
                # 确保不超出图像边界
                end_x = min(final_img.shape[1], end_x)
                if start_x < end_x:
                    final_img[:, start_x:end_x, :] = 0  # 黑色分割线
            else:
                # 水平分割线，只覆盖图1的下侧边缘
                divider_y = h1  # 图1和图2的交界处
                start_y = max(0, divider_y - divider_thickness)
                end_y = divider_y
                # 确保不超出图像边界
                end_y = min(final_img.shape[0], end_y)
                if start_y < end_y:
                    final_img[start_y:end_y, :, :] = 0  # 黑色分割线
        

        if size_mode != "auto":
            current_h, current_w = final_img.shape[:2]
            
            # 计算新尺寸
            if size_mode == "输出宽度":
                ratio = target_size / current_w
                new_h = int(current_h * ratio)
                new_w = target_size
            else:  # 输出高度
                ratio = target_size / current_h
                new_w = int(current_w * ratio)
                new_h = target_size
            
            # 通用的图像缩放函数
            def resize_image_array(img_array, new_width, new_height):
                # 统一处理输入格式
                if isinstance(img_array, torch.Tensor):
                    img_np = img_array.cpu().numpy()
                else:
                    img_np = img_array
                    
                # 转换为PIL并缩放
                img_pil = Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))
                img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                return np.array(img_pil) / 255.0
            
            # 应用缩放
            final_img = resize_image_array(final_img, new_w, new_h)
            final_mask = resize_image_array(final_mask, new_w, new_h)
            
            # 同步调整box2参数
            box2_w = int(box2_w * ratio)
            box2_h = int(box2_h * ratio)
            box2_x = int(box2_x * ratio)
            box2_y = int(box2_y * ratio)
            
            # 同步调整输出的new_img2和new_mask2
            # 添加检查以防止adjusted_img2_np为None
            if adjusted_img2_np is not None and adjusted_img2_np.size > 0:
                adjusted_h, adjusted_w = adjusted_img2_np.shape[:2]
                adjusted_new_h = int(adjusted_h * ratio)
                adjusted_new_w = int(adjusted_w * ratio)
                
                # 应用缩放到调整后的图像和遮罩
                adjusted_img2_np = resize_image_array(adjusted_img2_np, adjusted_new_w, adjusted_new_h)
                
                if adjusted_mask2_np is not None:
                    adjusted_mask2_np = resize_image_array(adjusted_mask2_np, adjusted_new_w, adjusted_new_h)
                else:
                    adjusted_mask2_np = np.ones((adjusted_new_h, adjusted_new_w))
            else:
                # 如果adjusted_img2_np为None或空，创建默认值
                adjusted_new_h = int(h1 * ratio) if 'h1' in locals() else int(512 * ratio)
                adjusted_new_w = int(w1 * ratio) if 'w1' in locals() else int(512 * ratio)
                adjusted_img2_np = np.zeros((adjusted_new_h, adjusted_new_w, 3), dtype=np.float32)
                adjusted_mask2_np = np.ones((adjusted_new_h, adjusted_new_w))


        else:
            if adjusted_mask2_np is None:
                if adjusted_img2_np is not None and adjusted_img2_np.size > 0:
                    adjusted_h, adjusted_w = adjusted_img2_np.shape[:2]
                    adjusted_mask2_np = np.ones((adjusted_h, adjusted_w))
                else:
                    # 如果adjusted_img2_np也为None或空，使用默认尺寸
                    adjusted_h = h1 if 'h1' in locals() else 512
                    adjusted_w = w1 if 'w1' in locals() else 512
                    adjusted_mask2_np = np.ones((adjusted_h, adjusted_w))


        # 透明背景处理
        if bg_mode == "transparent" and layout_mode not in ["左右-宽度对齐", "上下-高度对齐"]:
            # 如果 final_img 已经是 tensor
            if isinstance(final_img, torch.Tensor):
                if final_img.dim() == 3:  # (H, W, C) 格式
                    if final_img.shape[-1] == 3:  # RGB tensor
                        # 创建 RGBA 图像，将 final_mask 作为 alpha 通道
                        alpha_channel = final_mask if isinstance(final_mask, torch.Tensor) else torch.from_numpy(final_mask)
                        rgba_img = torch.cat([final_img, alpha_channel.unsqueeze(-1)], dim=-1)
                    else:  # 已经是 RGBA tensor
                        rgba_img = final_img
                elif final_img.dim() == 4:  # (1, H, W, C) 格式
                    if final_img.shape[-1] == 3:  # RGB tensor
                        # 创建 RGBA 图像，将 final_mask 作为 alpha 通道
                        alpha_channel = final_mask if isinstance(final_mask, torch.Tensor) else torch.from_numpy(final_mask)
                        if alpha_channel.dim() == 2:  # (H, W) 格式
                            alpha_channel = alpha_channel.unsqueeze(-1)  # 变为 (H, W, 1)
                        rgba_img = torch.cat([final_img, alpha_channel.unsqueeze(0).unsqueeze(-1)], dim=-1)
                    else:  # 已经是 RGBA tensor
                        rgba_img = final_img
                final_img_tensor = rgba_img.float()
            else:  # numpy array
                # 统一处理 numpy 数组
                final_img_np = final_img if not isinstance(final_img, torch.Tensor) else final_img.cpu().numpy()
                final_mask_np = final_mask if not isinstance(final_mask, torch.Tensor) else final_mask.cpu().numpy()
                
                # 确保 final_img_np 是正确的形状
                if final_img_np.ndim == 3 and final_img_np.shape[-1] == 4:
                    # 已经是 RGBA 格式
                    rgba_img = final_img_np
                else:
                    # 创建 RGBA 图像
                    if final_img_np.ndim == 3 and final_img_np.shape[-1] == 3:
                        # RGB 格式
                        rgba_img = np.zeros((final_img_np.shape[0], final_img_np.shape[1], 4), dtype=np.float32)
                        rgba_img[:, :, :3] = final_img_np
                        rgba_img[:, :, 3] = final_mask_np if final_mask_np.ndim == 2 else final_mask_np[:, :, 0]
                    elif final_img_np.ndim == 3 and final_img_np.shape[-1] == 1:
                        # 灰度图格式
                        rgba_img = np.zeros((final_img_np.shape[0], final_img_np.shape[1], 4), dtype=np.float32)
                        rgba_img[:, :, 0] = final_img_np[:, :, 0]
                        rgba_img[:, :, 1] = final_img_np[:, :, 0]
                        rgba_img[:, :, 2] = final_img_np[:, :, 0]
                        rgba_img[:, :, 3] = final_mask_np if final_mask_np.ndim == 2 else final_mask_np[:, :, 0]
                    else:
                        # 其他情况，假定是 RGB
                        rgba_img = np.zeros((final_img_np.shape[0], final_img_np.shape[1], 4), dtype=np.float32)
                        rgba_img[:, :, :3] = final_img_np
                        rgba_img[:, :, 3] = final_mask_np if final_mask_np.ndim == 2 else final_mask_np[:, :, 0]
                
                final_img_tensor = torch.from_numpy(rgba_img).float()
                if final_img_tensor.dim() == 3:
                    final_img_tensor = final_img_tensor.unsqueeze(0)

        else:
            # 非透明背景处理
            if isinstance(final_img, torch.Tensor):
                # 确保是3通道
                if final_img.dim() == 4 and final_img.shape[-1] == 4:
                    final_img = final_img[:, :, :, :3]  # 移除alpha通道
                elif final_img.dim() == 3 and final_img.shape[-1] == 4:
                    final_img = final_img[:, :, :3]  # 移除alpha通道
                if final_img.dim() == 3:
                    final_img_tensor = final_img.float().unsqueeze(0)
                else:
                    final_img_tensor = final_img.float()
            else:
                # numpy array 转 tensor
                final_img_np = final_img if not isinstance(final_img, torch.Tensor) else final_img.cpu().numpy()
                if final_img_np.ndim == 3 and final_img_np.shape[-1] == 4:
                    # 如果是 RGBA，只取 RGB 通道
                    final_img_np = final_img_np[:, :, :3]
                final_img_tensor = torch.from_numpy(final_img_np).float()
                if final_img_tensor.dim() == 3:
                    final_img_tensor = final_img_tensor.unsqueeze(0)

        final_mask_tensor = torch.from_numpy(final_mask).float().unsqueeze(0)
        box2 = (box2_w, box2_h, box2_x, box2_y)

        if adjusted_img2_np is None:
            adjusted_img2_np = np.zeros((h1, w1, 3), dtype=np.float32) if 'h1' in locals() and 'w1' in locals() else np.zeros((512, 512, 3), dtype=np.float32)
            
        if adjusted_mask2_np is None:
            adjusted_mask2_np = np.ones((h1, w1), dtype=np.float32) if 'h1' in locals() and 'w1' in locals() else np.ones((512, 512), dtype=np.float32)

        adjusted_img2_tensor = torch.from_numpy(adjusted_img2_np).float().unsqueeze(0)
        adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_np).float().unsqueeze(0)



        # 确保所有返回值都是tensor类型
        if not isinstance(final_img_tensor, torch.Tensor):
            final_img_tensor = torch.from_numpy(final_img_tensor).float() if isinstance(final_img_tensor, np.ndarray) else torch.tensor(final_img_tensor, dtype=torch.float32)

        if not isinstance(final_mask_tensor, torch.Tensor):
            final_mask_tensor = torch.from_numpy(final_mask_tensor).float() if isinstance(final_mask_tensor, np.ndarray) else torch.tensor(final_mask_tensor, dtype=torch.float32)

        if not isinstance(adjusted_img2_tensor, torch.Tensor):
            adjusted_img2_tensor = torch.from_numpy(adjusted_img2_tensor).float() if isinstance(adjusted_img2_tensor, np.ndarray) else torch.tensor(adjusted_img2_tensor, dtype=torch.float32)

        if not isinstance(adjusted_mask2_tensor, torch.Tensor):
            adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_tensor).float() if isinstance(adjusted_mask2_tensor, np.ndarray) else torch.tensor(adjusted_mask2_tensor, dtype=torch.float32)

        box2 = (box2_w, box2_h, box2_x, box2_y)
        return (final_img_tensor, final_mask_tensor, box2, adjusted_img2_tensor, adjusted_mask2_tensor)




class Image_Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout_mode": (["居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐", 
                                 "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                 "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐"],), 
                "bg_mode": (BJ_MODE,),  # 假设BJ_MODE已定义
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
        # 处理mask1_stack
        if mask1_stack and mask1 is not None:
            if hasattr(mask1, 'convert'):
                mask1_tensor = pil2tensor(mask1.convert('L'))
            else:
                if isinstance(mask1, torch.Tensor):
                    mask1_tensor = mask1 if len(mask1.shape) <= 3 else mask1.squeeze(-1) if mask1.shape[-1] == 1 else mask1
                else:
                    mask1_tensor = mask1
            (bg_mode_stack, mask_mode, output_mask, ignore_threshold, opacity, outline_thickness, smoothness, expand_width, expand_height, tapered_corners, sum_mask_min, sum_mask_max) = mask1_stack
            separated_result = Mask_transform_sum().separate(
                bg_mode=bg_mode_stack, mask_mode=mask_mode, output_mask=output_mask, 
                ignore_threshold=ignore_threshold, opacity=opacity, outline_thickness=outline_thickness, 
                smoothness=smoothness, expand_width=expand_width, expand_height=expand_height, 
                tapered_corners=tapered_corners, sum_mask_min=sum_mask_min, sum_mask_max=sum_mask_max, 
                base_image=image1, mask=mask1_tensor
            )
            mask1 = separated_result[1]

        # 处理mask2_stack
        if mask2_stack and mask2 is not None: 
            if hasattr(mask2, 'convert'):
                mask2_tensor = pil2tensor(mask2.convert('L'))
            else:  
                if isinstance(mask2, torch.Tensor):
                    mask2_tensor = mask2 if len(mask2.shape) <= 3 else mask2.squeeze(-1) if mask2.shape[-1] == 1 else mask2
                else:
                    mask2_tensor = mask2
            (bg_mode_stack, mask_mode, output_mask, ignore_threshold, opacity, outline_thickness, smoothness, expand_width, expand_height, tapered_corners, sum_mask_min, sum_mask_max) = mask2_stack
            separated_result = Mask_transform_sum().separate(  # 假设Mask_transform_sum类存在
                bg_mode=bg_mode_stack, mask_mode=mask_mode, output_mask=output_mask, 
                ignore_threshold=ignore_threshold, opacity=opacity, outline_thickness=outline_thickness, 
                smoothness=smoothness, expand_width=expand_width, expand_height=expand_height, 
                tapered_corners=tapered_corners, sum_mask_min=sum_mask_min, sum_mask_max=sum_mask_max, 
                base_image=image2, mask=mask2_tensor
            )
            mask2 = separated_result[1]

        # 转换图像为张量并检查形状
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
        
        # 处理图像形状，确保正确的维度
        if image1 is not None:
            # 确保有批次维度
            if len(image1.shape) == 3:  # (高度, 宽度, 通道) -> (1, 高度, 宽度, 通道)
                image1 = image1.unsqueeze(0)
            # 检查通道数
            if image1.shape[-1] != 3:
                raise ValueError(f"image1 应该是3通道图像，实际形状: {image1.shape}")
        
        if image2 is not None:
            if len(image2.shape) == 3:
                image2 = image2.unsqueeze(0)
            if image2.shape[-1] != 3:
                raise ValueError(f"image2 应该是3通道图像，实际形状: {image2.shape}")

        # 处理图像为空的情况
        if image1 is None and image2 is not None:
            image1 = image2.clone()
            
        if image1 is None and image2 is None:
            image1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image2 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        # 处理掩码
        if mask1 is None:
            # 根据image1形状创建掩码
            mask1 = torch.ones((image1.shape[0], image1.shape[1], image1.shape[2]), dtype=torch.float32)
        else:
            # 确保掩码是张量
            if not isinstance(mask1, torch.Tensor):
                mask1 = torch.from_numpy(np.array(mask1)) if hasattr(mask1, 'numpy') else torch.from_numpy(mask1)
            
            # 调整掩码形状
            if len(mask1.shape) == 2:  # (高度, 宽度) -> (1, 高度, 宽度)
                mask1 = mask1.unsqueeze(0)
            # 检查掩码与图像尺寸是否匹配
            if mask1.shape[1:] != image1.shape[1:3]:
                raise ValueError(f"mask1 尺寸 {mask1.shape[1:]} 与 image1 尺寸 {image1.shape[1:3]} 不匹配")

        if mask2 is None:
            mask2 = torch.ones((image2.shape[0], image2.shape[1], image2.shape[2]), dtype=torch.float32)
        else:
            if not isinstance(mask2, torch.Tensor):
                mask2 = torch.from_numpy(np.array(mask2)) if hasattr(mask2, 'numpy') else torch.from_numpy(mask2)
            
            if len(mask2.shape) == 2:
                mask2 = mask2.unsqueeze(0)
            if mask2.shape[1:] != image2.shape[1:3]:
                raise ValueError(f"mask2 尺寸 {mask2.shape[1:]} 与 image2 尺寸 {image2.shape[1:3]} 不匹配")

        # 执行合并操作
        final_img_tensor, final_mask_tensor, box2, image2, mask2 = Pair_Merge().composite(  # 假设Pair_Merge类存在
            layout_mode=layout_mode, bg_mode=bg_mode, size_mode=size_mode, 
            target_size=target_size, divider_thickness=divider_thickness, 
            image1=image1, image2=image2, mask1=mask1, mask2=mask2
        )

        return (final_img_tensor, final_mask_tensor, box2, image2, mask2, )


#endregion----图像-双图合并---总控制---------



#region -----Image_transform_layer----------------------------------#



class Image_transform_layer:   # 唯一问题：对齐是按照左上角初始化的，，并安装背景图进行裁切
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
                "bg_mode": (["image", "transparent", "white", "black", "red", "green", "blue"],),
                "x_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "front_img_sacle": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "extrant_to_block": ("BOOLEAN", {"default": False}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (COLORSMAP,{"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS , {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bg_img": ("IMAGE",),
                "front_img": ("IMAGE",),
                "front_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "line_mask")
    FUNCTION = "process_image"
    CATEGORY = "Apt_Preset/image"




    def process_image(self, bg_mode, blending_mode, blend_strength, front_img_sacle, x_position, y_position, rotation, opacity, edge_detection, edge_thickness, edge_color, mask_expansion, mask_blur, extrant_to_block, front_img=None, bg_img=None, front_mask=None):
        error_img = Image.new("RGB", (64, 64), color="red")
        error_tensor = pil2tensor(error_img)
        error_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        
        if front_img is None and bg_img is None:
            return (error_tensor, error_mask, error_mask)
        
        # 确保有前景图像
        if front_img is None:
            front_img, bg_img = bg_img, None
            
        fg_pil = tensor2pil(front_img)
        width, height = fg_pil.size
        
        # 确保前景是RGBA模式
        if fg_pil.mode != "RGBA":
            fg_pil = fg_pil.convert("RGBA")
        
        # 处理前景遮罩
        if front_mask is not None:
            mask_pil = tensor2pil(front_mask).convert('L')
            if mask_pil.size != fg_pil.size:
                mask_pil = mask_pil.resize(fg_pil.size, Image.LANCZOS)
        else:
            mask_pil = Image.new('L', fg_pil.size, 255)
        
        # 处理遮罩（矩形化、模糊、扩展）
        if extrant_to_block:
            mask_pil = apply_extrant_to_block(mask_pil)
        
        mask_pil = blur_and_expand_mask(mask_pil, mask_blur, mask_expansion)
        
        # 创建背景
        background = create_background(bg_mode, width, height, bg_img)
        if background is None:
            return (error_tensor, error_mask, error_mask)
        bg_pil = tensor2pil(background).convert("RGBA")
        
        # 记录背景尺寸
        bg_width, bg_height = bg_pil.size
        
        # 创建与背景相同尺寸的画布
        canvas = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
        
        # 应用变换（缩放和旋转）
        scale = front_img_sacle
        transformed_fg, transformed_mask, _, _ = apply_position_transform(
            canvas,  # 传递一个空画布作为背景
            fg_pil, 
            mask_pil, 
            scale, 
            x_position, 
            y_position, 
            rotation
        )
        
        # 确保变换后的图像和遮罩尺寸与背景一致
        if transformed_fg.size != (bg_width, bg_height):
            temp_fg = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
            temp_fg.paste(transformed_fg, (0, 0))
            transformed_fg = temp_fg
            
            temp_mask = Image.new('L', (bg_width, bg_height), 0)
            temp_mask.paste(transformed_mask, (0, 0))
            transformed_mask = temp_mask
        
        # 透明度处理
        if opacity < 1.0:
            # 应用透明度到变换后的前景图像
            r, g, b, a = transformed_fg.split()
            a = a.point(lambda x: int(x * opacity))
            transformed_fg = Image.merge("RGBA", (r, g, b, a))
        
        # === 关键修改：使用遮罩限制混合模式的应用范围 ===
        if blending_mode != "normal":
            # 创建一个临时背景，只在遮罩区域应用混合模式
            masked_bg = bg_pil.copy()
            masked_fg = transformed_fg.copy()
            
            # 创建与背景相同尺寸的遮罩
            full_size_mask = transformed_mask.copy()
            
            # 将前景和背景转换为RGBA模式
            masked_bg = masked_bg.convert("RGBA")
            masked_fg = masked_fg.convert("RGBA")
            
            # 在遮罩区域应用混合模式
            temp_result = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
            temp_result.paste(masked_bg, (0, 0))
            
            # 只在遮罩区域应用混合模式
            for x in range(bg_width):
                for y in range(bg_height):
                    if full_size_mask.getpixel((x, y)) > 127:  # 仅处理遮罩非透明区域
                        bg_pixel = masked_bg.getpixel((x, y))
                        fg_pixel = masked_fg.getpixel((x, y))
                        
                        # 创建一个单像素图像用于混合
                        pixel_bg = Image.new('RGBA', (1, 1), bg_pixel)
                        pixel_fg = Image.new('RGBA', (1, 1), fg_pixel)
                        
                        # 应用混合模式
                        blended_pixel_img = apply_blending_mode(
                            pixel_bg, pixel_fg, blending_mode, blend_strength
                        )
                        
                        # 获取混合后的像素值并设置到结果图像
                        blended_pixel = blended_pixel_img.getpixel((0, 0))
                        temp_result.putpixel((x, y), blended_pixel)
            
            # 将混合后的区域与原始背景合并
            result_img = Image.alpha_composite(bg_pil, temp_result)
        else:
            # 如果是正常混合模式，直接合并
            result_img = Image.alpha_composite(bg_pil, transformed_fg)
        
        # 更新结果遮罩
        result_mask = transformed_mask
        
        # 边缘检测
        if edge_detection:
            result_img, edge_mask_img = apply_edge_detection(
                result_img, result_mask, edge_detection, edge_thickness, edge_color
            )
        else:
            edge_mask_img = Image.new('L', result_img.size, 0)
        
        # 确保最终输出尺寸与背景一致
        if result_img.size != (bg_width, bg_height):
            result_img = result_img.resize((bg_width, bg_height), Image.LANCZOS)
        
        # 转换为RGB模式输出
        result_img = result_img.convert("RGB")
        
        # 转换回张量
        result_tensor = pil2tensor(result_img)
        mask_tensor = pil2mask(result_mask)
        line_mask_tensor = pil2mask(edge_mask_img)
        
        return (result_tensor, mask_tensor, line_mask_tensor)




def apply_edge_detection(input_img, input_mask, edge_detection, edge_thickness, edge_color):   # 图像边缘检测--
    edge_mask = np.zeros_like(np.array(input_mask)) 
    if edge_detection:
        # 支持 color_mapping 中的颜色名称或十六进制颜色值
        if edge_color in color_mapping:
            r, g, b = color_mapping[edge_color]
        else:
            edge_color_hex = edge_color.lstrip('#')
            r, g, b = tuple(int(edge_color_hex[i:i+2], 16) for i in (0, 2, 4))
        edge_image = Image.new("RGBA", input_img.size, (0, 0, 0, 0))
        edge_draw = ImageDraw.Draw(edge_image)
        contours, _ = cv2.findContours(np.array(input_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            for i in range(edge_thickness):
                points = [tuple(point[0]) for point in contour]
                edge_draw.line(points, fill=(r, g, b, 255), width=edge_thickness-i+1)       
        output_img = Image.alpha_composite(input_img, edge_image)
        cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)    
    else:
        output_img = input_img.copy()  
    edge_mask =Image.fromarray(edge_mask)
    return output_img, edge_mask



def apply_position_transform(bj_img, img, mask, scale, x_position, y_position, rotation):
    # 确保所有输入都是正确的模式
    img = img.convert('RGBA')
    mask = mask.convert('L')
    bg_width, bg_height = bj_img.size
    bj_img = bj_img.convert('RGBA')
    
    # 计算mask的边界框和中心
    mask_array = np.array(mask)
    mask_coords = np.where(mask_array > 128)
    
    if len(mask_coords[0]) > 0:
        x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
        y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
        mask_center_x = (x_min + x_max) // 2
        mask_center_y = (y_min + y_max) // 2
    else:
        # 如果mask为空，使用图像中心
        mask_center_x, mask_center_y = img.width // 2, img.height // 2
    
    # 创建足够大的画布以容纳所有变换
    max_dim = int(max(img.width, img.height, bg_width, bg_height) * 3)
    canvas_width, canvas_height = max_dim, max_dim
    
    # 计算在画布上放置背景的位置
    bj_pos_x = (canvas_width - bg_width) // 2
    bj_pos_y = (canvas_height - bg_height) // 2
    
    # 创建画布并粘贴背景图像
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    canvas.paste(bj_img, (bj_pos_x, bj_pos_y))
    
    # 计算前景图像在画布上的初始位置
    img_pos_x = bj_pos_x + x_position
    img_pos_y = bj_pos_y + y_position
    
    # 应用变换
    if scale != 1.0 or rotation != 0:
        # 计算变换后的图像尺寸
        new_width, new_height = int(img.width * scale), int(img.height * scale)
        
        # 缩放图像和遮罩
        transformed_img = img.resize((new_width, new_height), Image.LANCZOS)
        transformed_mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        # 计算变换后的mask中心（考虑缩放）
        new_mask_center_x = int(mask_center_x * scale)
        new_mask_center_y = int(mask_center_y * scale)
        
        # 应用旋转（围绕mask中心）
        if rotation != 0:
            # 旋转图像和遮罩
            transformed_img = transformed_img.rotate(
                -rotation,  # PIL的rotate函数使用逆时针旋转
                center=(new_mask_center_x, new_mask_center_y),
                expand=True,  # 允许图像扩展以包含所有旋转后的内容
                resample=Image.BICUBIC
            )
            
            transformed_mask = transformed_mask.rotate(
                -rotation,
                center=(new_mask_center_x, new_mask_center_y),
                expand=True,
                resample=Image.NEAREST
            )
        
        # 计算旋转后的图像中心偏移
        offset_x = (transformed_img.width - new_width) // 2
        offset_y = (transformed_img.height - new_height) // 2
        
        # 更新mask中心位置
        new_mask_center_x += offset_x
        new_mask_center_y += offset_y
        
        # 计算粘贴位置（左上角）
        paste_x = img_pos_x - new_mask_center_x + mask_center_x
        paste_y = img_pos_y - new_mask_center_y + mask_center_y
        
        # 将变换后的图像和遮罩粘贴到画布上
        canvas.paste(transformed_img, (paste_x, paste_y), transformed_mask)
        
        # 创建最终的遮罩图像
        result_mask = Image.new('L', (canvas_width, canvas_height), 0)
        result_mask.paste(transformed_mask, (paste_x, paste_y))
        
        # 计算最终的mask中心位置（相对于原始背景）
        final_x = x_position
        final_y = y_position
    else:
        # 如果没有变换，直接粘贴原图
        canvas.paste(img, (img_pos_x, img_pos_y), mask)
        
        # 创建最终的遮罩图像
        result_mask = Image.new('L', (canvas_width, canvas_height), 0)
        result_mask.paste(mask, (img_pos_x, img_pos_y))
        
        # 计算最终的mask中心位置（相对于原始背景）
        final_x = x_position
        final_y = y_position
    
    # === 关键修改：在输出前进行裁切 ===
    # 计算相对于原始背景的可见区域
    bg_visible_x1 = max(0, bj_pos_x)
    bg_visible_y1 = max(0, bj_pos_y)
    bg_visible_x2 = min(bj_pos_x + bg_width, canvas_width)
    bg_visible_y2 = min(bj_pos_y + bg_height, canvas_height)
    
    # 裁切画布和遮罩到背景大小
    cropped_img = canvas.crop((bg_visible_x1, bg_visible_y1, bg_visible_x2, bg_visible_y2))
    cropped_mask = result_mask.crop((bg_visible_x1, bg_visible_y1, bg_visible_x2, bg_visible_y2))
    
    # 如果裁切后的尺寸小于背景尺寸，创建与背景相同大小的新图像并居中粘贴
    if cropped_img.size != (bg_width, bg_height):
        final_img = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
        final_mask = Image.new('L', (bg_width, bg_height), 0)
        
        paste_x = (bg_width - cropped_img.width) // 2
        paste_y = (bg_height - cropped_img.height) // 2
        
        final_img.paste(cropped_img, (paste_x, paste_y))
        final_mask.paste(cropped_mask, (paste_x, paste_y))
    else:
        final_img = cropped_img
        final_mask = cropped_mask
    
    return final_img, final_mask, final_x, final_y




#endregion ----------------Image_transform_sum---------------------------------------#



class Mask_transform_sum:
    def __init__(self):
        self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["image", "transparent", "white", "black", "red", "green", "blue"],),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block"], {"default": "original"}),
                "output_mask": (["mask", "1_white", "2_black", "3_red", "4_green", "5_blue", "6_yellow", "7_cyan", "8_magenta"], {"default": "mask"}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "sum_mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "sum_mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            },
            "optional": {"base_image": ("IMAGE",), "mask": ("MASK",)}
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"
    
    def separate(self, bg_mode, mask_mode="fill", output_mask="mask", 
                 ignore_threshold=100, opacity=0.8, outline_thickness=1, 
                 smoothness=1, expand_width=0, expand_height=0, 
                 tapered_corners=True, sum_mask_min=0.0, sum_mask_max=1.0,
                 base_image=None, mask=None):
        import cv2
        import numpy as np
        import torch
        from PIL import Image, ImageFilter
        
        if mask is None:
            empty_mask = torch.zeros(64, 64, dtype=torch.float32)
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32), empty_mask.unsqueeze(0))
        
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
        individual_masks = [np.zeros_like(binary_mask) for _ in range(8)]
        
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=np.uint8)
        
        for i, contour in enumerate(sorted_contours[:8]):
            temp_mask = np.zeros_like(binary_mask)
            use_block = "_block" in mask_mode
            
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
            
            if expand_width != 0 or expand_height != 0:
                expand_amount = int((abs(expand_width) + abs(expand_height)) / 2)
                if expand_amount > 0:
                    if expand_width > 0 or expand_height > 0:
                        temp_mask = cv2.dilate(temp_mask, kernel, iterations=expand_amount)
                    else:
                        temp_mask = cv2.erode(temp_mask, kernel, iterations=expand_amount)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
            individual_masks[i] = temp_mask
        
        if smoothness > 0:
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_pil = final_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(final_mask_pil)
            
            for i in range(8):
                if np.sum(individual_masks[i]) > 0:
                    mask_pil = Image.fromarray(individual_masks[i])
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
                    individual_masks[i] = np.array(mask_pil)
        
        if base_image is None:
            h, w = final_mask.shape
            base_image_np = np.zeros((h, w, 3), dtype=np.float32)
        else:
            base_image_np = base_image[0].cpu().numpy() * 255.0
            base_image_np = base_image_np.astype(np.float32)
        
        h, w, _ = base_image_np.shape
        
        # 直接使用bg_mode和base_image创建背景
        background = create_background(
            bg_mode=bg_mode, 
            width=w, 
            height=h, 
            f_imge=base_image[0] if base_image is not None else None
        ).cpu().numpy() * 255.0
        
        if background.shape[2] == 4:
            background = background[:, :, :3]
        
        combined_image = background.copy()
        
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        mask_max = torch.max(final_mask_tensor)
        mask_max = mask_max if mask_max > 0 else 1
        final_mask_tensor = (final_mask_tensor / mask_max) * (sum_mask_max - sum_mask_min) + sum_mask_min
        final_mask_tensor = torch.clamp(final_mask_tensor, min=0.0, max=1.0)
        final_mask = final_mask_tensor.cpu().numpy() * 255
        final_mask = final_mask.astype(np.uint8)
        
        for i, contour in enumerate(sorted_contours[:8]):
            if i >= 8:
                break
            color_name = ["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"][i]
            color = np.array(self.colors[color_name], dtype=np.float32)
            mask_float = individual_masks[i].astype(np.float32) / 255.0
            
            for c in range(3):
                combined_image[:, :, c] = (mask_float * (opacity * color[c] + (1 - opacity) * combined_image[:, :, c]) + 
                                         (1 - mask_float) * combined_image[:, :, c])
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)
        
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        
        white_1_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 0 else torch.from_numpy(individual_masks[0]).float() / 255.0
        black_2_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 1 else torch.from_numpy(individual_masks[1]).float() / 255.0
        red_3_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 2 else torch.from_numpy(individual_masks[2]).float() / 255.0
        green_4_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 3 else torch.from_numpy(individual_masks[3]).float() / 255.0
        blue_5_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 4 else torch.from_numpy(individual_masks[4]).float() / 255.0
        yellow_6_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 5 else torch.from_numpy(individual_masks[5]).float() / 255.0
        cyan_7_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 6 else torch.from_numpy(individual_masks[6]).float() / 255.0
        magenta_8_tensor = torch.zeros_like(mask) if len(sorted_contours) <= 7 else torch.from_numpy(individual_masks[7]).float() / 255.0
        
        white_1_tensor = white_1_tensor.unsqueeze(0)
        black_2_tensor = black_2_tensor.unsqueeze(0)
        red_3_tensor = red_3_tensor.unsqueeze(0)
        green_4_tensor = green_4_tensor.unsqueeze(0)
        blue_5_tensor = blue_5_tensor.unsqueeze(0)
        yellow_6_tensor = yellow_6_tensor.unsqueeze(0)
        cyan_7_tensor = cyan_7_tensor.unsqueeze(0)
        magenta_8_tensor = magenta_8_tensor.unsqueeze(0)
        
        mask_outputs = {
            "mask": final_mask_tensor, 
            "1_white": white_1_tensor, 
            "2_black": black_2_tensor, 
            "3_red": red_3_tensor, 
            "4_green": green_4_tensor, 
            "5_blue": blue_5_tensor, 
            "6_yellow": yellow_6_tensor, 
            "7_cyan": cyan_7_tensor, 
            "8_magenta": magenta_8_tensor
        }
        
        selected_mask = mask_outputs[output_mask]
        return (combined_image_tensor, selected_mask)



class Image_solo_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),
            }
        }

    CATEGORY = "Apt_Preset/image"
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



class Image_solo_crop:
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

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "stitch")
    FUNCTION = "inpaint_crop"

    def inpaint_crop(self, image, mask, expand_width, expand_height, rescale_factor, smoothness):


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
    

    

class XXXmask_stack_sum:   # 建设中--旋转方法可行，遮罩总控设计----
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "align_mode": (["height", "width", "original"], {"default": "height"}),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "bj_img": ("IMAGE",),
                "mask1": ("MASK",),   
                "fj_img": ("IMAGE",),            
                "mask2": ("MASK",),
                "mask1_stack": ("MARK_STACK",),
                "mask2_stack": ("MASK_STACK",),

            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "BOX")
    RETURN_NAMES = ("composite", "mask", "line_mask", "box2")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"


    def process( self, align_mode, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, opacity, 
                bj_img=None, fj_img=None, mask1=None, mask2=None, mask1_stack=None, mask2_stack=None):
        # 颜色映射表
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

#------------------------------------------------------------------------

        if mask1_stack and mask1 is not None:
            (ignore_threshold, opacity, outline_thickness, extrant_to_block, smoothness, out_color, fill)= mask1_stack
            bj_img, mask1 = visualize_mask_on_image(bj_img, mask1,
                ignore_threshold, opacity, outline_thickness,
                extrant_to_block, smoothness, out_color, fill)
        else:
            bj_img, mask1 = visualize_mask_on_image( bj_img, mask1,
                ignore_threshold=0, opacity=1.00, outline_thickness=1,
                extrant_to_block=True, smoothness=1, out_color="blue", fill=True )

        #---------------------------------插入遮罩更新mask2,转张量-----------------------------
        if mask2_stack and mask2 is not None: 
            if hasattr(mask2, 'convert'):  # PIL Image
                mask2_tensor = pil2tensor(mask2.convert('L'))
            else:  
                if isinstance(mask2, torch.Tensor):
                    mask2_tensor = mask2 if len(mask2.shape) <= 3 else mask2.squeeze(-1) if mask2.shape[-1] == 1 else mask2
                else:
                    mask2_tensor = mask2
            (mask_mode, output_mask, ignore_threshold, opacity, outline_thickness, 
            smoothness, expand, tapered_corners, sum_mask_min, sum_mask_max) = mask2_stack 

            separated_result = Mask_transform_sum().separate(
                ignore_threshold, opacity, mask_mode, outline_thickness, smoothness, expand, 
                tapered_corners, sum_mask_min, sum_mask_max, output_mask, base_image=fj_img, mask=mask2_tensor)
            mask2 = separated_result[1]  



        if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
        if bj_img is None: bj_img = fj_img.clone()
            
        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        
        canvas_width, canvas_height = bj_pil.size
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2

#----------------------------------

        # 记录原始mask尺寸
        original_mask_width, original_mask_height = 0, 0
        if mask2 is not None:
            mask_np = mask2[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            original_mask_width, original_mask_height = mask_pil.size
            mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
        else:
            mask_pil = Image.new("L", fj_pil.size, 255)
            original_mask_width, original_mask_height = mask_pil.size
        
        mask_center_x, mask_center_y = mask_pil.size[0] // 2, mask_pil.size[1] // 2
        
        # 计算布局调整后的尺寸和缩放比例
        scale_x, scale_y = 1.0, 1.0
        if align_mode == "height":
            height_ratio = canvas_height / fj_pil.size[1]
            new_width = int(fj_pil.size[0] * height_ratio)
            scale_x = height_ratio
            scale_y = height_ratio
            fj_pil = fj_pil.resize((new_width, canvas_height), Image.LANCZOS)
            mask_pil = mask_pil.resize((new_width, canvas_height), Image.LANCZOS)
            mask_center_x, mask_center_y = new_width // 2, canvas_height // 2
        elif align_mode == "width":
            width_ratio = canvas_width / fj_pil.size[0]
            new_height = int(fj_pil.size[1] * width_ratio)
            scale_x = width_ratio
            scale_y = width_ratio
            fj_pil = fj_pil.resize((canvas_width, new_height), Image.LANCZOS)
            mask_pil = mask_pil.resize((canvas_width, new_height), Image.LANCZOS)
            mask_center_x, mask_center_y = canvas_width // 2, new_height // 2
        
        # 保存调整后但未旋转缩放的尺寸
        adjusted_mask_width, adjusted_mask_height = mask_pil.size
        
        # 确保rotation是数值类型
        rotation = float(rotation)  # 新增的类型转换
        
        # 应用旋转和缩放
        if rotation != 0 or scale != 1.0:
            mask_pil = mask_pil.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            fj_pil = fj_pil.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            
            if scale != 1.0:
                new_size = (int(fj_pil.size[0] * scale), int(fj_pil.size[1] * scale))
                scale_x *= scale
                scale_y *= scale
                fj_pil = fj_pil.resize(new_size, Image.LANCZOS)
                mask_pil = mask_pil.resize(new_size, Image.LANCZOS)
            
            mask_center_x, mask_center_y = fj_pil.size[0] // 2, fj_pil.size[1] // 2
        
        # 确保composite_pil正确初始化
        composite_pil = bj_pil.copy()
        
        # 计算mask在画布上的位置（考虑偏移，不应用缩放）
        x_position = canvas_center_x - mask_center_x + x_offset
        y_position = canvas_center_y - mask_center_y + y_offset
        
        # 计算实际粘贴位置（相对于画布左上角）
        paste_x = max(0, x_position)
        paste_y = max(0, y_position)
        
        # 应用遮罩到前景图像
        fj_with_mask = fj_pil.copy()
        fj_with_mask.putalpha(mask_pil)
        
        # 调整透明度
        if opacity < 1.0:
            r, g, b, a = fj_with_mask.split()
            a = a.point(lambda p: p * opacity)
            fj_with_mask = Image.merge("RGBA", (r, g, b, a))
        
        # 裁切超出画布的部分
        cropped_fj = fj_with_mask.crop((
            max(0, -x_position),
            max(0, -y_position),
            min(fj_pil.size[0], canvas_width - x_position),
            min(fj_pil.size[1], canvas_height - y_position)
        ))
        
        # 合成图像
        composite_pil.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
        
        # 重构完整尺寸的遮罩
        full_size_mask = Image.new("L", composite_pil.size, 0)
        full_size_mask.paste(mask_pil.crop((
            max(0, -x_position),
            max(0, -y_position),
            min(fj_pil.size[0], canvas_width - x_position),
            min(fj_pil.size[1], canvas_height - y_position)
        )), (paste_x, paste_y))
        
        # 计算mask的边界框（相对于mask自身）
        bbox = mask_pil.getbbox()  # 返回格式：(left, top, right, bottom)
        
        if bbox:
            bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
            bbox_width = bbox_right - bbox_left
            bbox_height = bbox_bottom - bbox_top
            
            # 计算mask边界框中心点相对于mask的坐标
            bbox_center_x = bbox_left + bbox_width // 2
            bbox_center_y = bbox_top + bbox_height // 2
            
            # 计算mask有效区域中心点在画布上的实际坐标
            bbox_actual_x = x_position + bbox_center_x
            bbox_actual_y = y_position + bbox_center_y
            
            # 计算相对于画布中心的偏移（不考虑缩放）
            bbox_center_x_rel = bbox_actual_x - canvas_center_x
            bbox_center_y_rel = bbox_actual_y - canvas_center_y
        else:
            # 如果mask为空，返回默认值
            bbox_width = 0
            bbox_height = 0
            bbox_center_x_rel = 0
            bbox_center_y_rel = 0
        
        # 应用边缘检测
        if edge_detection:
            # 获取颜色值
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0  # 默认黑色
            
            # 确保遮罩为二值图像
            threshold = 128
            mask_array = np.array(full_size_mask)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)
            
            # 应用边缘检测
            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            
            # 转换为OpenCV格式进行轮廓检测
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
            
            # 合并边缘到合成图
            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            
            # 生成边缘遮罩
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            line_mask_pil = Image.new("L", composite_pil.size, 0)
        
        # 准备输出
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
        
        if composite_np.shape[2] == 4: pass
        else: composite_np = np.dstack([composite_np, np.ones_like(composite_np[..., 0:1])])
        
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
        
        # box_info格式：(mask有效区域的宽度, mask有效区域的高度, 有效区域中心点相对于画布中心的X坐标, 有效区域中心点相对于画布中心的Y坐标)
        box_info = (bbox_width, bbox_height, bbox_center_x_rel, bbox_center_y_rel)
        
        return (composite_tensor, mask_tensor, line_mask_tensor, box_info)




























































