import os
import torch
import numpy as np
import folder_paths
from PIL import ImageOps, ImageEnhance, Image, ImageOps, ImageChops, ImageFilter, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import io
from typing import Literal, Any
import math
from comfy.utils import common_upscale
import typing as t
from pathlib import Path







from math import ceil, sqrt
from ..main_unit import *


#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False

try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False




#region--------------def--------layout----------------------

font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (160, 85, 15),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (102, 102, 102),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}


COLORS = ["white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]


ALIGN_OPTIONS = ["center", "top", "bottom"]                 
ROTATE_OPTIONS = ["text center", "image center"]
JUSTIFY_OPTIONS = ["center", "left", "right"]
PERSPECTIVE_OPTIONS = ["top", "bottom", "left", "right"]


def align_text(align, img_height, text_height, text_pos_y, margins):
    if align == "center":
        text_plot_y = img_height / 2 - text_height / 2 + text_pos_y
    elif align == "top":
        text_plot_y = text_pos_y + margins
    elif align == "bottom":
        text_plot_y = img_height - text_height + text_pos_y - margins
    return text_plot_y

def justify_text(justify, img_width, line_width, margins):
    if justify == "left":
        text_plot_x = 0 + margins
    elif justify == "right":
        text_plot_x = img_width - line_width - margins
    elif justify == "center":
        text_plot_x = img_width/2 - line_width/2
    return text_plot_x

def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

def draw_masked_text(text_mask, text,
                     font_name, font_size,
                     margins, line_spacing,
                     position_x, position_y, 
                     align, justify,
                     rotation_angle, rotation_options):
    draw = ImageDraw.Draw(text_mask)
    font_folder = "fonts"
    font_file = os.path.join(font_folder, font_name)
    resolved_font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), font_file)
    font = ImageFont.truetype(str(resolved_font_path), size=font_size)
    text_lines = text.split('\n')
    max_text_width = 0
    max_text_height = 0
    for line in text_lines:
        line_width, line_height = get_text_size(draw, line, font)
        line_height = line_height + line_spacing
        max_text_width = max(max_text_width, line_width)
        max_text_height = max(max_text_height, line_height)
    image_width, image_height = text_mask.size
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    text_pos_y = position_y
    sum_text_plot_y = 0
    text_height = max_text_height * len(text_lines)
    for line in text_lines:
        line_width, _ = get_text_size(draw, line, font)
        text_plot_x = position_x + justify_text(justify, image_width, line_width, margins)
        text_plot_y = align_text(align, image_height, text_height, text_pos_y, margins)
        draw.text((text_plot_x, text_plot_y), line, fill=255, font=font)
        text_pos_y += max_text_height
        sum_text_plot_y += text_plot_y
    text_center_x = text_plot_x + max_text_width / 2
    text_center_y = sum_text_plot_y / len(text_lines)
    if rotation_options == "text center":
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(text_center_x, text_center_y))
    elif rotation_options == "image center":
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(image_center_x, image_center_y))
    return rotated_text_mask

def draw_text_on_image(draw, y_position, bar_width, bar_height, text, font, text_color, font_outline):
    text_width, text_height = get_text_size(draw, text, font)
    if font_outline == "thin":
        outline_thickness = text_height // 40
    elif font_outline == "thick":
        outline_thickness = text_height // 20
    elif font_outline == "extra thick":
        outline_thickness = text_height // 10

    text_lines = text.split('\n')
    if len(text_lines) == 1:
        x = (bar_width - text_width) // 2
        y = y_position + (bar_height - text_height) // 2 - (bar_height * 0.10)
        if font_outline == "none":
            draw.text((x, y), text, fill=text_color, font=font)
        else:
            draw.text((x, y), text, fill=text_color, font=font, stroke_width=outline_thickness, stroke_fill='black')
    elif len(text_lines) > 1:
        text_width, text_height = get_text_size(draw, text_lines[0], font)
        x = (bar_width - text_width) // 2
        y = y_position + (bar_height - text_height * 2) // 2 - (bar_height * 0.15)
        if font_outline == "none":
            draw.text((x, y), text_lines[0], fill=text_color, font=font)
        else:
            draw.text((x, y), text_lines[0], fill=text_color, font=font, stroke_width=outline_thickness, stroke_fill='black')

        text_width, text_height = get_text_size(draw, text_lines[1], font)
        x = (bar_width - text_width) // 2
        y = y_position + (bar_height - text_height * 2) // 2 + text_height - (bar_height * 0.00)
        if font_outline == "none":
            draw.text((x, y), text_lines[1], fill=text_color, font=font)
        else:
            draw.text((x, y), text_lines[1], fill=text_color, font=font, stroke_width=outline_thickness, stroke_fill='black')

def get_font_size(draw, text, max_width, max_height, font_path, max_font_size):
    max_width = max_width * 0.9
    font_size = max_font_size
    font = ImageFont.truetype(str(font_path), size=font_size)
    text_lines = text.split('\n')[:2]
    if len(text_lines) == 2:
        font_size = min(max_height//2, max_font_size)
        font = ImageFont.truetype(str(font_path), size=font_size)

    max_text_width = 0
    longest_line = text_lines[0]
    for line in text_lines:
        line_width, line_height = get_text_size(draw, line, font)
        if line_width > max_text_width:
            longest_line = line
        max_text_width = max(max_text_width, line_width)

    text_width, text_height = get_text_size(draw, text, font)
    while max_text_width > max_width or text_height > 0.88 * max_height / len(text_lines):
        font_size -= 1
        font = ImageFont.truetype(str(font_path), size=font_size)
        max_text_width, text_height = get_text_size(draw, longest_line, font)
    return font

def combine_images(images, layout_direction='horizontal'):

    if layout_direction == 'horizontal':
        combined_width = sum(image.width for image in images)
        combined_height = max(image.height for image in images)
    else:
        combined_width = max(image.width for image in images)
        combined_height = sum(image.height for image in images)

    combined_image = Image.new('RGB', (combined_width, combined_height))

    x_offset = 0
    y_offset = 0  # Initialize y_offset for vertical layout
    for image in images:
        combined_image.paste(image, (x_offset, y_offset))
        if layout_direction == 'horizontal':
            x_offset += image.width
        else:
            y_offset += image.height

    return combined_image

def apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color):
    for i, image in enumerate(images):
        # Apply the outline
        if outline_thickness > 0:
            image = ImageOps.expand(image, outline_thickness, fill=outline_color)

        if border_thickness > 0:
            image = ImageOps.expand(image, border_thickness, fill=border_color)
        images[i] = image
    return images

def get_color_values(color, color_mapping):
    color_rgb = color_mapping.get(color, (0, 0, 0)) 
    return color_rgb 

def apply_resize_image(image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', supersample='true', factor: int = 2, width: int = 1024, height: int = 1024, resample='bicubic'):
    if mode == 'rescale':
        new_width, new_height = int(original_width * factor), int(original_height * factor)
    else:
        m = rounding_modulus
        original_ratio = original_height / original_width
        height = int(width * original_ratio)
        new_width = width if width % m == 0 else width + (m - width % m)
        new_height = height if height % m == 0 else height + (m - height % m)
    resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}
    if supersample == 'true':
        image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))
    return resized_image

def draw_text(panel, text, font_name, font_size, font_color, font_outline_thickness, font_outline_color, bg_color, margins, line_spacing, position_x, position_y, align, justify, rotation_angle, rotation_options):
    draw = ImageDraw.Draw(panel)
    font_folder = "fonts"
    font_file = os.path.join(font_folder, font_name)
    resolved_font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), font_file)
    font = ImageFont.truetype(str(resolved_font_path), size=font_size)
    text_lines = text.split('\n')
    max_text_width = 0
    max_text_height = 0
    for line in text_lines:
        line_width, line_height = get_text_size(draw, line, font)
        line_height = line_height + line_spacing
        max_text_width = max(max_text_width, line_width)
        max_text_height = max(max_text_height, line_height)
    image_center_x = panel.width / 2
    image_center_y = panel.height / 2
    text_pos_y = position_y
    sum_text_plot_y = 0
    text_height = max_text_height * len(text_lines)
    for line in text_lines:
        line_width, line_height = get_text_size(draw, line, font)
        text_plot_x = position_x + justify_text(justify, panel.width, line_width, margins)
        text_plot_y = align_text(align, panel.height, text_height, text_pos_y, margins)
        draw.text((text_plot_x, text_plot_y), line, fill=font_color, font=font, stroke_width=font_outline_thickness, stroke_fill=font_outline_color)
        text_pos_y += max_text_height
        sum_text_plot_y += text_plot_y
    text_center_x = text_plot_x + max_text_width / 2
    text_center_y = sum_text_plot_y / len(text_lines)
    if rotation_options == "text center":
        rotated_panel = panel.rotate(rotation_angle, center=(text_center_x, text_center_y), resample=Image.BILINEAR)
    elif rotation_options == "image center":
        rotated_panel = panel.rotate(rotation_angle, center=(image_center_x, image_center_y), resample=Image.BILINEAR)
    return rotated_panel

def text_panel(image_width, image_height, text, font_name, font_size, font_color, font_outline_thickness, font_outline_color, background_color, margins, line_spacing, position_x, position_y, align, justify, rotation_angle, rotation_options):
    size = (image_width, image_height)
    panel = Image.new('RGB', size, background_color)
    image_out = draw_text(panel, text, font_name, font_size, font_color, font_outline_thickness, font_outline_color, background_color, margins, line_spacing, position_x, position_y, align, justify, rotation_angle, rotation_options)
    return image_out

def crop_and_resize_image(image, target_width, target_height):
    width, height = image.size
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    if aspect_ratio > target_aspect_ratio:
        crop_width = int(height * target_aspect_ratio)
        crop_height = height
        left = (width - crop_width) // 2
        top = 0
    else:
        crop_height = int(width / target_aspect_ratio)
        crop_width = width
        left = 0
        top = (height - crop_height) // 2
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
    
    return cropped_image

def create_and_paste_panel(page, border_thickness, outline_thickness,
                        panel_width, panel_height, page_width,
                        panel_color, bg_color, outline_color,
                        images, i, j, k, len_images,):
    panel = Image.new("RGB", (panel_width, panel_height), panel_color)
    if k < len_images:
        img = images[k]
        image = crop_and_resize_image(img, panel_width, panel_height)
        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        panel.paste(image, (0, 0))
    panel = ImageOps.expand(panel, border=outline_thickness, fill=outline_color)
    panel = ImageOps.expand(panel, border=border_thickness, fill=bg_color)
    new_panel_width, new_panel_height = panel.size
    page.paste(panel, (j * new_panel_width, i * new_panel_height))


#endregion----------------------layout----------------------





#region------------------effect特效-------------------------------------------------------


class ImageEffects:
    @staticmethod
    def _convert_to_tensor(gray_img):
        """Helper method to convert grayscale numpy array to proper tensor format"""
        # Convert to 3 channels
        img_3ch = np.stack([gray_img, gray_img, gray_img], axis=-1)
        # Convert to float32 and normalize to 0-1
        img_float = img_3ch.astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension
        return torch.from_numpy(img_float).unsqueeze(0)

    @staticmethod
    def grayscale(img_tensor):
        rgb_coeff = torch.tensor([0.299, 0.587, 0.114]).to(img_tensor.device)
        grayscale = torch.sum(img_tensor * rgb_coeff, dim=-1, keepdim=True)
        return grayscale.repeat(1, 1, 1, 3)

    @staticmethod
    def flip_h(img_tensor):
        return torch.flip(img_tensor, dims=[2])

    @staticmethod
    def flip_v(img_tensor):
        return torch.flip(img_tensor, dims=[1])

    @staticmethod
    def posterize(img_tensor, levels):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        posterized = ImageOps.posterize(img_pil, bits=levels)
        img_np = np.array(posterized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def sharpen(img_tensor, factor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        enhancer = ImageEnhance.Sharpness(img_pil)
        sharpened = enhancer.enhance(factor)
        img_np = np.array(sharpened).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def contrast(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        contrasted = ImageOps.autocontrast(img_pil)
        img_np = np.array(contrasted).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def equalize(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        equalized = ImageOps.equalize(img_pil)
        img_np = np.array(equalized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def sepia(img_tensor, strength=1.0):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        r = strength
        sepia_filter = (
            0.393 + 0.607 * (1 - r), 0.769 - 0.769 * (1 - r), 0.189 - 0.189 * (1 - r), 0,
            0.349 - 0.349 * (1 - r), 0.686 + 0.314 * (1 - r), 0.168 - 0.168 * (1 - r), 0,
            0.272 - 0.272 * (1 - r), 0.534 - 0.534 * (1 - r), 0.131 + 0.869 * (1 - r), 0
        )
        
        sepia_img = img_pil.convert('RGB', sepia_filter)
        img_np = np.array(sepia_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def blur(img_tensor, strength):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=strength))
        img_np = np.array(blurred).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)
    
    @staticmethod
    def emboss(img_tensor, strength):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        embossed = img_pil.filter(ImageFilter.EMBOSS)
        enhancer = ImageEnhance.Contrast(embossed)
        embossed = enhancer.enhance(strength)
        img_np = np.array(embossed).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def palette(img_tensor, color_count):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        paletted = img_pil.convert('P', palette=Image.ADAPTIVE, colors=color_count)
        reduced = paletted.convert('RGB')
        img_np = np.array(reduced).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def enhance(img_tensor, strength=0.5):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        contrast = ImageEnhance.Contrast(img_pil)
        img_pil = contrast.enhance(1 + (0.2 * strength))
        
        sharpener = ImageEnhance.Sharpness(img_pil)
        img_pil = sharpener.enhance(1 + (0.3 * strength))
        
        color = ImageEnhance.Color(img_pil)
        img_pil = color.enhance(1 + (0.1 * strength))
        
        equalized = ImageOps.equalize(img_pil)
        equalized_np = np.array(equalized)
        original_np = np.array(img_pil)
        blend_factor = 0.2 * strength
        blended = (1 - blend_factor) * original_np + blend_factor * equalized_np
        
        final_np = np.clip(blended, 0, 255).astype(np.uint8)
        img_np = np.array(final_np).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def solarize(img_tensor, threshold=0.5):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        solarized = ImageOps.solarize(img_pil, threshold=int(threshold * 255))
        img_np = np.array(solarized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def denoise(img_tensor, strength=3):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        blurred = img_pil.filter(ImageFilter.MedianFilter(size=strength))
        img_np = np.array(blurred).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def vignette(img_tensor, intensity=0.75):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        height, width = img_np.shape[:2]
        
        # Create radial gradient
        center_x, center_y = width/2, height/2
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize and adjust intensity
        vignette_mask = 1 - (dist_from_center * intensity / max_dist)
        vignette_mask = np.clip(vignette_mask, 0, 1)
        vignette_mask = vignette_mask[..., np.newaxis]
        
        # Apply vignette
        vignetted = (img_np * vignette_mask).astype(np.uint8)
        img_np = vignetted.astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def glow_edges(img_tensor, strength=0.75):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Edge detection
        edges = img_pil.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Enhance edges
        enhancer = ImageEnhance.Brightness(edges)
        glowing = enhancer.enhance(1.5)
        
        # Blend with original
        blend_factor = strength
        blended = Image.blend(img_pil, glowing, blend_factor)
        
        img_np = np.array(blended).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    @staticmethod
    def new_effect(img_tensor, param1=1.0):
        # Your new effect implementation
        pass

    @staticmethod
    def edge_detect(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        edges = cv2.Canny(gray, 100, 200)
        return ImageEffects._convert_to_tensor(edges)

    @staticmethod
    def edge_gradient(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ImageEffects._convert_to_tensor(magnitude)

    @staticmethod
    def lineart_clean(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        edges = cv2.Canny(blur, 50, 150)
        return ImageEffects._convert_to_tensor(edges)

    @staticmethod
    def lineart_anime(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        edge = cv2.Canny(gray, 50, 150)
        edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
        return ImageEffects._convert_to_tensor(edge)

    @staticmethod
    def threshold(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return ImageEffects._convert_to_tensor(binary)

    @staticmethod
    def pencil_sketch(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (13, 13), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        return ImageEffects._convert_to_tensor(sketch)

    @staticmethod
    def sketch_lines(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
        return ImageEffects._convert_to_tensor(edges)

    @staticmethod
    def bold_lines(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        edges = cv2.Canny(gray, 100, 200)
        dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        return ImageEffects._convert_to_tensor(dilated)

    @staticmethod
    def depth_edges(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ImageEffects._convert_to_tensor(magnitude)

    @staticmethod
    def relief_light(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        embossed = cv2.filter2D(gray, -1, kernel) + 128
        return ImageEffects._convert_to_tensor(embossed)

    @staticmethod
    def edge_enhance(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        embossed = cv2.filter2D(gray, -1, kernel) + 128
        return ImageEffects._convert_to_tensor(embossed)

    @staticmethod
    def edge_morph(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        return ImageEffects._convert_to_tensor(gradient)

    @staticmethod
    def relief_shadow(img_tensor):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        kernel = np.array([[0,0,0], [0,1,0], [0,0,-1]])
        relief = cv2.filter2D(gray, -1, kernel) + 128
        return ImageEffects._convert_to_tensor(relief)


class img_effect_Load:
    def __init__(self):
        self.effects = ImageEffects()
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for filename in os.listdir(input_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                files.append(filename)

        available_styles = [
            "original", "grayscale", "enhance", "flip_h",
            "flip_v", "posterize", "sharpen", "contrast",
            "equalize", "sepia", "blur", "emboss", "palette",
            "solarize", "denoise", "vignette", "glow_edges",
            "edge_detect", "edge_gradient", "lineart_clean",
            "lineart_anime", "threshold", "pencil_sketch",
            "sketch_lines", "bold_lines", "depth_edges",
            "relief_light", "edge_enhance", "edge_morph",
            "relief_shadow"
        ]
        
        return {"required": {
            "image": (sorted(files), {"image_upload": True}),
            "output_01_fx": (available_styles, {"default": "original"}),
            "output_02_fx": (available_styles, {"default": "grayscale"}),
            "output_03_fx": (available_styles, {"default": "flip_h"}),
            "output_04_fx": (available_styles, {"default": "flip_v"})
        },
        "optional": {
            "image_input": ("IMAGE",)
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("output1", "output2", "output3", "output4", )
    FUNCTION = "load_image_and_process"
    CATEGORY = "Apt_Preset/imgEffect/😺backup"

    def load_image_and_process(self, image, output_01_fx, output_02_fx, output_03_fx, output_04_fx, image_input=None):
        
        if image_input is not None:
            output_image = image_input
            formatted_name = "piped_image"

        else:
            image_path = folder_paths.get_annotated_filepath(image)
            formatted_name = os.path.basename(image_path)
            # Always strip extension now
            formatted_name = os.path.splitext(formatted_name)[0]
            
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image = np.array(img).astype(np.float32) / 255.0
                    output_image = torch.from_numpy(image).unsqueeze(0)
            except Exception as e:
                print(f"Error processing image: {e}")
                raise e

        style_map = {
            "original": output_image,
            "grayscale": self.effects.grayscale(output_image),
            "enhance": self.effects.enhance(output_image),
            "flip_h": self.effects.flip_h(output_image),
            "flip_v": self.effects.flip_v(output_image),
            "posterize": self.effects.posterize(output_image, 4),
            "sharpen": self.effects.sharpen(output_image, 1.0),
            "contrast": self.effects.contrast(output_image),
            "equalize": self.effects.equalize(output_image),
            "sepia": self.effects.sepia(output_image, 1.0),
            "blur": self.effects.blur(output_image, 5.0),
            "emboss": self.effects.emboss(output_image, 1.0),
            "palette": self.effects.palette(output_image, 8),
            "solarize": self.effects.solarize(output_image, 0.5),
            "denoise": self.effects.denoise(output_image, 3),
            "vignette": self.effects.vignette(output_image, 0.75),
            "glow_edges": self.effects.glow_edges(output_image, 0.75),
            "edge_detect": self.effects.edge_detect(output_image),
            "edge_gradient": self.effects.edge_gradient(output_image),
            "lineart_clean": self.effects.lineart_clean(output_image),
            "lineart_anime": self.effects.lineart_anime(output_image),
            "threshold": self.effects.threshold(output_image),
            "pencil_sketch": self.effects.pencil_sketch(output_image),
            "sketch_lines": self.effects.sketch_lines(output_image),
            "bold_lines": self.effects.bold_lines(output_image),
            "depth_edges": self.effects.depth_edges(output_image),
            "relief_light": self.effects.relief_light(output_image),
            "edge_enhance": self.effects.edge_enhance(output_image),
            "edge_morph": self.effects.edge_morph(output_image),
            "relief_shadow": self.effects.relief_shadow(output_image)
        }

        return (
                style_map[output_01_fx],
                style_map[output_02_fx],
                style_map[output_03_fx],
                style_map[output_04_fx],)


class img_effect_CircleWarp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "radius": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp_image"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageCircleWarp"

    def ellipse_warp(self, img, strength, radius, center_x, center_y):
        height, width = img.shape[:2]
        center_x = int(width * center_x)
        center_y = int(height * center_y)
        
        y, x = np.indices((height, width))
        
        dx = (x - center_x) / (width/2)
        dy = (y - center_y) / (height/2)
        r = np.sqrt(dx**2 + dy**2)
        
        influence = np.clip(1.0 - r / radius, 0, 1)
        
        influence = influence * influence * (3 - 2 * influence)
        
        scale = 1.0 + strength * influence
        
        x_new = center_x + (x - center_x) * scale
        y_new = center_y + (y - center_y) * scale
        
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        return cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)

    def warp_image(self, image, strength, radius, center_x, center_y):
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        result = self.ellipse_warp(img, strength, radius, center_x, center_y)
        
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)


class img_effect_Stretch:
    """图像拉伸变形节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "direction": (["horizontal", "vertical"],),
                "position": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.01}),
                "stretch_width": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01}),
                "transition": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "funny_mirror"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageStretch"

    def create_control_points(self, size, center_pos, stretch_width, transition, strength):
        """创建样条插值的控制点"""
        # 计算关键区域的边界
        half_stretch = stretch_width * size / 2
        stretch_start = max(center_pos - half_stretch, half_stretch)  # 确保不会太靠近边界
        stretch_end = min(center_pos + half_stretch, size - half_stretch)  # 确保不会超出边界
        trans_pixels = min(transition * size, half_stretch)  # 限制过渡区域大小
        
        # 创建更多的控制点以实现更平滑的过渡
        x = np.array([
            0,                                     # 起始点（无变形）
            max(0.1 * size, stretch_start - trans_pixels * 2),  # 远过渡区开始
            max(0.1 * size, stretch_start - trans_pixels),      # 近过渡区开始
            stretch_start,                         # 拉伸区开始
            center_pos,                           # 中心点
            stretch_end,                          # 拉伸区结束
            min(size - 0.1 * size, stretch_end + trans_pixels),      # 近过渡区结束
            min(size - 0.1 * size, stretch_end + trans_pixels * 2),  # 远过渡区结束
            size                                  # 终止点（无变形）
        ])
        
        # 确保x坐标严格递增
        x = np.sort(x)
        eps = 1e-6 * size
        x[1:] = np.maximum(x[1:], x[:-1] + eps)
        
        # 归一化x坐标到[0,1]区间
        x = x / size
        
        # 计算变形量，使用正弦函数实现平滑过渡
        y = np.zeros_like(x)
        
        # 拉伸区域使用固定变形量
        center_region = slice(3, 6)  # 拉伸区域的索引范围
        y[center_region] = (strength - 1) * half_stretch
        
        # 过渡区域使用正弦函数实现平滑过渡
        left_transition = slice(1, 3)   # 左过渡区域
        right_transition = slice(6, 8)  # 右过渡区域
        
        # 左侧过渡
        t_left = np.linspace(0, np.pi/2, len(y[left_transition]))
        y[left_transition] = (strength - 1) * half_stretch * np.sin(t_left)
        
        # 右侧过渡
        t_right = np.linspace(np.pi/2, np.pi, len(y[right_transition]))
        y[right_transition] = (strength - 1) * half_stretch * np.cos(t_right)
        
        # 确保边界点无变形
        y[0] = 0   # 起始点
        y[-1] = 0  # 终止点
        
        return x, y

    def create_spline_mapping(self, size, center_pos, stretch_width, transition, strength):
        """创建基于样条的变形映射"""
        # 创建控制点
        x, y = self.create_control_points(size, center_pos, stretch_width, transition, strength)
        
        # 创建三次样条插值器
        spline = CubicSpline(x, y, bc_type='natural')
        
        # 生成所有位置的映射
        positions = np.linspace(0, 1, size)
        deformations = spline(positions)
        
        # 计算新的坐标映射
        new_positions = np.arange(size, dtype=np.float32)
        new_positions += deformations
        
        return new_positions

    def funny_mirror(self, image, strength, direction, position, stretch_width, transition):
        # 转换图像格式
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        height, width = img.shape[:2]
        
        # 创建坐标网格
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 根据方向应用变形
        if direction == "horizontal":
            center_pos = int(position * height)
            # 创建水平方向的变形映射
            y_new = self.create_spline_mapping(height, center_pos, stretch_width, transition, strength)
            # 应用变形
            y = y_new[y]
            x_new = x
        else:
            center_pos = int(position * width)
            # 创建垂直方向的变形映射
            x_new = self.create_spline_mapping(width, center_pos, stretch_width, transition, strength)
            # 应用变形
            x = x_new[x]
            y_new = y
        
        # 确保坐标在有效范围内
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
        
        # 应用变形并转换回tensor格式
        result = cv2.remap(img, x.astype(np.float32), 
                          y.astype(np.float32), 
                          cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_REFLECT)
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result,)


class img_effect_WaveWarp:
    """图像波浪扭曲节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "wave_frequency": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "wave_direction": (["horizontal", "vertical", "radial"],),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "wave_warp"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageWaveWarp"

    def wave_warp(self, image, strength, wave_frequency, wave_direction, center_x, center_y):
        # 转换图像格式
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        height, width = img.shape[:2]
        center_x = int(width * center_x)
        center_y = int(height * center_y)
        
        # 创建网格
        y, x = np.indices((height, width))
        
        # 根据波浪方向计算偏移
        if wave_direction == "horizontal":
            # 水平波浪
            phase = y / height * 2 * np.pi * wave_frequency
            x_offset = np.sin(phase) * strength * width / 20
            y_offset = np.zeros_like(x_offset)
        elif wave_direction == "vertical":
            # 垂直波浪
            phase = x / width * 2 * np.pi * wave_frequency
            x_offset = np.zeros_like(x)
            y_offset = np.sin(phase) * strength * height / 20
        else:  # radial
            # 径向波浪
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            phase = r / (width/2) * 2 * np.pi * wave_frequency
            angle = np.arctan2(dy, dx)
            magnitude = np.sin(phase) * strength * width / 20
            x_offset = magnitude * np.cos(angle)
            y_offset = magnitude * np.sin(angle)
        
        # 应用偏移
        x_new = x + x_offset
        y_new = y + y_offset
        
        # 确保坐标在有效范围内
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        # 应用变形并转换回tensor格式
        result = cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)


class img_effect_Liquify:
    """液化变形节点 - 支持多种液化效果"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "mode": (["PUSH", "PULL", "TWIST", "PINCH"],),
                "feather": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_liquify"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def liquify_effect(self, img, center_x, center_y, radius, strength, mode, feather):
        height, width = img.shape[:2]
        
        # 转换相对坐标到绝对坐标
        center_x = int(width * center_x)
        center_y = int(height * center_y)
        radius = int(width * radius)  # 使用图像宽度来缩放半径
        
        # 创建网格
        y, x = np.indices((height, width))
        
        # 计算到中心点的距离和角度
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # 计算影响因子
        influence = np.clip(1.0 - distance / (radius * feather), 0, 1)
        influence = influence * influence * (3 - 2 * influence)  # 平滑过渡
        
        # 根据模式计算变形
        if mode == "PUSH":
            # 向外推效果
            scale = 1.0 + strength * influence
            x_offset = dx * (scale - 1)
            y_offset = dy * (scale - 1)
        elif mode == "PULL":
            # 向内拉效果
            scale = 1.0 - strength * influence
            x_offset = dx * (scale - 1)
            y_offset = dy * (scale - 1)
        elif mode == "TWIST":
            # 扭转效果
            twist_angle = strength * np.pi * influence
            cos_theta = np.cos(twist_angle)
            sin_theta = np.sin(twist_angle)
            x_offset = (dx * cos_theta - dy * sin_theta - dx) * influence
            y_offset = (dx * sin_theta + dy * cos_theta - dy) * influence
        else:  # PINCH
            # 挤压效果
            scale = 1.0 + strength * influence * (distance / radius)
            x_offset = dx * (scale - 1)
            y_offset = dy * (scale - 1)
        
        # 应用变形
        x_new = x + x_offset
        y_new = y + y_offset
        
        # 确保坐标在有效范围内
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        # 使用双三次插值进行重映射
        return cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), 
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def apply_liquify(self, image, center_x, center_y, radius, strength, mode, feather):
        try:
            # 转换图像格式
            img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # 应用液化效果
            result = self.liquify_effect(img, center_x, center_y, radius, strength, mode, feather)
            
            # 转换回tensor格式
            result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
            return (result,)
            
        except Exception as e:
            print(f"液化效果应用失败: {str(e)}")
            return (image,)


#endregion---------------特效-------------------------------------------------------


#region -----------------create shape--------------------



class create_mulcolor_img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = 'color_images'
    CATEGORY = "Apt_Preset/imgEffect"

    def color_images(self, width, height):
        red_image = Image.new('RGB', (width, height), color=(255, 0, 0))  # RGB 值对应 #FFF000
        green_image = Image.new('RGB', (width, height), color=(0, 255, 0))  # RGB 值对应 #00FF00
        blue_image = Image.new('RGB', (width, height), color=(0, 0, 255))  # RGB 值对应 #0000FF
        cyan_image = Image.new('RGB', (width, height), color=(0, 255, 255))  # RGB 值对应 #00FFFF
        magenta_image = Image.new('RGB', (width, height), color=(255, 0, 255))  # RGB 值对应 #FF00FF
        yellow_image = Image.new('RGB', (width, height), color=(255, 255, 0))  # RGB 值对应 #FFFF00
        black_image = Image.new('RGB', (width, height), color=(0, 0, 0))  # RGB 值对应 #000000
        white_image = Image.new('RGB', (width, height), color=(255, 255, 255))  # RGB 值对应 #FFFFFF

        return (
            pil2tensor(red_image),
            pil2tensor(green_image),
            pil2tensor(blue_image),
            pil2tensor(cyan_image),
            pil2tensor(magenta_image),
            pil2tensor(yellow_image),
            pil2tensor(black_image),
            pil2tensor(white_image)
        )



#endregion-----------------------------------------------------------------------------------------------------



#region------------------layout----------------------


class lay_ImageGrid:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"batch_img": ("IMAGE",), "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}), "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grid_images"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    def grid_images(self, batch_img, rows, cols):
        batch_img = batch_img.cpu().numpy()
        batch_size, height, width, channels = batch_img.shape
        grid_width = width * cols
        grid_height = height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height))
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            img = Image.fromarray((batch_img[i] * 255).astype(np.uint8))
            x = col * width
            y = row * height
            grid_image.paste(img, (x, y))
        grid_image = np.array(grid_image).astype(np.float32) / 255.0
        grid_image = torch.from_numpy(grid_image)[None,]
        return (grid_image,)


class lay_MaskGrid:
    @classmethod 
    def INPUT_TYPES(cls):
        return {"required": {"batch_mask": ("MASK",), "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}), "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "grid_masks"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def grid_masks(self, batch_mask, rows, cols):
        batch_mask = batch_mask.cpu().numpy()
        batch_size, height, width = batch_mask.shape
        grid_width = width * cols
        grid_height = height * rows
        grid_mask = Image.new('L', (grid_width, grid_height))
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            mask = Image.fromarray((batch_mask[i] * 255).astype(np.uint8))
            x = col * width
            y = row * height
            grid_mask.paste(mask, (x, y))
        grid_mask = np.array(grid_mask).astype(np.float32) / 255.0
        grid_mask = torch.from_numpy(grid_mask)[None,]
        return (grid_mask,)


class lay_edge_cut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 10}),
                "row_split_method": (["uniform", "edge_detection"],),
                "col_split_method": (["uniform", "edge_detection"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "split_image")
    FUNCTION = "split_image"
    CATEGORY = "Apt_Preset/imgEffect"

    def find_split_positions(self, image, num_splits, is_vertical, split_method):
        if split_method == "uniform":
            size = image.shape[1] if is_vertical else image.shape[0]
            return [i * size // (num_splits + 1) for i in range(1, num_splits + 1)]
        else:  # edge_detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges, axis=0) if is_vertical else np.sum(edges, axis=1)
            
            window_size = len(edge_density) // (num_splits + 1) // 2
            smoothed_density = np.convolve(edge_density, np.ones(window_size)/window_size, mode='same')
            
            split_positions = []
            for i in range(1, num_splits + 1):
                start = i * len(smoothed_density) // (num_splits + 1) - window_size
                end = i * len(smoothed_density) // (num_splits + 1) + window_size
                split = start + np.argmin(smoothed_density[start:end])
                split_positions.append(split)
            return split_positions

    def split_image(self, image, rows, cols, row_split_method, col_split_method):
        
        # 确保输入图像是 4D tensor (B, H, W, C)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 将图像从 PyTorch tensor 转换为 NumPy 数组
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        height, width = img_np.shape[:2]
        print(f"Original image size: {width}x{height}")

        # 找到分割位置
        vertical_splits = self.find_split_positions(img_np, cols - 1, True, col_split_method)
        horizontal_splits = self.find_split_positions(img_np, rows - 1, False, row_split_method)

        # 创建预览图
        preview_img = image.clone()
        
        # 在预览图上画线
        green_line = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
        for x in vertical_splits:
            preview_img[:, :, x:x+2, :] = green_line
        for y in horizontal_splits:
            preview_img[:, y:y+2, :, :] = green_line
        
        # 分割图片
        split_images = []
        h_splits = [0] + horizontal_splits + [height]
        v_splits = [0] + vertical_splits + [width]
        
        # 计算最大的分割尺寸
        max_height = max([h_splits[i+1] - h_splits[i] for i in range(len(h_splits) - 1)])
        max_width = max([v_splits[i+1] - v_splits[i] for i in range(len(v_splits) - 1)])
        
        for i in range(len(h_splits) - 1):
            for j in range(len(v_splits) - 1):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell = image[:, top:bottom, left:right, :]
                
                # 填充到最大尺寸
                pad_bottom = max_height - (bottom - top)
                pad_right = max_width - (right - left)
                cell = torch.nn.functional.pad(cell, (0, 0, 0, pad_right, 0, pad_bottom), mode='constant')
                
                split_images.append(cell)

        split_images = torch.cat(split_images, dim=0)  
        return (preview_img, split_images)



class lay_compare_img:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": "text"}),
                    "text2": ("STRING", {"multiline": True, "default": "text"}),
                    "footer_height": ("INT", {"default": 100, "min": 0, "max": 1024}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),                
                    "mode": (["normal", "dark"],),
                    "border_thickness": ("INT", {"default": 20, "min": 0, "max": 1024}),                
                    "use_second_image_as_reference": ("BOOLEAN", {"default": False}),  # 新增布尔选项
            },
                "optional": {
                    "image1": ("IMAGE",),
                    "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "layout"
    CATEGORY = "Apt_Preset/imgEffect"
    
    def layout(self, text1, text2, footer_height, font_name, font_size, mode, border_thickness, 
               use_second_image_as_reference=False, image1=None, image2=None):  # 添加参数
        if mode == "normal":
            font_color = "black"
            bg_color = "white"
        else:
            font_color = "white"
            bg_color = "black"
        if image1 is not None and image2 is not None:
            img1 = tensor2pil(image1)
            img2 = tensor2pil(image2)
            
            # 根据布尔选项决定参考尺寸
            if use_second_image_as_reference:
                # 使用 image2 作为参考尺寸
                reference_img = img2
                image_width, image_height = reference_img.width, reference_img.height
                # 调整 image1 尺寸以匹配参考尺寸
                if img1.width != image_width or img1.height != image_height:
                    img1 = apply_resize_image(img1, image_width, image_height, 8, "rescale", "false", 1, 256, "lanczos")
            else:
                # 使用 image1 作为参考尺寸（默认行为）
                reference_img = img1
                image_width, image_height = reference_img.width, reference_img.height
                # 调整 image2 尺寸以匹配参考尺寸
                if img2.width != image_width or img2.height != image_height:
                    img2 = apply_resize_image(img2, image_width, image_height, 8, "rescale", "false", 1, 256, "lanczos")
                    
            margins = 50
            line_spacing = 0
            position_x = 0
            position_y = 0
            align = "center"
            rotation_angle = 0
            rotation_options = "image center"
            font_outline_thickness = 0
            font_outline_color = "black"
            footer_align = "center"
            outline_thickness = border_thickness//2
            border_thickness = border_thickness//2
            if footer_height > 0:
                text_panel1 = text_panel(image_width, footer_height, text1, font_name, font_size, font_color, font_outline_thickness, font_outline_color, bg_color, margins, line_spacing, position_x, position_y, align, footer_align, rotation_angle, rotation_options)
                combined_img1 = combine_images([img1, text_panel1], 'vertical')
            if outline_thickness > 0:
                combined_img1 = ImageOps.expand(combined_img1, outline_thickness, fill=bg_color)
            if footer_height > 0:
                text_panel2 = text_panel(image_width, footer_height, text2, font_name, font_size, font_color, font_outline_thickness, font_outline_color, bg_color, margins, line_spacing, position_x, position_y, align, footer_align, rotation_angle, rotation_options)
                combined_img2 = combine_images([img2, text_panel2], 'vertical')
            if outline_thickness > 0:
                combined_img2 = ImageOps.expand(combined_img2, outline_thickness, fill=bg_color)
            result_img = combine_images([combined_img1, combined_img2], 'horizontal')
        else:
            result_img = Image.new('RGB', (512,512), bg_color)
        if border_thickness > 0:
            result_img = ImageOps.expand(result_img, border_thickness, bg_color)
        return (pil2tensor(result_img),)


class lay_text_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": False, "default": "text"}),
                    "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                    "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),  
                    "background_color": ("STRING", {"default": "#000000"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "text_color": ("STRING", {"default": "#FF0000"}),
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                },   
                "optional": { 
                    "image_bg": ("IMAGE",),
                    "text_bg": ("IMAGE",),
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "composite_text"
    CATEGORY = "Apt_Preset/imgEffect"

    def composite_text(self, text, font_name, font_size,
                    position_x, position_y, align, justify, text_color,
                    image_width, image_height, background_color,
                    rotation_angle, 
                    text_bg=None, image_bg=None):
        
        line_spacing = 0
        margins = 0
        rotation_options="text center"
        if image_bg is not None:
            if isinstance(image_bg, torch.Tensor):
                image_3d = image_bg[0].cpu().numpy()
                back_image = Image.fromarray(np.clip(image_3d * 255.0, 0, 255).astype(np.uint8))
            else:
                back_image = image_bg.convert("RGB")
        else:
            # 如果没有提供 image_bg，但提供了 text_bg，则用 text_bg 的尺寸创建背景图
            if text_bg is not None:
                if isinstance(text_bg, torch.Tensor):
                    image_3d = text_bg[0].cpu().numpy()
                    tmp_img = Image.fromarray(np.clip(image_3d * 255.0, 0, 255).astype(np.uint8))
                else:
                    tmp_img = text_bg.convert("RGB")
                image_width, image_height = tmp_img.size
                back_image = Image.new('RGB', tmp_img.size, background_color)
            else:
                # 如果都没有输入，使用默认尺寸
                back_image = Image.new('RGB', (image_width, image_height), background_color)

        # 文本图层处理
        if text_bg is not None:
            if isinstance(text_bg, torch.Tensor):
                image_3d = text_bg[0].cpu().numpy()
                text_image = Image.fromarray(np.clip(image_3d * 255.0, 0, 255).astype(np.uint8))
            else:
                text_image = text_bg.convert("RGB")
        else:
            text_image = Image.new('RGB', back_image.size, text_color)

        # 确保所有图像尺寸一致
        if text_image.size != back_image.size:
            text_image = text_image.resize(back_image.size)

        # 创建文本遮罩
        text_mask = Image.new('L', back_image.size)
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing, 
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        # 合成最终图像
        image_out = Image.composite(text_image, back_image, rotated_text_mask)

        # 遮罩输出（保持与背景图相同尺寸）
        mask_out = np.array(rotated_text_mask).astype(np.float32) / 255.0
        mask_out = torch.from_numpy(mask_out).unsqueeze(0)

        return (pil2tensor(image_out), mask_out)





class lay_image_grid_note:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, t.Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "rows": ("INT", {"default": 1, "min": 1}),
                "columns": ("INT", {"default": 1, "min": 1}),
                "gap": ("FLOAT", {"default": 10.0, "min": 0, "max": 50}),
                "font_size": ("INT", {"default": 60, "min": 1}),
                "row_texts": ("STRING", {"default": "a@b@b"}),
                "col_texts": ("STRING", {"default": "1@2@3"}),
                "bg_color": ("STRING", {"default": "#0E0000"}),
            }
        }

    FUNCTION = "create_grid"
    CATEGORY = "Apt_Preset/imgEffect/😺backup"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def create_grid(
        self,
        images: torch.Tensor,
        rows: int,
        columns: int,
        gap: float,
        font_size: int,
        row_texts: str,
        col_texts: str,
        bg_color: str,
    ) -> tuple[torch.Tensor]:
        bg_color_tuple = self._parse_color(bg_color)
        pillow_images = [tensor_to_pillow(image) for image in images]

        total_images = len(pillow_images)
        calculated_rows, calculated_cols = self._calculate_grid_dimensions(rows, columns, total_images)

        row_labels = row_texts.split("@") if row_texts else []
        col_labels = col_texts.split("@") if col_texts else []

        # 空白填充
        required_slots = calculated_rows * calculated_cols
        if len(pillow_images) < required_slots:
            pillow_images = pillow_images + ["blank"] * (required_slots - len(pillow_images))
        else:
            pillow_images = pillow_images[:required_slots]

        grid_image = self._create_grid_image(
            pillow_images,
            calculated_rows,
            calculated_cols,
            gap,
            font_size,
            bg_color_tuple,
            row_labels,
            col_labels
        )

        tensor_grid = pillow_to_tensor(grid_image)
        return (tensor_grid,)

    def _calculate_grid_dimensions(self, rows: int, cols: int, total_images: int) -> tuple[int, int]:
        original_rows, original_cols = rows, cols

        if rows == 1 and cols == 1:
            size = math.ceil(math.sqrt(total_images))
            return size, size
        elif rows == 1:
            return math.ceil(total_images / cols), cols
        elif cols == 1:
            return rows, math.ceil(total_images / rows)
        else:
            return rows, cols

    def _create_grid_image(
        self,
        images: list[Image.Image],
        rows: int,
        cols: int,
        gap: float,
        font_size: int,
        bg_color: tuple[int, int, int],
        row_labels: list[str],
        col_labels: list[str]
    ) -> Image.Image:
        valid_images = [img for img in images if img != "blank"]
        if not valid_images:
            return Image.new("RGB", (100, 100), bg_color)

        image_size = valid_images[0].size
        grid_width = image_size[0] * cols + gap * (cols - 1)
        grid_height = image_size[1] * rows + gap * (rows - 1)

        grid_image = Image.new("RGB", (int(grid_width), int(grid_height)), bg_color)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        blank_image = Image.new("RGB", image_size, bg_color)

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            x = col * (image_size[0] + gap)
            y = row * (image_size[1] + gap)

            current_image = images[idx] if idx < len(images) and images[idx] != "blank" else blank_image
            grid_image.paste(current_image, (int(x), int(y)))

            self._draw_index_label(
                grid_image,
                (row, col),
                (int(x), int(y)),
                image_size,
                font,
                font_size,
                row_labels,
                col_labels
            )

        return grid_image

    def _draw_index_label(
        self,
        grid_image: Image.Image,
        position: tuple[int, int],
        offset: tuple[int, int],
        image_size: tuple[int, int],
        font: ImageFont.FreeTypeFont,
        font_size: int,
        row_labels: list[str],
        col_labels: list[str]
    ):
        draw = ImageDraw.Draw(grid_image)
        row_idx, col_idx = position

        row_text = row_labels[row_idx] if row_labels and row_idx < len(row_labels) else str(row_idx + 1)
        col_text = col_labels[col_idx] if col_labels and col_idx < len(col_labels) else str(col_idx + 1)
        label_text = f"({row_text}, {col_text})"

        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        padding = max(2, int(font_size * 0.3))
        bg_rect_size = (text_width + padding * 2, text_height + padding * 2)

        overlay = Image.new("RGBA", bg_rect_size, (255, 255, 255, 192))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.text((padding, 0), label_text, fill=(0, 0, 0, 255), font=font)

        grid_image.paste(overlay, (offset[0], offset[1]), overlay)

    def _parse_color(self, color_str: str) -> tuple[int, int, int]:
        if color_str.startswith("#"):
            r = int(color_str[1:3], 16)
            g = int(color_str[3:5], 16)
            b = int(color_str[5:7], 16)
            return (r, g, b)
        elif color_str.startswith("rgb("):
            values = [int(x) for x in color_str[4:-1].split(",")]
            return (values[0], values[1], values[2])
        else:
            return (255, 255, 255)

#endregion--------------layout---------------------------



class create_RadialGradient:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "gradient_distance": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
                    "radial_center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "radial_center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "start_color_hex": ("STRING", {"default": "#000000"}),
                    "end_color_hex": ("STRING", {"default": "#FFF2F2"}),
                    },
                "optional": {
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "draw"
    CATEGORY = "Apt_Preset/imgEffect"

    def draw(self, width, height, 
            radial_center_x=0.5, radial_center_y=0.5, gradient_distance=1,
            start_color_hex='#000000', end_color_hex='#ffffff'):

        color1_rgb = hex_to_rgb_tuple(start_color_hex)
        color2_rgb = hex_to_rgb_tuple(end_color_hex)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        center_x = int(radial_center_x * width)
        center_y = int(radial_center_y * height)                
        max_distance = (np.sqrt(max(center_x, width - center_x)**2 + max(center_y, height - center_y)**2))*gradient_distance

        for i in range(width):
            for j in range(height):
                distance_to_center = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                t = distance_to_center / max_distance
                t = max(0, min(t, 1))
                interpolated_color = [int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1_rgb, color2_rgb)]
                canvas[j, i] = interpolated_color 

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        ax.imshow(canvas)
        plt.axis('off')
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.autoscale(tight=True)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)

        image_out = pil2tensor(img.convert("RGB"))
        
        mask_out = torch.from_numpy(canvas[:, :, 0]).float() / 255.0
        mask_out = mask_out.unsqueeze(0)

        return (image_out, mask_out)


class create_lineGradient:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "gradient_distance": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
                    "linear_transition": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "orientation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 10}),
                    "start_color_hex": ("STRING", {"default": "#FFFFFF"}),
                    "end_color_hex": ("STRING", {"default": "#000000"}),
                    
                    },
                "optional": {
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "draw"
    CATEGORY = "Apt_Preset/imgEffect"

    def draw(self, width, height, orientation, start_color_hex='#000000', end_color_hex='#ffffff', 
            linear_transition=0.5, gradient_distance=1,): 
        
        color1_rgb = hex_to_rgb_tuple(start_color_hex)
        color2_rgb = hex_to_rgb_tuple(end_color_hex)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        angle = np.radians(orientation)
        
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        gradient = X * np.cos(angle) + Y * np.sin(angle)
        
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
        gradient = (gradient - (linear_transition - gradient_distance/2)) / gradient_distance
        gradient = np.clip(gradient, 0, 1)
        
        for j in range(height):
            for i in range(width):
                t = gradient[j, i]
                interpolated_color = [int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1_rgb, color2_rgb)]
                canvas[j, i] = interpolated_color
                    
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        ax.imshow(canvas)
        plt.axis('off')
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.autoscale(tight=True)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)
        
        image_out = pil2tensor(img.convert("RGB"))
        
        mask_out = torch.from_numpy(gradient).float()
        mask_out = mask_out.unsqueeze(0)

        return (image_out, mask_out)




#----------------------------------总控包------------------------------------------


class lay_images_free_layout:
    @classmethod
    def INPUT_TYPES(s):
        templates = ["custom",
                    "G21", "G22",
                    "H2", "H3",
                    "H12", "H13",
                    "V2", "V3",
                    "V31", "V32"]                           
        
        return {"required": {
                    "page_width": ("INT", {"default": 512, "min": 8, "max": 4096}),
                    "page_height": ("INT", {"default": 512, "min": 8, "max": 4096}),
                    "template": (templates,),
                    "border_thickness": ("INT", {"default": 5, "min": 0, "max": 1024}),
                    "outline_thickness": ("INT", {"default": 2, "min": 0, "max": 1024}),
                    "outline_color": ("STRING", {"default": "#000000"}),
                    "panel_color": ("STRING", {"default": "#00FF62"}),
                    "bg_color": ("STRING", {"default": "#FF0000"}),
            },
                "optional": {
                    "images": ("IMAGE",),
                    "custom_panel_layout": ("STRING", {"multiline": False, "default": "H123"}),
            }
    }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "layout"
    CATEGORY = "Apt_Preset/imgEffect"
    
    def layout(self, page_width, page_height, template, 
            border_thickness, outline_thickness, 
            outline_color, panel_color, bg_color,
            images=None, custom_panel_layout='G44',):

        panels = []
        k = 0
        len_images = 0
        
        if images is not None:
            images = [tensor2pil(image) for image in images]
            len_images = len(images)
        size = (page_width - (2 * border_thickness), page_height - (2 * border_thickness))
        page = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(page)

        if template == "custom":
            template = custom_panel_layout
        first_char = template[0]
        if first_char == "G":
            rows = int(template[1])
            columns = int(template[2])
            panel_width = (page.width - (2 * columns * (border_thickness + outline_thickness))) // columns
            panel_height = (page.height  - (2 * rows * (border_thickness + outline_thickness))) // rows
            # Row loop
            for i in range(rows):
                # Column Loop
                for j in range(columns):
                    # Draw the panel
                    create_and_paste_panel(page, border_thickness, outline_thickness,
                                        panel_width, panel_height, page.width,
                                        panel_color, bg_color, outline_color,
                                        images, i, j, k, len_images)
                    k += 1

        elif first_char == "H":
            rows = len(template) - 1
            panel_height = (page.height  - (2 * rows * (border_thickness + outline_thickness))) // rows
            for i in range(rows):
                columns = int(template[i+1])
                panel_width = (page.width - (2 * columns * (border_thickness + outline_thickness))) // columns
                for j in range(columns):
                    # Draw the panel
                    create_and_paste_panel(page, border_thickness, outline_thickness,
                                        panel_width, panel_height, page.width,
                                        panel_color, bg_color, outline_color,
                                        images, i, j, k, len_images)
                    k += 1
                    
        elif first_char == "V":
            columns = len(template) - 1
            panel_width = (page.width - (2 * columns * (border_thickness + outline_thickness))) // columns
            for j in range(columns):
                rows = int(template[j+1])
                panel_height = (page.height  - (2 * rows * (border_thickness + outline_thickness))) // rows
                for i in range(rows):
                    # Draw the panel
                    create_and_paste_panel(page, border_thickness, outline_thickness,
                                        panel_width, panel_height, page.width,
                                        panel_color, bg_color, outline_color,
                                        images, i, j, k, len_images)
                    k += 1 
        
        if border_thickness > 0:
            page = ImageOps.expand(page, border_thickness, bg_color)

        return (pil2tensor(page), )   






class create_Mask_match_shape:
    def __init__(self):
        self.bg_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bimg": ("IMAGE",),
                "bmask": ("MASK",),
                "b_color": (["black", "white", "red", "green", "blue"], {"default": "blue"}),
                "b_extrant_to_block": ("BOOLEAN", {"default": True}),
                "f_extrant_to_block": ("BOOLEAN", {"default": True}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue"], {"default": "white"}),
                "f_smoothness": ("INT", {"default": 1, "min": 0, "max": 150, "step": 1}),
                "f_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "fimg": ("IMAGE",),
                "fmask": ("MASK",),
                "f_x_offset": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "f_y_offset": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "scale_mode": ( ["custom", "width_align", "height_align", "auto-in", "auto-out"], {"default": "auto-in"}),
                "f_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "f_rot": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("composed_image", "new_background", "foreground_layer", "foreground_mask")
    FUNCTION = "compose"
    CATEGORY = "Apt_Preset/mask"

    def get_min_rect(self, mask_np):
        _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (mask_np.shape[1], mask_np.shape[0])
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (w, h)


    def compose(self, bimg, bmask, b_color, b_extrant_to_block, f_extrant_to_block, 
                edge_detection, edge_thickness, edge_color,
                f_smoothness, f_opacity, scale_mode, image_output,
                fimg=None, fmask=None, f_scale=1.0, f_x_offset=0, f_y_offset=0, f_rot=0,
                prompt=None, extra_pnginfo=None):

        def get_resample_method():
            try:
                if hasattr(Image, 'Resampling') and hasattr(Image.Resampling, 'BICUBIC'):
                    return Image.Resampling.BICUBIC
                else:
                    return Image.BICUBIC
            except:
                return Image.BILINEAR

        resample_method = get_resample_method()
        rotate_resample = get_resample_method()

        def tensor2pil(tensor):
            if len(tensor.shape) == 4:
                tensor = tensor[0]
            return Image.fromarray(np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8))
        
        def pil2tensor(image):
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        bimg_pil = tensor2pil(bimg)
        bmask_np = bmask.cpu().numpy().squeeze() * 255
        bmask_np = bmask_np.astype(np.uint8)
        bmask_height, bmask_width = bmask_np.shape[:2]

        _, binary_mask = cv2.threshold(bmask_np, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        b_center_x, b_center_y = bimg_pil.width // 2, bimg_pil.height // 2
        if contours:
            moments = [cv2.moments(cnt) for cnt in contours]
            contour_centers = []
            for m in moments:
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    contour_centers.append((cx, cy))
            if contour_centers:
                b_center_x = int(sum(x for x, y in contour_centers) / len(contour_centers))
                b_center_y = int(sum(y for x, y in contour_centers) / len(contour_centers))

        bg_color_mask = np.zeros_like(binary_mask)
        for contour in contours[:8]:
            if b_extrant_to_block:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(bg_color_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(bg_color_mask, [contour], 0, 255, thickness=cv2.FILLED)

        bimg_np = np.array(bimg_pil).astype(np.float32)
        color = np.array(self.bg_colors[b_color], dtype=np.float32)
        mask_float = bg_color_mask.astype(np.float32) / 255.0
        for c in range(3):
            bimg_np[:, :, c] = mask_float * color[c] + (1 - mask_float) * bimg_np[:, :, c]
        bimg_processed = Image.fromarray(np.clip(bimg_np, 0, 255).astype(np.uint8)).convert("RGBA")
        
        new_background_pil = bimg_processed.convert("RGB")

        has_foreground = fimg is not None
        edge_image = None
        foreground_layer_pil = Image.new("RGBA", bimg_processed.size, (0, 0, 0, 0))
        foreground_mask_np = np.zeros_like(bg_color_mask)

        if has_foreground:
            fimg_pil = tensor2pil(fimg)
            
            if fmask is None:
                fmask_np = np.ones((fimg_pil.height, fimg_pil.width), dtype=np.uint8) * 255
                fmask_pil = Image.fromarray(fmask_np)
            else:
                fmask_np = fmask.cpu().numpy().squeeze() * 255
                fmask_np = fmask_np.astype(np.uint8)
                fmask_pil = Image.fromarray(fmask_np)

            f_original_height, f_original_width = fmask_np.shape[:2]
            if f_original_width == 0 or f_original_height == 0:
                composed_tensor = pil2tensor(bimg_processed.convert("RGB"))
                new_background_tensor = pil2tensor(new_background_pil)
                foreground_layer_tensor = pil2tensor(foreground_layer_pil.convert("RGB"))
                foreground_mask_tensor = torch.from_numpy(foreground_mask_np).float() / 255.0
                foreground_mask_tensor = foreground_mask_tensor.unsqueeze(0)
                return {"ui": {}, "result": (composed_tensor, new_background_tensor, foreground_layer_tensor, foreground_mask_tensor)}

            # 获取背景和前景mask的最小矩形尺寸
            W1, H1 = self.get_min_rect(bmask_np)
            W2, H2 = self.get_min_rect(fmask_np)
            
            W2 = max(W2, 1)
            H2 = max(H2, 1)

            # 计算缩放比例 - 修复auto-out模式下的缩放问题
            if scale_mode == "width_align":
                f_scale = W1 / W2
            elif scale_mode == "height_align":
                f_scale = H1 / H2
            elif scale_mode == "auto-in":
                scale_w = W1 / W2
                scale_h = H1 / H2
                f_scale = min(scale_w, scale_h)
            elif scale_mode == "auto-out":  # all-out模式
                scale_w = W1 / W2
                scale_h = H1 / H2
                f_scale = max(scale_w, scale_h)
            f_scale = max(0.01, min(f_scale, 5.0))

            # 计算背景mask的轮廓中心（统一使用轮廓中心对齐）
            _, bmask_bin = cv2.threshold(bmask_np, 127, 255, cv2.THRESH_BINARY)
            b_contours, _ = cv2.findContours(bmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if b_contours:
                largest_b_contour = max(b_contours, key=cv2.contourArea)
                b_moments = cv2.moments(largest_b_contour)
                if b_moments["m00"] != 0:
                    b_center_x = int(b_moments["m10"] / b_moments["m00"])
                    b_center_y = int(b_moments["m01"] / b_moments["m00"])
            else:
                b_center_x, b_center_y = bimg_pil.width // 2, bimg_pil.height // 2

            # 使用计算好的f_scale直接缩放，不进行二次缩放
            new_size = (int(f_original_width * f_scale), int(f_original_height * f_scale))
            fimg_scaled = fimg_pil.resize(new_size, resample=resample_method)
            fmask_scaled = fmask_pil.resize(new_size, resample=resample_method)

            # 处理旋转
            if f_rot != 0:
                fimg_scaled = fimg_scaled.rotate(f_rot, expand=True, resample=rotate_resample)
                fmask_scaled = fmask_scaled.rotate(f_rot, expand=True, resample=rotate_resample)

            # 计算前景mask的轮廓中心
            fmask_scaled_np = np.array(fmask_scaled)
            _, fmask_bin = cv2.threshold(fmask_scaled_np, 127, 255, cv2.THRESH_BINARY)
            f_contours, _ = cv2.findContours(fmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rf_center_x, rf_center_y = fmask_scaled.width // 2, fmask_scaled.height // 2
            if f_contours:
                largest_f_contour = max(f_contours, key=cv2.contourArea)
                f_moments = cv2.moments(largest_f_contour)
                if f_moments["m00"] != 0:
                    rf_center_x = int(f_moments["m10"] / f_moments["m00"])
                    rf_center_y = int(f_moments["m01"] / f_moments["m00"])

            # 计算最终放置位置（确保中心对齐）
            base_x = b_center_x - rf_center_x + f_x_offset
            base_y = b_center_y - rf_center_y + f_y_offset

            # 处理前景mask
            fmask_processed = np.zeros_like(fmask_bin)
            edge_contours = []
            
            if np.count_nonzero(fmask_bin) > 0:
                f_contours, _ = cv2.findContours(fmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if f_contours:
                    for contour in f_contours:
                        if f_extrant_to_block:
                            x, y, w, h = cv2.boundingRect(contour)
                            rect_contour = np.array([
                                [[x, y]],
                                [[x+w, y]],
                                [[x+w, y+h]],
                                [[x, y+h]]
                            ], dtype=np.int32)
                            edge_contours.append(rect_contour)
                            cv2.rectangle(fmask_processed, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
                        else:
                            edge_contours.append(contour)
                            cv2.drawContours(fmask_processed, [contour], 0, 255, thickness=1)
            
            fmask_final = np.maximum(fmask_bin, fmask_processed)
            fmask_final_pil = Image.fromarray(fmask_final)

            # 处理前景mask放置
            foreground_mask_canvas = Image.new('L', bimg_processed.size, 0)
            foreground_mask_canvas.paste(fmask_final_pil, (base_x, base_y))
            foreground_mask_np = np.array(foreground_mask_canvas)

            # 处理边缘检测
            if edge_detection and len(edge_contours) > 0:
                edge_img = Image.new('RGBA', fimg_scaled.size, (0, 0, 0, 0))
                edge_mask = Image.new('L', fimg_scaled.size, 0)
                draw = ImageDraw.Draw(edge_img)
                draw_mask = ImageDraw.Draw(edge_mask)
                edge_color_rgb = self.bg_colors.get(edge_color, (255, 255, 255))
                
                for contour in edge_contours:
                    points = [(p[0][0], p[0][1]) for p in contour]
                    if len(points) >= 2:
                        draw.line(points, fill=(*edge_color_rgb, 255), width=edge_thickness)
                        draw_mask.line(points, fill=255, width=edge_thickness)
                
                edge_image = edge_img
                edge_mask_image = edge_mask

            # 合成前景图层
            fg_with_alpha = Image.new('RGBA', bimg_processed.size, (0, 0, 0, 0))
            fg_with_alpha.paste(fimg_scaled, (base_x, base_y), mask=fmask_final_pil)

            if edge_image is not None:
                fg_with_alpha.paste(edge_image, (base_x, base_y), mask=edge_image.split()[-1])

            # 处理平滑度
            if f_smoothness > 0:
                fg_np = np.array(fg_with_alpha)
                alpha_channel = fg_np[:, :, 3]
                blurred_alpha = cv2.GaussianBlur(alpha_channel, (0, 0), sigmaX=f_smoothness)
                fg_np[:, :, 3] = blurred_alpha
                fg_with_alpha = Image.fromarray(fg_np)

            # 处理不透明度
            alpha = int(f_opacity * 255)
            alpha_channel = fg_with_alpha.split()[-1]
            alpha_channel = Image.eval(alpha_channel, lambda x: int(x * alpha / 255))
            fg_with_alpha.putalpha(alpha_channel)

            # 最终合成
            composed_pil = Image.alpha_composite(bimg_processed, fg_with_alpha)
            composed_pil = composed_pil.convert('RGB')
            
            foreground_layer_pil = fg_with_alpha

            if f_smoothness > 0:
                foreground_mask_np = cv2.GaussianBlur(foreground_mask_np, (0, 0), sigmaX=f_smoothness)

        else:
            if edge_detection and len(contours) > 0:
                edge_img = Image.new('RGBA', bimg_processed.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(edge_img)
                edge_color_rgb = self.bg_colors.get(edge_color, (255, 255, 255))
                for contour in contours:
                    points = [(p[0][0], p[0][1]) for p in contour]
                    if len(points) >= 2:
                        draw.line(points, fill=(*edge_color_rgb, 255), width=edge_thickness)
                composed_pil = Image.alpha_composite(bimg_processed, edge_img).convert('RGB')
            else:
                composed_pil = bimg_processed.convert('RGB')

        # 转换为张量输出
        composed_tensor = pil2tensor(composed_pil)
        new_background_tensor = pil2tensor(new_background_pil)
        foreground_layer_tensor = pil2tensor(foreground_layer_pil.convert("RGB"))
        foreground_mask_tensor = torch.from_numpy(foreground_mask_np).float() / 255.0
        foreground_mask_tensor = foreground_mask_tensor.unsqueeze(0)

        results = easySave(composed_tensor, 'composedPreview', image_output, prompt, extra_pnginfo)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {}, "result": (composed_tensor, new_background_tensor, foreground_layer_tensor, foreground_mask_tensor)}        
        return {"ui": {"images": results}, "result": (composed_tensor, new_background_tensor, foreground_layer_tensor, foreground_mask_tensor)}



class create_Mask_visual_tag:
    def __init__(self):
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "中灰": (128, 128, 128)  # 1. 新增中灰颜色定义（RGB标准中灰值）
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 8}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "fill": ("BOOLEAN", {"default": True, }),
                "mask_mode": (["原始", "方形", "圆形", "五角星", "菱形", "六边形"], {"default": "原始"}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1,}),
                "out_color": (["colorful", "white", "black", "red", "green", "blue", "中灰"], {"default": "colorful"}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "mask_info": ("MASK_INFO",)
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "fill_mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"

    def separate(self, mask, base_image, ignore_threshold=100, opacity=0.8, 
                outline_thickness=1, mask_mode="原始", fill=True, smoothness=1, 
                image_output=None, out_color="colorful", mask_info=None, 
                prompt=None, extra_pnginfo=None):

        def tensor2pil(image):
            return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        def tensorMask2cv2img(tensor_mask):
            if isinstance(tensor_mask, torch.Tensor):
                mask_np = tensor_mask.squeeze().cpu().numpy()
                return (mask_np * 255).astype(np.uint8)
            return tensor_mask

        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_with_positions = [(cv2.boundingRect(c)[0], cv2.boundingRect(c)[1], c) for c in contours]
        contours_with_positions.sort(key=lambda x: (x[1], x[0]))
        sorted_contours = [c[2] for c in contours_with_positions[:8]]

        fill_mask = np.zeros_like(binary_mask)
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < ignore_threshold:
                continue
            cv2.drawContours(fill_mask, [contour], 0, 255, cv2.FILLED)
        if smoothness > 0:
            fill_mask = np.array(Image.fromarray(fill_mask).filter(ImageFilter.GaussianBlur(radius=smoothness)))

        base_image_np = base_image[0].cpu().numpy() * 255.0
        base_image_np = base_image_np.astype(np.float32)
        mask_color_layer = base_image_np.copy()

        final_mask = np.zeros_like(binary_mask)

        for i, contour in enumerate(sorted_contours):
            area = cv2.contourArea(contour)
            if area < ignore_threshold:
                continue
            if out_color in ["white", "black", "red", "green", "blue", "中灰"]:
                color = np.array(self.colors[out_color], dtype=np.float32)
                fill_current = fill
                outline_thickness_current = outline_thickness
                mask_mode_current = mask_mode
            elif mask_info and f"mask{i+1}" in mask_info:
                # mask_info存在时：完全接管颜色、填充、线宽、形状模式
                config = mask_info[f"mask{i+1}"]
                color = np.array(config["rgb"], dtype=np.float32)
                fill_current = config.get("fill", fill)
                outline_thickness_current = config.get("outline_thickness", outline_thickness)
                mask_mode_current = config.get("mask_mode", mask_mode)
            else:
                # 无mask_info的colorful模式：使用默认彩色循环（新增"中灰"到循环列表）
                if out_color == "colorful":
                    color_names = ["white", "black", "red", "green", "blue", "中灰", "yellow", "cyan", "magenta"]
                    color_name = color_names[i % len(color_names)]
                    color = np.array(self.colors[color_name], dtype=np.float32)
                else:
                    color = np.array(self.colors[out_color], dtype=np.float32)
                fill_current = fill
                outline_thickness_current = outline_thickness
                mask_mode_current = mask_mode

            temp_mask = np.zeros_like(binary_mask)
            thickness = cv2.FILLED if fill_current else outline_thickness_current

            # 根据当前形状模式绘制mask（原逻辑不变）
            if mask_mode_current == "原始":
                cv2.drawContours(temp_mask, [contour], 0, 255, thickness)
                if not fill_current:
                    temp_mask = cv2.bitwise_and(opencv_gray_image, temp_mask)
            elif mask_mode_current == "方形":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness)
            elif mask_mode_current == "圆形":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness)
            elif mask_mode_current == "五角星":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                pts = []
                for j in range(10):
                    angle_rad = np.pi / 5 * j
                    r = radius if j % 2 == 0 else radius * 0.4
                    px = center[0] + r * np.cos(angle_rad)
                    py = center[1] + r * np.sin(angle_rad)
                    pts.append((int(px), int(py)))
                if fill_current:
                    cv2.fillPoly(temp_mask, [np.array(pts)], (255, 255, 255))
                else:
                    cv2.polylines(temp_mask, [np.array(pts)], True, (255, 255, 255), outline_thickness_current)
            elif mask_mode_current == "菱形":
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                pts = [
                    (center_x, y),
                    (x + w, center_y),
                    (center_x, y + h),
                    (x, center_y)
                ]
                if fill_current:
                    cv2.fillPoly(temp_mask, [np.array(pts)], (255, 255, 255))
                else:
                    cv2.polylines(temp_mask, [np.array(pts)], True, (255, 255, 255), outline_thickness_current)
            elif mask_mode_current == "六边形":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                pts = []
                for j in range(6):
                    angle_rad = np.pi / 3 * j
                    px = center[0] + radius * np.cos(angle_rad)
                    py = center[1] + radius * np.sin(angle_rad)
                    pts.append((int(px), int(py)))
                if fill_current:
                    cv2.fillPoly(temp_mask, [np.array(pts)], (255, 255, 255))
                else:
                    cv2.polylines(temp_mask, [np.array(pts)], True, (255, 255, 255), outline_thickness_current)

            if smoothness > 0:
                temp_mask = np.array(Image.fromarray(temp_mask).filter(ImageFilter.GaussianBlur(radius=smoothness)))

            final_mask = cv2.bitwise_or(final_mask, temp_mask)
            mask_float = temp_mask.astype(np.float32) / 255.0

            for c in range(3):
                mask_color_layer[:, :, c] = (
                    mask_float * color[c] +
                    (1 - mask_float) * mask_color_layer[:, :, c]
                )

        mask_float_global = final_mask.astype(np.float32) / 255.0
        combined_image = (
            opacity * mask_color_layer +
            (1 - opacity) * base_image_np
        )
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)

        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)

        fill_mask_tensor = torch.from_numpy(fill_mask).float() / 255.0
        fill_mask_tensor = fill_mask_tensor.unsqueeze(0)

        results = easySave(combined_image_tensor, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {}, "result": (combined_image_tensor, final_mask_tensor, fill_mask_tensor)}
        return {"ui": {"images": results}, "result": (combined_image_tensor, final_mask_tensor, fill_mask_tensor)}


class stack_Mask2color:
    COLORS = [
        "Default",
        "Red", "Green", "Blue", 
        "Yellow", "Magenta", "Cyan", 
        "White", "Black",
        "Medium Gray"  
    ]
    
    def __init__(self):
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "medium gray": (128, 128, 128)  
        }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Mask1 参数组
                "mask_mode1": (["原始", "方形", "圆形", "五角星", "菱形", "六边形"], {"default": "原始"}),
                "fill1": ("BOOLEAN", {"default": True, }),            
                "mask1_color": (s.COLORS, {"default": "Default"}),
                "outline_thickness1": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                
                # Mask2 参数组
                "mask_mode2": (["原始", "方形", "圆形", "五角星", "菱形", "六边形"], {"default": "原始"}),
                "fill2": ("BOOLEAN", {"default": True, }),
                "mask2_color": (s.COLORS, {"default": "Default"}),
                "outline_thickness2": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                
                # Mask3 参数组
                "mask_mode3": (["原始", "方形", "圆形", "五角星", "菱形", "六边形"], {"default": "原始"}),
                "fill3": ("BOOLEAN", {"default": True, }),
                "outline_thickness3": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "mask3_color": (s.COLORS, {"default": "Default"}),
            }
        }
    
    RETURN_TYPES = ("MASK_INFO",)
    RETURN_NAMES = ("mask_info",)
    FUNCTION = "pack_mask_info"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def pack_mask_info(self, 
                      # Mask1 对应参数
                      mask1_color, fill1, outline_thickness1, mask_mode1,
                      # Mask2 对应参数
                      mask2_color, fill2, outline_thickness2, mask_mode2,
                      # Mask3 对应参数
                      mask3_color, fill3, outline_thickness3, mask_mode3):
        def get_rgb_color(color_name):
            lower_color = color_name.lower()  
            return self.colors.get(lower_color, self.colors["white"])  
        
        mask_info = {
            "mask1": {
                "color_name": mask1_color.lower(),
                "rgb": get_rgb_color(mask1_color),
                "fill": fill1,
                "outline_thickness": outline_thickness1,
                "mask_mode": mask_mode1
            },
            "mask2": {
                "color_name": mask2_color.lower(),
                "rgb": get_rgb_color(mask2_color),
                "fill": fill2,
                "outline_thickness": outline_thickness2,
                "mask_mode": mask_mode2
            },
            "mask3": {
                "color_name": mask3_color.lower(),
                "rgb": get_rgb_color(mask3_color),
                "fill": fill3,
                "outline_thickness": outline_thickness3,
                "mask_mode": mask_mode3
            }
        }
        
        return (mask_info,)





class lay_image_conbine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
            }),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = "Apt_Preset/imgEffect"

    def concatenate(self, image1, image2, direction,):
        match_image_size =True 
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2
            
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)
        first_image_shape=None
        if match_image_size:
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                target_height = target_shape[1]
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                target_width = target_shape[2]
                target_height = int(target_width / original_aspect_ratio)
            
            image2_for_upscale = image2.movedim(-1, 1)
            image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)

        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)
        return concatenated_image,



class lay_image_add_text:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),  
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
            "font": (file_list,),
            "text": ("STRING", {"default": "Text"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "addlabel"
    CATEGORY = "Apt_Preset/imgEffect"

    def addlabel(self, image, text, font_size, font):
        # 文字位置参数
        text_x = 20
        text_y = font_size  # 调整初始Y坐标，避免文字过靠上
        font_color = "red"  # 纯色字体，可根据需求修改
        
        batch_size = image.shape[0]
        width = image.shape[2]
        
        # 正确获取字体文件路径
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        font_path = os.path.join(font_dir, font)

        def process_image(input_image, caption_text):
            # 使用正确的字体路径
            font_obj = ImageFont.truetype(font_path, font_size)
            words = caption_text.split()
            lines = []
            current_line = []
            current_line_width = 0

            # 文本换行处理：根据图片宽度拆分文字为多行
            for word in words:
                word_width = font_obj.getbbox(word)[2]
                if current_line_width + word_width <= width - 2 * text_x:
                    current_line.append(word)
                    current_line_width += word_width + font_obj.getbbox(" ")[2]  # 增加空格宽度
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_line_width = word_width
            if current_line:
                lines.append(" ".join(current_line))

            # 直接使用原始图片作为绘制载体，不创建额外背景
            pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_image)
            
            # 逐行绘制文字
            y_offset = text_y
            for line in lines:
                try:
                    draw.text((text_x, y_offset), line, font=font_obj, fill=font_color, features=['-liga'])
                except:
                    draw.text((text_x, y_offset), line, font=font_obj, fill=font_color)
                y_offset += font_size  # 每行文字间隔一个字体高度

            # 转换回tensor格式
            processed_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
            return processed_image
        
        # 处理批量图片
        processed_images = [process_image(img, text) for img in image]
        processed_batch = torch.cat(processed_images, dim=0)
        
        # 直接返回处理后的图片（原代码的拼接逻辑已删除，因无需额外背景块）
        return (processed_batch,)



class lay_image_mul:
    @classmethod
    def INPUT_TYPES(s):
        templates = [
            "H11", "H12", "H13", 
            "H21", "H31", 
            "V11", "V12", "V13", 
            "V21", "V31", 
        ]                           
        
        return {
            "required": {
                "long_size": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "template": (templates,),
                "out_text": ("STRING", {"default": ""}),
                "input_text1": ("STRING", {"default": ""}),
                "input_text2": ("STRING", {"default": ""}),
                "input_text3": ("STRING", {"default": ""}),
                "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "font": (file_list,),
            },
            "optional": {
                "out_img": ("IMAGE",),
                "input_img1": ("IMAGE",),
                "input_img2": ("IMAGE",),
                "input_img3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "layout"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def layout(self, long_size, template, out_text, input_text1, input_text2, input_text3, font_size, font,
               out_img=None, input_img1=None, input_img2=None, input_img3=None):

        # 为图片添加文字（使用修改后的lay_image_add_text，无背景）
        if font_size > 0:
            if out_img is not None and out_text != "":
                out_img = lay_image_add_text().addlabel(out_img, out_text, font_size, font)[0]
            if input_img1 is not None and input_text1 != "":
                input_img1 = lay_image_add_text().addlabel(input_img1, input_text1, font_size, font)[0]
            if input_img2 is not None and input_text2 != "":
                input_img2 = lay_image_add_text().addlabel(input_img2, input_text2, font_size, font)[0]
            if input_img3 is not None and input_text3 != "":
                input_img3 = lay_image_add_text().addlabel(input_img3, input_text3, font_size, font)[0]

        # 初始化结果图片
        Nimage = None
        # 调用图片拼接类（需确保lay_image_conbine已正确实现）
        combine_cls = lay_image_conbine()

        # 根据模板处理图片布局
        if template == "H11":
            if input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="right")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "H12":
            if input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="right")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="right")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "H13":
            if input_img1 is not None and input_img2 is not None and input_img3 is not None:
                temp1 = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                temp2 = combine_cls.concatenate(input_img3, temp1, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp2, direction="right")[0] if out_img is not None else temp2
            elif input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="right")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="right")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "H21":
            if input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="left")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="left")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "H31":
            if input_img1 is not None and input_img2 is not None and input_img3 is not None:
                temp1 = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                temp2 = combine_cls.concatenate(input_img3, temp1, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp2, direction="left")[0] if out_img is not None else temp2
            elif input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="up")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="left")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="left")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "V11":
            if input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="up")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "V12":
            if input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="up")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="up")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "V13":
            if input_img1 is not None and input_img2 is not None and input_img3 is not None:
                temp1 = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                temp2 = combine_cls.concatenate(input_img3, temp1, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp2, direction="up")[0] if out_img is not None else temp2
            elif input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="up")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="up")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "V21":
            if input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="down")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="down")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        elif template == "V31":
            if input_img1 is not None and input_img2 is not None and input_img3 is not None:
                temp1 = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                temp2 = combine_cls.concatenate(input_img3, temp1, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp2, direction="down")[0] if out_img is not None else temp2
            elif input_img1 is not None and input_img2 is not None:
                temp = combine_cls.concatenate(input_img1, input_img2, direction="right")[0]
                Nimage = combine_cls.concatenate(out_img, temp, direction="down")[0] if out_img is not None else temp
            elif input_img1 is not None:
                Nimage = combine_cls.concatenate(out_img, input_img1, direction="down")[0] if out_img is not None else input_img1
            else:
                Nimage = out_img if out_img is not None else torch.zeros(1, 512, 512, 3)

        # 确保结果为tensor格式，并按长边缩放
        if isinstance(Nimage, torch.Tensor):
            Nimage_pil = tensor2pil(Nimage)
        elif isinstance(Nimage, Image.Image):
            Nimage_pil = Nimage
        else:
            Nimage_pil = Image.new('RGB', (512, 512), (0, 0, 0))
        
        # 根据长边尺寸调整图片大小
        Nimage_pil = self.resize_by_long_side(Nimage_pil, long_size)
        # 转换回tensor返回
        return (pil2tensor(Nimage_pil),)

    def resize_by_long_side(self, image, long_size):
        """根据长边调整图像尺寸，支持放大和缩小"""
        if long_size <= 0 or not isinstance(image, Image.Image):
            return image
            
        width, height = image.size
        long_side = max(width, height)
        
        if long_side == long_size:
            return image
            
        # 计算缩放比例
        scale = long_size / long_side
        new_width = max(100, int(width * scale))
        new_height = max(100, int(height * scale))
        
        # 高质量缩放
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)





import torch
import os
import sys
from PIL import Image, ImageFont, ImageDraw
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_font_list():
    font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    font_extensions = ('.ttf', '.otf', '.ttc')
    font_list = []
    for file in os.listdir(font_dir):
        if file.lower().endswith(font_extensions):
            font_list.append(file)
    return font_list

file_list = get_font_list()


class lay_mul_image:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        self.file_list = [f for f in os.listdir(self.font_dir) if f.endswith(('.ttf', '.otf'))]

    @classmethod
    def INPUT_TYPES(cls):
        inst = cls()
        file_list = inst.file_list
        
        return {
            "required": {
                "sub_team_direction": (["row", "column"], {"default": "row"}),
                "main_position": (["top", "bottom", "left", "right"], {"default": "top"}),
                "font": (file_list,),
            },
            "optional": {
                "main_image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                
                "main_text": ("STRING", {"multiline": False, "default": ""}),
                "main_text_size": ("INT", {"default": 60, "min": 8, "max": 128, "step": 2}),
                "main_long_size": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 32}),

                "image1_text": ("STRING", {"multiline": False, "default": ""}),
                "image2_text": ("STRING", {"multiline": False, "default": ""}),
                "image3_text": ("STRING", {"multiline": False, "default": ""}),
                "image4_text": ("STRING", {"multiline": False, "default": ""}),
                "sub_text_size": ("INT", {"default": 30, "min": 8, "max": 128, "step": 2}),
                "sub_team_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 32}),
                "border": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'process_images'
    CATEGORY = "Apt_Preset/imgEffect"
    DESCRIPTION = """
    - 子图排列方向：4个子图横向或纵向排列，
    - 主图位置：围绕子图组合的位置
    """


    def process_images(self, sub_team_direction, sub_team_size, sub_text_size, main_text_size, font, border,
                      image1=None, image1_text="", image2=None, image2_text="",
                      image3=None, image3_text="", image4=None, image4_text="",
                      main_image=None, main_text="", main_position="top", main_long_size=1024):
        
        sub_images = []
        sub_texts = []
        
        img_text_pairs = [
            (image1, image1_text),
            (image2, image2_text),
            (image3, image3_text),
            (image4, image4_text)
        ]
        
        for img, text in img_text_pairs:
            if img is not None:
                pil_img = tensor2pil(img.unsqueeze(0))
                sub_images.append(pil_img)
                sub_texts.append(text)
        
        if not sub_images:
            return (pil2tensor(Image.new('RGB', (512, 512), color='white')),)
        
        sub_grid = self.create_sub_grid(sub_images, sub_texts, sub_team_direction, sub_team_size, sub_text_size, font, border)
        
        if main_image is not None:
            main_pil = tensor2pil(main_image)
            if main_text:
                main_pil = self.add_text_to_image(main_pil, main_text, main_text_size, font, border)
            main_resized = self.resize_main_image(main_pil, main_long_size)
            result_image = self.combine_main_sub(main_resized, sub_grid, main_position, border)
        else:
            result_image = sub_grid
            
        return (pil2tensor(result_image),)
    
    def resize_sub_image(self, image, direction, target_size) -> Image:
        if direction == "row":
            scale = target_size / image.height
            new_width = int(image.width * scale)
            return image.resize((new_width, target_size), Image.LANCZOS)
        else:
            scale = target_size / image.width
            new_height = int(image.height * scale)
            return image.resize((target_size, new_height), Image.LANCZOS)
    
    def add_text_to_image(self, image, text, text_size, font_name, border) -> Image:
        if not text.strip():
            return image
            
        text_height = int(text_size * 1.8)
        new_image = Image.new('RGB', (image.width, image.height + text_height), color='white')
        new_image.paste(image, (0, 0))
        
        draw = ImageDraw.Draw(new_image)
        
        font_path = os.path.join(self.font_dir, font_name)
        font_obj = ImageFont.truetype(font_path, text_size)
        
        bbox = font_obj.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x = (image.width - text_width) // 2
        y = image.height + (text_height - text_size) // 2
        
        try:
            draw.text((x, y), text, fill=(0, 0, 0), font=font_obj, features=['-liga'])
        except:
            draw.text((x, y), text, fill=(0, 0, 0), font=font_obj)
        
        return new_image
    
    def create_sub_grid(self, images, texts, direction, target_size, text_size, font_name, border) -> Image:
        processed_images = []
        for img, text in zip(images, texts):
            resized = self.resize_sub_image(img, direction, target_size)
            if text:
                with_text = self.add_text_to_image(resized, text, text_size, font_name, border)
                processed_images.append(with_text)
            else:
                processed_images.append(resized)
        
        spacing = border
        
        if direction == "row":
            total_width = sum(img.width for img in processed_images) + spacing * (len(processed_images) - 1)
            max_height = max(img.height for img in processed_images)
            grid_width = total_width + border * 2
            grid_height = max_height + border * 2
            
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            x = border
            for img in processed_images:
                y = border + (max_height - img.height) // 2
                grid_img.paste(img, (x, y))
                x += img.width + spacing
        else:
            max_width = max(img.width for img in processed_images)
            total_height = sum(img.height for img in processed_images) + spacing * (len(processed_images) - 1)
            grid_width = max_width + border * 2
            grid_height = total_height + border * 2
            
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            y = border
            for img in processed_images:
                x = border + (max_width - img.width) // 2
                grid_img.paste(img, (x, y))
                y += img.height + spacing
        
        return grid_img
    
    def resize_main_image(self, main_image, long_side) -> Image:
        width, height = main_image.size
        if width >= height:
            scale = long_side / width
            new_width = long_side
            new_height = int(height * scale)
        else:
            scale = long_side / height
            new_height = long_side
            new_width = int(width * scale)
        return main_image.resize((new_width, new_height), Image.LANCZOS)
    
    def combine_main_sub(self, main_img, sub_grid, position, border) -> Image:
        spacing = border * 2
        
        if position in ['top', 'bottom']:
            max_width = max(main_img.width, sub_grid.width)
            total_width = max_width + 2 * border
            total_height = main_img.height + sub_grid.height + spacing + 2 * border

            result = Image.new('RGB', (total_width, total_height), color='white')

            if position == 'top':
                main_y = border
                sub_y = main_y + main_img.height + spacing
            else:
                sub_y = border
                main_y = sub_y + sub_grid.height + spacing

            main_x = border + (max_width - main_img.width) // 2
            sub_x = border + (max_width - sub_grid.width) // 2

        else:
            max_height = max(main_img.height, sub_grid.height)
            total_height = max_height + 2 * border
            total_width = main_img.width + sub_grid.width + spacing + 2 * border

            result = Image.new('RGB', (total_width, total_height), color='white')

            if position == 'left':
                main_x = border
                sub_x = main_x + main_img.width + spacing
            else:
                sub_x = border
                main_x = sub_x + sub_grid.width + spacing

            main_y = border + (max_height - main_img.height) // 2
            sub_y = border + (max_height - sub_grid.height) // 2

        result.paste(main_img, (main_x, main_y))
        result.paste(sub_grid, (sub_x, sub_y))
        return result






#region-----------------纹理组---------------------------------


ort.disable_telemetry_events()

class ModelNotFound(Exception):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(f"The model {model_name} could not be found.", *args, **kwargs)

def tensor2pil(tensor: torch.Tensor) -> list[Image.Image]:
    if tensor.ndim == 3:
        if tensor.shape[2] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[3] in [1, 3, 4]:
            tensor = tensor.permute(0, 3, 1, 2)
    
    images = []
    for t in tensor:
        np_array = t.permute(1, 2, 0).cpu().detach().numpy()
        if np_array.dtype in [np.float32, np.float64]:
            np_array = (np_array * 255).astype(np.uint8)
        
        if np_array.ndim == 2:
            img = Image.fromarray(np_array, mode="L")
        elif np_array.shape[2] == 3:
            img = Image.fromarray(np_array, mode="RGB")
        elif np_array.shape[2] == 4:
            img = Image.fromarray(np_array, mode="RGBA")
        else:
            raise ValueError(f"不支持的通道数: {np_array.shape[2]}（张量形状: {t.shape}）")
        images.append(img)
    return images

def corner_mask(side_length):
    corner = np.zeros([side_length, side_length])
    for h in range(side_length):
        for w in range(side_length):
            if h >= w:
                sh = h / (side_length - 1)
                corner[h, w] = 1 - sh
            if h <= w:
                sw = w / (side_length - 1)
                corner[h, w] = 1 - sw
    return corner - 0.25 * scaling_mask(side_length)

def scaling_mask(side_length):
    scaling = np.zeros([side_length, side_length])
    for h in range(side_length):
        for w in range(side_length):
            sh = h / (side_length - 1)
            sw = w / (side_length - 1)
            if h >= w and h <= side_length - w:
                scaling[h, w] = sw
            if h <= w and h <= side_length - w:
                scaling[h, w] = sh
            if h >= w and h >= side_length - w:
                scaling[h, w] = 1 - sh
            if h <= w and h >= side_length - w:
                scaling[h, w] = 1 - sw
    return 2 * scaling

def generate_mask(tile_size, stride_size):
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    ramp_h = tile_h - stride_h
    ramp_w = tile_w - stride_w
    mask = np.ones((tile_h, tile_w))
    mask[ramp_h:-ramp_h, :ramp_w] = np.linspace(0, 1, num=ramp_w)
    mask[ramp_h:-ramp_h, -ramp_w:] = np.linspace(1, 0, num=ramp_w)
    mask[:ramp_h, ramp_w:-ramp_w] = np.transpose(np.linspace(0, 1, num=ramp_h)[None], (1, 0))
    mask[-ramp_h:, ramp_w:-ramp_w] = np.transpose(np.linspace(1, 0, num=ramp_h)[None], (1, 0))
    assert ramp_h == ramp_w
    corner = np.rot90(corner_mask(ramp_h), 2)
    mask[:ramp_h, :ramp_w] = corner
    corner = np.flip(corner, 1)
    mask[:ramp_h, -ramp_w:] = corner
    corner = np.flip(corner, 0)
    mask[-ramp_h:, -ramp_w:] = corner
    corner = np.flip(corner, 1)
    mask[-ramp_h:, :ramp_w] = corner
    return mask

def pad(img, left, right, top, bottom):
    pad_width = np.array(((0, 0), (top, bottom), (left, right)))
    return np.pad(img, pad_width, mode="wrap")

def tiles_infer(tiles, ort_session, progress_callback=None):
    out_channels = 3
    tiles_nb = tiles.shape[0]
    pred_tiles = np.empty((tiles_nb, out_channels, tiles.shape[2], tiles.shape[3]))
    for i in range(tiles_nb):
        if progress_callback:
            progress_callback(i + 1, tiles_nb)
        pred_tiles[i] = ort_session.run(None, {"input": tiles[i:i+1].astype(np.float32)})[0]
    return pred_tiles

def tiles_merge(tiles, stride_size, img_size, paddings):
    _, tile_h, tile_w = tiles[0].shape
    pad_left, pad_right, pad_top, pad_bottom = paddings
    height = img_size[1] + pad_top + pad_bottom
    width = img_size[2] + pad_left + pad_right
    stride_h, stride_w = stride_size
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    assert (stride_h >= tile_h / 2) and (stride_w >= tile_w / 2)
    assert (stride_h <= tile_h) and (stride_w <= tile_w)
    merged = np.zeros((img_size[0], height, width))
    mask = generate_mask((tile_h, tile_w), stride_size)
    h_range = ((height - tile_h) // stride_h) + 1
    w_range = ((width - tile_w) // stride_w) + 1
    idx = 0
    for h in range(h_range):
        for w in range(w_range):
            h_from, h_to = h * stride_h, h * stride_h + tile_h
            w_from, w_to = w * stride_w, w * stride_w + tile_w
            merged[:, h_from:h_to, w_from:w_to] += tiles[idx] * mask
            idx += 1
    return merged[:, pad_top:-pad_bottom, pad_left:-pad_right]

def tiles_split(img, tile_size, stride_size):
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    img_h, img_w = img.shape[0], img.shape[1]
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    assert (stride_h >= tile_h / 2) and (stride_w >= tile_w / 2)
    assert (stride_h <= tile_h) and (stride_w <= tile_w)
    pad_h, pad_w = 0, 0
    remainer_h = (img_h - tile_h) % stride_h
    remainer_w = (img_w - tile_w) % stride_w
    if remainer_h != 0:
        pad_h = stride_h - remainer_h
    if remainer_w != 0:
        pad_w = stride_w - remainer_w
    if tile_h > img_h:
        pad_h = tile_h - img_h
    if tile_w > img_w:
        pad_w = tile_w - img_w
    pad_left = pad_w // 2 + stride_w
    pad_right = pad_left if pad_w % 2 == 0 else pad_left + 1
    pad_top = pad_h // 2 + stride_h
    pad_bottom = pad_top if pad_h % 2 == 0 else pad_top + 1
    img = pad(img, pad_left, pad_right, pad_top, pad_bottom)
    img_h, img_w = img.shape[1], img.shape[2]
    h_range = ((img_h - tile_h) // stride_h) + 1
    w_range = ((img_w - tile_w) // stride_w) + 1
    tiles = np.empty([h_range * w_range, img.shape[0], tile_h, tile_w])
    idx = 0
    for h in range(h_range):
        for w in range(w_range):
            h_from, h_to = h * stride_h, h * stride_h + tile_h
            w_from, w_to = w * stride_w, w * stride_w + tile_w
            tiles[idx] = img[:, h_from:h_to, w_from:w_to]
            idx += 1
    return tiles, (pad_left, pad_right, pad_top, pad_bottom)

def color_to_normals(color_img, overlap, progress_callback=None):
    img = np.mean(color_img[:3], axis=0, keepdims=True)
    tile_size = 256
    overlaps = {"SMALL": tile_size // 6, "MEDIUM": tile_size // 4, "LARGE": tile_size // 2}
    stride_size = tile_size - overlaps[overlap]
    tiles, paddings = tiles_split(img, (tile_size, tile_size), (stride_size, stride_size))
    
    # 本地模型路径（已按你的正确路径修改）
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "deepbump" / "deepbump256.onnx"
    if not model_path.exists():
        raise ModelNotFound(f"deepbump ({model_path})")
    
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLProvider", "CPUExecutionProvider"]
    available_providers = [p for p in providers if p in ort.get_available_providers()]
    if not available_providers:
        raise RuntimeError("No valid ONNX Runtime providers available.")
    
    ort_session = ort.InferenceSession(model_path.as_posix(), providers=available_providers)
    pred_tiles = tiles_infer(tiles, ort_session, progress_callback=progress_callback)
    pred_img = tiles_merge(pred_tiles, (stride_size, stride_size), (3, img.shape[1], img.shape[2]), paddings)
    return normalize(pred_img)

def conv_1d(array, kernel_1d):
    k_l = len(kernel_1d)
    assert k_l % 2 != 0
    extended = np.pad(array, k_l // 2, mode="wrap")
    output = np.empty(array.shape)
    for i in range(array.shape[0]):
        output[i] = np.convolve(extended[i + (k_l // 2)], kernel_1d, mode="valid")
    return output * -1

def gaussian_kernel(length, sigma):
    space = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    kernel = np.exp(-0.5 * np.square(space) / np.square(sigma))
    return kernel / np.sum(kernel)

def normalize(np_array):
    return (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array) + 1e-16)

def normals_to_curvature(normals_img, blur_radius, progress_callback=None):
    if progress_callback:
        progress_callback(0, 4)
    diff_kernel = np.array([-1, 0, 1])
    h_conv = conv_1d(normals_img[0, :, :], diff_kernel)
    if progress_callback:
        progress_callback(1, 4)
    v_conv = conv_1d(-1 * normals_img[1, :, :].T, diff_kernel).T
    if progress_callback:
        progress_callback(2, 4)
    edges_conv = h_conv + v_conv
    blur_factors = {"SMALLEST": 1/256, "SMALLER": 1/128, "SMALL": 1/64, "MEDIUM": 1/32, "LARGE": 1/16, "LARGER": 1/8, "LARGEST": 1/4}
    if blur_radius not in blur_factors:
        raise ValueError(f"{blur_radius} not in {blur_factors}")
    blur_radius_px = int(np.mean(normals_img.shape[1:3]) * blur_factors[blur_radius])
    if blur_radius_px < 2:
        edges_conv = normalize(edges_conv)
        return np.stack([edges_conv, edges_conv, edges_conv])
    if blur_radius_px % 2 == 0:
        blur_radius_px += 1
    sigma = blur_radius_px // 8 if blur_radius_px // 8 != 0 else 1
    g_kernel = gaussian_kernel(blur_radius_px, sigma)
    h_blur = conv_1d(edges_conv, g_kernel)
    if progress_callback:
        progress_callback(3, 4)
    v_blur = conv_1d(h_blur.T, g_kernel).T
    if progress_callback:
        progress_callback(4, 4)
    curvature = normalize(v_blur)
    return np.stack([curvature, curvature, curvature])

def normals_to_grad(normals_img):
    return (normals_img[0] - 0.5) * 2, (normals_img[1] - 0.5) * 2

def copy_flip(grad_x, grad_y):
    grad_x_top = np.hstack([grad_x, -np.flip(grad_x, axis=1)])
    grad_x_bottom = np.hstack([np.flip(grad_x, axis=0), -np.flip(grad_x)])
    new_grad_x = np.vstack([grad_x_top, grad_x_bottom])
    grad_y_top = np.hstack([grad_y, np.flip(grad_y, axis=1)])
    grad_y_bottom = np.hstack([-np.flip(grad_y, axis=0), -np.flip(grad_y)])
    new_grad_y = np.vstack([grad_y_top, grad_y_bottom])
    return new_grad_x, new_grad_y

def frankot_chellappa(grad_x, grad_y, progress_callback=None):
    if progress_callback:
        progress_callback(0, 3)
    rows, cols = grad_x.shape
    rows_scale = (np.arange(rows) - (rows // 2 + 1)) / (rows - rows % 2)
    cols_scale = (np.arange(cols) - (cols // 2 + 1)) / (cols - cols % 2)
    u_grid, v_grid = np.meshgrid(cols_scale, rows_scale)
    u_grid = np.fft.ifftshift(u_grid)
    v_grid = np.fft.ifftshift(v_grid)
    if progress_callback:
        progress_callback(1, 3)
    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)
    if progress_callback:
        progress_callback(2, 3)
    nominator = (-1j * u_grid * grad_x_F) + (-1j * v_grid * grad_y_F)
    denominator = (u_grid**2) + (v_grid**2) + 1e-16
    Z_F = nominator / denominator
    Z_F[0, 0] = 0.0
    Z = np.real(np.fft.ifft2(Z_F))
    if progress_callback:
        progress_callback(3, 3)
    return (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-16)

def normals_to_height(normals_img, seamless, progress_callback=None):
    flip_img = np.flip(normals_img, axis=1)
    grad_x, grad_y = normals_to_grad(flip_img)
    grad_x = np.flip(grad_x, axis=0)
    grad_y = np.flip(grad_y, axis=0)
    if not seamless:
        grad_x, grad_y = copy_flip(grad_x, grad_y)
    pred_img = frankot_chellappa(-grad_x, grad_y, progress_callback=progress_callback)
    if not seamless:
        height, width = normals_img.shape[1], normals_img.shape[2]
        pred_img = pred_img[:height, :width]
    return np.stack([pred_img, pred_img, pred_img])



class texture_create:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normals_overlap": (["SMALL", "MEDIUM", "LARGE"],),
                "curvature_blur": (["SMALLEST", "SMALLER", "SMALL", "MEDIUM", "LARGE", "LARGER", "LARGEST"],),
                "height_seamless": ("BOOLEAN", {"default": True}),
                "highlight_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "highlight_contrast": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "highlight_sharpen": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Normals", "Curvature", "Height", "Highlight")
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/imgEffect"
    
    def apply(self, *, image, normals_overlap="SMALL", 
              curvature_blur="SMALL", height_seamless=True,
              highlight_threshold=200, highlight_contrast=1.5, highlight_sharpen=True):
        images = tensor2pil(image)
        out_normals = []
        out_curvature = []
        out_height = []
        out_highlight = []
        
        for img in images:
            in_img = np.transpose(img, (2, 0, 1)) / 255
            
            normals_img = color_to_normals(in_img, normals_overlap)
            curvature_img = normals_to_curvature(normals_img, curvature_blur)
            height_img = normals_to_height(normals_img, height_seamless)
            
            gray_img = img.convert("L")
            gray_arr = np.array(gray_img, dtype=np.float32)
            highlight_arr = np.where(gray_arr < highlight_threshold, 0, gray_arr)
            highlight_arr = (highlight_arr - highlight_threshold) * highlight_contrast
            highlight_arr = np.clip(highlight_arr, 0, 255)
            
            if highlight_sharpen:
                laplacian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                pad_arr = np.pad(highlight_arr, 1, mode="edge")
                sharpened_arr = np.zeros_like(highlight_arr)
                for i in range(highlight_arr.shape[0]):
                    for j in range(highlight_arr.shape[1]):
                        sharpened_arr[i, j] = np.sum(pad_arr[i:i+3, j:j+3] * laplacian_kernel)
                highlight_arr = np.clip(sharpened_arr, 0, 255)
            
            highlight_arr = highlight_arr.astype(np.float32) / 255.0
            highlight_arr = np.stack([highlight_arr, highlight_arr, highlight_arr], axis=-1)
            
            out_normals.append(torch.from_numpy(np.transpose(normals_img, (1, 2, 0)).astype(np.float32)).unsqueeze(0))
            out_curvature.append(torch.from_numpy(np.transpose(curvature_img, (1, 2, 0)).astype(np.float32)).unsqueeze(0))
            out_height.append(torch.from_numpy(np.transpose(height_img, (1, 2, 0)).astype(np.float32)).unsqueeze(0))
            out_highlight.append(torch.from_numpy(highlight_arr).unsqueeze(0))
        
        return (
            torch.cat(out_normals, dim=0),
            torch.cat(out_curvature, dim=0),
            torch.cat(out_height, dim=0),
            torch.cat(out_highlight, dim=0)
        )


#endregion-----------------纹理组---------------------------------



#region------------------------------------
def blend_normal(backdrop, source, opacity):
    alpha = source[..., 3:4] / 255.0 * opacity
    # 修复：截取backdrop前3通道（RGB），与source[..., :3]通道数保持一致
    return backdrop[..., :3] * (1 - alpha) + source[..., :3] * alpha

def blend_add(backdrop, source, opacity):
    alpha = opacity
    result = backdrop[..., :3] + source[..., :3] * alpha
    return np.clip(result, 0.0, 255.0)

def blend_multiply(backdrop, source, opacity):
    alpha = opacity
    backdrop_norm = backdrop[..., :3] / 255.0
    source_norm = source[..., :3] / 255.0
    result = backdrop_norm * source_norm * alpha + backdrop_norm * (1 - alpha)
    return result * 255.0

def blend_screen(backdrop, source, opacity):
    alpha = opacity
    backdrop_norm = backdrop[..., :3] / 255.0
    source_norm = source[..., :3] / 255.0
    result = (1 - (1 - backdrop_norm) * (1 - source_norm)) * alpha + backdrop_norm * (1 - alpha)
    return result * 255.0

def blend_overlay(backdrop, source, opacity):
    alpha = opacity
    backdrop_norm = backdrop[..., :3] / 255.0
    source_norm = source[..., :3] / 255.0
    result = np.where(backdrop_norm < 0.5, 2 * backdrop_norm * source_norm, 1 - 2 * (1 - backdrop_norm) * (1 - source_norm))
    result = result * alpha + backdrop_norm * (1 - alpha)
    return result * 255.0

def blend_soft_light(backdrop, source, opacity):
    alpha = opacity
    backdrop_norm = backdrop[..., :3] / 255.0
    source_norm = source[..., :3] / 255.0
    def _g(x):
        return np.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, np.sqrt(x))
    result = np.where(source_norm <= 0.5, backdrop_norm - (1 - 2 * source_norm) * backdrop_norm * (1 - backdrop_norm), backdrop_norm + (2 * source_norm - 1) * (_g(backdrop_norm) - backdrop_norm))
    result = result * alpha + backdrop_norm * (1 - alpha)
    return result * 255.0

def blend_linear_light(backdrop, source, opacity):
    alpha = opacity
    backdrop_norm = backdrop[..., :3] / 255.0
    source_norm = source[..., :3] / 255.0
    result = np.where(source_norm > 0.5, backdrop_norm + 2 * (source_norm - 0.5), backdrop_norm + 2 * source_norm - 1)
    result = np.clip(result, 0.0, 1.0)
    result = result * alpha + backdrop_norm * (1 - alpha)
    return result * 255.0

BLEND_FUNCTIONS = {
    'normal': blend_normal, 'add': blend_add, 'multiply': blend_multiply,
    'screen': blend_screen, 'overlay': blend_overlay, 'soft light': blend_soft_light, 'linear light': blend_linear_light
}

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def chop_image_v2(background_image: Image, layer_image: Image, blend_mode: str, opacity: int) -> Image:
    # 优化：先统一转为RGB再转RGBA，确保输入图像基础模式一致
    background_image = background_image.convert('RGB').convert('RGBA')
    layer_image = layer_image.convert('RGB').convert('RGBA')
    
    backdrop_prepped = np.asarray(background_image, dtype=np.float32)
    source_prepped = np.asarray(layer_image, dtype=np.float32)
    
    # 优化：增加通道数校验，提前暴露异常
    if backdrop_prepped.shape[-1] != 4 or source_prepped.shape[-1] != 4:
        raise ValueError(f"图像通道数错误，需为RGBA（4通道），当前背景图通道：{backdrop_prepped.shape[-1]}，图层通道：{source_prepped.shape[-1]}")
    
    blend_func = BLEND_FUNCTIONS.get(blend_mode.lower())
    if not blend_func:
        raise ValueError(f"不支持的混合模式: {blend_mode}，可选模式：{list(BLEND_FUNCTIONS.keys())}")
    
    opacity_norm = opacity / 100.0
    blended_np = blend_func(backdrop_prepped, source_prepped, opacity_norm)
    return Image.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8)).convert('RGB')

def gaussian_blur(image: Image, radius: int) -> Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

class Image_Detail_HL_frequencye:
    def __init__(self):
        self.NODE_NAME = 'HLFrequencyDetailRestore'
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE",),  # 目标图（打光图）
                "detail_image": ("IMAGE",),  # 细节源图（原图）
                "keep_high_freq": ("INT", {"default": 64, "min": 0, "max": 1023}),  # 细节提取模糊半径
                "erase_low_freq": ("INT", {"default": 32, "min": 0, "max": 1023}),  # 光影平滑模糊半径
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 1023}),  # 遮罩边缘模糊
                "blend_mode": (list(BLEND_FUNCTIONS.keys()), {"default": "linear light"}),  # 混合模式
                "blend_opacity": ("INT", {"default": 100, "min": 0, "max": 100}),  # 混合强度
                "detail_strength": ("INT", {"default": 100, "min": 0, "max": 200}),  # 细节强度（0-200%）
                "high_freq_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 高频细节对比度阈值
            },
            "optional": {
                "mask": ("MASK",),  # 细节应用区域遮罩
                "invert_mask": ("BOOLEAN", {"default": False}),  # 是否反转遮罩
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'frequency_detail_restore'
    CATEGORY = "Apt_Preset/image"
    DESCRIPTION = """
    细节太粗 / 太细 调 高频保留keep_high_freq（小→细，大→粗）。
    光影太硬 / 太糊 调 擦除低频erase_low_freq（小→硬，大→糊）。
    效果太淡 / 太浓 调 混合强度blend_opacity（小→淡，大→浓）。
    细节太弱 / 过曝 调 细节强度detail_strength（小→弱，大→强）。
    噪点多 / 细节杂 调 高频细节对比度（小→留杂，大→去杂）。
    边缘太硬 / 太糊 调 mask_blur（小→硬，大→糊）。
    """
    def frequency_detail_restore(self, image, detail_image, keep_high_freq, erase_low_freq, mask_blur, blend_mode="overlay", blend_opacity=100, detail_strength=100, high_freq_threshold=0.5, mask=None, invert_mask=False):
        b_images = [torch.unsqueeze(b, 0) for b in image]
        l_images = [torch.unsqueeze(l, 0) for l in detail_image]
        l_masks = []
        
        for l in detail_image:
            m = tensor2pil(l)
            l_masks.append(m.split()[-1] if m.mode == 'RGBA' else Image.new('L', m.size, 'white'))
        
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = [tensor2pil(torch.unsqueeze(m, 0)).convert('L') for m in mask]
            if invert_mask:
                l_masks = [ImageChops.invert(m) for m in l_masks]
        
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        ret_images = []
        
        for i in range(max_batch):
            bg_img = tensor2pil(b_images[i % len(b_images)]).convert('RGB')
            dt_img = tensor2pil(l_images[i % len(l_images)]).convert('RGB')
            _mask = l_masks[i % len(l_masks)]
            
            # 提取高频细节（增强强度控制）
            blurred_dt = gaussian_blur(dt_img, keep_high_freq)
            high_freq = chop_image_v2(ImageChops.invert(dt_img), blurred_dt, 'normal', 50)
            high_freq = ImageChops.invert(high_freq)
            
            # 细节强度调整
            if detail_strength != 100:
                high_freq = ImageChops.blend(dt_img, high_freq, alpha=detail_strength/100)
            
            # 高频阈值过滤（弱化低对比度细节）
            high_freq_np = np.array(high_freq).astype(np.float32)
            blurred_high = gaussian_blur(high_freq, 1)
            high_contrast = np.abs(high_freq_np - np.array(blurred_high)) / 255.0
            mask_threshold = (high_contrast >= high_freq_threshold).astype(np.float32)
            high_freq = Image.fromarray(np.uint8(high_freq_np * mask_threshold + np.array(blurred_high) * (1 - mask_threshold)))
            
            # 低频处理
            low_freq = gaussian_blur(bg_img, erase_low_freq) if erase_low_freq > 0 else bg_img.copy()
            
            # 混合
            ret_image = chop_image_v2(low_freq, high_freq, blend_mode, blend_opacity)
            
            # 遮罩应用
            _mask_inv = ImageChops.invert(_mask)
            if mask_blur > 0:
                _mask_inv = gaussian_blur(_mask_inv, mask_blur)
            ret_image.paste(bg_img, mask=_mask_inv)
            
            ret_images.append(pil2tensor(ret_image))
        
        return (torch.cat(ret_images, dim=0),)
    
#endregion-------------------------------------------------------------------------------

























































