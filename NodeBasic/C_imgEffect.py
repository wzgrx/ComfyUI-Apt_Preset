import os
import torch
import numpy as np
import cv2
import folder_paths
from PIL import Image, ImageOps, ImageEnhance, Image, ImageOps, ImageChops, ImageFilter, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import io
from typing import Literal, Any
import math
from comfy.utils import common_upscale
import typing as t

from math import ceil, sqrt
from ..main_unit import *





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


COLORS = ["custom", "white", "black", "red", "green", "blue", "yellow",
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
    CATEGORY = "Apt_Preset/imgEffect"

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
    CATEGORY = "Apt_Preset/imgEffect"
    
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
    CATEGORY = "Apt_Preset/imgEffect"
    
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
    CATEGORY = "Apt_Preset/imgEffect"
    
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
    CATEGORY = "Apt_Preset/imgEffect"

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



class create_overShape:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wide": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "shape": (
                    [
                        'circle',
                        'square',
                        'semicircle',
                        'quarter_circle',
                        'ellipse',
                        'triangle',
                        'cross',
                        'star',
                        'radial',
                    ],
                    {"default": "circle"},
                ),
                "X_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "Y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "background_color": ("COLOR",{"default": "#FFFFFF"},),
                "shape_color": ("COLOR", {"default": "#000000"},),
            },
            "optional": {
                "base_image": ("IMAGE", {"default": None}),
            },
        }

    CATEGORY = "Apt_Preset/imgEffect"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "drew_light_shape"

    def drew_light_shape(self, wide, height, shape, X_offset, Y_offset, scale, rotation, opacity, blur_radius, background_color,
                         shape_color, base_image=None):

        if base_image is None:
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius)
        else:
            img_cv = Image.fromarray((base_image.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius, base_image=img_cv)

        rst = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)


class create_RadialGradient:
    @classmethod
    def INPUT_TYPES(s):
    
        return {"required": {
                    "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    
                    "gradient_distance": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
                    "radial_center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "radial_center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    
                    "start_color_hex": ("COLOR", {"default": "#000000"}),
                    "end_color_hex": ("COLOR", {"default": "#ffffff"}),
                    
                    },
                "optional": {

                }
        }

    RETURN_TYPES = ("IMAGE",  )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "draw"
    CATEGORY = "Apt_Preset/imgEffect"

    def draw(self, width, height, 
            radial_center_x=0.5, radial_center_y=0.5, gradient_distance=1,
            start_color_hex='#000000', end_color_hex='#ffffff'): # Default to .5 if the value is not found

        color1_rgb = hex_to_rgb_tuple(start_color_hex)

        color2_rgb = hex_to_rgb_tuple(end_color_hex)

        # Create a blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        center_x = int(radial_center_x * width)
        center_y = int(radial_center_y * height)                
        # Computation for max_distance
        max_distance = (np.sqrt(max(center_x, width - center_x)**2 + max(center_y, height - center_y)**2))*gradient_distance

        for i in range(width):
            for j in range(height):
                distance_to_center = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                t = distance_to_center / max_distance
                # Ensure t is between 0 and 1
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


        return (image_out, )


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
                    "start_color_hex": ("COLOR", {"default": "#000000"}),
                    "end_color_hex": ("COLOR", {"default": "#ffffff"}),
                    
                    },
                "optional": {
                }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "draw"
    CATEGORY = "Apt_Preset/imgEffect"

    def draw(self, width, height, orientation, start_color_hex='#000000', end_color_hex='#ffffff', 
            linear_transition=0.5, gradient_distance=1,): 
        

            
        color1_rgb = hex_to_rgb_tuple(start_color_hex)

        color2_rgb = hex_to_rgb_tuple(end_color_hex)

        # Create a blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Convert orientation angle to radians
        angle = np.radians(orientation)
        
        # Create gradient based on angle
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate gradient based on angle
        gradient = X * np.cos(angle) + Y * np.sin(angle)
        
        # Normalize gradient to 0-1 range
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
        # Apply gradient distance and transition
        gradient = (gradient - (linear_transition - gradient_distance/2)) / gradient_distance
        gradient = np.clip(gradient, 0, 1)
        
        # Apply gradient to colors
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


        return (image_out,  )


class create_AD_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "method": (["linear", "in", "out", "in-out"],),
                
                "invertMask": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "Apt_Preset/imgEffect"

    def linear(self, i, t):
        return i/t
    def ease_in(self, i, t):
        return pow(i/t, 2)
    def ease_out(self, i, t):
        return 1 - pow(1 - i/t, 2)
    def ease_in_out(self, i, t):
        if i < t/2:
            return pow(i/(t/2), 2) / 2
        else:
            return 1 - pow(1 - (i - t/2)/(t/2), 2) / 2

    def run(self, width, height, frames, start_frame, end_frame, transition_type, method, invertMask):
        if method == 'in':
            method = self.ease_in
        elif method == 'out':
            method = self.ease_out
        elif method == 'in-out':
            method = self.ease_in_out
        else:
            method = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = method(i, transition-1)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)

        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)
        
        if invertMask:
            out = 1.0 - out


        return (out, )


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


class create_mulcolor_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/imgEffect"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)


#endregion-----------------------------------------------------------------------------------------------------



#region------------------layout----------------------


class lay_ImageGrid:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"images": ("IMAGE",), "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}), "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grid_images"
    CATEGORY = "Apt_Preset/imgEffect"
    def grid_images(self, images, rows, cols):
        images = images.cpu().numpy()
        batch_size, height, width, channels = images.shape
        grid_width = width * cols
        grid_height = height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height))
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            img = Image.fromarray((images[i] * 255).astype(np.uint8))
            x = col * width
            y = row * height
            grid_image.paste(img, (x, y))
        grid_image = np.array(grid_image).astype(np.float32) / 255.0
        grid_image = torch.from_numpy(grid_image)[None,]
        return (grid_image,)


class lay_MaskGrid:
    @classmethod 
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}), "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "grid_masks"
    CATEGORY = "Apt_Preset/imgEffect"

    def grid_masks(self, masks, rows, cols):
        masks = masks.cpu().numpy()
        batch_size, height, width = masks.shape
        grid_width = width * cols
        grid_height = height * rows
        grid_mask = Image.new('L', (grid_width, grid_height))
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            mask = Image.fromarray((masks[i] * 255).astype(np.uint8))
            x = col * width
            y = row * height
            grid_mask.paste(mask, (x, y))
        grid_mask = np.array(grid_mask).astype(np.float32) / 255.0
        grid_mask = torch.from_numpy(grid_mask)[None,]
        return (grid_mask,)


class lay_image_match_W_or_H:
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


    def concatenate(self, image1, image2, direction, first_image_shape=None):

        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2
            
            # Repeat the last image to match the largest batch size
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        target_shape = first_image_shape if first_image_shape is not None else image1.shape
        original_height = image2.shape[1]
        original_width = image2.shape[2]
        original_aspect_ratio = original_width / original_height

        if direction in ['left', 'right']:
            target_height = target_shape[1]  # B, H, W, C format
            target_width = int(target_height * original_aspect_ratio)
        elif direction in ['up', 'down']:

            target_width = target_shape[2]  # B, H, W, C format
            target_height = int(target_width / original_aspect_ratio)

        image2_for_upscale = image2.movedim(-1, 1) 
        image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
        image2_resized = image2_resized.movedim(1, -1)


        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)

        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,


class lay_image_match_W_and_H:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "border_color": ("COLOR", {"default": "#000000"}),
                "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "outline_color": ("COLOR", {"default": "#000000"}),
                "rows": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "cols": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = "Apt_Preset/imgEffect"

    def process_images(self, count,
                    border_thickness,
                    outline_thickness, 
                    rows, cols, 
                    outline_color='#000000', 
                    border_color='#000000', **kwargs):
        
        # Get all images
        images = []
        for i in range(1, count + 1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                images.append(tensor2pil(image))

        # Resize images to match first image size
        if len(images) > 0:
            first_size = images[0].size
            for i in range(1, len(images)):
                if images[i].size != first_size:
                    images[i] = images[i].resize(first_size)

        # Apply borders and outlines
        images = apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color)

        # Combine images into a grid
        combined_image = self.combine_images_grid(images, rows, cols)
        return (pil2tensor(combined_image),)

    def combine_images_grid(self, images, rows, cols):
        if not images:
            return Image.new('RGB', (0, 0))

        # Calculate grid size
        img_width, img_height = images[0].size
        grid_width = img_width * cols
        grid_height = img_height * rows

        # Create blank canvas
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # Paste images into grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(img, (x, y))

        return grid_image


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


class lay_fill_inpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img1": ("IMAGE",),
                "img2": ("IMAGE",),
                "scale_by_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "scale_by_height": ("INT", {"default": 0, "min": 0, "max": 4096})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "concat_two_images"
    CATEGORY = "Apt_Preset/imgEffect"

    def concat_two_images(self, img1, img2, scale_by_width=0, scale_by_height=0):
        i1 = 255. * img1[0].cpu().numpy()
        image1 = Image.fromarray(np.clip(i1, 0, 255).astype(np.uint8))
        
        i2 = 255. * img2[0].cpu().numpy()
        image2 = Image.fromarray(np.clip(i2, 0, 255).astype(np.uint8))

        width1, height1 = image1.size
        width2, height2 = image2.size

        if height2 != height1:
            new_width = int(width2 * (height1 / height2))
            image2 = image2.resize((new_width, height1), resample=Image.LANCZOS)
            width2, height2 = image2.size

        total_width = width1 + width2
        max_height = height1

        new_img = Image.new("RGB", (total_width, max_height))
        new_img.paste(image1, (0, 0))
        new_img.paste(image2, (width1, 0))

        mask_img = Image.new("L", (total_width, max_height), color=0)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([width1, 0, total_width, max_height], fill=255)

        target_size = (total_width, max_height)
        if scale_by_width > 0 and scale_by_height == 0:
            target_ratio = max_height / total_width
            target_size = (scale_by_width, int(scale_by_width * target_ratio))
        elif scale_by_height > 0:
            target_ratio = total_width / max_height
            target_size = (int(scale_by_height * target_ratio), scale_by_height)

        resized_img = new_img.resize(target_size, resample=Image.LANCZOS)
        resized_mask = mask_img.resize(target_size, resample=Image.LANCZOS)

        new_img_tensor = torch.from_numpy(np.array(resized_img).astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(resized_mask).astype(np.float32) / 255.0).unsqueeze(0)

        return (new_img_tensor, mask_tensor)


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
    
    def layout(self, text1, text2, footer_height, font_name, font_size, mode, border_thickness, image1=None, image2=None):
        if mode == "normal":
            font_color = "black"
            bg_color = "white"
        else:
            font_color = "white"
            bg_color = "black"
        if image1 is not None and image2 is not None:
            img1 = tensor2pil(image1)
            img2 = tensor2pil(image2)
            image_width, image_height = img1.width, img1.height
            if img2.width != img1.width or img2.height != img1.height:
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


class lay_iamges_free_layout:
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
                    "outline_color": ("COLOR",{"default": "#FFFFFF"}),
                    "panel_color": ("COLOR",{"default": "#00AAFF"}),
                    "bg_color": ("COLOR",{"default": "#000000"}),
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


class lay_text_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": False, "default": "text"}),
                    "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                    "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),  
                    "background_color": ("COLOR",{"default": "#000000"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "text_color": ("COLOR",), 
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
                "bg_color": ("COLOR", {"default": "#FFFFFF"}),
            }
        }

    FUNCTION = "create_grid"
    CATEGORY = "Apt_Preset/imgEffect"
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

#endregion--------------layout----------------------------------