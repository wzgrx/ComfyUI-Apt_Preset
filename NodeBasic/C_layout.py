import torch
import numpy as np
from comfy.utils import  common_upscale
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from ..main_unit import tensor2pil, pil2tensor



ALIGN_OPTIONS = ["center", "top", "bottom"]                 
ROTATE_OPTIONS = ["text center", "image center"]
JUSTIFY_OPTIONS = ["center", "left", "right"]
PERSPECTIVE_OPTIONS = ["top", "bottom", "left", "right"]



#region--------------def--------textlayout----------------------
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

#endregion----------------------textlayout----------------------



class lay_ImageGrid:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"images": ("IMAGE",), "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}), "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grid_images"
    CATEGORY = "Apt_Preset/layout"
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
    CATEGORY = "Apt_Preset/layout"

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



class lay_text:
    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")       
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        return {"required": {

                    "image_bg": ("IMAGE",),
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "text_color": ("COLOR",), 
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (ROTATE_OPTIONS,),
                } ,   
                
            "optional": { 
                "text_bg": ("IMAGE",),},
                
    }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "composite_text"
    CATEGORY = "Apt_Preset/layout"
    
    def composite_text(self, image_bg, text,
                    font_name, font_size, 
                    margins, line_spacing,
                    position_x, position_y,
                    align, justify,text_color,
                    rotation_angle, rotation_options,text_bg=None, ):

        image_3d = image_bg[0, :, :, :]
        back_image = tensor2pil(image_3d)
        text_image = Image.new('RGB', back_image.size, text_color)
        
        if text_bg is not None:
            text_image = tensor2pil(text_bg[0, :, :, :])
        

        text_mask = Image.new('L', back_image.size)
    
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing, 
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        image_out = Image.composite(text_image, back_image, rotated_text_mask)       

        return (pil2tensor(image_out), )



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
    CATEGORY = "Apt_Preset/layout"


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



#region ---------------lay_image_match_W_and_Hh----------------


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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove the '#' character, if present
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

def get_color_values(color, color_hex, color_mapping):
    

    if color == "custom":
        color_rgb = hex_to_rgb(color_hex)
    else:
        color_rgb = color_mapping.get(color, (0, 0, 0))  # Default to black if the color is not found

    return color_rgb 

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 


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
                # Optional inputs can be added here if needed
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = "Apt_Preset/layout"

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


#endregion---------------------------



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
    CATEGORY = "Apt_Preset/layout"

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


