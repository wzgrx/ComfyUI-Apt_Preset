import os
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageColor
import numpy as np
import torch
from torchvision import transforms
import random






font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

class AD_font2img:
    def __init__(self):
        pass
    

    @classmethod
    def INPUT_TYPES(self):
        alignment_options = ["left top", "left center", "left bottom", "center top", "center center", "center bottom", "right top", "right center", "right bottom"]
        text_interpolation_options = ["strict","interpolation","cumulative"]
        return {
            "required": {
                "font_file": (file_list,),
                "font_color": ("STRING", {"default": "white", "display": "text"}),
                "background_color": ("STRING", {"default": "black", "display": "text"}),
                "line_spacing": ("INT", {"default": 5, "step": 1, "display": "number"}),
                "kerning": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "padding": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "image_height": ("INT", {"default": 100, "step": 1, "display": "number"}),
                "image_width": ("INT", {"default": 100, "step": 1, "display": "number"}),
                "text_alignment": (alignment_options, {"default": "center center", "display": "dropdown"}),
                "text_interpolation_options": (text_interpolation_options, {"default": "cumulative", "display": "dropdown"}),
                "text": ("STRING", {"multiline": True, "placeholder": "Text"}),
                "start_font_size": ("INT", {"default": 20, "min": 1, "step": 1, "display": "number"}),
                "end_font_size": ("INT", {"default": 20, "min": 1, "step": 1, "display": "number"}),
                "start_x_offset": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "end_x_offset": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "start_y_offset": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "end_y_offset": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "start_rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "end_rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "rotation_anchor_x": ("INT", {"default": 0, "step": 1}),
                "rotation_anchor_y": ("INT", {"default": 0, "step": 1})
            },
            "optional": {
                "input_images": ("IMAGE", {"default": None, "display": "input_images"}),
                "transcription": ("TRANSCRIPTION", {"default": None, "display": "transcription","forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "transcription_framestamps",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def run(self, end_font_size, start_font_size, text_interpolation_options, line_spacing, start_x_offset, end_x_offset, start_y_offset, end_y_offset, start_rotation, end_rotation, font_file, frame_count, text, font_color, background_color, image_width, image_height, text_alignment, rotation_anchor_x, rotation_anchor_y, kerning, padding, **kwargs):
        transcription = kwargs.get('transcription')
        if transcription != None:
            formatted_transcription = self.format_transcription(transcription, image_width, font_file, start_font_size,padding)
            text = formatted_transcription 
        else:
            formatted_transcription = None
        frame_text_dict, is_structured_input = self.parse_text_input(text, frame_count)
        frame_text_dict = self.process_text_mode(frame_text_dict, text_interpolation_options, is_structured_input, frame_count)
        rotation_increment = (end_rotation - start_rotation) / max(frame_count - 1, 1)
        x_offset_increment = (end_x_offset - start_x_offset) / max(frame_count - 1, 1)
        y_offset_increment = (end_y_offset - start_y_offset) / max(frame_count - 1, 1)
        font_size_increment = (end_font_size - start_font_size) / max(frame_count - 1, 1)
        input_images = kwargs.get('input_images', [None] * frame_count)
        images= self.generate_images(start_font_size, font_size_increment, frame_text_dict, rotation_increment, x_offset_increment, y_offset_increment, start_x_offset, end_x_offset, start_y_offset, end_y_offset, font_file, font_color, background_color, image_width, image_height, text_alignment, line_spacing, frame_count, input_images, rotation_anchor_x, rotation_anchor_y, kerning,padding)
        image_batch = torch.cat(images, dim=0)
        return (image_batch, formatted_transcription,)

    def format_transcription(self, transcription, image_width, font_file, font_size, padding):
        if not transcription:
            return ""
        _, _, _, transcription_fps = transcription[0]
        formatted_transcription = ""
        current_sentence = ""
        for i, (word, start_time, _, _) in enumerate(transcription):
            frame_number = round(start_time * transcription_fps)
            if not current_sentence:
                current_sentence = word
            else:
                new_sentence = current_sentence + " " + word
                width = self.get_text_width(new_sentence, font_file, font_size)
                if width <= image_width - padding:
                    current_sentence = new_sentence
                else:
                    formatted_transcription += f'"{frame_number}": "{current_sentence}",\n'
                    current_sentence = word
            formatted_transcription += f'"{frame_number}": "{current_sentence}",\n'
        if current_sentence:
            last_frame_number = round(transcription[-1][1] * transcription_fps)
            formatted_transcription += f'"{last_frame_number}": "{current_sentence}",\n'
        return formatted_transcription
    
    def get_text_width(self,text, font_file, font_size):
        font = self.get_font(font_file, font_size, os.path.dirname(__file__))
        text_width, _ = font.getsize(text)
        return text_width
    
    def calculate_text_position(self, image_width, image_height, text_width, text_height, text_alignment, x_offset, y_offset, padding):        
        if text_alignment == "left top":
            base_x, base_y = padding, padding
        elif text_alignment == "left center":
            base_x, base_y = padding, padding + (image_height - text_height) // 2
        elif text_alignment == "left bottom":
            base_x, base_y = padding, image_height - text_height - padding
        elif text_alignment == "center top":
            base_x, base_y = (image_width - text_width) // 2, padding
        elif text_alignment == "center center":
            base_x, base_y = (image_width - text_width) // 2, (image_height - text_height) // 2
        elif text_alignment == "center bottom":
            base_x, base_y = (image_width - text_width) // 2, image_height - text_height - padding
        elif text_alignment == "right top":
            base_x, base_y = image_width - text_width - padding, padding
        elif text_alignment == "right center":
            base_x, base_y = image_width - text_width - padding, (image_height - text_height) // 2
        elif text_alignment == "right bottom":
            base_x, base_y = image_width - text_width - padding, image_height - text_height - padding
        else:
            base_x, base_y = (image_width - text_width) // 2, (image_height - text_height) // 2
        final_x = base_x + x_offset
        final_y = base_y + y_offset
        return final_x, final_y

    def process_image_for_output(self, image):
        i = ImageOps.exif_transpose(image)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]
    
    def get_font(self, font_file, font_size, script_dir):
        if font_file == "default":
            return ImageFont.load_default()
        else:
            font_dir1 = os.path.join(script_dir, 'font')
            font_dir2 = os.path.join(script_dir, 'font_files')
            if os.path.exists(os.path.join(font_dir1, font_file)):
                full_font_file = os.path.join(font_dir1, font_file)
            elif os.path.exists(os.path.join(font_dir2, font_file)):
                full_font_file = os.path.join(font_dir2, font_file)
            else:
                raise FileNotFoundError(f"Font file '{font_file}' not found in 'font' or 'font_files' directories.")
            return ImageFont.truetype(full_font_file, font_size)
        
    def calculate_text_block_size(self, draw, text, font, line_spacing, kerning):
        lines = text.split('\n')
        max_width = 0
        font_size = int(font.size)
        for line in lines:
            line_width = sum(draw.textlength(char, font=font) + kerning for char in line)
            line_width -= kerning
            max_width = max(max_width, line_width)
        total_height = font_size * len(lines) + line_spacing * (len(lines) - 1)
        return max_width, total_height
    
    def parse_text_input(self, text_input, frame_count):
        structured_format = False
        frame_text_dict = {}
        lines = [line for line in text_input.split('\n') if line.strip()]
        if all(':' in line and line.split(':')[0].strip().replace('"', '').isdigit() for line in lines):
            structured_format = True
            for line in lines:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    frame_number = parts[0].strip().replace('"', '')
                    text = parts[1].strip().replace('"', '').replace(',', '')
                    frame_text_dict[frame_number] = text
        else:
            frame_text_dict = {str(i): text_input for i in range(1, frame_count + 1)}
        return frame_text_dict, structured_format

    def interpolate_text(self, frame_text_dict, frame_count):
        sorted_frames = sorted(int(k) for k in frame_text_dict.keys())
        if len(sorted_frames) == 1:
            single_frame_text = frame_text_dict[str(sorted_frames[0])]
            return {str(i): single_frame_text for i in range(1, frame_count + 1)}
        sorted_frames.append(frame_count)
        interpolated_text_dict = {}
        for i in range(len(sorted_frames) - 1):
            start_frame, end_frame = sorted_frames[i], sorted_frames[i + 1]
            start_text = frame_text_dict[str(start_frame)]
            end_text = frame_text_dict.get(str(end_frame), start_text)
            for frame in range(start_frame, end_frame):
                interpolated_text_dict[str(frame)] = self.calculate_interpolated_text(start_text, end_text, frame - start_frame, end_frame - start_frame)
        last_frame_text = frame_text_dict.get(str(frame_count), end_text)
        interpolated_text_dict[str(frame_count)] = last_frame_text
        return interpolated_text_dict

    def calculate_interpolated_text(self, start_text, end_text, frame_delta, total_frames):
        start_len = len(start_text)
        end_len = len(end_text)
        interpolated_len = int(start_len + (end_len - start_len) * frame_delta / total_frames)
        interpolated_text = ""
        for i in range(interpolated_len):
            if i < start_len and i < end_len:
                start_char_fraction = 1 - frame_delta / total_frames
                end_char_fraction = frame_delta / total_frames
                interpolated_char = self.interpolate_char(start_text[i], end_text[i], start_char_fraction, end_char_fraction)
                interpolated_text += interpolated_char
            elif i < end_len:
                interpolated_text += end_text[i]
            else:
                interpolated_text += " "
        return interpolated_text

    def interpolate_char(self, start_char, end_char, start_fraction, end_fraction):
        if random.random() < start_fraction:
            return start_char
        else:
            return end_char
        
    def cumulative_text(self, frame_text_dict, frame_count):
        cumulative_text_dict = {}
        last_text = ""
        for i in range(1, frame_count + 1):
            if str(i) in frame_text_dict:
                last_text = frame_text_dict[str(i)]
            cumulative_text_dict[str(i)] = last_text
        return cumulative_text_dict
    
    def process_text_mode(self, frame_text_dict, mode, is_structured_input, frame_count):
        if mode == "interpolation" and is_structured_input:
            return self.interpolate_text(frame_text_dict, frame_count)
        elif mode == "cumulative" and is_structured_input:
            return self.cumulative_text(frame_text_dict, frame_count)
        return frame_text_dict
    
    def prepare_image(self, input_image, image_width, image_height, background_color):
        if not isinstance(input_image, list):
            if isinstance(input_image, torch.Tensor):
                if input_image.dtype == torch.float:
                    input_image = (input_image * 255).byte()
                if input_image.ndim == 4:
                    processed_images = []
                    for img in input_image:
                        tensor_image = img.permute(2, 0, 1)
                        transform = transforms.ToPILImage()
                        try:
                            pil_image = transform(tensor_image)
                        except Exception as e:
                            print("Error during conversion:", e)
                            raise
                        processed_images.append(pil_image.resize((image_width, image_height), Image.ANTIALIAS))
                    return processed_images
                elif input_image.ndim == 3 and input_image.shape[0] in [3, 4]:
                    tensor_image = input_image.permute(1, 2, 0)
                    pil_image = transforms.ToPILImage()(tensor_image)
                    return pil_image.resize((image_width, image_height), Image.ANTIALIAS)
                else:
                    raise ValueError(f"Input image tensor has an invalid shape or number of channels: {input_image.shape}")
            else:
                return input_image.resize((image_width, image_height), Image.ANTIALIAS)
        else:
            background_color_tuple = ImageColor.getrgb(background_color)
            return Image.new('RGB', (image_width, image_height), color=background_color_tuple)

    def generate_images(self, start_font_size, font_size_increment, frame_text_dict, rotation_increment, x_offset_increment, y_offset_increment, start_x_offset, end_x_offset, start_y_offset, end_y_offset, font_file, font_color, background_color, image_width, image_height, text_alignment, line_spacing, frame_count, input_images, rotation_anchor_x, rotation_anchor_y, kerning, margin):        
        images = []
        prepared_images = self.prepare_image(input_images, image_width, image_height, background_color)
        if not isinstance(prepared_images, list):
            prepared_images = [prepared_images]
        for i in range(1, frame_count + 1):
            text = frame_text_dict.get(str(i), "")
            current_font_size = int(start_font_size + font_size_increment * (i - 1))
            font = self.get_font(font_file, current_font_size, os.path.dirname(__file__))
            image_index = min(i - 1, len(prepared_images) - 1)
            selected_image = prepared_images[image_index]
            draw = ImageDraw.Draw(selected_image)
            text_width, text_height = self.calculate_text_block_size(draw, text, font, line_spacing, kerning)
            x_offset = start_x_offset + x_offset_increment * (i - 1)
            y_offset = start_y_offset + y_offset_increment * (i - 1)
            text_position = self.calculate_text_position(image_width, image_height, text_width, text_height, text_alignment, x_offset, y_offset, margin)
            processed_image= self.process_single_image(selected_image, text, font, font_color, rotation_increment * i, x_offset, y_offset, text_alignment, line_spacing, text_position, rotation_anchor_x, rotation_anchor_y, background_color, kerning)
            images.append(processed_image)
        return images
    
    def process_single_image(self, image, text, font, font_color, rotation_angle, x_offset, y_offset, text_alignment, line_spacing, text_position, rotation_anchor_x, rotation_anchor_y, background_color, kerning):        
        orig_width, orig_height = image.size
        canvas_size = int(max(orig_width, orig_height) * 1.5)
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        text_block_width, text_block_height = self.calculate_text_block_size(draw, text, font, line_spacing, kerning)
        text_x, text_y = text_position
        text_x += (canvas_size - orig_width) / 2 + x_offset
        text_y += (canvas_size - orig_height) / 2 + y_offset
        text_center_x = text_x + text_block_width / 2
        text_center_y = text_y + text_block_height / 2
        overlay = Image.new('RGBA', (int(text_block_width), int(text_block_height)), (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        self.draw_text_on_overlay(draw_overlay, text, font, font_color, line_spacing, kerning)
        canvas.paste(overlay, (int(text_x), int(text_y)), overlay)
        anchor = (text_center_x + rotation_anchor_x, text_center_y + rotation_anchor_y)
        rotated_canvas = canvas.rotate(rotation_angle, center=anchor, expand=0)
        new_canvas = Image.new('RGBA', rotated_canvas.size, (0, 0, 0, 0))
        new_canvas.paste(image, (int((rotated_canvas.size[0] - orig_width) / 2), int((rotated_canvas.size[1] - orig_height) / 2)))
        new_canvas.paste(rotated_canvas, (0, 0), rotated_canvas)
        cropped_image = new_canvas.crop(((canvas_size - orig_width) / 2, (canvas_size - orig_height) / 2, (canvas_size + orig_width) / 2, (canvas_size + orig_height) / 2))
        return self.process_image_for_output(cropped_image)

    def draw_text_on_overlay(self, draw_overlay, text, font, font_color, line_spacing, kerning):
        y_text_overlay = 0
        x_text_overlay = 0
        for line in text.split('\n'):
            for char in line:
                draw_overlay.text((x_text_overlay, y_text_overlay), char, font=font, fill=font_color)
                char_width = draw_overlay.textlength(char, font=font)
                x_text_overlay += char_width + kerning
            x_text_overlay = 0
            y_text_overlay += int(font.size) + line_spacing