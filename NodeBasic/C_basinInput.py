import comfy
from ..main_unit import *


class basicIn_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "pass_seed"
    CATEGORY = "Apt_Preset/View_IO"

    def pass_seed(self, seed):
        return (seed,)

class basicIn_float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert_to_float"
    CATEGORY = "Apt_Preset/View_IO"

    def convert_to_float(self, input):
        try:
            return (float(input),)
        except (ValueError, TypeError):
            raise ValueError("请输入有效的数字")

class basicIn_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": ( comfy.samplers.KSampler.SAMPLERS, ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "pass_sampler"
    CATEGORY = "Apt_Preset/View_IO"

    def pass_sampler(self, sampler):
        return (sampler,)

class basicIn_Scheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "pass_scheduler"
    CATEGORY = "Apt_Preset/View_IO"

    def pass_scheduler(self, scheduler):
        return (scheduler,)


class basicIn_string:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pass_text"
    CATEGORY = "Apt_Preset/View_IO"

    def pass_text(self, input_text):
        return (input_text,)




class basicIn_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", { "min": 0, "max": 16384,  "step": 1,})
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert_to_int"
    CATEGORY = "Apt_Preset/View_IO"

    def convert_to_int(self, input):
        try:
            return (int(input),)
        except (ValueError, TypeError):
            return (None,)






class basicIn_color:
 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 定义颜色预设映射
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
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "lightgray": (211, 211, 211),
            "darkgray": (169, 169, 169),
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
        
        # 准备预设选项，添加自定义选项在最前面
        preset_options = ["custom"] + list(color_mapping.keys())
        # 预设选项的显示标签
        preset_labels = ["自定义颜色"] + [name.capitalize() for name in color_mapping.keys()]
        
        return {
            "required": {
                "preset": (
                    preset_options, 
                    {"default": "custom", "label": preset_labels}
                ),
                "hex_str": ("STRING", {"default": "#FFFFFF"}),  # 默认为白色
            }
        }

    RETURN_TYPES = ("COLOR",)  # 输出数据类型为"COLOR"
    RETURN_NAMES = ("color",)
    FUNCTION = "output_color"
    CATEGORY = "Apt_Preset/View_IO"

    def output_color(self, preset, hex_str):
        # 定义颜色预设映射
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
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "lightgray": (211, 211, 211),
            "darkgray": (169, 169, 169),
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
        
        # 根据预设选择颜色
        if preset != "custom":
            # 将RGB元组转换为十六进制字符串
            r, g, b = color_mapping[preset]
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
        
        return (hex_str,)
    
























