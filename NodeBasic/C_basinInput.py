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
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pass_text"
    CATEGORY = "Apt_Preset/View_IO"

    def pass_text(self, input):
        return (input,)




class basicIn_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"step": 1, "default": 0})
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

