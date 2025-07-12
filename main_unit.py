
#region------------------导入库----------------------------------------------------#
import numpy as np
import torch
import os
import comfy.utils
from comfy.cli_args import args
import comfy.samplers
import json
import csv
import re
import math
from comfy_extras.chainner_models import model_loading
import folder_paths
from comfy import model_management
from math import ceil
import torchvision.transforms.functional as TF
from typing import cast
import numbers
from io import BytesIO
import matplotlib.pyplot as plt
from typing import Any, Callable, Mapping
import logging
from PIL import Image, ImageChops, ImageFilter
from nodes import PreviewImage, SaveImage
import numpy.typing as npt
from PIL import Image, ImageDraw,  ImageFilter,  ImageChops, ImageDraw, ImageFont
import hashlib
import torch.nn.functional as F
from scipy.fft import fft
from dataclasses import dataclass
from comfy.utils import ProgressBar
from comfy.utils import common_upscale
import node_helpers








#endregion------------------导入库----------------------------------------------------#



#region-------------------路径、数据------------------------------------------------------------------------------#




logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")
any_type = AnyType("*")
anyType = AnyType("*")
anytype = AnyType("*")

MAX_RESOLUTION = 88888
CLIP_TYPE = ["sdxl", "sd3", "flux", "hunyuan_video", "stable_diffusion", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan"]



available_ckpt = folder_paths.get_filename_list("checkpoints")
available_unets = list(set(folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf")))
available_clips = list(set(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")))
available_loras = folder_paths.get_filename_list("loras")
available_vaes = folder_paths.get_filename_list("vae")
available_embeddings = folder_paths.get_filename_list("embeddings")
available_style_models = folder_paths.get_filename_list("style_models")
available_clip_visions = folder_paths.get_filename_list("clip_vision")
available_controlnet=folder_paths.get_filename_list("controlnet")
available_samplers = comfy.samplers.KSampler.SAMPLERS
available_schedulers = comfy.samplers.KSampler.SCHEDULERS






def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    out = model_loading.load_state_dict(sd).eval()
    return out




#endregion-----------路径、数据------------------------------------------------------------------------------#



#region-------------------context全局定义------------------------------------------------------------------------------#




_all_contextput_output_data = {
    "context": ("context", "RUN_CONTEXT", "context"),
    "model": ("model", "MODEL", "model"),
    "positive": ("positive", "CONDITIONING", "positive"),
    "negative": ("negative", "CONDITIONING", "negative"),
    "latent": ("latent", "LATENT", "latent"),
    "vae": ("vae", "VAE", "vae"),
    "clip": ("clip","CLIP", "clip"),
    "images": ("images", "IMAGE", "images"),
    "mask": ("mask", "MASK", "mask"),
    "guidance": ("guidance", "FLOAT", "guidance"),
    "steps": ("steps", "INT", "steps"),
    "cfg": ("cfg", "FLOAT", "cfg"),
    "sampler": ("sampler", available_samplers, "sampler"),
    "scheduler": ("scheduler", available_schedulers, "scheduler"),

    "clip1": ("clip1", available_clips, "clip1"),
    "clip2": ("clip2", available_clips, "clip2"),
    "clip3": ("clip3", available_clips, "clip3"),
    "clip4": ("clip4", available_clips, "clip3"),
    "unet_name": ("unet_name", available_unets, "unet_name"),
    "ckpt_name": ("ckpt_name", available_ckpt, "ckpt_name"),
    "pos": ("pos", "STRING", "pos"),
    "neg": ("neg", "STRING", "neg"),
    "width": ("width", "INT","width" ),
    "height": ("height", "INT","height"),
    "batch": ("batch", "INT","batch"),
    "data": ("data", ANY_TYPE, "data"),
    "data1": ("data1", ANY_TYPE, "data1"),
    "data2": ("data2", ANY_TYPE, "data2"),
    "data3": ("data3", ANY_TYPE, "data3"),
    "data4": ("data4", ANY_TYPE, "data4"),
    "data5": ("data5", ANY_TYPE, "data5"),
    "data6": ("data6", ANY_TYPE, "data6"),
    "data7": ("data7", ANY_TYPE, "data7"),
    "data8": ("data8", ANY_TYPE, "data8"),

}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler", "scheduler","clip1", "clip2", "clip3","clip4", "unet_name", "ckpt_name"]


def _create_context_data(input_list=None):
    if input_list is None:
        input_list = _all_contextput_output_data.keys()
    list_ctx_return_types = []
    list_ctx_return_names = []
    ctx_optional_inputs = {}
    for inp in input_list:
        data = _all_contextput_output_data[inp]
        list_ctx_return_types.append(data[1])
        list_ctx_return_names.append(data[2])
        ctx_optional_inputs[data[0]] = tuple([data[1]] + (
            [{"forceInput": True}] if data[1] in force_input_types or data[0] in force_input_names else []
        ))
    ctx_return_types = tuple(list_ctx_return_types)
    ctx_return_names = tuple(list_ctx_return_names)
    return (ctx_optional_inputs, ctx_return_types, ctx_return_names)

ALL_CTX_OPTIONAL_INPUTS, ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES = _create_context_data()



_original_ctx_inputs_list = ["context", "model", "positive", "negative", "latent", "vae", "clip", "images", "mask",]
ORIG_CTX_OPTIONAL_INPUTS, ORIG_CTX_RETURN_TYPES, ORIG_CTX_RETURN_NAMES = _create_context_data(_original_ctx_inputs_list)


_load_ctx_inputs_list = ["context", "clip1", "clip2", "clip3","clip4", "ckpt_name","unet_name", "pos","neg" ,"width", "height", "batch"]
LOAD_CTX_OPTIONAL_INPUTS, LOAD_CTX_RETURN_TYPES, LOAD_CTX_RETURN_NAMES = _create_context_data(_load_ctx_inputs_list)



def new_context(context, **kwargs):
    context = context if context is not None else None
    new_ctx = {}
    for key in _all_contextput_output_data:
        if key == "context":
            continue
        v = kwargs[key] if key in kwargs else None
        new_ctx[key] = v if v is not None else (
            context[key] if context is not None and key in context else None
        )
    return new_ctx


def merge_new_context(*args):
    new_ctx = {}
    for key in _all_contextput_output_data:
        if key == "base_ctx":
            continue
        v = None
        for ctx in reversed(args):
            v = ctx[key] if not is_context_empty(ctx) and key in ctx else None
            if v is not None:
                break
        new_ctx[key] = v
    return new_ctx


def get_context_return_tuple(ctx, inputs_list=None):
    if inputs_list is None:
        inputs_list = _all_contextput_output_data.keys()
    tup_list = [ctx]
    for key in inputs_list:
        if key == "context":
            continue
        tup_list.append(ctx[key] if ctx is not None and key in ctx else None)
    return tuple(tup_list)


def get_orig_context_return_tuple(ctx):
    return get_context_return_tuple(ctx, _original_ctx_inputs_list)

def get_load_context_return_tuple(ctx):
    return get_context_return_tuple(ctx, _load_ctx_inputs_list)



def is_context_empty(ctx):
    return not ctx or all(v is None for v in ctx.values())



#endregion



# region-----------------缓动函数------------------------------------------------------
def easeInBack(t):
    s = 1.70158
    return t * t * ((s + 1) * t - s)

def easeInOutSinSquared(t):
    if t < 0.5:
        return 0.5 * (1 - math.cos(t * 2 * math.pi))
    else:
        return 0.5 * (1 + math.cos((t - 0.5) * 2 * math.pi))

def easeOutBack(t):
    s = 1.70158
    return ((t - 1) * t * ((s + 1) * t + s)) + 1

def easeInOutBack(t):
    s = 1.70158 * 1.525
    if t < 0.5:
        return (t * t * (t * (s + 1) - s)) * 2
    return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2

# Elastic easing functions
def easeInElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return -(math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))

def easeOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

def easeInOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3 * 1.5
    s = p / 4
    t = t * 2
    if t < 1:
        return -0.5 * (
            math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        )
    return (
        0.5 * math.pow(2, -10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        + 1
    )

# Bounce easing functions
def easeInBounce(t):
    return 1 - easeOutBounce(1 - t)

def easeOutBounce(t):
    if t < (1 / 2.75):
        return 7.5625 * t * t
    elif t < (2 / 2.75):
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < (2.5 / 2.75):
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

def easeInOutBounce(t):
    if t < 0.5:
        return easeInBounce(t * 2) * 0.5
    return easeOutBounce(t * 2 - 1) * 0.5 + 0.5

# Quart easing functions
def easeInQuart(t):
    return t * t * t * t

def easeOutQuart(t):
    t -= 1
    return -(t**2 * t * t - 1)

def easeInOutQuart(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t * t
    t -= 2
    return -0.5 * (t**2 * t * t - 2)

# Cubic easing functions
def easeInCubic(t):
    return t * t * t

def easeOutCubic(t):
    t -= 1
    return t**2 * t + 1

def easeInOutCubic(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t
    t -= 2
    return 0.5 * (t**2 * t + 2)

# Circ easing functions
def easeInCirc(t):
    return -(math.sqrt(1 - t * t) - 1)

def easeOutCirc(t):
    t -= 1
    return math.sqrt(1 - t**2)

def easeInOutCirc(t):
    t *= 2
    if t < 1:
        return -0.5 * (math.sqrt(1 - t**2) - 1)
    t -= 2
    return 0.5 * (math.sqrt(1 - t**2) + 1)

# Sine easing functions
def easeInSine(t):
    return -math.cos(t * (math.pi / 2)) + 1

def easeOutSine(t):
    return math.sin(t * (math.pi / 2))

def easeInOutSine(t):
    return -0.5 * (math.cos(math.pi * t) - 1)

def easeLinear(t):
    return t

easing_functions = {
    "Linear": easeLinear,
    "Sine_In": easeInSine,
    "Sine_Out": easeOutSine,
    "Sine_InOut": easeInOutSine,
    "Sin_Squared": easeInOutSinSquared,
    "Quart_In": easeInQuart,
    "Quart_Out": easeOutQuart,
    "Quart_InOut": easeInOutQuart,
    "Cubic_In": easeInCubic,
    "Cubic_Out": easeOutCubic,
    "Cubic_InOut": easeInOutCubic,
    "Circ_In": easeInCirc,
    "Circ_Out": easeOutCirc,
    "Circ_InOut": easeInOutCirc,
    "Back_In": easeInBack,
    "Back_Out": easeOutBack,
    "Back_InOut": easeInOutBack,
    "Elastic_In": easeInElastic,
    "Elastic_Out": easeOutElastic,
    "Elastic_InOut": easeInOutElastic,
    "Bounce_In": easeInBounce,
    "Bounce_Out": easeOutBounce,
    "Bounce_InOut": easeInOutBounce,
}

EASING_TYPES = list(easing_functions.keys())

def apply_easing(value, easing_type):
    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)
    
    raise ValueError(f"Unknown easing type: {easing_type}")



# endregion---------------插值算法------------------------



#region  ----------------math 算法-------------------------------------------------------------#


DEFAULT_FLOAT = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001,})

FLOAT_UNARY_OPERATIONS: Mapping[str, Callable[[float], float]] = {
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Abs": lambda a: abs(a),
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Sqrt": lambda a: math.sqrt(a),
    "Exp": lambda a: math.exp(a),
    "Ln": lambda a: math.log(a),
    "Log10": lambda a: math.log10(a),
    "Log2": lambda a: math.log2(a),
    "Sin": lambda a: math.sin(a),
    "Cos": lambda a: math.cos(a),
    "Tan": lambda a: math.tan(a),
    "Asin": lambda a: math.asin(a),
    "Acos": lambda a: math.acos(a),
    "Atan": lambda a: math.atan(a),
    "Sinh": lambda a: math.sinh(a),
    "Cosh": lambda a: math.cosh(a),
    "Tanh": lambda a: math.tanh(a),
    "Asinh": lambda a: math.asinh(a),
    "Acosh": lambda a: math.acosh(a),
    "Atanh": lambda a: math.atanh(a),
    "Round": lambda a: round(a),
    "Floor": lambda a: math.floor(a),
    "Ceil": lambda a: math.ceil(a),
    "Trunc": lambda a: math.trunc(a),
    "Erf": lambda a: math.erf(a),
    "Erfc": lambda a: math.erfc(a),
    "Gamma": lambda a: math.gamma(a),
    "Radians": lambda a: math.radians(a),
    "Degrees": lambda a: math.degrees(a),
}

FLOAT_UNARY_CONDITIONS: Mapping[str, Callable[[float], bool]] = {
    "IsZero": lambda a: a == 0.0000,
    "IsPositive": lambda a: a > 0.000,
    "IsNegative": lambda a: a < 0.000,
    "IsNonZero": lambda a: a != 0.000,
    "IsPositiveInfinity": lambda a: math.isinf(a) and a > 0.000,
    "IsNegativeInfinity": lambda a: math.isinf(a) and a < 0.000,
    "IsNaN": lambda a: math.isnan(a),
    "IsFinite": lambda a: math.isfinite(a),
    "IsInfinite": lambda a: math.isinf(a),
    "IsEven": lambda a: a % 2 == 0.000,
    "IsOdd": lambda a: a % 2 != 0.000,
}

FLOAT_BINARY_OPERATIONS: Mapping[str, Callable[[float, float], float]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "FloorDiv": lambda a, b: a // b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
    "Log": lambda a, b: math.log(a, b),
    "Atan2": lambda a, b: math.atan2(a, b),
}

FLOAT_BINARY_CONDITIONS: Mapping[str, Callable[[float, float], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Gte": lambda a, b: a >= b,
    "Lt": lambda a, b: a < b,
    "Lte": lambda a, b: a <= b,
}


DEFAULT_INT = ("INT", {"default": 0})

INT_UNARY_OPERATIONS: Mapping[str, Callable[[int], int]] = {
    "Abs": lambda a: abs(a),
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Not": lambda a: ~a,
    "Factorial": lambda a: math.factorial(a),
}

INT_UNARY_CONDITIONS: Mapping[str, Callable[[int], bool]] = {
    "IsZero": lambda a: a == 0,
    "IsNonZero": lambda a: a != 0,
    "IsPositive": lambda a: a > 0,
    "IsNegative": lambda a: a < 0,
    "IsEven": lambda a: a % 2 == 0,
    "IsOdd": lambda a: a % 2 == 1,
}

INT_BINARY_OPERATIONS: Mapping[str, Callable[[int, int], int]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a // b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "And": lambda a, b: a & b,
    "Nand": lambda a, b: ~a & b,
    "Or": lambda a, b: a | b,
    "Nor": lambda a, b: ~a & b,
    "Xor": lambda a, b: a ^ b,
    "Xnor": lambda a, b: ~a ^ b,
    "Shl": lambda a, b: a << b,
    "Shr": lambda a, b: a >> b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
}

INT_BINARY_CONDITIONS: Mapping[str, Callable[[int, int], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Lt": lambda a, b: a < b,
    "Geq": lambda a, b: a >= b,
    "Leq": lambda a, b: a <= b,
}

# endregion-------------------------math 算法--------------------------------------------------#



#region------------------调度数据-AD--sch prompt  def----------------------------------------------------#

def adapt_to_batch(value, frame):    # 适应批处理数据处理，value对应当前的帧数
    if not hasattr(value, '__iter__'):
        value = [value]
    if frame < len(value):
        value = value[:frame]
    elif frame > len(value):
        last_value = value[-1]
        value = value + [last_value] * (frame - len(value))
    return value




DefaultPromp = """0: a girl @Sine_In@
7: a boy
15: a dog
"""



DefaultValue = """0:0.5 @Sine_In@
30:1
60:0.5
90:1
120:0.5
"""



def lerp_tensors(tensor_from: torch.Tensor, tensor_to: torch.Tensor, weight: float):
    return tensor_from * (1.0 - weight) + tensor_to * weight


@dataclass
class PromptKeyframe:
    index: int
    prompt: str
    interp_method: str = "linear"


class ValueKeyframe:
    def __init__(self, index, value, interp_method):
        self.index = index
        self.value = value
        self.interp_method = interp_method


def parse_prompt_schedule(text: str, easing_type="Linear"):
    keyframes = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        idx_part, prompt_part = line.split(':', 1)
        idx = int(idx_part.strip())
        prompt = prompt_part.strip()

        interp_method = easing_type
        match = re.search(r'@([a-zA-Z0-9_ ]+)@', prompt)
        if match:
            custom_ease = match.group(1).strip()
            if custom_ease in easing_functions:
                interp_method = custom_ease
            prompt = re.sub(r'\s*@([a-zA-Z0-9_ ]+)@\s*', ' ', prompt).strip()  # 移除标签

        if not prompt:
            continue
        keyframes.append(PromptKeyframe(index=idx, prompt=prompt, interp_method=interp_method))

    return sorted(keyframes, key=lambda x: x.index)


def build_conditioning(keyframes, clip, max_length, f_text="", b_text=""):
    if len(keyframes) == 0:
        raise ValueError("No valid keyframes found.")

    if max_length <= 0:
        max_length = keyframes[-1].index + 1

    conds = [None] * max_length
    pooleds = [None] * max_length

    prev_idx, prev_cond, prev_pooled = None, None, None

    pbar = ProgressBar(max_length)

    all_weights = [None] * max_length  # 存储所有帧的权重用于绘图

    for i, kf in enumerate(keyframes):
        curr_idx, curr_prompt, curr_method = kf.index, kf.prompt, kf.interp_method
        if curr_idx >= max_length:
            break

        # 添加前后缀
        prefix = f"{f_text}, " if f_text else ""
        suffix = f", {b_text}" if b_text else ""
        full_prompt = f"{prefix}{curr_prompt}{suffix}".strip()

        # tokenize and encode prompt
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conds[curr_idx] = cond
        pooleds[curr_idx] = pooled
        pbar.update(1)

        # 插值逻辑：当前帧和上一帧之间
        if prev_idx is not None and prev_idx < curr_idx:
            diff_len = curr_idx - prev_idx
            weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
            easing_weights = [apply_easing(w.item(), curr_method) for w in weights]

            print(f"Frame {prev_idx} to {curr_idx}, method: {curr_method}")
            print("Raw weights:     ", [round(w.item(), 3) for w in weights])
            print("Easing weights:  ", [round(w, 3) for w in easing_weights])

            for j, w in enumerate(easing_weights):
                idx = prev_idx + j + 1
                if idx >= max_length:
                    break
                conds[idx] = lerp_tensors(prev_cond, cond, w)
                pooleds[idx] = pooleds[idx - 1]
                all_weights[idx] = w  # 记录当前帧的权重

        # 更新 prev 变量
        prev_idx, prev_cond, prev_pooled = curr_idx, cond, pooled
        all_weights[curr_idx] = 1.0  # 当前帧权重设为1

    # 填充开头和结尾缺失帧
    first_valid = next((i for i in range(max_length) if conds[i] is not None), None)
    last_valid = None
    for i in range(max_length):
        if conds[i] is not None:
            last_valid = i
        elif last_valid is not None:
            conds[i] = conds[last_valid]
            pooleds[i] = pooleds[last_valid]
            all_weights[i] = all_weights[last_valid]

    if first_valid is not None:
        for i in range(first_valid):
            conds[i] = conds[first_valid]
            pooleds[i] = pooleds[first_valid]
            all_weights[i] = all_weights[first_valid]

    final_cond = torch.cat(conds, dim=0)
    final_pooled_dict = {"pooled_output": torch.cat(pooleds, dim=0)}

    return [[final_cond, final_pooled_dict]]


def generate_frame_weight_curve_image(keyframes, max_length):
    current_weights = [None] * max_length  # 当前帧影响权重
    previous_weights = [None] * max_length  # 上一帧影响权重

    prev_idx = None

    for kf in keyframes:
        curr_idx = kf.index
        if prev_idx is not None and prev_idx < curr_idx:
            diff_len = curr_idx - prev_idx
            weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
            easing_weights = [apply_easing(w.item(), kf.interp_method) for w in weights]

            for j, w in enumerate(easing_weights):
                idx = prev_idx + j + 1
                if idx >= max_length:
                    break
                current_weights[idx] = w           # 当前帧权重 (curr)
                previous_weights[idx] = 1.0 - w    # 上一帧权重 (prev)

        prev_idx = curr_idx
        current_weights[curr_idx] = 1.0
        previous_weights[curr_idx] = 0.0

    # 补全缺失帧（保持连续性）
    def fill_weights(weights_list):
        last_valid = None
        for i in range(len(weights_list)):
            if weights_list[i] is not None:
                last_valid = i
            elif last_valid is not None:
                weights_list[i] = weights_list[last_valid]

    fill_weights(current_weights)
    fill_weights(previous_weights)

    # 如果开头没有值，则复制第一个有效值
    first_valid = next((i for i, w in enumerate(current_weights) if w is not None), None)
    if first_valid is not None:
        for i in range(first_valid):
            current_weights[i] = current_weights[first_valid]
            previous_weights[i] = previous_weights[first_valid]

    # 转换为浮点数
    y_current = [w if w is not None else 0.0 for w in current_weights]
    y_previous = [w if w is not None else 0.0 for w in previous_weights]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_previous)), y_previous, marker='x', linestyle='--', color='blue', markersize=2, label='Prev Frame Weight')
    plt.plot(range(len(y_current)), y_current, marker='o', linestyle='-', color='green', markersize=2, label='Curr Frame Weight')
    plt.title("Interpolation Weights per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image = Image.open(buffer)
    return pil2tensor(image)


def generate_value_curve_image_with_data(values_seq, max_length, frame_methods=None):
    y = [v if v is not None else np.nan for v in values_seq]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y)), y, marker='o', linestyle='-', markersize=3, label="Value Curve")

    # 如果有插值区间信息，则绘制缓动标记
    if frame_methods:
        for start, end, method in frame_methods:
            plt.axvspan(start, end, alpha=0.1, color='gray', label=f"{method} Interpolation" if start == 0 else "")

    plt.title("Interpolated Value Curve per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(loc="upper left")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image = Image.open(buffer)
    return pil2tensor(image)



#endregion---------------调度数据----------------------------------------------------#



#region------------------风格处理----------------------------------------------------#
def style_list():   # 读取风格csv文件
    current_dir = os.path.dirname(__file__)
    csv_dir = os.path.join(current_dir, "csv")
    file_path = os.path.join(csv_dir, "styles.csv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data_list = [row for row in reader]
    my_path = os.path.join(csv_dir, "my_styles.csv")
    if os.path.exists(my_path):
        with open(my_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            my_data_list = [row for row in reader]
        data_list = my_data_list + data_list
    
    card_list = []
    for i in data_list:
        card_list += [i[0]]
    return (card_list, data_list)




def add_style_to_subject(style, positive, negative):
    if style != "None":
        style_info = style_list()
        style_index = style_info[0].index(style)
        style_text = style_info[1][style_index][1]

        if "{prompt}" in style_text:
            positive = style_text.replace("{prompt}", positive)
        else:
            positive += f", {style_text}"

        if len(style_info[1][style_index]) > 2:
            negative += f", {style_info[1][style_index][2]}"

    return positive, negative

#endregion------------------风格处理----------------------------------------------------#




#region------------------类型转换----------------------------------------------------#  





def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return np_array





def try_cast(x, dst_type: str): #按值的类型返回
    result = x
    if dst_type == "STRING":
        result = str(x)
    elif dst_type == "INT":
        result = int(x)
    elif dst_type == "FLOAT" or dst_type == "NUMBER":
        result = float(x)
    elif dst_type == "BOOLEAN":
        if isinstance(x, numbers.Number):
            if x > 0:
                result = True
            else:
                result = False
        elif isinstance(x, str):
            try:
                x = float(x)
                if x > 0:
                    result = True
                else:
                    result = False
            except:
                result = bool(x)
        else:
            result = bool(x)
    return result


def pil2tensor(image):  #多维度的图像也可以
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def tensor_to_pillow(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pillow_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)





def list_pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)
    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)


def list_tensor2pil(tensor: torch.Tensor) -> list[Image.Image]: 
    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 1:  # 处理 [H, W, 1] 形状的张量
                return Image.fromarray(np_array.squeeze(-1), mode="L")
            elif np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")
    return handle_batch(tensor, single_tensor2pil)


def handle_batch(       #输入张量进行批量处理，将指定的转换函数应用到张量的每个元素上，并返回处理结果的列表
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]






#endregion------------------类型转换----------------------------------------------------#



#region------------------图像处理----------------------------------------------------#


def latentrepeat(samples, amount):
    try:
        # 复制原始样本字典，避免修改原始数据
        s = samples.copy()
        s_in = samples["samples"]

        # 动态生成重复维度
        num_dims = s_in.dim()
        if num_dims < 4:
            raise ValueError(f"输入张量的维度为 {num_dims}，期望至少为 4 维。")
        repeat_dims = (amount,) + (1,) * (num_dims - 1)
        s["samples"] = s_in.repeat(repeat_dims)

        # 处理 noise_mask
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                repeat_factor = math.ceil(s_in.shape[0] / masks.shape[0])
                masks = masks.repeat(repeat_factor, 1, 1, 1)[:s_in.shape[0]]
            # 动态生成 noise_mask 的重复维度
            mask_num_dims = masks.dim()
            if mask_num_dims < 4:
                raise ValueError(f"noise_mask 张量的维度为 {mask_num_dims}，期望至少为 4 维。")
            mask_repeat_dims = (amount,) + (1,) * (mask_num_dims - 1)
            s["noise_mask"] = masks.repeat(mask_repeat_dims)

        # 处理 batch_index
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            additional_indices = [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
            s["batch_index"] += additional_indices

        return (s,)
    except Exception as e:
        print(f"在 latentrepeat 函数中发生错误: {str(e)}")
        return (samples,)  # 发生错误时返回原始样本


def read_ratios():           # 读取ratios.json文件,生成图像的尺寸
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(current_dir, 'web')
    file_path = os.path.join(p, 'ratios.json')
    
    # 显式指定使用 utf-8 编码打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    ratio_sizes = list(data['ratios'].keys())
    ratio_dict = data['ratios']
    return ratio_sizes, ratio_dict


def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):  # 预览、保存图片

    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']



def append_helper(t, mask, c, set_area_to_bounds, strength): #没有用到？？
    if mask is not None:  
        n = [t[0], t[1].copy()]
        _, h, w = mask.shape
        n[1]['mask'] = mask
        n[1]['set_area_to_bounds'] = set_area_to_bounds
        n[1]['mask_strength'] = strength
        c.append(n)



def rescale(samples, width, height, algorithm):
    if algorithm == "nearest":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="nearest")
    elif algorithm == "bilinear":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")
    elif algorithm == "bicubic":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic")
    elif algorithm == "bislerp":
        return comfy.utils.bislerp(samples, width, height)
    return samples


def upscale_with_model(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s        


def image_upscale(image, upscale_method, scale_by):
    samples = image.movedim(-1,1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return (s,)


def upscale(image, upscale_method, width,height):
    samples = image.movedim(-1,1)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return (s,)


def image_2dtransform(             # 2D图像变换
        image,
        x,
        y,
        zoom,
        angle,
        shear=0,
        border_handling="edge",
    ):
        x = int(x)
        y = int(y)
        angle = int(angle)

        if image.size(0) == 0:
            return (torch.zeros(0),)
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        # - Calculate diagonal of the original image
        diagonal = math.sqrt(frame_width**2 + frame_height**2)
        max_padding = math.ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [max(0, pw + x), max(0, ph + y), max(0, pw - x), max(0, ph - y)]

        img = tensor2pil(image)

        img = TF.pad(
            img,  # transformed_frame,
            padding=padding,
            padding_mode=border_handling,
        )

        img = cast(
            Image.Image,
            TF.affine(img, angle=angle, scale=zoom, translate=[x, y], shear=shear),
        )

        left = abs(padding[0])
        upper = abs(padding[1])
        right = img.width - abs(padding[2])
        bottom = img.height - abs(padding[3])

        img = img.crop((left, upper, right, bottom))

        return pil2tensor(img)



def get_image_resize(image, target_image=None):  # 尺寸参考
    B, H, W, C = image.shape
    upscale_method = "lanczos"
    crop = "center"
    if target_image is None:
        width = W - (W % 8)
        height = H - (H % 8)
        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)
        return image
    _, height, width, _ = target_image.shape

    image = image.movedim(-1, 1)
    image = common_upscale(image, width, height, upscale_method, crop)
    image = image.movedim(1, -1)
    return image




#endregion------------------图像处理----------------------------------------------------#



#region------------------颜色处理----------------------------------------------------#


def hex_to_float(color):
    if not isinstance(color, str):
        raise ValueError("Color must be a hex string")
    color = color.strip("#")
    return int(color, 16) / 255.0


def hex_to_rgba(hex_color, alpha): 
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, int(alpha * 255 / 100))


def hex_to_rgb(hex_color):    #只能处理 6 位十六进制颜色字符串作为输入，例如 "#FF0000"
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

def hex_to_rgb_tuple(color):   # 处理 3 位和 6 位十六进制颜色字符串作为输入，例如 "#F00" 和 "#FF0000"
    if isinstance(color, list) and len(color) == 3:
        return tuple(int(c * 255) for c in color)
    elif isinstance(color, str):
        color = color.lstrip('#')
        if len(color) == 6:
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        elif len(color) == 3:
            return tuple(int(c * 2, 16) for c in color)
    raise ValueError(f"不支持的颜色格式: {color}")


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


COLORS = ["custom", "white", "black", "red", "green", "blue", "yellow",
        "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
        "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
        "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
        "violet", "coral", "indigo"]


STYLES = ["Accent","afmhot","autumn","binary","Blues","bone","BrBG","brg",
    "BuGn","BuPu","bwr","cividis","CMRmap","cool","coolwarm","copper","cubehelix","Dark2","flag",
    "gist_earth","gist_gray","gist_heat","gist_rainbow","gist_stern","gist_yarg","GnBu","gnuplot","gnuplot2","gray","Greens",
    "Greys","hot","hsv","inferno","jet","magma","nipy_spectral","ocean","Oranges","OrRd",
    "Paired","Pastel1","Pastel2","pink","PiYG","plasma","PRGn","prism","PuBu","PuBuGn",
    "PuOr","PuRd","Purples","rainbow","RdBu","RdGy","RdPu","RdYlBu","RdYlGn","Reds","seismic",
    "Set1","Set2","Set3","Spectral","spring","summer","tab10","tab20","tab20b","tab20c","terrain",
    "turbo","twilight","twilight_shifted","viridis","winter","Wistia","YlGn","YlGnBu","YlOrBr","YlOrRd"]


# endregion------------------颜色处理----------------------------------------------------#



#region------------------文本处理----------------------------------------------------#
def replace_text(text, target, replace):
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
    
    targets = split_with_quotes(target)
    exchanges = split_with_quotes(replace)
    
    word_map = {}
    for target_item, exchange_item in zip(targets, exchanges):
        target_clean = target_item.strip('"').strip().lower()
        exchange_clean = exchange_item.strip('"').strip()
        word_map[target_clean] = exchange_clean

    sorted_targets = sorted(word_map.keys(), key=len, reverse=True)
    result = text
    
    for target_item in sorted_targets:
        if ' ' in target_item:
            pattern = re.escape(target_item)
        else:
            pattern = r'\b' + re.escape(target_item) + r'\b'
        
        result = re.sub(pattern, word_map[target_item], result, flags=re.IGNORECASE)
    return result  # 直接返回字符串

def clean_prompt(text, words_to_remove):
    remove_words = [word.strip().lower() for word in words_to_remove.split(',')]
    words = re.findall(r'\b\w+\b|[^\w\s]', text)

    cleaned_words = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in remove_words:
            continue
        cleaned_words.append(word)
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text  # 直接返回字符串
#endregion------------------文本处理----------------------------------------------------#



#region------------------图层混合----------------------------------------------------#

blend_methods = ["None", "add", "subtract", "multiply", "screen", "overlay", "difference", "hard_light", "soft_light",
        "add_modulo", "blend", "darker", "duplicate", "lighter", "subtract_modulo"]


def blend_images(img_batch1, img_batch2, method, strength):
    # 检查输入是否为单张图像
    if isinstance(img_batch1, Image.Image):
        img_batch1 = [img_batch1]
    if isinstance(img_batch2, Image.Image):
        img_batch2 = [img_batch2]

    blended_batch = []
    for img1, img2 in zip(img_batch1, img_batch2):
        if method == "None":
            blended = None
        elif method == "add":
            blended = ImageChops.add(img1, img2)
        elif method == "subtract":
            blended = ImageChops.subtract(img1, img2)
        elif method == "multiply":
            blended = ImageChops.multiply(img1, img2)
        elif method == "screen":
            blended = ImageChops.screen(img1, img2)
        elif method == "overlay":
            blended = ImageChops.overlay(img1, img2)
        elif method == "difference":
            blended = ImageChops.difference(img1, img2)
        elif method == "hard_light":
            blended = ImageChops.hard_light(img1, img2)
        elif method == "soft_light":
            blended = ImageChops.soft_light(img1, img2)
        elif method == "add_modulo":
            blended = ImageChops.add_modulo(img1, img2)
        elif method == "blend":
            blended = ImageChops.blend(img1, img2, strength / 100.0)
        elif method == "darker":
            blended = ImageChops.darker(img1, img2)
        elif method == "duplicate":
            blended = ImageChops.duplicate(img1)
        elif method == "lighter":
            blended = ImageChops.lighter(img1, img2)
        elif method == "subtract_modulo":
            blended = ImageChops.subtract_modulo(img1, img2)
        else:
            raise ValueError("Unsupported blend method")
        if blended is not None:
            outimage = Image.blend(img1, blended, strength / 100.0)
        else:
            outimage = img1
        blended_batch.append(outimage)

    # 如果输入是单张图像，返回单张图像
    if len(blended_batch) == 1:
        return blended_batch[0]
    return blended_batch




#endregion------------------图像处理----------------------------------------------------#


#region------------------mask ----------------------------------------------------#
def mask_smoothness(mask,smoothness):
    mask=tensor2pil(mask)
    feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
    mask=pil2tensor(feathered_image)
    return (mask,)


def mask2image(input_mask_pil):
    input_mask_tensor = pil2tensor(input_mask_pil)
    result_tensor = input_mask_tensor.expand(-1, 3, -1, -1)
    return result_tensor



def image2mask(image_pil):
    # Convert image to grayscale
    image_pil = image_pil.convert("L")
    # Convert grayscale image to binary mask
    threshold = 128
    mask_array = np.array(image_pil) > threshold
    return Image.fromarray((mask_array * 255).astype(np.uint8))

def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


def tensorMask2cv2img(tensor) -> np.ndarray:   
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    return array



def mask_crop(image, mask):
    image_pil = tensor2pil(image)
    mask_pil = tensor2pil(mask)
    mask_array = np.array(mask_pil) > 0
    
    coords = np.where(mask_array)
    if coords[0].size == 0 or coords[1].size == 0:
        return (image, None)
    x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
    # 移除边界调整逻辑
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_pil.width)
    y1 = min(y1, image_pil.height)
    cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
    cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
    cropped_image_tensor = pil2tensor(cropped_image_pil)
    cropped_mask_tensor = pil2tensor(cropped_mask_pil)
    return (cropped_image_tensor, cropped_mask_tensor)



def img_and_mask_merge(images: torch.Tensor, channel_data: torch.Tensor = None):
    if channel_data is None:
        # 如果没有 mask，直接返回原图 PIL 列表
        return list_pil2tensor(images)

    merged_images = []
    for image in images:
        image = image.cpu().clone()
        if image.shape[2] < 4:
            image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)
        image[:, :, 3] = channel_data  
        merged_images.append(image)

    outimg = torch.stack(merged_images)
    return list_tensor2pil(outimg)  # 返回 PIL.Image 列表




def set_mask(samples, mask):
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return (s,)







#endregion------------------mask ----------------------------------------------------#



#region------------------图像绘制形状----------------------------------------------------#
#该函数用于在图像上绘制指定形状。支持多种形状，如圆形、半圆形、正方形等
DRAW_SHAPE_LIST = ["circle", "semicircle", "quarter_circle", "ellipse", "square", "triangle","cross", "star", "radial"]

def draw_shape(shape, size=(200, 200), offset=(0, 0), scale=1.0, rotation=0, bg_color=(255, 255, 255),
               shape_color=(0, 0, 0), opacity=1.0, blur_radius=0, base_image=None):
    width, height = size
    offset_x, offset_y = offset
    center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
    max_dim = min(width, height) * scale

    diagonal = int(math.sqrt(width ** 2 + height ** 2))
    img_tmp = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(img_tmp)

    tmp_center = diagonal // 2

    alpha = int(opacity * 255)
    shape_color = shape_color + (alpha,)

    if shape == 'circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'semicircle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=180, fill=shape_color)

    elif shape == 'quarter_circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=90, fill=shape_color)

    elif shape == 'ellipse':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 4, tmp_center + max_dim / 2, tmp_center + max_dim / 4)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'square':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.rectangle(bbox, fill=shape_color)

    elif shape == 'triangle':
        points = [
            (tmp_center, tmp_center - max_dim / 2),
            (tmp_center - max_dim / 2, tmp_center + max_dim / 2),
            (tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        ]
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'cross':
        vertical = [(tmp_center - max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center + max_dim / 2),
                    (tmp_center - max_dim / 6, tmp_center + max_dim / 2)]
        horizontal = [(tmp_center - max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center + max_dim / 6),
                      (tmp_center - max_dim / 2, tmp_center + max_dim / 6)]
        draw_tmp.polygon(vertical, fill=shape_color)
        draw_tmp.polygon(horizontal, fill=shape_color)

    elif shape == 'star':
        points = []
        for i in range(10):
            angle = i * 36 * math.pi / 180
            radius = max_dim / 2 if i % 2 == 0 else max_dim / 4
            points.append((tmp_center + radius * math.sin(angle), tmp_center - radius * math.cos(angle)))
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'radial':
        num_rays = 12
        for i in range(num_rays):
            angle = i * (360 / num_rays) * math.pi / 180
            x1 = tmp_center + max_dim / 4 * math.cos(angle)
            y1 = tmp_center + max_dim / 4 * math.sin(angle)
            x2 = tmp_center + max_dim / 2 * math.cos(angle)
            y2 = tmp_center + max_dim / 2 * math.sin(angle)
            draw_tmp.line([(x1, y1), (x2, y2)], fill=shape_color, width=int(max_dim / 20))

    img_tmp = img_tmp.rotate(rotation, resample=Image.BICUBIC, expand=True)
    if base_image is None:
        img = Image.new('RGBA', size, bg_color + (255,))
    else:
        img = base_image.copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

    paste_x = center_x - img_tmp.width // 2
    paste_y = center_y - img_tmp.height // 2

    img.alpha_composite(img_tmp, (paste_x, paste_y))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


#endregion------------------图像形状----------------------------------------------------#



#region------------------mask bridge----------------------------------------------------#

def tensor_to_hash(tensor):
    np_array = tensor.cpu().numpy()
    byte_data = np_array.tobytes()
    hash_value = hashlib.md5(byte_data).hexdigest()
    return hash_value


def create_temp_file(image):
    output_dir = folder_paths.get_temp_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path('material', output_dir)
    image = tensor2pil(image)
    image_file = f"{filename}_{counter:05}.png"
    image_path = os.path.join(full_output_folder, image_file)
    image.save(image_path, compress_level=4)
    return (image_path, [{"filename": image_file, "subfolder": subfolder, "type": "temp"}])



def generate_masked_black_image(image, mask):
    b, h, w = image.shape[0], image.shape[1], image.shape[2]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (b, h, w):
        mask = torch.zeros_like(image[:, :, :, 0])
    black_mask = torch.zeros_like(image)
    masked_image = image * (1 - mask.unsqueeze(-1)) + black_mask * mask.unsqueeze(-1)

    return {"ui": {"images": []}, "result": (masked_image,)}






#endregion------------------mask bridge----------------------------------------------------#




#region------------------lora----------------------------------------------------#

def apply_lora_stack( model, clip, lora_stack=None):
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


#endregion------------------lora----------------------------------------------------#





#region------------------vae----------------------------------------------------#
def decode(vae, samples):
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return (images, )


def encode(vae, pixels):
    t = vae.encode(pixels[:,:,:,:3])
    return ({"samples":t}, )

#endregion------------------vae----------------------------------------------------#


#region------------------latent----------------------------------------------------#

def set_latent_mask(latent, mask):
    newlatent = latent.copy()
    if mask is not None:
        newlatent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return newlatent

def reshape_latent_to(target_shape, latent, repeat_batch=True):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[-1], target_shape[-2], "bilinear", "center")
    if repeat_batch:
        return comfy.utils.repeat_to_batch_size(latent, target_shape[0])
    else:
        return latent


def latent_inter_polate(samples1, samples2, ratio):
    samples_out = samples1.copy()

    s1 = samples1["samples"]
    s2 = samples2["samples"]

    s2 = reshape_latent_to(s1.shape, s2)

    m1 = torch.linalg.vector_norm(s1, dim=(1))
    m2 = torch.linalg.vector_norm(s2, dim=(1))

    s1 = torch.nan_to_num(s1 / m1)
    s2 = torch.nan_to_num(s2 / m2)

    t = (s1 * ratio + s2 * (1.0 - ratio))
    mt = torch.linalg.vector_norm(t, dim=(1))
    st = torch.nan_to_num(t / mt)

    samples_out["samples"] = st * (m1 * ratio + m2 * (1.0 - ratio))
    return samples_out



def set_last_layer(clip, clipnum):
    clip = clip.clone()
    clip.clip_layer(clipnum)
    return clip

def create_latent_tensor(width, height, batch):
    width = width - (width % 8)
    height = height - (height % 8)
    latent = torch.zeros([batch, 4, height // 8, width // 8])
    if latent.shape[1] != 16:
        latent = latent.repeat(1, 16// 4, 1, 1)
    return latent




def condi_zero_out(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    return (c, )


#endregion------------------latent----------------------------------------------------#



#region------------------kontext---------------------------------------------------#

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def resize_to_multiple_of_8(x):
    if isinstance(x, torch.Tensor):
        # Tensor 分支
        _, h, w, _ = x.shape
        new_w = w // 8 * 8
        new_h = h // 8 * 8
        if new_w != w or new_h != h:
            x = comfy.utils.common_upscale(
                x.movedim(-1, 1), new_w, new_h, "bilinear", "center"
            ).movedim(1, -1)
        return x
    elif isinstance(x, Image.Image):
        # PIL 分支
        width, height = x.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        return x.resize((new_width, new_height), Image.LANCZOS)
    else:
        raise TypeError("Input must be PIL.Image or torch.Tensor")


def XXXXkontext_adjust_image_resolution(image, auto_adjust_image):

    if image is None:
        raise ValueError("Input image cannot be None.")
        
    image = resize_to_multiple_of_8(image)

    if auto_adjust_image:
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, target_width, target_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
        )

        scaled_image = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            target_width,
            target_height,
            "lanczos",
            "center"
        ).movedim(1, -1)

        image = scaled_image[:, :, :, :3]

    return image


def XXXkontext_latent_and_conditioning(context, vae, image, mask, prompt_weight, relevance, positive): #相关与无关
    if image is None:
        latent_data = context.get("latent", None)
        if latent_data is None or "samples" not in latent_data:
            raise ValueError("Context latent or samples not found when image is None")
        encoded_latent = latent_data["samples"]
    else:
        encoded_latent = vae.encode(image)

    # 确保 encoded_latent 是正确的形状
    if len(encoded_latent.shape) == 4:
        if encoded_latent.shape[0] > 1:
            encoded_latent = encoded_latent[:1]
    elif len(encoded_latent.shape) == 3:
        encoded_latent = encoded_latent.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected encoded_latent shape: {encoded_latent.shape}")
    
    # 规范化提示权重并计算影响因子
    prompt_weight = max(0.0, min(prompt_weight, 1.0))
    influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
    scaled_latent = encoded_latent * influence
    
    # 处理掩码（如果提供）
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) != 4:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        mask = mask[:1, :1]
    
    # 构建潜在空间字典
    latent = {"samples": encoded_latent}
    ref_latent = scaled_latent
    
    # 根据相关性设置参考潜在空间（仅当 relevance 为 True 时处理 positive）
    if relevance:
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [ref_latent]}, append=True
        )
    
    # 无论相关性如何，只要有掩码就添加到 latent 中
    if mask is not None:
        latent["noise_mask"] = mask

    # 验证最终样本形状
    samples = latent["samples"]
    if len(samples.shape) != 4:
        raise ValueError(f"Unexpected samples shape: {samples.shape}")
        
    return latent, positive
