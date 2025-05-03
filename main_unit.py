import numpy as np
import torch
import os
import comfy.utils
from comfy.cli_args import args
import comfy.samplers
from PIL import Image
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


#region--------------------高频-------------


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


available_ckpt = folder_paths.get_filename_list("checkpoints")
available_unets = folder_paths.get_filename_list("unet")
available_clips = folder_paths.get_filename_list("text_encoders")
available_loras = folder_paths.get_filename_list("loras")
available_vaes = folder_paths.get_filename_list("vae")

#region-----------context全局定义------------------------------------------------------------------------------#




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
    "sampler": ("sampler", comfy.samplers.KSampler.SAMPLERS, "sampler"),
    "scheduler": ("scheduler", comfy.samplers.KSampler.SCHEDULERS, "scheduler"),


    "clip1": ("clip1", folder_paths.get_filename_list("clip"), "clip1"),
    "clip2": ("clip2", folder_paths.get_filename_list("clip"), "clip2"),
    "clip3": ("clip3", folder_paths.get_filename_list("clip"), "clip3"),
    "unet_name": ("unet_name", folder_paths.get_filename_list("unet"), "unet_name"),
    "ckpt_name": ("ckpt_name", folder_paths.get_filename_list("checkpoints") ,"ckpt_name"),
    "pos": ("pos", "STRING", "pos"),
    "neg": ("neg", "STRING", "neg"),
    "width": ("width", "INT","width" ),
    "height": ("height", "INT","height"),
    "batch": ("batch", "INT","batch"),



}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler", "scheduler","clip1", "clip2", "clip3", "unet_name", "ckpt_name"]


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


_load_ctx_inputs_list = ["context", "clip1", "clip2", "clip3", "ckpt_name","unet_name", "pos","neg" ,"width", "height", "batch"]
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


# region---------------缓动函数------------------------
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
    "Sine In": easeInSine,
    "Sine Out": easeOutSine,
    "Sine In+Out": easeInOutSine,
    "Quart In": easeInQuart,
    "Quart Out": easeOutQuart,
    "Quart In+Out": easeInOutQuart,
    "Cubic In": easeInCubic,
    "Cubic Out": easeOutCubic,
    "Cubic In+Out": easeInOutCubic,
    "Circ In": easeInCirc,
    "Circ Out": easeOutCirc,
    "Circ In+Out": easeInOutCirc,
    "Back In": easeInBack,
    "Back Out": easeOutBack,
    "Back In+Out": easeInOutBack,
    "Elastic In": easeInElastic,
    "Elastic Out": easeOutElastic,
    "Elastic In+Out": easeInOutElastic,
    "Bounce In": easeInBounce,
    "Bounce Out": easeOutBounce,
    "Bounce In+Out": easeInOutBounce,
    "Sin Squared": easeInOutSinSquared
}

EASING_TYPES = list(easing_functions.keys())

def apply_easing(value, easing_type):
    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)
    
    raise ValueError(f"Unknown easing type: {easing_type}")



# endregion---------------插值算法------------------------





#region  ----math 算法-------------------------------------------------------------#


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




def adapt_to_batch(value, num_samples):    # 适应批处理数据处理
    if not hasattr(value, '__iter__'):
        value = [value]
    if num_samples < len(value):
        value = value[:num_samples]
    elif num_samples > len(value):
        last_value = value[-1]
        value = value + [last_value] * (num_samples - len(value))
    return value



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


def read_ratios():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(current_dir, 'web')
    file_path = os.path.join(p, 'ratios.json')
    
    # 显式指定使用 utf-8 编码打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    ratio_sizes = list(data['ratios'].keys())
    ratio_dict = data['ratios']
    return ratio_sizes, ratio_dict



def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):  # 保存图片
    """Save or Preview Image"""
    from nodes import PreviewImage, SaveImage
    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']



def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))



def batch_tensor_to_pil(img_tensor):
    return [tensor2pil(img_tensor, i) for i in range(img_tensor.shape[0])]

def batched_pil_to_tensor(images):
    return torch.cat([pil2tensor(image) for image in images], dim=0)




def none2list(folderlist):
    list = ["None"]
    list += folderlist
    return list


def append_helper(t, mask, c, set_area_to_bounds, strength):
    if mask is not None:  
        n = [t[0], t[1].copy()]
        _, h, w = mask.shape
        n[1]['mask'] = mask
        n[1]['set_area_to_bounds'] = set_area_to_bounds
        n[1]['mask_strength'] = strength
        c.append(n)




def mask2image(input_mask_pil):
    input_mask_tensor = pil2tensor(input_mask_pil)
    result_tensor = input_mask_tensor.expand(-1, 3, -1, -1)
    return result_tensor


def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    out = model_loading.load_state_dict(sd).eval()
    return out


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


def replace_text(text, target, replace):
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
    
    targets = split_with_quotes(target)
    exchanges = split_with_quotes(replace)
    

    word_map = {}
    for target, exchange in zip(targets, exchanges):

        target_clean = target.strip('"').strip().lower()
        exchange_clean = exchange.strip('"').strip()
        word_map[target_clean] = exchange_clean
    

    sorted_targets = sorted(word_map.keys(), key=len, reverse=True)
    
    result = text
    for target in sorted_targets:
        if ' ' in target:
            pattern = re.escape(target)
        else:
            pattern = r'\b' + re.escape(target) + r'\b'
        
        result = re.sub(pattern, word_map[target], result, flags=re.IGNORECASE)
    return (result,)




def AD_schdule_graph(normalized_amp):     #数据图示
    width = int(len(normalized_amp) / 10)
    if width < 10:
        width = 10
    if width > 100:
        width = 100
    plt.figure(figsize=(width, 6))
    plt.plot(normalized_amp)
    plt.xlabel("Frame(s)")
    plt.ylabel("Amplitude")
    plt.grid()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()  
    buffer.seek(0)
    image = Image.open(buffer)

    return (pil2tensor(image),)







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


def hex_to_rgb(hex_color: str, bgr: bool = False):
    hex_color = hex_color.lstrip("#")
    if bgr:
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))




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



#endregion-----------------高频-------------
