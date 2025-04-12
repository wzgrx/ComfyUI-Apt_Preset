import numpy as np
import torch
import os
import comfy.utils
from comfy.cli_args import args
import comfy.samplers
from PIL import Image
import json
import csv

import math


from comfy_extras.chainner_models import model_loading
import folder_paths
from comfy import model_management
from nodes import common_ksampler,  VAEDecode, VAEEncode
from math import ceil
import torchvision.transforms.functional as TF
from typing import cast

#region--------------------高频-------------


#region-----------context全局定义------------------------------------------------------------------------------#


class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")




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





class Data_chx_Merge:
    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "required": {  },
            "optional": {
            "context": ("RUN_CONTEXT",),  
                "chx1": ("RUN_CONTEXT",),
                "chx2": ("RUN_CONTEXT",),
            },
        }

    RETURN_TYPES = "RUN_CONTEXT",
    RETURN_NAMES = "context",
    CATEGORY = "Apt_Preset/load"
    FUNCTION = "merge"
    

    def get_return_tuple(self, ctx):
        return get_orig_context_return_tuple(ctx)

    def merge(self, context=None, chx1=None, chx2=None):
        ctxs = [context, chx1, chx2]  
        ctx = merge_new_context(*ctxs)
        return self.get_return_tuple(ctx)


class Data_chx_MergeBig:
    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "required": {  },
            
            "optional": { "context": ("RUN_CONTEXT",),  
                "chx1": ("RUN_CONTEXT",),
                "chx2": ("RUN_CONTEXT",),
                "chx3": ("RUN_CONTEXT",),
                "chx4": ("RUN_CONTEXT",),
                "chx5": ("RUN_CONTEXT",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING","CONDITIONING","IMAGE","MASK",)
    RETURN_NAMES = ("context","model","positive","negative","images","mask",)
    CATEGORY = "Apt_Preset/load"
    FUNCTION = "merge"
    

    def get_return_dict(self, ctx):
        # 假设 merge_new_context 返回一个字典
        return merge_new_context(ctx)

    def merge(self, context=None, chx1=None, chx2=None, chx3=None, chx4=None, chx5=None):
        ctxs = [context, chx1, chx2, chx3, chx4, chx5]  
        ctx = merge_new_context(*ctxs)
        context_dict = self.get_return_dict(ctx)
        
        # 从字典中获取值
        model = context_dict.get("model", None)
        positive = context_dict.get("positive", None)
        negative = context_dict.get("negative", None)
        images = context_dict.get("images", None)
        mask = context_dict.get("mask", None)
        
        return (context_dict, model, positive, negative, images, mask)









#endregion


#region-----------读取json文件风格,分割成子文件夹-------------------------------------
# 读取 JSON 文件
def read_json_file(file_path):
    if not os.access(file_path, os.R_OK):
        print(f"Warning: No read permissions for file {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            # 检查内容是否符合预期格式
            if not all(['name' in item and 'prompt' in item and 'negative_prompt' in item for item in content]):
                print(f"Warning: Invalid content in file {file_path}")
                return None
            return content
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None

# 从指定的 JSON 文件加载样式数据
def load_styles_from_file():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    ################构建相对路径指向新位置的 sdxl_styles.json 文件################
    file_path = os.path.join(current_directory, 'web', 'sdxl_styles.json')
    ###########################################################################
    json_data = read_json_file(file_path)
    if not json_data:
        return [], []

    combined_data = []
    seen = set()

    for item in json_data:
        original_style = item['name']
        style = original_style
        suffix = 1
        while style in seen:
            style = f"{original_style}_{suffix}"
            suffix += 1
        item['name'] = style
        seen.add(style)
        combined_data.append(item)

    unique_style_names = [item['name'] for item in combined_data if isinstance(item, dict) and 'name' in item]
    return combined_data, unique_style_names

# 验证 JSON 数据结构
def validate_json_data(json_data):
    if not isinstance(json_data, list):
        return False
    for template in json_data:
        if 'name' not in template or 'prompt' not in template:
            return False
    return True

# 根据名称查找模板
def find_template_by_name(json_data, template_name):
    for template in json_data:
        if template['name'] == template_name:
            return template
    return None

# 替换模板中的提示词
def replace_prompts_in_template(template, positive_prompt, negative_prompt):
    positive_result = template['prompt'].replace('{prompt}', positive_prompt)
    json_negative_prompt = template.get('negative_prompt', "")
    negative_result = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt and negative_prompt else json_negative_prompt or negative_prompt
    return positive_result, negative_result

# 读取 SDXL 模板并替换组合提示词
def read_sdxl_templates_replace_and_combine(json_data, template_name, positive_prompt, negative_prompt):
    if not validate_json_data(json_data):
        return positive_prompt, negative_prompt
    template = find_template_by_name(json_data, template_name)
    if template:
        return replace_prompts_in_template(template, positive_prompt, negative_prompt)
    else:
        return positive_prompt, negative_prompt

# 填充样式项，添加预览信息
def populate_items(styles, item_type):
    for idx, item_name in enumerate(styles):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        preview_path = os.path.join(current_directory, item_type, item_name + ".png")

        if len(item_name.split('-')) > 1:
            content = f"{item_name.split('-')[0]} /{item_name}"
        else:
            content = item_name

        if os.path.exists(preview_path):
            styles[idx] = {
                "content": content,
                "preview": preview_path,
                "original_name": item_name  # 添加原始风格名称
            }
        else:
            styles[idx] = {
                "content": content,
                "preview": None,
                "original_name": item_name  # 添加原始风格名称
            }
#endregion------------------读取json文件风格-------------------------------------


# region---------------#deform 插值算法------------------------

def image_2dtransform(
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


def easeInBack(t):
        s = 1.70158
        return t * t * ((s + 1) * t - s)

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
}

def apply_easing(value, easing_type):
    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)
    
    raise ValueError(f"Unknown easing type: {easing_type}")



# endregion---------------插值算法------------------------




def style_list():   # 读取csv文件
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


def latentrepeat(samples, amount): # 重复latent向量

    s = samples.copy()
    s_in = samples["samples"]

    s["samples"] = s_in.repeat((amount, 1, 1, 1))
    if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
        masks = samples["noise_mask"]
        if masks.shape[0] < s_in.shape[0]:
            masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
        s["noise_mask"] = samples["noise_mask"].repeat((amount, 1, 1, 1))
    if "batch_index" in s:
        offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
        s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
    return (s,)

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

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))



def batch_tensor_to_pil(img_tensor):
    return [tensor2pil(img_tensor, i) for i in range(img_tensor.shape[0])]

def batched_pil_to_tensor(images):
    return torch.cat([pil2tensor(image) for image in images], dim=0)



#endregion-----------------高频-------------



#region-----------------收纳不常用-------------

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



#region---------------class chx_ksampler_tile-----------------------
def split_image(img, tile_size=1024):
    """Generate tiles for a given image."""
    
    # 检查 img 的类型
    if isinstance(img, list):
        print("Warning: img is a list, selecting the first element.")
        img = img[0]  # 如果 img 是列表，选择第一个元素

    # 确保 img 是图像对象（如 PIL Image 或 torch tensor）
    if not hasattr(img, 'width') or not hasattr(img, 'height'):
        raise TypeError("The input 'img' must be an image object (e.g., PIL Image or torch tensor).")

    tile_width, tile_height = tile_size, tile_size
    width, height = img.width, img.height

    # Determine the number of tiles needed
    num_tiles_x = ceil(width / tile_width)
    num_tiles_y = ceil(height / tile_height)

    # 如果 num_tiles_x 或 num_tiles_y 的计算结果是 1，确保它们至少为 2
    if num_tiles_x < 2:
        num_tiles_x = 2
    if num_tiles_y < 2:
        num_tiles_y = 2

    # If width or height is an exact multiple of the tile size, add an additional tile for overlap
    if width % tile_width == 0:
        num_tiles_x += 1
    if height % tile_height == 0:
        num_tiles_y += 1

    # Calculate the overlap, ensuring no division by zero
    if num_tiles_x > 1:
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    else:
        overlap_x = 0  # Avoid division by zero

    if num_tiles_y > 1:
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)
    else:
        overlap_y = 0  # Avoid division by zero

    # If overlap is smaller than a threshold, increase the number of tiles
    if overlap_x < 256:
        num_tiles_x += 1
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    if overlap_y < 256:
        num_tiles_y += 1
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)

    tiles = []

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_width - j * overlap_x
            y_start = i * tile_height - i * overlap_y

            # Correct for potential float precision issues
            x_start = round(x_start)
            y_start = round(y_start)

            # Crop the tile from the image
            tile_img = img.crop((x_start, y_start, x_start + tile_width, y_start + tile_height))
            tiles.append(((x_start, y_start, x_start + tile_width, y_start + tile_height), tile_img))

    return tiles


def stitch_images(upscaled_size, tiles):
    """Stitch tiles together to create the final upscaled image with overlaps."""
    
    # Ensure upscaled_size is a tuple (width, height)
    if isinstance(upscaled_size, tuple):
        width, height = upscaled_size
    elif hasattr(upscaled_size, 'size'):
        # If upscaled_size is a PIL image, get its size
        width, height = upscaled_size.size
    elif hasattr(upscaled_size, 'shape'):
        # If upscaled_size is a torch tensor, get its shape
        _, height, width = upscaled_size.shape
    else:
        raise TypeError("upscaled_size should be a tuple, PIL.Image, or torch.Tensor.")
    
    result = torch.zeros((3, height, width))

    # We assume tiles come in the format [(coordinates, tile), ...]
    sorted_tiles = sorted(tiles, key=lambda x: (x[0][1], x[0][0]))  # Sort by upper then left

    # Variables to keep track of the current row's starting point
    current_row_upper = None

    for (left, upper, right, lower), tile in sorted_tiles:

        # Check if we're starting a new row
        if current_row_upper != upper:
            current_row_upper = upper
            first_tile_in_row = True
        else:
            first_tile_in_row = False

        tile_width = right - left
        tile_height = lower - upper
        feather = tile_width // 8  # Assuming feather size is consistent with the example

        mask = torch.ones(tile.shape[0], tile.shape[1], tile.shape[2])

        if not first_tile_in_row:  # Left feathering for tiles other than the first in the row
            for t in range(feather):
                mask[:, :, t:t+1] *= (1.0 / feather) * (t + 1)

        if upper != 0:  # Top feathering for all tiles except the first row
            for t in range(feather):
                mask[:, t:t+1, :] *= (1.0 / feather) * (t + 1)

        # Apply the feathering mask
        tile = tile.squeeze(0).squeeze(0)  # Removes first two dimensions
        tile_to_add = tile.permute(2, 0, 1)
        # Use the mask to correctly feather the new tile on top of the existing image
        combined_area = tile_to_add * mask.unsqueeze(0) + result[:, upper:lower, left:right] * (1.0 - mask.unsqueeze(0))
        result[:, upper:lower, left:right] = combined_area

    # Expand dimensions to get (1, 3, height, width)
    tensor_expanded = result.unsqueeze(0)

    # Permute dimensions to get (1, height, width, 3)
    tensor_final = tensor_expanded.permute(0, 2, 3, 1)
    return tensor_final


def ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, start_step=11, end_step=20):
    """Upscale a tile using the AI model."""
    vaedecoder = VAEDecode()
    vaeencoder = VAEEncode()
    tile = pil2tensor(tile)
    #print('Tile Complexity:', complexity)
    encoded_tile = vaeencoder.encode(vae, tile)[0]
    tile = common_ksampler(base_model, seed, end_step, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, encoded_tile,
                        start_step=start_step, force_full_denoise=True)[0]
    tile = vaedecoder.decode(vae, tile)[0]
    return tile


def run_tiler_for_steps(enlarged_img, base_model, vae, seed, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, steps=20, denoise=0.25, tile_size=1024):
    # Ensure enlarged_img is a valid image object, not a list
    if isinstance(enlarged_img, list):
        print("Warning: enlarged_img is a list, selecting the first element.")
        enlarged_img = enlarged_img[0]  # If it's a list, select the first element
    
    # Ensure enlarged_img is a PIL Image or torch tensor
    if not hasattr(enlarged_img, 'size') and not hasattr(enlarged_img, 'shape'):
        raise TypeError("enlarged_img should be a valid image object (e.g., PIL.Image or torch tensor).")

    # Split the enlarged image into overlapping tiles
    tiles = split_image(enlarged_img, tile_size=tile_size)

    # Resample each tile using the AI model
    start_step = int(steps - (steps * denoise))
    end_step = steps
    resampled_tiles = [(coords, ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler,
                                            positive_cond_base, negative_cond_base, start_step, end_step)) for coords, tile in tiles]

    # Stitch the tiles to get the final upscaled image
    result = stitch_images(enlarged_img.size, resampled_tiles)

    return result


#endregion---------------class chx_ksampler_tile-----------------------
