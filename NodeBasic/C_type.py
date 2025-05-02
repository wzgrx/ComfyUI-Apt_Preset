
import torch
import comfy
import torch
import numpy as np
from PIL import Image, ImageOps
from ..main_unit import *


PACK_PREFIX = 'value'

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask



#region---------------type---------------


class ByPassTypeTuple(tuple):
	def __getitem__(self, index):
		if index > 0:
			index = 0
		item = super().__getitem__(index)
		if isinstance(item, str):
			return AnyType(item)
		return item


class Pack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Pack"
    RETURN_TYPES = ("PACK", )
    RETURN_NAMES = ("PACK", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        data = {}
        pack = {
            "id": unique_id,
            "data": data,
        }
        for k, v in kwargs.items():
            if k.startswith('value'):
                i = int(k.split("_")[1])
                data[i - 1] = {
                    "name": cur_node["inputs"][i - 1]["name"],
                    "type": cur_node["inputs"][i - 1]["type"],
                    "value": v,
                }

        return (pack, )


class Unpack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "PACK": ("PACK", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Unpack"
    RETURN_TYPES = ByPassTypeTuple(("*", ))
    RETURN_NAMES = ByPassTypeTuple(("value_1", ))
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, PACK: dict, unique_id, prompt, extra_pnginfo):
        length = len(PACK["data"])
        types = []
        names = []
        outputs = []
        for i in range(length):
            d = PACK["data"][i]
            names.append(d["name"])
            types.append(d["type"])
            outputs.append(d["value"])
        return tuple(outputs)


class type_BasiPIPE:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "optional": {
                "context": ("RUN_CONTEXT", ),
            },
        }
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    CATEGORY = "Apt_Preset/type"
    
    FUNCTION = "fn"

    def fn(self, context):
        pipe = (context['model'], context['clip'], context['vae'], context['positive'], context['negative'])
        return pipe,


class type_text_list2batch :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text_list": (any_type,),
                    "delimiter":(["newline","comma","backslash","space"],),
                            },
                }
    
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    INPUT_IS_LIST = True # 当true的时候，输入时list，当false的时候，如果输入是list，则会自动包一层for循环调用
    OUTPUT_IS_LIST = (False,)

    def run(self,text_list,delimiter):
        delimiter=delimiter[0]
        if delimiter =='newline':
            delimiter='\n'
        elif delimiter=='comma':
            delimiter=','
        elif delimiter=='backslash':
            delimiter='\\'
        elif delimiter=='space':
            delimiter=' '
        t=''
        if isinstance(text_list, list):
            t=delimiter.join(text_list)
        return (t,)


class type_BasiPIPE:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "optional": {
                "context": ("RUN_CONTEXT", ),
            },
        }
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    CATEGORY = "Apt_Preset/type"
    
    FUNCTION = "fn"

    def fn(self, context):
        pipe = (context['model'], context['clip'], context['vae'], context['positive'], context['negative'])
        return pipe,


class type_text_list2batch :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text_list": (any_type,),
                    "delimiter":(["newline","comma","backslash","space"],),
                            },
                }
    
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    INPUT_IS_LIST = True # 当true的时候，输入时list，当false的时候，如果输入是list，则会自动包一层for循环调用
    OUTPUT_IS_LIST = (False,)

    def run(self,text_list,delimiter):
        delimiter=delimiter[0]
        if delimiter =='newline':
            delimiter='\n'
        elif delimiter=='comma':
            delimiter=','
        elif delimiter=='backslash':
            delimiter='\\'
        elif delimiter=='space':
            delimiter=' '
        t=''
        if isinstance(text_list, list):
            t=delimiter.join(text_list)
        return (t,)


class type_text_2_UTF8:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
            }
        }


    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    CATEGORY = "Apt_Preset/type"
    FUNCTION = "encode_utf8"

    def encode_utf8(self, text):
        try:
            encoded_bytes = text.encode('utf-8', 'ignore')
            encoded_text = encoded_bytes.decode('utf-8', 'replace')
            return (encoded_text,)
        except Exception as e:
            return (f"Error during encoding: {e}",)


class type_Image_List2Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "images": ("IMAGE", ),
                    }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/type"

    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


class type_Image_Batch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/type"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )


class type_Mask_Batch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "masks": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("MASK", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/type"

    def doit(self, masks):
        if masks is None:
            empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([empty_mask], )

        res = []

        for mask in masks:
            res.append(mask)

        print(f"mask len: {len(res)}")

        res = [make_3d_mask(x) for x in res]

        return (res, )


class type_Mask_List2Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask": ("MASK", ),
                    }
                }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK", )
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/type"

    def doit(self, mask):
        if len(mask) == 1:
            mask = make_3d_mask(mask[0])
            return (mask,)
        elif len(mask) > 1:
            mask1 = make_3d_mask(mask[0])

            for mask2 in mask[1:]:
                mask2 = make_3d_mask(mask2)
                if mask1.shape[1:] != mask2.shape[1:]:
                    mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
                mask1 = torch.cat((mask1, mask2), dim=0)

            return (mask1,)
        else:
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu").unsqueeze(0)
            return (empty_mask,)


class AnyCast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}),
                "TYPE": (["*", "STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE", "LATENT", "MASK", "NOISE", "SAMPLER", "SIGMAS", "GUIDER", "MODEL", "CLIP", "VAE", "CONDITIONING"], {}),
            },
        }
    
    TITLE = "Any Cast"
    RETURN_TYPES = (ANY_TYPE,)  #输出名字，动态

    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, ANY, TYPE):
        result = try_cast(ANY, TYPE)
        return (result, )


class AnyToDict:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}), 
            },
        }
    
    TITLE = "Any To Dict"
    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("DICT", "str()")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, ANY):
        if type(ANY) is dict:
            return (ANY, str(ANY))
        elif not hasattr(ANY, '__dict__'):
            print(f"Object of type {type(ANY).__name__} doesn't have a __dict__ attribute")
            return ({}, str(ANY))
        else:
            return (vars(ANY), str(ANY))




class type_Anyswitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "data_": ("INT", {"default": 0, "min": 1, "max": 49, "step": 1}),  # 添加 data_ 输入
            }
        }
    
    RETURN_TYPES = (ANY_TYPE,)  # 修改返回类型为 ANY_TYPE
    RETURN_NAMES = ("data",)  # 修改返回名称为更通用的 "data"
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Preset/type"

    def stack_image(self, count, data_, **kwargs):

        data_ = data_ - 1  # 调整 data_ 为 0 开始的索引

        data_list = []
        
        for i in range(1, count + 1):
            data = kwargs.get(f"data_{i}")  # 修改输入参数名
            if data is not None:
                data_list.append(data)
        
        if data_ < len(data_list):
            return (data_list[data_],)  # 根据 data_ 输出对应元素
        return (None,)  # data_ 超出范围返回 None




#endregion---------------type---------------



#region---------------create--------------


class creat_any_List:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Create List"
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith(PACK_PREFIX):
                output_list.append(v)
        return (output_list, )





class CreateBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Create Batch"
    RETURN_TYPES = ("LIST",)  
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith(PACK_PREFIX):
                output_list.append(v)
        return (output_list, )


class creat_image_batch:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/type"

    def doit(self, unique_id, prompt, extra_pnginfo, **kwargs):
        images = [value for value in kwargs.values() if value is not None]
        
        if len(images) == 0:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        image1 = images[0]
        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
            image1 = torch.cat((image1, image2), dim=0)
        return (image1,)


class creat_mask_batch:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/type"

    def doit(self, unique_id, prompt, extra_pnginfo, **kwargs):
        masks = [make_3d_mask(value) for value in kwargs.values() if value is not None]
        
        if len(masks) == 0:
            return (torch.zeros((1, 64, 64), dtype=torch.float32),)
            
        mask1 = masks[0]
        for mask2 in masks[1:]:
            if mask1.shape[1:] != mask2.shape[1:]:
                mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
            mask1 = torch.cat((mask1, mask2), dim=0)
        return (mask1,)


class creat_mask_batch_input:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Preset/type"

    def stack_image(self, count, **kwargs):
        mask_list = []
        
        for i in range(1, count + 1):
            mask = kwargs.get(f"mask_{i}")
            if mask is not None:
                mask_list.append(mask)
        if len(mask_list) > 0:
            mask_batch = torch.cat(mask_list, dim=0)
            return (mask_batch,)
        return (None,)


class creat_image_batch_input:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Preset/type"

    def stack_image(self, count, **kwargs):
        image_list = []
        
        for i in range(1, count + 1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                image_list.append(image)
        if len(image_list) > 0:
            image_batch = torch.cat(image_list, dim=0)
            return (image_batch,)
        return (None,)






#endregion---------------create--------------


