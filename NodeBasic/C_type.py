
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


class ByPassTypeTuple(tuple):
	def __getitem__(self, index):
		if index > 0:
			index = 0
		item = super().__getitem__(index)
		if isinstance(item, str):
			return AnyType(item)
		return item



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

    INPUT_IS_LIST = True
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

    INPUT_IS_LIST = True
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


class type_Anyswitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "output_index": ("INT", {"default": 0, "min": 1, "max": 49, "step": 1}),  # 添加 data_ 输入
            }
        }
    
    RETURN_TYPES = (ANY_TYPE,)  # 修改返回类型为 ANY_TYPE
    RETURN_NAMES = ("data",)  # 修改返回名称为更通用的 "data"
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Preset/type"

    def stack_image(self, count, output_index, **kwargs):

        output_index = output_index - 1  # 调整 data_ 为 0 开始的索引

        data_list = []
        
        for i in range(1, count + 1):
            data = kwargs.get(f"data_{i}")  # 修改输入参数名
            if data is not None:
                data_list.append(data)
        
        if output_index < len(data_list):
            return (data_list[output_index],)  # 根据 data_ 输出对应元素
        return (None,)  # data_ 超出范围返回 None


class type_BatchToList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
            }
        }
    
    TITLE = "Batch To List"
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, LIST: list):
        return (LIST, )


class type_ListToBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
            }
        }
    
    TITLE = "List To Batch"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("LIST", )
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, ANY: list):
        return (ANY, )


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



class creat_any_batch:
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
    CATEGORY = "Apt_Preset/Deprecated"

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
    CATEGORY = "Apt_Preset/Deprecated"

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




class type_AnyCast:
    def __init__(self):
        # 类型构造函数映射
        self.type_constructor = {
            "LIST": list,
            "SET": set,
            "DICTIONARY": dict,
            "TUPLE": tuple,
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {}),
                "TYPE": (["anytype", "STRING", "INT", "FLOAT", "LIST", "SET", "TUPLE", "DICTIONARY", "BOOLEAN", "UTF8_STRING"], {}),
            },
        }
    
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/type"

    def run(self, ANY, TYPE):
        # 处理 None 值
        if ANY is None:
            if TYPE in ["INT", "FLOAT"]:
                return (0 if TYPE == "INT" else 0.0,)
            elif TYPE == "STRING":
                return ("",)
            elif TYPE == "BOOLEAN":
                return (False,)
            elif TYPE in ["LIST", "SET", "DICTIONARY", "TUPLE"]:
                return (self.type_constructor[TYPE](),)
            elif TYPE == "UTF8_STRING":
                return ("",)
            else:
                return (None,)

        if TYPE == "LIST":
            if isinstance(ANY, list):
                return ([self.try_cast(item, "ANY") for item in ANY],)
            elif isinstance(ANY, set):
                return (list(ANY),)
            else:
                return ([ANY],)
                
        elif TYPE == "SET":
            if isinstance(ANY, set):
                return (ANY,)
            elif isinstance(ANY, list):
                return (set(ANY),)
            elif isinstance(ANY, dict):
                return (set(ANY.keys()),)
            else:
                return ({ANY},)
                
        elif TYPE == "TUPLE":
            if isinstance(ANY, tuple):
                return (ANY,)
            elif isinstance(ANY, list):
                return (tuple(ANY),)
            else:
                return ((ANY,),)
                
        elif TYPE == "DICTIONARY":
            if isinstance(ANY, dict):
                return ({k: self.try_cast(v, "ANY") for k, v in ANY.items()},)
            elif isinstance(ANY, str):
                try:
                    import json
                    parsed = json.loads(ANY)
                    return (self.try_cast(parsed, "DICTIONARY")[0],)
                except (json.JSONDecodeError, TypeError):
                    try:
                        import ast
                        parsed = ast.literal_eval(ANY)
                        return (self.try_cast(parsed, "DICTIONARY")[0],)
                    except (SyntaxError, ValueError):
                        return ({},)
            elif isinstance(ANY, list) and len(ANY) > 0:
                if isinstance(ANY[0], (list, tuple)) and len(ANY[0]) == 2:
                    return (dict(ANY),)
            return ({},)
            
        elif TYPE == "BOOLEAN":
            if isinstance(ANY, str):
                return (ANY.lower() in ["true", "1", "yes"],)
            return (bool(ANY),)
            
        elif TYPE == "INT":
            if isinstance(ANY, str):
                try:
                    return (int(float(ANY)),)
                except ValueError:
                    return (0,)
            return (int(ANY),)
            
        elif TYPE == "FLOAT":
            if isinstance(ANY, str):
                try:
                    return (float(ANY),)
                except ValueError:
                    return (0.0,)
            return (float(ANY),)
            
        elif TYPE == "STRING":
            return (str(ANY),)
            
        elif TYPE == "UTF8_STRING":
            try:
                # 执行 UTF-8 编码转换
                encoded_bytes = str(ANY).encode('utf-8', 'ignore')
                encoded_text = encoded_bytes.decode('utf-8', 'replace')
                return (encoded_text,)
            except Exception as e:
                return (f"Error during UTF-8 encoding: {e}",)
                
        else:  # 其他类型或 ANY_TYPE
            return (ANY,)

    def try_cast(self, value, target_type):
        # 递归调用 run 方法处理嵌套结构
        return self.run(value, target_type)[0]

#endregion---------------create--------------


