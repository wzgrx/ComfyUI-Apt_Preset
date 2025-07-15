import math
import torch
from typing import Any, Callable, Mapping
import comfy
import numpy as np

from ..main_unit import *


#region-----------------def---------------


def get_input_nodes(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_nodes = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_nodes.append(in_node)
    return input_nodes


def get_input_types(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_types = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_type = in_node["outputs"][in_socket_id]["type"]
        input_types.append(input_type)
    return input_types


def keyframe_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_params = ""
    for item in schedule:   
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        if not line.strip():
            print(f"[Warning] Skipped blank line at line {i}")
            continue
        frame_str, params = line.split('@', 1)
        frame = int(frame_str)
        params = params.lstrip()
        if frame < current_frame:
            previous_params = params
            continue
        if frame == current_frame:
            previous_params = params
        else:
            params = previous_params
        return params
    return previous_params

def prompt_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_prompt = ""
    previous_keyframe = 0
    for item in schedule:   
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        frame_str, prompt = line.split('@', 1)
        frame_str = frame_str.strip('\"')
        frame = int(frame_str)
        prompt = prompt.lstrip()
        prompt = prompt.replace('"', '')        
        if frame < current_frame:
            previous_prompt = prompt
            previous_keyframe = frame
            continue
        elif frame == current_frame:
            next_prompt = prompt
            next_keyframe = frame             
            previous_prompt = prompt
            previous_keyframe = frame
        else:
            next_prompt = prompt
            next_keyframe = frame            
            prompt = previous_prompt
        return prompt, next_prompt, previous_keyframe, next_keyframe
    return previous_prompt, previous_prompt, previous_keyframe, previous_keyframe


#endregion---------------def---------------



#region---------------math---------------------


class math_Remap_data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "value": (ANY_TYPE,),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "easing": (EASING_TYPES,{"default": "Linear"},
                ),
            },
            "optional": {

            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("float", "int",)
    CATEGORY = "Apt_Preset/data"

    def set_range(
        self,
        clamp,
        source_min,
        source_max,
        target_min,
        target_max,
        easing,
        value,
    ):
        
        
        try:
            float_value = float(value)
        except ValueError:
            raise ValueError("Invalid value for conversion to float")
        
        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (float_value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)
        eased_value = apply_easing(normalized_value, easing)
        res_float = target_min + (target_max - target_min) * eased_value
        res_int = int(res_float)

        return (res_float, res_int)



class math_Exec:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "FUNC": ("STRING", {"forceInput": False, "multiline": True, "default": "result = x[0]"})
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Exec"
    RETURN_TYPES = (ANY_TYPE, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        # 创建一个空字典，用于存储结果
        result_dict = {}
        
        # 获取kwargs中的FUNC参数
        func_str = kwargs["FUNC"]
        
        # 如果FUNC参数中没有result，则打印错误信息并返回None
        if "result" not in func_str:
            print("error: no `result` in FUNC")
            return (None, )
        
        # 将FUNC参数中的result替换为result_dict['result']
        func_str = func_str.replace("result", "result_dict['result']")
        print(func_str)

        # 创建一个空字符串，用于存储变量定义的代码
        vars_str = "x = []\n"
        # 遍历kwargs中的所有键
        for k in kwargs.keys():
            # 如果键以"x["开头，则将其添加到变量定义的代码中
            if k.startswith("x["):
                vars_str += f"x.append(kwargs['{k}'])\n"
        
        # 将变量定义的代码和FUNC参数的代码合并
        exec_str = vars_str + func_str

        # 执行合并后的代码
        exec(exec_str)
        # 返回结果字典中的result
        return (result_dict["result"], )



class math_calculate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {"default": "a", "multiline": False,}),
                "a": (ANY_TYPE, {"forceInput": True}),
            },
            "optional": {
                "b": (ANY_TYPE,),
                "c": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data",)
    FUNCTION = "calculate"
    CATEGORY = "Apt_Preset/data"
    DESCRIPTION = """
    - 基本运算：加 (+)、减 (-)、乘 (*)、除 (/)、模 (%)
    - 三角函数: sin(a)、cos、tan、asin、acos、atan
    - 幂运算与开方: pow(a,2)=a*a、sqrt
    - 对数运算: log、log10(a)
    - 双曲函数: sinh、cosh、tanh、asinh、acosh、atanh
    - 角度与弧度转换: radians、degrees
    - 绝对值与取整: fabs、ceil(a/b)向上取整、floor(a/b)向下取整
    - 比大小: max(a,b,c) ,min(a,b,c)
    - 布尔运算: a>b,a<b,a>=b,a<=b,a==b,a!=b ,返回True或False
    """


    def calculate(self, expression, a, b=None, c=None):
        try:
            # 定义命名空间，将输入变量和常用数学函数添加到其中
            namespace = {
                'a': a,
                'b': b if b is not None else 0,
                'c': c if c is not None else 0,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'pow': math.pow,
                'sqrt': math.sqrt,
                'log': math.log,
                'log10': math.log10,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'asinh': math.asinh,
                'acosh': math.acosh,
                'atanh': math.atanh,
                'radians': math.radians,
                'degrees': math.degrees,
                'fabs': math.fabs,
                'ceil': math.ceil,
                'floor': math.floor
            }
            # 执行表达式计算
            result = eval(expression, namespace)
            return (result,)
        except Exception as e:
            print(f"Error performing calculation: {e}")
            return (None,)



#endregion---------------------math---------------------



#region---------------list---------------

class list_GetByIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
                "index": ("INT", {"forceInput": False, "default": 0}),
            }
        }
    
    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data",)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, ANY: list, index: list[int]):
        index = index[0]
        if index >= len(ANY):
            print("Error: index out of range")
            return (None, )
        return (ANY[index], )


class list_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": 0, "min": -9007199254740991}),
            }
        }
    

    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, ANY: list, start: list[int], end: list[int]):
        start = start[0]
        end = end[0]
        if end == 0:
            return (ANY[start:], )
        return (ANY[start:end], )


class list_Merge:
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
    
    TITLE = "Merge List"
    INPUT_IS_LIST = True
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        unique_id = unique_id[0]
        prompt = prompt[0]
        extra_pnginfo = extra_pnginfo[0]
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )



class list_sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                             "max_frames": ("INT", {"default": 100, "min": 1, "max": 99999}),  # 添加 max_frames 参数
                             "easing_type": (list(easing_functions.keys()), ),
                },
        }
    RETURN_TYPES = ("FLOAT","INT",  "FLOAT")
    RETURN_NAMES = ("float","int",  "weight")
    OUTPUT_IS_LIST = (True, True, True)  # 强制输出结果为列表
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data"

    def adv_schedule(self, schedule, max_frames, easing_type):
        schedule_lines = list()
        if schedule == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
            return ([], [], [])  # 返回空列表

        lines = schedule.split('\n')
        for line in lines:
            schedule_lines.extend([("ADV", line)])

        int_out_list = []
        value_out_list = []
        weight_list = []

        for current_frame in range(max_frames):
            params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
            if params == "":
                print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame {current_frame}. Advanced schedules must start at frame 0.")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)
                continue

            try:
                current_params, next_params, from_index, to_index = prompt_scheduler(schedule_lines, "ADV", current_frame)
                if to_index == from_index:
                    t = 1.0
                else:
                    t = (current_frame - from_index) / (to_index - from_index)
                if t < 0 or t > 1:
                    t = 1.0
                weight = apply_easing(t, easing_type)
                current_value = float(current_params)
                next_value = float(next_params)
                value_out = current_value + (next_value - current_value) * weight
                int_out = int(value_out)

                int_out_list.append(int_out)
                value_out_list.append(value_out)
                weight_list.append(weight)
            except ValueError:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)

        return ( value_out_list, int_out_list, weight_list)



class list_num_range:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "num": ("INT", {"default": 10, "min": 2}),
            },
        }
    
    TITLE = "Create Linspace"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("data", "list", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, start: float, stop: float, num: int):
        range_list = list(np.linspace(start, stop, num))
        return (range_list, range_list, len(range_list))


#endregion---------------list---------------



#region---------------sch---------------


class sch_split_text:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,"default":"a,b,c"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "delimiter": ('STRING', {"forceInput": False}, {"default": ","}),
            }
        }

    RETURN_TYPES = ("STRING",'INT',)
    RETURN_NAMES = ('i_text', "length",)
    FUNCTION = "text_to_list"

    CATEGORY = "Apt_Preset/data"

    def text_to_list(self,text,current_frame,delimiter):
        delimiter=delimiter.replace("\\n","\n")
        strList=text.split(delimiter)

        return (strList[current_frame],len(strList) )



class sch_text:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),  
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                },
                "optional": {
                }

        }
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("current_prompt", "next_prompt", "weight")
    FUNCTION = "simple_schedule"
    CATEGORY = "Apt_Preset/data"

    def simple_schedule(self, keyframe_list,  current_frame, easing_type,):
        keyframes = list()
        if keyframe_list == "":
            print(f"[Error] CR Simple Prompt Scheduler. No lines in keyframe list") 
            return ()   
        lines = keyframe_list.split('\n')
        for line in lines:
            if not line.strip():
                print(f"[Warning] CR Simple Prompt Scheduler. Skipped blank line at line {i}")
                continue                  
            keyframes.extend([("SIMPLE", line)])        
        current_prompt, next_prompt, current_keyframe, next_keyframe = prompt_scheduler(keyframes, "SIMPLE", current_frame)
        if current_prompt == "":
            print(f"[Warning] CR Simple Prompt Scheduler. No prompt found for frame. Simple schedules must start at frame 0.")
        else:        
            try:
                current_prompt_out = str(current_prompt)
                next_prompt_out = str(next_prompt)
                from_index = int(current_keyframe)
                to_index = int(next_keyframe)
            except ValueError:
                print(f"[Warning] CR Simple Text Scheduler. Invalid keyframe at frame {current_frame}")
            
            if from_index == to_index:
                weight_out = 1.0
            else:
                # 缓入缓出效果
                t = (to_index - current_frame) / (to_index - from_index)

                weight_out =  apply_easing(t, easing_type) 


            return(current_prompt_out, next_prompt_out, weight_out)



class sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                             "easing_type": (list(easing_functions.keys()), ),
                },
        }
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("INT", "FLOAT", "weight")
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data"

    def adv_schedule(self, schedule, current_frame, easing_type):
        schedule_lines = list()
        if schedule == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
            return ()
        lines = schedule.split('\n')
        for line in lines:        
            schedule_lines.extend([("ADV", line)])        
        params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
        if params == "":
            print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame. Advanced schedules must start at frame 0.")
        else:
            try:
                current_params, next_params, from_index, to_index = prompt_scheduler(schedule_lines, "ADV", current_frame)
                if to_index == from_index:
                    t = 1.0
                else:
                    t = (current_frame - from_index) / (to_index - from_index)
                if t < 0 or t > 1:
                    t = 1.0
                weight = apply_easing(t, easing_type)
                current_value = float(current_params)
                next_value = float(next_params)
                value_out = current_value + (next_value - current_value) * weight
                int_out = int(value_out)
            except ValueError:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}")
            return (int_out, value_out, weight)



class sch_Prompt:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",),
                            "keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),  
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                            }
        }
    
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("CONDITIONING", )
    FUNCTION = "condition"
    CATEGORY = "Apt_Preset/data"

    def condition(self, clip, keyframe_list, current_frame, easing_type):      
        
        (current_prompt, next_prompt, weight) = sch_text().simple_schedule( keyframe_list, current_frame, easing_type)
        
        # CLIP text encoding
        tokens = clip.tokenize(str(next_prompt))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(str(current_prompt))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
        print(weight)
        
        # Average conditioning
        conditioning_to_strength = weight
        conditioning_from = [[cond_from, {"pooled_output": pooled_from}]]
        conditioning_to = [[cond_to, {"pooled_output": pooled_to}]]
        out = []

        if len(conditioning_from) > 1:
            print("Warning: Conditioning from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)

        return (out,)



class sch_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_image",)
    FUNCTION = "select_image"
    CATEGORY = "Apt_Preset/data"

    def select_image(self, images, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(images) - 1)  # 调整当前帧
        selected_image = images[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or image count. Using frame {adjusted_frame}.")
        return (selected_image,)



class sch_mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("selected_mask",)
    FUNCTION = "select_mask"
    CATEGORY = "Apt_Preset/data"

    def select_mask(self, masks, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(masks) - 1)  # 调整当前帧
        selected_mask = masks[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or mask count. Using frame {adjusted_frame}.")
        return (selected_mask,)


#endregion---------------batch_cycler---------------




class BatchGetByIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
                "index": ("INT", {"default": 0}),
            }
        }
    
    TITLE = "Batch: Get By Index"
    RETURN_TYPES = (ANY_TYPE, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, LIST: list, index: int):
        if index >= len(LIST):
            print("Error: index out of range")
            return (None, )
        return (LIST[index], )



class BatchSlice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": 0, "min": -9007199254740991}),
            }
        }
    
    TITLE = "Batch: Slice"
    RETURN_TYPES = ("LIST", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, LIST: list, start: int, end: int):
        if end == 0:
            return (LIST[start:], )
        return (LIST[start:end], )




class MergeBatch:
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
    
    TITLE = "Merge Batch"
    RETURN_TYPES = ("LIST", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )


