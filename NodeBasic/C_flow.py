import folder_paths
from comfy import model_management

from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy.utils import common_upscale


from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader



#region----------------lowcpu--------------------------

try:
    import pynvml
    pynvml_installed = True
    pynvml.nvmlInit()
except ImportError:
    pynvml_installed = False
    print("警告：未安装pynvml库，auto选项将不可用。")


def get_gpu_memory_info():
    """获取GPU显存信息"""
    if not pynvml_installed:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = memory_info.total / (1024 * 1024 * 1024)  
        used = memory_info.used / (1024 * 1024 * 1024)    
        return total, used
    except Exception as e:
        print(f"获取GPU信息出错: {e}")
        return None, None
#endregion----------------lowcpu--------------------------






class AlwaysEqual(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class AlwaysTuple(tuple):
    def __getitem__(self, i):
        if i < super().__len__():
            return AlwaysEqual(super().__getitem__(i))
        else:
            return AlwaysEqual(super().__getitem__(-1))


class flow_judge:
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "judge": (AlwaysEqual("*"),),
                "true": (AlwaysEqual("*"), {"lazy": True}),
                "false": (AlwaysEqual("*"), {"lazy": True}),
            }
        }

    RETURN_TYPES = (AlwaysEqual("*"),)
    RETURN_NAMES = ("data",)
    FUNCTION = "judge_bool"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    def check_lazy_status(self, judge, true, false):
        needed = []
        if judge:
            needed.append('true')
        else:
            needed.append('false')
        return needed


    def judge_bool(self, judge, true, false):
        return {"ui": {"value": [True if judge else False]}, "result": (true if judge else false,)}
    




class flow_auto_pixel:
    upscale_methods = ["bicubic","nearest-exact", "bilinear", "area",  "lanczos"]
    crop_methods = ["disabled", "center"]
    # 包含英文的选项列表
    threshold_types = ["(W+H) < threshold", "W*H < threshold", "width <= height", "width > height"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                "image": ("IMAGE",), 
                "threshold_type": (s.threshold_types,),  # 使用更新后的选项列表
                "pixels_threshold": ("INT", { "min": 0, "max": 90000,  "step": 1,}),
                "upscale_method_True": (s.upscale_methods,),
                "upscale_method_False": (s.upscale_methods,),
                "low_pixels_True": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),      # 名称修改
                "high_pixels_False": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),   # 名称修改
                "divisible_by": ("INT", { "default": 8, "min": 0, "max": 512, "step": 1, }),
                }
                }
    

    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("image", )  
    FUNCTION = "auto_pixel"
    CATEGORY = "Apt_Preset/flow"

    def auto_pixel(self, model_name, image, threshold_type, 
                pixels_threshold, upscale_method_True, upscale_method_False, low_pixels_True, high_pixels_False, divisible_by):


        # 处理不同维度的图像张量
        if len(image.shape) == 3:
            # 形状为 (H, W, C) 的单张图像
            height, width, channels = image.shape
            batch_size = 1
        elif len(image.shape) == 4:
            # 形状为 (B, H, W, C) 的批次图像
            batch_size, height, width, channels = image.shape
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # 根据选择的threshold_type确定使用哪种逻辑
        if threshold_type == "(W+H) < threshold":
            if (width + height) < pixels_threshold:
                megapixels = low_pixels_True
                upscale_method = upscale_method_True
            else:
                megapixels = high_pixels_False
                upscale_method = upscale_method_False
        elif threshold_type == "W*H < threshold":
            if (width * height) < pixels_threshold:
                megapixels = low_pixels_True
                upscale_method = upscale_method_True
            else:
                megapixels = high_pixels_False
                upscale_method = upscale_method_False
        elif threshold_type == "width <= height":
            megapixels = low_pixels_True
            upscale_method = upscale_method_True
        elif threshold_type == "width > height":
            megapixels = high_pixels_False
            upscale_method = upscale_method_False
            
        model = UpscaleModelLoader().load_model(model_name)[0]
        image = ImageUpscaleWithModel().upscale(model, image)[0]

        if len(image.shape) == 3:
            H, W, C = image.shape
        else:  # len(image.shape) == 4
            B, H, W, C = image.shape
        
        if divisible_by > 1:
            new_width = W - (W % divisible_by)
            new_height = H - (H % divisible_by)
            
            if new_width == 0:
                new_width = divisible_by
            if new_height == 0:
                new_height = divisible_by
            if new_width != W or new_height != H:
                # 根据图像维度调整处理方式
                if len(image.shape) == 3:
                    image = image.movedim(-1, 0)  # (H, W, C) -> (C, H, W)
                    image = common_upscale(image.unsqueeze(0), new_width, new_height, upscale_method, "center")
                    image = image.squeeze(0).movedim(0, -1)  # (C, H, W) -> (H, W, C)
                else:  # len(image.shape) == 4
                    image = image.movedim(-1, 1)  # (B, H, W, C) -> (B, C, H, W)
                    image = common_upscale(image, new_width, new_height, upscale_method, "center")
                    image = image.movedim(1, -1)  # (B, C, H, W) -> (B, H, W, C)

        return (image,)



class flow_switch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_method": ("BOOLEAN", {"default": True, "label_on": "first_valid_value", "label_off": "select_index"}),
                "input_index": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "output_method": ("BOOLEAN", {"default": True, "label_on": "output_all", "label_off": "select_index"}),
                "output_index": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "in1": (any_type,),
                "in2": (any_type,),
                "in3": (any_type,),
                "in4": (any_type,),
                "in5": (any_type,),
            }
        }

    RETURN_TYPES = (any_type, any_type, any_type, any_type, any_type,)
    RETURN_NAMES = ('out1', 'out2', 'out3', 'out4', 'out5',)
    CATEGORY = "Apt_Preset/flow"
    FUNCTION = "switch"

    DESCRIPTION = """
        - input_method: 自动检测并选择第一个非空输入数据
        - input_index: 手动选择输入端口索引（1-5）
        - output_method: 是否按output_index输出，为真时按索引输出，为假时全部输出
        - output_index: 输出端口索引（1-5）
        - 支持任意类型数据输入，适用于流程控制和条件分支场景
        """

    def switch(self, input_method, input_index, output_method, output_index, 
               in1=None, in2=None, in3=None, in4=None, in5=None):
        inputs = [in1, in2, in3, in4, in5]
        
        if input_method:
            selected_value = None
            for value in inputs:
                if not self.is_none(value):
                    selected_value = value
                    break
        else:
            index = input_index - 1
            if 0 <= index < len(inputs):
                selected_value = inputs[index]
            else:
                selected_value = None
        
        if output_method:
            output = [selected_value] * 5

        else:
            output = [None] * 5
            output_port = output_index - 1
            if output_port < 0 or output_port >= 5:
                output_port = 0
            output[output_port] = selected_value
       
        return tuple(output)

    def is_none(self, value):
        if value is not None:
            if isinstance(value, dict) and 'model' in value and 'clip' in value:
                return all(v is None for v in value.values())
        return value is None





class flow_low_gpu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_type, {}),
                "reserved": ("FLOAT", {
                    "default": 0.6,
                    "min": -2.0,
                    "step": 0.1,
                    "display": "reserved (GB)"
                }),
                "mode": (["manual", "auto"], {
                    "default": "auto",
                    "display": "Mode"
                })
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "set_vram"
    CATEGORY = "Apt_Preset/flow"

    def set_vram(self, anything, reserved, mode="auto", unique_id=None, extra_pnginfo=None):
        if mode == "auto":
            if pynvml_installed:
                total, used = get_gpu_memory_info()
                if total and used:
                    auto_reserved = used + reserved
                    auto_reserved = max(0, auto_reserved)  # 确保不小于0
                    model_management.EXTRA_RESERVED_VRAM = int(auto_reserved * 1024 * 1024 * 1024)
                    print(f'set EXTRA_RESERVED_VRAM={auto_reserved:.2f}GB (自动模式: 总显存={total:.2f}GB, 已用={used:.2f}GB)')
                else:
                    model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)
            else:
                model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)
        else:
            # 手动模式
            reserved = max(0, reserved)
            model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)

        return (anything,)

































