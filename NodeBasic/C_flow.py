import folder_paths

from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy.utils import common_upscale



from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader





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
        model = UpscaleModelLoader().load_model(model_name)[0]

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

        # 执行 upscale 操作
        image = ImageUpscaleWithModel().upscale(model, image)[0]
        image = ImageScaleToTotalPixels().upscale(image, upscale_method, megapixels)[0]
        
        # 调整尺寸以满足divisible_by要求
        # 同样需要处理不同维度的情况
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



























