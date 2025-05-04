
#from .AdvancedControlNet.utils import *
#from .AdvancedControlNet.nodes_main import AdvancedControlNetApply


import sys
import os
from collections.abc import Iterable
import folder_paths
import comfy
from ..NodeBasic.C_AD import batch_get_inbetweens,batch_parse_key_frames
from ..main_unit import *


defaultValue = """0:(0),
30:(1),
60:(0),
90:(0),
120:(0)
"""


#region---------------------------从外部导入模块，加载检查-------------------------------

#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Apt_Preset\NodeChx\AdvancedCN.py"
#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet"
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.path.join(custom_nodes_dir, 'ComfyUI-Advanced-ControlNet')
#target_dir = r"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet"


# 初始化可能从 ComfyUI-Advanced-ControlNet 导入的模块和类
AdvancedControlNetApply = None
TimestepKeyframeGroup = None
ControlWeights = None
LatentKeyframeGroup = None
LatentKeyframe = None


if not os.path.exists(target_dir):
    print("ComfyUI-Advanced-ControlNet 目录不存在，请安装该插件。")
else:
    sys.path.append(target_dir)  # 将 target_dir 目录添加到 Python 模块搜索路径 sys.path 列表的末尾。
    print(f"已添加 {target_dir} 到 sys.path")
    try:
        # 显式导入需要的类和函数
        from adv_control.utils import (
            TimestepKeyframeGroup,
            ControlWeights,
            LatentKeyframeGroup,
            LatentKeyframe
        )
        print("成功导入 adv_control.utils 中的类")
    except ImportError as e:
        print(f"无法导入 adv_control.utils 中的部分模块: {e}")
    try:
        from adv_control.nodes_main import AdvancedControlNetApply
        print("成功导入 adv_control.nodes_main 中的 AdvancedControlNetApply")
    except ImportError:
        print("无法导入 adv_control.nodes_main 模块，请检查该模块是否存在。")


# 定义一个函数用于检查插件是否安装
def check_advanced_controlnet_installed():
    if AdvancedControlNetApply is None:
        raise RuntimeError(" Please install ComfyUI-Advanced-ControlNet  before using this function.。")


#endregion---------------------------加载检查-------------------------------



class AD_sch_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                #"timestep_kf": ("TIMESTEP_KEYFRAME", ),
                #"weights_override": ("CONTROL_NET_WEIGHTS", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    DEPRECATED = True
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "Apt_Preset/AD"

    def apply_controlnet(self, conditioning, control_net, image, strength, 
                        mask_optional=None, model_optional=None, vae_optional=None,
                        timestep_kf: TimestepKeyframeGroup=None, latent_kf_override=None,
                        weights_override: ControlWeights=None):
        check_advanced_controlnet_installed()
        values = AdvancedControlNetApply.apply_controlnet(self, positive=conditioning, negative=None, control_net=control_net, image=image,
                                                        strength=strength, start_percent=0, end_percent=1,
                                                        mask_optional=mask_optional, vae_optional=vae_optional,
                                                        timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                        control_apply_to_uncond=True)
        return (values[0], model_optional)



class AD_sch_latent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": defaultValue}),
                "max_frames": ("INT", {"default": 300, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Apt_Preset/AD"

    def load_keyframe(self, text, max_frames, print_output=None,
                    prev_latent_kf: LatentKeyframeGroup=None,
                    prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                    ):
        check_advanced_controlnet_installed()
        print_output=None
        
        t, _ = self.animate(text, max_frames, print_output)
        float_strengths = t

        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        # if received a normal float input, do nothing
        if type(float_strengths) in (float, int):
            logger.info("No batched float_strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        # if iterable, attempt to create LatentKeyframes with chosen strengths
        elif isinstance(float_strengths, Iterable):
            for idx, strength in enumerate(float_strengths):
                keyframe = LatentKeyframe(idx, strength)
                curr_latent_keyframe.add(keyframe)
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(float_strengths).__repr__}.")    

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

    def animate(self, text, max_frames, print_output=None):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)),)




class xxStack_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "cn_stack": ("ADV_CN_STACK",),
                "controlnet": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("ADV_CN_STACK","CONTROL_NET",)
    RETURN_NAMES = ("cn_stack","controlnet")
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack"

    def controlnet_stacker(self, controlnet, image, strength, cn_stack=None, mask_optional=None,
                        latent_kf_override=None):
        controlnet_list = []

        if cn_stack is not None:
            controlnet_list.extend([cn for cn in cn_stack if cn[0] != "None"])

        if controlnet != "None" and image is not None:
            controlnet_list.append((
                controlnet,    # 第 1 个元素
                image,         # 第 2 个元素
                strength,      # 第 3 个元素
                mask_optional, # 第 6 个元素
                latent_kf_override  # 第 7 个元素
            ))
        return (controlnet_list,controlnet)



class Stack_adv_CN:

    #controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

            },
            "optional": {
                "cn_stack": ("ADV_CN_STACK",),
                "controlnet": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "image": ("IMAGE", ),
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("ADV_CN_STACK","CONTROL_NET",)
    RETURN_NAMES = ("cn_stack","controlnet")
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack"

    def controlnet_stacker(self, controlnet, image, strength, cn_stack=None, mask_optional=None,
                        latent_kf_override=None):
        check_advanced_controlnet_installed()
        controlnet_list = []

        if cn_stack is not None:
            controlnet_list.extend([cn for cn in cn_stack if cn[0] != "None"])

        if controlnet != "None" and image is not None:

            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)

            controlnet_list.append(( controlnet, image,  strength, mask_optional, latent_kf_override ))

        return (controlnet_list,controlnet)



class Apply_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "cn_stack": ("ADV_CN_STACK",),
            },
            
            }


    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Apt_Preset/stack/apply"

    def apply_controlnet(self, positive, cn_stack=None):
        
        check_advanced_controlnet_installed()
        if cn_stack is not None:
            # 创建 AdvancedControlNetApply 的实例
            for cn in cn_stack:
                # 解包 cn，确保变量数量与元组结构一致
                controlnet, image, strength, mask_optional, latent_kf_override = cn
                
                positive =  AdvancedControlNetApply.apply_controlnet(self, 
                    positive=positive, 
                    negative=None, 
                    control_net=controlnet, 
                    image=image,
                    mask_optional=mask_optional, 
                    strength=strength, 
                    start_percent=0, 
                    end_percent=1,
                    timestep_kf=None,  
                    latent_kf_override=latent_kf_override, 
                    weights_override=None,
                    control_apply_to_uncond=True
                )[0]
        return (positive,)


