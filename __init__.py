WEB_DIRECTORY = "./web"
#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


from .NodeChx.main_nodes import *
from .NodeChx.main_stack import *

from .NodeChx.AdvancedCN import AD_sch_adv_CN,AD_sch_latent
from .NodeChx.IPAdapterPlus import chx_IPA_basic,chx_IPA_faceID,chx_IPA_XL_adv,chx_IPA_region_combine, chx_IPA_apply_combine
from .NodeChx.IPAdapterSD3 import IPA_dapterSD3LOAD,Stack_IPA_SD3,Apply_IPA_SD3
from .NodeChx.video_node import *
from .NodeChx.Ksampler_all import *


from .NodeBasic.c_packdata import *
from .NodeBasic.C_math import *
from .NodeBasic.C_model import *
from .NodeBasic.C_mask import *
from .NodeBasic.C_latent import *
from .NodeBasic.C_viewIO import *
from .NodeBasic.C_AD import *
from .NodeBasic.C_image import *
from .NodeBasic.C_promp import *
from .NodeBasic.C_imgEffect import *
from .NodeBasic.C_type import *
from .NodeBasic.C_layout import *
from .NodeBasic.C_GPT import *

from .NodeBasic.C_test import *





NODE_CLASS_MAPPINGS = {





#-sum-------------------------------------------#
"sum_load": sum_load,
"sum_editor": sum_editor,
"sum_latent": sum_latent,
"sum_lora": sum_lora,
"sum_controlnet": sum_controlnet,
"sum_text": sum_text,   
"sum_stack_AD": sum_stack_AD,
"sum_stack_image": sum_stack_image,
"sum_stack_all": sum_stack_all,
"sum_stack_Wan": sum_stack_Wan,

"load_FLUX": load_FLUX,
"load_basic": load_basic,
"load_SD35": load_SD35,
"load_only_clip": load_only_clip,
"load_clip": load_clip,

"Data_chx_Merge":Data_chx_Merge,
"Data_presetData":Data_presetData,
"Data_basic": Data_basic,
"Data_basic_easy": Data_basic_easy,
"Data_sample": Data_sample,
"Data_select": Data_select,
"Data_preset_save": Data_preset_save,



#---------tool------------------------------------------#

"chx_re_fluxguide": chx_re_fluxguide,
"chx_vae_encode": chx_vae_encode,
"chx_vae_encode_tile": chx_vae_encode_tile,
"chx_controlnet": chx_controlnet, 
"chx_controlnet_union": chx_controlnet_union,
"chx_mask_Mulcondi": mask_Mulcondition,
"chx_Upscale_simple": chx_Upscale_simple,




#-IPA-------------------------------------------#
"chx_IPA_basic": chx_IPA_basic,
"chx_IPA_faceID": chx_IPA_faceID,
"chx_IPA_XL_adv": chx_IPA_XL_adv,
"IPA_dapterSD3LOAD":IPA_dapterSD3LOAD,
"chx_IPA_region_combine": chx_IPA_region_combine,
"chx_IPA_apply_combine": chx_IPA_apply_combine,
"chx_YC_LG_Redux": chx_YC_LG_Redux,
"chx_Style_Redux": chx_Style_Redux,
"chx_StyleModelApply":chx_StyleModelApply,



#-sample-------------------------------------------#
"pre_sample_data":  pre_sample_data,  
"pre_make_context": pre_make_context,
"pre_inpaint_xl": pre_inpaint_xl,


"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_mid": basic_Ksampler_mid,    
"basic_Ksampler_full": basic_Ksampler_full,     
"basic_Ksampler_custom": basic_Ksampler_custom,
"basic_Ksampler_adv": basic_Ksampler_adv,
"basic_Ksampler_batch": basic_Ksampler_batch,

"chx_Ksampler_texture": chx_Ksampler_texture,
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_Ksampler_refine": chx_Ksampler_refine,
"chx_Ksampler_dual_paint": chx_Ksampler_dual_paint,
"chx_Ksampler_dual_area": chx_Ksampler_dual_area,
"chx_Ksampler_VisualStyle": chx_Ksampler_VisualStyle,  
"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,
"chx_ksampler_Deforum": chx_ksampler_Deforum,
"chx_ksampler_Deforum_math": chx_ksampler_Deforum_math,
"chx_ksampler_Deforum_sch":chx_ksampler_Deforum_sch,

"sampler_InpaintCrop": InpaintCrop,  #wed
"sampler_InpaintStitch": InpaintStitch,  #wed
"sampler_DynamicTileSplit": DynamicTileSplit, 
"sampler_DynamicTileMerge": DynamicTileMerge,
"sampler_enhance": sampler_enhance,
"sampler_sigmas": sampler_sigmas,





#-stack-------------------------------------------#
"Apply_IPA": Apply_IPA,
"Apply_prompt_Schedule": Apply_prompt_Schedule,
"Apply_adv_CN": Apply_adv_CN,
"Apply_AD_diff": Apply_AD_diff,
"Apply_condiStack": Apply_condiStack,
"Apply_textStack": Apply_textStack,
"Apply_Redux": Apply_Redux,
"Apply_latent": Apply_latent,
"Apply_CN_union":Apply_CN_union,
"Apply_ControlNetStack": Apply_ControlNetStack,
"Apply_LoRAStack": Apply_LoRAStack,
"Apply_IPA_SD3":Apply_IPA_SD3,


"Stack_IPA_SD3":Stack_IPA_SD3,
"Stack_latent": Stack_latent,
"Stack_Redux":Stack_Redux,
"stack_AD_diff": stack_AD_diff,
"Stack_adv_CN_easy": Stack_adv_CN_easy,
"Stack_adv_CN": Stack_adv_CN,
"Stack_IPA": Stack_IPA,
"Stack_text": Stack_text,
"Stack_condi": Stack_condi,
"Stack_LoRA": Stack_LoRA,
"Stack_ControlNet": Stack_ControlNet,
"Stack_ControlNet1":Stack_ControlNet1,
"Stack_CN_union":Stack_CN_union,
"stack_sum_pack": stack_sum_pack,


"Stack_WanImageToVideo": Stack_WanImageToVideo,
"Stack_WanFunControlToVideo": Stack_WanFunControlToVideo,
"Stack_WanFirstLastFrameToVideo": Stack_WanFirstLastFrameToVideo,
"Stack_WanFunInpaintToVideo": Stack_WanFunInpaintToVideo,




#--------AD------------------------------------------

"AD_stack_prompt" : AD_stack_prompt,
"AD_sch_IPA": AD_sch_IPA,
"AD_sch_latent": AD_sch_latent,
"AD_sch_adv_CN":AD_sch_adv_CN,

"AD_sch_prompt_chx":AD_sch_prompt_chx,
"AD_sch_prompt_preset": AD_sch_prompt_preset,
"AD_sch_mask":AD_sch_mask,
"AD_sch_value": AD_sch_value,
"Amp_drive_value": Amp_drive_value,
"Amp_drive_String": Amp_drive_String,
"Amp_audio_Normalized": Amp_audio_Normalized,
"Amp_drive_mask": Amp_drive_mask,

"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_Dynamic_MASK": AD_Dynamic_MASK,
"AD_batch_replace": AD_batch_replace,




#--unpack------------------------------------------#
"param_preset_pack": param_preset,
"param_preset_Unpack": Unpack_param,
"Model_Preset_pack":Model_Preset,
"Model_Preset_Unpack": Unpack_Model,

"CN_preset1_pack": CN_preset1,
"CN_preset1_Unpack": Unpack_CN,
"photoshop_preset_pack":photoshop_preset,
"photoshop_preset_Unpack":Unpack_photoshop,




#-------------view-IO-------------------
"IO_inputbasic": IO_inputbasic,
"IO_load_anyimage":IO_load_anyimage,
"IO_clip_vision": IO_clip_vision,
"IO_clear_cache": IO_clear_cache,


"view_Data": view_Data,  #wed
"view_bridge_image": view_bridge_image,  #wed
"view_bridge_Text": view_bridge_Text, #wed
"view_mask": view_mask,
"view_latent": view_LatentAdvanced,
"view_combo": view_combo,#wed
"view_node_Script": view_node_Script,
"view_GetLength": view_GetLength, #wed----utils
"view_GetShape": view_GetShape, #wed----utils
"view_GetWidgetsValues": view_GetWidgetsValues, #wed----utils




#-------------math-------------------

"list_num_range": list_num_range,
"list_cycler_Value":list_cycler_Value,
"list_input_text": list_input_text,
"list_input_Value": list_input_Value,
"list_ListGetByIndex": ListGetByIndex,
"list_ListSlice": ListSlice,
"list_MergeList": MergeList, #wed   

"batch_cycler_Prompt":batch_cycler_Prompt,
"batch_cycler_Value":batch_cycler_Value,
"batch_cycler_text" :batch_cycler_text,
"batch_cycler_split_text":batch_cycler_split_text,
"batch_cycler_image":batch_cycler_image,
"batch_cycler_mask":batch_cycler_mask,


"Remap_basic_data": Remap_basic_data,  
"Remap_mask": Remap_mask,
"math_BinaryOperation": math_BinaryOperation,  
"math_BinaryCondition": math_BinaryCondition,
"math_UnaryOperation": math_UnaryOperation,
"math_UnaryCondition": math_UnaryCondition,
"math_Exec": math_Exec,#wed----utils





#-----------math---type-------------------
"pack_Pack": Pack, #wed
"pack_Unpack": Unpack, #wed

"creat_mask_batch": creat_mask_batch, #wed
"creat_mask_batch_input": creat_mask_batch_input,#wed
"creat_image_batch": creat_image_batch, #wed
"creat_image_batch_input": creat_image_batch_input,#wed
"creat_any_List": creat_any_List,#wed


"type_AnyCast": AnyCast, #wed
"type_Anyswitch": type_Anyswitch,
"type_BasiPIPE": type_BasiPIPE,
"type_Image_List2Batch":type_Image_List2Batch,
"type_Image_Batch2List":type_Image_Batch2List,
"type_Mask_Batch2List":type_Mask_Batch2List,
"type_Mask_List2Batch":type_Mask_List2Batch,
"type_text_list2batch ": type_text_list2batch ,  
"type_text_2_UTF8": type_text_2_UTF8 ,  




#---------------model--------
"model_adjust_color": Model_adjust_color,
"model_diff_inpaint": model_diff_inpaint,
"model_Regional": model_Regional,


#----------------image------------------------
"pad_uv_fill": pad_uv_fill,
"pad_color_fill": pad_color_fill,
"Image_LightShape": Image_LightShape,    
"Image_Normal_light": Image_Normal_light,
"Image_keep_OneColorr": Image_keep_OneColorr,  
"Image_transform": Image_transform,    
"Image_cutResize": Image_cutResize,
"Image_Resize": Image_Resize,
"Image_scale_adjust": Image_scale_adjust,


"image_sumTransform": image_sumTransform,
"Image_overlay": Image_overlay,
"Image_overlay_mask": Image_overlay_mask,
"Image_overlay_composite": Image_overlay_composite,
"Image_overlay_transform": Image_overlay_transform,
"Image_overlay_sum": Image_overlay_sum,
"Image_Extract_Channel": Image_Extract_Channel,
"Image_Apply_Channel": Image_Apply_Channel,
"Image_RemoveAlpha": Image_RemoveAlpha,
"image_selct_batch": image_selct_batch,
"Image_scale_match": Image_scale_match,




#-----------------mask----------------------
"Mask_AD_generate": Mask_AD_generate,
"Mask_inpaint_Grey": Mask_inpaint_Grey,
"Mask_math": Mask_math,
"Mask_Detect_label": Mask_Detect_label,
"Mask_mulcolor_img": Mask_mulcolor_img,   
"Mask_mulcolor_mask": Mask_mulcolor_mask,   
"Mask_Outline": Mask_Outline,

"Mask_Smooth": Mask_Smooth,
"Mask_Offset": Mask_Offset,
"Mask_cut_mask": Mask_cut_mask,
"Mask_image2mask": Mask_image2mask,
"Mask_mask2mask": Mask_mask2mask,
"Mask_mask2img": Mask_mask2img,
"Mask_splitMask": Mask_splitMask,



#------------latent---------------------
"latent_chx_noise": latent_chx_noise,
"latent_Image2Noise": latent_Image2Noise,
"latent_ratio": latent_ratio,
"latent_mask":latent_mask,


#----------prompt----------------
"text_CSV_load": text_CSV_load,
"text_SuperPrompter": text_SuperPrompter,
"text_mul_replace": text_mul_replace,
"text_mul_remove": text_mul_remove,
"text_free_wildcards": text_free_wildcards,
"text_stack_wildcards": text_stack_wildcards,





#---------Gpt modle---------------
"GPT_ChineseToEnglish": GPT_ChineseToEnglish,
"GPT_EnglishToChinese": GPT_EnglishToChinese,

"GPT_deepseek_api_text": GPT_deepseek_api_text,
"GPT_Janus_img_2_text": GPT_Janus_img_2_text,
"GPT_Janus_generate_img": GPT_Janus_generate_img,
"GPT_MiniCPM": GPT_MiniCPM,







#----------------imgEffect--------
"img_Loadeffect": img_Loadeffect,
"img_Upscaletile": img_Upscaletile,
"img_Remove_bg": img_Remove_bg,
"img_CircleWarp": img_CircleWarp,
"img_Stretch": img_Stretch,
"img_WaveWarp": img_WaveWarp,
"img_Liquify": img_Liquify,
"img_Seam_adjust_size": img_Seam_adjust_size,
"img_texture_Offset": img_texture_Offset,
"img_White_balance":img_White_balance,  #白平衡重定向
"img_HDR": img_HDR,


"color_adjust": color_adjust,
"color_Match": color_Match,
"color_match_adv":color_match_adv,
"color_input": color_input,
"color_color2hex":color_color2hex,
"color_hex2color":color_hex2color,
"color_image_Replace": ImageReplaceColor,
"color_pure_img": color_pure_img,
"color_Gradient": color_Gradient,
"color_RadialGradient": color_RadialGradient,




#-----------------layout----------------
"lay_ImageGrid": lay_ImageGrid,
"lay_MaskGrid": lay_MaskGrid,
"lay_image_match_W_or_H": lay_image_match_W_or_H,
"lay_image_match_W_and_H": lay_image_match_W_and_H,
"lay_edge_cut": lay_edge_cut,   
"lay_text":lay_text,





}








NODE_DISPLAY_NAME_MAPPINGS = {

"chx_ksampler_tile": "chx_ksampler_tile(sole)",   
"chx_Ksampler_inpaint": "chx_Ksampler_inpaint(sole)",

"chx_IPA_basic": "IPA_basic",
"chx_IPA_faceID": "IPA_faceID",
"chx_IPA_XL_adv": "IPA_XL_adv",


}


