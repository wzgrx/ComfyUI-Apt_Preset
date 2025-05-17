WEB_DIRECTORY = "./web"
#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']




from .NodeChx.main_nodes import *
from .NodeChx.main_stack import *

from .NodeChx.AdvancedCN import AD_sch_adv_CN,AD_sch_latent
from .NodeChx.IPAdapterPlus import chx_IPA_basic,chx_IPA_faceID,chx_IPA_XL_adv,chx_IPA_region_combine, chx_IPA_apply_combine
from .NodeChx.style_node import *

from .NodeChx.video_node import *
from .NodeChx.Ksampler_all import *
from .NodeChx.Keyframe_schedule import *

from .NodeBasic.C_packdata import *

from .NodeBasic.C_model import *
from .NodeBasic.C_mask import *
from .NodeBasic.C_latent import *
from .NodeBasic.C_viewIO import *
from .NodeBasic.C_AD import *

from .NodeBasic.C_promp import *
from .NodeBasic.C_imgEffect import *

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
"load_create_chx": load_create_chx,
"load_create_basic_chx": load_create_basic_chx,

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
"IPA_dapterSD3LOAD":IPA_dapterSD3LOAD,
"chx_IPA_basic": chx_IPA_basic,
"chx_IPA_faceID": chx_IPA_faceID,
"chx_IPA_XL_adv": chx_IPA_XL_adv,
"chx_IPA_region_combine": chx_IPA_region_combine,
"chx_IPA_apply_combine": chx_IPA_apply_combine,
"chx_StyleModelApply":chx_StyleModelApply,
"chx_Style_Redux":chx_Style_Redux,
"chx_YC_LG_Redux": chx_YC_LG_Redux,


#-sample-------------------------------------------#
"pre_sample_data":  pre_sample_data,  
"pre_inpaint_xl": pre_inpaint_xl,


"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_mid": basic_Ksampler_mid,    
"basic_Ksampler_full": basic_Ksampler_full,     
"basic_Ksampler_custom": basic_Ksampler_custom,
"basic_Ksampler_adv": basic_Ksampler_adv,
"basic_Ksampler_batch": basic_Ksampler_batch,

"chx_Ksampler_Relight": chx_Ksampler_Relight,
"chx_Ksampler_texture": chx_Ksampler_texture,
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_Ksampler_refine": chx_Ksampler_refine,
"chx_Ksampler_dual_paint": chx_Ksampler_dual_paint,
"chx_Ksampler_dual_area": chx_Ksampler_dual_area,
"chx_Ksampler_VisualStyle": chx_Ksampler_VisualStyle,  
"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,

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
"Apply_IPA_SD3":Apply_IPA_SD3,
"Apply_adv_CN": Apply_adv_CN,
"Apply_AD_diff": Apply_AD_diff,
"Apply_condiStack": Apply_condiStack,
"Apply_textStack": Apply_textStack,
"Apply_Redux": Apply_Redux,
"Apply_latent": Apply_latent,
"Apply_CN_union":Apply_CN_union,
"Apply_ControlNetStack": Apply_ControlNetStack,
"Apply_LoRAStack": Apply_LoRAStack,



"Stack_IPA_SD3":Stack_IPA_SD3,
"Stack_latent": Stack_latent,
"Stack_Redux":Stack_Redux,
"stack_AD_diff": stack_AD_diff,
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
#"AD_sch_prompt_basic": AD_sch_prompt_basic,
#"AD_sch_prompt_apply": AD_sch_prompt_apply,
#"AD_sch_prompt_preset": AD_sch_prompt_preset,
#"AD_sch_prompt_chx":AD_sch_prompt_chx,


"AD_sch_prompt_adv": AD_sch_prompt_adv,
"AD_sch_prompt_stack": AD_sch_prompt_stack,
"AD_sch_IPA": AD_sch_IPA,
"AD_sch_latent": AD_sch_latent,
"AD_sch_mask":AD_sch_mask,
"AD_sch_value": AD_sch_value,
"AD_sch_adv_CN":AD_sch_adv_CN,

"AD_sch_image_merge":AD_sch_image_merge,
"AD_DrawSchedule": AD_DrawSchedule,
"AD_slice_Condi": AD_slice_Condi,
"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_batch_replace": AD_batch_replace,
"AD_pingpong_vedio":AD_pingpong_vedio,

"Amp_drive_value": Amp_drive_value,
"Amp_drive_String": Amp_drive_String,
"Amp_audio_Normalized": Amp_audio_Normalized,
"Amp_drive_mask": Amp_drive_mask,




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
"IO_save_image": IO_save_image,  




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
"view_Mask_And_Img": view_Mask_And_Img,







#---------------model--------
"model_adjust_color": model_adjust_color,
"model_diff_inpaint": model_diff_inpaint,
"model_Regional": model_Regional,
"model_Style_Align":model_Style_Align,

#----------------image------------------------



#----------------imgEffect--------


"create_lineGradient": create_lineGradient,
"create_RadialGradient": create_RadialGradient,
"create_overShape": create_overShape,
"create_AD_mask": create_AD_mask,
"create_mulcolor_img": create_mulcolor_img,   
"create_mulcolor_mask": create_mulcolor_mask,  


"img_effect_Load": img_effect_Load,
"img_effect_CircleWarp": img_effect_CircleWarp,
"img_effect_Stretch": img_effect_Stretch,
"img_effect_WaveWarp": img_effect_WaveWarp,
"img_effect_Liquify": img_effect_Liquify,

"lay_texture_Offset": lay_texture_Offset,
"lay_ImageGrid": lay_ImageGrid,
"lay_MaskGrid": lay_MaskGrid,
"lay_image_match_W_or_H": lay_image_match_W_or_H,
"lay_image_match_W_and_H": lay_image_match_W_and_H,
"lay_edge_cut": lay_edge_cut,   
"lay_text":lay_text,

"lay_fill_inpaint":lay_fill_inpaint,








#-----------------mask----------------------


"Mask_math": Mask_math,
"Mask_Detect_label": Mask_Detect_label,
"Mask_Remove_bg": Mask_Remove_bg,
"Mask_transform": Mask_transform,
"Mask_image2mask": Mask_image2mask,
"Mask_mask2mask": Mask_mask2mask,
"Mask_splitMask": Mask_splitMask,
"Mask_inpaint_light": Mask_inpaint_light,


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
"text_sum ": text_sum,  #wed
"text_manual" :text_manual, #wed



#---------Gpt modle---------------
"GPT_ChineseToEnglish": GPT_ChineseToEnglish,
"GPT_EnglishToChinese": GPT_EnglishToChinese,
"GPT_deepseek_api_text": GPT_deepseek_api_text,
"GPT_Janus_img_2_text": GPT_Janus_img_2_text,
"GPT_Janus_generate_img": GPT_Janus_generate_img,
"GPT_MiniCPM": GPT_MiniCPM,








}








NODE_DISPLAY_NAME_MAPPINGS = {

"chx_ksampler_tile": "chx_ksampler_tile(sole)",   
"chx_Ksampler_inpaint": "chx_Ksampler_inpaint(sole)",

"chx_IPA_basic": "IPA_basic",
"chx_IPA_faceID": "IPA_faceID",
"chx_IPA_XL_adv": "IPA_XL_adv",


}


