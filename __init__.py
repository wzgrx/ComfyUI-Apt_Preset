WEB_DIRECTORY = "./web"
#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

from .NodeChx.main_nodes import *
from .NodeChx.main_stack import *
from .NodeChx.AdvancedCN import AD_sch_latent
from .NodeChx.IPAdapterPlus import chx_IPA_basic,chx_IPA_faceID,chx_IPA_XL,chx_IPA_region_combine, chx_IPA_apply_combine,chx_IPA_adv,chx_IPA_faceID_adv
from .NodeChx.style_node import *
from .NodeChx.video_node import *
from .NodeChx.Ksampler_all import *
from .NodeChx.Keyframe_schedule import *

from .NodeBasic.highway import Data_Highway
from .NodeBasic.C_packdata import *
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

from .NodeBasic.C_test import *
from .NodeCollect.mask_face import *
from .NodeCollect.text_font2img import *
from .NodeCollect.mask_mutil_lay import *
from .NodeChx.sum_text_yaml import *




#-load------------------------------------------#

NODE_CLASS_MAPPINGS= {

#"sum_text": sum_text,  
# "sum_load_only_model": sum_load_only_model, 

"sum_load_adv": sum_load_adv,
"sum_create_chx": sum_create_chx,
"sum_editor": sum_editor,
"load_FLUX": load_FLUX,
"load_basic": load_basic,
"load_SD35": load_SD35,
"load_GGUF": UnetLoaderGGUF2,

#"Data_basic_easy": Data_basic_easy,
"Data_Highway":Data_Highway,
"Data_bus_chx":Data_bus_chx,
"Data_basic": Data_basic,
"Data_select": Data_select,
"Data_chx_Merge":Data_chx_Merge,
"Data_sampleData": Data_sampleData,
"Data_presetData":Data_presetData,
"Data_preset_save": Data_preset_save,


#-sample-------------------------------------------#

"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_mid": basic_Ksampler_mid,    
"basic_Ksampler_full": basic_Ksampler_full,     
"basic_Ksampler_custom": basic_Ksampler_custom,
"basic_Ksampler_adv": basic_Ksampler_adv,
"basic_Ksampler_batch": basic_Ksampler_batch,
"chx_Ksampler_refine": chx_Ksampler_refine,
"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,
"chx_Ksampler_texture": chx_Ksampler_texture,
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_Ksampler_Kontext": chx_Ksampler_Kontext,

"chx_Ksampler_dual_paint": chx_Ksampler_dual_paint,
"chx_Ksampler_dual_area": chx_Ksampler_dual_area,
"chx_Ksampler_VisualStyle": chx_Ksampler_VisualStyle,  
"chx_ksampler_Deforum_math": chx_ksampler_Deforum_math,
"chx_ksampler_Deforum_sch":chx_ksampler_Deforum_sch,


"sampler_InpaintCrop": InpaintCrop,  #wed
"sampler_InpaintStitch": InpaintStitch,  #wed
"sampler_DynamicTileSplit": DynamicTileSplit, 
"sampler_DynamicTileMerge": DynamicTileMerge,
"sampler_enhance": sampler_enhance,


#---------control tool------------------------------------------#
#"chx_controlnet": chx_controlnet, 

"sum_latent": sum_latent,
"sum_lora": sum_lora,
"sum_stack_all": sum_stack_all,
"sum_stack_image": sum_stack_image,
"sum_stack_Wan": sum_stack_Wan,
"sum_stack_AD": sum_stack_AD,



"pre_Kontext": pre_Kontext, 
"pre_sample_data": pre_sample_data,
"pre_controlnet": pre_controlnet,
"pre_controlnet_union": pre_controlnet_union,
"pre_mask_Mulcondi": pre_mask_Mulcondi,
"pre_ic_light_sd15":  pre_ic_light_sd15,
"pre_latent_light": pre_latent_light,
"pre_Flex2": pre_Flex2,



#-IPA-------------------------------------------#

"chx_IPA_basic": chx_IPA_basic,
"chx_IPA_faceID": chx_IPA_faceID,
"chx_IPA_faceID_adv": chx_IPA_faceID_adv,
"chx_IPA_XL": chx_IPA_XL,
"chx_IPA_adv":chx_IPA_adv,
"chx_IPA_region_combine": chx_IPA_region_combine,
"chx_IPA_apply_combine": chx_IPA_apply_combine,
"chx_StyleModelApply":chx_StyleModelApply,
"chx_Style_Redux":chx_Style_Redux,
"chx_YC_LG_Redux": chx_YC_LG_Redux,

"IPA_dapterSD3LOAD":IPA_dapterSD3LOAD,
"IPA_XL_PromptInjection": PromptInjection,
"IPA_clip_vision": IPA_clip_vision,



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
"Stack_WanVaceToVideo_mul": stack_WanVaceToVideo_mul,
"Stack_WanCameralToVideo": stack_WanCameraImageToVideo,

#--------AD------------------------------------------

#"AD_sch_prompt_preset": AD_sch_prompt_preset,
#"AD_sch_adv_CN":AD_sch_adv_CN,
"AD_sch_IPA": AD_sch_IPA,
"AD_sch_prompt_adv": AD_sch_prompt_adv,
"AD_sch_prompt_stack": AD_sch_prompt_stack,
"AD_sch_latent": AD_sch_latent,

"AD_sch_image_merge":AD_sch_image_merge,
"AD_DrawSchedule": AD_DrawSchedule,
"AD_slice_Condi": AD_slice_Condi,
"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_batch_replace": AD_batch_replace,
"AD_pingpong_vedio":AD_pingpong_vedio,
"AD_font2img":AD_font2img,

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
"unpack_box2":unpack_box2,



#-------------view-IO-------------------
"IO_input_any": IO_input_any,
"IO_inputbasic": IO_inputbasic,
"IO_load_anyimage":IO_load_anyimage,
"IO_adjust_image": IO_adjust_image,
"IO_clear_cache": IO_clear_cache,
"IO_save_image": IO_save_image,  
"IO_video_encode": IO_video_encode,


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


#-------------data-------------------

"AD_sch_prompt_basic": AD_sch_prompt_basic,
"AD_sch_mask":AD_sch_mask,
"AD_sch_value": AD_sch_value,

"list_ListGetByIndex": list_GetByIndex,
"list_ListSlice": list_Slice,
"list_MergeList": list_Merge, #wed   
"list_num_range": list_num_range,
"list_cycler_Value":list_cycler_Value,

"batch_cycler_Prompt":batch_cycler_Prompt,
"batch_cycler_Value":batch_cycler_Value,
"batch_cycler_text" :batch_cycler_text,
"batch_cycler_split_text":batch_cycler_split_text,
"batch_cycler_image":batch_cycler_image,
"batch_cycler_mask":batch_cycler_mask,

"math_Remap_data": math_Remap_data,  
"math_calculate": math_calculate, 

#-----------math---type-------------------
"pack_Pack": Pack, #wed
"pack_Unpack": Unpack, #wed
"creat_mask_batch": creat_mask_batch, #wed
"creat_mask_batch_input": creat_mask_batch_input,#wed
"creat_image_batch": creat_image_batch, #wed
"creat_image_batch_input": creat_image_batch_input,#wed
"creat_any_List": creat_any_List,#wed

"type_AnyCast": type_AnyCast, #wed
"type_Anyswitch": type_Anyswitch,
"type_BasiPIPE": type_BasiPIPE,
"type_Image_List2Batch":type_Image_List2Batch,
"type_Image_Batch2List":type_Image_Batch2List,
"type_Mask_Batch2List":type_Mask_Batch2List,
"type_Mask_List2Batch":type_Mask_List2Batch,
"type_text_list2batch ": type_text_list2batch ,  
"type_text_2_UTF8": type_text_2_UTF8 ,  


#---------------model--------
"model_adjust_color": model_adjust_color,
"model_diff_inpaint": model_diff_inpaint,
"model_Regional": model_Regional,
"model_Style_Align":model_Style_Align,


#----------------image------------------------

"Image_batch_select": Image_batch_select,
"Image_batch_composite": Image_batch_composite,
"Image_transform_batch": Image_transform_batch,
"Image_transform_sum": Image_transform_sum, 
"Image_pad_outfill": Image_pad_outfill,
"Image_pad_overlay": Image_pad_overlay,
"Image_Resize_sum": Image_Resize_sum,
"Image_Resize2": Image_Resize2,
"Image_Upscaletile": Image_Upscaletile,
"Image_crop_box2": Image_crop_box2,

"Image_Channel_Extract": Image_Channel_Extract,
"Image_Channel_Apply": Image_Channel_Apply,
"Image_Channel_RemoveAlpha": Image_Channel_RemoveAlpha,


"color_tool": color_tool,
"color_adjust_HDR": color_adjust_HDR,
"color_adjust_light": color_adjust_light,
"color_adjust_HSL": color_adjust_HSL,
"color_adjust_WB_balance":color_adjust_WB_balance,  #白平衡重定向
"color_match_adv":color_match_adv,
"color_OneColor_replace": color_OneColor_replace,
"color_OneColor_keep": color_OneColor_keep,     



#----------------imgEffect--------


"create_lineGradient": create_lineGradient,
"create_RadialGradient": create_RadialGradient,
"create_overShape": create_overShape,
"create_AD_mask": create_AD_mask,
"create_mulcolor_img": create_mulcolor_img,   
"create_mulcolor_mask": create_mulcolor_mask,  
"create_Mask_sole": create_Mask_sole,
"create_Mask_lay_X": create_Mask_lay_X,
"create_Mask_lay_Y": create_Mask_lay_Y,
"create_Mask_Rectangles": create_Mask_Rectangles,

"img_effect_Load": img_effect_Load,
"img_effect_CircleWarp": img_effect_CircleWarp,
"img_effect_Stretch": img_effect_Stretch,
"img_effect_WaveWarp": img_effect_WaveWarp,
"img_effect_Liquify": img_effect_Liquify,

"lay_texture_Offset": lay_texture_Offset,
"lay_ImageGrid": lay_ImageGrid,
"lay_MaskGrid": lay_MaskGrid,
"lay_image_match_W_and_H": lay_image_match_W_and_H,
"lay_edge_cut": lay_edge_cut,   
"lay_text_sum":lay_text_sum,
"lay_fill_inpaint":lay_fill_inpaint,
"lay_compare_img": lay_compare_img,
"lay_images_free_layout":lay_images_free_layout,
"lay_image_grid_note": lay_image_grid_note,


#-----------------mask----------------------

"Mask_combine_sum": Mask_combine_sum,
"Mask_combine_crop": Mask_combine_crop,
"Mask_math": Mask_math,
"Mask_Detect_label": Mask_Detect_label,
"Mask_Remove_bg": Mask_Remove_bg,
"Mask_transform": Mask_transform,
"Mask_image2mask": Mask_image2mask,
"Mask_mask2mask": Mask_mask2mask,
"Mask_splitMask": Mask_splitMask,
"Mask_inpaint_light": Mask_inpaint_light,
"Mask_face_detect": Mask_face_detect,



#------------latent---------------------
#"chx_latent_blend": chx_latent_blend, 
"latent_chx_noise": latent_chx_noise,
"latent_Image2Noise": latent_Image2Noise,
"latent_ratio": latent_ratio,
"chx_latent_adjust": chx_latent_adjust,



#----------prompt----------------

"excel_read":excel_read,
"excel_write_data":excel_write_data,
"excel_insert_image":excel_insert_image,
"excel_search_data":excel_search_data,
"excel_row_diff":excel_row_diff,

"text_mul_Split":text_mul_Split,
"text_mul_Join":text_mul_Join,
"text_mul_replace": text_mul_replace,
"text_mul_remove": text_mul_remove,
"text_CSV_load": text_CSV_load,
"text_SuperPrompter": text_SuperPrompter,
"text_free_wildcards": text_free_wildcards,
"text_stack_wildcards": text_stack_wildcards,
"text_sum": NS_PromptList,#web


}








NODE_DISPLAY_NAME_MAPPINGS = {

"chx_ksampler_tile": "chx_ksampler_tile(sole)",   
"chx_Ksampler_inpaint": "chx_Ksampler_inpaint(sole)",

}
