
WEB_DIRECTORY = "./web"

from .CollectNode.AutomaticCFG import *
from .CollectNode.GeekyRembv2 import *
from .CollectNode.Keyframe_schedule import *
from .CollectNode.MultiAreaConditioning import *
from .CollectNode.GIMM_VFI.nodes import GIMMVFI_interpolate
from .CollectNode.layered_infinite_zoom import AD_InfiniteZoom
from .CollectNode.prompt_injection import *

from .ExpendNode.CSV_loader import *

from .Finenode.C_math import *
from .Finenode.C_model import *
from .Finenode.C_mask import *
from .Finenode.C_latent import *
from .Finenode.C_viewIO import *
from .Finenode.C_AD import *
from .Finenode.C_sample import *
from .Finenode.C_image import *
from .Finenode.C_promp import *
from .Finenode.C_utils import *
from .Finenode.C_imgEffect import *
from .Finenode.C_color import *
from .Finenode.C_condition import *
from .Finenode.C_type import *
from .Finenode.C_test import *

from .web_node.edit_mask import *
from .web_node.inpaint_cropandstitch import *
from .web_node.info import *
from .web_node.latent_Selector import *
from .web_node.image_color_analyzer import *
from .web_node.TranslationNode import *
from .web_node.presetsTXT import *
from .web_node.stack_Wildcards import *
from .web_node.Mul_Clip_adv import *
from .promptStyle.promptStyle import *

#--------------------------------------------------------------------------------------------







NODE_CLASS_MAPPINGS = {  

#-----------test-------------------







#-------------view-IO-------------------
"IO_inputAnything": C_inputAnything,
"IO_MultiOutput": C_MultiOutput,    
"IO_Textlist": Output_Textlist,
"IO_Valuelist": Output_Valuelist,
"IO_load_anyimage":C_load_anyimage,
"IO_Hub_load": Hub_load_Tool,
"view_bridge_image": CEditMask,  #wed
"view_Data": view_Data,  #wed
"view_bridge_Text": view_bridge_Text, #wed
"view_mask": CMaskPreview,
"view_latent": PreviewLatentAdvanced,
"view_combo": view_combo,



#-------------utils-------------------

"list_ListGetByIndex": ListGetByIndex,
"list_ListSlice": ListSlice,
"list_ListToBatch": ListToBatch, #wed
"list_CreateList": CreateList,#wed
"list_MergeList": MergeList, #wed   
"batch_BatchGetByIndex": BatchGetByIndex,
"batch_BatchSlice": BatchSlice,
"batch_BatchToList": BatchToList,
"batch_CreateBatch": CreateBatch,  #wed
"batch_MergeBatch": MergeBatch, #wed
"pack_Pack": Pack, #wed
"pack_Unpack": Unpack, #wed
"view_GetLength": GetLength, #wed----utils
"view_GetShape": GetShape, #wed----utils
"text_SplitString": SplitString,
#"type_AnyToDict": AnyToDict, 
"type_AnyCast": AnyCast, #wed
"type_BatchItemCast": BatchItemCast, 
"view_GetWidgetsValues": GetWidgetsValues, #wed----utils
"math_Exec": Exec,#wed----utils



#---------math------------------
"math_Float_Op": CFloatUnaryOperation,  
"math_Float_Condi": CFloatUnaryCondition,  
"math_Float_Binary_Op": CFloatBinaryOperation,  
"math_Float_Binary_Condin": CFloatBinaryCondition,  
"math_Int_Unary_Op": CIntUnaryOperation,
"math_Int_Unary_Condi": CIntUnaryCondition,
"math_Int_Binary_Op": CIntBinaryOperation,
"math_Int_Binary_Condi": CIntBinaryCondition,
"math_Remap_Data": C_Remap_DataRange,  
"math_Remap_Mask": C_Remap_MaskRange, 
"math_CreateRange": CreateRange,
"math_CreateArange": CreateArange,
"math_CreateLinspace": CreateLinspace,



#---------------model--------
"model_adjust_color": Model_adjust_color,
"model_diff_inpaint": model_diff_inpaint,
"CFG_Automatic": simpleDynamicCFG,

"Style_BatchAlign": Style_BatchAlign,




#----------------image------------------------
"pad_uv_fill": pad_uv_fill,
"pad_color_fill": pad_color_fill,
"Image_LightShape": Image_LightShape,    
"Image_Normal_light": Image_Normal_light,
"Image_keep_OneColorr": Image_keep_OneColorr,  
"Image_transform": Image_transform,    
"Image_cutResize": Image_cutResize,
"Image_Adjust": Image_Adjust,
"Image_Extract_Channel": Image_Extract_Channel,
"Image_Apply_Channel": Image_Apply_Channel,
"Image_GeekyRemB": GeekyRemB,     
"Image_edge_Splitter": Image_edge_Splitter,   
"Image_RemoveAlpha": Image_RemoveAlpha,
"image_sumTransform": image_sumTransform,
"image_selct_batch": image_selct_batch,

#-----------------mask
"Mask_ColortoMask": Mask_ColortoMask,
"Mask_Grey": Mask_Grey,
"Mask_AlphaMatte":Mask_AlphaMatte,
"Mask_Smooth": Mask_Smooth,
"Mask_lightSource": Mask_lightSource,
"Mask_math": Mask_math,
"Mask_blur_edge": Mask_blur_edge,
"Mask_Detect_label": Mask_Detect_label,
"Mask_mulcolor_img": Mask_mulcolor_img,   
"Mask_mulcolor_mask": Mask_mulcolor_mask,   
"Mask_with_depth": Mask_with_depth,
"Mask_contrast": Mask_contrast,         


#------------latent---------------------
"latent_chx_Beta": CImageNoiseBeta,
"latent_chx_Binomial": CImageNoiseBinomial,
"latent_chx_Gaussian": CImageNoiseGaussian,
"latent_ImageToNoise": CImageToNoise,
"latent_Selector": latent_Selector,



#----------prompt----------------
"text_deepseek": text_deepseek,
"text_CSV_load": text_CSV_load,
"text_saveTXT": text_saveTXT,
"text_mul_replace": text_mul_replace,
"text_mul_remove": text_mul_remove,
"text_free_wildcards": text_free_wildcards,
"text_randomTXT": text_randomTXT,
"text_style": text_style,
"text_SuperPrompter": text_SuperPrompter,
"text_presetsTXT": text_presetsTXT,
"text_cameraPrompt": text_cameraPrompt,
"text_Trim_Tokens": text_Trim_Tokens,
"text_selectOutput": text_selectOutput,


"PromptInjection": PromptInjection,
"PromptInjectionIdx": PromptInjectionIdx,
"SimplePromptInjection": SimplePromptInjection,
"AdvancedPromptInjection": AdvancedPromptInjection,


"stack_Wildcards": stack_Wildcards,
"stack_text_combine": stack_text_combine,
"Stack_LoRA2": Stack_LoRA2,


"ChineseToEnglish": ChineseToEnglish,
"EnglishToChinese": EnglishToChinese,
"Cutoff_BasePrompt": CLIPRegionsBasePrompt,
"Cutoff_SetRegions": CLIPSetRegion,
"Cutoff_RegionsToConditioning": CLIPRegionsToConditioning,
"Cutoff_RegionsToConditioning_ADV": CLIPRegionsToConditioningADV,



#---------condition----------------
"condi_IPAdapterSD3": condi_IPAdapterSD3,
"condi_ReduxAdvanced": condi_ReduxAdvanced,
"condi_mask_Mulcondi": mask_Mulcondition,
"condi_pos_neg": condi_pos_neg,    
"condi_Mul_Clip": condi_Mul_Clip,   
"condi_Mul_adv": condi_Mul_adv,   

"condi_Mul_controlnet": condi_Mul_controlnet,
"condi_Mul_lora": condi_Mul_lora,
"MultiAreaConditioning": MultiAreaConditioning,  #wed




#-------sample----------------------------------------
"InpaintCrop": InpaintCrop,  #wed
"InpaintStitch": InpaintStitch,  #wed
"C_DynamicTileSplit": DynamicTileSplit, 
"C_DynamicTileMerge": DynamicTileMerge,
"C_xy_Tiling_KSampler": xy_Tiling_KSampler,
"C-SeamCarving": SeamCarving,
"C-SeamlessTile": SeamlessTile,
"C-CircularVAEDecode": CircularVAEDecode,
"C-MakeCircularVAE": MakeCircularVAE,
"C-OffsetImage": OffsetImage,
"C_local_sample": C_local_sample,





#---------AD--------------------
"AD_sch_prompt": AD_sch_prompt,
"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_Dynamic_MASK": AD_Dynamic_MASK,

"AD_DrawSchedule": AD_DrawSchedule,
"AD_schedule_MulCondi": AD_schedule_MulCondi,
"AD_slice_Condi": AD_slice_Condi,
"GIMMVFI_interpolate": GIMMVFI_interpolate,
"AD_InfiniteZoom": AD_InfiniteZoom,


#-----------------Color----------------
"color_Flux_Palette": Flux_ColorPalette,
"color_Picker": ColorPicker,
"color_to RGB": ColorToRGB,
"color_to Hex": ColorToHex,
"color_image_Replace": ImageReplaceColor,
"color_Invert": InvertColor,
"color_Analyzer_Image": Image_Color_Analyzer,




#-----------------type----------------
"type_make_imagesBatch": type_make_imagesBatch,
"type_make_maskBatch": type_make_maskBatch,
"type_make_condition": type_make_condition,
"type_Anyswitch": type_Anyswitch,
"type_BasiPIPE": type_BasiPIPE,
"type_Image_List2Batch":type_Image_List2Batch,
"type_Image_Batch2List":type_Image_Batch2List,
"type_Mask_Batch2List":type_Mask_Batch2List,
"type_Mask_List2Batch":type_Mask_List2Batch,
"type_text_list2batch ": type_text_list2batch ,  

#----------------imgEffect--------
"img_Loadeffect": img_Loadeffect,
"img_Upscaletile": img_Upscaletile,
"img_Remove_bg": img_Remove_bg,
"img_gen_abstract": img_gen_abstract,
"img_gen_fractal":img_gen_fractal,

}



NODE_DISPLAY_NAME_MAPPINGS = {    }


