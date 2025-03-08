WEB_DIRECTORY = "./web"
#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


from .nodes import *
from .packdata import *






NODE_CLASS_MAPPINGS = {





#--------AD------------------------------------------


#-sum-------------------------------------------#
"sum_load": sum_load,
"sum_editor": sum_editor,
"sum_latent": sum_latent,
"sum_lora": sum_lora,
"sum_controlnet": sum_controlnet,

"load_FLUX": load_FLUX,
"load_basic": load_basic,
"load_SD35": load_SD35,

#-pre_sample-------------------------------------------#
"pre_sample_data":  pre_sample_data,  
"pre_make_context": pre_make_context,




#-sample-------------------------------------------#

"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_mid": basic_Ksampler_mid,    
"basic_Ksampler_full": basic_Ksampler_full,     
"basic_Ksampler_custom": basic_Ksampler_custom,


"chx_Ksampler_adv": chx_Ksampler_adv,
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_Ksampler_texture": chx_Ksampler_texture,
"chx_ksampler_Deforum": chx_ksampler_Deforum,

"chx_Ksampler_refine": chx_Ksampler_refine,

"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,


#--context data-------------------------------------------#
"Data_preset_save": Data_preset_save,
"Data_chx_Merge":Data_chx_Merge,
"Data_chx_MergeBig":Data_chx_MergeBig,
"Data_fullData": Data_fullData,
"Data_presetData":Data_presetData,

"Date_basic": Date_basic,
"Date_basic_easy": Date_basic_easy,
"Data_sample": Data_sample,
"Data_select": Data_select,


#---------tool------------------------------------------#
"chx_YC_LG_Redux": chx_YC_LG_Redux,
"chx_Style_Redux": chx_Style_Redux,
"chx_StyleModelApply":chx_StyleModelApply,
"chx_re_fluxguide": chx_re_fluxguide,
"chx_easy_text": chx_easy_text,   
"chx_controlnet": chx_controlnet, 
"chx_controlnet_union": chx_controlnet_union,
"chx_mask_Mulcondi": mask_Mulcondition,
#"chx_condi_hook": chx_condi_hook,
"chx_Upscale_simple": chx_Upscale_simple,



#--unpack------------------------------------------#
"param_preset": param_preset,
"Unpack_param": Unpack_param,
"Model_Preset":Model_Preset,
"Unpack_Model": Unpack_Model,

"CN_preset1": CN_preset1,
"Unpack_CN": Unpack_CN,
"photoshop_preset":photoshop_preset,
"Unpack_photoshop":Unpack_photoshop,


#-stack-------------------------------------------#




#-IPA-------------------------------------------#



}








NODE_DISPLAY_NAME_MAPPINGS = {

"pre_inpaint":" pre_inpaint(xl)",
"chx_ksampler_tile": "chx_ksampler_tile(sole)",   
"chx_Ksampler_inpaint": "chx_Ksampler_inpaint(sole)",

"chx_IPA_basic": "IPA_basic",
"chx_IPA_faceID": "IPA_faceID",
"chx_IPA_XL_adv": "IPA_XL_adv",


}


