![image](https://github.com/user-attachments/assets/00457d93-ead4-4083-902b-2ae32dfbb8f0)

一些资源和工作流放在Some resources and workflows are placed in  [ComfyUI-Apt_Collect_](https://github.com/cardenluo/ComfyUI-Apt_Collect)
# <font color="#000000"> 一、Update record更新记录</font>
2025.5.3   Update to V 2.0 and add a large number of basic nodes.

# <font color="#000000">二、Usage Guide使用指南</font> 
**1、加载器 Loader：Sum_load 全能加载器 All - in - one Loader**
包含五种模式，全部参数的设置都可以自己保存为预设，也配备了每个模式的简版加载器 It includes five modes. All parameter settings can be saved as presets by yourself, and there is also a simplified loader for each mode.

| <font color="#ff0000">run_Mode</font> | <font color="#ff0000">module Combination </font> | 
| --------------------------------------------- | ------------------------------------------------ |
| Basic                                     | checkpoint                                       |
| Clip                                      | clip 1+checkpoint or unet                        | 
| Flux                                      | clip 1+clip 2 + unet                             | 
| SD3.5                                     | clip 1+clip 2+clip 3 + unet                      | 
| only_clip                                 | clip 1 or clip 1+clip 2 or clip 1+clip 2+clip 3  | 

![image](https://github.com/user-attachments/assets/83d25557-dd7b-43b4-8bbd-07421a05761a)

**2、采样器 Sampler：丰富的采样样式Rich sampling styles**

| 名称nodes                   | 描述        | Description                  |
|---------------------------|-----------|------------------------------|
| Basic_Ksampler_simple     | 精简参数采样器   | Simplified Parameter Sampler |
| Basic_Ksampler_mid        | 半参数采样器    | Semi-parametric Sampler      |
| Basic_Ksampler_full       | 完整参数采样器   | Full Parameter Sampler       |
| Basic_Ksampler_custom     | 自定义基础采样   | Custom Base Sampling         |
| Basic_Ksampler_adv        | 高级采样      | Advanced Sampling            |
| Basic_Ksampler_batch      | 调度批次采样器   | Batch Scheduler Sampler      |
| Chx_Ksampler_refine       | 基础放大采样    | Basic Amplified Sampling     |
| Chx_ksampler_tile         | tile 放大采样 | Tile Amplified Sampling      |
| Chx_Ksampler_mix          | 混合采样器     | Hybrid Sampler               |
| Chx_Ksampler_texture      | 纹理采样器     | Texture Sampler              |
| Chx_Ksampler_dual_paint   | 双区重绘      | Dual Zone Redraw             |
| Chx_Ksampler_dual_area    | 双区采样      | Dual Zone Sampling           |
| Chx_Ksampler_inpaint      | 重绘采样器     | Redraw Sampler               |
| Chx_Ksampler_VisualStyle  | 风格一致性采样   | Style Consistency Sampling   |
| Chx_ksampler_Deforum      | 简单的首尾帧    | Simple First and Last Frames |
| Chx_ksampler_Deforum_math | 数学公式驱动    | Driven by Formulas           |
| Chx_ksampler_Deforum_sch  | 调度数据驱动    | Driven by Scheduling Data    |

**3、控制器 controller：总控设计 Overall control design**
采用sum 汇总控制节点，特别是 stack 集中化控制 
Adopt the sum to summarize the control nodes, especially the stack centralized control

| **sum_stack_image** | **图像生成控制堆**             |
|---------------------|-------------------------------|
| **sum_stack_AD**    | **动画生成控制堆 AnimateDiff** |
| **sum_stack_Wan**   | **视频生成控制堆 Wan 2.1**     |
| **sum_editor**      | **可以编辑所有的基础参数**     |
| **sum_latent**      | **各种 latent 接入方式**       |
| **sum_lora**        | **批量 lora**                 |
| **sum_controlnet**  | **批量 controlnet**           |
| **sum_text**        | **功能齐全的文本编辑**         |

![image](https://github.com/user-attachments/assets/95a4f2fb-fb36-4c90-bb3a-72ed879de618)

**4、基础节点 Basic nodes：系统分类，完整规划 System classification, complete planning**

| **<font color="#ff0000">mask</font>** | **<font color="#ff0000">math</font>** | **<font color="#ff0000">image</font>** | **<font color="#ff0000">imgEffect</font>** |
| --------------------------------- | --------------------------------- | ---------------------------------- | -------------------------------------- |
| Mask_AD_generate                  | list_num_range                    | pad_uv_fill                        | img_Loadeffect                         |
| Mask_inpaint_Grey                 | list_cycler_Value                 | pad_color_fill                     | img_Upscaletile                        |
| Mask_math                         | list_input_text                   | Image_LightShape                   | img_Remove_bg                          |
| Mask_Detect_label                 | list_input_Value                  | Image_Normal_light                 | img_CircleWarp                         |
| Mask_mulcolor_img                 | ListGetByIndex                    | Image_keep_OneColorr               | img_Stretch                            |
| Mask_mulcolor_mask                | ListSlice                         | Image_transform                    | img_WaveWarp                           |
| Mask_Outline                      | MergeList                         | Image_cutResize                    | img_Liquify                            |
| Mask_Smooth                       | batch_cycler_Prompt               | Image_Resize                       | img_Seam_adjust_size                   |
| Mask_Offset                       | batch_cycler_Value                | Image_scale_adjust                 | img_texture_Offset                     |
| Mask_cut_mask                     | batch_cycler_text                 | image_sumTransform                 | img_White_balance                      |
| Mask_image2mask                   | batch_cycler_split_text           | Image_overlay                      | img_HDR                                |
| Mask_mask2mask                    | batch_cycler_image                | Image_overlay_mask                 | color_adjust                           |
| Mask_mask2img                     | batch_cycler_mask                 | Image_overlay_composite            | color_Match                            |
| Mask_splitMask                    | Remap_basic_data                  | Image_overlay_transform            | color_match_adv                        |
|                                   | Remap_mask                        | Image_overlay_sum                  | color_input                            |
|                                   | math_BinaryOperation              | Image_Extract_Channel              | color_color2hex                        |
|                                   | math_BinaryCondition              | Image_Apply_Channel                | color_hex2color                        |
|                                   | math_UnaryOperation               | Image_RemoveAlpha                  | color_image_Replace                    |
|                                   | math_UnaryCondition               | image_selct_batch                  | color_pure_img                         |
|                                   | math_Exec                         | Image_scale_match                  | color_Gradient                         |
|                                   |                                   |                                    | color_RadialGradient                   |

| <font color="#ff0000">View_IO</font> | <font color="#ff0000">prompt</font> | <font color="#ff0000">type</font> | <font color="#ff0000">layout</font> |
| ------------------------------------ | ----------------------------------- | --------------------------------- | ----------------------------------- |
| IO_inputbasic                        | text_CSV_load                       | Pack                              | lay_ImageGrid                       |
| IO_load_anyimage                     | text_SuperPrompter                  | Unpack                            | lay_MaskGrid                        |
| IO_clip_vision                       | text_mul_replace                    | creat_mask_batch                  | lay_match_W_or_H                    |
| IO_clear_cache                       | text_mul_remove                     | creat_mask_batch_input            | lay_match_W_and_H                   |
| view_Data                            | text_free_wildcards                 | creat_image_batch                 | lay_edge_cut                        |
| view_bridge_image                    | text_stack_wildcards                | creat_image_batch_input           | lay_text                            |
| view_bridge_Text                     |                                     | creat_any_List                    |                                     |
| view_mask                            |                                     | AnyCast                           |                                     |
| view_LatentAdvanced                  |                                     | type_Anyswitch                    |                                     |
| view_combo                           |                                     | type_BasiPIPE                     |                                     |
| view_node_Script                     |                                     | type_Image_List2Batch             |                                     |
| view_GetLength                       |                                     | type_Image_Batch2List             |                                     |
| view_GetShape                        |                                     | type_Mask_Batch2List              |                                     |
| view_GetWidgetsValues                |                                     | type_Mask_List2Batch              |                                     |
|                                      |                                     | type_text_list2batch              |                                     |
|                                      |                                     | type_text_2_UTF8                  |                                     |

# <font color="#000000"> 三 、Installation</font>
Clone the repository to the **custom_nodes** directory and install dependencies

```

#1. git下载
git clone https://github.com/cardenluo/ComfyUI-Apt_Preset.git

#2. 安装依赖
双击install.bat安装依赖

```
注意Note：
要使用AD简化的功能，请先安装 To use the simplified functions of AD, you need to install it first.

[Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)

要使用功能controlNet schdule控制，请先安装To use the function of controlNet schdule control, please install it first.

[ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) 



# <font color="#000000">四、Reference Node Packages参考节点包</font>
The code of this open-source project has referred to the following code during the development process. We express our sincere gratitude for their contributions in the relevant fields.

| [ComfyUI](https://github.com/comfyanonymous/ComfyUI)                      | [ComfyUI\_VisualStylePrompting](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting) | [Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [ComfyUI-EasyDeforum](https://github.com/Chan-0312/ComfyUI-EasyDeforum)   | [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)      | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)         |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)                 | [ComfyUI\_mittimiLoadPreset2](https://github.com/mittimi/ComfyUI_mittimiLoadPreset2)           | [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)           |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)      | [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)                   |
| [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)         | [Comfyui__Flux_Style_Adjust](https://github.com/yichengup/Comfyui_Flux_Style_Adjust)           | [ComfyUI_LayerStyle_](https://github.com/chflame163/ComfyUI_LayerStyle)          |
| [_ComfyUI_-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)             | [ComfyUI_-Easy_Deforum_](https://github.com/Chan-0312/ComfyUI-EasyDeforum)                     | [ComfyUI-AudioScheduler](https://github.com/a1lazydog/ComfyUI-AudioScheduler)    |
|                                                                           |                                                                                                |                                                                                  |


## Disclaimer免责声明
This open-source project and its contents are provided "AS IS" without any express or implied warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are responsible for ensuring compliance with all applicable laws and regulations in their respective jurisdictions when using this software or publishing content generated by it. The authors and copyright holders are not responsible for any violations of laws or regulations by users in their respective locations.


## 关注我，持续分享插件的使用方法，轻松搭建各种工作流
![image](https://github.com/user-attachments/assets/d1f53fe7-4a31-49e4-b8bb-fb5e71e5cf5f)
