![image](https://github.com/user-attachments/assets/3227c43c-d82d-47d3-bb9a-096f8fdca21c)


一些资源和工作流放在Some resources and workflows are placed in  [ComfyUI-Apt_Collect_](https://github.com/cardenluo/ComfyUI-Apt_Collect)
# <font color="#000000"> 一、Update record更新记录</font>

2025.5.17  Now the system supports switching to Chinese.支持系统切换中文

2025.5.31 Sum_load_adv 的Unet支持GGUF模型加载 ；The Unet supports the loading of GGUF models

2025.6.30 更新内容比较多 
          ①增加excel互动节点，在comfyui中实现多维表格功能 
          ②增加pre_kontext节点，让工作流和flux一样简洁
          ③text_sum节点，增加预设内容，kontext和wan常用提示词
          ④移除一些重复功能的节点，要找回原来的节点，只能回到旧版本了
          On June 30, 2025, there are quite a few updates. 
          ① An Excel interaction node is added to implement the multi - dimensional table function in ComfyUI. 
          ② A pre_kontext node is added to make the workflow as concise as Flux. 
          ③ For the text_sum node, preset content is added, including kontext and commonly used prompt words of Wan. 
          ④ Some nodes with duplicate functions are removed. To retrieve the original nodes, you can only go back to the old version. 
          
2025.7.3  增加Kontext采样器，连续编辑

2025.7.6  添加Kontext采样器尺寸选择，自定义或官方自动尺寸

2025.7.9  添加在线PS工具和lora触发词获取添加到stack lora

2025.7.12  添加节点 kontext_inpaint

2025.7.13  添加 kontext官方presets。放入了 text_sum

2025.7.15  添加 kontext多条件遮罩 pre kontext_mul 

2025.7.18  添加节点，创建_遮罩_视角打标  create_Mask_visual_tag 

2025.7.23  添加节点，创建_遮罩_匹配打标  create_Mask_match_shape

Simultaneously issue commands to generate for multiple objects

<img width="2829" height="1347" alt="image" src="https://github.com/user-attachments/assets/5b32b4dd-2f2e-4324-95ad-2b646854073c" />





连续编辑Continuous editing     ksampler_Kontext
![image](https://github.com/user-attachments/assets/e84c2e03-32af-4608-bb5b-5ed683628c69)



精细化控制Fine-grained control    pre_Kontext
![image](https://github.com/user-attachments/assets/152014f4-5d0e-421c-abb8-b7e5a0ff4742)




遮罩高级处理kontext_inpaint--遮罩部分太小，识别不了的时候，用放大再采样When the masked part is too small to be recognized, use magnification and then resampling.
<img width="3060" height="1145" alt="image" src="https://github.com/user-attachments/assets/f82d292b-f586-4248-88bb-47379fda8b2e" />



# <font color="#000000">二、Usage Guide使用指南</font> 
**1、预设加载器 Load：Sum_load_adv 组合各种流程需要的模型加载，Unet还支持GGUF模型**

**Load the models required for combining various processes，Unet also supports GGUF models**

1)选择模型： checkpoint or Unet 或者 over model

2)按顺序添加 clip：clip1, clip2, clip3, clip4  ，会根据 clip 的数量自动适应模式，是 flux还是 sd 3.5 等

3)全部参数的设置都可以自己保存为预设

1)Select model: checkpoint or Unet or over model. 

2)Add clips in sequence: clip 1, clip 2, clip 3, clip 4. The mode will be automatically adapted according to the number of clips, such as flux or sd 3.5, etc. 

3)All parameter settings can be saved as presets by yourself.


| <font color="#ff0000">run_Mode</font> | <font color="#ff0000">composition </font> | <font color="#ff0000">Pattern represents </font> |
| ------------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| ckpt                                  | checkpoint or Unet                        | XL, sd 1.5                                       |
| clip 1                                | clip 1+checkpoint or unet                 | Wan 2.1、LTX                                      |
| clip 2                                | clip 1+clip 2 + unet                      | Flux                                             |
| clip 3                                | clip 1+clip 2+clip 3 + unet               | SD 3.5                                           |
| clip 4                                | clip 1+clip 2+clip 3 +clip 4+ unet        | Hidream                                          |
| clip(only)                            | clip 1+&clip 2+&clip 3 +&clip 4           | output clip without model                        |
| over model                            | replace checkpoint or Unet                | nunchaku-flux                                    |
| over clip                             | replace all clip                          |                                                  |

![image](https://github.com/user-attachments/assets/c937203d-6ada-4b58-a882-512290e30dcd)


或者你可以使用 `load create_chx` 创建一个自定义的加载器，就像下面 Nunchaku一样，也会让工作流变的非常简单 
Or you can use `load create_chx` to create a custom loader. Just like Nunchaku below, it will also make the workflow very simple.

![image](https://github.com/user-attachments/assets/40f0df16-31cd-4645-9a7f-1c730452385d)


或者使用具体类型加载器，像load_FLUX,load_basic,load_SD35, 


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
| Chx_ksampler_Deforum_math | 数学公式驱动    | Driven by Formulas           |
| Chx_ksampler_Deforum_sch  | 调度数据驱动    | Driven by Scheduling Data    |
| Chx_ksampler_kontext      | kontext采样器    | kontext Sampler           |
| Chx_ksampler_kontext_avd  | kontext高级采样器   |Advanced   kontext Sampler    |

集成好的采样器能快速实现常用功能，一般的工作流要实现生成图，修复、放大全流程，需要大量的节点配合，像下面会变的非常简单 
The integrated sampler can quickly implement common functions. To achieve the full process of generating images, repairing, and enlarging in a general workflow, a large number of nodes are required. For example, it will become very simple as follows.
![image](https://github.com/user-attachments/assets/0c62a1f7-e92f-41bc-a6fb-447b3cc7ea48)




**3、控制器 controller：总控设计 Overall control design**
采用sum 汇总控制节点，特别是 stack 集中化控制 
Adopt the sum to summarize the control nodes, especially the stack centralized control

| **sum_stack_image** | **图像生成控制堆**             |
| ------------------- | ----------------------- |
| **sum_stack_AD**    | **动画生成控制堆 AnimateDiff** |
| **sum_stack_Wan**   | **视频生成控制堆 Wan 2.1**     |
| **sum_editor**      | **可以编辑所有的基础参数**         |
| **sum_latent**      | **各种 latent 接入方式**      |
| **sum_lora**        | **批量 lora**             |
| **sum_controlnet**  | **批量 controlnet**       |
| **sum_text**        | **功能齐全的文本编辑**           |


![image](https://github.com/user-attachments/assets/a477dd17-80c1-4759-9a10-d69c855c1c52)


**4、基础节点 Basic nodes**
持续更新中Continuously updating

# <font color="#000000"> 三 、Installation</font>
Clone the repository to the **custom_nodes** directory and install dependencies

```

#1. git下载
git clone https://github.com/cardenluo/ComfyUI-Apt_Preset.git

#2. 安装依赖
双击install.bat安装依赖

```
注意Note：

要使用功能controlNet schdule控制，请先安装To use the function of controlNet schdule control, please install it first.

[ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) 

要加载GGUF模型，请先安装To load the GGUF model, please install it first.

[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) 

# <font color="#000000">四、Reference Node Packages参考节点包</font>
The code of this open-source project has referred to the following code during the development process. We express our sincere gratitude for their contributions in the relevant fields.

| [ComfyUI](https://github.com/comfyanonymous/ComfyUI)                      | [ComfyUI\_VisualStylePrompting](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting) | [Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [ComfyUI-EasyDeforum](https://github.com/Chan-0312/ComfyUI-EasyDeforum)   | [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)      | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)         |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)                 | [ComfyUI\_mittimiLoadPreset2](https://github.com/mittimi/ComfyUI_mittimiLoadPreset2)           | [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)           |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)      | [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)                   |
| [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)         | [Comfyui__Flux_Style_Adjust](https://github.com/yichengup/Comfyui_Flux_Style_Adjust)           | [ComfyUI_LayerStyle_](https://github.com/chflame163/ComfyUI_LayerStyle)          |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)               | [ComfyUI_-Easy_Deforum_](https://github.com/Chan-0312/ComfyUI-EasyDeforum)                     | [ComfyUI-AudioScheduler](https://github.com/a1lazydog/ComfyUI-AudioScheduler)    |
| [ComfyUI-IC-Light](https://github.com/kijai/ComfyUI-IC-Light)             | [ComfyUI-Inspyrenet-Rembg](https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg)               | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)                                                                                |


## Disclaimer免责声明
This open-source project and its contents are provided "AS IS" without any express or implied warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are responsible for ensuring compliance with all applicable laws and regulations in their respective jurisdictions when using this software or publishing content generated by it. The authors and copyright holders are not responsible for any violations of laws or regulations by users in their respective locations.


## 关注我，分享工作流轻松搭建的方法
<img width="333" height="395" alt="PixPin_2025-07-18_09-12-23" src="https://github.com/user-attachments/assets/35238903-a94c-4802-a67a-5aea3580d533" />

