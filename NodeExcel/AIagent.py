
import os
import json
import base64
import random
from PIL import Image
import numpy as np
import io
import requests

try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZhipuAI = None
    ZHIPUAI_AVAILABLE = False


#region--------------------------------------------------------------------------


class AI_Ollama:
    """Ollama生成节点，用于在ComfyUI中调用Ollama模型生成文本"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama3:8b"}),
                "prompt": ("STRING", {"multiline": True, "default": "请输入提示词..."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    RETURN_NAMES = ("pos", )
    CATEGORY = "Apt_Preset/prompt"


    def run(self, model_name, prompt, temperature, max_tokens, system_prompt="", ollama_host="http://localhost:11434"):
        """调用Ollama API生成文本"""
        try:
            url = f"{ollama_host}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()

            # 处理流式响应
            result = ""
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode('utf-8'))
                    if 'response' in line_data:
                        result += line_data['response']
                    if line_data.get('done', False):
                        break

            return (result,)
        except Exception as e:
            print(f"Ollama API调用错误: {str(e)}")
            return (f"错误: {str(e)}",)



def _log_info(message):
    print(f"[GLM_Nodes] 信息：{message}")

def _log_warning(message):
    print(f"[GLM_Nodes] 警告：{message}")

def _log_error(message):
    print(f"[GLM_Nodes] 错误：{message}")

def get_zhipuai_api_key():
    env_api_key = os.getenv("ZHIPUAI_API_KEY")
    if env_api_key:
        _log_info("使用环境变量 API Key。")
        return env_api_key
    
    _log_warning("未设置环境变量 ZHIPUAI_API_KEY。")
    return ""




ZHIPU_MODELS = [
    "GLM-4.5-Flash",
    "glm-4v-flash", 
    "XX----下面的要开通支付-----XX",
    "glm-4.5-air", 
    "glm-4.5", 
    "glm-4.5-x", 
    "glm-4.5-airx", 
    "glm-4.5-flash", 
    "glm-4-plus", 
    "glm-z1-air", 
    "glm-4v-plus-0111", 
    "glm-4.1v-thinking-flash"
]

BUILT_IN_PROMPTS = {

    "Text字体生成": '''核心内容："文字内容"  ，双引号里面的就是要生成的艺术字体内容。  
你是一个专业的海报、画板、艺术字体和logo设计稿生成助手。会根据用户提供的文字类型，文字排版，文字效果，和整体氛围内容等。明确各元素的排列方式、疏密关系、视觉重心位置，描绘主体所处的背景设定，包括空间层次、装饰元素、色彩基调、光影效果等将其扩写成一个详细、具体、富有设计感，和高端美感的创作提示词，将所有要素融合为一段连贯的描述性文字,当文字类型和文字效果匹配产生冲突时，优先保证字体类型，进行取舍，确保逻辑流畅。输出格式要求：不要包含任何解释性文字或额外的对话，只需提供一个创意内容输出。''',

    "Text字体公式助手": '''你是一个提示词整理助手，会对输入的内容进行处理：
接收内容：获取需分析的包含文字相关信息的内容​
特征识别：从内容中提取文字载体、字体类型、文字排版、文字效果、整体氛围、中文内容、英文内容等特征​
归类划分：将提取的特征分别对应至各标签类别​
输出格式为：​
文字载体 {text_medium}：具体内容
字体类型 {text_font}：具体内容
文字排版 {text_array}：具体内容
文字效果 {text_effect}：具体内容
整体氛围 {sum_toon}：具体内容
中文内容 {text_cn}：具体带双引号的中文​
英文内容 {text_en}：具体带双引号的英文​
输出规则：
1、分析过程，取原话，不要添加和删减内容
2、无归类的，就留直接空着，不需要解释和描述。（错误：“无具体描述”）
3、不包含任何解释性文字，仅呈现分析结果
现在对如下输入的内容进行整理：''',  

    "Wan-视频生成": '''你是文生视频提示生成助手，需严格模仿以下核心风格创作：1. 以 “视频展示了 / 呈现了” 开头；2. 只写画面可见的核心元素（如物体形态、环境状态、角色动作、镜头运动），不添加主观抒情或长篇铺垫；3. 用短句连贯衔接，突出视觉先后顺序（如从局部到整体、从静态到动态）。
生成时需包含这些关键信息：
先点明核心主体（如石块、森林、飞天仙女）；
再补充主体状态 / 动作（如伫立、狂风肆虐、翱翔）；
接着描述环境细节（如广场石砖、乌云、废土地景）；
最后说明镜头运动（如拉远、跟随、拉高）。
输出要求：仅生成 1 段符合上述风格的文字，控制在 150 字内，纯视觉描述、无多余叙述，严格匹配示例的简洁表达节奏。''',
    "img英文描述": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用英文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写“角度”。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 'a woman', 'the man'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', best quality, high resolution, 4k, high quality, masterpiece, photorealistic'",
    "img中文描述": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的中文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的中文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用中文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写“角度”。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 '女人', '男人'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', 最佳质量, 高分辨率, 4k, 高质量, 艺术品, 照片级真实'",
    "img动漫描述": "请以动漫风格描述图片内容，包括角色特征、场景设定、色彩搭配和艺术风格等要素。",   
    "Text-Ai问答": "你是一个智能助手，需友好、准确地回应用户的各类问题，包括生活常识、信息查询、简单建议等。回答要简洁易懂，保持礼貌，若无法解答需如实告知。",
    "Text节点分析": "你是节点分析助手，根据用户输入comfyui节点的原代码，对这个代码进行分析总结。说明一下输入，输出的功能，以及是怎么实现功能的，并给出注意事项。",
    "Text-kontext助手": '''你是图像编辑提示词助手，核心能力:
    根据用户意图生成精准的图像编辑英文提示词。提示词需同时包含命令式指令与关键元素的详细描述，确保AI模型能准确理解并执行编辑操作；此外，必须明确指定对“非修改元素”的保护要求，避免图像中未涉及修改的部分出现意外变化。
    提示词生成规范:
    1. 命令与描述结合：针对用户提出的修改目标（如“让女孩结冰”），提示词需包含三部分：
    - 命令式语句（明确编辑动作）；
    - 修改目标的细节描述（如“a girl with her face and hands covered in translucent, frosty ice—preserving the subtle contours of her features while adding a glossy, cold sheen to the icy layers”）；
    - 必要的关联场景补充（若有助于提升准确性，如“set against the original snowy background with no changes to the snow’s texture or color”）。
    2. 非修改元素保护：必须在提示词中明确“保护范围”，例如用户需求为“让女孩微笑”时，需补充“keep the girl’s original facial features (eyes, nose, hair style), clothing details (fabric texture, color, design), and the entire background completely unchanged—only adjust the shape of her mouth to a natural, soft smile”，确保仅修改目标部分，其余元素保持原样。
    3. 细节精准性：对于修改涉及的关键特征（如结冰范围、微笑弧度、物体颜色等），需补充具体描述（如“ice only covers her exposed skin, not her clothing”“smile should show slight dimples on her cheeks without altering the size of her lips”），避免模型产生歧义。
    输出要求:仅提供符合上述规范的英文图像编辑提示词，不包含任何解释性文字、额外对话或中文内容。'''
}





class AI_GLM4:
    @classmethod
    def INPUT_TYPES(cls):
        prompt_keys = list(BUILT_IN_PROMPTS.keys())
        default_selection = prompt_keys[0] if prompt_keys else ""

        return {
            "required": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "此处是语言模型的文本输入，图片分析用系统提示词输入"
                }),
                "prompt_preset": (prompt_keys, {"default": default_selection}),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": " 输入系统提示词，留空则用预设"
                }),

                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
              }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "可选：智谱AI API Key (留空则尝试从环境变量或config.json读取)"
                }),
                "model_name": (ZHIPU_MODELS, {  "default": "glm-4v-flash",
                    "placeholder": "请输入模型名称，如 glm-4v-flash "
                }),
            },
            "optional": {
                "image_input": ("IMAGE", {"optional": True, "tooltip": "直接输入ComfyUI IMAGE对象"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, prompt_preset, prompt_override, api_key, model_name, 
                max_tokens, seed, text_input, image_input=None):
        
        image_input_provided = image_input is not None
        
        if image_input_provided:
            return self._process_image(api_key, prompt_preset, prompt_override, model_name, seed,
                                     image_input, max_tokens)
        else:
            return self._process_text(text_input, api_key, prompt_preset, prompt_override, model_name,
                                    max_tokens, seed)

    def _process_text(self, text_input, api_key, prompt_preset, prompt_override, model_name, 
                      max_tokens, seed):
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)

        _log_info("初始化智谱AI客户端。")

        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        final_system_prompt = ""
        if prompt_override and prompt_override.strip():
            final_system_prompt = prompt_override.strip()
            _log_info("使用 'prompt_override'。")
        elif prompt_preset in BUILT_IN_PROMPTS:
            final_system_prompt = BUILT_IN_PROMPTS[prompt_preset]
            _log_info(f"使用预设提示词: '{prompt_preset}'。")
        else:
            final_system_prompt = list(BUILT_IN_PROMPTS.values())[0]
            _log_warning("预设提示词未找到，使用默认提示词。")

        if not final_system_prompt:
            _log_error("系统提示词不能为空。")
            return ("系统提示词不能为空。",)

        if not isinstance(final_system_prompt, str):
            final_system_prompt = str(final_system_prompt)

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": text_input}
        ]

        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        _log_info(f"内部种子: {effective_seed}。")
        random.seed(effective_seed)

        _log_info(f"调用 GLM-4 ({model_name})...")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.9,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            response_text = response.choices[0].message.content
            _log_info("GLM-4 响应成功。")
            return (response_text,)
        except Exception as e:
            error_message = f"GLM-4 API 调用失败: {e}"
            return (error_message,)



    def _process_image(self, api_key, prompt_preset, prompt_override, model_name, seed,
                    image_input=None, max_tokens=1024):
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)
        _log_info("初始化智谱AI客户端。")

        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        # 验证图片输入是否有效
        if image_input is None:
            _log_error("必须提供有效的IMAGE对象。")
            return ("必须提供有效的IMAGE对象。",)

        # 处理ComfyUI的IMAGE对象（PyTorch张量）
        try:
            # 将张量转换为PIL Image
            # ComfyUI的IMAGE格式为：[B, H, W, C]，值范围[0,1]
            i = 255. * image_input.cpu().numpy()  # 转换为[0,255]范围
            img_array = np.clip(i, 0, 255).astype(np.uint8)[0]  # 取第一个批次的图片
            img = Image.fromarray(img_array)
            
            # 转换为Base64编码
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")  # 使用PNG格式确保无损转换
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            final_image_data = f"data:image/png;base64,{image_base64}"
            _log_info("IMAGE对象成功转换为Base64格式")
        except Exception as e:
            _log_error(f"图片格式转换失败: {str(e)}")
            return (f"图片格式转换失败: {str(e)}",)

        # 确定提示词内容
        final_prompt_text = ""
        if prompt_override and prompt_override.strip():
            final_prompt_text = prompt_override.strip()
            _log_info("使用自定义提示词")
        elif prompt_preset in BUILT_IN_PROMPTS:
            final_prompt_text = BUILT_IN_PROMPTS[prompt_preset]
            _log_info(f"使用预设提示词: {prompt_preset}")
        else:
            final_prompt_text = list(BUILT_IN_PROMPTS.values())[0]
            _log_warning("使用默认提示词")

        if not final_prompt_text:
            _log_error("提示词不能为空")
            return ("提示词不能为空",)

        # 构建消息内容
        content_parts = [
            {"type": "text", "text": final_prompt_text},
            {"type": "image_url", "image_url": {"url": final_image_data}}
        ]

        # 种子处理
        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        _log_info(f"使用内部种子: {effective_seed}")
        random.seed(effective_seed)

        # 调用GLM-4V API
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.9,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            response_content = str(response.choices[0].message.content)
            _log_info("GLM-4V图片识别成功")
            return (response_content,)
        except Exception as e:
            error_message = f"GLM-4V API调用失败: {str(e)}"
            _log_error(error_message)
            return (error_message,)





























#endregion--------------------------------------------------------------------------















































