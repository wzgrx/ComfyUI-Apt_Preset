import requests
import json
import os
from PIL import Image
import base64
import io



class DoubaoAPI:
    """豆包API配置节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING", 
                    {
                        "default": os.environ.get("DOUBAO_API_KEY", ""),
                        "placeholder": "请输入豆包API密钥（可在火山引擎控制台获取）"
                    }
                ),
            },
            "optional": {
                "endpoint": (
                    "STRING", 
                    {
                        "default": "https://ark.cn-beijing.volces.com/api/v3/",
                        "placeholder": "豆包API端点URL"
                    }
                ),
            },
        }

    RETURN_TYPES = ("DOUBAO_API",)
    FUNCTION = "configure"

    def configure(self, api_key, endpoint="https://ark.cn-beijing.volces.com/api/v3/"):
        return ({
            "api_key": api_key,
            "endpoint": endpoint
        },)

class DoubaoConfig:
    """豆包模型配置节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "STRING", 
                    {
                        "default": "doubao-seed-1.6-250615",
                        "options": [
                            # 豆包1.6系列（支持视觉）
                            "doubao-seed-1.6-250615",
                            "doubao-seed-1.6-250615-32k",
                            "doubao-seed-1.6-250615-128k",
                            "doubao-seed-1.6-250615-256k",
                            # 豆包1.5系列
                            "doubao-seed-1.5-250615",
                            "doubao-seed-1.5-250615-32k",
                            "doubao-seed-1.5-250615-128k",
                            "doubao-seed-1.5-250615-256k",
                            # DeepSeek系列
                            "deepseek-chat",
                            "deepseek-coder",
                            #  legacy models
                            "doubao-pro-4k", "doubao-pro-32k", "doubao-pro-128k", "doubao-pro-256k",
                            "doubao-lite-4k", "doubao-lite-32k", "doubao-lite-128k"
                        ]
                    }
                ),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("DOUBAO_CONFIG",)
    FUNCTION = "configure"

    def configure(self, model, max_tokens=1024, temperature=0.7, top_p=0.9, repetition_penalty=1.0):
        return ({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        },)

class DoubaoTextChat:
    """豆包文本对话节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "请输入提示词..."}),
                "doubao_api": ("DOUBAO_API",),
                "doubao_config": ("DOUBAO_CONFIG",),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "ignore_errors": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"

    def chat(self, user_prompt, doubao_api, doubao_config, system_prompt="", ignore_errors=True):
        try:
            url = f"{doubao_api['endpoint']}chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {doubao_api['api_key']}"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            data = {
                "model": doubao_config['model'],
                "messages": messages,
                "max_tokens": doubao_config['max_tokens'],
                "temperature": doubao_config['temperature'],
                "top_p": doubao_config['top_p'],
                "repetition_penalty": doubao_config['repetition_penalty']
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result['choices']) > 0:
                return (result['choices'][0]['message']['content'],)
            else:
                error_msg = f"API响应格式错误: {result}"
                if ignore_errors:
                    print(error_msg)
                    return ("",)
                else:
                    return (error_msg,)

        except Exception as e:
            error_msg = f"豆包API调用错误: {str(e)}"
            if ignore_errors:
                print(error_msg)
                return ("",)
            else:
                return (error_msg,)

class DoubaoVisionChat:
    """豆包视觉对话节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"multiline": True, "default": "请描述这张图片..."}),
                "doubao_api": ("DOUBAO_API",),
                "doubao_config": ("DOUBAO_CONFIG",),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "ignore_errors": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"

    def chat(self, image, user_prompt, doubao_api, doubao_config, system_prompt="", ignore_errors=True):
        try:
            # 检查是否使用支持视觉的模型
            vision_models = ["doubao-seed-1.6-250615", "doubao-seed-1.6-250615-32k", 
                            "doubao-seed-1.6-250615-128k", "doubao-seed-1.6-250615-256k"]
            if doubao_config['model'] not in vision_models and not doubao_config['model'].startswith('ep-'):
                warning = f"警告: 当前模型 {doubao_config['model']} 可能不支持视觉功能。建议使用 {vision_models[0]} 或视觉专用Endpoint ID。"
                print(warning)

            # 处理图像
            img = Image.fromarray((image[0] * 255).astype('uint8'))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            url = f"{doubao_api['endpoint']}chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {doubao_api['api_key']}"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            })

            data = {
                "model": doubao_config['model'],
                "messages": messages,
                "max_tokens": doubao_config['max_tokens'],
                "temperature": doubao_config['temperature'],
                "top_p": doubao_config['top_p'],
                "repetition_penalty": doubao_config['repetition_penalty']
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result['choices']) > 0:
                return (result['choices'][0]['message']['content'],)
            else:
                error_msg = f"API响应格式错误: {result}"
                if ignore_errors:
                    print(error_msg)
                    return ("",)
                else:
                    return (error_msg,)

        except Exception as e:
            error_msg = f"豆包API调用错误: {str(e)}"
            if ignore_errors:
                print(error_msg)
                return ("",)
            else:
                return (error_msg,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "DoubaoAPI": DoubaoAPI,
    "DoubaoConfig": DoubaoConfig,
    "DoubaoTextChat": DoubaoTextChat,
    "DoubaoVisionChat": DoubaoVisionChat
}







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
    "glm-4-air-250414", 
    "glm-4-airx", 
    "glm-4-flashx", 
    "glm-4-flashx-250414", 
    "glm-z1-air", 
    "glm-z1-airx", 
    "glm-z1-flash", 
    "glm-z1-flashx", 
    "glm-4v-plus-0111", 
    "glm-4v-flash", 
    "glm-4.1v-thinking-flashx", 
    "glm-4.1v-thinking-flash"
]


BUILT_IN_PROMPTS = {
    "Text字体生成": "请将用户提供的，关键词信息，组合优化成合适的场景。输出不要包含任何解释性文字或额外的对话，只需提共一个创意内容输出。注意: \字体类型\"文本\"\是这个固定组合，不可拆分，例如:\火焰体\"美少女\"\是个固定合体",
    "Text视频生成": "请将用户提供的简单描述扩展为详细的视频生成提示词，包含场景、动作、镜头语言等要素。",
    "img英文描述": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用英文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写“角度”。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 'a woman', 'the man'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', best quality, high resolution, 4k, high quality, masterpiece, photorealistic'",
    "img中文描述": "你是一个专业的图像描述专家，请详细描述图片内容，包括主体对象、场景、动作、光线、材质、色彩、构图和艺术风格等要素。",
    "img动漫描述": "请以动漫风格描述图片内容，包括角色特征、场景设定、色彩搭配和艺术风格等要素。",
    "AI助手-基础通用版": "你是一个智能助手，需友好、准确地回应用户的各类问题，包括生活常识、信息查询、简单建议等。回答要简洁易懂，保持礼貌，若无法解答需如实告知。",
    "AI助手-功能聚焦版": "作为AI助手，你的核心任务是帮用户解决问题：提供实用信息、解释概念、整理思路。沟通时用口语化表达，避免复杂术语，优先满足用户的直接需求。",
    "AI助手-交互风格版": "你是一个高效贴心的助手，回应需快速、清晰，语气亲切自然。能处理多类需求（如问答、提醒、建议），遇到不确定的内容时，可引导用户补充信息。"
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
                    "default": "请扩写XX。",
                    "placeholder": "输入处理内容"
                }),
                "prompt_preset": (prompt_keys, {"default": default_selection}),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": " 优先加载系统提示词,留空则从预设加载"
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

        image_input_provided = image_input is not None

        if not image_input_provided:
            _log_error("必须提供IMAGE对象。")
            return ("必须提供IMAGE对象。",)

        final_image_data = None
        if image_input_provided:
            _log_info("检测到 IMAGE 对象输入，正在转换为 Base64。")
            try:
                i = 255. * image_input.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                final_image_data = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
                _log_info("IMAGE 对象成功转换为 Base64。")
            except Exception as e:
                _log_error(f"将 IMAGE 对象转换为 Base64 失败: {e}")
                return (f"将 IMAGE 对象转换为 Base64 失败: {e}",)

        if not final_image_data:
            _log_error("未能获取有效的图片数据。")
            return ("未能获取有效的图片数据。",)

        final_prompt_text = ""
        if prompt_override and prompt_override.strip():
            final_prompt_text = prompt_override.strip()
            _log_info("使用 'prompt_override'。")
        elif prompt_preset in BUILT_IN_PROMPTS:
            final_prompt_text = BUILT_IN_PROMPTS[prompt_preset]
            _log_info(f"使用预设提示词: '{prompt_preset}'。")
        else:
            final_prompt_text = list(BUILT_IN_PROMPTS.values())[0]
            _log_warning("预设提示词未找到，使用默认提示词。")

        if not final_prompt_text:
            _log_error("提示词不能为空。")
            return ("提示词不能为空。",)

        if not isinstance(final_prompt_text, str):
            final_prompt_text = str(final_prompt_text)

        content_parts = [{"type": "text", "text": final_prompt_text}]
        content_parts.append({"type": "image_url", "image_url": {"url": final_image_data}})

        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        _log_info(f"内部种子: {effective_seed}。")
        random.seed(effective_seed)

        _log_info(f"调用 GLM-4V ({model_name})...")
    
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.9,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            response_content = str(response.choices[0].message.content)
            _log_info("GLM-4V 响应成功。")
            return (response_content,)
        except Exception as e:
            error_message = f"GLM-4V API 调用失败: {e}"
            _log_error(error_message)
            return (error_message,)

