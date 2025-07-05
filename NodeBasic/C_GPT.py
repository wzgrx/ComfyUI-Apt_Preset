import torch
import numpy as np
from PIL import Image
import os
from openai import OpenAI
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TRANSFORMERS_CACHE
import shutil
import folder_paths

from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.v2 import ToPILImage
from decord import VideoReader, cpu  # pip install decord




api_key = "Your API key here"       # -deepseek-   Your API key here
base_url = "https://api.deepseek.com/v1"     # -deepseek-   Your API key here

class GPT_deepseek_api_text:
    # 定义助手字典为类变量
    ASSISTANTS = {
        "Helpful Assistant": "You are a helpful assistant",
        "Art Style Expert": "You are an AI art assistant specialized in generating prompts for various art styles. Your task is to provide detailed descriptions of art styles, including color schemes, brush strokes, and composition.",
        "Landscape Painter": "You are an AI assistant specialized in creating prompts for landscape paintings. Your task is to describe scenic views, including mountains, rivers, forests, and skies, with a focus on lighting and atmosphere.",
        "Abstract Art Specialist": "You are an AI assistant specialized in abstract art. Your task is to generate prompts that describe abstract concepts, shapes, and colors, encouraging creative and non-representational artwork.",
        "Portrait Artist": "You are an AI assistant specialized in creating prompts for portrait paintings. Your task is to describe human faces, expressions, and emotions, with attention to detail in features like eyes, hair, and skin tones.",
        "Sci-Fi & Fantasy Artist": "You are an AI assistant specialized in generating prompts for sci-fi and fantasy art. Your task is to describe futuristic cities, alien worlds, mythical creatures, and magical landscapes, with a focus on imaginative and otherworldly elements.",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "system": (list(s.ASSISTANTS.keys()), {"default": "Helpful Assistant"}), 
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "topic_range": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "reduce_repetition": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "change_topic": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                #"stop_generation_mark": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"multiline": False,"default": api_key}),
                "base_url": ("STRING", {"multiline": False,"default": base_url}),

            },
            "optional": {

            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/GPT"

    @classmethod
    def execute(self, prompt,system="Helpful Assistant", 
                temperature=1.0, max_tokens=512, topic_range=1.0,
                reduce_repetition=0.0, change_topic=0.0, api_key=api_key,
                base_url=base_url,
                ):
        # 根据选择的助手名称获取对应的 system 内容
        system_content = self.ASSISTANTS.get(system, "You are a helpful assistant")

        if not api_key:
            raise Exception("API key is not set.")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        params = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_content},  # 使用选择的助手内容
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature, 
            "max_tokens": max_tokens,
            "top_p": topic_range,
            "frequency_penalty": reduce_repetition,
            "presence_penalty": change_topic,
            "stream": True,
            "response_format": {"type": "text"}
        }
        
        #if stop_generation_mark:
        #    params["stop"] = [stop_generation_mark]
        
        try:
            response = client.chat.completions.create(**params)
            # 检查 response 对象是否有 choices 属性，并且 choices 列表不为空
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return (response.choices[0].message.content,)
            else:
                # 若不符合条件，抛出异常提示 API 响应中没有有效的选择
                raise Exception("API 响应中不包含有效的选择项。")
        except openai.OpenAIError as e:
            if hasattr(e, 'http_status') and e.http_status == 402:
                raise Exception("账户余额不足，请充值后再试。")
            else:
                raise e




class GPT_Janus_img_2_text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "model_name": ([ "deepseek-ai/Janus-Pro-1B"],"deepseek-ai/Janus-Pro-7B",),
                "seed": ("INT", {
                    "default": 666666666666666,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze_image"
    CATEGORY = "Apt_Preset/GPT"

    def analyze_image(self, model_name, image, question, seed, temperature, top_p, max_new_tokens):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # 获取ComfyUI根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建模型路径
        model_dir = os.path.join(comfy_path, 
                               "models", 
                               "Janus-Pro",
                               os.path.basename(model_name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        model = vl_gpt.to(dtype).to(device).eval()
        processor = vl_chat_processor

        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


        if len(image.shape) == 4:  # BCHW format
            if image.shape[0] == 1:
                image = image.squeeze(0)  # 移除batch维度，现在是 [H, W, C]
        

        image = (torch.clamp(image, 0, 1) * 255).cpu().numpy().astype(np.uint8)
        

        pil_image = Image.fromarray(image, mode='RGB')

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [pil_image],
            },
            {"role": " $<|User|>", "content": ""},
        ]

        prepare_inputs = processor(
            conversations=conversation, 
            images=[pil_image], 
            force_batchify=True
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return (answer,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


class GPT_Janus_generate_img:
    # 合并JanusModelLoader的初始化方法
    def __init__(self):
        pass

    # 合并JanusModelLoader的INPUT_TYPES类方法
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful photo of"
                }),
                "model_name": ([ "deepseek-ai/Janus-Pro-1B"],"deepseek-ai/Janus-Pro-7B",),
                "seed": ("INT", {
                    "default": 666666666666666,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "Apt_Preset/GPT"

    # 合并JanusModelLoader的load_model方法
    def load_model(self, model_name):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # 获取ComfyUI根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建模型路径
        model_dir = os.path.join(comfy_path, 
                               "models", 
                               "Janus-Pro",
                               os.path.basename(model_name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        vl_gpt = vl_gpt.to(dtype).to(device).eval()
        
        return (vl_gpt, vl_chat_processor)

    # 修改generate_images方法，调用合并进来的load_model方法
    def generate_images(self, model_name, prompt, seed, batch_size=1, temperature=1.0, cfg_weight=5.0, top_p=0.95):
        model, processor = self.load_model(model_name)
        try:
            from janus.models import MultiModalityCausalLM
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 图像参数设置
        image_token_num = 576  # 24x24 patches
        img_size = 384  # 输出图像大小
        patch_size = 16  # 每个patch的大小
        parallel_size = batch_size

        # 准备对话格式
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 准备输入
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + processor.image_start_tag

        # 编码输入文本
        input_ids = processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        # 准备条件和无条件输入
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:  # 无条件输入
                tokens[i, 1:-1] = processor.pad_id

        # 获取文本嵌入
        inputs_embeds = model.language_model.get_input_embeddings()(tokens)

        # 生成图像tokens
        generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()
        outputs = None

        # 自回归生成
        for i in range(image_token_num):
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state
            
            # 获取logits并应用CFG
            logits = model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # 准备下一步的输入
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # 解码生成的tokens为图像
        dec = model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        
        # 转换为numpy进行处理
        dec = dec.to(torch.float32).cpu().numpy()
        
        # 确保是BCHW格式
        if dec.shape[1] != 3:
            dec = np.repeat(dec, 3, axis=1)
        
        # 从[-1,1]转换到[0,1]
        dec = (dec + 1) / 2
        
        # 确保值范围在[0,1]之间
        dec = np.clip(dec, 0, 1)
        
        # 转换为ComfyUI需要的格式 [B,C,H,W] -> [B,H,W,C]
        dec = np.transpose(dec, (0, 2, 3, 1))
        
        # 转换为tensor
        images = torch.from_numpy(dec).float()
        
        # 打印详细的形状信息
        # print(f"Initial dec shape: {dec.shape}")
        # print(f"Final tensor: shape={images.shape}, dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
        
        # 确保格式正确
        assert images.ndim == 4 and images.shape[-1] == 3, f"Unexpected shape: {images.shape}"
        
        return (images,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed 



class GPT_MiniCPM:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "Describe this image in detail", "multiline": True}),
                "model": (
                    ["MiniCPM-V-2_6-int4", "MiniCPM-Llama3-V-2_5-int4"],
                    {"default": "MiniCPM-V-2_6-int4"},
                ),

                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),

                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                    },
                ),

                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
            "optional": {
                "ref_image": ("IMAGE",),
            },
        }


    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Apt_Preset/GPT"

    def encode_video(self, source_video_path, MAX_NUM_FRAMES):
        def uniform_sample(l, n):  # noqa: E741
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(source_video_path, ctx=cpu(0))
        total_frames = len(vr) + 1
        print("Total frames:", total_frames)
        avg_fps = vr.get_avg_fps()
        print("Get average FPS(frame per second):", avg_fps)
        sample_fps = round(avg_fps / 1)  # FPS
        duration = len(vr) / avg_fps
        print("Total duration:", duration, "seconds")
        width = vr[0].shape[1]
        height = vr[0].shape[0]
        print("Video resolution(width x height):", width, "x", height)

        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print("num frames:", len(frames))
        return frames

    def inference(
        self,
        text,
        model,
        temperature,
        max_new_tokens,
        seed,
        ref_image=None,

    ):
        
        
        keep_model_loaded=True
        video_max_num_frames=64
        video_max_slice_nums=2
        keep_model_loaded=True
        repetition_penalty=1.05
        top_p=0.8   
        top_k=100



        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"openbmb/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_checkpoint,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
            )

        with torch.no_grad():
            if ref_image is not None:
                images = ref_image.permute([0, 3, 1, 2])
                images = [ToPILImage()(img).convert("RGB") for img in images]
                msgs = [{"role": "user", "content": images + [text]}]
            else:
                msgs = [{"role": "user", "content": [text]}]
                # raise ValueError("Either image or video must be provided")

            params = {"use_image_id": False, "max_slice_nums": video_max_slice_nums}


            result = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                **params,
            )

            if not keep_model_loaded:
                del self.tokenizer  # release tokenizer memory
                del self.model  # release model memory
                self.tokenizer = None  # set tokenizer to None
                self.model = None  # set model to None
                torch.cuda.empty_cache()  # release GPU memory
                torch.cuda.ipc_collect()

            return (result,)



class BaseTranslatorNode:
    def __init__(self, model_name):
        self.use_gpu = True  # 默认使用GPU
        self.model_name = model_name
        self.model_path = os.path.join(folder_paths.base_path, "models", "translator", model_name)
        
        
        self.set_device()
        self.load_or_download_model()

    def set_device(self):
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

    def load_or_download_model(self):
        try:
            # 尝试从本地加载模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试下载模型")
            
            try:
                # 从Hugging Face下载模型
                model_id = f"Helsinki-NLP/{self.model_name}"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # 保存模型到本地
                self.model.save_pretrained(self.model_path)
                self.tokenizer.save_pretrained(self.model_path)            
                
                self.clear_cache() # 清理缓存
            except Exception as e:
                print(f"模型下载或保存失败: {e}")
                raise  # 重新抛出异常，让调用者处理

    def clear_cache(self):
        if os.path.exists(TRANSFORMERS_CACHE):
            try:
                shutil.rmtree(TRANSFORMERS_CACHE)
                print(f"缓存已清除: {TRANSFORMERS_CACHE}")
            except Exception as e:
                print(f"清除缓存时出错: {e}")
        else:
            print(f"缓存目录不存在: {TRANSFORMERS_CACHE}")  

    @classmethod
    def INPUT_TYPES(s):
        return {            
            "required": {
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No"
                }), 
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入要翻译的文本"
                }),                
            },
            "optional": {
                "optional_input_text": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "placeholder": "连接的输入（如果提供则优先使用）"
                }),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"
    CATEGORY = "Apt_Preset/GPT"

    def translate(self, input_text, use_gpu, optional_input_text=""):
        if self.use_gpu != use_gpu:
            self.use_gpu = use_gpu
            self.set_device()
            self.model = self.model.to(self.device)

        text_to_translate = optional_input_text if optional_input_text.strip() else input_text
        
        if not text_to_translate.strip():
            return {"ui": {"text": ""}, "result": ("",)}
        
        inputs = self.tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  

        return {"ui": {"text": translated_text}, "result": (translated_text,)}



class GPT_ChineseToEnglish(BaseTranslatorNode):
    def __init__(self):
        super().__init__("opus-mt-zh-en")



class GPT_EnglishToChinese(BaseTranslatorNode):
    def __init__(self):
        super().__init__("opus-mt-en-zh")


