
import torch
import numpy as np
from typing import Any
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance
import random
import numexpr
import torch.nn.functional as F
import pandas as pd
import re
import json
from io import BytesIO
import hashlib
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fft import fft
from ..main_unit import *



#region-----------------收纳--------------------



class AD_ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)



class AD_MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"

    def execute(self, mask, size, method):
        orig_size = mask.shape[0]

        if orig_size == size:
            return (mask,)

        if size <= 1:
            return (mask[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(mask.shape)[1:], dtype=mask.dtype, device=mask.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = mask[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = mask[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = mask.repeat([math.ceil(size / mask.shape[0])] + [1] * (len(mask.shape) - 1))[:size]
        elif 'first' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat([mask[:1].repeat(size-mask.shape[0], 1, 1), mask], dim=0)
        elif 'last' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat((mask, mask[-1:].repeat((size-mask.shape[0], 1, 1))), dim=0)

        return (out,)



class AD_sch_prompt_preset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyframe_interval": ("INT", {"default": 30, "min": 0, "max": 999, "step": 1}),
                "loops": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "prompt_1": ("STRING", {"multiline": True, "default": "dancing,(neon:1.3)"}),
            },
            "optional": {
                "prompt_2": ("STRING", {"multiline": True, "default": "(dancing:`pw_a`)"}),
                "prompt_3": ("STRING", {"multiline": True, "default": "(smiling:`(0.5+0.5*sin(t/max_f))`"}),
                "prompt_4": ("STRING", {"multiline": True, "default": "(b:`(1.5*cos(1.57*(t-30)/30)*cos(1.57*(t-30)/30))`)"}),
                "prompt_5": ("STRING", {"multiline": True}),
                "prompt_6": ("STRING", {"multiline": True}),
                "prompt_7": ("STRING", {"multiline": True}),
                "prompt_8": ("STRING", {"multiline": True}),
                "prompt_9": ("STRING", {"multiline": True}),
                "prompt_10": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sch_prompt",)
    FUNCTION = "make_keyframes"
    CATEGORY = "Apt_Preset/AD"

    def make_keyframes(self, keyframe_interval, loops, prompt_1, prompt_2="", prompt_3="", prompt_4="", prompt_5="", prompt_6="", prompt_7="", prompt_8="", prompt_9="", prompt_10=""):
        prompts = []

        if prompt_1 != "":
            prompts.append(prompt_1)

        if prompt_2 != "":
            prompts.append(prompt_2)

        if prompt_3 != "":
            prompts.append(prompt_3)

        if prompt_4 != "":
            prompts.append(prompt_4)

        if prompt_5 != "":
            prompts.append(prompt_5)

        if prompt_6 != "":
            prompts.append(prompt_6)

        if prompt_7 != "":
            prompts.append(prompt_7)

        if prompt_8 != "":
            prompts.append(prompt_8)

        if prompt_9 != "":
            prompts.append(prompt_9)

        if prompt_10 != "":
            prompts.append(prompt_10)

        keyframe_list = []
        
        i = 0
        for j in range(1, loops + 1): 
            for index, prompt in enumerate(prompts):
                if i == 0:
                    keyframe_list.append(f"\"0\": \"{prompt}\",\n")
                    i += keyframe_interval  
                    continue
                
                new_keyframe = f"\"{i}\": \"{prompt}\",\n"
                keyframe_list.append(new_keyframe)
                i += keyframe_interval 
        
        sch_prompt = " ".join(keyframe_list)[:-2]
        
        return (sch_prompt,)



class AD_batch_replace:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                # 添加节点工作类型选择
                "type": (["choose frame output", "replace  frame and  output all"], {"default": "choose frame output"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "replace_img": ("IMAGE",),
                "replace_mask": ("MASK",),
            }
        } 
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "Apt_Preset/AD"

    def imagesfrombatch(self, start_index, num_frames, type, images=None, masks=None, replace_img=None, replace_mask=None):
        chosen_images = None
        chosen_masks = None

        # Process images if provided
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            if start_index < 0 or start_index >= len(images):
                raise ValueError("Start index is out of range")
            end_index = min(start_index + num_frames, len(images))

            if replace_img is not None:
                # 尺寸处理
                processed_input_img = []
                for img in replace_img:
                    if img.shape != images[0].shape:
                        # 中心对齐裁切逻辑
                        img_height, img_width = img.shape[0], img.shape[1]
                        target_height, target_width = images[0].shape[0], images[0].shape[1]
                        y_start = (img_height - target_height) // 2
                        x_start = (img_width - target_width) // 2
                        cropped_img = img[y_start:y_start + target_height, x_start:x_start + target_width]
                        processed_input_img.append(cropped_img)
                    else:
                        processed_input_img.append(img)
                processed_input_img = torch.stack(processed_input_img)

                # 补齐或舍弃图像
                if len(processed_input_img) < num_frames:
                    last_img = processed_input_img[-1:]
                    repeat_times = num_frames - len(processed_input_img)
                    padded_img = last_img.repeat(repeat_times, 1, 1, 1)
                    processed_input_img = torch.cat([processed_input_img, padded_img], dim=0)
                elif len(processed_input_img) > num_frames:
                    processed_input_img = processed_input_img[:num_frames]

                # 替换对应位置的图像
                images = torch.cat([images[:start_index], processed_input_img, images[end_index:]], dim=0)

            if type == "choose frame output":
                chosen_images = images[start_index:end_index]
            elif type == "replace  frame and  output all":
                chosen_images = images

        # Process masks if provided
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            if start_index < 0 or start_index >= len(masks):
                raise ValueError("Start index is out of range for masks")
            end_index = min(start_index + num_frames, len(masks))

            if replace_mask is not None:
                if len(replace_mask) < num_frames:
                    last_mask = replace_mask[-1:]
                    repeat_times = num_frames - len(replace_mask)
                    padded_mask = last_mask.repeat(repeat_times, 1, 1)
                    replace_mask = torch.cat([replace_mask, padded_mask], dim=0)
                elif len(replace_mask) > num_frames:
                    replace_mask = replace_mask[:num_frames]
                masks = torch.cat([masks[:start_index], replace_mask, masks[end_index:]], dim=0)

            if type == "choose frame output":
                chosen_masks = masks[start_index:end_index]
            elif type == "replace  frame and  output all":
                chosen_masks = masks

        return (chosen_images, chosen_masks,)


#endregion-----------------收纳--------------------



#region---------------------def----------------------


class ScheduleSettings:
    def __init__(
            self,
            text_g: str,
            pre_text_G: str,
            app_text_G: str,
            text_L: str,
            pre_text_L: str,
            app_text_L: str,
            max_frames: int,
            current_frame: int,
            print_output: bool,
            pw_a: float,
            pw_b: float,
            pw_c: float,
            pw_d: float,
            start_frame: int,
            end_frame:int,
            width: int,
            height: int,
            crop_w: int,
            crop_h: int,
            target_width: int,
            target_height: int,
    ):
        self.text_g=text_g
        self.pre_text_G=pre_text_G
        self.app_text_G=app_text_G
        self.text_l=text_L
        self.pre_text_L=pre_text_L
        self.app_text_L=app_text_L
        self.max_frames=max_frames
        self.current_frame=current_frame
        self.print_output=print_output
        self.pw_a=pw_a
        self.pw_b=pw_b
        self.pw_c=pw_c
        self.pw_d=pw_d
        self.start_frame=start_frame
        self.end_frame=end_frame
        self.width=width
        self.height=height
        self.crop_w=crop_w
        self.crop_h=crop_h
        self.target_width=target_width
        self.target_height=target_height

    def set_sync_option(self, sync_option: bool):
        self.sync_context_to_pe = sync_option



class AudioData:
    def __init__(self, audio_file) -> None:
        
        # Extract the sample rate
        sample_rate = audio_file.frame_rate

        # Get the number of audio channels
        num_channels = audio_file.channels

        # Extract the audio data as a NumPy array
        audio_data = np.array(audio_file.get_array_of_samples())
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def get_channel_audio_data(self, channel: int):
        if channel < 0 or channel >= self.num_channels:
            raise IndexError(f"Channel '{channel}' out of range. total channels is '{self.num_channels}'.")
        return self.audio_data[channel::self.num_channels]
    
    def get_channel_fft(self, channel: int):
        audio_data = self.get_channel_audio_data(channel)
        return fft(audio_data)


class AudioFFTData:
    def __init__(self, audio_data, sample_rate) -> None:

        self.fft = fft(audio_data)
        self.length = len(self.fft)
        self.frequency_bins = np.fft.fftfreq(self.length, 1 / sample_rate)
    
    def get_max_amplitude(self):
        return np.max(np.abs(self.fft))
    
    def get_normalized_fft(self) -> float:
        max_amplitude = self.get_max_amplitude()
        return np.abs(self.fft) / max_amplitude

    def get_indices_for_frequency_bands(self, lower_band_range: int, upper_band_range: int):
        return np.where((self.frequency_bins >= lower_band_range) & (self.frequency_bins < upper_band_range))

    def __len__(self):
        return self.length


defaultPrompt = """"0" :"",
"30" :"",
"60" :"",
"90" :"",
"120" :""
"""


defaultText="""Rabbit
Dog
Cat
One prompt per line
"""


defaultValue = """0:(0),
30:(1),
60:(0),
90:(1),
120:(0)
"""


#region-------------------AD--def-------------------------------

def batch_parse_key_frames(string, max_frames):
    string = re.sub(r',\s*$', '', string)
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        frame = int(sanitize_value(frameParam[0])) if check_is_number(
            sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
            frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",1)[::-1]))
        frames[frame] = frameParam[1].strip()
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def sanitize_value(value):
    # Remove single quotes, double quotes, and parentheses
    value = value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")
    return value


def batch_get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
    value_is_number = False
    for i in range(0, max_frames):
        if i in key_frames:
            value = str(key_frames[i])  # Convert to string to ensure it's treated as an expression
            value_is_number = check_is_number(sanitize_value(value))
            if value_is_number:
                key_frame_series[i] = sanitize_value(value)
        if not value_is_number:
            t = i
            # workaround for values formatted like 0:("I am test") //used for sampler schedules
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitize_value(value)
        elif is_single_string:  # take previous string value and replicate it
            key_frame_series[i] = key_frame_series[i - 1]
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

    # 获取有效的关键帧索引和值
    valid_indices = key_frame_series.index[key_frame_series.notna()].tolist()
    valid_values = key_frame_series[valid_indices].tolist()

    if len(valid_indices) < 2:
        # 只有一个关键帧，直接填充整个序列
        key_frame_series.fillna(valid_values[0] if valid_values else 0, inplace=True)
    else:
        # 使用自定义缓动函数进行插值
        for i in range(len(valid_indices) - 1):
            start_index = valid_indices[i]
            end_index = valid_indices[i + 1]
            start_value = valid_values[i]
            end_value = valid_values[i + 1]

            for j in range(start_index, end_index):
                t = (j - start_index) / (end_index - start_index)
                easing_function = easing_functions.get(interp_method)
                if easing_function:
                    eased_t = easing_function(t)
                    interpolated_value = start_value + (end_value - start_value) * eased_t
                    key_frame_series[j] = interpolated_value

        # 填充最后一个关键帧之后的值
        key_frame_series[valid_indices[-1]:] = valid_values[-1]

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def batch_prompt_schedule(settings:ScheduleSettings,clip):
    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # Add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    p = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, settings)
    n = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, settings)

    # return positive and negative conditioning as well as the current and next prompts for each
    return (p, n,)



def process_input_text(text: str) -> dict:
    input_text = text.replace('\n', '')
    input_text = "{" + input_text + "}"
    input_text = re.sub(r',\s*}', '}', input_text)
    animation_prompts = json.loads(input_text.strip())
    return animation_prompts


def BatchPoolAnimConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip, settings:ScheduleSettings):
    pooled_out = []
    cond_out = []
    max_size = 0

    if settings.end_frame == 0:
        settings.end_frame = settings.max_frames
        print("end_frame at 0, using max_frames instead!")

    if settings.start_frame >= settings.end_frame:
        settings.start_frame = 0
        print("start_frame larger than or equal to end_frame, using max_frames instead!")

    if max_size == 0:
        for i in range(0, settings.end_frame):
            tokens = clip.tokenize(str(cur_prompt_series[i]))
            cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
            max_size = max(max_size, cond_to.shape[1])
    for i in range(settings.start_frame, settings.end_frame):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        if i < len(nxt_prompt_series):
            tokens = clip.tokenize(str(nxt_prompt_series[i]))
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            cond_from, pooled_from = torch.zeros_like(cond_to), torch.zeros_like(pooled_to)

        interpolated_conditioning = addWeighted([[cond_to, {"pooled_output": pooled_to}]],
                                                [[cond_from, {"pooled_output": pooled_from}]],
                                                weight_series[i],max_size)

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from)

        cond_out.append(interpolated_cond)
        pooled_out.append(interpolated_pooled)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return [[final_conditioning, {"pooled_output": final_pooled_output}]]


def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength, max_size=0):
    out = []

    if len(conditioning_from) > 1:
        print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        if max_size == 0:
            max_size = max(t1.shape[1], cond_from.shape[1])
        t0, max_size = pad_with_zeros(cond_from, max_size)
        t1, max_size = pad_with_zeros(t1, t0.shape[1])  # Padding t1 to match max_size
        t0, max_size = pad_with_zeros(t0, t1.shape[1])

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        t_to = conditioning_to[i][1].copy()

        t_to["pooled_output"] = pooled_output_from
        n = [tw, t_to]
        out.append(n)

    return out


def pad_with_zeros(tensor, target_length):
    current_length = tensor.shape[1]

    if current_length < target_length:
        # Calculate the required padding length
        pad_length = target_length - current_length

        # Calculate padding on both sides to maintain the tensor's original shape
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        # Pad the tensor along the second dimension
        tensor = F.pad(tensor, (0, 0, left_pad, right_pad))

    return tensor, target_length


def batch_split_weighted_subprompts(text, pre_text, app_text):
    pos = {}
    neg = {}
    pre_text = str(pre_text)
    app_text = str(app_text)

    if "--neg" in pre_text:
        pre_pos, pre_neg = pre_text.split("--neg")
    else:
        pre_pos, pre_neg = pre_text, ""

    if "--neg" in app_text:
        app_pos, app_neg = app_text.split("--neg")
    else:
        app_pos, app_neg = app_text, ""

    for frame, prompt in text.items():
        negative_prompts = ""
        positive_prompts = ""
        prompt_split = prompt.split("--neg")

        if len(prompt_split) > 1:
            positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
        else:
            positive_prompts = prompt_split[0]

        pos[frame] = ""
        neg[frame] = ""
        pos[frame] += (str(pre_pos) + " " + positive_prompts + " " + str(app_pos))
        neg[frame] += (str(pre_neg) + " " + negative_prompts + " " + str(app_neg))
        if pos[frame].endswith('0'):
            pos[frame] = pos[frame][:-1]
        if neg[frame].endswith('0'):
            neg[frame] = neg[frame][:-1]
    return pos, neg


def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)


def convert_pw_to_tuples(settings):
    if isinstance(settings.pw_a, (int, float, np.float64)):
        settings.pw_a = tuple([settings.pw_a] * settings.max_frames)
    if isinstance(settings.pw_b, (int, float, np.float64)):
        settings.pw_b = tuple([settings.pw_b] * settings.max_frames)
    if isinstance(settings.pw_c, (int, float, np.float64)):
        settings.pw_c = tuple([settings.pw_c] * settings.max_frames)
    if isinstance(settings.pw_d, (int, float, np.float64)):
        settings.pw_d = tuple([settings.pw_d] * settings.max_frames)


def interpolate_prompt_seriesA(animation_prompts, settings:ScheduleSettings):

    max_f = settings.max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}


    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(settings.max_frames):
        sorted_prompts.append((str(settings.max_frames), sorted_prompts[-1][1]))

    # Setup containers for interpolated prompts
    nan_list = [np.nan for a in range(settings.max_frames)]
    cur_prompt_series = pd.Series(nan_list,dtype=object)
    nxt_prompt_series = pd.Series(nan_list,dtype=object)

    # simple array for strength values
    weight_series = [np.nan] * settings.max_frames

    # in case there is only one keyed prompt, set all prompts to that prompt
    if settings.max_frames == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

    #make sure prompt weights are tuples and convert them if not
    convert_pw_to_tuples(settings)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0

    index_offset = 0

    # Evaluate the current and next prompt's expressions
    for i in range(settings.start_frame, min(settings.end_frame,len(cur_prompt_series))):
        cur_prompt_series[i] = prepare_batch_promptA(cur_prompt_series[i], settings, i)
        nxt_prompt_series[i] = prepare_batch_promptA(nxt_prompt_series[i], settings, i)
        if settings.print_output == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            if(settings.start_frame >= i):
                if(settings.end_frame > 0):
                    if(settings.end_frame > i):
                        print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i),
                              "\n", "Current Prompt: ",
                              cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                              weight_series[i], "\n")
                else:
                    print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i), "\n", "Current Prompt: ",
                          cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                          weight_series[i], "\n")
        index_offset = index_offset + 1

    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)


def prepare_batch_promptA(prompt, settings:ScheduleSettings, index):
    max_f = settings.max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace(
            't',
            f'{index}').replace("pw_a",
            f"{settings.pw_a[index]}").replace("pw_b",
            f"{settings.pw_b[index]}").replace("pw_c",
            f"{settings.pw_c[index]}").replace("pw_d",
            f"{settings.pw_d[index]}").replace("max_f",
            f"{max_f}").replace('`', '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()


#endregion-------------------AD--def-------------------------------



#endregion--------------------------------def-------------------------------------------------



class AD_sch_prompt_basic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),

                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": False, }),
                    "app_text": ("STRING", {"multiline": False, }),

                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/AD"
    
    def animate(self, text, max_frames, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        

        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=None,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=max_frames,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        
        positive, negative = batch_prompt_schedule(settings, clip)
        
        return (positive, negative)


class AD_sch_prompt_chx:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("RUN_CONTEXT",),
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": False, }),
                    "app_text": ("STRING", {"multiline": False, }),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                }
        }

    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/AD"
    def animate(self, text, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='',context=None):
        
        clip=context.get("clip")
        
        positive=None
        negative=None
        
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=None,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=max_frames,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        
        positive, negative = batch_prompt_schedule(settings, clip)
        context = new_context(context, positive=positive, negative=negative,)  
        
        return (context,)


class AD_sch_mask:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                "invert": ("BOOLEAN", {"default": False}),
                "frames": ("INT", {"default": 16,"min": 2, "max": 255, "step": 1}),
                "width": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
        },
    } 
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "createfademask"
    CATEGORY = "Apt_Preset/AD"
    def createfademask(self, frames, width, height, invert, points_string, easing_type):
        points = []
        points_string = points_string.rstrip(',\n')
        for point_str in points_string.split(','):
            frame_str, color_str = point_str.split(':')
            frame = int(frame_str.strip())
            color = float(color_str.strip()[1:-1])
            points.append((frame, color))

        if len(points) == 0 or points[-1][0] != frames - 1:
            points.append((frames - 1, points[-1][1] if points else 0))

        points.sort(key=lambda x: x[0])

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        next_point = 1

        for i in range(batch_size):
            while next_point < len(points) and i > points[next_point][0]:
                next_point += 1

            prev_point = next_point - 1
            t = (i - points[prev_point][0]) / (points[next_point][0] - points[prev_point][0])

            easing_function = easing_functions.get(easing_type)
            if easing_function:
                t = easing_function(t)

            color = points[prev_point][1] - t * (points[prev_point][1] - points[next_point][1])
            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)


class AD_sch_value:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": defaultValue}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 999999, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01}),  # 添加放大系数
                "offset": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01})  # 添加偏移值
            }
        }

    RETURN_TYPES = ("FLOAT", "INT","IMAGE")
    RETURN_NAMES = ("float", "int","graph")
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/AD"

    def animate(self, text, max_frames, easing_type, scale_factor, offset):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames, interp_method=easing_type, is_single_string=False)

        t = [val * scale_factor + offset for val in t]
        graph_image = AD_schdule_graph(t)[0]

        return (t, list(map(int, t)), graph_image)


class Amp_drive_value:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_amp": ("FLOAT", {"forceInput": True}),
                "add_to": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "threshold_for_add": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_ceiling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    CATEGORY = "Apt_Preset/AD"

    RETURN_TYPES = ("FLOAT", "INT", "IMAGE")
    RETURN_NAMES = ("float", "int", "graph")
    FUNCTION = "convert_and_graph"

    def convert(self, normalized_amp, add_to, threshold_for_add, add_ceiling, scale):
        normalized_amp[np.isnan(normalized_amp)] = 0.0
        normalized_amp[np.isinf(normalized_amp)] = 1.0
        modified_values = np.where(normalized_amp > threshold_for_add, normalized_amp + add_to, normalized_amp)
        modified_values = np.clip(modified_values, 0.0, add_ceiling)
        # 使用 scale 放大 modified_values
        scaled_values = modified_values * scale
        return scaled_values, scaled_values.astype(int)

    def graph(self, normalized_amp):
        width = int(len(normalized_amp) / 10)
        if width < 10:
            width = 10
        if width > 100:
            width = 100
        plt.figure(figsize=(width, 6))
        plt.plot(normalized_amp)
        plt.xlabel("Frame(s)")
        plt.ylabel("Amplitude")
        plt.grid()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()  
        buffer.seek(0)
        image = Image.open(buffer)
        print(f"Image mode: {image.mode}, Image size: {image.size}")
        return (pil2tensor(image),)


    def convert_and_graph(self, normalized_amp, add_to, threshold_for_add, add_ceiling, scale):
        float_value, int_value = self.convert(normalized_amp, add_to, threshold_for_add, add_ceiling, scale)
        graph_image = self.graph(float_value)[0]
        return float_value, int_value, graph_image


class Amp_drive_String:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultText}),
                    "normalized_amp": ("FLOAT", {"forceInput": True}),
                    "triggering_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },                          
               "optional": {
                    "loop": ("BOOLEAN", {"default": True},),
                    "shuffle": ("BOOLEAN", {"default": False},),
                    }
                }

    @classmethod
    def IS_CHANGED(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        if shuffle:
            return float("nan")
        m = hashlib.sha256()
        m.update(text)
        m.update(normalized_amp)
        m.update(triggering_threshold)
        m.update(loop)
        return m.digest().hex()


    CATEGORY = "Apt_Preset/AD"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "convert"
        

    def convert(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        prompts = text.splitlines()

        keyframes = self.get_keyframes(normalized_amp, triggering_threshold)

        if loop and len(prompts) < len(keyframes): # Only loop if there's more prompts than keyframes
            i = 0
            result = []
            for _ in range(len(keyframes) // len(prompts)):
                if shuffle:
                    random.shuffle(prompts)
                for prompt in prompts:
                    result.append('"{}": "{}"'.format(keyframes[i], prompt))
                    i += 1
        else: # normal
            if shuffle:
                random.shuffle(prompts)
            result = ['"{}": "{}"'.format(keyframe, prompt) for keyframe, prompt in zip(keyframes, prompts)]

        result_string = ',\n'.join(result)

        return (result_string,)

    def get_keyframes(self, normalized_amp, triggering_threshold):
        above_threshold = normalized_amp >= triggering_threshold
        above_threshold = np.insert(above_threshold, 0, False)  # Add False to the beginning
        transition = np.diff(above_threshold.astype(int))
        keyframes = np.where(transition == 1)[0]
        return keyframes


class Amp_audio_Normalized:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "frame_rate": ("INT", {"default": 12, "min": 0, "max": 240, "step": 1}),
                    "operation": (["avg","max","sum"], {"default": "max"}),
                    },                            
                "optional": {
                    "start_frame": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                    "limit_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                    }
                }

    CATEGORY = "Apt_Preset/AD"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("normalized_amp",)
    FUNCTION = "process_audio"

    def load_audio(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        waveform_np = waveform.squeeze().numpy()
        
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            waveform_int16.tobytes(), 
            frame_rate=sample_rate, 
            sample_width=waveform_int16.dtype.itemsize, 
            channels=1
        )
        audio_data = AudioData(audio_segment)
        return (audio_data,)

    def get_ffts(self, audio, frame_rate:int, start_frame:int=0, limit_frames:int=0):
        audio = self.load_audio(audio)[0]

        audio_data = audio.get_channel_audio_data(0)
        total_samples = len(audio_data)
        
        samples_per_frame = audio.sample_rate / frame_rate
        total_frames = int(np.ceil(total_samples / samples_per_frame))

        if (np.abs(start_frame) > total_frames):
            raise IndexError(f"Absolute value of start_frame '{start_frame}' cannot exceed the total_frames '{total_frames}'")
        if (start_frame < 0):
            start_frame = total_frames + start_frame

        ffts = []
        if (limit_frames > 0 and start_frame + limit_frames < total_frames):
            end_at_frame = start_frame + limit_frames
            total_frames = limit_frames
        else:
            end_at_frame = total_frames
        
        for i in range(start_frame, end_at_frame):
            i_next = (i + 1) * samples_per_frame

            if i_next >= total_samples:
                i_next = total_samples
            i_current = i * samples_per_frame
            frame = audio_data[round(i_current) : round(i_next)]
            ffts.append(AudioFFTData(frame, audio.sample_rate))

        return ffts

    def process_amplitude(self, audio_fft, operation):
        lower_band_range =100
        upper_band_range = 20000

        max_frames = len(audio_fft)
        # 修复未存取变量 a 的问题
        key_frame_series = pd.Series([np.nan for _ in range(max_frames)])
        
        for i in range(0, max_frames):
            fft = audio_fft[i]
            indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)
            amplitude = (2 / len(fft)) * np.abs(fft.fft[indices])

            if "avg" in operation:
                key_frame_series[i] = np.mean(amplitude)
            elif "max" in operation:
                key_frame_series[i] = np.max(amplitude)
            elif "sum" in operation:
                key_frame_series[i] = np.sum(amplitude)

        normalized_amplitude =  key_frame_series / np.max( key_frame_series)
        return normalized_amplitude

    def process_audio(self, audio, frame_rate:int, operation, start_frame:int=0, limit_frames:int=0):
        ffts = self.get_ffts(audio, frame_rate, start_frame, limit_frames)
        normalized_amplitude = self.process_amplitude(ffts, operation)
        return (normalized_amplitude,)


class Amp_drive_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("FLOAT", {"forceInput": True}),
                    "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "frame_offset": ("INT", {"default": 0,"min": -255, "max": 255, "step": 1}),
                    "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "size": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                    "shape": (
                        [   
                            'none',
                            'circle',
                            'square',
                            'triangle',
                        ],
                        {
                        "default": 'none'
                        }),
                    "color": (
                        [   
                            'white',
                            'amplitude',
                        ],
                        {
                        "default": 'amplitude'
                        }),
                    },}

    CATEGORY = "Apt_Preset/AD"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"

    def convert(self, normalized_amp, width, height, frame_offset, shape, location_x, location_y, size, color):
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        normalized_amp = np.roll(normalized_amp, frame_offset)
        out = []
        for amp in normalized_amp:
            if color == 'amplitude':
                grayscale_value = int(amp * 255)
            elif color == 'white':
                grayscale_value = 255
            gray_color = (grayscale_value, grayscale_value, grayscale_value)
            finalsize = size * amp
            
            if shape == 'none':
                shapeimage = Image.new("RGB", (width, height), gray_color)
            else:
                shapeimage = Image.new("RGB", (width, height), "black")

            draw = ImageDraw.Draw(shapeimage)
            if shape == 'circle' or shape == 'square':
                left_up_point = (location_x - finalsize, location_y - finalsize)
                right_down_point = (location_x + finalsize,location_y + finalsize)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=gray_color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=gray_color)
                    
            elif shape == 'triangle':
                left_up_point = (location_x - finalsize, location_y + finalsize)
                right_down_point = (location_x + finalsize, location_y + finalsize)
                top_point = (location_x, location_y)
                draw.polygon([top_point, left_up_point, right_down_point], fill=gray_color)
            
            shapeimage = pil2tensor(shapeimage)
            mask = shapeimage[:, :, :, 0]
            out.append(mask)
        
        return (torch.cat(out, dim=0),)





NODE_CLASS_MAPPINGS = {
    "Amp_drive_value": Amp_drive_value,
    "Amp_drive_String": Amp_drive_String,
    "Amp_audio_Normalized": Amp_audio_Normalized,
    "Amp_drive_mask": Amp_drive_mask,
}

NODE_DISPLAY_NAME_MAPPINGS = {

}

