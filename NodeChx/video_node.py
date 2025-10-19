

import torch
import comfy
import folder_paths
from nodes import  CLIPTextEncode
import numpy as np

from comfy_extras.nodes_wan import *

from ..main_unit import *




class IO_video_encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {                     
                "context": ("RUN_CONTEXT",),},
            "optional": {
                    "samples": ("LATENT", ), 
                    "trim_latent": ("INT", {"default": 0, "min": 0, "max": 500, "step": 4, "tooltip": "Only used for vace_video."}),
                    "tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
                    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                    
            }
                }
    
    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    CATEGORY = "Apt_Preset/View_IO"
    FUNCTION = "encode_tile"


    def encode_tile(self, context, samples=None, tile_size=256, trim_latent=0, overlap=64, temporal_size=64, temporal_overlap=8):


        vae = context.get("vae", None)
        if samples is None :
            samples = context.get("latent", None)
        if samples:
            samples = samples

        trim_latent = trim_latent or 0  # 如果 trim_latent 是 None，则设为 0
        if trim_latent > 0:
            # 修改为一致的调用方式
            trim_node = TrimVideoLatent()
            node_output = trim_node.execute(samples=samples, trim_amount=trim_latent)
            samples = node_output[0]  #

        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        context = new_context(context, latent=samples ,images=images,) 
        return (context, images)





def clip_vision_output_encode(clip_vision_name, image):
    clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name)
    clip_vision = comfy.clip_vision.load(clip_path)
    output = clip_vision.encode_image(image, crop=True)
    return (output,)


#region----------------源码 ---------------------



class sum_stack_Wan:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "funControl": ("FUNCONTROL", ),
                "funControl22": ("FUNCONTROL22", ),
                "funInpaint": ("FUNINPAINT", ),
                "ImageToVideo": ("IMAGETOVIDEO", ),
                "FirstLastFrame": ("FIRSTLASTFRAME", ),
                "Vace": ("WANVACE", ),
                "AnimateVideo": ("ANIMATEVIDEO", ),
                "CameraImage": ("CAMERAIMAGE", ),
                "Track": ("WANTRACKT", ),
                "SoundImage": ("SOUNDIMAGE", ),
                "SoundImage_ex": ("SOUNDIMAGE_EX", ),
                "HuMoImage": ("HUMOIMAGE", ),
                "Phantom": ("PHANTOM", ),

            },
            "hidden": {},
        }
        
    # 修改sum_stack_Wan类的RETURN_TYPES和RETURN_NAMES
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT","INT","INT","INT")
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent","vace_trim_latent","trim_image","video_frame_offset")
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"

    def merge(self, context=None, model=None, funControl=None, funControl22=None, funInpaint=None, ImageToVideo=None, FirstLastFrame=None, Vace=None, AnimateVideo=None, CameraImage=None, Track=None, SoundImage=None, SoundImage_ex=None, HuMoImage=None, Phantom=None):
        clip = context.get("clip", None)
        vae = context.get("vae")
        latent = None
        trim_latent = 0  # 初始化trim_latent默认值
        trim_image = 0  # 初始化trim_image默认值
        video_frame_offset = 0  # 初始化video_frame_offset默认值
        
        if model is None:
            model = context.get("model")

#------------------条件--------------------------------

        pos = context.get("pos","a girl")
        neg = context.get("neg","bad quality")  
        positive, = CLIPTextEncode().encode(clip, pos)
        negative, = CLIPTextEncode().encode(clip, neg)

#---------------------------------------------------

        # 按照Stack_类的顺序处理各个视频类型
        if funControl is not None:
            width, height, length, start_image, clip_img, clip_vision_name, control_video = funControl
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanFunControlToVideo().execute(
                positive=positive, 
                negative=negative, 
                vae=vae, 
                width=width, 
                height=height, 
                length=length, 
                batch_size=1, 
                start_image=start_image, 
                clip_vision_output=clip_vision_output, 
                control_video=control_video
            )


        if funControl22 is not None:
            width, height, length, ref_image, control_video = funControl22

            positive, negative, latent = Wan22FunControlToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                ref_image=ref_image,
                start_image=None,  # 明确传递 None
                control_video=control_video
            )



        if funInpaint is not None:
            width, height, length, start_image, end_image, clip_img, clip_vision_name = funInpaint
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]
            positive, negative, latent = WanFunInpaintToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                start_image=start_image,
                end_image=end_image,
                clip_vision_output=clip_vision_output
            )

        if ImageToVideo is not None:
            width, height, length, start_image, clip_img, clip_vision_name = ImageToVideo
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanImageToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                start_image=start_image,
                clip_vision_output=clip_vision_output
            )


        if FirstLastFrame is not None:
            width, height, length, start_image, end_image, clip_start_img, clip_end_img, clip_vision_start, clip_vision_end = FirstLastFrame
            clip_vision_start_image = None
            clip_vision_end_image = None
            if clip_start_img is not None and clip_vision_start is not None:
                clip_vision_start_image = clip_vision_output_encode(clip_vision_start, clip_start_img)[0]
            if clip_end_img is not None and clip_vision_end is not None:
                clip_vision_end_image = clip_vision_output_encode(clip_vision_end, clip_end_img)[0]

            positive, negative, latent = WanFirstLastFrameToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                start_image=start_image,
                end_image=end_image,
                clip_vision_start_image=clip_vision_start_image,
                clip_vision_end_image=clip_vision_end_image
            )

        if Vace is not None:
            width, height, length, strength, control_video, control_masks, reference_image = Vace        
            positive, negative, latent, trim_latent = WanVaceToVideo().execute(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    width=width,
                    height=height,
                    length=length,
                    batch_size=1,
                    strength=strength,
                    control_video=control_video,
                    control_masks=control_masks,
                    reference_image=reference_image
                )

        if AnimateVideo is not None:
            width, height, length, continue_motion_max_frames, video_frame_offset, reference_image, clip_img, clip_vision_name, face_video, pose_video, continue_motion, background_video, character_mask = AnimateVideo
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            node_output = WanAnimateToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                continue_motion_max_frames=continue_motion_max_frames,
                video_frame_offset=video_frame_offset,  
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                face_video=face_video,
                pose_video=pose_video,
                continue_motion=continue_motion,
                background_video=background_video,
                character_mask=character_mask
            )
            positive, negative, latent, animate_trim_latent, trim_image, video_frame_offset = node_output
                    

        if CameraImage is not None:
            width, height, length, start_image, clip_img, clip_vision_name, camera_conditions = CameraImage 
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]
            positive, negative, latent = WanCameraImageToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                start_image=start_image,
                clip_vision_output=clip_vision_output,
                camera_conditions=camera_conditions
            )

        if Track is not None:
            width, height, length, temperature, topk, tracks, start_image, clip_img, clip_vision_name = Track
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]
            positive, negative, latent = WanTrackToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                tracks=tracks,  
                width=width,
                height=height,
                length=length,
                batch_size=1,
                temperature=temperature,
                topk=topk,
                start_image=start_image,
                clip_vision_output=clip_vision_output
            )

        if SoundImage is not None:
            width, height, length, ref_image, control_video, ref_motion, audio_encoder_output = SoundImage
            positive, negative, latent = WanSoundImageToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                ref_image=ref_image,
                audio_encoder_output=audio_encoder_output,
                control_video=control_video,
                ref_motion=ref_motion
            )

        if SoundImage_ex is not None:
            length, video_latent, ref_image, control_video, audio_encoder_output = SoundImage_ex
            positive, negative, latent = WanSoundImageToVideoExtend().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                length=length,
                video_latent=video_latent,  
                ref_image=ref_image,
                audio_encoder_output=audio_encoder_output,
                control_video=control_video
            )

        if HuMoImage is not None:
            width, height, length, ref_image, audio_encoder_output = HuMoImage
            positive, negative, latent = WanHuMoImageToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                ref_image=ref_image,
                audio_encoder_output=audio_encoder_output
            )

        if Phantom is not None:
            width, height, length, images = Phantom
            positive, negative, latent = WanPhantomSubjectToVideo().execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                images=images
            )
        context = new_context(context, clip=clip, positive=positive, latent=latent, negative=negative, model=model)
        return (context, model, positive, negative, latent, trim_latent, trim_image, video_frame_offset)





class Stack_WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("FUNCONTROL",)
    RETURN_NAMES = ("funControl",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, start_image=None, clip_img=None, clip_vision_name=None, control_video=None):
        image_2V = (width, height, length, start_image, clip_img, clip_vision_name, control_video)
        return (image_2V,)


class Stack_Wan22FunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("FUNCONTROL22",)
    RETURN_NAMES = ("funControl22",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, ref_image=None, control_video=None):
        image_2V = (width, height, length, ref_image, control_video)
        return (image_2V,)


class Stack_WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("FUNINPAINT",)
    RETURN_NAMES = ("funInpaint",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, start_image=None, end_image=None, clip_img=None, clip_vision_name=None):
        image_2V = (width, height, length, start_image, end_image, clip_img, clip_vision_name)
        return (image_2V,)



class Stack_WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {    
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGETOVIDEO",)
    RETURN_NAMES = ("ImageToVideo",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, start_image=None, clip_img=None, clip_vision_name=None):
        image_2V = (width, height, length, start_image, clip_img, clip_vision_name)
        return (image_2V,)


class Stack_WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name_start": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "clip_vision_name_end": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "clip_img_start": ("IMAGE", ),
                "clip_img_end": ("IMAGE", ),
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("FIRSTLASTFRAME",)
    RETURN_NAMES = ("FirstLastFrame",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, start_image=None, end_image=None, clip_img_start=None, clip_img_end=None, clip_vision_name_start=None, clip_vision_name_end=None):
        image_2V = (width, height, length, start_image, end_image, clip_img_start, clip_img_end, clip_vision_name_start, clip_vision_name_end)
        return (image_2V,)


class Stack_WanVaceToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE", ),
                "control_masks": ("MASK", ),
                "reference_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("WANVACE",)
    RETURN_NAMES = ("Vace",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, strength, control_video=None, control_masks=None, reference_image=None):
        image_2V = (width, height, length, strength, control_video, control_masks, reference_image)
        return (image_2V,)


class Stack_WanAnimateToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 77, "min": 1, "max": 4096, "step": 4}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 4}),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "clip_img": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "face_video": ("IMAGE", ),
                "pose_video": ("IMAGE", ),
                "continue_motion": ("IMAGE", ),
                "background_video": ("IMAGE", ),
                "character_mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("ANIMATEVIDEO",)
    RETURN_NAMES = ("AnimateVideo",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, continue_motion_max_frames, video_frame_offset, reference_image=None, clip_img=None, clip_vision_name=None, face_video=None, pose_video=None, continue_motion=None, background_video=None, character_mask=None):
        image_2V = (width, height, length, continue_motion_max_frames, video_frame_offset, reference_image, clip_img, clip_vision_name, face_video, pose_video, continue_motion, background_video, character_mask)
        return (image_2V,)




class Stack_WanCameraImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
                "camera_conditions": ("WANCAMERA", ),
            }
        }

    RETURN_TYPES = ("CAMERAIMAGE",)
    RETURN_NAMES = ("CameraImage",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, start_image=None, clip_img=None, clip_vision_name=None, camera_conditions=None):
        image_2V = (width, height, length, start_image, clip_img, clip_vision_name, camera_conditions)
        return (image_2V,)


class Stack_WanTrackToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] + folder_paths.get_filename_list("clip_vision"), {"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "temperature": ("FLOAT", {"default": 220.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "topk": ("INT", {"default": 2, "min": 1, "max": 10}),
                "tracks": ("STRING", {"multiline": True, "default": "[]"}),
            },
            "optional": {
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("WANTRACKT",)
    RETURN_NAMES = ("Track",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, temperature, topk, tracks, start_image=None, clip_img=None, clip_vision_name=None):
        image_2V = (width, height, length, temperature, topk, tracks, start_image, clip_img, clip_vision_name)
        return (image_2V,)




class Stack_WanSoundImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 77, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
                "ref_motion": ("IMAGE", ),
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("SOUNDIMAGE",)
    RETURN_NAMES = ("SoundImage",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, ref_image=None, control_video=None, ref_motion=None, audio_encoder_output=None):
        image_2V = (width, height, length, ref_image, control_video, ref_motion, audio_encoder_output)
        return (image_2V,)


class Stack_WanSoundImageToVideoExtend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "length": ("INT", {"default": 77, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "video_latent": ("LATENT", ),
                "ref_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("SOUNDIMAGE_EX",)
    RETURN_NAMES = ("SoundImage_ex",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, length, video_latent=None, ref_image=None, control_video=None, audio_encoder_output=None):
        image_2V = (length, video_latent, ref_image, control_video, audio_encoder_output)
        return (image_2V,)



class Stack_WanHuMoImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 97, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
                "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("HUMOIMAGE",)
    RETURN_NAMES = ("HuMoImage",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, ref_image=None, audio_encoder_output=None):
        image_2V = (width, height, length, ref_image, audio_encoder_output)
        return (image_2V,)


class Stack_WanPhantomSubjectToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "images": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("PHANTOM",)
    RETURN_NAMES = ("Phantom",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"
    
    def encode(self, width, height, length, images=None):
        image_2V = (width, height, length, images)
        return (image_2V,)



















