import nodes
import node_helpers
import torch
import comfy
import folder_paths
from ..main_unit import *



def clip_vision_output_encode(clip_vision_name, image):
    clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name)
    clip_vision = comfy.clip_vision.load(clip_path)
    output = clip_vision.encode_image(image, crop=True)
    return (output,)

#region----------------源码 ---------------------

class WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, control_video=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,16:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(control_video[:, :, :, :3])
            concat_latent[:,:16,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                             "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_output=None):
        flfv = WanFirstLastFrameToVideo()
        return flfv.encode(positive, negative, vae, width, height, length, batch_size, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_output)


class WanVaceToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                },
                "optional": {"control_video": ("IMAGE", ),
                             "control_masks": ("MASK", ),
                             "reference_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength, control_video=None, control_masks=None, reference_image=None):
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent, trim_latent)


class TrimVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "trim_amount": ("INT", {"default": 0, "min": 0, "max": 99999}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/video"

    EXPERIMENTAL = True

    def op(self, samples, trim_amount):
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1[:, :, trim_amount:]
        return (samples_out,)


class WanCameraImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "camera_conditions": ("WAN_CAMERA_EMBEDDING", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, camera_conditions=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if camera_conditions is not None:
            positive = node_helpers.conditioning_set_values(positive, {'camera_conditions': camera_conditions})
            negative = node_helpers.conditioning_set_values(negative, {'camera_conditions': camera_conditions})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanPhantomSubjectToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"images": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative_text", "negative_img_text", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, images):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        cond2 = negative
        if images is not None:
            images = comfy.utils.common_upscale(images[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in images:
                latent_images += [vae.encode(i.unsqueeze(0)[:, :, :, :3])]
            concat_latent_image = torch.cat(latent_images, dim=2)

            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            cond2 = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, cond2, negative, out_latent)


#endregion----------------源码 ---------------------




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
            samples = TrimVideoLatent().op(samples, trim_latent)[0]

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



class Stack_WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),
                                    
                },
                "optional": {    
                    "clip_img": ("IMAGE", ),
                    
                    "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("IMAGE_2V",)
    RETURN_NAMES = ("Image_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"
    
    def encode(self, width, height, length, start_image=None, clip_img=None, clip_vision_name=None):
        #Image_2V =(width, height, length, start_image, clip_img, clip_vision_name)
        return ((width, height, length, start_image, clip_img, clip_vision_name),)


class Stack_WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),

                },
                "optional": {                    
                    "clip_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("FCTL_2V",)
    RETURN_NAMES = ("FunControl_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, control_video=None, clip_img=None, clip_vision_name=None): 

        return ((width, height, length, start_image, control_video, clip_img, clip_vision_name),)


class Stack_WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "clip_vision_start": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                    "clip_vision_end": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),

                },
                "optional": {
                    "clip_start_img": ("IMAGE", ),
                    "clip_end_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "end_image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("FFL_2V",)
    RETURN_NAMES = ("FirstLast_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, end_image=None, clip_start_img=None, clip_end_img=None, clip_vision_start=None, clip_vision_end=None):

        return ((width, height, length, start_image, end_image, clip_start_img, clip_end_img, clip_vision_start, clip_vision_end),)


class Stack_WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),
                                    
                },
                "optional": {                    
                    "clip_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "end_image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("FIP_2V",)
    RETURN_NAMES = ("FunInpaint_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, end_image=None, clip_img=None, clip_vision_name=None):
        return ((width, height, length, start_image, end_image, clip_img, clip_vision_name),)


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

    RETURN_TYPES = ("VACE_2V",)
    RETURN_NAMES = ("vace_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length,strength, control_video=None, control_masks=None,

            reference_image=None):
        # 返回包含新字段的元组
        return (
            (width, height, length, strength, control_video, control_masks, reference_image),
        )




class Stack_WanCameraImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                },
                "optional": {
                "clip_img": ("IMAGE", ),
                "start_image": ("IMAGE", ),
                "camera_conditions": ("WAN_CAMERA_EMBEDDING", ),
                }
                }
    RETURN_TYPES = ("CAME_2V",)
    RETURN_NAMES = ("cameral_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None,clip_img=None, clip_vision_name=None, camera_conditions=None):


        #cameral_2V = ( width, height, length, start_image,clip_img,clip_vision_name, camera_conditions)
        return (( width, height, length, start_image,clip_img,clip_vision_name, camera_conditions),)



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
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "Image_2V": ("IMAGE_2V", ),
                "vace_2V": ("VACE_2V", ),
                "FunControl_2V": ("FCTL_2V", ),
                "FirstLast_2V": ("FFL_2V", ),
                "cameral_2V": ("CAME_2V", ),

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT","INT")
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent","vace_trim_latent")
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"


    def merge(self, context=None, model=None, positive=None, negative=None, Image_2V=None, FunControl_2V=None, FirstLast_2V=None, vace_2V=None, cameral_2V=None):
        
        clip = context.get("clip", None)
        vae = context.get("vae")
        latent = None
        trim_latent = 0  # 初始化trim_latent默认值
        
        if model is None:
            model = context.get("model")

        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")


        if Image_2V is not None:
            width, height, length, start_image, clip_img, clip_vision_name = Image_2V
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanImageToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, clip_vision_output=clip_vision_output)


        if FunControl_2V is not None:
            width, height, length, start_image, control_video, clip_img, clip_vision_name = FunControl_2V
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanFunControlToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, clip_vision_output=clip_vision_output, control_video=control_video)


        if FirstLast_2V is not None:
            width, height, length, start_image, end_image, clip_start_img, clip_end_img, clip_vision_start, clip_vision_end = FirstLast_2V
            clip_vision_start_image = None
            clip_vision_end_image = None
            if clip_start_img is not None and clip_vision_start is not None:
                clip_vision_start_image = clip_vision_output_encode(clip_vision_start, clip_start_img)[0]
            if clip_end_img is not None and clip_vision_end is not None:
                clip_vision_end_image = clip_vision_output_encode(clip_vision_end, clip_end_img)[0]


            positive, negative, latent = WanFirstLastFrameToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_start_image, clip_vision_end_image=clip_vision_end_image)


        if vace_2V is not None:
            width, height, length, strength, control_video, control_masks, reference_image = vace_2V       
            positive, negative, latent, trim_latent = WanVaceToVideo().encode(
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



        if cameral_2V is not None:
            width, height, length, start_image,clip_img, clip_vision_name, camera_conditions = cameral_2V 
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]
            positive, negative, latent = WanCameraImageToVideo().encode (positive, negative, vae, width, height, length, 1, start_image=start_image, clip_vision_output=clip_vision_output, camera_conditions=camera_conditions)


        context = new_context(context, clip=clip, positive=positive, latent=latent, negative=negative, model=model)

        return (context, model, positive, negative, latent, trim_latent )











