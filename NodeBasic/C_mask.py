
#region---------
import torch
import torch.nn.functional as F
import numpy as np

import folder_paths as comfy_paths
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageChops
import os,sys
import comfy.utils

import cv2

from PIL import Image, ImageOps, ImageFilter
import math
import scipy.ndimage
from torchvision.transforms import functional as TF
from comfy_extras.nodes_mask import GrowMask
import folder_paths
from ultralytics import YOLO,settings
from transparent_background import Remover
from tqdm import tqdm
from ..main_unit import *


#endregion-------------------------------------------------------------------------------#


#region---------DetectByLabel检测遮罩--------------------------------------

MODELS_DIR =  comfy_paths.models_dir
sys.path.append(os.path.join(__file__,'../../'))
settings.update({'weights_dir':os.path.join(folder_paths.models_dir,'ultralytics')})


def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_name = os.path.splitext(file)[0]
                file_list.append(file_name)
    return file_list


def createMask(image,x,y,w,h):
    mask = Image.new("L", image.size)
    pixels = mask.load()
    # 遍历指定区域的像素，将其设置为黑色（0 表示黑色）
    for i in range(int(x), int(x + w)):
        for j in range(int(y), int(y + h)):
            pixels[i, j] = 255
    # mask.save("mask.png")
    return mask


def grow(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                            [1, 1, 1],
                            [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def combine(destination, source, x, y):
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1]))

    left, top = (x, y,)
    right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
    visible_width, visible_height = (right - left, bottom - top,)

    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = destination[:, top:bottom, left:right]

    #operation == "subtract":
    output[:, top:bottom, left:right] = destination_portion - source_portion
        
    output = torch.clamp(output, 0.0, 1.0)

    return output

#endregion-----------------



class Mask_Detect_label:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "confidence":("FLOAT", {"default": 0.1, "min": 0.0, "max": 1, "step":0.01, "display": "number"}),
            "model":(get_files_with_extension(os.path.join(folder_paths.models_dir,'ultralytics'),'.pt'),),
            },
            "optional":{ }
        }
    
    RETURN_TYPES = ("MASK","IMAGE",)
    RETURN_NAMES = ("masks","image",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/mask"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,)

    def run(self,image,confidence,model,target_label="",debug="on"):
        target_labels=target_label.split('\n')
        target_labels=[t.strip() for t in target_labels if t.strip()!='']
        model = YOLO(model+'.pt')  
        image=tensor2pil(image)
        image=image.convert('RGB')
        images=[image]
        results = model(images)  
        masks=[]
        names=[]
        grids=[]
        images_debug=[]
        for i in range(len(results)):
            result=results[i]
            img=images[i]
            boxes = result.boxes
            bb=boxes.xyxy.cpu().numpy()
            confs=boxes.conf.cpu().numpy()
            if debug=='on':
                im_bgr = result.plot()
                im_rgb = Image.fromarray(im_bgr[..., ::-1])
                images_debug.append(pil2tensor(im_rgb))
            for j in range(len(bb)):
                name=result.names[boxes[j].cls.item()]
                is_target=True
                if len(target_labels)>0:
                    is_target=False
                    for t in target_labels:
                        if t==name:
                            is_target=True
                if is_target:
                    b=bb[j]
                    conf=confs[j]
                    if conf >= confidence:
                        x,y,xw,yh=b
                        w=xw-x
                        h=yh-y
                        mask=createMask(img,x,y,w,h)
                        mask=pil2tensor(mask)
                        masks.append(mask)
                        names.append(name)
                        grids.append((x,y,w,h))
        if len(masks)==0:
            mask = Image.new("L", image.size)
            mask=pil2tensor(mask)
            masks.append(mask)
            grids.append((0,0,image.size[0],image.size[1]))
            names.append(['-'])
        return (masks,images_debug,)



class Mask_Remove_bg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_img": (["image", "white", "black", "green"],),
            },
            "optional": {
                "mask": ("MASK",),  
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", )
    RETURN_NAMES = ("image", "mask", "invert_mask", "alpha_img",)
    FUNCTION = "removebg"
    CATEGORY = "Apt_Preset/mask"



    def mask_crop(self, image, mask, up=0, down=0, left=0, right=0):
        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)
        mask_array = np.array(mask_pil) > 0
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            return (image, None)
        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
        x0 -= left
        y0 -= up
        x1 += right
        y1 += down
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        return (cropped_image_tensor, cropped_mask_tensor)



    def removebg(self, bg_img, image, threshold,mask=None):
        
        if mask is not None:
            out= self.mask_crop(image, mask, up=0, down=0, left=0, right=0)
            image = out[0]

        remover = Remover()
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba', threshold=threshold)
            out =  pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        invert_mask = 1.0 - mask

        if bg_img == "image":
            image2 = image
        elif bg_img == "white":
            # 创建白色背景
            white_bg = torch.ones_like(img_stack[:, :, :, :3])
            alpha = img_stack[:, :, :, 3:4]
            image2 = alpha * img_stack[:, :, :, :3] + (1 - alpha) * white_bg
        elif bg_img == "black":
            # 创建黑色背景
            black_bg = torch.zeros_like(img_stack[:, :, :, :3])
            alpha = img_stack[:, :, :, 3:4]
            image2 = alpha * img_stack[:, :, :, :3] + (1 - alpha) * black_bg
        elif bg_img == "green":
            # 创建绿色背景
            green_bg = torch.zeros_like(img_stack[:, :, :, :3])
            green_bg[:, :, :, 1] = 1  # 将绿色通道设置为1
            alpha = img_stack[:, :, :, 3:4]
            image2 = alpha * img_stack[:, :, :, :3] + (1 - alpha) * green_bg

        return (image2, mask, invert_mask, img_stack)



class Mask_inpaint_light:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_light": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "bg_light": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "smoothness": ("INT", {"default": 1, "min":0, "max": 150, "step": 1}),

            }
        }

    CATEGORY = "Apt_Preset/mask"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("mask_img", "mask","bg_img")
    FUNCTION = "apply"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, bg_light, mask_light, smoothness ):
        # Clone the original image and mask to avoid modifying them
        image1_input = image.clone()  # Clone for image1 processing
        mask1_input = mask.clone()    # Clone for image1 processing

        # GrayScaler Logic for Image1 (use clones, not the original)
        if image1_input.ndim != 4 or mask1_input.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        if mask1_input.ndim == 3:
            mask1_input = mask1_input.unsqueeze(-1)

        grey_value = 0.5 * bg_light
        image1 = image1_input * mask1_input + (1 - mask1_input) * grey_value

        # Clone the original image and mask again to ensure independence for image2
        image2_input = image.clone()  # Clone for image2 processing
        mask2_input = mask.clone()    # Clone for image2 processing

        # CZeSTGrayoutSubject Logic for Image2 (use clones, not the original)
        subject_mask = mask2_input.reshape((-1, 1, mask2_input.shape[-2], mask2_input.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        target = Image.fromarray((255. * image2_input[0]).cpu().numpy().astype(np.uint8))
        subjectmask = Image.fromarray((255. * subject_mask[0]).cpu().numpy().astype(np.uint8))

        grayall_target = target.convert('L').convert('RGB')
        grayall_target = ImageEnhance.Brightness(grayall_target)
        grayall_target = grayall_target.enhance(mask_light)

        graysubject = ImageChops.darker(grayall_target, subjectmask)
        colorbackground = ImageChops.darker(target, ImageChops.invert(subjectmask))

        grayoutsubject = ImageChops.lighter(colorbackground, graysubject)

        image2 = np.array(grayoutsubject).astype(np.float32) / 255.0
        image2 = torch.from_numpy(image2)[None,]


        mask=tensor2pil(mask)
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)

        return (image2, mask, image1,)



class Mask_image2mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_type": (["White_Balance", "color", "channel", "depth"],),
                "WB_low_threshold": ("INT", {"default": 1, "min": 1, "max": 255, "step": 1}),
                "WB_high_threshold": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),

                "color": ("COLOR",),
                "color_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "channel": (["red", "green", "blue", "alpha"],),
                "depth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "display": "number"}),


                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 32768, "step": 1}),
                "expand": ("INT", {"default": 0, "min": -150, "max": 150, "step": 1}),

            },
        }

    RETURN_TYPES = ( "MASK", "MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "invert_mask", "mask2_img")
    FUNCTION = "image_to_mask"
    CATEGORY = "Apt_Preset/mask"

    def image_to_mask(self, image, WB_low_threshold, WB_high_threshold, blur_radius, mask_type, depth, color, color_threshold, channel, expand):
        
        tapered_corners = True
        
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        out_image = image

        if mask_type == "White_Balance":
            image = 255. * image_np[0]
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image = ImageOps.grayscale(image)
            threshold_filter = lambda x: 255 if x > WB_high_threshold else 0 if x < WB_low_threshold else x
            image = image.convert("L").point(threshold_filter, mode="L")
            image = np.array(image).astype(np.float32) / 255.0
            mask = 1- torch.from_numpy(image)
            #mask2_img = torch.from_numpy(image)[None,]


        elif mask_type == "color":
            images = 255. * image_np
            images = np.clip(images, 0, 255).astype(np.uint8)
            images = [Image.fromarray(img) for img in images]
            images = [np.array(img) for img in images]

            black = [0, 0, 0]
            white = [255, 255, 255]
            new_images = []

            # 将十六进制颜色字符串转换为 RGB 值
            if isinstance(color, str) and color.startswith('#'):
                color = color.lstrip('#')
                rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                color = np.array(rgb, dtype=np.float32)

            for img in images:
                new_image = np.full_like(img, black)

                color_distances = np.linalg.norm(img - color, axis=-1)
                complement_indexes = color_distances <= color_threshold
                new_image[complement_indexes] = white
                new_images.append(new_image)

            new_images = np.array(new_images).astype(np.float32) / 255.0
            new_images = torch.from_numpy(new_images).permute(3, 0, 1, 2)
            mask = new_images[0]

        elif mask_type == "channel":
            channels = ["red", "green", "blue", "alpha"]
            mask = image[:, :, :, channels.index(channel)]


        elif mask_type == "depth":
            bs = image.size()[0]
            width = image.size()[2]
            height = image.size()[1]
            mask1 = torch.zeros((bs, height, width))
            image = upscale(image, 'lanczos', width, height)[0]
            for k in range(bs):
                for i in range(width):
                    for j in range(height):
                        now_depth = image[k][j][i][0].item()
                        if now_depth < depth:
                            mask1[k][j][i] = 1
            mask = mask1


        if blur_radius > 0:
            mask=tensor2pil(mask)
            feathered_image = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            mask=pil2tensor(feathered_image)


        mask = GrowMask().expand_mask(mask, expand, tapered_corners)[0]
        mask2_img = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        invert_mask = 1.0 - mask
        
        invert_mask2_img =1 - mask2_img

        return (mask, invert_mask,  mask2_img )


class Mask_mask2mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0,"max": 100,"step": 0.1}),
                "expand": ("INT", {"default": 0,"min": -100,"max": 100,"step": 1}),
                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "outline_width":("INT", {"default": 10,"min": 1, "max": 50, "step": 1}),

            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Apt_Preset/mask"
    RETURN_TYPES = ("MASK", "MASK", "MASK","IMAGE")
    RETURN_NAMES = ("mask", "invert_mask", "outline_mask", "image")
    FUNCTION = "expand_mask"

    
    def expand_mask(self, mask, expand, tapered_corners, blur_radius, min, max, outline_width,  fill_holes=False):


        mask_max = torch.max(mask)
        mask_max = mask_max if mask_max > 0 else 1
        scaled_mask = (mask / mask_max) * (max - min) + min
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)

        mask = scaled_mask


        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []

        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)

            out.append(output)

        if blur_radius != 0:

            for idx, tensor in enumerate(out):
                pil_image = tensor2pil(tensor.cpu().detach())
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))

                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)

            mask = blurred
            invert_mask = 1.0 - mask

        else:
            mask = torch.stack(out, dim=0)
            invert_mask = 1.0 - mask
        


        m1=grow(mask,outline_width,tapered_corners)
        m2=grow(mask,-outline_width,tapered_corners)
        outline_mask=combine(m1,m2,0,0)

        image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        return (mask, invert_mask, outline_mask, image)



class Mask_transform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": 10240, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": 10240, "step": 1, "display": "number" }),
                "angle": ("INT", { "default": 0, "min": -360, "max": 360, "step": 1, "display": "number" }),
                "duplication_factor": ("INT", { "default": 1, "min": 1, "max": 1000, "step": 1, "display": "number" }),
                "roll": ("BOOLEAN", { "default": False }),
                "incremental": ("BOOLEAN", { "default": False }),
                "padding_mode": (
            [   
                'empty',
                'border',
                'reflection',
                
            ], {
            "default": 'empty'
            }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "Apt_Preset/mask"


    def offset(self, mask, x, y, angle, roll=False, incremental=False, duplication_factor=1, padding_mode="empty"):
        # Create duplicates of the mask batch
        mask = mask.repeat(duplication_factor, 1, 1).clone()

        batch_size, height, width = mask.shape

        if angle != 0 and incremental:
            for i in range(batch_size):
                rotation_angle = angle * (i+1)
                mask[i] = TF.rotate(mask[i].unsqueeze(0), rotation_angle).squeeze(0)
        elif angle > 0:
            for i in range(batch_size):
                mask[i] = TF.rotate(mask[i].unsqueeze(0), angle).squeeze(0)

        if roll:
            if incremental:
                for i in range(batch_size):
                    shift_x = min(x*(i+1), width-1)
                    shift_y = min(y*(i+1), height-1)
                    if shift_x != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_x, dims=1)
                    if shift_y != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_y, dims=0)
            else:
                shift_x = min(x, width-1)
                shift_y = min(y, height-1)
                if shift_x != 0:
                    mask = torch.roll(mask, shifts=shift_x, dims=2)
                if shift_y != 0:
                    mask = torch.roll(mask, shifts=shift_y, dims=1)
        else:
            
            for i in range(batch_size):
                if incremental:
                    temp_x = min(x * (i+1), width-1)
                    temp_y = min(y * (i+1), height-1)
                else:
                    temp_x = min(x, width-1)
                    temp_y = min(y, height-1)
                if temp_x > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((height, temp_x)), mask[i, :, :-temp_x]], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, :-temp_x], (0, temp_x), mode=padding_mode)
                elif temp_x < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :, :temp_x], torch.zeros((height, -temp_x))], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, -temp_x:], (temp_x, 0), mode=padding_mode)

                if temp_y > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((temp_y, width)), mask[i, :-temp_y, :]], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :-temp_y, :], (0, temp_y), mode=padding_mode)
                elif temp_y < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :temp_y, :], torch.zeros((-temp_y, width))], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, -temp_y:, :], (temp_y, 0), mode=padding_mode)
        
        return  (mask,)



class Mask_splitMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks", )
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"

    def separate(self, mask, ignore_threshold=100, index=0):
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < ignore_threshold:
                continue
            segmented_mask = np.zeros_like(binary_mask)
            cv2.drawContours(segmented_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            segmented_masks.append(segmented_mask)
        output_masks = []
        for segmented_mask in segmented_masks:
            numpy_mask = np.array(segmented_mask).astype(np.float32) / 255.0
            i_mask = torch.from_numpy(numpy_mask)
            output_masks.append(i_mask.unsqueeze(0))
        mask = output_masks
        if isinstance(mask, list):
            result = mask[index]
        else:
            result = mask
        if result is None:
            result = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (result,)



class Mask_math:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1":("MASK",),
                "mask2":("MASK",),
                "operation":(["-","+","*","&"],{"default": "+"}),
                "algorithm":(["cv2","torch"],{"default":"cv2"}),
                "invert_mask1":("BOOLEAN",{"default":False}),
                "invert_mask2":("BOOLEAN",{"default":False}),
            }
        }
    CATEGORY = "Apt_Preset/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math"
    def mask_math(self, mask1, mask2, operation, algorithm, invert_mask1, invert_mask2):
        #invert mask
        if invert_mask1:
            mask1 = 1-mask1
        if invert_mask2:
            mask2 = 1-mask2

        #repeat mask
        if mask1.dim() == 2:
            mask1 = mask1.unsqueeze(0)
        if mask2.dim() == 2:
            mask2 = mask2.unsqueeze(0)
        if mask1.shape[0] == 1 and mask2.shape[0] != 1:
            mask1 = mask1.repeat(mask2.shape[0],1,1)
        elif mask1.shape[0] != 1 and mask2.shape[0] == 1:
            mask2 = mask2.repeat(mask1.shape[0],1,1)

        #check cv2
        if algorithm == "cv2":
            try:
                import cv2
            except:
                print("prompt-mask_and_mask_math: cv2 is not installed, Using Torch")
                print("prompt-mask_and_mask_math: cv2 未安装, 使用torch")
                algorithm = "torch"

        #algorithm
        if algorithm == "cv2":
            if operation == "-":
                return (self.subtract_masks(mask1, mask2),)
            elif operation == "+":
                return (self.add_masks(mask1, mask2),)
            elif operation == "*":
                return (self.multiply_masks(mask1, mask2),)
            elif operation == "&":
                return (self.and_masks(mask1, mask2),)
        elif algorithm == "torch":
            if operation == "-":
                return (torch.clamp(mask1 - mask2, min=0, max=1),)
            elif operation == "+":
                return (torch.clamp(mask1 + mask2, min=0, max=1),)
            elif operation == "*":
                return (torch.clamp(mask1 * mask2, min=0, max=1),)
            elif operation == "&":
                mask1 = torch.round(mask1).bool()
                mask2 = torch.round(mask2).bool()
                return (mask1 & mask2, )

    def subtract_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1

    def add_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def multiply_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def and_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1)
        cv2_mask2 = np.array(mask2)
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1






class Mask_combine_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "img1": ("IMAGE",),
                "lay_mode": (["auto", "lay_right", "lay_bottom"], {"default": "auto"}),
                "out_width": ("INT", {"default": 1536, "step":8,}),
                "bj_color": (["#FF0000", "#00FF00","#0000FF", "#FFFFFF"], {"default": "#FF0000"}),
            },
            "optional": {
                "mask1": ("MASK",),
                "img2": ("IMAGE",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2")
    RETURN_NAMES = ("image", "mask", "box2")
    FUNCTION = "add_mask"
    OUTPUT_NODE = True
    CATEGORY = "Apt_Preset/mask"

    def __init__(self):
        self.default_interpolation = cv2.INTER_CUBIC
        self.mask_interpolation = cv2.INTER_NEAREST_EXACT

    def resize(self, img, resolution, interpolation=None):
        """实例方法确保正确绑定"""
        interp = interpolation if interpolation else self.default_interpolation
        return cv2.resize(img, resolution, interpolation=interp)

    def create_image_from_color(self, width, height, color="#FFFFFF"):
        """创建纯色背景图"""
        if isinstance(color, str) and color.startswith('#'):
            color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5) if i+2 <= len(color))
            color = color + (0,) * (3 - len(color))  # 补全RGB
        return np.full((height, width, 3), color[:3], dtype=np.uint8)

    def fit_image(self, image, mask=None, out_width=1536, lay_mode="auto"):
        """核心图像适配逻辑"""
        # 转换为numpy并确保float32类型
        image = image.float().cpu().numpy() if torch.is_tensor(image) else image.astype(np.float32)
        if mask is not None:
            mask = mask.float().cpu().numpy() if torch.is_tensor(mask) else mask.astype(np.float32)

        # 计算目标尺寸
        base_length = int(out_width * 2 / 3)
        half_length = out_width // 2
        target_width = half_length if lay_mode != "lay_bottom" else base_length
        target_height = base_length if lay_mode != "lay_bottom" else half_length

        # 自动判断布局方向
        if lay_mode == "auto":
            h, w = image.shape[:2]
            lay_mode = "lay_bottom" if w > h else "lay_right"
            target_width, target_height = (base_length, half_length) if lay_mode == "lay_bottom" else (half_length, base_length)


        if image.shape[1] < target_width or image.shape[0] < target_height:
            img, mask, w, h = self._handle_upscale(image, mask, target_width, target_height)
        else:
            img, mask, w, h = self._handle_downscale(image, mask, target_width, target_height, lay_mode)[:4]
        
        return img, mask, w, h, lay_mode  # 确保始终返回5个值


    def _handle_upscale(self, image, mask, target_w, target_h):
        """处理放大情况"""
        h, w = image.shape[:2]
        if h > w:  # 竖图
            new_w = int(w * (target_h / h))
            new_h = target_h
        else:  # 横图
            new_w = target_w
            new_h = int(h * (target_w / w))

        # 缩放图像
        resized_img = self.resize(image, (new_w, new_h))
        resized_mask = self.resize(mask, (new_w, new_h), self.mask_interpolation) if mask is not None else None

        # 计算填充
        pad_x = max(0, (target_w - new_w) // 2)
        pad_y = max(0, (target_h - new_h) // 2)

        # 应用填充
        if pad_x > 0 or pad_y > 0:
            resized_img = cv2.copyMakeBorder(
                resized_img, pad_y, target_h - new_h - pad_y,
                pad_x, target_w - new_w - pad_x,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            if resized_mask is not None:
                resized_mask = cv2.copyMakeBorder(
                    resized_mask, pad_y, target_h - new_h - pad_y,
                    pad_x, target_w - new_w - pad_x,
                    cv2.BORDER_CONSTANT, value=0
                )
        
        if resized_mask is None:
            resized_mask = np.zeros((target_h, target_w), dtype=np.float32)
        return resized_img, resized_mask, target_w, target_h

    def _handle_downscale(self, image, mask, target_w, target_h, lay_mode):
        """处理缩小情况"""
        h, w = image.shape[:2]
        scale_ratio = target_w / target_h
        img_ratio = w / h

        # 计算裁剪区域
        if img_ratio > scale_ratio:  # 宽图
            crop_h = h
            crop_w = int(h * scale_ratio)
        else:  # 高图
            crop_w = w
            crop_h = int(w / scale_ratio)

        # 居中裁剪
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        cropped_img = image[y:y+crop_h, x:x+crop_w]
        cropped_mask = mask[y:y+crop_h, x:x+crop_w] if mask is not None else None

        # 缩放到目标尺寸
        resized_img = self.resize(cropped_img, (target_w, target_h))
        if cropped_mask is not None:
            resized_mask = self.resize(cropped_mask, (target_w, target_h), self.mask_interpolation)
        else:
            resized_mask = np.zeros((target_h, target_w), dtype=np.float32)

        return resized_img, resized_mask, target_w, target_h, lay_mode
    
    def add_mask(self, img1, lay_mode, out_width, bj_color, mask1=None, img2=None, mask2=None):
        """主处理函数"""
        # 确保输出长度是8的倍数
        out_width = out_width - (out_width % 8)
        
        # 处理第一张图像
        img1 = img1[0]  # 取batch中的第一张
        mask1 = mask1[0] if mask1 is not None else torch.zeros(img1.shape[:2])
        img1, mask1, img1_w, img1_h, lay_mode = self.fit_image(img1, mask1, out_width, lay_mode)

        # 处理第二张图像
        if img2 is not None:
            img2 = img2[0]
            mask2 = mask2[0] if mask2 is not None else torch.zeros(img2.shape[:2])
            img2, mask2, img2_w, img2_h = self.fit_image(img2, mask2, out_width, lay_mode)[:4]
        else:
            # 创建纯色背景
            img2 = self.create_image_from_color(img1_w, img1_h, bj_color)
            img2 = torch.from_numpy(img2).float() / 255.0
            mask2 = torch.ones(img1.shape[:2]) if mask2 is None else mask2[0]
            img2_w, img2_h = img1_w, img1_h

        # 合并图像
        if lay_mode == "lay_right":
            combined_img = np.hstack((img1, img2))
            combined_mask = np.hstack((mask1, mask2))
            offset_x, offset_y = img1_w, 0  # 偏移量为图像1的宽度
        else:
            combined_img = np.vstack((img1, img2))
            combined_mask = np.vstack((mask1, mask2))
            offset_x, offset_y = 0, img1_h  # 偏移量为图像1的高度

        # 转换为torch tensor
        combined_img = torch.from_numpy(combined_img).unsqueeze(0)  # 添加batch维度
        combined_mask = torch.from_numpy(combined_mask).unsqueeze(0)
        
        w=img2_w
        h=img2_h
        x=offset_x
        y=offset_y
        box2 = (w, h, x, y)  # w, h, x, y
        return (combined_img, combined_mask, box2)



class Mask_combine_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["top-bottom", "left-right"], {"default": "left-right"}),
                "out_heigh": ("INT", {"default": 0, "max": 88888, "min": 0, "step": 8, "tooltip": "height = 0 means not set"}),
            },
            "optional": {
                "img1": ("IMAGE",),
                "mask1": ("MASK",),
                "img2": ("IMAGE",),
                "mask2": ("MASK",),
            },
        }
    
    CATEGORY = "Apt_Preset/mask"
    FUNCTION = "make"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX2")
    RETURN_NAMES = ("combine_img", "combine_mask", "box2")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False, False)

    @staticmethod
    def pil2tensor(image):
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:
            return torch.from_numpy(np_image[None, ...])
        elif np_image.ndim == 3:
            return torch.from_numpy(np_image[None, ...])
        raise ValueError(f"Unsupported image dimensions: {np_image.ndim}")

    def fillMask(self, width, height, mask, box=(0, 0), color=0):
        bg = Image.new("L", (width, height), color)
        bg.paste(mask, box, mask)
        return bg

    def emptyImage(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return torch.cat((r, g, b), dim=-1)

    def make(self, direction, out_heigh=0, img1=None, img2=None, mask1=None, mask2=None):
        # 初始化处理
        if img2 is None:
            img2 = self.emptyImage(img1.shape[2], img1.shape[1])
            mask2 = torch.ones((1, img1.shape[1], img1.shape[2]), dtype=torch.float32, device=img1.device)
        elif img2 is not None and mask2 is None:
            mask2 = torch.zeros((1, img2.shape[1], img2.shape[2]), dtype=torch.float32, device=img2.device)
        
        # 调整尺寸逻辑
        if out_heigh > 0:
            _, img2_h, img2_w, _ = img2.shape
            h = out_heigh if direction == 'left-right' else int(img2_h * (out_heigh / img2_w))
            w = out_heigh if direction == 'top-bottom' else int(img2_w * (out_heigh / img2_h))
            img2 = img2.movedim(-1, 1)
            img2 = comfy.utils.common_upscale(img2, w, h, 'bicubic', 'disabled')
            img2 = img2.movedim(1, -1)
            mask2 = F.interpolate(mask2.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)


            orig_img2 = tensor2pil(img2)
            orig_mask2 = tensor2pil(mask2).convert('L')
            orig_mask2 = orig_mask2.resize(orig_img2.size)
            mask2 = pil2tensor(orig_mask2)

        _, img1_h, img1_w, _ = img1.shape
        _, img2_h, img2_w, _ = img2.shape

        # resize 图像以匹配尺寸
        if img1_h != img2_h and img1_w != img2_w:
            width, height = img2_w, img2_h
            if direction == 'left-right' and img1_h != img2_h:
                scale_factor = img2_h / img1_h
                width = round(img1_w * scale_factor)
            elif direction == 'top-bottom' and img1_w != img2_w:
                scale_factor = img2_w / img1_w
                height = round(img1_h * scale_factor)
            img1 = img1.movedim(-1, 1)
            img1 = comfy.utils.common_upscale(img1, width, height, 'bicubic', 'disabled')
            img1 = img1.movedim(1, -1)

        if mask1 is None:
            mask1 = torch.full((1, img1.shape[1], img1.shape[2]), 0, dtype=torch.float32, device="cpu")

        orig_img1 = tensor2pil(img1)
        orig_mask1 = tensor2pil(mask1).convert('L')

        if orig_mask1.size != orig_img1.size:
            orig_mask1 = orig_mask1.resize(orig_img1.size)

        img1_w, img1_h = orig_img1.size
        img1 = pil2tensor(orig_img1)
        image = torch.cat((img1, img2), dim=2) if direction == 'left-right' else torch.cat((img1, img2), dim=1)

        context_mask = self.fillMask(image.shape[2], image.shape[1], orig_mask1)
        orig_mask2 = tensor2pil(mask2).convert('L')
        x = img1_w if direction == 'left-right' else 0
        y = img1_h if direction == 'top-bottom' else 0
        mask = self.fillMask(image.shape[2], image.shape[1], orig_mask2, (x, y))

        combine_mask = Image.new("L", (image.shape[2], image.shape[1]), 0)
        combine_mask.paste(context_mask, (0, 0))
        combine_mask.paste(mask.crop((x, y, x + orig_mask2.width, y + orig_mask2.height)), (x, y))

        combine_mask = pil2tensor(combine_mask)
        mask = pil2tensor(mask)
        context_mask = pil2tensor(context_mask)

        w=img2_w 
        h=img2_h
        box2 = (w, h, x, y)  # w, h, x, y
        return (image, combine_mask, box2)

