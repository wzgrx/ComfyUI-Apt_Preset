
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import torch
import os
import sys
import folder_paths
import random
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from comfy import latent_formats
import json
import latent_preview
from comfy.cli_args import args
import numpy as np
import inspect

import traceback
import itertools
from typing import Optional
import comfy
from typing import Any
from server import PromptServer
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import hashlib
from comfy_extras.nodes_mask import ImageCompositeMasked
import re





from ..main_unit import *




#优先从当前文件所在目录下的 comfy 子目录中查找模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))  

def updateTextWidget(node, widget, text):
    PromptServer.instance.send_sync("view_Data_text_processed", {"node": node, "widget": widget, "text": text})




#region-----------------------收纳-------------------------------------------------------#




class view_LatentAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"latent": ("LATENT",),
                    "base_model": (["SD15","SDXL"],),
                    "preview_method": (["auto","taesd","latent2rgb"],),
                    },
            "hidden": {"prompt": "PROMPT",
                        "extra_pnginfo": "EXTRA_PNGINFO",
                        "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = True
    FUNCTION = "lpreview"
    CATEGORY = "Apt_Preset/View_IO"

    def lpreview(self, latent, base_model, preview_method, prompt=None, extra_pnginfo=None, my_unique_id=None):
        previous_preview_method = args.preview_method
        if preview_method == "taesd":
            temp_previewer = latent_preview.LatentPreviewMethod.TAESD
        elif preview_method == "latent2rgb":
            temp_previewer = latent_preview.LatentPreviewMethod.Latent2RGB
        else:
            temp_previewer = latent_preview.LatentPreviewMethod.Auto

        results = list()

        try:
            args.preview_method=temp_previewer
            preview_format = "PNG"
            load_device=comfy.model_management.vae_offload_device()
            latent_format = {"SD15":latent_formats.SD15,
                            "SDXL":latent_formats.SDXL}[base_model]()

            result=[]
            for i in range(len(latent["samples"])):
                x=latent.copy()
                x["samples"] = latent["samples"][i:i+1].clone()
                x_sample = x["samples"]
                x_sample = x_sample /  {"SD15":6,"SDXL":7.5}[base_model]

                img = latent_preview.get_previewer(load_device, latent_format).decode_latent_to_preview(x_sample)
                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("",folder_paths.get_temp_directory(), img.height, img.width)
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                file = "latent_"+"".join(random.choice("0123456789") for x in range(8))+".png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                results.append({"filename": file, "subfolder": subfolder, "type": "temp"})

        finally:
            # Restore global changes
            args.preview_method=previous_preview_method

        return {"result": (latent,), "ui": { "images": results } }



class view_mask(SaveImage):
    
    def __init__(self):
        pass

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },  
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/View_IO"
    DESCRIPTION = "show mask"
    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)




class IO_load_anyimage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "fill_color": (["None", "white", "gray", "black"], {}),
                "smooth": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    RETURN_TYPES = ('IMAGE', 'MASK',)
    FUNCTION = "get_transparent_image"
    CATEGORY = "Apt_Preset/View_IO"
    
    def get_transparent_image(self, file_path, smooth, seed, fill_color):
        try:
            if os.path.isdir(file_path):
                images = []
                for filename in os.listdir(file_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(file_path, filename)
                        image = Image.open(img_path).convert('RGBA')
                        images.append(image)
                
                if not images:
                    return None, None
                
                target_size = images[0].size
                
                resized_images = []
                for image in images:
                    if image.size != target_size:
                        image = image.resize(target_size, Image.BILINEAR)
                    resized_images.append(image)
                
                batch_images = np.stack([np.array(img) for img in resized_images], axis=0).astype(np.float32) / 255.0
                batch_tensor = torch.from_numpy(batch_images)
                
                mask_tensor = None
                
                return batch_tensor, mask_tensor        
            else:
                file_path = file_path.strip('"')
                image = Image.open(file_path)
                if image is not None:
                    image_rgba = image.convert('RGBA')
                    image_rgba.save(file_path.rsplit('.', 1)[0] + '.png')
            
                    mask = np.array(image_rgba.getchannel('A')).astype(np.float32) / 255.0
                    if smooth:
                        mask = 1.0 - mask
                    mask_tensor = torch.from_numpy(mask)[None, None, :, :]
            
                    if fill_color == 'white':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (255, 255, 255, 255))
                    elif fill_color == 'gray':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (128, 128, 128))
                    elif fill_color == 'black':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (0, 0, 0))
                    elif fill_color == 'None':
                        pass
                    else:
                        raise ValueError("Invalid fill color specified.")
            
                    image_np = np.array(image_rgba).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None, :, :, :]
            
                    return (image_tensor, mask_tensor)
            
        except Exception as e:
            print(f"出错请重置节点：{e}")
        return None, None



class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False



class view_combo:     # web_node/view_Data_text.js

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "text"}),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                    "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
                    },
            "hidden":{
                "workflow_prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING", AlwaysEqualProxy('*'))
    RETURN_NAMES = ("STRING", "COMBO")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_strings"
    CATEGORY = "Apt_Preset/View_IO"

    def generate_strings(self, prompt, start_index, max_rows, workflow_prompt=None, my_unique_id=None):
        lines = prompt.split('\n')

        start_index = max(0, min(start_index, len(lines) - 1))
        end_index = min(start_index + max_rows, len(lines))
        rows = lines[start_index:end_index]

        return (rows, rows)



class view_node_Script:
    def __init__(self):
        self.node_list = []
        self.custom_node_list = []
        self.update_node_list()

    def update_node_list(self):
        """Scan and update the list of available nodes in ComfyUI"""
        try:
            import nodes
            self.node_list = []
            self.custom_node_list = []
            
            for node_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
                try:
                    # Determine if it's a custom node
                    module = inspect.getmodule(node_class)
                    module_path = getattr(module, '__file__', '')
                    is_custom = 'custom_nodes' in module_path

                    node_info = {
                        'name': node_name,
                        'class_name': node_class.__name__,
                        'category': getattr(node_class, 'CATEGORY', 'Uncategorized'),
                        'description': getattr(node_class, 'DESCRIPTION', ''),
                        'is_custom': is_custom
                    }
                    
                    self.node_list.append(node_info)
                    if is_custom:
                        self.custom_node_list.append(node_info)
                except Exception as e:
                    logging.error(f"Error processing node {node_name}: {str(e)}")
                    continue
            
            # Sort nodes alphabetically
            self.node_list.sort(key=lambda x: x['name'])
            self.custom_node_list.sort(key=lambda x: x['name'])
            
        except Exception as e:
            logging.error(f"Error updating node list: {str(e)}")
            traceback.print_exc()

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import nodes
            node_names = sorted(list(nodes.NODE_CLASS_MAPPINGS.keys()))
            if not node_names:
                node_names = ["No nodes found"]
                
            return {
                "required": {
                    "mode": (["All Nodes", "Custom Nodes Only", "Built-in Nodes Only"], {
                        "default": "All Nodes"
                    }),
                    "view_mode": (["List Nodes", "View Source Code", "Usage Guide"], {
                        "default": "List Nodes"
                    }),
                    "selected_node": (node_names, {
                        "default": node_names[0]
                    }),
                    "search": ("STRING", {
                        "default": "",
                        "multiline": False
                    }),
                    "show_all": ("BOOLEAN", {
                        "default": True,
                        "label": "Show All Nodes"
                    }),
                    "refresh_list": ("BOOLEAN", {
                        "default": False,
                        "label": "Refresh Node List"
                    })
                }
            }
        except Exception as e:
            print(f"Error in INPUT_TYPES: {str(e)}")
            return {
                "required": {
                    "mode": (["All Nodes", "Custom Nodes Only", "Built-in Nodes Only"], {"default": "All Nodes"}),
                    "view_mode": (["List Nodes", "View Source Code", "Usage Guide"], {"default": "List Nodes"}),
                    "search": ("STRING", {"default": "", "multiline": False}),
                    "show_all": ("BOOLEAN", {"default": True, "label": "Show All Nodes"}),
                    "refresh_list": ("BOOLEAN", {"default": False, "label": "Refresh Node List"})
                }
            }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("node_info", "node_source",)
    FUNCTION = "find_script"
    CATEGORY = "Apt_Preset/View_IO"

    def get_node_source_code(self, node_name):
        """Get the source code of a node"""
        try:
            import nodes
            import inspect
            import os

            # Get the node class
            node_class = nodes.NODE_CLASS_MAPPINGS.get(node_name)
            if not node_class:
                return f"Node '{node_name}' not found"

            # Get the module
            module = inspect.getmodule(node_class)
            if not module:
                return f"Could not find module for {node_name}"

            # Get file path
            try:
                file_path = inspect.getfile(module)
            except TypeError:
                return f"Could not determine file path for {node_name}"

            # Read entire file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

            # Find the class definition
            class_def = f"class {node_class.__name__}:"
            class_start = file_content.find(class_def)
            
            if class_start == -1:
                return f"Could not find class definition for {node_name}"

            # Extract class source code
            lines = file_content[class_start:].split('\n')
            class_lines = []
            indent_level = None

            for line in lines:
                # Determine initial indent level
                if indent_level is None:
                    if line.strip().startswith('class'):
                        indent_level = len(line) - len(line.lstrip())
                    continue

                # Check if we've reached the end of the class
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break

                class_lines.append(line)

            # Construct formatted output
            source_output = f"=== Node: {node_name} ===\n"
            source_output += f"File: {file_path}\n\n"
            source_output += "=== Source Code ===\n"
            source_output += "\n".join(class_lines)

            return source_output

        except Exception as e:
            return f"Error retrieving source code: {str(e)}"

    def find_script(self, mode, view_mode, selected_node, search, show_all, refresh_list):
        """Main function to find and return selected node"""
        try:
            # Refresh node list if requested
            if refresh_list:
                self.update_node_list()

            # Handle source code view mode
            if view_mode == "View Source Code":
                if selected_node:
                    source_code = self.get_node_source_code(selected_node)
                    return f"Source Code for {selected_node}", source_code
                return "No node selected", "Please select a node to view its source code"

            # Default fallback
            return "Node Source Finder", "Select a node and choose 'View Source Code' mode"

        except Exception as e:
            logging.error(f"Error in find_script: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}", traceback.format_exc()



class IPA_clip_vision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
            "image": ("IMAGE",),
        }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "combined_process"

    CATEGORY = "Apt_Preset/chx_IPA"

    def combined_process(self, clip_name, image):
        # 加载 CLIP Vision 模型
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")
        
        output = clip_vision.encode_image(image, crop="center")
        return (output,)



#region----------------------cache -clear all-------------------------------------------------------#

class TaggedCache:
    def __init__(self, tag_settings: Optional[dict]=None):
        self._tag_settings = tag_settings or {}  
        self._data = {}

    def __getitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        raise KeyError(f'Key `{key}` does not exist')

    def __setitem__(self, key, value: tuple):

        for tag_data in self._data.values():
            if key in tag_data:
                tag_data.pop(key, None)
                break

        tag = value[0]
        if tag not in self._data:

            try:
                from cachetools import LRUCache
                default_size = 20
                if 'ckpt' in tag:
                    default_size = 5
                elif tag in ['latent', 'image']:
                    default_size = 100
                self._data[tag] = LRUCache(maxsize=self._tag_settings.get(tag, default_size))
            except (ImportError, ModuleNotFoundError):
                self._data[tag] = {}
        self._data[tag][key] = value

    def __delitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                del tag_data[key]
                return
        raise KeyError(f'Key `{key}` does not exist')

    def __contains__(self, key):
        return any(key in tag_data for tag_data in self._data.values())

    def items(self):
        yield from itertools.chain(*map(lambda x :x.items(), self._data.values()))

    def get(self, key, default=None):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        return default

    def clear(self):
        self._data = {}

cache_settings = {}
cache = TaggedCache(cache_settings)
cache_count = {}

def update_cache(k, tag, v):
    cache[k] = (tag, v)
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0
        cache_count[k] = cnt
    else:
        cache_count[k] += 1
def remove_cache(key):
    global cache
    if key == '*':
        cache = TaggedCache(cache_settings)
    elif key in cache:
        del cache[key]
    else:
        print(f"invalid {key}")



class IO_clear_cache:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any_type, {}),
        }, "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "Apt_Preset/View_IO"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        remove_cache('*')
        return (anything,)

#endregion-----------------------clear all-------------------------------------------------------#



class IO_inputbasic:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("int", "float", "string")
    FUNCTION = "convert_number_types"
    CATEGORY = "Apt_Preset/View_IO"
    def convert_number_types(self, input):
        try:
            float_num = float(input)
            int_num = int(float_num)
            str_num = input
        except ValueError:
            return (None, None, input)
        return (int_num, float_num, str_num)



class view_Data:   # web_node/view_Data_text.js

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any": (anyType, {"forceInput": True}),
                "data": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    INPUT_IS_LIST = (True,)
    OUTPUT_NODE = True

    CATEGORY = "Apt_Preset/View_IO"
    FUNCTION = "process"

    def process(self, data, unique_id, any=None):
        displayText = self.render(any)

        updateTextWidget(unique_id, "data", displayText)
        return {"ui": {"data": displayText}}

    def render(self, any):
        if not isinstance(any, list):
            return str(any)

        listLen = len(any)

        if listLen == 0:
            return ""

        if listLen == 1:
            return str(any[0])

        result = "List:\n"

        for i, element in enumerate(any):
            result += f"- {str(any[i])}\n"

        return result



class view_bridge_Text:    # web_node/view_Data_text.js

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "text": ("STRING", {"forceInput": True}),
                "display": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True

    CATEGORY = "Apt_Preset/View_IO"
    FUNCTION = "process"

    def process(self, text, display, unique_id):
        displayText = self.render(text)

        updateTextWidget(unique_id, "display", displayText)
        return {"ui": {"display": displayText}, "result": (text,)}

    def render(self, input):
        if not isinstance(input, list):
            return input

        listLen = len(input)

        if listLen == 0:
            return ""

        if listLen == 1:
            return input[0]

        result = "List:\n"

        for i, element in enumerate(input):
            result += f"- {input[i]}\n"

        return result




class view_GetLength:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}), 
            },
        }
    
    TITLE = "Get Length"
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("length", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/View_IO"
    OUTPUT_NODE = True

    def run(self, ANY):
        length = len(ANY)
        return { "ui": {"text": (f"{length}",)}, "result": (length, ) }



class view_GetShape:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor" : ("IMAGE,LATENT,MASK", {}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Get Shape"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/View_IO"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # if input_types["tensor"] not in ("IMAGE", "LATENT", "MASK"):
        #     return "Input must be an IMAGE or LATENT or MASK type"
        return True

    def run(self, tensor, unique_id, prompt, extra_pnginfo):  
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        link_id = cur_node["inputs"][0]["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_type = in_node["outputs"][in_socket_id]["type"]
        
        B, H, W, C = 1, 1, 1, 1
        # IMAGE: [B,H,W,C]
        # LATENT: ["samples"][B,C,H,W]
        # MASK: [H,W] or [B,C,H,W]
        if input_type == "IMAGE":
            B, H, W, C = tensor.shape
        elif input_type == "LATENT" or (type(tensor) is dict and "samples" in tensor):
            B, C, H, W = tensor["samples"].shape
            H *= 8
            W *= 8
        else:  # MASK or type deleted IMAGE
            shape = tensor.shape
            if len(shape) == 2:  # MASK
                H, W = shape
            elif len(shape) == 3:  # MASK
                B, H, W = shape
            elif len(shape) == 4:
                if shape[3] <= 4:  # IMAGE?
                    B, H, W, C = tensor.shape
                else:  # MASK
                    B, C, H, W = shape
        return { "ui": {"text": (f"{W}, {H}, {B}, {C}",)}, "result": (W, H, B, C) }



class view_GetWidgetsValues:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    TITLE = "Get Widgets Values"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("LIST", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/View_IO"
    OUTPUT_NODE = True

    def run(self, ANY, unique_id, prompt, extra_pnginfo):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        link_id = cur_node["inputs"][0]["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        return { "ui": {"text": (f"{in_node['widgets_values']}",)}, "result": (in_node["widgets_values"], ) }



class view_bridge_image:
    def __init__(self):
        self.image_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
            "optional": {
                "mask": ("MASK",),  # 可选原始遮罩
                "operation": (["+", "-", "*", "&", "None"], {"default": "+"}),  # 新增 "None"
                "image_update": ("IMAGE_FILE",)  # 用户编辑后的图像
            }
        }

    CATEGORY = "Apt_Preset/View_IO"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "edit"
    OUTPUT_NODE = True

    def edit(self, image, mask=None, operation="None", image_update=None):
        if self.image_id is None:
            self.image_id = tensor_to_hash(image)
            image_update = None
        else:
            image_id = tensor_to_hash(image)
            if image_id != self.image_id:
                image_update = None
                self.image_id = image_id

        # ==========【优先使用 image_update 中的图像】==========
        if image_update is not None and 'images' in image_update:
            images = image_update['images']
            filename = images[0]['filename']
            subfolder = images[0]['subfolder']
            type = images[0]['type']
            name, base_dir = folder_paths.annotated_filepath(filename)

            if type.endswith("output"):
                base_dir = folder_paths.get_output_directory()
            elif type.endswith("input"):
                base_dir = folder_paths.get_input_directory()
            elif type.endswith("temp"):
                base_dir = folder_paths.get_temp_directory()

            image_path = os.path.join(base_dir, subfolder, name)
            img = node_helpers.pillow(Image.open, image_path)
        else:
            # ==========【否则使用 preview_image】==========
            if mask is not None:
                try:
                    masked_result = generate_masked_black_image(image, mask)
                    preview_image = masked_result["result"][0]
                except Exception as e:
                    print(f"[Error] Failed to apply mask for preview: {e}")
                    preview_image = image
            else:
                preview_image = image

            image_path, images = create_temp_file(preview_image)
            img = node_helpers.pillow(Image.open, image_path)

        # ==========【从图像中提取 mask】==========
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_pil = i.convert("RGB")

            if len(output_masks) == 0:
                w = image_pil.size[0]
                h = image_pil.size[1]

            if image_pil.size[0] != w or image_pil.size[1] != h:
                continue

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1. - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_masks.append(mask_tensor.unsqueeze(0))

        if len(output_masks) > 1 and img.format not in excluded_formats:
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_mask = output_masks[0] if output_masks else torch.zeros_like(image[0, :, :, 0])

        # ==========【新增 Mask 运算逻辑】==========
        mask1 = mask
        mask2 = output_mask

        # 如果没有 mask1 或 operation 为 None，则直接返回 mask2
        if mask1 is None or operation == "None":
            result = mask2
        else:
            invert_mask1 = False
            invert_mask2 = False

            if invert_mask1:
                mask1 = 1 - mask1
            if invert_mask2:
                mask2 = 1 - mask2

            if mask1.dim() == 2:
                mask1 = mask1.unsqueeze(0)
            if mask2.dim() == 2:
                mask2 = mask2.unsqueeze(0)

            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            if mask1.shape != (b, h, w):
                mask1 = torch.zeros((b, h, w), dtype=mask1.dtype, device=mask1.device)
            if mask2.shape != (b, h, w):
                mask2 = torch.zeros((b, h, w), dtype=mask2.dtype, device=mask2.device)

            algorithm = "cv2"
            if algorithm == "cv2":
                try:
                    import cv2
                except ImportError:
                    print("[Info] cv2 not installed, using Torch mode for mask math.")
                    algorithm = "torch"

            if algorithm == "cv2":
                if operation == "-":
                    result = self.subtract_masks(mask1, mask2)
                elif operation == "+":
                    result = self.add_masks(mask1, mask2)
                elif operation == "*":
                    result = self.multiply_masks(mask1, mask2)
                elif operation == "&":
                    result = self.and_masks(mask1, mask2)
                else:
                    result = mask2  # 默认操作为 mask2
            else:
                if operation == "-":
                    result = torch.clamp(mask1 - mask2, min=0, max=1)
                elif operation == "+":
                    result = torch.clamp(mask1 + mask2, min=0, max=1)
                elif operation == "*":
                    result = torch.clamp(mask1 * mask2, min=0, max=1)
                elif operation == "&":
                    result = (torch.round(mask1).bool() & torch.round(mask2).bool()).float()
                else:
                    result = mask2  # 默认操作为 mask2

        # ==========【返回结果】==========
        return {"ui": {"images": images}, "result": (image, result)}



    @staticmethod
    def subtract_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for this operation but is not installed.")

        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def add_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for this operation but is not installed.")

        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def multiply_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for this operation but is not installed.")

        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def and_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for this operation but is not installed.")

        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            print("Warning: The two masks have different shapes")
            return mask1




#endregion-----------------------旧-------------------------------------------------------#.



class view_Mask_And_Img(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()  
    RETURN_NAMES = ()
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/View_IO"

    def execute(self, mask_opacity, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            color_list = [0, 0, 0]  # 黑色
            mask_image[:, :, :, 0] = color_list[0] / 255  # Red channel
            mask_image[:, :, :, 1] = color_list[1] / 255  # Green channel
            mask_image[:, :, :, 2] = color_list[2] / 255  # Blue channel

            preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)

        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)



class IO_adjust_image:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f.name for f in Path(input_dir).iterdir() if f.is_file()]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "max_dimension": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "size_option": (["No Change", "Custom", "Million Pixels", "Small", "Medium", "Large", 
                                "480P-H(vid 4:3)", "480P-V(vid 3:4)", "720P-H(vid 16:9)", "720P-V(vid 9:16)", "832×480", "480×832"], 
                                {"default": "No Change"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "load_image"
    CATEGORY = "Apt_Preset/View_IO"

    def IS_CHANGED(): return float("NaN")

    def load_image(self, image, max_dimension, size_option):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        W, H = img.size
        aspect_ratio = W / H

        def get_target_size():
            if size_option == "No Change":
                # No resizing or cropping, just return the original size
                return W, H
            elif size_option == "Million Pixels":
                return self._resize_to_million_pixels(W, H)
            elif size_option == "Custom":
                ratio = min(max_dimension / W, max_dimension / H)
                return round(W * ratio), round(H * ratio)
            
            size_options = {
                "Small": (
                    (768, 512) if aspect_ratio >= 1.23 else
                    (512, 768) if aspect_ratio <= 0.82 else
                    (768, 768)
                ),
                "Medium": (
                    (1216, 832) if aspect_ratio >= 1.23 else
                    (832, 1216) if aspect_ratio <= 0.82 else
                    (1216, 1216)
                ),
                "Large": (
                    (1600, 1120) if aspect_ratio >= 1.23 else
                    (1120, 1600) if aspect_ratio <= 0.82 else
                    (1600, 1600)
                ),
                "Million Pixels": self._resize_to_million_pixels(W, H),  # Million Pixels option
                "480P-H(vid 4:3)": (640, 480),  # 480P-H, 640x480
                "480P-V(vid 3:4)": (480, 640),  # 480P-V, 480x640
                "720P-H(vid 16:9)": (1280, 720),  # 720P-H, 1280x720
                "720P-V(vid 9:16)": (720, 1280),  # 720P-V, 720x1280
                "832×480": (832, 480),  # 832x480
                "480×832": (480, 832),  # 480x832
            }
            return size_options[size_option]
        
        target_width, target_height = get_target_size()
        output_images = []
        output_masks = []

        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == 'P':
                frame = frame.convert("RGBA")
            elif 'A' in frame.getbands():
                frame = frame.convert("RGBA")
            
            if size_option == "No Change":
                # No resizing, just use the original frame
                image_frame = frame.convert("RGB")
            else:
                if size_option == "Custom" or size_option == "Million Pixels":
                    ratio = min(target_width / W, target_height / H)
                    adjusted_width = round(W * ratio)
                    adjusted_height = round(H * ratio)
                    image_frame = frame.convert("RGB").resize((adjusted_width, adjusted_height), Image.Resampling.BILINEAR)
                else:
                    image_frame = frame.convert("RGB")
                    image_frame = ImageOps.fit(image_frame, (target_width, target_height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))

            image_array = np.array(image_frame).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image_array)[None,])

            # Process the mask if available
            if 'A' in frame.getbands():
                mask_frame = frame.getchannel('A')
                if size_option == "Custom" or size_option == "Million Pixels":
                    mask_frame = mask_frame.resize((adjusted_width, adjusted_height), Image.Resampling.BILINEAR)
                else:
                    mask_frame = ImageOps.fit(mask_frame, (target_width, target_height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
                mask = np.array(mask_frame).astype(np.float32) / 255.0
                mask = 1. - mask
            else:
                if size_option == "Custom" or size_option == "Million Pixels":
                    mask = np.zeros((adjusted_height, adjusted_width), dtype=np.float32)
                else:
                    mask = np.zeros((target_height, target_width), dtype=np.float32)
            output_masks.append(torch.from_numpy(mask).unsqueeze(0))
        
        output_image = torch.cat(output_images, dim=0) if len(output_images) > 1 else output_images[0]
        output_mask = torch.cat(output_masks, dim=0) if len(output_masks) > 1 else output_masks[0]
        info = f"Image Path: {image_path}\nOriginal Size: {W}x{H}\nAdjusted Size: {target_width}x{target_height}"
        return (output_image, output_mask, info)

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
    def _resize_to_million_pixels(self, W, H):
        aspect_ratio = W / H
        target_area = 1000000  # 1 million pixels
        if aspect_ratio > 1:  # Landscape
            width = int(np.sqrt(target_area * aspect_ratio))
            height = int(target_area / width)
        else:  # Portrait
            height = int(np.sqrt(target_area / aspect_ratio))
            width = int(target_area / height)
        width = (width + 7) // 8 * 8
        height = (height + 7) // 8 * 8
        return width, height




class IO_save_image:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "Apt"}),
                "file_format": (["png","webp", "jpg", "tif"],),
            },
            "optional": {
                "output_path": ("STRING", {"default": "./output/Apt", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_path",)
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)  # 指定第一个返回值为列表类型
    CATEGORY = "Apt_Preset/View_IO"

    @staticmethod
    def find_highest_numeric_value(directory, filename_prefix):
        highest_value = -1
        for filename in os.listdir(directory):
            if filename.startswith(filename_prefix):
                try:
                    numeric_part = filename[len(filename_prefix):]
                    numeric_str = re.search(r'\d+', numeric_part).group()
                    numeric_value = int(numeric_str)
                    if numeric_value > highest_value:
                        highest_value = numeric_value
                except (ValueError, AttributeError):  # 增加对正则匹配失败的处理
                    continue
        return highest_value
        
    def save_image(self, image, file_format, filename_prefix="Apt", output_path=""):
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]

        # 设置输出路径
        output_dir = folder_paths.get_output_directory()
        results = []

        # 存储每张图的输出路径
        output_paths = []

        if isinstance(output_path, str):
            # 单一路径：所有图像都保存到该目录
            os.makedirs(output_path, exist_ok=True)
            output_paths = [output_path] * batch_size
        elif isinstance(output_path, list) and len(output_path) == batch_size:
            # 多路径列表：每个图像对应一个目录
            for path in output_path:
                os.makedirs(path, exist_ok=True)
            output_paths = output_path
        else:
            # 路径数量不匹配，默认使用默认输出目录
            print("Invalid output_path format. Using default output directory.")
            output_paths = [output_dir] * batch_size

        # 获取当前前缀下的最大序号
        base_dir = output_paths[0]
        if os.path.exists(base_dir):  # 避免目录不存在时的错误
            counter = self.find_highest_numeric_value(base_dir, filename_prefix) + 1
        else:
            counter = 0

        for idx, img_tensor in enumerate(images_list):
            output_image = img_tensor.cpu().numpy()
            img_np = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np[0])

            out_path = output_paths[idx]

            output_filename = f"{filename_prefix}_{counter + idx:05}"
            resolved_image_path = os.path.join(out_path, f"{output_filename}.{file_format}")

            img_params = {
                'png': {'compress_level': 4},
                'webp': {'method': 6, 'lossless': False, 'quality': 80},
                'jpg': {'quality': 95, 'format': 'JPEG'},
                'tif': {'format': 'TIFF'}
            }

            img.save(resolved_image_path, **img_params[file_format])

            results.append({
                "filename": f"{output_filename}.{file_format}",
                "subfolder": os.path.basename(out_path),  # 只保存子文件夹名而非完整路径
                "type": self.type
            })

        # 生成所有保存图像的绝对路径列表
        absolute_paths = [os.path.abspath(os.path.join(path, f"{filename_prefix}_{counter + i:05}.{file_format}")) 
                         for i, path in enumerate(output_paths)]
        
        # 关键修改：直接返回路径列表的元组，与RETURN_TYPES和OUTPUT_IS_LIST匹配
        return (absolute_paths,)
    



class IO_input_any:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", }),
            },
            "optional": {                
                "delimiter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "分隔行\\n, 分隔制符表\\t, 分隔空格\\s"
                }),
                "output_type": (["float", "int", "string", "anytype"], {"default": "anytype"}),
                "current_index": ("INT", {"default": -1, "min": -1, "max": 99999, "tooltip": "当前索引=-1时,不进行分割"}),
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "process_text"
    CATEGORY = "Apt_Preset/View_IO"
    OUTPUT_IS_LIST = (True, ) 

    def process_text(self, text, delimiter="", output_type="anytype", current_index=-1):
        if delimiter == "\\n":
            actual_delimiter = "\n"
        elif delimiter == "\\t":
            actual_delimiter = "\t"
        elif delimiter == "\\s":
            actual_delimiter = " "
        elif delimiter == "":
            actual_delimiter = "@@@@$" #不分隔
        else:
            actual_delimiter = delimiter
        
        if text == "":
            return ([],)
        values = text.split(actual_delimiter)
        
        try:
            # 输出完整列表
            if current_index == -1:
                if output_type == "float":
                    data_list = [float(value) for value in values]
                elif output_type == "int":
                    data_list = [int(float(value)) for value in values]
                elif output_type == "string":
                    data_list = [str(value) for value in values]
                else:  # anytype
                    data_list = []
                    for value in values:
                        value = value.strip()
                        if value.lower() == "true":
                            data_list.append(True)
                        elif value.lower() == "false":
                            data_list.append(False)
                        elif value.lower() in ["none", "null"]:
                            data_list.append(None)
                        elif re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$', value):
                            if '.' in value or 'e' in value.lower():
                                data_list.append(float(value))
                            else:
                                data_list.append(int(value))
                        else:
                            data_list.append(value)
                return (data_list,)
            
            # 输出单个元素
            elif 0 <= current_index < len(values):
                data = values[current_index]
                
                # 根据输出类型进行转换
                if output_type == "float":
                    return (float(data),)
                elif output_type == "int":
                    return (int(float(data)),)  # 先转为float再转为int
                elif output_type == "string":
                    return (str(data),)
                else:  # anytype
                    # 自动识别数据类型
                    data = data.strip()
                    if data.lower() == "true":
                        return (True,)
                    elif data.lower() == "false":
                        return (False,)
                    elif data.lower() in ["none", "null"]:
                        return (None,)
                    elif re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$', data):
                        if '.' in data or 'e' in data.lower():
                            return (float(data),)
                        else:
                            return (int(data),)
                    else:
                        return (data,)
            else:
                # 索引超出范围
                return ("",)
                
        except Exception as e:
            raise ValueError(f"Error processing text: {str(e)}")
