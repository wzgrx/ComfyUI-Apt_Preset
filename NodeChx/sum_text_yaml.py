
"""
Declaration, the original nodes come from  NakamuraShippo/ComfyUI-NS-PromptList
https://github.com/NakamuraShippo/ComfyUI-NS-PromptList.git

"""


from ..main_unit import *
from nodes import CLIPTextEncode

import os
import yaml
import json
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from server import PromptServer




class YAMLFileHandler(FileSystemEventHandler):
    """Handles file system events for YAML files"""
    def __init__(self, node_instance):
        self.node_instance = node_instance
        
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.yaml'):
            self.node_instance.refresh_enums()
    
    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.yaml'):
            self.node_instance.refresh_enums()
    
    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.yaml'):
            self.node_instance.refresh_enums()


class text_sum:
    """NS-PromptList Node for ComfyUI"""

    def __init__(self):
        # 修改此处路径
        self.yaml_dir = Path(__file__).parent / "yaml"
        self.yaml_dir.mkdir(exist_ok=True)
        self.write_lock = asyncio.Lock()
        self.observer = None
        self.file_handler = YAMLFileHandler(self)
        
        # Store server instance
        self.server = PromptServer.instance
        
        # Start watchdog observer
        self._start_watchdog()
        
        # Register socket handlers
        self._register_socket_handlers()
        
        # Initial enum refresh
        self.refresh_enums()
    
    @classmethod
    def INPUT_TYPES(cls):
        # 修改此处路径
        yaml_dir = Path(__file__).parent / "yaml"
        yaml_dir.mkdir(exist_ok=True)
        
        yaml_files = [f.name for f in yaml_dir.glob("*.yaml")]
        if not yaml_files:
            # Create default yaml if none exist
            default_yaml = yaml_dir / "default.yaml"
            default_yaml.write_text("example:\n  prompt: 'Enter your prompt here'\n")
            yaml_files = ["default.yaml"]
        
        # Get all possible titles from all YAML files for validation
        all_titles = set([""])  # Always include empty string
        for yaml_file in yaml_files:
            yaml_path = yaml_dir / yaml_file
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    all_titles.update(data.keys())
            except:
                pass
        
        return {
            "required": {
                "select_yaml": (sorted(yaml_files), {"default": yaml_files[0] if yaml_files else ""}),
                "select": (sorted(list(all_titles)), {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True, }),
                "negative": ("STRING", {"default": "", "multiline": False,}),
                "add_pos": ("STRING", {"default": "", "multiline": False, }),
                "add_neg":  ("STRING", {"default": "", "multiline": False, }),
                "style": (["None"] + style_list()[0], {"default": "None"}),
                "remove": ("STRING", {"multiline": False,"default": ""}),
                "replace_target": ("STRING", {"multiline": False,"default": ""}),
                "replace": ("STRING", {"multiline": False,"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("pos", "neg",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """节点说明:
- 风格路径: ..\\ComfyUI-Apt_Preset\\csv\\styles.csv
- 文档路径: ..\\ComfyUI-Apt_Preset\\NodeChx\\yaml
- 
- 1)多个词汇同时替换用逗号隔开，例如:
- 替换目标: object, prompt
- 替换内容: a girl, smile
- => object替换为 a girl, prompt 替换为 smile
- 
- 2)多词汇移除，用法和替换类似:
- 
- 3)输出逻辑:
- Positive Prompt =先合并文本 (prompt + style + add_pos) , 再移除或替换
- Negative Prompt = (negative + style + add_neg)
"""  
#region------------------------
    @classmethod
    def _get_instance(cls):
        """Get or create singleton instance"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance
    
    def _get_yaml_files(self) -> List[str]:
        """Get list of YAML files in yaml directory"""
        if not self.yaml_dir.exists():
            return ["default.yaml"]
        
        yaml_files = [f.name for f in self.yaml_dir.glob("*.yaml")]
        if not yaml_files:
            # Create default yaml if none exist
            default_yaml = self.yaml_dir / "default.yaml"
            default_yaml.write_text("example:\n  prompt: 'Enter your prompt here'\n")
            yaml_files = ["default.yaml"]
        
        return sorted(yaml_files)
    
    def _get_titles_from_yaml(self, yaml_file: str) -> List[str]:
        """Get titles from a YAML file"""
        yaml_path = self.yaml_dir / yaml_file
        if not yaml_path.exists():
            return [""]
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return list(data.keys())
        except Exception as e:
            print(f"Error reading YAML {yaml_file}: {e}")
            self._handle_corrupt_yaml(yaml_path)
            return [""]
    
    def _handle_corrupt_yaml(self, yaml_path: Path):
        """Handle corrupt YAML by renaming and creating new"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad_path = yaml_path.parent / f"bad_{timestamp}_{yaml_path.name}"
        shutil.move(str(yaml_path), str(bad_path))
        yaml_path.write_text("# Recovered from corrupt file\n")
    
    def _start_watchdog(self):
        """Start file system observer"""
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(self.file_handler, str(self.yaml_dir), recursive=False)
            self.observer.start()
    
    def refresh_enums(self):
        """Refresh YAML and title enums, broadcast update"""
        yaml_files = self._get_yaml_files()
        
        # Build enum data for all YAML files
        enum_data = {
            "yaml_files": yaml_files,
            "titles_by_yaml": {}
        }
        
        for yaml_file in yaml_files:
            titles = self._get_titles_from_yaml(yaml_file)
            enum_data["titles_by_yaml"][yaml_file] = titles
        
        self._broadcast_enum(enum_data)
    
    def _broadcast_enum(self, enum_data: Dict):
        """Broadcast enum update to frontend"""
        if hasattr(self.server, 'send_sync'):
            self.server.send_sync("sum_text_list_enum", enum_data)
        else:
            # Alternative method for older ComfyUI versions
            try:
                from aiohttp import web
                if hasattr(self.server, 'socketio'):
                    self.server.socketio.emit("sum_text_list_enum", enum_data)
            except:
                pass
    
    def _register_socket_handlers(self):
        """Register socket.io handlers"""
        server = PromptServer.instance
        
        @server.routes.post("/sum_text_list/get_prompt")
        async def get_prompt(request):
            data = await request.json()
            yaml_file = data.get("yaml", "")
            title = data.get("title", "")
            node_id = data.get("node_id", None)
            
            response_data = self._get_prompt_data(yaml_file, title)
            # Include node_id in response
            response_data["node_id"] = node_id
            
            # Send via websocket
            if hasattr(server, 'send_sync'):
                server.send_sync("sum_text_list_set_widgets", response_data)
            
            return {"success": True}
        
        @server.routes.post("/sum_text_list/reload_yamls")
        async def reload_yamls(request):
            """Force reload YAML list"""
            self.refresh_enums()
            return {"success": True}
    
    def _get_prompt_data(self, yaml_file: str, title: str) -> Dict[str, str]:
        """Get prompt data from YAML"""
        yaml_path = self.yaml_dir / yaml_file
    
        if not yaml_path.exists(): 
            return {"title": title, "prompt": "", "negative": ""}
    
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
    
            if title in data and isinstance(data[title], dict):
                prompt = data[title].get("prompt", "")
                negative = data[title].get("negative", "")
            else:
                prompt = ""
                negative = ""
    
            return {"title": title, "prompt": prompt, "negative": negative}
        except Exception as e:
            print(f"Error reading prompt: {e}")
            return {"title": title, "prompt": "", "negative": ""}
    
    async def _save_yaml(self, yaml_file: str, data: Dict):
        """Atomic save YAML with lock"""
        async with self.write_lock:
            yaml_path = self.yaml_dir / yaml_file
            
            # Use temporary file for atomic write
            with tempfile.NamedTemporaryFile(mode='w', dir=str(self.yaml_dir), 
                                           delete=False, encoding='utf-8') as tmp:
                yaml.dump(data, tmp, default_flow_style=False, allow_unicode=True, 
                         sort_keys=True)
                tmp_path = tmp.name
            
            # Atomic replace
            os.replace(tmp_path, str(yaml_path))
    
    def delete_title(self, yaml_file: str, title: str):
        """Delete a title from YAML"""
        yaml_path = self.yaml_dir / yaml_file
        
        if not yaml_path.exists():
            return
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            if title in data:
                del data[title]
                # Run async save in sync context
                asyncio.run(self._save_yaml(yaml_file, data))
                self.refresh_enums()
        except Exception as e:
            print(f"Error deleting title: {e}")

#endregion------------------------  


    def run(self, style="default", negative="", replace_target="", replace="",remove="", add_pos="", add_neg="", select_yaml: str = "", select: str = "", prompt: str = "", unique_id: str = "") -> Tuple[str]:
        """Main execution function"""
        
        # Check prompt length warning
        if len(prompt) > 4096:
            print(f"Warning: Prompt length ({len(prompt)}) exceeds recommended 4096 chars")
    
        # Save current prompt and negative to YAML
        if select and prompt:
            yaml_path = self.yaml_dir / select_yaml
            try:
                if yaml_path.exists(): 
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                else:
                    data = {}
                if select not in data:
                    data[select] = {}
                data[select]["prompt"] = prompt
                # 仅在 negative 有值时才写入
                if negative:
                    data[select]["negative"] = negative
                asyncio.run(self._save_yaml(select_yaml, data))
            except Exception as e:
                print(f"Error saving prompt: {e}")
    

        if add_pos is not None and add_pos != "":
            prompt = prompt + "," + add_pos
        if add_neg is not None and add_neg != "":
            negative = negative + "," + add_neg
            

        if isinstance(prompt, tuple):
            prompt = ", ".join(str(x) for x in prompt if x is not None)
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        if isinstance(negative, tuple):
            negative = ", ".join(str(x) for x in negative if x is not None)
        elif not isinstance(negative, str):
            negative = str(negative)
        prompt, negative = add_style_to_subject(style,  prompt, negative)

        if remove is not None and remove!= "":
            prompt = clean_prompt(prompt, remove)
        if replace_target is not None and replace_target!= "":
            prompt = replace_text(prompt, replace_target, replace)

        pos= prompt  #更新纯文本
        neg = negative  #更新纯文本
        return (pos, neg)





    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution on each run"""
        # This ensures the node re-validates inputs each time
        return float("NaN")
    
    @classmethod
    def VALIDATE_INPUTS(cls, select_yaml, select, prompt, unique_id):
        """Validate inputs - always return True to avoid validation errors"""
        # Since we're dynamically updating the select options,
        # we need to bypass ComfyUI's static validation
        return True


# Singleton instance
_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = text_sum()
    return _instance


get_instance()
