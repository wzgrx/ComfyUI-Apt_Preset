import os
import re
import folder_paths
import comfy
import os, re
from typing import Dict, List, Tuple

from PIL import Image as PILImage
from io import BytesIO



from ..main_unit import *


#------------------------------------------------------------
# 安全导入检查 -- 将导入语句修改为以下形式

try:
    import openpyxl
except ImportError:
    openpyxl = None
    print("Warning: openpyxl not installed, Excel-related nodes will not be available")

try:
    from openpyxl.drawing.image import Image as OpenpyxlImage
except ImportError:
    OpenpyxlImage = None
    print("Warning: openpyxl.drawing.image not available")

try:
    from openpyxl.utils import get_column_letter
except ImportError:
    get_column_letter = None
    print("Warning: openpyxl.utils.get_column_letter not available")


#------------------------------------------------------------


class excel_search_data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "search_content": ("STRING", {"default": ""}),
                "search_mode": (["Precise_search", "Fuzzy_search"], {"default": "Precise_search"}),
            },
            "optional": {   } 
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("debug", "row", "col")
    FUNCTION = "search_data"
    CATEGORY = "Apt_Preset/prompt"

    def IS_CHANGED(): return float("NaN")

    def search_data(self, excel_path, sheet_name, search_content, search_mode):
        try:
            if not os.path.exists(excel_path):
                return (f"Error: File does not exist at path: {excel_path}", None, None)
            if not os.access(excel_path, os.R_OK):
                return (f"Error: No read permission for file at path: {excel_path}", None, None)
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook[sheet_name]

            results = []
            found_row = None
            found_col = None
            for row in range(1, sheet.max_row + 1):
                for col in range(1, sheet.max_column + 1):
                    cell = sheet.cell(row=row, column=col)
                    cell_value = cell.value if cell.value is not None else ""
                    cell_value_str = str(cell_value)
                    if (search_mode == "Precise_search" and cell_value_str == search_content) or \
                        (search_mode == "Fuzzy_search" and search_content in cell_value_str):
                        results.append(f"{sheet_name}|{row}|{col}|{cell_value}")
                        found_row = row
                        found_col = col

            workbook.close()
            del workbook
            if not results:
                return ("No results found.", None, None)
            return ("\n".join(results), found_row, found_col)
        except Exception as e:
            return (f"Error: {str(e)}", None, None)


class excel_row_diff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "col_data": ("INT", {"default": 1, "min": 1, "step": 1}),
                "col_finish": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {} 
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("count", "count_data", "count_finish")
    FUNCTION = "excel_row_diff"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    - col_data=2: 统计第2列,从上到下连续非空单元格总数count1
    - col_finish=3: 统计第3列,从上到下连续非空单元格总数count2
    - 输出未对齐的数量:count_data-count_finish
    """

    def IS_CHANGED(cls): 
        return float("NaN")

    def excel_row_diff(self, excel_path, sheet_name, col_data, col_finish):
        try:
            if not os.path.exists(excel_path):
                raise Exception(f"Error: File does not exist at path: {excel_path}")

            if not os.access(excel_path, os.R_OK):
                raise Exception(f"Error: No read permission for file at path: {excel_path}")

            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook[sheet_name]

            def count_cells(col_index):
                if col_index == 0:  # 跳过无效列
                    return 0
                count = 0
                for row in range(1, sheet.max_row + 1):
                    cell_value = sheet.cell(row=row, column=col_index).value
                    if cell_value is not None:
                        count += 1
                    else:
                        break
                return count

            count1 = count_cells(col_data)
            count2 = count_cells(col_finish)

            result = abs(count1 - count2)

            workbook.close()
            del workbook

            return (result, count1, count2)

        except Exception as e:
            raise Exception(f"Error: {str(e)}")


class excel_column_diff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "row_data": ("INT", {"default": 1, "min": 1, "step": 1}),
                "row_finish": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("count", "count_data", "count_finish")
    FUNCTION = "excel_column_diff"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    - row_data=2: 统计第2行,从左到右连续非空单元格总数count1
    - row_finish=3: 统计第3行,从左到右连续非空单元格总数count2
    - 输出未对齐的数量:count_data-count_finish
    """

    def IS_CHANGED(cls):
        return float("NaN")

    def excel_column_diff(self, excel_path, sheet_name, row_data, row_finish):
        try:
            if not os.path.exists(excel_path):
                raise Exception(f"Error: File does not exist at path: {excel_path}")

            if not os.access(excel_path, os.R_OK):
                raise Exception(f"Error: No read permission for file at path: {excel_path}")

            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook[sheet_name]

            def count_cells(row_index):
                if row_index == 0:  # 跳过无效行
                    return 0
                count = 0
                for col in range(1, sheet.max_column + 1):
                    cell_value = sheet.cell(row=row_index, column=col).value
                    if cell_value is not None:
                        count += 1
                    else:
                        break
                return count

            count1 = count_cells(row_data)
            count2 = count_cells(row_finish)

            result = abs(count1 - count2)

            workbook.close()
            del workbook

            return (result, count1, count2)

        except Exception as e:
            raise Exception(f"Error: {str(e)}")


class excel_read:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "row_start": ("INT", {"default": 0, "min": 0}),
                "row_end": ("INT", {"default": 3, "min": 1}),
                "col_start": ("INT", {"default": 0, "min": 0}),
                "col_end": ("INT", {"default": 4, "min": 1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("data",)
    FUNCTION = "excel_read"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    - #excle最小单元行或列不能为0,将无效跳过
    - row_start=0, row_end=3: 单行输出: row=3    
    - row_start=1, row_end=3: 多行输出: row=1,2,3
    - row_start = row_end=1 : 行数相同,单行输出row=1
    - row_end=3, row_end=1 : 报错, 起始行必须小于=结束行
    """
    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")

    def excel_read(self, excel_path, sheet_name, row_start, row_end, col_start, col_end):
        try:
            # 校验 start <= end 且 >= 0
            if row_start > row_end:
                raise Exception(f"Error: row_start ({row_start}) must be <= row_end ({row_end})!")
            if col_start > col_end:
                raise Exception(f"Error: col_start ({col_start}) must be <= col_end ({col_end})!")

            # 处理 row_start == 0 的情况：只取 row_end 行
            if row_start == 0:
                start_row = end_row = max(1, row_end)
            else:
                start_row, end_row = row_start, row_end

            # 处理 col_start == 0 的情况：只取 col_end 列
            if col_start == 0:
                start_col = end_col = max(1, col_end)
            else:
                start_col, end_col = col_start, col_end

            # 确保行列编号 ≥ 1
            start_row = max(1, start_row)
            end_row = max(1, end_row)
            start_col = max(1, start_col)
            end_col = max(1, end_col)

            # 打开 Excel 文件并读取数据
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook[sheet_name]

            output_lines = []
            for row in range(start_row, end_row + 1):
                row_data = []
                for col in range(start_col, end_col + 1):
                    cell_value = sheet.cell(row=row, column=col).value
                    row_data.append(str(cell_value) if cell_value is not None else "")
                output_lines.append("|".join(row_data))

            workbook.close()
            del workbook

            return ("\n".join(output_lines),)

        except Exception as e:
            return (f"Error: {str(e)}",)


class excel_write_data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "row_start": ("INT", {"default": 0, "min": 0}),
                "row_end": ("INT", {"default": 3, "min": 1}),
                "col_start": ("INT", {"default": 0, "min": 0}),
                "col_end": ("INT", {"default": 5, "min": 1}),
                "data": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    FUNCTION = "write_data"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    - 示例: data 输入如下数据
    - 1 | 2 | 3 | 4
    - a | b | c | d
    - row_start=2, row_end=3  
    - col_start=2, col_end=5
    - 数据会依次填入第2行第2列 到 第3行第5列
    - 
    - row_start=0, row_end=3  
    - col_start=0, col_end=5
    - 数据会只填入到第3行第5列的单元格
    """

    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")

    def write_data(self, excel_path, sheet_name, row_start, row_end, col_start, col_end, data):
        try:
            # 校验 start <= end 且 >= 0
            if row_start > row_end:
                return (f"Error: row_start ({row_start}) must be <= row_end ({row_end})",)
            if col_start > col_end:
                return (f"Error: col_start ({col_start}) must be <= col_end ({col_end})",)

            # 处理 start == 0 的情况：只写 end 行/列
            if row_start == 0:
                start_row = end_row = max(1, row_end)
            else:
                start_row, end_row = row_start, row_end

            if col_start == 0:
                start_col = end_col = max(1, col_end)
            else:
                start_col, end_col = col_start, col_end

            # 确保最小值为1（兼容 Excel）
            start_row = max(1, start_row)
            end_row = max(1, end_row)
            start_col = max(1, start_col)
            end_col = max(1, end_col)

            # 文件存在性及权限检查
            if not os.path.exists(excel_path):
                return (f"Error: File does not exist at path: {excel_path}",)
            if not os.access(excel_path, os.W_OK):
                return (f"Error: No write permission for file at path: {excel_path}",)

            # 加载工作簿和工作表
            workbook = openpyxl.load_workbook(excel_path, read_only=False, data_only=True)
            sheet = workbook[sheet_name]

            # 解析输入数据
            data_lines = data.strip().split("\n")

            # 写入数据
            for row_index, line in enumerate(data_lines, start=start_row):
                if row_index > end_row:
                    break
                cell_values = line.split("|")
                for col_index, cell_value in enumerate(cell_values, start=start_col):
                    if col_index > end_col:
                        break
                    if cell_value.strip():
                        sheet.cell(row=row_index, column=col_index).value = cell_value.strip()

            # 保存并关闭工作簿
            workbook.save(excel_path)
            workbook.close()
            del workbook

            return ("Data written successfully!",)

        except PermissionError as pe:
            return (f"Permission Error: {str(pe)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)


class excel_insert_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_path": ("STRING", {"default": "excel_file_path.xlsx"}),
                "sheet_name": ("STRING", {"default": "Sheet1"}),
                "row_start": ("INT", {"default": 0, "min": 0}),
                "row_end": ("INT", {"default": 1, "min": 1}),
                "col_start": ("INT", {"default": 0, "min": 0}),
                "col_end": ("INT", {"default": 1, "min": 1}),
                "img_height": ("INT", {"default": 256, "max": 2048, "min": 64}),
                "image_path": ("STRING", {"default": "image_file_path.png"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    FUNCTION = "write_image"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    - 示例: 图片输入如下
    - row_start=2, row_end=3  
    - col_start=2, col_end=5
    - 图片会依次填入第2行第2列 到 第3行第5列
    - 
    - row_start=0, row_end=3  
    - col_start=0, col_end=5
    - 图片会只插入到第3行第5列的单元格
    """
    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")

    def write_image(self, excel_path, sheet_name, row_start, row_end, col_start, col_end, image_path,img_height):
        try:
            # 校验 start <= end 且 >= 0
            if row_start > row_end:
                return (f"Error: row_start ({row_start}) must be <= row_end ({row_end})",)
            if col_start > col_end:
                return (f"Error: col_start ({col_start}) must be <= col_end ({col_end})",)

            # 处理 start == 0：只插入 end 所在行/列
            if row_start == 0:
                start_row = end_row = max(1, row_end)
            else:
                start_row, end_row = row_start, row_end

            if col_start == 0:
                start_col = end_col = max(1, col_end)
            else:
                start_col, end_col = col_start, col_end

            # 确保最小值为 1（兼容 Excel）
            start_row = max(1, start_row)
            end_row = max(1, end_row)
            start_col = max(1, start_col)
            end_col = max(1, end_col)

            # 文件存在性及权限检查
            if not os.path.exists(excel_path):
                return (f"Error: Excel file does not exist at path: {excel_path}",)
            if not os.access(excel_path, os.W_OK):
                return (f"Error: No write permission for Excel file at path: {excel_path}",)
            if not os.path.exists(image_path):
                return (f"Error: Image file does not exist at path: {image_path}",)
            if not os.access(image_path, os.R_OK):
                return (f"Error: No read permission for image file at path: {image_path}",)

            # 加载工作簿和工作表
            workbook = openpyxl.load_workbook(excel_path, read_only=False, data_only=True)
            sheet = workbook[sheet_name]

            # 插入图片的目标位置（仅使用 start 坐标）
            target_row = start_row
            target_col = start_col
            cell_address = get_column_letter(target_col) + str(target_row)

            # 打开图片并按比例缩放（高度固定为256像素，宽度按比例计算）
            with PILImage.open(image_path) as img:
                width, height = img.size
                scale = img_height / height
                target_width = int(width * scale)
                resized_img = img.resize((target_width, img_height), PILImage.LANCZOS)

                # 转换为字节流供 openpyxl 使用
                img_byte_arr = BytesIO()
                resized_img.save(img_byte_arr, format=img.format)
                openpyxl_img = OpenpyxlImage(img_byte_arr)

            # 调整单元格尺寸以适应图片
            column_letter = get_column_letter(target_col)
            
            # 设置列宽（按像素转磅的近似公式：1像素 ≈ 0.75磅）
            sheet.column_dimensions[column_letter].width = target_width * 0.75 / 7
            
            # 设置行高为192磅（对应256像素）
            sheet.row_dimensions[target_row].height = 192

            # 插入图片（图片会自动以单元格左上角对齐）
            sheet.add_image(openpyxl_img, cell_address)

            # 保存并关闭工作簿
            workbook.save(excel_path)
            workbook.close()

            return ("Image inserted and scaled successfully!",)

        except PermissionError as pe:
            return (f"Permission Error: {str(pe)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)
        



class XXXexcel_Prompter:   #定式  #  简单
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_DIR = os.path.join(BASE_DIR, "excel")

    @staticmethod
    def load_excel(excel_path: str) -> dict:
        if not os.path.exists(excel_path):
            print(f"Excel文件不存在: {excel_path}")
            return {"文件不存在": ["", ""]}
            
        try:
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook.active
            headers = next(sheet.iter_rows(values_only=True))
            id_col = headers.index('ID') if 'ID' in headers else 0
            pos_col = headers.index('Positive') if 'Positive' in headers else 1
            neg_col = headers.index('Negative') if 'Negative' in headers else 2
            
            data = {}
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[id_col]:
                    pos_val = str(row[pos_col] or '')
                    neg_val = str(row[neg_col] or '')
                    data[row[id_col]] = [pos_val, neg_val]
            
            workbook.close()
            return data if data else {"无有效数据": ["", ""]}
            
        except Exception as e:
            print(f"加载Excel失败: {excel_path} - {e}")
            return {"加载失败": ["", ""]}

    @staticmethod
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]

    def single_replace(self, text, target, replacement):
        if not target or not replacement:
            return text
            
        target_clean = target.strip('"').strip()
        replacement_clean = replacement.strip('"').strip()
        
        pattern = re.escape(target_clean)
        return re.sub(pattern, replacement_clean, text)
    
    def multi_replace(self, text, multi_targets, multi_replacements):
        if not multi_targets or not multi_replacements:
            return text
            
        targets = multi_targets.split('@')
        replacements = multi_replacements.split('@')
        
        min_len = min(len(targets), len(replacements))
        targets = targets[:min_len]
        replacements = replacements[:min_len]
        
        result = text
        for target, replacement in zip(targets, replacements):
            result = self.single_replace(result, target.strip(), replacement.strip())
        
        return result

    @classmethod
    def INPUT_TYPES(cls):
        cls.人工建筑 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "人工建筑.xlsx"))
        cls.光线 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "光线.xlsx"))
        cls.影视摄影 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "影视摄影.xlsx"))
        cls.材质 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "材质.xlsx"))
        cls.自定义 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "自定义.xlsx"))
        cls.自然场景 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "自然场景.xlsx"))
        cls.自然现象 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "自然现象.xlsx"))
        cls.视觉镜头 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "视觉镜头.xlsx"))
        cls.起手式 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "起手式.xlsx"))
        cls.风格 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "风格.xlsx"))

        return {
            "required": {
                "文本控制": ("EXL_TEXT_STACK"),

                "前置文本": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "label": "前置文本内容"
                }),
                "起手式": (list(cls.起手式.keys()),),
                "风格": (list(cls.风格.keys()),),
                "光线": (list(cls.光线.keys()),),
                "材质": (list(cls.材质.keys()),),
                "人工建筑": (list(cls.人工建筑.keys()),),
                "自然场景": (list(cls.自然场景.keys()),),
                "自然现象": (list(cls.自然现象.keys()),),
                "视觉镜头": (list(cls.视觉镜头.keys()),),
                "影视摄影": (list(cls.影视摄影.keys()),),
                "自定义": (list(cls.自定义.keys()),),
                "replace_prompt": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "label": "替换 {prompt}"
                }),
                "replace_object": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "label": "替换 {object}"
                }),
            },
            "optional": {
                "custom_targets": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "label": "自定义目标标识符(多个用@分隔)"
                }),
                "custom_replace": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "label": "自定义替换内容(多个用@分隔)"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, 前置文本, 人工建筑, 光线, 影视摄影, 材质, 自定义, 自然场景, 自然现象, 视觉镜头, 起手式, 风格, 
                replace_prompt="", replace_object="", custom_targets="", custom_replace=""):
        positive_prompt = 前置文本 + " "
        
        negative_prompt = ""

        if 人工建筑 in self.人工建筑:
            positive_prompt += self.人工建筑[人工建筑][0] + " "
            negative_prompt += self.人工建筑[人工建筑][1] + " "
            
        if 光线 in self.光线:
            positive_prompt += self.光线[光线][0] + " "
            negative_prompt += self.光线[光线][1] + " "
            
        if 影视摄影 in self.影视摄影:
            positive_prompt += self.影视摄影[影视摄影][0] + " "
            negative_prompt += self.影视摄影[影视摄影][1] + " "
            
        if 材质 in self.材质:
            positive_prompt += self.材质[材质][0] + " "
            negative_prompt += self.材质[材质][1] + " "
            
        if 自定义 in self.自定义:
            positive_prompt += self.自定义[自定义][0] + " "
            negative_prompt += self.自定义[自定义][1] + " "
            
        if 自然场景 in self.自然场景:
            positive_prompt += self.自然场景[自然场景][0] + " "
            negative_prompt += self.自然场景[自然场景][1] + " "
            
        if 自然现象 in self.自然现象:
            positive_prompt += self.自然现象[自然现象][0] + " "
            negative_prompt += self.自然现象[自然现象][1] + " "
            
        if 视觉镜头 in self.视觉镜头:
            positive_prompt += self.视觉镜头[视觉镜头][0] + " "
            negative_prompt += self.视觉镜头[视觉镜头][1] + " "
            
        if 起手式 in self.起手式:
            positive_prompt += self.起手式[起手式][0] + " "
            negative_prompt += self.起手式[起手式][1] + " "
            
        if 风格 in self.风格:
            positive_prompt += self.风格[风格][0] + " "
            negative_prompt += self.风格[风格][1] + " "

        pos = positive_prompt.strip()
        neg = negative_prompt.strip()

        if replace_prompt.strip():
            pos = self.single_replace(pos, "{prompt}", replace_prompt)
            
        if replace_object.strip():
            pos = self.single_replace(pos, "{object}", replace_object)
            
        if custom_targets and custom_replace:
            pos = self.multi_replace(pos, custom_targets, custom_replace)
        
        return (pos, neg)








#------------------------------------------------------------建设中------------------------


class Stack_excel_Prompter:
    pass


class excel_kontext:
    pass

class Stack_excel_kontext:
    pass





class XXexcel_Prompter:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_DIR = os.path.join(BASE_DIR, "global")
    CONFIG_FILE = os.path.join(EXCEL_DIR, "config.xlsx")
    excel_data: Dict[str, Dict] = {}
    excel_files: List[str] = []
    @classmethod
    def load_config(cls) -> List[str]:
        if not os.path.exists(cls.CONFIG_FILE):
            print(f"配置文件不存在: {cls.CONFIG_FILE}")
            return []
        try:
            workbook = openpyxl.load_workbook(cls.CONFIG_FILE, read_only=True, data_only=True)
            sheet = workbook.active
            xlsx_files = []
            row = 1
            while True:
                cell_value = sheet.cell(row=row, column=2).value
                if cell_value is None:
                    break
                file_name = str(cell_value).strip()
                if file_name and file_name.lower().endswith('.xlsx'):
                    xlsx_files.append(file_name)
                row += 1
            workbook.close()
            cls.excel_files = [os.path.splitext(f)[0] for f in xlsx_files]
            return xlsx_files
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return []
    @classmethod
    def load_all_excels(cls) -> None:
        xlsx_files = cls.load_config()
        cls.excel_data = {}
        for file in xlsx_files:
            key = os.path.splitext(file)[0]
            file_path = os.path.join(cls.EXCEL_DIR, file)
            cls.excel_data[key] = cls.load_excel(file_path)
    @staticmethod
    def load_excel(excel_path: str) -> dict:
        if not os.path.exists(excel_path):
            print(f"Excel文件不存在: {excel_path}")
            return {"文件不存在": ["", ""]}
        try:
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook.active
            headers = next(sheet.iter_rows(values_only=True))
            id_col = headers.index('ID') if 'ID' in headers else 0
            pos_col = headers.index('Positive') if 'Positive' in headers else 1
            neg_col = headers.index('Negative') if 'Negative' in headers else 2
            data = {}
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[id_col]:
                    pos_val = str(row[pos_col] or '')
                    neg_val = str(row[neg_col] or '')
                    data[row[id_col]] = [pos_val, neg_val]
            workbook.close()
            return data if data else {"无有效数据": ["", ""]}
        except Exception as e:
            print(f"加载Excel失败: {excel_path} - {e}")
            return {"加载失败": ["", ""]}
    @staticmethod
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
    def single_replace(self, text, target, replacement):
        if not target or not replacement:
            return text
        target_clean = target.strip('"').strip()
        replacement_clean = replacement.strip('"').strip()
        pattern = re.escape(target_clean)
        return re.sub(pattern, replacement_clean, text)
    def multi_replace(self, text, multi_targets, multi_replacements):
        if not multi_targets or not multi_replacements:
            return text
        targets = multi_targets.split('@')
        replacements = multi_replacements.split('@')
        min_len = min(len(targets), len(replacements))
        targets = targets[:min_len]
        replacements = replacements[:min_len]
        result = text
        for target, replacement in zip(targets, replacements):
            result = self.single_replace(result, target.strip(), replacement.strip())
        return result
    @classmethod
    def INPUT_TYPES(cls):
        cls.load_all_excels()
        input_config = {
            "required": {

                },
            "optional": {
                "pos_stack": ("EXL_STACK", { "default": "" }),
                "pre_pos": ("STRING", { "multiline": True, "default": "",}),
                "back_pos": ("STRING", {"multiline": False,"default": "", }),
                
                "replace_content": ("STRING", {"multiline": False,"default": "", }),
                "replace_item": ("STRING", {"multiline": False,"default": "", }),
                "custom_targets": ("STRING", { "multiline": False,"default": "", }),
                "custom_replace": ("STRING", { "multiline": False, "default": "", })
                }
            }
        for file_key in cls.excel_files:
            input_config["required"][file_key] = (list(cls.excel_data[file_key].keys()),)
        return input_config
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, replace_content="", replace_item="", 
                pos_stack="", pre_pos="", back_pos="",
                custom_targets="", custom_replace="", **kwargs):
        # 处理正面提示
        excel_positives = []
        for file_key in self.excel_files:
            selected_value = kwargs.get(file_key)
            if selected_value in self.excel_data[file_key]:
                excel_positives.append(self.excel_data[file_key][selected_value][0])
        
        parts = []
        if pre_pos.strip():
            parts.append(pre_pos.strip())
        if excel_positives:
            # 过滤掉空值后再连接
            filtered_positives = list(filter(None, excel_positives))
            if filtered_positives:
                parts.append(", ".join(filtered_positives))  # Excel内部用逗号分隔
        if back_pos.strip():
            parts.append(back_pos.strip())
        
        # 只有非空部分才用逗号连接
        positive_prompt = ", ".join(filter(None, parts))
        
        # 处理负面提示（使用同样的逻辑）
        excel_negatives = []
        for file_key in self.excel_files:
            selected_value = kwargs.get(file_key)
            if selected_value in self.excel_data[file_key]:
                excel_negatives.append(self.excel_data[file_key][selected_value][1])
        
        # 过滤掉空值后用逗号连接
        negative_prompt = ", ".join(filter(None, excel_negatives))
        
        # 处理替换逻辑
        if replace_content.strip():
            positive_prompt = self.single_replace(positive_prompt, "{content}", replace_content)
        if replace_item.strip():
            positive_prompt = self.single_replace(positive_prompt, "{item}", replace_item)
        if custom_targets and custom_replace:
            positive_prompt = self.multi_replace(positive_prompt, custom_targets, custom_replace)
        
        return (positive_prompt, negative_prompt)




class excel_Prompter:  #help
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_DIR = os.path.join(BASE_DIR, "global")
    CONFIG_FILE = os.path.join(EXCEL_DIR, "config.xlsx")
    excel_data: Dict[str, Dict] = {}
    excel_files: List[str] = []
    
    @classmethod
    def load_config(cls) -> List[str]:
        if not os.path.exists(cls.CONFIG_FILE):
            print(f"配置文件不存在: {cls.CONFIG_FILE}")
            return []
        try:
            workbook = openpyxl.load_workbook(cls.CONFIG_FILE, read_only=True, data_only=True)
            sheet = workbook.active
            xlsx_files = []
            row = 1
            while True:
                cell_value = sheet.cell(row=row, column=2).value
                if cell_value is None:
                    break
                file_name = str(cell_value).strip()
                if file_name and file_name.lower().endswith('.xlsx'):
                    xlsx_files.append(file_name)
                row += 1
            workbook.close()
            cls.excel_files = [os.path.splitext(f)[0] for f in xlsx_files]
            return xlsx_files
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return []
    
    @classmethod
    def load_all_excels(cls) -> None:
        xlsx_files = cls.load_config()
        cls.excel_data = {}
        for file in xlsx_files:
            key = os.path.splitext(file)[0]
            file_path = os.path.join(cls.EXCEL_DIR, file)
            cls.excel_data[key] = cls.load_excel(file_path)
    
    @staticmethod
    def load_excel(excel_path: str) -> dict:
        if not os.path.exists(excel_path):
            print(f"Excel文件不存在: {excel_path}")
            return {"文件不存在": ["", "", ""]}  # 添加空的帮助信息
        try:
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook.active
            headers = next(sheet.iter_rows(values_only=True))
            id_col = headers.index('ID') if 'ID' in headers else 0
            pos_col = headers.index('Positive') if 'Positive' in headers else 1
            neg_col = headers.index('Negative') if 'Negative' in headers else 2
            help_col = headers.index('Help') if 'Help' in headers else 3  # 添加帮助列
            data = {}
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[id_col]:
                    pos_val = str(row[pos_col] or '') if len(row) > pos_col else ''
                    neg_val = str(row[neg_col] or '') if len(row) > neg_col else ''
                    help_val = str(row[help_col] or '') if len(row) > help_col else ''  # 获取帮助信息
                    data[row[id_col]] = [pos_val, neg_val, help_val]
            workbook.close()
            return data if data else {"无有效数据": ["", "", ""]}
        except Exception as e:
            print(f"加载Excel失败: {excel_path} - {e}")
            return {"加载失败": ["", "", ""]}
    
    @staticmethod
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
    
    def single_replace(self, text, target, replacement):
        if not target or not replacement:
            return text
        target_clean = target.strip('"').strip()
        replacement_clean = replacement.strip('"').strip()
        pattern = re.escape(target_clean)
        return re.sub(pattern, replacement_clean, text)
    
    def multi_replace(self, text, multi_targets, multi_replacements):
        if not multi_targets or not multi_replacements:
            return text
        targets = multi_targets.split('@')
        replacements = multi_replacements.split('@')
        min_len = min(len(targets), len(replacements))
        targets = targets[:min_len]
        replacements = replacements[:min_len]
        result = text
        for target, replacement in zip(targets, replacements):
            result = self.single_replace(result, target.strip(), replacement.strip())
        return result



    @classmethod
    def INPUT_TYPES(cls):
        cls.load_all_excels()
        input_config = {
            "required": {},
            "optional": {
                "pre_pos": ("STRING", {"multiline": True, "default": "",}),
                "back_pos": ("STRING", {"multiline": False, "default": "", }),
                "replace_content": ("STRING", {"multiline": False, "default": "", }),

                "custom_targets": ("STRING", {"multiline": False, "default": "", }),
                "custom_replace": ("STRING", {"multiline": False, "default": "", })
            }
        }
        for file_key in cls.excel_files:
            input_config["required"][file_key] = (list(cls.excel_data[file_key].keys()),)
        return input_config
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")  
    RETURN_NAMES = ("pos", "neg", "help")  
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, replace_content="", replace_item="", 
                 pre_pos="", back_pos="",
                custom_targets="", custom_replace="", **kwargs):
        # 处理正面提示
        excel_positives = []
        excel_negatives = []
        excel_helps = [] 
        
        for file_key in self.excel_files:
            selected_value = kwargs.get(file_key)
            if selected_value in self.excel_data[file_key]:
                excel_data = self.excel_data[file_key][selected_value]
                excel_positives.append(excel_data[0])
                excel_negatives.append(excel_data[1])
                if len(excel_data) > 2: 
                    excel_helps.append(excel_data[2])
        
        parts = []
        if pre_pos.strip():
            parts.append(pre_pos.strip())
        if excel_positives:
            filtered_positives = list(filter(None, excel_positives))
            if filtered_positives:
                parts.append(", ".join(filtered_positives))  
        if back_pos.strip():
            parts.append(back_pos.strip())

        positive_prompt = ", ".join(filter(None, parts))
        negative_prompt = ", ".join(filter(None, excel_negatives))
        
        help_info = "\n".join(filter(None, excel_helps)) 
        
        if replace_content.strip():
            positive_prompt = self.single_replace(positive_prompt, "{prompt}", replace_content)

        if custom_targets and custom_replace:
            positive_prompt = self.multi_replace(positive_prompt, custom_targets, custom_replace)
        
        return (positive_prompt, negative_prompt, help_info)



class excel_qwen_font:   # 定式  # 简单
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_DIR = os.path.join(BASE_DIR, "qwen_Image")

    @staticmethod
    def load_excel(excel_path: str) -> dict:
        if not os.path.exists(excel_path):
            print(f"Excel文件不存在: {excel_path}")
            return {"文件不存在": ["", ""]}
            
        try:
            workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet = workbook.active
            headers = next(sheet.iter_rows(values_only=True))
            id_col = headers.index('ID') if 'ID' in headers else 0
            pos_col = headers.index('Positive') if 'Positive' in headers else 1
            neg_col = headers.index('Negative') if 'Negative' in headers else 2
            
            data = {}
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[id_col]:
                    pos_val = str(row[pos_col] or '')
                    neg_val = str(row[neg_col] or '')
                    data[row[id_col]] = [pos_val, neg_val]
            
            workbook.close()
            return data if data else {"无有效数据": ["", ""]}
            
        except Exception as e:
            print(f"加载Excel失败: {excel_path} - {e}")
            return {"加载失败": ["", ""]}

    @staticmethod
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]

    def single_replace(self, text, target, replacement):
        if not target or not replacement:
            return text
        target_clean = target.strip('"').strip()
        replacement_clean = replacement.strip('"').strip()
        pattern = re.escape(target_clean)
        return re.sub(pattern, replacement_clean, text)

    @classmethod
    def INPUT_TYPES(cls):
        cls.样式 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "样式.xlsx"))
        cls.字体类型 = cls.load_excel(os.path.join(cls.EXCEL_DIR, "字体类型.xlsx"))

        return {
            "required": {
                "prefix_text": ("STRING", {"default": "", "multiline": True}),
                "style": (list(cls.样式.keys()),),
                "font_type": (list(cls.字体类型.keys()),),
                "font_detail_enhance": ("BOOLEAN", {"default": False}),
                "text_location": ("STRING", {"default": "", "multiline": False}),
                "text_content": ("STRING", {"default": "", "multiline": False}),
                "suffix_text": ("STRING", {"default": "", "multiline": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": False}),
            },
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, style, font_type, font_detail_enhance, prefix_text, 
                text_location, text_content, suffix_text, negative_prompt):
        # 从样式.xlsx 中读取 A 列，对应输出 B 列和 C 列内容
        if style in self.样式:
            B1 = self.样式[style][0]
            C1 = self.样式[style][1]
        else:
            B1 = ""
            C1 = ""

        # 从字体类型.xlsx 中读取 A 列，对应输出 B 列和 C 列内容
        if font_type in self.字体类型:
            B2 = self.字体类型[font_type][0]
            C2 = self.字体类型[font_type][1]
        else:
            B2 = ""
            C2 = ""

        # 构建正面提示
        pos = prefix_text
        if pos and not pos.endswith((' ', ',')):
            pos += ' '
        pos += f"在{{object}}，有一行字，内容如下：{B2}\"{{text}}\""
        if B1:
            pos += f", {B1}"
        if font_detail_enhance and C2:
            pos += f", {C2}"
        if suffix_text:
            pos += f", {suffix_text}"
            
        # 处理替换逻辑
        if text_location:
            pos = self.single_replace(pos, "{object}", text_location)
        if text_content:
            pos = self.single_replace(pos, "{text}", text_content)

        # 构建负面提示
        neg = C1
        if neg and negative_prompt:
            neg += f", {negative_prompt}"
        elif negative_prompt:
            neg = negative_prompt

        return (pos, neg)

















