import torch
import logging

from ..main_unit import *


class pre_Condition_mode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
                "mode": (["combine", "average", "concat"], ),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})  
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool"

    DESCRIPTION = "strength1-strength4调节各条件强度：\n" \
    "- combine：混合特征（如红色+圆形→融合为红色圆形，无法拆分。文本串联）\n" \
    "- average：平衡融合（归一化权重，避免某特征过强）\n" \
    "- concat：保留独立特征（如红色+圆形→同时保留红色和圆形通道，可分别处理）"


    def merge(self, conditioning_1, conditioning_2, mode, 
              conditioning_3=None, conditioning_4=None,
              strength1=1.0, strength2=1.0, strength3=1.0, strength4=1.0):
        
        # 收集有效的条件输入
        conditionings = [conditioning_1, conditioning_2]
        if conditioning_3 is not None:
            conditionings.append(conditioning_3)
        if conditioning_4 is not None:
            conditionings.append(conditioning_4)
            
        # 收集与条件对应的强度参数
        strengths = [strength1, strength2, strength3, strength4]
        valid_strengths = strengths[:len(conditionings)]  # 只保留与条件数量匹配的强度值
        
        if mode == "combine":
            # Combine模式：简单连接所有条件（类似于原生ConditioningCombine）
            result = []
            for i, conditioning in enumerate(conditionings):
                # 应用强度参数到每个条件
                for cond_item in conditioning:
                    cond_tensor = cond_item[0]
                    cond_dict = cond_item[1].copy()
                    
                    # 应用强度到张量
                    cond_tensor = torch.mul(cond_tensor, valid_strengths[i])
                    
                    # 如果有pooled_output且不为None，也应用强度
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[i])
                    
                    result.append([cond_tensor, cond_dict])
            
            return (result, )
        
        elif mode == "average":
            # Average模式：加权平均
            if len(conditionings) == 1:
                # 如果只有一个条件，直接应用强度后返回
                result = []
                for cond_item in conditionings[0]:
                    cond_tensor = torch.mul(cond_item[0], valid_strengths[0])
                    cond_dict = cond_item[1].copy()
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[0])
                    result.append([cond_tensor, cond_dict])
                return (result,)
            
            # 对于两个条件，完全模拟原始ConditioningAverage的行为
            if len(conditionings) == 2:
                out = []
                # 对应原始的conditioning_to
                conditioning_to = conditionings[0]
                # 对应原始的conditioning_from
                conditioning_from = conditionings[1]
                
                # 原始实现只使用第一个conditioning_from
                if len(conditioning_from) > 1:
                    logging.warning("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied.")
                cond_from = conditioning_from[0][0]
                pooled_output_from = conditioning_from[0][1].get("pooled_output", None)
                
                # 使用strength1作为原始的conditioning_to_strength
                # 第二个条件的权重固定为1 - strength1，与原始实现一致
                conditioning_to_strength = valid_strengths[0]
                
                for i in range(len(conditioning_to)):
                    t1 = conditioning_to[i][0]
                    pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
                    
                    # 确保张量维度匹配（与原始实现完全一致）
                    t0 = cond_from[:,:t1.shape[1]]
                    if t0.shape[1] < t1.shape[1]:
                        # 使用与原始相同的零填充方式
                        t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]),
                                            dtype=t0.dtype, device=t0.device)], dim=1)
                    
                    # 完全采用原始的加权公式
                    tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
                    t_to = conditioning_to[i][1].copy()
                    
                    # 处理pooled_output（与原始实现一致）
                    if pooled_output_from is not None and pooled_output_to is not None:
                        t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
                    elif pooled_output_from is not None:
                        t_to["pooled_output"] = pooled_output_from
                    
                    out.append([tw, t_to])
                
                return (out, )
            
            # 对于三个或更多条件，使用归一化权重
            total_strength = sum(valid_strengths)
            if total_strength > 0:
                normalized_strengths = [s/total_strength for s in valid_strengths]
            else:
                normalized_strengths = [1.0/len(conditionings) for _ in valid_strengths]
            
            # 以第一个条件作为基础
            base_conditioning = conditionings[0]
            out = []
            
            for i in range(len(base_conditioning)):
                # 初始化加权和
                t_base = torch.mul(base_conditioning[i][0], normalized_strengths[0])
                t_result = base_conditioning[i][1].copy()
                
                # 处理pooled_output
                pooled_output = None
                if "pooled_output" in t_result and t_result["pooled_output"] is not None:
                    pooled_output = torch.mul(t_result["pooled_output"], normalized_strengths[0])
                
                # 累加其他条件
                for j in range(1, len(conditionings)):
                    cond_tensor = conditionings[j][i][0]
                    # 确保张量维度匹配
                    if cond_tensor.shape[1] != t_base.shape[1]:
                        if cond_tensor.shape[1] > t_base.shape[1]:
                            cond_tensor = cond_tensor[:, :t_base.shape[1], :]
                        else:
                            pad_size = t_base.shape[1] - cond_tensor.shape[1]
                            padding = torch.zeros((cond_tensor.shape[0], pad_size, cond_tensor.shape[2]), 
                                                dtype=cond_tensor.dtype, device=cond_tensor.device)
                            cond_tensor = torch.cat([cond_tensor, padding], dim=1)
                    
                    t_base = t_base + torch.mul(cond_tensor, normalized_strengths[j])
                    
                    # 处理pooled_output
                    if pooled_output is not None and "pooled_output" in conditionings[j][i][1] and conditionings[j][i][1]["pooled_output"] is not None:
                        pooled2 = conditionings[j][i][1]["pooled_output"]
                        if pooled2.shape[0] != pooled_output.shape[0]:
                            if pooled2.shape[0] > pooled_output.shape[0]:
                                pooled2 = pooled2[:pooled_output.shape[0]]
                            else:
                                pad_size = pooled_output.shape[0] - pooled2.shape[0]
                                padding = torch.zeros((pad_size, pooled2.shape[1]), 
                                                    dtype=pooled2.dtype, device=pooled2.device)
                                pooled2 = torch.cat([pooled2, padding], dim=0)
                        pooled_output = pooled_output + torch.mul(pooled2, normalized_strengths[j])
                
                # 更新pooled_output
                if pooled_output is not None:
                    t_result["pooled_output"] = pooled_output
                
                out.append([t_base, t_result])
                
            return (out, )
        
        elif mode == "concat":
            # Concat模式：张量拼接（类似于原生ConditioningConcat）
            if len(conditionings) == 1:
                # 如果只有一个条件，直接应用强度后返回
                result = []
                for cond_item in conditionings[0]:
                    cond_tensor = torch.mul(cond_item[0], valid_strengths[0])
                    cond_dict = cond_item[1].copy()
                    if "pooled_output" in cond_dict and cond_dict["pooled_output"] is not None:
                        cond_dict["pooled_output"] = torch.mul(cond_dict["pooled_output"], valid_strengths[0])
                    result.append([cond_tensor, cond_dict])
                return (result,)
            
            # 以第一个条件作为基础
            base_conditioning = conditionings[0]
            out = []
            
            for i in range(len(base_conditioning)):
                # 应用强度到第一个条件
                t_result = torch.mul(base_conditioning[i][0], valid_strengths[0])
                t_dict = base_conditioning[i][1].copy()
                
                # 处理pooled_output
                if "pooled_output" in t_dict and t_dict["pooled_output"] is not None:
                    t_dict["pooled_output"] = torch.mul(t_dict["pooled_output"], valid_strengths[0])
                
                # 拼接其他条件（应用强度后）
                for j in range(1, len(conditionings)):
                    cond_tensor = torch.mul(conditionings[j][i][0], valid_strengths[j])
                    t_result = torch.cat((t_result, cond_tensor), 1)
                    
                    # 拼接pooled_output（如果存在且不为None）
                    if "pooled_output" in t_dict and t_dict["pooled_output"] is not None and \
                       "pooled_output" in conditionings[j][i][1] and conditionings[j][i][1]["pooled_output"] is not None:
                        pooled1 = t_dict["pooled_output"]
                        pooled2 = torch.mul(conditionings[j][i][1]["pooled_output"], valid_strengths[j])
                        t_dict["pooled_output"] = torch.cat((pooled1, pooled2), dim=0)
                
                out.append([t_result, t_dict])
            
            return (out, )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")