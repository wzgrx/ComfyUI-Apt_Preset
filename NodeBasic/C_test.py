




import torch


#ReferenceOnly文生图节点
class ReferenceOnlySimple_T2i:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only_T2i"
    CATEGORY = "custom_node_experiments"

    def reference_only_T2i(self, model, reference, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size
        latent = {}
        latent["samples"] = torch.zeros(size_latent)
  
        batch = latent["samples"].shape[0] + reference["samples"].shape[0]
        # batch = reference["samples"].shape[0]
        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch

            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o,:]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        out_latent = torch.cat((reference["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))})
    





#ReferenceOnly遮罩节点
class ReferenceOnlySimple_Masks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "Mask_latent": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only_mask"
    CATEGORY = "custom_node_experiments"

    def reference_only_mask(self, model, reference, Mask_latent, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size

        batch = Mask_latent["samples"].shape[0] + reference["samples"].shape[0]
        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch
            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o,:]
            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        out_latent = torch.cat((reference["samples"], Mask_latent["samples"]))
        if "noise_mask" in Mask_latent:
            mask = Mask_latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < Mask_latent["samples"].shape[0]:
            print(Mask_latent["samples"].shape, mask.shape)
            mask = mask.repeat(Mask_latent["samples"].shape[0], 1, 1)

        if len(mask.shape) < 4:
            out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        else:
            out_mask = torch.zeros((1,1,mask.shape[2],mask.shape[3]), dtype=torch.float32, device="cpu")

        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))})  




class ReferenceOnly_TwoReference_Img2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "reference2": ("LATENT",),
                              "latent": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only_tworeference_img2img"
    CATEGORY = "custom_node_experiments"

    def reference_only_tworeference_img2img(self, model, reference, reference2,latent,batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size

        batch = latent["samples"].shape[0] + reference["samples"].shape[0] + reference2["samples"].shape[0]

        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch

            re = extra_options["transformer_index"] % 2
            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o + re,:]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        out_latent = torch.cat((reference["samples"], reference2["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask,out_mask, mask))})


    
#ReferenceOnly图生图节点
class ReferenceOnlySimple_Img2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "latent": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only_img2img"
    CATEGORY = "custom_node_experiments"

    def reference_only_img2img(self, model, reference, latent, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size

        batch = latent["samples"].shape[0] + reference["samples"].shape[0]
        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch

            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o,:]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        out_latent = torch.cat((reference["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask, mask))})




class ReferenceOnlyFlexible_Img2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "model": ("MODEL",),
                              "latent": ("LATENT",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                              },
                "optional": {
                              "reference": ("LATENT",),
                              "reference2": ("LATENT",),
                            }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only_flexible_img2img"
    CATEGORY = "custom_node_experiments"

    def reference_only_flexible_img2img(self, model, latent, batch_size, reference=None, reference2=None):
        model_reference = model.clone()
        
        # 确定使用的参考图像数量
        references = [ref for ref in [reference, reference2] if ref is not None]
        
        # 如果没有提供任何参考图像，则退化为普通img2img
        if not references:
            return (model_reference, latent)
        
        # 计算批次大小
        batch = latent["samples"].shape[0]
        for ref in references:
            batch += ref["samples"].shape[0]

        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch

            re = extra_options["transformer_index"] % len(references) if len(references) > 1 else 0
            for o in range(0, q.shape[0], batch):
                for x in range(1, len(references)+1):  # 根据实际参考图数量调整
                    if x <= len(references):
                        k[x + o, q.shape[1]:] = q[o + min(re, len(references)-1),:]

            return q, k, k

        model_reference.set_model_attn1_patch(reference_apply)
        
        # 构建输出潜空间张量
        tensors_to_cat = [ref["samples"] for ref in references] + [latent["samples"]]
        out_latent = torch.cat(tensors_to_cat)
        
        # 处理噪声遮罩
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] < latent["samples"].shape[0]:
            print(latent["samples"].shape, mask.shape)
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)

        # 创建与参考图像数量匹配的零遮罩
        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        masks_to_cat = [out_mask] * len(references) + [mask]
        
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat(masks_to_cat)})













