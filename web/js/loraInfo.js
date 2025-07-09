import { app } from "../../../scripts/app.js";
import { LoraInfoDialog } from "../../ComfyUI-Custom-Scripts/js/modelInfo.js";

const infoHandlers = {
    "Stack_LoRA": true,
    "Stack_LoRA": true
}

app.registerExtension({
    name: "autotrigger.LoraInfo",
    beforeRegisterNodeDef(nodeType) {
        if (!infoHandlers[nodeType.comfyClass]) {
            return;
        }
        
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const loraWidgets = this.widgets.filter(w => w.name.startsWith("lora_name_"));
            
            if (loraWidgets.length === 0) {
                return;
            }
            
            // 为每个LoRA添加单独的菜单项
            for (const widget of loraWidgets) {
                const loraIndex = widget.name.split('_').pop();
                let value = widget.value;
                
                if (!value || (value === "None")) {
                    continue;
                }
                
                if (value.content) {
                    value = value.content;
                }
                
                options.unshift({
                    content: `View info for LoRA ${loraIndex} (${value})`,
                    callback: async () => {
                        new LoraInfoDialog(value).show("loras", value);
                    },
                });
            }

            return getExtraMenuOptions?.apply(this, arguments);
        };
    }
});