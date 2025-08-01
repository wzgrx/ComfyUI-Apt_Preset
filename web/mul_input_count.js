import { app } from "/scripts/app.js";

const findInputByName = (node, name) => node.inputs ? node.inputs.find((w) => w.name === name) : null;
const findWidgetByName = (node, name) => node.widgets ? node.widgets.find((w) => w.name === name) : null;

function handleInputsVisibility(node, countValue, targets, type) {
    console.log(`Handling inputs for node ${node.constructor.title} with count value: ${countValue}`);
    for (let i = 1; i <= 50; i++) {
        targets.forEach((target) => {
            const name = `${target}_${i}`;
            const input = findInputByName(node, name);
            if (input) {
                if (i > countValue) {
                    console.log(`Removing input ${name} from node ${node.constructor.title}`);
                    node.removeInput(input);
                }
            } else {
                if (i <= countValue) {
                    console.log(`Adding input ${name} of type ${type} to node ${node.constructor.title}`);
                    node.addInput(name, type);
                }
            }
        });
    }
}

// 定义 ANY_TYPE
const ANY_TYPE = '*';

app.registerExtension({
    name: "type",
    nodeCreated(node) {
        console.log(`Node created: ${node.constructor.title}`);
        const nodeHandlers = {
            //"lay_image_match_W_and_H": ["image", "IMAGE"],
            "creat_mask_batch_input": ["mask", "MASK"],
            "creat_image_batch_input": ["image", "IMAGE"],
            "type_Anyswitch": ["data", ANY_TYPE]
        };

        for (const [nodeTitle, [target, type]] of Object.entries(nodeHandlers)) {
            if (node.constructor.title === nodeTitle) {
                console.log(`Node title matched: ${nodeTitle}`);
                if (!node.widgets) {
                    console.warn(`Node ${nodeTitle} has no widgets`);
                    continue;
                }
                const countWidget = findWidgetByName(node, "count");
                // 检查 countWidget 是否存在
                if (!countWidget) {
                    console.warn(`Count widget not found for node ${nodeTitle}`);
                    continue;
                }
                let widgetValue = countWidget.value;
                console.log(`Count widget value for node ${nodeTitle}: ${widgetValue}`);
                handleInputsVisibility(node, widgetValue, [target], type);

                Object.defineProperty(countWidget, 'value', {
                    get: () => widgetValue,
                    set: (newVal) => {
                        if (newVal !== widgetValue) {
                            console.log(`Count widget value changed for node ${nodeTitle} to ${newVal}`);
                            widgetValue = newVal;
                            handleInputsVisibility(node, newVal, [target], type);
                        }
                    }
                });
            }
        }
    }
});
