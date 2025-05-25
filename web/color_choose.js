import { app } from '../../scripts/app.js'
import parseCss from './extern/parse-css.js'

// #region Shared Utils

export const CONVERTED_TYPE = 'converted-widget'

export function hideWidget(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.hidden = true
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4]
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    const { link } = node.inputs.find(i => i.widget?.name === widget.name)
    if (!link) return undefined
    return widget.origSerializeValue ? widget.origSerializeValue() : widget.value
  }
  if (widget.linkedWidgets) for (const w of widget.linkedWidgets) hideWidget(node, w, `:${widget.name}`)
}

export function showWidget(widget) {
  widget.type = widget.origType
  widget.computeSize = widget.origComputeSize
  widget.serializeValue = widget.origSerializeValue
  delete widget.origType; delete widget.origComputeSize; delete widget.origSerializeValue
  if (widget.linkedWidgets) for (const w of widget.linkedWidgets) showWidget(w)
}

export function convertToWidget(node, widget) {
  showWidget(widget)
  const sz = node.size
  node.removeInput(node.inputs.findIndex(i => i.widget?.name === widget.name))
  for (const w of node.widgets) w.last_y -= LiteGraph.NODE_SLOT_HEIGHT
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function convertToInput(node, widget, config) {
  hideWidget(node, widget)
  const { linkType } = getWidgetType(config)
  const sz = node.size
  node.addInput(widget.name, linkType, { widget: { name: widget.name, config } })
  for (const w of node.widgets) w.last_y += LiteGraph.NODE_SLOT_HEIGHT
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, 0]
  widget.type = CONVERTED_TYPE + suffix
  if (widget.linkedWidgets) for (const w of widget.linkedWidgets) hideWidgetForGood(node, w, `:${widget.name}`)
}

export function fixWidgets(node) {
  if (!node.inputs || !node.widgets) return
  for (const input of node.inputs) {
    const matching = node.widgets.find(w => w.name === input.name)
    if (!matching) continue
    if (matching.type !== CONVERTED_TYPE) hideWidget(node, matching)
    else convertToWidget(node, input)
  }
}

export const hasWidgets = node => node.widgets?.[Symbol.iterator] ? true : false

export const cleanupNode = node => {
  if (!hasWidgets(node)) return
  for (const w of node.widgets) {
    w.canvas?.remove()
    w.inputEl?.remove()
    w.onRemoved?.()
  }
}

export function getWidgetType(config) {
  let type = config?.[0]
  let linkType = type
  if (Array.isArray(type)) { type = 'COMBO'; linkType = linkType.join(',') }
  return { type, linkType }
}

export function isColorBright(rgb, threshold = 240) {
  return getBrightness(rgb) > threshold
}

function getBrightness(rgbObj) {
  return Math.round((Number.parseInt(rgbObj[0]) * 299 +
                    Number.parseInt(rgbObj[1]) * 587 +
                    Number.parseInt(rgbObj[2]) * 114) / 1000)
}

export function extendPrototype(object, property, callback) {
  if (!object) return
  if (property in object) {
    const orig = object[property]
    object[property] = function(...args) {
      const r = orig.apply(this, args)
      callback.apply(this, args)
      return r
    }
  } else {
    object[property] = callback
  }
}

export function addMenuHandler(nodeType, cb) {
  const orig = nodeType.prototype.getExtraMenuOptions
  nodeType.prototype.getExtraMenuOptions = function(app, options) {
    const r = orig?.apply(this, [app, options]) || []
    const newItems = cb?.apply(this, [app, options]) || []
    return [...r, ...newItems]
  }
}

export const getNodes = (skip_unused) => {
  const nodes = []
  for (const outerNode of app.graph.computeExecutionOrder(false)) {
    const skipNode = (outerNode.mode === 2 || outerNode.mode === 4) && skip_unused
    const innerNodes = !skipNode && outerNode.getInnerNodes ? outerNode.getInnerNodes() : [outerNode]
    for (const node of innerNodes) {
      if ((node.mode === 2 || node.mode === 4) && skip_unused) continue
      nodes.push(node)
    }
  }
  return nodes
}

// #endregion

// #region Color Widget

const newTypes = ['COLOR']

export const ColorWidgets = {
  COLOR: (key, val) => {
    const widget = {}
    widget.y = 0
    widget.name = key
    widget.type = 'COLOR'
    widget.options = { default: '#ff0000' }
    widget.value = val || '#ff0000'
    widget.draw = function(ctx, node, w, y, h) {
      if (this.type !== 'COLOR' && app.canvas.ds.scale > 0.5) return
      ctx.beginPath()
      ctx.fillStyle = this.value
      ctx.roundRect(15, y, w - 30, h, h)
      ctx.stroke()
      ctx.fill()
      const color = parseCss(this.value.default || this.value)
      if (!color) return
      ctx.fillStyle = isColorBright(color.values, 125) ? '#000' : '#fff'
      ctx.font = '14px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(this.name, w * 0.5, y + 15)
    }
    widget.mouse = function(e, pos, node) {
      if (e.type !== 'pointerdown') return
      const widgets = node.widgets.filter(w => w.type === 'COLOR')
      for (const w of widgets) {
        const rect = [w.last_y, w.last_y + 32]
        if (pos[1] > rect[0] && pos[1] < rect[1]) {
          const picker = document.createElement('input')
          picker.type = 'color'
          picker.value = this.value
          picker.style.position = 'absolute'
          picker.style.left = '-9999px'
          picker.style.top = '-9999px'
          document.body.appendChild(picker)
          picker.addEventListener('change', () => {
            this.value = picker.value
            this.callback?.(this.value)
            node.graph._version++
            node.setDirtyCanvas(true)
            picker.remove()
          })
          picker.click()
        }
      }
    }
    widget.computeSize = w => [w, 20]
    return widget
  }
}

// #endregion

// #region Register Extension

const widgets = {
  name: 'mtb.widgets',
  getCustomWidgets: () => ({
    COLOR: (node, inputName, inputData) => ({
      widget: node.addCustomWidget(ColorWidgets.COLOR(inputName, inputData[1]?.default || '#ff0000')),
      minWidth: 150,
      minHeight: 30
    })
  }),
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    let has_custom = false
    if (nodeData.input?.required) {
      for (const i of Object.keys(nodeData.input.required)) {
        if (newTypes.includes(nodeData.input.required[i][0])) {
          has_custom = true
          break
        }
      }
    }
    if (has_custom) {
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
        this.serialize_widgets = true
        this.setSize?.(this.computeSize())
        this.onRemoved = () => shared.cleanupNode(this)
        return r
      }

      const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined
        if (this.widgets) {
          const toInput = []
          for (const w of this.widgets) {
            if (w.type === CONVERTED_TYPE) continue
            if (newTypes.includes(w.type)) {
              const config = nodeData?.input?.required[w.name] || nodeData?.input?.optional[w.name] || [w.type, w.options || {}]
              toInput.push({
                content: `Convert ${w.name} to input`,
                callback: () => convertToInput(this, w, config)
              })
            }
          }
          if (toInput.length) options.push(...toInput, null)
        }
        return r
      }
    }
  }
}

app.registerExtension(widgets)
