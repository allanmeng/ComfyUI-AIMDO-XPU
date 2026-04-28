# ComfyUI-AIMDO-XPU

Intel XPU 版动态模型卸载器（comfy-aimdo XPU 替代实现）

[**English**](README_EN.md) | 中文

---

## 原版 comfy-aimdo（CUDA）核心能力

comfy-aimdo（Rattus 著，v0.2.12）是 ComfyUI 的 DynamicVRAM 功能核心依赖，提供了两项硬件级能力：

### 1. VBAR 虚拟地址缺页机制
- 依赖 NVIDIA GPU 的 `aimdo.dll` 注入 CUDA 虚拟地址基址寄存器（VBAR）
- 当模型权重超出显存时，GPU 触发缺页中断，触发按需换入（fault-in）
- 换入后显存中只保留实际需要的部分，卸载（fault-out）后腾出空间
- **效果**：可在有限显存中运行大于显存的模型，避免整体加载带来的 OOM

### 2. 固定内存与文件映射
- `cudaHostAlloc`： pinned memory 用于 GPU-CPU 高效数据传输
- `VirtualAlloc + CreateFileMapping`：将大模型 checkpoint 文件映射到虚拟地址空间，避免一次性加载到内存
- **效果**：降低内存占用，支持超大模型（>10GB）的流式加载

> 原版完全依赖 NVIDIA CUDA API，在 Intel XPU（Arc B 系列）上无法运行。

---

## ComfyUI-AIMDO-XPU 项目介绍

本项目以相同的 Python API 接口，在 Intel XPU 上重新实现上述能力，**无需修改 ComfyUI 官方代码**。

### 架构决策：PYTHONPATH 劫持

官方 comfy-aimdo 安装在 `site-packages/comfy_aimdo/`，本项目在 custom_nodes 目录下放置同名 `comfy_aimdo/` 包，通过启动脚本将项目路径加入 `PYTHONPATH` 最前面，使 `import comfy_aimdo` 优先命中本项目，实现透明替换。

> **为什么不用 custom_node __init__.py 做劫持？** `main.py` 第 34 行的 `import comfy.aimdo.control` 早于 custom_nodes 加载，custom_nodes 的 `__init__.py` 介入时已经太晚了。

### 替代策略

| 原版 CUDA 机制 | XPU 替代实现 |
|---|---|
| aimdo.dll VBAR 虚拟地址缺页 | torch.xpu LRU 张量缓存（model_vbar.py） |
| CUDA `page fault handler` | 显式 `fault()` 换入/换出 |
| `cudaHostAlloc` 固定内存 | `torch.empty(pin_memory=True)` |
| `VirtualAlloc + CreateFileMapping` | Python `mmap` 模块 |
| CUDA 设备属性查询 | `torch.xpu.get_device_properties()` |

### 项目结构

```
ComfyUI-AIMDO-XPU/
├── comfy_aimdo/              ← 劫持包（目录名必须与官方包相同）
│   ├── __init__.py           ← torch.cuda 全局 shim；加载时一次性 patch
│   ├── control.py            ← init/init_device/get_total_vram_usage
│   ├── model_vbar.py         ← ModelVBAR + 全部模块级函数
│   ├── torch.py              ← aimdo_to_tensor / hostbuf_to_tensor
│   ├── host_buffer.py       ← HostBuffer（用 torch pin_memory 代替 CUDA）
│   └── model_mmap.py         ← ModelMMAP（用 Python mmap 代替 aimdo.dll）
├── __init__.py               ← custom_node 入口；含启动验证 + XPUAIMDOStatus 节点
└── README.md
```

---

## 里程碑列表

| 阶段 | 里程碑 | 状态 | 说明 |
|------|--------|------|------|
| **① 基础路基** | PYTHONPATH 劫持通道打通 | ✅ 已完成 | `comfy_aimdo/` 包优先于 `site-packages` 官方版被加载 |
| **② 清除 CUDA 硬编码** | `torch.cuda` 全局 shim | ✅ 已完成 | `_lazy_init`、`device()`、`get_device_properties` 等全部拦截 |
| **③ 清除 CUDA 硬编码** | `is_nvidia()` DynamicVRAM 门卫 | ✅ 已完成 | 让 XPU 也能触发 ComfyUI 的 DynamicVRAM 初始化逻辑 |
| **④ 核心 API 对齐** | VBAR 虚拟地址缺页模拟 | ✅ 已完成 | `model_vbar.py` LRU 缓存替代 aimdo.dll VBAR 机制 |
| **⑤ 核心 API 对齐** | Pin Memory 对齐 | ✅ 已完成 | `host_buffer.py` 用 `torch.empty(pin_memory=True)` 替代 `cudaHostAlloc` |
| **⑥ 核心 API 对齐** | 文件映射对齐 | ✅ 已完成 | `model_mmap.py` 用 Python `mmap` 替代 Windows `VirtualAlloc + CreateFileMapping` |
| **⑦ 第三方兼容** | XPUSYSMonitor 插件兼容 | ✅ 已完成 | `torch.cuda.is_available()` 捕获 AssertionError 返回 False |
| **⑧ 第三方兼容** | SeedVR2 插件兼容 | ✅ 已完成 | `torch.cuda.device('cuda:N')` 映射到 `xpu:N` |
| **⑨ 功能验证** | SDXL 推理跑通 | ✅ 已完成 | DynamicVRAM 开启，82s 执行，无 OOM |
| **⑩ 第三方兼容** | 其他插件 CUDA 硬编码清理 | 🔄 进行中 | 按需发现、按需修复 |
| **⑪ 稳定性验证** | 长时间/大批量推理测试 | ⬜ 待做 | 验证无内存泄漏、无资源泄露 |
| **⑫ 功能补全** | ComfyUI 其他 DynamicVRAM 场景 | ⬜ 待做 | 如 Video models、Diffusion models larger than VRAM |
| **⑬ 性能优化** | reserve-vram 精细化调优 | ⬜ 待做 | 针对 B580 11.67GB 做显存分段策略优化 |
| **⑭ 性能优化** | 模型按层/模块卸载 | ⬜ 待做 | 比整体卸载更细粒度，进一步降低 OOM 风险 |
| **⑮ 工程化** | 一键安装/配置脚本 | ⬜ 待做 | 用户无需手动改 bat，只需装插件 |
| **⑯ 上游反馈** | 向 ComfyUI 官方提 PR 建议 | ⬜ 待做 | 将 XPU 兼容代码合并入官方，减少下游补丁依赖 |

---

## 激活方式

在启动 ComfyUI **之前**，将本项目根目录加到 `PYTHONPATH` 最前面：

```batch
:: 假设你的启动脚本位于 ComfyUI-aki-v3\ 目录下
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"
```

`%~dp0` 自动指向 bat 文件所在目录，适配任意安装路径。

> Python 模块搜索顺序：`PYTHONPATH` > `site-packages`，因此
> `import comfy_aimdo` 会优先找到本目录下的 `comfy_aimdo/` 包，
> 而不是 `site-packages` 中的官方 CUDA 版本。

### 临时禁用劫持

注释掉上述 `set PYTHONPATH` 行即可切回官方 comfy-aimdo（CUDA），用于对比测试。

---

## 验证方法

### 方法一：查看启动日志

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

### 方法二：工作流节点

在工作流中添加 **"XPU AIMDO Status"** 节点，运行后会显示完整状态报告。

---

## 与原版的差异

- **无硬件级缺页中断（VBAR）**：改为 Python 层 LRU 换入/换出，性能略低但功能等价
- **不依赖任何 CUDA 组件**
- `comfy.memory_management.aimdo_enabled` 会被设为 `True`，
  `CoreModelPatcher` 切换为 `ModelPatcherDynamic`，与原版路径完全一致

---

## 已知限制

### ❌ 不支持 NF4 / FP4 量化

Intel XPU（Arc B 系列）**不支持** NF4 / FP4 量化推理。原因：`bitsandbytes.matmul_4bit` 底层调用 CUDA-specific kernel，XPU 后端无对应实现。

**表现**：工作流中使用 NF4/FP4 量化模型时会报错或回退到非量化路径。

**影响范围**：
- Stable Diffusion 模型的 NF4/FP4 量化权重（通常标记为 `-NF4`、`-FP4` 后缀）
- 第三方量化工具导出的 4-bit 量化版本

**替代方案**：
- 使用 **Q4_K** / **Q5_K** / **Q6_K** 等 GGUF 量化格式（XPU 支持）
- 使用 **GGUF-f16** / **GGUF-bf16** 全精度格式

> 日志中 `gguf qtypes` 字段可查看当前模型实际使用的量化类型。
