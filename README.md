# ComfyUI-AIMDO-XPU

Intel XPU 版动态模型卸载器（comfy-aimdo XPU 替代实现）

> 本质是通过调整模型的加载方式，提高显存利用率，降低OOM发生的几率
> 简单来说：OFF（全量加载）是“空间换时间”，而 ON（DynamicVRAM / dyvram）是“时间换空间”。


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

### 【新方案尝试】DLL 后端（Level Zero 硬件级）

> 🆕 这是本项目的一次新尝试：**以预编译 DLL 的形式提供硬件级 DynamicVRAM 后端**，
> 与下方原有的纯 Python 劫持方案互补。目前以 **Release 试点** 形式发布，
> 欢迎试用并反馈。

**来源与原理**

- DLL 后端基于社区项目 `xiangyuT/comfy-aimdo-xpu` 的 `dev/xpu-level-zero-vbar` 分支
  （Intel **Level Zero** + oneAPI SYCL 实现），在 Windows 上编译为 `aimdo_xpu.dll`；
- 与 Python 劫持方案的"LRU 张量缓存模拟"不同，DLL 后端使用**真实的 Level Zero 虚拟地址预留
  （VBAR）+ 缺页换入机制**，更接近 NVIDIA 原版 comfy-aimdo 的硬件行为；
- 它通过替换 `site-packages` 中的 `comfy_aimdo` 包生效，而不是 PYTHONPATH 劫持。

**与劫持方案对比**

| 维度 | Python 劫持方案（默认） | DLL 后端方案（试点） |
|---|---|---|
| 实现方式 | 纯 Python，零编译 | 预编译 C/C++（Level Zero + SYCL） |
| VBAR 机制 | LRU 缓存 + watermark 驱逐（软件模拟） | 真实虚拟地址预留 + 缺页换入（硬件级） |
| 部署方式 | custom_nodes + PYTHONPATH 劫持 | 部署到 site-packages（替换官方包） |
| 依赖 | 无编译依赖 | 需 oneAPI 2026.1（与编译版本一致） |
| 性能取向 | 轻量、易分享 | 更贴近原版 aimdo 性能 |

**如何尝试 DLL 方案**

- 从 **GitHub Releases** 下载最新发布包（`comfy_aimdo_xpu_win_v*.zip`）；
- 解压后按包内 `README-DEPLOY-CN.md` / `README-DEPLOY-EN.md` 操作即可；
- ⚠️ 如果你之前使用过本插件的劫持版，请先按部署文档开头的"升级提示"清理，
  否则 DLL 不会生效。

---

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
│   ├── vram_buffer.py       ← VRAMBuffer（ComfyUI 新版接口）
├── __init__.py               ← custom_node 入口；含 XPUAIMDOStatus 节点（开关 + debug）
└── README.md
```

---

## 下载与安装

### 方式一：Git Clone（推荐）

```batch
cd f:\ComfyUI-aki-v3\ComfyUI\custom_nodes
git clone https://github.com/allanmeng/ComfyUI-AIMDO-XPU.git
```

### 方式二：下载 ZIP

1. 访问 [GitHub 仓库](https://github.com/allanmeng/ComfyUI-AIMDO-XPU)
2. 点击 **Code** → **Download ZIP**
3. 解压到 `ComfyUI/custom_nodes/ComfyUI-AIMDO-XPU` 目录

### 前置要求

- Intel Arc B 系列显卡（B580 / B570 等）
- Intel oneAPI Base Toolkit（提供 oneMKL 和 SYCL 运行时）
- PyTorch XPU 版本：`pip install torch --index-url https://download.pytorch.org/whl/xpu`
- ComfyUI-aki-v3 或其他支持 Intel XPU 的 ComfyUI 发行版

### 验证安装

启动 ComfyUI 后，查看日志中是否出现：

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE
[ComfyUI-AIMDO-XPU] v0.5
```

---

## 激活方式

在启动 ComfyUI **之前**，将本项目根目录赋值到 `PYTHONPATH` ，放到bat启动文件中， main.py的前面：

```batch
:: 假设你的启动脚本位于 ComfyUI-aki-v3\ 目录下
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"

:: 下面这句是原来的启动项
"%PYTHON_PATH%\python.exe" "%COMFYUI_PATH%\main.py" --lowvram --disable-smart-memory --reserve-vram 0.1 --preview-method none --use-pytorch-cross-attention
```

`%~dp0` 自动指向 bat 文件所在目录，适配任意安装路径。

> Python 模块搜索顺序：`PYTHONPATH` > `site-packages`，因此
> `import comfy_aimdo` 会优先找到本目录下的 `comfy_aimdo/` 包，
> 而不是 `site-packages` 中的官方 CUDA 版本。

### 如何禁用劫持

注释掉上述 `set PYTHONPATH` 行即可切回官方 comfy-aimdo（CUDA），用于对比测试。

### 工作台中切换 DynamicVRAM

XPU AIMDO Status 节点新增 `Enable_DynamicVRAM` 开关（v0.5）：

<img width="45%" height="45%" alt="image" src="https://github.com/user-attachments/assets/afde7138-bdf5-4d09-88a7-d31718e81e05" />

- **ON** = VBAR 显存分级加载（节省 VRAM，推理较慢）
- **OFF** = 全速全量加载（占用更多 VRAM，推理最快）
- **删除节点** = 自动恢复 OFF
- **debug** 勾选框开启 Proxy 诊断日志

| 模式 | 空间（显存）占用 | 运行速度 | 相当于什么？ | 
|--------|--------|------|--------|
|OFF（全量加载）|❌ 极高（容易 OOM 爆显存）|🚀 极快（满血全速）|所有的参考书全部平铺在巨大的书桌上，伸手就能拿到，翻书速度最快。|
|ON（dyvram 模式）|极低（12G 也能跑巨型大图）|🐢 变慢（取决于策略）|书桌很小，放不下几本书。每次要看下一章，都得把桌上的书收起来，再从书架上拿一本新书摆出来。|



> 只要把该节点独立的放在工作流中，保持启用状态即可，不用链接任何其他节点
> 切换Enable_DynamicVRAM开关，立即生效，无需重启 ComfyUI。切换 OFF 后会自动清理 VBAR 缓存和显存碎片。


---

## 验证方法

### 方法一：查看启动日志

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

### 方法二：工作流节点

在工作流中添加 **"XPU AIMDO Status"** 节点，运行后日志中会显示完整状态报告。

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
