# ComfyUI-AIMDO-XPU

Intel Arc XPU 上的 DynamicVRAM 早期实现（PYTHONPATH 劫持方案）

> [!IMPORTANT]
> ## ⛔ 本项目已停止更新
>
> **不是 comfy-aimdo XPU 被废弃了，而是它被一套更好的方案接手了。**
>
> Intel 端已经有一个自成体系的 XPU 方案介入 aimdo，由
> **`omni_xpu_kernel` + `ComfyUI-OmniXPU` + aimdo XPU provider + kitchen XPU provider** 四部分组成。
> 本插件在最后一次更新（v1.0-alpha，2026-08-08）之后，整体切换到这套新方案中。
>
> 👉 **要在 Intel Arc 上跑 ComfyUI + DynamicVRAM，请直接看下面的「后续方案」章节，不要再安装本插件。**

[**English**](README_EN.md) | 中文

---

## 为什么停更

本插件用**纯 Python 劫持**的方式，在 Intel XPU 上模拟了官方 comfy-aimdo（NVIDIA CUDA 版）的 DynamicVRAM 行为：
不依赖任何编译、不修改 ComfyUI 源码，在 2026-04 ~ 2026-08 期间是 A / B 系列卡少数的可用方案。

之后 Intel 官方介入这一领域（`intel/llm-scaler` 的 `omni` 目录），给出了 **Level Zero + SYCL 的硬件级实现**：
真实的虚拟地址预留（VBAR）+ 缺页换入，而不是软件层的 LRU 缓存模拟。
方案质量与维护力度都远超本项目，因此本项目的「软件模拟」使命就此结束。

| 对比项 | 本插件（旧） | 后续方案（现用） |
|---|---|---|
| 实现层次 | 纯 Python，零编译 | C/C++ + Level Zero / SYCL |
| VBAR 机制 | LRU 张量缓存 + watermark 驱逐（软件模拟） | 真实虚拟地址预留 + 缺页换入（硬件级） |
| 生效方式 | custom_nodes + `PYTHONPATH` 劫持 | 安装 provider 包 + 启动参数 `--enable-dynamic-vram` |
| 性能取向 | 轻量、易分享，略有损耗 | 贴近 NVIDIA 原版 aimdo 的硬件行为 |
| 维护状态 | ❌ 已停止（本仓库） | ✅ 活跃维护 |

> 也就是说：**旧插件不装、新方案照装**。两者的接口目标一致（都是让 ComfyUI 的 DynamicVRAM 在 XPU 上跑起来），
> 但实现路径完全不同，不能混用。

---

## 后续方案：四组件结构

新方案由四个组件拼成，各司其职：

| 组件 | 作用 | 说明 |
|---|---|---|
| **omni_xpu_kernel** | 高性能 Intel XPU / SYCL kernel（attention、量化等） | 从 `intel/llm-scaler` 提取 |
| **ComfyUI-OmniXPU** | ComfyUI 侧的薄适配层（custom node） | 把 kernel 接进 ComfyUI |
| **aimdo XPU provider** | DynamicVRAM 后端 | **取代本插件的劫持方案** |
| **kitchen XPU provider** | 推理 kernel 后端 | comfy-kitchen 的 XPU fork |

### 按显卡分线选择（重要）

A 系列（DG2）与 B 系列（BMG）走的是**两条不同的发行线**，不要混装：

| 组件 | A770 / DG2（Xe-HPG） | B 系列 / BMG（Battlemage） |
|---|---|---|
| **omni_xpu_kernel**（kernel） | <https://github.com/Blackwood416/omni-xpu-kernel> | <https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel> |
| **ComfyUI-OmniXPU** | <https://github.com/Blackwood416/ComfyUI-OmniXPU> | <https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU> |
| **comfy-aimdo-xpu**（aimdo XPU provider） | <https://github.com/Blackwood416/comfy-aimdo-xpu> | <https://github.com/xiangyuT/comfy-aimdo-xpu> |
| **comfy-kitchen-xpu**（kitchen XPU provider） | <https://github.com/xiangyuT/comfy-kitchen-xpu> | <https://github.com/xiangyuT/comfy-kitchen-xpu> |

> [!NOTE]
> **关于「官方」的一点说明（避免踩坑）**
>
> - `intel/llm-scaler` 是 Intel 官方仓库，其 `omni` 目录当前**面向 B 系列 / Battlemage**（native wheel AOT 编译给 BMG），
>   且 `0.2.0-b2` 仍是 **beta、实验性、仅源码构建**。
> - `Blackwood416/omni-xpu-kernel` 与 `Blackwood416/ComfyUI-OmniXPU` 是从 `intel/llm-scaler` **提取出来的 A770 / DG2 独立 fork**
>   ——因为上游只覆盖 BMG，A 系列需要走这条线。
> - `xiangyuT/comfy-aimdo-xpu` 与 `xiangyuT/comfy-kitchen-xpu` 是社区维护的 XPU fork，
>   仓库自述为**实验性**，验证范围主要是 Ubuntu 24.04 LTS + Arc Pro B70；
>   其它操作系统、显卡型号、软件环境的兼容性、正确性与性能**不作保证**。
>
> 也就是说：**这些组件的适用范围各不相同，装之前先确认自己的卡型对应的那条线。**

### 预编译包使用手册

不想自己编译、想直接下载整合包的用户，请看：

📖 **《Intel GPU 的 ComfyUI 系统优化指南》**
<https://allanmeng.github.io/IntelGpu-ComfyUI-Collection/intel-comfyui-guide/>

手册覆盖：背景说明 → 优化思路 → 安装 → 验证与故障排查 → 实测数据与 FAQ → 内核升级后的组件检查与恢复。

---

## 社区

我们有 Intel GPU 的社区网页和 QQ 群，有兴趣可以关注：

- 项目集主页：<https://allanmeng.github.io/IntelGpu-ComfyUI-Collection/>
- QQ 群：**220819365**（<https://qm.qq.com/q/gls9aI3lgA>）

---

## 如果你还在用本插件

本插件**不会再适配新的 ComfyUI 内核**。继续使用会导致插件与新内核（以及新 aimdo）的调用约定不匹配，
最坏情况是无法正常启动工作台（v0.4 就踩过一次这个坑）。

迁移步骤：

1. **先摘掉劫持**：把启动 bat 里那行 `set "PYTHONPATH=..."` 注释掉或删除；
   如果你装过 DLL 试点版，用部署文档里的方式回退 `site-packages/comfy_aimdo` 备份。
2. **清理插件目录**（可选）：删除 `custom_nodes/ComfyUI-AIMDO-XPU`。
3. **按上面的分线表安装新方案**，并参考使用手册操作。

> 回退插件的原理很简单：本插件只是在 `PYTHONPATH` 最前面插了一个同名 `comfy_aimdo/` 包。
> 去掉那行 `set`，`import comfy_aimdo` 就会重新命中 `site-packages` 里的官方包。

---

## 历史存档：原方案是怎么做的（仅作原理参考，勿再安装）

<details>
<summary>展开查看 v1.0-alpha 的架构说明</summary>

### 架构决策：PYTHONPATH 劫持

原方案以相同的 Python API 接口，在 Intel XPU 上重新实现 comfy-aimdo 的能力，**无需修改 ComfyUI 官方代码**。

官方 comfy-aimdo 安装在 `site-packages/comfy_aimdo/`，本项目在 `custom_nodes` 目录下放置同名 `comfy_aimdo/` 包，
通过启动脚本把项目路径加到 `PYTHONPATH` 最前面，使 `import comfy_aimdo` 优先命中本项目，实现透明替换。

> **为什么不用 custom_node 的 `__init__.py` 做劫持？** `main.py` 里的 `import comfy.aimdo.control`
> 早于 custom_nodes 加载，等 custom_nodes 的 `__init__.py` 介入时已经太晚了。

#### 替代策略

| 原版 CUDA 机制 | XPU 替代实现 |
|---|---|
| `aimdo.dll` VBAR 虚拟地址缺页 | `torch.xpu` LRU 张量缓存（`model_vbar.py`） |
| CUDA page fault handler | 显式 `fault()` 换入 / 换出 |
| `cudaHostAlloc` 固定内存 | `torch.empty(pin_memory=True)` |
| `VirtualAlloc + CreateFileMapping` | Python `mmap` 模块 |
| CUDA 设备属性查询 | `torch.xpu.get_device_properties()` |

#### 项目结构

```
ComfyUI-AIMDO-XPU/
├── comfy_aimdo/              ← 劫持包（目录名必须与官方包相同）
│   ├── __init__.py           ← torch.cuda 全局 shim；加载时一次性 patch
│   ├── control.py            ← init / init_device / get_total_vram_usage
│   ├── model_vbar.py         ← ModelVBAR + 全部模块级函数
│   ├── torch.py              ← aimdo_to_tensor / hostbuf_to_tensor
│   ├── host_buffer.py        ← HostBuffer（用 torch pin_memory 代替 CUDA）
│   ├── model_mmap.py         ← ModelMMAP（用 Python mmap 代替 aimdo.dll）
│   └── vram_buffer.py        ← VRAMBuffer（ComfyUI 新版接口）
├── __init__.py               ← custom_node 入口；含 XPUAIMDOStatus 节点（开关 + debug）
├── README.md
└── README_EN.md
```

#### 激活方式（历史）

在启动 ComfyUI **之前**，把项目根目录加到 `PYTHONPATH`：

```batch
:: 假设你的启动脚本位于 ComfyUI-aki-v3\ 目录下
set "PYTHONPATH=%~dp0ComfyUI\custom_nodes\ComfyUI-AIMDO-XPU;%PYTHONPATH%"
```

Python 模块搜索顺序为 `PYTHONPATH` > `site-packages`，因此 `import comfy_aimdo` 会优先找到本目录下的包。

#### 验证方法（历史）

启动日志中应出现：

```
[ComfyUI-AIMDO-XPU] ✅ XPU hijack ACTIVE  →  ...ComfyUI-AIMDO-XPU\comfy_aimdo\__init__.py
[ComfyUI-AIMDO-XPU] init_device(0) OK  name='Intel Arc ...'  total_vram=X.XX GB
DynamicVRAM support detected and enabled
```

#### 工作台内开关（v0.5）

`XPU AIMDO Status` 节点提供 `Enable_DynamicVRAM` 开关：

- **ON** = VBAR 分级加载（省显存，推理较慢）
- **OFF** = 全速全量加载（占更多显存，推理最快）
- **删除节点** = 自动恢复 OFF；`debug` 勾选框开启 Proxy 诊断日志

一句话理解：**OFF 是「空间换时间」，ON（dyvram）是「时间换空间」。**
OFF 把书全摊在大桌子上，随手就拿；ON 桌子小，每看一章都要收书、换书。

#### 与原版的差异

- **无硬件级缺页中断（VBAR）**：改为 Python 层 LRU 换入 / 换出，性能略低但功能等价
- **不依赖任何 CUDA 组件**
- `comfy.memory_management.aimdo_enabled` 会被设为 `True`，
  `CoreModelPatcher` 切换为 `ModelPatcherDynamic`，与原版路径一致

#### 已知限制

**❌ 不支持 NF4 / FP4 量化**：Intel XPU 上 `bitsandbytes.matmul_4bit` 底层调用 CUDA-specific kernel，
XPU 后端无对应实现。表现是报错或回退到非量化路径。
替代方案：使用 GGUF 的 **Q4_K / Q5_K / Q6_K** 或 **GGUF-f16 / bf16** 格式。

**❌ 抗内核变动能力弱**：因为要时刻保持与官方 aimdo、ComfyUI 内核的调用约定一致，
任何一方升级都可能让插件报错 —— 这也是最终被 provider 形态取代的原因之一。

#### 附：DLL 试点

v1.0-alpha 期间还尝试过一次 **DLL 后端试点**：基于
[`xiangyuT/comfy-aimdo-xpu`](https://github.com/xiangyuT/comfy-aimdo-xpu) 的 `dev/xpu-level-zero-vbar` 分支，
编译 Windows 版 `aimdo_xpu.dll`，以「替换 `site-packages/comfy_aimdo`」的方式生效。
该试点现已被上游正式的 provider 形态取代，不再单独维护。

</details>

---

## 版本历史

完整记录见 [CHANGELOG.md](CHANGELOG.md)。

| 版本 | 日期 | 主要变化 |
|---|---|---|
| v1.0-alpha | 2026-08-08 | 新增 DLL 后端试点；文档重构（**最终版本**，此后项目转入新方案） |
| v0.5 | 2026-06-02 | `Enable_DynamicVRAM` 运行时开关、`__class__` 双向切换、四步清洗 |
| v0.4 | 2026-06-02 | 修复 ComfyUI 0.23.0 升级导致的无法启动，新增 `init_devices()` |
| v0.3 | 2026-05-17 | `model_mmap` 大模型零拷贝修复（ACCESS_COPY + from_buffer） |
| v0.2 | 2026-05-04 | 新增 `VRAMBuffer`，启动时 API 兼容性检查 |
| v0.1 | 2026-04-28 | 首次发布：XPU 上的 comfy-aimdo 重实现 |

## License

本仓库未附带独立的 License 文件。代码作为**历史存档**保留，不再接受功能类 PR / issue。
有使用问题请转向上面的新方案仓库与社区渠道。
