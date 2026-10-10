# 更新日志

## 文档更新（2026-10-10）— 停更交接说明

> 纯文档改动（中文 + 英文），**不含任何代码变更**。

### 变更
- README / README_EN 重写为「项目已停止更新」的交接说明，让访客能快速了解后续去向：
  - 顶部新增停更声明，明确「不是 comfy-aimdo XPU 被废弃，而是整体切换到了新方案」；
  - 新增新旧方案对比表（软件模拟 VBAR ↔ 硬件级 Level Zero / SYCL）；
  - 新增后续方案的「四组件结构」说明：`omni_xpu_kernel` + `ComfyUI-OmniXPU` + aimdo XPU provider + kitchen XPU provider；
  - 新增按显卡分线的组件链接表（A770 / DG2 与 B 系列 / BMG 两条发行线，勿混装）；
  - 新增「官方」适用范围的澄清：`intel/llm-scaler` 的 `omni` 目录当前只覆盖 B 系列 / BMG，
    A770 / DG2 走独立 fork，aimdo / kitchen 的 XPU provider 为社区维护的实验性 fork；
  - 新增旧用户迁移步骤（摘掉 `PYTHONPATH` 劫持 → 清理插件目录 → 安装新方案）；
  - 原劫持方案的架构说明（替代策略 / 项目结构 / 激活与验证 / 已知限制）移入折叠的「历史存档」区，
    并标注「仅作原理参考，勿再安装」。
- 新增「社区」章节：项目集主页与 QQ 群入口。

### 说明
- 本仓库自此作为**历史存档**保留，不再适配新的 ComfyUI 内核。
- 使用问题请转向上表对应的新方案仓库与社区渠道。

## v1.0-alpha (2026-08-08)

### 新增
- **【新方案尝试】DLL 后端（Level Zero 硬件级）** — 以预编译 `aimdo_xpu.dll` 形式提供硬件级 DynamicVRAM 后端
  - 基于社区项目 `xiangyuT/comfy-aimdo-xpu` 的 `dev/xpu-level-zero-vbar` 分支构建
  - 使用真实的 Level Zero 虚拟地址预留（VBAR）+ 缺页换入机制，更接近 NVIDIA 原版 aimdo
  - 通过替换 `site-packages` 中的 `comfy_aimdo` 包生效（deploy.bat 自动备份，可回退）
  - 以 **GitHub Release 试点** 形式发布，详见 README 中【新方案尝试】章节
- 新增 `release/comfy_aimdo_xpu_win_v0.4.13.zip` 发布包（含中英文部署文档）

### 文档
- README 删除里程碑列表
- README 项目介绍新增【新方案尝试】DLL 后端章节（来源 / 原理 / 环境要求 / 与劫持方案对比 / 如何尝试）
- "架构决策：PYTHONPATH 劫持" 更名为 "原方案架构决策：PYTHONPATH 劫持"，明确与 DLL 新方案的定位区分
- 更新 README_EN.md 与中文版同步

## v0.5 (2026-06-02)

### 新增
- **DynamicVRAM 运行时开关** — XPU AIMDO Status 节点新增 `Enable_DynamicVRAM` 开关
  - ON：VBAR 分级加载（省显存）
  - OFF：全速全量加载
  - 删除节点自动恢复 OFF
- 节点新增 `debug` 勾选框，控制 Proxy 诊断日志
- 节点新增 `usage` 使用说明文本框

### 技术改动
- `__class__` 双向切换：运行时切换 ModelPatcher/ModelPatcherDynamic
- 四步清洗机制：VBAR 销毁 + 显存碎片回收 + 权重缓存重载
- 全新 `set_dynamic_vram()` 函数
- per-prompt reset 确保每次 Prompt 从 OFF 开始

### 修复
- 适配 ComfyUI 0.23.0 节点输出缓存的时序问题

## v0.4 (2026-06-02)

### 修复
- **解决 ComfyUI 0.23.0 升级带来的无法启动问题**

【问题】
- ComfyUI 内核更新到了 0.23.0 版本
- 其中包括调用 AIMDO 的函数发生了变化
- 导致安装过 ComfyUI-AIMDO-XPU 插件的用户在更新内核后，无法正常启动工作台

【说明】
- 该插件是通过劫持方案实现显存的动态管理，需要时刻保持与原插件、内核调用插件的方式保持一致
- 所以一旦CUDA版的Aimdo DLL升级、内核调用Aimdo方式升级，则插件就会报错，最坏情况是无法正常进入工作台
- 我尽可能保证及时更新此插件，以适配官方方案的更新
- 如果未能及时更新，导致不能正常启动；两个方案：1. 注销到bat中的劫持语句（你安装的时候肯定改过，推荐方案） 2. 如果是更新内核导致的问题，回退内核版本

【解决】
- ComfyUI-AIMDO-XPU 插件的版本已更新，已经可以解决0.23.0 内核版本遇到的问题
- 你需要用启动器的插件管理来更新此插件，或者直接在 git 上更新（这个插件我没放出过网盘安装包，所以都是用git安装的，应该都会）
- 如果你有这个插件，看到这个内容，升级了该插件，但没有升级内核到 0.23.0 ，不会报错

### 技术改动
- 新增 `control.init_devices()` 函数，适配 ComfyUI 0.23.0 的新调用方式
- API 兼容性检查更新至 **22 项**覆盖

## v0.3 (2026-05-17)

### 修复
- **`model_mmap.py`：大模型文件零拷贝修复** — 将 `mmap.ACCESS_READ` + `from_buffer_copy()`（全量复制）
   改为 `mmap.ACCESS_COPY` + `from_buffer()`（零拷贝），避免加载大模型时因双倍内存占用导致 OOM
- 改进异常诊断：错误信息中加入 `[异常类型]`，方便定位根因
- 新增文件存在性预检查，文件缺失时直接报 `FileNotFoundError`

### 其他
- 版本号提升至 v0.3

## v0.2 (2026-05-04)

### 新增
- `vram_buffer.py`：新增 `VRAMBuffer` 类，兼容 ComfyUI 新版 `comfy_aimdo.vram_buffer` 接口
- `control.py`：启动时打印 API 兼容性检查（`_check_api_compat()`），20 项接口覆盖
- 版本号提升至 v0.2

### 修复
- API 兼容性检查中使用 `torch_aimdo` 别名正确引用子模块

### 其他
- 小深（Deepseek V4）接手维护

## v0.1 (2026-04-28)

### 首次发布

- Intel XPU (Arc B-Series) 重新实现 comfy-aimdo 的 DynamicVRAM 功能
- PYTHONPATH 劫持，替换官方 CUDA 版 comfy_aimdo
- 基于 LRU 的 VBAR 模拟（model_vbar.py）
- Pin memory 通过 `torch.empty(pin_memory=True)` 替代 `cudaHostAlloc`
- 文件映射通过 Python `mmap` 替代 `VirtualAlloc + CreateFileMapping`
- `torch.cuda` shim 实现透明的 XPU 兼容
- `comfy.model_management` patch 使 XPU 启用 DynamicVRAM
- `XPU AIMDO Status` 诊断节点

### 已知限制

- NF4/FP4 量化不支持（`bitsandbytes.matmul_4bit` 无 XPU kernel）
- torch.compile VAE compile 在 XPU dynamo 上不支持

---

# Changelog

## Docs update (2026-10-10) — project handover notice

> Documentation-only change (Chinese + English), **no code changes**.

### Changed
- README / README_EN rewritten as a "no longer maintained" handover notice, so visitors
  immediately understand the status and where to go next:
  - added a prominent notice at the top clarifying that comfy-aimdo for XPU was **not**
    abandoned — the work moved to the successor stack;
  - added an old-vs-new comparison table (simulated VBAR vs hardware-level Level Zero / SYCL);
  - added the successor stack layout: `omni_xpu_kernel` + `ComfyUI-OmniXPU` +
    aimdo XPU provider + kitchen XPU provider;
  - added a per-architecture component link table (A770 / DG2 vs B-Series / BMG — two
    separate distribution lines, do not mix);
  - added a clarification of the scope of "official": the `omni` directory of
    `intel/llm-scaler` currently covers B-Series / BMG only, the A770 / DG2 line is a
    standalone fork, and the aimdo / kitchen XPU providers are community-maintained
    experimental forks;
  - added migration steps for existing users (remove the `PYTHONPATH` hijack → clean up the
    plugin directory → install the successor stack);
  - moved the original hijack architecture notes (replacement strategy / project structure /
    activation & verification / known limitations) into a collapsed "archive" section marked
    "reference only — do not install".
- Added a "Community" section (project collection page and QQ group).

### Notes
- This repository is kept as a **historical archive** and will not be adapted to new ComfyUI kernels.
- For usage questions, please use the corresponding successor repositories and community channels.

## v1.0-alpha (2026-08-08)

### Added
- **【New pilot】DLL backend (Level Zero, hardware-level)** — provides a hardware-level
  DynamicVRAM backend as a precompiled `aimdo_xpu.dll`
  - Built from the `dev/xpu-level-zero-vbar` branch of the community project `xiangyuT/comfy-aimdo-xpu`
  - Uses real Level Zero virtual address reservation (VBAR) + page-fault fault-in, closer to the original NVIDIA aimdo
  - Takes effect by replacing the `comfy_aimdo` package in `site-packages` (deploy.bat backs up automatically, rollback available)
  - Released as a **GitHub Release pilot**; see the 【New pilot】section in the README
- Added the `release/comfy_aimdo_xpu_win_v0.4.13.zip` package (with CN/EN deployment docs)

### Docs
- Removed the milestone list from the README
- Added the 【New pilot】DLL backend section to the README (origin / principle / requirements / comparison / how to try)
- Renamed "Architecture decision: PYTHONPATH hijack" to "Original architecture decision: PYTHONPATH hijack"
- Synced README_EN.md with the Chinese version

## v0.5 (2026-06-02)

### Added
- **Runtime DynamicVRAM toggle** — XPU AIMDO Status node now has an `Enable_DynamicVRAM` switch
  - ON: VBAR staged loading (VRAM-efficient)
  - OFF: full-speed full load
  - Auto-reverts to OFF when node is deleted
- `debug` checkbox on the Status node for Proxy diagnostic logs
- `usage` text input explaining how to use the node

### Changed
- `__class__` swapping: switch between ModelPatcher/ModelPatcherDynamic at runtime
- 4-step cleanup: VBAR destruction + VRAM defrag + weight cache reload
- New `set_dynamic_vram()` function
- Per-prompt reset ensures every prompt starts in OFF mode

### Fixed
- ComfyUI 0.23.0 node output cache timing issues

## v0.4 (2026-06-02)

### Fixed
- **Compatibility fix for ComfyUI 0.23.0 startup crash**
- Added `control.init_devices()` to match the new API call in ComfyUI 0.23.0
- API compat check now covers **22 interfaces**

## v0.3 (2026-05-17)

### Fixed
- **`model_mmap.py`: zero-copy fix for large model files** — switched from
  `mmap.ACCESS_READ` + `from_buffer_copy()` (full file copy) to
  `mmap.ACCESS_COPY` + `from_buffer()` (zero-copy), preventing OOM on
  large safetensors files caused by doubled memory usage
- Improved error diagnostics: exception type name `[TypeName]` now included
  in error messages
- Added file existence pre-check with explicit `FileNotFoundError`

### Misc
- Version bumped to v0.3

## v0.2 (2026-05-04)

### Added
- `vram_buffer.py`: `VRAMBuffer` class for comfy-aimdo upstream API compatibility
- `control.py`: API compat check on startup (`_check_api_compat()`), covering 20 interfaces
- Version bumped to v0.2

### Fixed
- `torch_aimdo` alias properly used in API compat check

### Misc
- 小深 (Deepseek V4) takes over maintenance

## v0.1 (2026-04-28)

### Initial Release

- Intel XPU (Arc B-Series) reimplementation of comfy-aimdo's DynamicVRAM
- PYTHONPATH hijack replacing official CUDA comfy_aimdo
- LRU-based VBAR simulation (model_vbar.py)
- Pin memory via `torch.empty(pin_memory=True)` instead of `cudaHostAlloc`
- File mapping via Python `mmap` instead of `VirtualAlloc + CreateFileMapping`
- `torch.cuda` shim for transparent XPU compatibility
- `comfy.model_management` patch to enable DynamicVRAM on XPU
- `XPU AIMDO Status` diagnostic node

### Known Limitations

- NF4/FP4 quantization not supported (`bitsandbytes.matmul_4bit` has no XPU kernel)
- torch.compile VAE compile not supported on XPU dynamo
