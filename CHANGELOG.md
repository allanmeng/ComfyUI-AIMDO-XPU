# 更新日志

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
