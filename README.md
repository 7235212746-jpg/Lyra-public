# Lyra

VASP 电子结构绘图与分析工作台。公开源码预览版 `v0.2.0-preview.1`，应用内部版本为 `0.2.0`。

默认使用经典桌面界面和 Lite Mode，可在不读取 `vasprun.xml` 的情况下绘制能带、投影能带和 DOS。当前主要验证平台为 Windows，Python 3.12。

## 已实装功能

| 模块 | 功能 |
| --- | --- |
| Band | 完整能带、自旋分辨、分段高对称路径 |
| Projected Band | 元素、轨道组与细分轨道投影；自选通道颜色和完整能带底色；RGB 与 fat-band |
| DOS / PDOS | 总 DOS、元素/轨道分解、自旋向下镜像、手动或自动密度范围 |
| Band + DOS | 独立选择两个计算文件夹，共享能量轴，默认以 DOS/SCF 的费米能级对齐 |
| 查点与能带对照 | 点击真实采样点查看投影组成；并排比较两套计算，列出相近能带候选 |
| 局部放大 | 保留坐标轴的放大窗口，字号、线宽、范围调整和 PNG/PDF/SVG 保存 |
| 带边分析 | 手动两点能量差；基于占据数的采样路径 VBM/CBM 与带隙提示 |
| 费米面 | 完整规则三维 k 网格上的三维等能面与二维切片 |
| INCAR 助手 | 离线中文说明、参数写法、重复覆盖和基础风险检查，保留官方来源 |
| 缓存与报告 | 完整投影缓存、实际解析阶段进度、计算质量证据报告、设置记忆 |

RGB 支持一至三个互不重叠的投影通道；更多通道使用 fat-band。投影百分比表示已解析投影内占比。

远程 Linux 和手机功能处于预览阶段，具体范围见 [公开版状态与科学限制](docs/PUBLIC_PREVIEW.md)。成键、价态和 WAVECAR 波函数身份识别尚未开放。

## 安装与启动

在 Windows 安装包含 Tk 支持的 Python 3.12 或更新版本。以下示例使用 Python 3.12：

```powershell
git clone https://github.com/7235212746-jpg/Lyra-public.git
cd Lyra-public
py -3.12 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install --upgrade pip
.\.venv-build\Scripts\python.exe -m pip install -e .
.\.venv-build\Scripts\pythonw.exe -m lyra
```

Lite Mode 已包含在基本安装中。需要 Full Mode 时，额外安装：

```powershell
.\.venv-build\Scripts\python.exe -m pip install -e ".[full]"
```

创建桌面启动按钮：

```powershell
.\install_shortcut.ps1
```

启动按钮使用当前源码环境。更新源码和所需依赖后，无需重新封装 EXE。

## 需要哪些文件

| 功能 | Lite Mode | Full Mode |
| --- | --- | --- |
| Band | `EIGENVAL`、`KPOINTS`、`POSCAR`、`OUTCAR` | `vasprun.xml(.gz)`、`KPOINTS` |
| Projected Band | Band 文件加 `PROCAR` | XML 中需包含投影数据 |
| DOS / PDOS | `DOSCAR`、`POSCAR`、`OUTCAR` | 含 DOS 数据的 `vasprun.xml(.gz)` |
| Combined | 分别选择 Band 和 DOS/SCF 文件夹 | 两侧可分别指定读取模式 |
| Fermi Surface | `EIGENVAL`、`KPOINTS`、`POSCAR`、`OUTCAR` | 含本征能和 k 点的 `vasprun.xml(.gz)` |

费米面需要完整规则三维 k 网格；Line-mode 路径和未展开的对称约化网格不满足当前实现要求。当前费米面未提供元素或轨道投影着色。

缺少投影数据时，普通能带或总 DOS 仍可使用。Full Mode 由用户主动选择；默认 Lite Mode 不读取 XML。

## 缓存与输出

- 第一次解析 PROCAR 或 Full Mode XML 时，在计算目录的 `.lyra-cache` 中保存规范化数据。
- 源文件大小、修改时间或解析器版本变化后，缓存会失效。
- 更换投影元素或轨道可以复用完整缓存；耗时仍取决于数据量和绘图规模。
- 预览为临时 PNG；点击保存后生成正式 PNG、PDF 或 SVG。
- 正式导出附带 JSON/HTML 质量报告，列出证据和限制。

## 验证与开发

```powershell
.\.venv-build\Scripts\python.exe -m pip install pytest pytest-subtests
.\.venv-build\Scripts\python.exe -m pytest -q
```

后台测试默认跳过部分真实 GUI 测试。仅在独立测试桌面设置 `LYRA_RUN_GUI_TESTS=1` 后运行完整 GUI 测试，避免影响正在使用的桌面窗口。

本预览版提供源码，不附带打包 EXE。维护者可使用 `build_lyra.ps1 -PythonPath <python.exe>` 构建；本次发布不宣称已验收独立 EXE。

## 来源与许可说明

本次发布未为 Lyra 自有代码选择新的开源许可证。详见 [NOTICE.md](NOTICE.md)。

内置 INCAR 知识库基于 VASP Wiki，保留来源、修订信息与 GNU FDL 1.2 许可文本；中文内容由 Lyra 辅助整理，非 VASP 官方翻译。参见 [知识库来源说明](lyra/resources/incar/ATTRIBUTION.md)。

仓库不包含 VASP 程序、POTCAR、用户计算数据、服务器配置或登录凭据。
