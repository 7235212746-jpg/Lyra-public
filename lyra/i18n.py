from __future__ import annotations

import re


LANGUAGE_CHOICES = {"中文": "zh_CN", "English": "en_US"}
LANGUAGE_NAMES = {value: key for key, value in LANGUAGE_CHOICES.items()}


ENGLISH_TEXT = {
    "电子结构绘图工作台": "Electronic Structure Workbench",
    "读取模式": "Read Mode",
    "绘图模块": "Plot Modules",
    "能带": "Band",
    "态密度": "DOS",
    "能带 + 态密度": "Band + DOS",
    "能带对照": "Band Comparison",
    "费米面": "Fermi Surface",
    "成键分析 · 规划中": "Bonding Analysis · Planned",
    "价态与电荷 · 规划中": "Valence & Charge · Planned",
    "远程 Linux": "Remote Linux",
    "INCAR 助手": "INCAR Assistant",
    "手机遥控": "Mobile Control",
    "设置": "Settings",
    "v0.2\nLite 优先，Full 按需": "v0.2\nLite by default, Full on demand",
    "选择一次能带计算文件夹，Lite Mode 默认不读取 vasprun.xml。":
        "Select one band-calculation folder. Lite Mode does not read vasprun.xml by default.",
    "DOS/PDOS 从独立的 DOS 或 SCF 文件夹读取。":
        "Read DOS/PDOS from a dedicated DOS or SCF folder.",
    "两侧文件夹分别选择；默认统一采用 DOS/SCF 的 E-fermi。":
        "Choose the two folders separately; DOS/SCF E-fermi is the shared reference by default.",
    "并排比较两个计算；点击真实 k 点后给出投影相似候选与两侧组成。":
        "Compare two calculations side by side; click a real k point to inspect ranked projection candidates and both compositions.",
    "选择完整三维均匀 k 网格计算文件夹；高对称 Line-mode 能带不能用于费米面。":
        "Select a full 3D uniform k-mesh calculation; a high-symmetry line path cannot produce a Fermi surface.",
    "论文图默认输出 PNG 600 dpi，并同时保留 PDF 与 SVG 矢量文件。":
        "Publication figures default to 600 dpi PNG with PDF and SVG vector copies.",
    "浏览服务器计算目录、检查 VASP 任务，并选择性同步绘图数据和缓存。":
        "Browse server calculations, inspect VASP jobs, and selectively sync plot data and caches.",
    "离线查询 VASP 标签，逐条解释当前 INCAR，并显示来源和风险证据。":
        "Look up VASP tags offline, explain each INCAR entry, and show sources and risk evidence.",
    "在手机浏览器中管理 Lyra 任务、查看进度、放大预览和下载结果。":
        "Manage Lyra jobs, inspect progress, enlarge previews, and download results from a phone browser.",
    "打开远程工作台": "Open Remote Workspace",
    "打开 INCAR 助手": "Open INCAR Assistant",
    "打开手机控制中心": "Open Mobile Control Center",
    "语言": "Language",
    "输出文件夹": "Output Folder",
    "选择文件夹": "Choose Folder",
    "格式": "Formats",
    "图宽 (mm)": "Figure Width (mm)",
    "预览与输出": "Preview & Output",
    "生成图片后在此预览": "The generated figure will appear here",
    "打开图片": "Open Image",
    "放大预览": "Enlarge Preview",
    "局部放大图": "Local Zoom Plot",
    "准备局部图…": "Preparing Local Zoom…",
    "打开文件夹": "Open Folder",
    "保存图片": "Save Image",
    "取消": "Cancel",
    "查看质量报告": "View Quality Report",
    "数据类型": "Data Type",
    "完整能带": "Complete Band",
    "元素 / 轨道投影": "Element / Orbital Projection",
    "投影样式": "Projection Style",
    "完整能带颜色": "Complete Band Colour",
    "绘制元素 / 轨道投影": "Plot Element / Orbital Projections",
    "能带投影": "Band Projection",
    "DOS 投影": "DOS Projection",
    "能带样式": "Band Style",
    "查看方式": "View",
    "三维费米面": "3D Fermi Surface",
    "二维切片": "2D Slice",
    "能量偏移 E-Ef (eV)": "Energy Offset E-Ef (eV)",
    "自旋": "Spin",
    "三维视角：仰角": "3D View: Elevation",
    "方位角": "Azimuth",
    "切片法向": "Slice Normal",
    "位置 (-0.5 至 0.5)": "Position (-0.5 to 0.5)",
    "预览": "Preview",
    "读取模式": "Read Mode",
    "清缓存": "Clear Cache",
    "重建缓存": "Rebuild Cache",
    "+ 添加投影通道": "+ Add Projection Channel",
    "至": "to",
    "Band 文件夹": "Band Folder",
    "左侧 Band 文件夹": "Left Band Folder",
    "右侧 Band 文件夹": "Right Band Folder",
    "打开能带对照": "Open Band Comparison",
    "DOS / SCF 文件夹": "DOS / SCF Folder",
    "费米面计算文件夹": "Fermi-Surface Folder",
    "能量范围 (eV)": "Energy Range (eV)",
    "DOS 纵轴范围 (states/eV)": "DOS Y Range (states/eV)",
    "DOS 横轴范围 (states/eV)": "DOS X Range (states/eV)",
    "就绪": "Ready",
    "尚未选择文件夹": "No folder selected",
    "缓存：待验证": "Cache: pending validation",
    "选择文件夹后检测 PROCAR 或 XML 投影数据。":
        "Select a folder to detect PROCAR or XML projection data.",
    "计算质量：尚无分析结果": "Calculation quality: no result yet",
    "准备分析…": "Preparing Analysis…",
}


def translate(text: str, language: str) -> str:
    if language != "en_US":
        return text
    translated = ENGLISH_TEXT.get(text)
    if translated is not None:
        return translated
    channel = re.fullmatch(r"通道\s+(\d+)", text)
    if channel:
        return f"Channel {channel.group(1)}"
    return text
