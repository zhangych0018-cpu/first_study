# DSG-SWS 贝叶斯优化仿真项目进度总览

更新时间：2026-06-04  
项目目录：`C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization`

## 1. 当前项目定位

本项目已经从早期的 RGS-SWG 慢波结构路线切换为 **DSG-only** 主线，当前目标是构建一个面向 **W 波段 double-staggered grating (DSG) 慢波结构** 的贝叶斯优化与仿真联动平台。

当前软件主线分为两层：

1. **可稳定运行的 mock 优化链路**  
   用于算法验证、接口联调、失败容错、基线与消融实验。
2. **正在接入的真实 CST 冷结构仿真链路**  
   用于把真实 `.cst` 工程纳入自动化求解、结果导出与后处理。

## 2. 已完成事项

### 2.1 DSG-only 优化框架

以下能力已经完成并接入到 DSG 主线：

- DSG 问题定义：`W / P / T / G / H`
- 初始实验设计：LHS / Sobol / hybrid
- 几何合法性检查
- mock DSG 仿真器
- independent GP / ModelListGP 代理模型
- Matern-5/2 ARD 核
- constrained qNEHVI 采集函数
- BO 主循环
- baseline
- ablation
- robust optimization
- multifidelity demo
- 失败仿真容错与结果记录
- DSG postprocessing parser

### 2.2 真实 CST 单点链路已打通一部分

已完成真实 `.cst` 工程的自动化读取与导出验证，核心脚本为：

- `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\dsg_cst_automation.py`

已验证能力：

- 自动加载本机 CST Python 接口
- 打开真实工程 `C:\Users\87007\Desktop\SWS\DSG_SWS.cst`
- 调用 CST solver
- 枚举结果树
- 导出 S 参数文本和复数 CSV
- 导出全部可读 1D Results
- 保存参数快照与导出摘要

## 3. 当前已验证状态

### 3.1 测试状态

已于 2026-06-04 实际运行：

```bash
C:\Users\87007\.conda\envs\sws_predict_env\python.exe -m pytest -q
```

结果：

- `28 passed, 7 warnings in 12.28s`

说明：

- 当前仓库在 DSG 主线下测试通过。
- warning 主要来自 BoTorch / GPyTorch 数值告警和一个 pandas FutureWarning，当前不阻塞功能运行。

### 3.2 mock demo 与结果目录

当前主要结果目录：

- `data/results/dsg_default_run`
- `data/results/dsg_mock_demo`
- `data/results/multifidelity_demo`
- `data/results/paper_figures`

其中 `data/results/dsg_mock_demo` 已包含：

- `evaluations.csv`
- `history.csv`
- `pareto.csv`
- `representative_designs.csv`
- `summary.json`
- `figures/`

### 3.3 真实 CST 导出状态

当前真实 CST 导出目录：

- `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\dsg_cst_exports\run_20260604_153424`

该目录已经包含：

- `sparameters.txt`
- `sparameters_complex.csv`
- `project_parameters.json`
- `tree_items.txt`
- `export_summary.json`
- `all_1d_results/`

根据 `export_summary.json`，当前真实工程已识别到：

- `S-Parameters`
- `Port Information`
- `Power`
- `Port signals`

并且 `all_tree_items_count = 41`。

## 4. 真实 CST 接入中学到的关键结论

### 4.1 Python 接口调用方式

本机默认 Python 环境不能直接可靠导入 `cst`，需要优先考虑：

- 显式补充 CST 安装路径
- 使用 `cst.interface`
- 使用 `cst.results`

真实工程打开的推荐模式是：

1. `DesignEnvironment.connect_to_any_or_new()`
2. `open_project(...)`
3. `project.model3d.run_solver()`
4. 使用 `cst.results.ProjectFile(...)` 读取结果树

### 4.2 COM 可启动，但不能证明导出链路可用

本机已经验证：

- `win32com.client.Dispatch("CSTStudio.Application")` 可以启动

但同时也验证到：

- COM `ASCIIExport` 并不可靠
- 会报错：`The ASCII export option is not available for the current view.`

结论：

- **不能把 COM 可启动误判为“后处理自动化已打通”**
- 当前更稳的导出方式是 **`cst.results` 直接读结果树后写出仓库控制的 CSV/TXT**

### 4.3 当前真实工程还不满足完整 DSG BO 指标输入

当前真实工程虽然已经能稳定导出 S 参数，但在 `export_summary.json` 中仍显示：

- `mode_frequencies = []`
- `dispersion_like = []`
- `coupling_impedance_like = []`
- `dsg_bo_required_complete = false`

这说明：

- 真实工程已经能支撑一部分冷测数据链路
- 但还没有直接提供完整 DSG 贝叶斯优化所需的 `dispersion / Kc / mode-frequency` 标准后处理输入

## 5. 当前的核心阻塞点

要把真实 CST 全面接入 DSG BO 主循环，还缺下面这些内容中的至少一部分：

1. **真实工程中的色散曲线结果节点**
2. **真实工程中的 coupling impedance / Kc 结果节点**
3. **能稳定映射到 `TM21-like` 与基模竞争的模式识别信息**
4. **与现有 parser 对齐的导出格式，或新的 parser 适配规则**

换句话说，当前不是算法层卡住，而是：

- **真实工程模板的结果组织方式，还没有完全对上 BO 所需指标**

## 6. 下一步最值得做的工作

### 方案 A：优先继续打通真实 CST 全链路（推荐）

按这个顺序推进最稳：

1. 在 CST 里确认当前工程是否存在：
   - dispersion 相关结果
   - Kc / interaction impedance 相关结果
   - 模式识别相关结果
2. 如果存在，确定它们在结果树中的精确名称
3. 在 `dsg_cst_automation.py` 中新增自动导出这些结果节点的逻辑
4. 把这些导出文件与 `sws_bo/utils/postprocessing.py` 对接
5. 先跑一次真实 **单点** 评估，不直接上 BO
6. 单点评估稳定后，再接回 `sws_bo/utils/cst_interface.py` 与 BO 主循环

### 方案 B：如果当前 CST 工程没有这些结果节点

则需要先改 CST 模板本身：

1. 在工程里补充需要的监视器、结果节点或宏导出逻辑
2. 手工导出一组标准样例文件
3. 再让 Python 自动化脚本对这些标准文件做批处理接入

## 7. 继续工作时优先阅读的文件

如果在新对话中继续本项目，建议优先阅读以下文件：

1. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\README.md`
2. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\dsg_cst_automation.py`
3. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\sws_bo\utils\cst_interface.py`
4. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\sws_bo\utils\postprocessing.py`
5. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\dsg_cst_exports\run_20260604_153424\export_summary.json`
6. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\dsg_cst_exports\run_20260604_153424\tree_items.txt`
7. `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\data\results\dsg_mock_demo\summary.json`

## 8. 新对话续工建议

如果你准备开一个新对话窗口继续本项目，建议直接把下面这句话作为新对话开头：

> 请先阅读 `C:\Users\87007\Desktop\CC\SWS_predict\sws_bayesian_optimization\DSG_PROJECT_STATUS_2026-06-04.md`，然后继续把真实 DSG CST 的 dispersion / Kc / mode-frequency 结果接入到当前贝叶斯优化流程中。

说明：

- 当前工具环境里没有直接可用的“新建线程并自动切换过去”的接口。
- 因此这份文档已经按“可直接交接”的方式整理好了，适合作为下一轮工作的唯一入口。
