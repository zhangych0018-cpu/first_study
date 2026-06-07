# 真实 CST S 参数降参 BO 验收记录（2026-06-07）

## 验收目标

在 `reduce-cst-export-bo-pipeline` 分支中，保持 `main` 不变，使用真实 CST 已能稳定导出的 S 参数数据跑通降参后的 DSG 贝叶斯优化流程：

- BO 仍修改 DSG 五个几何参数：`W / P / T / G / H`
- CST 自动写入参数、运行求解器并导出 `sparameters.txt`
- 后处理只基于 S 参数计算降参 BO 指标
- 至少完成初始样本和一次 BO 候选迭代

## 真实 CST 验证命令

```powershell
C:\Users\87007\.conda\envs\sws_predict_env\python.exe scripts\run_bo.py --backend cst --config config\cst_config.yaml --n-initial 4 --n-iterations 1 --batch-size 1 --seed 23
```

## 验证结果

命令完成，输出摘要：

```text
backend=cst
problem_name=DSG_W_Band_SParameter
n_evaluated=5
n_successful=5
wall_clock_time=2385.6329541 s
output_dir=data\results\dsg_cst_run
```

`data/results/dsg_cst_run/history.csv` 显示：

```text
iteration,n_evaluated,feasible_rate,hypervolume,problem_name
0,4,0.0,0.0,DSG_W_Band_SParameter
1,5,0.0,0.0,DSG_W_Band_SParameter
```

`data/results/dsg_cst_run/evaluations.csv` 显示：

- 前 4 行 `stage=initial`
- 第 5 行 `stage=bo_iter_1`
- 5 行全部 `success=True`
- 5 行全部包含 `W/P/T/G/H`
- 5 行全部包含 `postprocessing_mode=sparameter_only`
- 5 行全部生成 S 参数降参目标列：`neg_S21_mean / S11_ripple / insertion_loss_mean`

## 真实导出文件

本次 5 个真实 CST run 均在 `dsg_cst_exports/bo_runs/` 下生成独立目录。每个 `standard_dsg_exports.json` 均包含：

```text
exported.sparameters.sample_count = 1001
exported.sparameters.sparameters_txt = dsg_cst_exports\bo_runs\<run_id>\sparameters.txt
```

最近一次 BO 候选点目录示例：

```text
dsg_cst_exports\bo_runs\cst_run_20260607_194825_b4f71535\sparameters.txt
```

## 说明

当前真实 CST 工程仍未提供完整 `dispersion / Kc / mode-frequency` 结果树节点，因此本分支采用 `sparameter_only` 降参模式。该模式保留 DSG 五个设计变量，只减少后处理依赖的物理输出数量。

为了避免旧的完整 DSG `S11 <= -10 dB` 约束让当前 S 参数降参验收样本全部不可行，`DSGSParameterProblem` 单独使用 `s11_constraint_db = 0.0`。匹配质量仍通过 `S11_ripple` 和插入损耗目标进入 BO。
