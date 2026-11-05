# G20 成员分工

本项目复现 SelfCheckGPT 的 unigram 分支，比较平均负对数词频与最大负对数词频，并完成 K=1/3/5 的参考输出数量消融。

| 成员 | 负责内容 | 对应材料 |
|---|---|---|
| 张昊 1120233632 | 论文阅读、复现范围与原文实验定位、报告中的方法说明和参考文献 | 论文 §5.3、表2、表7、图8；report.pdf 第1—2页 |
| 赖轲楠 1120231192 | 项目联系、环境配置、主实验运行流程、代码整理与官方实现对照 | code/run_experiment.py；results/pilot/；results/upstream_verification.json；results/run_log.jsonl |
| 李方洲 1120232973 | AUROC/AP 评价、采样数量消融、结果汇总与统计分析 | code/evaluate.py；results/metrics_by_run.csv；results/summary.csv；results/bootstrap_ci.json |
| 李沐曦 1120232679 | 错误案例分析、报告图表、PPT与汇报材料整理 | results/cases.json；results/cases.md；slides.pptx；presentation_notes.txt |

## 项目完成内容

采用作者公开的238篇文本、1,908个标注句子，运行三种参考采样预算和三个随机种子，另以20个参考输出对照原论文。共保存19,080条逐句结果、12个完整案例、采样索引、配置和原始日志。

完整评分耗时约127.27秒，指标计算与bootstrap约5.21秒。K=5时，Max的AP为84.04±0.65%，Avg为78.43±0.47%。20参考设置下，Max的AP为85.64%，与论文AUC-PR 85.63%接近。3篇样本、6项评分与官方实现的最大绝对差小于1e-12。

## 代码与数据来源

作者仓库：https://github.com/potsawee/selfcheckgpt 。固定版本：19b492a2a380931bf1ed0ca94a9565c9aa7b03e1。

code/upstream/modeling_ngram.py原样保留，MIT许可见同目录LICENSE。文件SHA-256为9284824d5ad519b2e6636a781f0072eaae9495bf6557388c49a7aa3a2783fbee。

本项目新增分词缓存、固定数据校验、嵌套参考采样、逐句结果保存、评价和bootstrap分析。保留作者的小写处理、原文计数、标点、零平滑和自然对数评分设置。未使用其他小组的代码或实验结果。

数据来自作者公开的WikiBio GPT-3 Hallucination Dataset v3，采用CC-BY-SA-3.0许可，来源与取得方式见data/README.md。PPT沿用课程提供的7页模板。

## AI使用说明

使用Codex辅助资料检索、代码编写、实验执行和文稿整理。报告中的数值来自本次运行日志；评分实现通过与作者代码的逐句对照检查。依赖工具包括spaCy、NumPy、NLTK、scikit-learn、SciPy、Matplotlib和ReportLab。
