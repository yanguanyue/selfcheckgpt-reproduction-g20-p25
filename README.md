# G20 P25 SelfCheckGPT 课程复现

[完整作业压缩包](NLP2026_G20_P25.zip) · [报告 PDF](report.pdf) · [汇报 PPT](slides.pptx) · [汇报 PDF](slides.pdf)

本仓库保存本次真实运行的代码、配置、公开数据、报告与实验结果。

课程：自然语言处理项目实践 2026。联系人：赖轲楠（1120231192）。
成员：张昊（1120233632）、赖轲楠（1120231192）、李方洲（1120232973）、李沐曦（1120232679）。

## 范围与服务器

原设置部分复现，只复现 n-gram 分支的 unigram average / maximum 两种一致性评分。沿用作者公开的全部 238 篇、1,908 句与 20 个参考输出。必做消融为 K=1/3/5，随机种子 42/43/44，另有 K=20 的论文锚点。无需训练、GPU、服务器、API Key 或付费接口。32 GB 内存 Windows CPU 实测主实验 127.27 秒，评价与 bootstrap 5.21 秒；首次安装、下载耗时另计。机器与网络不同，建议预留 5—15 分钟运行和更长的首次安装时间。

## 环境与运行

Python 3.12。建议创建独立环境，在本目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe code/run_experiment.py --limit 3 --output results/pilot --verify-upstream
.venv\Scripts\python.exe code/run_experiment.py --verify-upstream
python -m venv .eval
.eval\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.eval\Scripts\python.exe code/evaluate.py
.eval\Scripts\python.exe code/verify_results.py
```

数据已按 CC-BY-SA-3.0 随包提供，可离线运行。若缺失，执行 `python code/download_data.py`，其校验值必须匹配 `data/source.json`。不要从其他仓库下载同名但不同版本的数据后混用。

实际运行使用两个环境：评分为 Python 3.12.7、spaCy 3.8.11、en_core_web_sm 3.8.0、NumPy 2.5.4；评价为同版 Python、NumPy 1.26.4、scikit-learn 1.8.0、SciPy 1.14.1、Matplotlib 3.9.2。实际环境完整记录在 `results/environment.json` 和 `results/evaluation_environment.json`。为避免 NumPy 版本冲突，上述命令复用相同的两个环境方案。原始评分依赖清单另存于 `configs/scoring-freeze.txt`；评价的直接依赖见 `requirements-evaluation.txt`。

## 复现流程与结果入口

1. 阅读论文第 5.3 节、表 2 的 Unigram(max)、附录 D 表 7 的 Avg/Max 1-gram 行，以及图 8 的采样数量实验。
2. `code/download_data.py` 取得固定数据；`results/dataset_manifest.json` 保存全部人物 ID 与标签统计。
3. 先做 3 篇资源试跑，再运行完整评分。每个人物按 `[seed, doc_id]` 创建 NumPy SeedSequence；同一 seed 内 K=1/3/5 为同一随机排列的前缀，两种评分共享参考输出。
4. `code/run_experiment.py` 用 spaCy 分句并重新分词，合并原文与 K 个参考输出，统一小写，计算词频概率；输出每句 avg/max 负对数词频（自然对数、单位 nats）。保留标点，不平滑；加入待检测原文是作者实现的原始设置，不是额外事实来源。
5. `code/evaluate.py` 计算 AP、AUROC、梯形 PR-AUC、事实类 AP、段落相关性、三种子均值/样本标准差，以及 K=5/seed=42 的 1,000 次按篇聚类配对 bootstrap。不得把种子标准差与置信区间混为一谈。
6. 报告 `report.pdf` 正文 5 页，后接参考文献与附录；`slides.pptx` 和 `slides.pdf` 为 7 页主讲，`presentation_notes.txt` 提供 5 分钟讲稿与 1 分钟证据演示。

关键证据：

- `results/predictions.jsonl`：19,080 条记录；含句 ID、作者三类标签、二类标签、avg/max 输出、K、seed、参考索引与配置路径。
- `results/metrics_by_run.csv`：逐次指标；`results/summary.csv`：三种子汇总。
- `results/sample_selections.json`：每篇实际抽中的参考索引；`results/bootstrap_ci.json`：配对区间。
- `results/run_log.jsonl` 与 `results/console.log`：真实运行日志；`results/pilot/`：资源试跑证据。
- `results/upstream_verification.json`：3 篇、6 项评分与固定版本作者实现一致，误差小于 1e-12。
- `results/cases.json` / `cases.md`：12 例完整输入、标签、两种输出、原始 WikiBio 参照和 5 个参考文本。参照仅用于事后分析，绝不作为评分输入。
- `results/budget_curve.png`：K=1/3/5 预算曲线；误差棒为三个参考抽样种子的样本标准差。

非事实类定义：minor_inaccurate 或 major_inaccurate 为 1，accurate 为 0；分数越大越偏非事实。无技能 AP=1392/1908=72.956%。K=5 时 avg AP=78.43±0.47%，max AP=84.04±0.65%；max AUROC=69.87±0.30%。K=20 max AP=85.639%，梯形 PR-AUC=85.624%，论文表 2 的 AUC-PR=85.63%，属于近似重现而非宣称完全相等。

这里没有训练、验证集选参数或在评估集调分类阈值；公开 evaluation 集上的多配置比较属于复现与探索性分析，不是新盲测。12 例案例按排序边界筛选，不能当成预先确定分类器的误报率统计。

## 来源与改动

- 论文：https://aclanthology.org/2023.emnlp-main.557/ ，Manakul、Liusie、Gales，EMNLP 2023，9004—9017。
- 作者代码：https://github.com/potsawee/selfcheckgpt ，固定 commit `19b492a2a380931bf1ed0ca94a9565c9aa7b03e1`。
- `code/upstream/modeling_ngram.py` 原样保留，MIT 许可随附；本组实现合并分词缓存以减少重复加载，用原实现核对实际分数。没有使用后增的 NLI 分支替代课程限定分支。
- 数据：https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination ，v3，许可 CC-BY-SA-3.0。
- 课程要求：https://nlp.hltbit.top/2026/materials/w01/classroom/papers.html 与同目录 `requirements.html`。
- AI 辅助及建议个人分工见 `contributions.md`，自动运行不冒充四名成员本人已完成核验。

## 提交前

按 `contributions.md` 完成各自核验后更新实际贡献。核对教师确认组号/论文号、最后修改材料、检查压缩包可解压。截止 2026-11-22 24:00，由联系人上传延河课堂并确认提交，登记表填写教师可访问的材料链接。当前包不会自动上传、创建公开仓库或修改在线登记表。当前进展记录写明实际日期，不回填 10 月的完成时间。
