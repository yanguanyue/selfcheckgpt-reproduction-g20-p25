# 数据来源与许可

作者数据集 `potsawee/wiki_bio_gpt3_hallucination` v3（2023-05-05 发布规模），取得日期 2026-11-05。

来源：https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination

许可 CC-BY-SA-3.0：https://creativecommons.org/licenses/by-sa/3.0/ 。原作者 Potsawee Manakul 等。随包 dataset.json 为作者公开 JSON 原样副本；没有改动文本、标签或数据顺序。本项目的抽样与错误分析为衍生研究材料，继续遵守相同数据归属与相同方式共享条件。

共 238 篇、1,908 句；每篇 20 个同一 GPT-3 模型生成的参考输出。原三类标签 accurate=516、minor_inaccurate=631、major_inaccurate=761。非事实正类=后两类，共 1,392 句；不反转分数方向。

使用完整 evaluation split，不新设训练/验证划分，不拟合参数或选择阈值。全部人物 ID 见 `../results/dataset_manifest.json`；K=1/3/5 的具体参考索引、种子见 `../results/sample_selections.json`。同一篇同一种子严格嵌套抽样。

SHA-256：`9fea0263501d74b192b6f75dae5c25a67db7938e096df4aab18c65c42e83189c`；文件 5,114,676 字节。完整下载地址及返回元数据见 `source.json`。如需重新获取，在包根目录执行 `python code/download_data.py`；版本变化时校验失败，不能静默更换数据。

`wiki_bio_text` 只用于报告中的事后参照与错误解释，不参与算法输入。算法读取原生成文本、作者句子切分，以及采样生成文本。人工标签只用于评价。原样保留截断句，案例中明确讨论此边界。
