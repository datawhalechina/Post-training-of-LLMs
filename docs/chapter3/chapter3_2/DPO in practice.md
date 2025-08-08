# 直接偏好优化实践(DPO in Practice)

&emsp;&emsp;在本节课程中，你将进行直接偏好优化(DPO)的实践,在实践中更深刻的理解DPO。

&emsp;&emsp;DPO（直接偏好优化）是一种对比学习方法，它同时从正样本（优选）和负样本（劣选）中学习。  

&emsp;&emsp;在这个实验中，我们将从一个小的 Qwen instruct 模型开始。这个模型有自己的身份标识“Qwen”。当用户问“你是谁？”时，它会回答“我是 Qwen”。然后，我们创建一些对比数据。具体来说，当询问身份时，我们将身份名称从“Qwen”改为“Deep Qwen”，并使用“Deep Qwen”作为正样本（优选回答），“Qwen”作为负样本（劣选回答）。我们使用了一个大规模（数量）的对比数据集，并在现有的 instruct 模型之上进行 DPO 排序训练。之后，我们将得到一个微调后的 Qwen 模型，它拥有了新的身份。当用户问“你是谁？”时，希望模型会回答“我是 Deep Qwen”。
[](https://github.com/datawhalechina/Post-training-of-LLMs/blob/18ad0f15a868d27a255f232d0fd886d9acecdd5f/docs/images/DPO%20in%20Practice.png)

## 导入相关的python库
&emsp;&emsp;我们将从导入相关的库开始，这些库将用于 DPO 编程部分。包括 `torch`、`pandas` 和来自 `transformers` 的库，例如 `AutoTokenizer`、`AutoModelForCausalLM`（如我们之前讨论的）。对于 `TRL` 库，我们也将引入新的 `DPOTrainer` 和 `DPOConfig` 来进行 DPO 训练。我们还会用到 `datasets`，导入 `load_dataset` 和数据集类型。稍后，我们还会使用一个辅助函数（我们上次实现过），它包含生成响应、用问题测试模型以及在此处加载模型和分词器的功能。
```
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
```