# 各类评估基准（Benchmark）介绍

以下是常用于评估语言模型能力的基准测试，包括评估的能力、指标以及官网链接：

---

## 📘 1. MMLU（Massive Multitask Language Understanding）

- **评估能力**：测试模型在57个学科（包括STEM、人文、社会科学等）中的多项选择题表现，涵盖从小学到专业级别的难度。
- **指标**：准确率（Accuracy），即模型在零样本（zero-shot）或少样本（few-shot）设置下答对的题目比例。
- **官网链接**：[https://paperswithcode.com/dataset/mmlu](https://paperswithcode.com/dataset/mmlu)

---

## 🧠 2. BBH（BIG-Bench Hard）

- **评估能力**：专注于23个被认为超出当前语言模型能力的挑战性任务，如逻辑推理、规划和复杂的语言理解。
- **指标**：准确率（Accuracy）或任务特定的评分标准。
- **官网链接**：[https://paperswithcode.com/dataset/bbh](https://paperswithcode.com/dataset/bbh)

---

## 📚 3. DROP（Discrete Reasoning Over Paragraphs）

- **评估能力**：测试模型在阅读理解中进行离散推理的能力，如加法、计数和排序。
- **指标**：Exact Match（EM）和 F1 分数，分别衡量模型答案与参考答案的精确匹配程度和部分匹配程度。
- **官网链接**：[https://paperswithcode.com/dataset/drop](https://paperswithcode.com/dataset/drop)

---

## 💻 4. HumanEval

- **评估能力**：评估大型语言模型在代码生成任务中的性能，特别是生成功能正确的代码。
- **指标**：pass@k，表示在 k 次尝试中至少有一次生成的代码通过所有单元测试的比例。
- **官网链接**：[https://docs.confident-ai.com/docs/benchmarks-human-eval](https://docs.confident-ai.com/docs/benchmarks-human-eval)

---

## 🔄 5. CRASS（Counterfactual Reasoning Assessment）

- **评估能力**：测试模型处理反事实条件句的能力，即理解和推理与现实相反的假设情境。
- **指标**：Top-1 Accuracy，即模型最有信心的预测与正确答案匹配的比例。
- **官网链接**：[https://aclanthology.org/2022.lrec-1.229](https://aclanthology.org/2022.lrec-1.229)

---

## 🧑‍💻 6. Vicuna Benchmark

- **评估能力**：通过与 GPT-4 等模型的对比，评估 Vicuna 模型在对话生成中的质量和连贯性。
- **指标**：使用 GPT-4 作为评审，进行对话质量的主观评分。
- **官网链接**：[https://lmsys.org/blog/2023-03-30-vicuna](https://lmsys.org/blog/2023-03-30-vicuna)

---

## 🗣️ 7. MT-Bench（Multi-turn Benchmark）

- **评估能力**：测试模型在多轮对话中的连贯性、信息性和指令遵循能力。
- **指标**：通常使用 GPT-4 等模型作为评审，对每轮对话进行评分，最终得分为最低轮次得分，以反映对话中的最弱环节。
- **官网链接**：[https://klu.ai/glossary/mt-bench-eval](https://klu.ai/glossary/mt-bench-eval)

---

