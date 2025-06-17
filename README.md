# DMLM: A Diffusion Molecular Language Model for SELFIES Generation
## 基于扩散语言模型的SELFIES分子生成方法

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./毕业论文终稿.pdf) ![Model Architecture](https://github.com/MikeWYX/Diffusion-Molecular-Language-Model/blob/main/assets/model_architecture.png) *图：DMLM模型框架图，展示了基于MOLGEN去噪器的离散扩散过程。*

---

### 简介 (Introduction)

本项目是论文 **《基于扩散语言模型的分子生成方法研究》** 的官方实现。

**DMLM (Diffusion Molecular Language Model)** 是一种新颖的分子生成模型，它将离散扩散模型 (Discrete Diffusion Models) 的迭代细化能力与预训练掩码语言模型 (Masked Language Models, MLM) 的强大上下文理解能力相结合，专门用于生成 **SELFIES (Self-Referencing Embedded Strings)** 分子表示。

我们的核心思想是：
- **利用 SELFIES**：从根本上保证生成分子的100%语法有效性，使模型能专注于学习化学结构和语义。
- **借鉴 DiffusionBERT**：引入了带离散吸收态 (`[MASK]`) 的扩散过程、能够感知符号信息量的 **纺锤形噪声调度 (Spindle Noise Schedule)**，以及无需显式时间步输入的 **时间无关解码 (Time-Agnostic Decoding)** 策略。
- **融合 MOLGEN**：采用专为 SELFIES 优化的预训练模型 **MOLGEN** 作为核心去噪网络，将大规模分子数据中学习到的知识迁移到生成任务中。

实验证明，DMLM 在 MOSES 基准数据集上表现出色，在有效性、新颖性、唯一性和稳定性方面均达到了SOTA水平。
