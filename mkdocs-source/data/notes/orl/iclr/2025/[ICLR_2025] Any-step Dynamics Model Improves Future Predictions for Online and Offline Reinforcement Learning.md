---
title: "Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning"
description: "ADM 用任意步回溯的直接预测降低模型 rollout 的误差累积，并在离线 MBRL 中用单模型差异估计不确定性。"
tags:
  - Offline RL
  - Model-Based RL
  - Dynamics Model
  - Uncertainty
  - D4RL
---

# Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning

| 字段 | 内容 |
|---|---|
| Title | Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning |
| Year | 2025；arXiv first version 2024 |
| Source | ICLR 2025 Conference Paper |
| Authors | Haoxin Lin, Yu-Yan Xu, Yihao Sun, Zhilong Zhang, Yi-Chen Li, Chengxing Jia, Junyin Ye, Jiaji Zhang, Yang Yu |
| Affiliations | Nanjing University; Polixir Technologies; Peng Cheng Laboratory，见论文首页 |
| Tags | Model-based RL; Offline RL; dynamics model; uncertainty quantification; ADMPO |

**一句话概括：** 论文提出 Any-step Dynamics Model (ADM)，让动力学模型用任意长度的历史状态-动作片段直接预测未来状态，从而减少 rollout 中反复单步 bootstrapping 带来的误差累积。

**我对这篇论文的定位：** 这是 model-based offline RL 中“模型误差如何随 rollout horizon 累积”的方法论文。它和 MOPO/MOReL/COMBO/MOBILE 的共同点是利用模型生成数据并做 pessimism；不同点是用一个可变回溯长度的 ADM 同时改进未来预测和不确定性估计，而不是依赖 ensemble dynamics。

## 1. 第一作者相关信息

第一作者 **Haoxin Lin**。论文首页显示其来自 Nanjing University / LAMDA 系列单位并与 Polixir Technologies 有合作。

## 2. 研究问题

传统 MBRL 用单步模型 $\hat{T}(s_{t+1},r_{t+1}|s_t,a_t)$ rollout。问题是：第 2 步预测依赖第 1 步预测出的状态，第 3 步又依赖第 2 步，误差会随 horizon 累积。这在 online MBRL 中限制 sample efficiency，在 offline MBRL 中则会把 policy 引到模型虚构但数据不支持的区域。

本文设定包括 online 和 offline 两种场景；offline 部分沿用 MOPO 式思想：学到 dynamics model 后生成 model data，并把不确定性作为 pessimistic penalty 放进 Bellman operator。

核心目标是把“频繁单步递归预测”改成“可变长度直接预测”：

$$
\hat{T}_\theta(s_{t+k}, r_{t+k}\mid s_t, a_{t:t+k-1}),\quad k\in\{1,\dots,m\}.
$$

直观含义：模型不只学“下一步”，还学“从较早状态加一串动作直接到未来”的预测，因此 rollout 时可随机回溯，减少当前预测状态对上一预测状态的完全依赖。

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| MBRL | 先学环境模型，再在模型里生成数据或规划 | ADM 的应用框架 |
| Compounding error | rollout 越长，预测误差越积越大 | 论文要解决的核心失败模式 |
| Direct prediction | 直接从较早状态和多步动作预测未来 | ADM 的核心机制 |
| Random backtracking | rollout 时随机选择回溯长度 $k$ | 降低连续 bootstrapping 次数 |
| Model uncertainty | 模型对某状态动作预测不可靠的程度 | offline ADMPO-OFF 的惩罚来源 |
| Pessimism | 对不确定区域降低价值估计 | 防止 policy 利用模型幻觉 |

offline MBRL 的风险来自“模型外推”。如果数据集没有覆盖某些状态-动作区域，模型在那里可能看似自信但实际错得很远。MOPO 用 ensemble 方差给 reward 加惩罚；MOBILE 用 Model-Bellman inconsistency 估计误差。ADM 的新意是：不用多个模型，而是利用不同回溯长度给出的预测差异作为不确定性信号。

## 4. 问题分析

论文先指出单步模型 rollout 的 bootstrapping prediction 是误差爆炸的结构性原因。即使单步误差很小，长 rollout 也可能偏离真实轨迹，导致 policy gradient 被虚假样本误导。

Figure 2 直接比较 ADM、ensemble dynamics model 和 bootstrapping RNN dynamics model 的 compounding error。结果显示在 hopper/walker2d 的 D4RL 数据上，ADM 的误差曲线接近零，而另外两类模型在 rollout 长度增大后呈指数增长。

理论上，论文定义 ADM 的不确定性 $U_{\mathrm{ADM}}$，并在 Theorem 3.4 中说明 $\beta U_{\mathrm{ADM}}$ 可作为 PEVI 风格的 uncertainty quantifier，用来上界 proxy Bellman operator 与真实 Bellman operator 的差异：

$$
\hat{T}^{\pi}Q(s_t,a_t)-T^{\pi}Q(s_t,a_t)\le \beta U_{\mathrm{ADM}}(s_t,a_t).
$$

含义：如果不同回溯长度下模型预测差异大，说明该区域不可靠，应在 value update 中惩罚。

## 5. 思想与方法

ADM 的指导原则是：**不要让每一步预测都完全站在上一预测状态上，而是让模型能从历史片段直接预测未来，并把不同回溯长度的分歧当作不确定性。**

方法组件：

- **Any-step dynamics model**：用 RNN/GRU 结构接收 $s_{t-k+1}$ 和动作序列 $a_{t-k+1:t}$，预测 $s_{t+1}, r_{t+1}$。
- **ADM-Roll**：rollout 时每一步随机采样 $k\in[1,m]$，从真实或已预测的历史片段回溯生成下一状态。
- **ADMPO-ON**：在线版本，类似 MBPO，用 ADM 替换 ensemble dynamics。
- **ADMPO-OFF**：离线版本，类似 MOPO，用 ADM 不确定性构造 pessimistic Bellman operator：

$$
\hat{T}_{\mathrm{ADM}} Q(s,a)
=\hat{T}^{\pi}Q(s,a)-\beta U_{\mathrm{ADM}}(s,a).
$$

这使得 policy 在模型不一致区域得到较低价值，减少利用模型错误的倾向。

## 6. 算法与伪代码

算法名：**ADMPO-ON / ADMPO-OFF**。

1. 从真实数据或离线数据中采样长度至少为 $m$ 的状态-动作片段。
2. 训练 ADM，使其对任意 $k\le m$ 的输入片段预测对应未来状态和奖励。
3. rollout 时给定起始片段，重复以下步骤：
4. 用当前 policy 采样动作。
5. 随机选择回溯长度 $k$。
6. 将回溯状态和后续动作序列输入 ADM，得到下一状态和奖励。
7. online 版本把生成样本加入 replay buffer 后用 SAC/MBPO 式更新。
8. offline 版本计算 $U_{\mathrm{ADM}}$ 并在 Bellman target 中减去惩罚项，形成 ADMPO-OFF。

重要超参数是最大回溯长度 $m$。Appendix 的 Table 10 显示 $m=1$ 或固定多步模型效果很差，$m$ 增至 3 或 5 后显著改善，继续增大收益有限。

## 7. 实验与消融

实验回答三个问题：ADM 是否减少 compounding error；ADMPO-ON 是否提升 online sample efficiency；ADMPO-OFF 是否提升 offline MBRL 并提供更好不确定性。

主要结果：

- Figure 2：ADM 长 rollout 误差显著低于 ensemble dynamics 和 bootstrapping RNN。
- Figure 3：online MuJoCo-v3 上，ADMPO-ON 相比 MBPO、BMPO、DDPPO、STEVE 等更快接近 SAC asymptotic performance，在 Humanoid 上 sample efficiency 尤其明显。
- Table 1：D4RL MuJoCo 上 ADMPO-OFF 平均 normalized score 为 81.0，超过 MOPO、COMBO、RAMBO、MOBILE 等 model-based baseline，也高于若干 model-free 方法。
- Appendix Table 6：AntMaze 中 ADMPO-OFF 的平均分高于 MOPO/COMBO/RAMBO/MOBILE，但 large-diverse 仍为 0，说明长程稀疏目标仍困难。
- Table 10：any-step 设计和合适的 $m$ 对性能很关键；固定 multi-step 或 $m=1$ 明显退化。
- Table 11：ADMPO-OFF 参数量小于 MOPO ensemble，但 GPU memory 可更高，runtime 大体相近。

实验弱点是主要集中在 MuJoCo/AntMaze 等连续控制模拟环境；论文也承认高随机环境中 ADM 的不确定性可能受随机性干扰。

## 8. 展望

研究启发：

- dynamics model 的结构可以直接影响 pessimism 的质量，而不只是后处理 uncertainty。
- ensemble-free uncertainty 对大模型 MBRL 有吸引力，但需要辨别 epistemic uncertainty 与环境 stochasticity。
- variable-horizon prediction 可能和 temporal abstraction、options、latent dynamics planning 结合。

局限：

- 高随机环境下不同回溯长度的预测差异未必等于模型错误。
- ADM 需要更高 memory，且 $m$ 的选择有任务依赖。
- AntMaze large 等任务仍存在明显不足。

后续方向：

- 将 ADM 与 latent dynamics 或 diffusion planner 结合，减少高维观测 rollout 的误差。
- 在真实机器人数据上测试 single-model uncertainty 是否可靠。
- 用 ADM 的不确定性指导数据采集或 offline-to-online fine-tuning。

## Links

- Paper page: https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0506debbf49e31d25690fbd1e69cd2f-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2405.17031
- Code: https://github.com/LAMDA-RL/ADMPO
