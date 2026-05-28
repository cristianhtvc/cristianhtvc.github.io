---
title: "Online-to-Offline RL for Agent Alignment"
description: "将已在线训练好的游戏智能体用少量离线人类行为数据对齐到目标偏好。"
tags:
  - Agent Alignment
  - Preference Learning
  - Online-to-Offline RL
  - Game AI
---

# Online-to-Offline RL for Agent Alignment

| 字段 | 内容 |
|---|---|
| Title | Online-to-Offline RL for Agent Alignment |
| Year | ICLR 2025；OpenReview 发布于 2025-01-22，最终修改于 2025-02-28 |
| Source | ICLR 2025 Poster |
| Authors | Xu Liu, Haobo Fu, Stefano V. Albrecht, Qiang Fu, Shuai Li |
| Affiliations | Shanghai Jiao Tong University; Tencent; University of Edinburgh |
| Tags | Agent Alignment; Preference Learning; RLHF; Game AI; Online-to-Offline RL |

**一句话概括**：ALIGN-GAP 形式化 online-to-offline RL：先保留在线训练 agent 的环境能力，再用极少离线人类偏好轨迹，通过 reward model calibration 和 preference curriculum 把行为风格对齐到目标人群。

**我对这篇论文的定位**：这篇论文不是传统 offline RL，也不是 offline-to-online，而是反向设定：已有强 agent，只有少量人类行为数据，要改变风格而避免 unlearning。它把 LLM/RLHF 的 reward model 思路迁移到 Game AI agent，但强调 RL agent 已有环境动力学知识，不能像从头训练那样直接切换奖励。

## 1. 第一作者相关信息

Xu Liu 的 OpenReview 资料显示其为上海交通大学硕士生，2018-2022 年本科、2022 年起硕士；论文列表包括本文、ICML 2025 `Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation`，以及 ICLR 2024 withdrawn submission `Adaptive Offline Data Replay in Offline-to-Online Reinforcement Learning`。

| Year | Title | Venue/source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2025 | Online-to-Offline RL for Agent Alignment | ICLR 2025 Poster | Agent alignment, preference RL | 本文主体工作 |
| 2025 | Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation | ICML 2025 poster / OpenReview | Offline-to-online RL, diffusion | 相邻方向：离线策略向在线微调迁移 |
| 2024 | Adaptive Offline Data Replay in Offline-to-Online Reinforcement Learning | ICLR 2024 withdrawn / OpenReview | Replay, offline-to-online RL | 早期相关工作，关注在线微调稳定性 |

从可核验记录看，第一作者的研究集中在 online/offline RL 切换、策略迁移和 agent alignment。合作者包括腾讯 AI Lab 研究者、University of Edinburgh 的 Stefano V. Albrecht，以及上海交大 Shuai Li。

## 2. 研究问题

论文提出 online-to-offline RL：先通过在线 RL 或高回报模仿得到一个强 agent $\pi_0$，再用少量离线人类行为数据把它对齐到特定偏好。目标不是最大化环境奖励，而是让行为更像某类人类玩家，例如谨慎新手、速通玩家、成就收集者。

直接 fine-tuning 会失败有两个原因。第一，离线人类数据太少，论文主实验中每种偏好只有 10 条轨迹。第二，若直接用 reward model 替代环境奖励，会产生 reward distribution shift，使 agent 忘掉原有环境能力，即 unlearning。

设定上，论文把人类偏好编码为离线行为轨迹，而不是显式 pairwise 标注；通过从人类轨迹和在线 agent 轨迹中采样子轨迹对，训练 transformer-structured reward model。

## 3. 背景知识

**Preference-based RL/RLHF**：标准 RL 需要手写奖励；偏好学习用“轨迹 A 比轨迹 B 更好”训练奖励模型。LLM RLHF 中常用 Bradley-Terry loss，让 preferred response 的 reward 高于 dispreferred response。

**Agent alignment 和 LLM alignment 的差异**：LLM 对齐通常从大模型分布中采样回答，在线 PPO 主要改变文本偏好；Game AI agent 已经在环境奖励上很强，直接改奖励可能破坏已有能力。因此本文强调“增量对齐”。

**Reward calibration**：如果在线 agent 已经会做某些人类偏好行为，那么 reward model 再给高分并不能提供新信息。校准项用当前行为和 $\pi_0$ 在同一状态序列上的行为作差，只奖励“相对初始 agent 新学到的人类偏好”。

**Reward curriculum**：训练初期继续依赖环境奖励，随后逐步切换到校准后偏好奖励：

$$
R_{\mathrm{cur}}(t)=(1-\alpha(t))r_{\mathrm{env}}+\alpha(t)R_{\mathrm{cal}}.
$$

这样做的直觉是先稳住原有能力，再逐步引导风格变化。

| 术语 | 直白含义 | 在本文中的作用 |
|---|---|---|
| Online-to-offline RL | 已有在线强 agent，再用离线偏好数据对齐 | 新问题设定 |
| Human preference data | 少量人类/代理人风格轨迹 | 对齐信号来源 |
| Transformer RM | 处理子轨迹序列的奖励模型 | 捕捉时序风格 |
| Calibration | 减去初始 agent 的偏好得分 | 只学习增量偏好 |
| Curriculum | 从环境奖励平滑过渡到偏好奖励 | 防止 unlearning |

## 4. 问题分析

Figure 1 的滑行游戏展示了核心现象：环境奖励训练出的 agent 往往追求最高分/最短路径，但人类玩家可能重视避坑、速通或收集物。也就是说，“高回报”不等于“符合人类风格”。

论文把失败模式拆为两类。第一，少量离线轨迹无法直接支持 BC 或 SAC/DQN fine-tuning，Table 1 中多项 baseline 与人类回报差距很大。第二，直接用 reward model fine-tune 可能导致 agent unlearning，尤其在 reward 从环境目标骤然切到偏好目标时。

这解释了为什么 ALIGN-GAP 需要两个组件：calibration 解决“学什么”的问题，curriculum 解决“怎样平滑切换”的问题。

## 5. 思想与方法

ALIGN-GAP 的核心思想是：不要让强 agent 从头模仿少量人类轨迹，而是在已有能力上学习风格残差。

方法包含两阶段。第一阶段预训练 online agent $\pi_0$，并收集 $\pi_0$ 轨迹以及少量人类偏好轨迹。第二阶段训练 reward model，再用校准和课程奖励进行对齐训练。

Reward model 使用子轨迹而不是单步 $(s,a)$，因为人类偏好通常体现为行为模式，例如“接近坑前减速”或“绕路收集星星”。Calibration 的机制是：

$$
R_{\mathrm{cal}}(\tau)=R_\phi(\tau)-R_\phi(\tau_{\pi_0}),
$$

其中 $\tau_{\pi_0}$ 是在同一状态上下文中用初始策略动作替换得到的参考轨迹。若当前行为只是 $\pi_0$ 本来就会的，差值接近 0；若包含人类偏好中新颖的行为，差值为正。

## 6. 算法与伪代码

算法名称：ALIGNment of Game AI to Preferences，简称 ALIGN-GAP。

1. 用 SAC/DQN 等在线 RL 训练强基线 agent $\pi_0$。
2. 为每类目标偏好收集少量离线人类轨迹；D4RL 实验中用 human proxy，Atari 中用 replay dataset 聚类得到风格。
3. 从人类轨迹和 $\pi_0$ 轨迹中采样子轨迹对，构造 preference/reward model 训练数据。
4. 用 Bradley-Terry 风格目标训练 transformer reward model $R_\phi$。
5. 初始化待对齐策略 $\pi \leftarrow \pi_0$。
6. 每个训练 step 中，计算当前轨迹的 $R_\phi(\tau)$ 和参考轨迹 $R_\phi(\tau_{\pi_0})$，得到 $R_{\mathrm{cal}}$。
7. 用 $R_{\mathrm{cur}}=(1-\alpha)r_{\mathrm{env}}+\alpha R_{\mathrm{cal}}$ 更新策略。
8. 让 $\alpha(t)$ 随训练进度从 0 增至 1；论文比较了线性和指数 schedule。

训练和推理区别：训练阶段需要 RM、校准参考和课程奖励；推理阶段只部署对齐后的策略。

## 7. 实验与消融

环境包括 D4RL locomotion 的 HalfCheetah、Walker2d、Hopper，以及 Atari Pac-Man、Space Invaders、Alien 等。D4RL 中作者通过修改 reward 权重构造多种 human proxy preference，每类偏好只收集 10 条轨迹；Atari 中用 Google Atari Replay Dataset，经 UMAP 无监督聚类得到不同风格。

基线包括 Finetune w/ SAC/DQN、Finetune w/ BC、SACfD/DQNfD、BC、Finetune w/ RM，以及附录中的 RLPD 对比。Table 1 报告与 human return 的差距，越接近 0 越好；Table 2 报告 oracle reward 下的对齐排名。ALIGN-GAP 在 21 个任务中的 17 个上优于 fine-tuning baseline，并在多数 locomotion 偏好上排名第一或接近第一。

消融显示 curriculum 的影响通常比 calibration 更大，但二者都重要。Table 3 显示线性和指数 $\alpha(t)$ 表现接近，说明方法不特别依赖 schedule 形式。Table 7 显示 calibration/curriculum 的计算开销很小，ALIGN-GAP 约 42.61 iter/s，去掉 calibration 约 44.59 iter/s。

实验的弱点是“人类”偏好主要来自 human proxy 或 replay 聚类，真实 human-in-the-loop 噪声、非理性偏好和多用户冲突还没有被充分验证。

## 8. 展望

启发：第一，online-to-offline 是一个值得独立研究的迁移方向。第二，reward calibration 可以看作行为残差学习，适用于个性化 agent。第三，curriculum 是防止 reward switching 造成 unlearning 的简单强基线。第四，偏好往往是时序模式，子轨迹 RM 比单步 reward 更自然。

局限：依赖可访问的强初始 agent 和其轨迹；实验偏好主要由 proxy 构造；未处理多用户偏好冲突；最终阶段环境奖励可能被偏好奖励压过；Game AI 外的机器人/医疗等高风险场景还未验证。

后续方向：1. 用真实人类交互数据替代 human proxy，测试噪声鲁棒性。2. 将 calibration 用于 offline-to-online fine-tuning，缓解在线阶段遗忘。3. 研究多偏好/多用户条件化 ALIGN-GAP，让一个 agent 可按用户画像切换风格。

## Links

- Paper page: https://openreview.net/forum?id=ruv3HdK6he
- ICLR poster: https://iclr.cc/virtual/2025/poster/28165
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/8ce291bd0bcbe06da618ac2f630dbaa3-Abstract-Conference.html
- First author OpenReview: https://openreview.net/profile?id=~Xu_Liu12
- Code: 未能从论文/OpenReview/公开搜索确认官方代码仓库。

