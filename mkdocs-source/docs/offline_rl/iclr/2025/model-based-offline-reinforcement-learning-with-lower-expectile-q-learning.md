---
title: "Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning"
description: "LEQ 用 lower expectile regression 学习模型 rollout 的保守 λ-return，从而让 model-based offline RL 能在 AntMaze 等长时域任务上摆脱不可靠不确定性惩罚。"
tags:
  - Offline RL
  - Model-based RL
  - Expectile Regression
  - Lambda Return
  - AntMaze
---

# Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning

<div class="paper-hero">
<p class="paper-meta">Kwanyoung Park, Youngwoon Lee · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Model-based RL</span><span>Expectile Regression</span><span>Lambda Return</span><span>AntMaze</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning |
| Year | 2025；arXiv 首版 2024-06-30 |
| Source | ICLR 2025 Poster / Published as a conference paper |
| Authors | Kwanyoung Park, Youngwoon Lee |
| Affiliations | Yonsei University |
| Tags | Model-based Offline RL; Lower Expectile; Conservative Q-learning; λ-return; D4RL AntMaze; Visual Offline RL |
| Local PDF | `[ICLR_2025] Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning.pdf` |

**一句话概括**：LEQ 用 $\tau<0.5$ 的 lower expectile regression 对 world model 生成的多步 $\lambda$-return 做保守估计，避免手工不确定性惩罚和海量 rollout 分布估计，使 model-based offline RL 首次在 AntMaze 长时域稀疏奖励任务上稳定接近或超过强 model-free baseline。

**我对这篇论文的定位**：这篇文章是 model-based offline RL 的一次“去复杂化”尝试。MOPO/MOBILE/COMBO/RAMBO 等方法通常依赖模型或价值不确定性惩罚，CBOP 依赖大量 rollout 显式估计 Q 分布；LEQ 则把保守性压缩到一个简单的非对称平方损失里。它和 IQL 都用 expectile，但方向相反：IQL 用 upper expectile 近似 $\max_a Q(s,a)$，LEQ 用 lower expectile 从有噪模型 rollout 中学低偏差、保守的 return。

## 1. 第一作者相关信息

**Kwanyoung Park** 是 Yonsei University 研究者，论文和项目页均显示其与 Youngwoon Lee 合作完成本文。Youngwoon Lee 是 Yonsei University 助理教授，研究方向包括机器人学习、模型强化学习、模仿学习和泛化策略学习。

Kwanyoung Park 的公开研究轨迹集中在 model-based RL、offline RL 与长时域任务。他的 LEQ 项目页强调 model-based offline RL 在 AntMaze 等长时域任务上失败的核心原因是 model rollout 的 value estimation bias，而 LEQ 的解决方案是 lower expectile 与 λ-return 的结合。

| 年份 | 标题 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2024 | Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning | arXiv:2407.00699 | Model-based Offline RL, AntMaze | 本文预印本版本，提出 LEQ 思想 |
| 2025 | Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning | ICLR 2025 | Lower Expectile, λ-return | 本文正式会议版本 |

主要研究方向可概括为：**让基于模型的离线 RL 在长时域、稀疏奖励和视觉输入任务中稳定利用模型生成数据，而不是被模型误差拖垮**。来源包括论文 PDF、项目页、arXiv 与官方代码仓库。

## 2. 研究问题

本文解决的具体问题是：**model-based offline RL 虽然能用 learned world model 生成想象轨迹补充数据，但 model rollout 的状态、奖励和 bootstrapped Q 值都可能有误，导致 Q-target 过估计；现有不确定性惩罚往往不准且会让策略优化偏离真实价值**。

标准 model-based offline RL 流程是：

1. 用离线数据训练 world model $p_\psi(s_{t+1},r_t\mid s_t,a_t)$。
2. 从离线状态或模型状态出发生成短 rollout。
3. 将模型数据与真实离线数据混合，用 offline RL 更新策略和 Q。

问题在长时域 AntMaze 中特别严重：稀疏奖励使邻近状态 Q 值接近，错误不确定性惩罚会把好路径打坏；多步 rollout 又会累积模型误差。LEQ 的假设设置是固定 offline dataset、无额外在线交互、单智能体连续控制，使用 ensemble world model 生成 imagined trajectories，并训练确定性策略。

## 3. 背景知识

**Model-based offline RL** 的吸引力在于用模型生成额外经验，突破离线数据支持不足。但它的风险是模型在 OOD 状态动作上不准，策略可能 exploitation learned model。MOPO 用模型不确定性惩罚奖励，MOBILE 用 Bellman uncertainty 惩罚 Q，RAMBO 学对抗模型，CBOP 用大量 rollout 显式构建 Q 分布。

**Expectile regression** 是期望的非对称平方损失推广：

$$
L_2^\tau(u)=|\tau-\mathbf{1}(u>0)|u^2.
$$

当 $\tau=0.5$ 时就是普通均值；当 $\tau<0.5$ 时，小于预测值的目标被更高权重惩罚，最优解会落在分布均值以下，形成 lower expectile。和 quantile regression 相比，它仍是平方损失，更适合神经网络平滑训练。

**λ-return** 在一阶 TD 和完整 Monte Carlo 之间折中。LEQ 在 H 步模型轨迹 $T$ 上定义：

$$
Q_t^\lambda(T)=\frac{1-\lambda}{1-\lambda^{H-t-1}}\sum_{i=1}^{H-t}\lambda^{i-1}G_{t:t+i}(T).
$$

直觉：多步 return 减少单步 bootstrapping 的偏差，$\lambda$ 控制对长短 horizon 的权重。LEQ 同时将 λ-return 用于 critic 和 policy learning。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| World model | 学到的环境转移与奖励模型 | 生成 imaginary rollouts |
| Model rollout | 模型生成的状态-动作-奖励序列 | critic/policy 的主要训练信号之一 |
| Lower expectile | 比均值更保守的平方损失统计量 | 替代不确定性惩罚 |
| λ-return | 多步 return 的指数加权平均 | 降低 Q-target 偏差 |
| Dataset expansion | 用模型从离线状态扩充状态池 | 改善训练覆盖和稳定性 |
| FQE pretraining | 用离线数据预训练 Q | 给 LEQ 初始 critic 提供稳定基础 |

本文动机成立，是因为 model-based offline RL 的失败不一定需要“更准的不确定性估计”来解决；如果目标分布本身很吵，可以直接学习它的保守统计量。

## 4. 问题分析

论文的诊断从 Section 4.1 开始。对模型生成数据，1-step target 为：

$$
\hat{y}_{model}=
\mathbb{E}_{\psi}\mathbb{E}_{(s',r)\sim p_\psi(\cdot\mid s,a)}
[r+\gamma Q_\phi(s',\pi_\theta(s'))].
$$

它有三类误差：预测状态 $s'$、预测奖励 $r$、以及 $Q_\phi(s',\pi_\theta(s'))$ 的外推误差。真实离线 transition 的 target

$$
\hat{y}_{env}=r+\gamma Q_\phi(s',\pi_\theta(s'))
$$

没有模型预测误差，因此论文主张：**模型数据上的 Q 学习要保守，真实数据上的 Q 学习可用标准 Bellman loss**。

Figure 1 直观展示了 LEQ：从少量 model rollouts 得到 return samples，不显式估计完整 Q 分布，而用 lower expectile 学一个保守 return estimate。Figure 2 将普通 Q-learning 与 LEQ 对比，LEQ 只是给 Q loss 和 policy gradient 乘上非对称权重：

$$
|\tau-\mathbf{1}(Q_t^\lambda(T)>Q_\phi(s_t,a_t))|.
$$

这揭示 prior methods 的局限：惩罚项常依赖启发式 uncertainty，策略最大化的是“被惩罚的 Q”，而 LEQ 学的是目标 return 分布的保守统计量，更贴近“真实值的低估计”。

## 5. 思想与方法

LEQ 的指导原则是：**对于不可靠模型 rollout，不要估计不确定性后再惩罚；直接让 Q 和 policy 学 return 分布的 lower expectile**。

方法包含四个关键设计。

**Lower expectile critic on model data**：

$$
L_{Q,model}(\phi)=
\mathbb{E}_{s_0\in D_{model},T\sim p_\psi,\pi_\theta}
\left[\sum_{t=0}^{H-1}L_2^\tau(Q_\phi(s_t,\pi_\theta(s_t))-Q_t^\lambda(T))\right].
$$

当模型 rollout 给出过高 return 时，$\tau<0.5$ 会降低其权重；当 return 较低时，损失权重更高，critic 被拉向保守值。

**Standard Bellman on real data**：真实离线 transition 使用普通平方 Bellman loss，避免对可靠数据过度保守。总 critic loss 是

$$
L_Q(\phi)=\beta L_{Q,model}(\phi)+(1-\beta)L_{Q,env}(\phi)+\omega_{EMA}L_{Q,EMA}(\phi).
$$

**Lower expectile policy learning**：策略不直接最大化即时 $Q_\phi(s,a)$，而最大化模型 rollout 的 lower-expectile λ-return。由于 expectile 不易直接求梯度，论文推导了 Eq. 11 的 surrogate loss，用 learned Q 近似 expectile 位置。

**Dataset expansion**：用当前策略和 world model 从离线状态出发扩充 $D_{model}$，并在生成时加入探索噪声，缓解状态覆盖过窄。

新意在于：lower expectile 同时用于 critic 和 actor，λ-return 同时用于价值估计和策略优化，真实 transition 仍保留标准 Bellman 更新。这三个点构成了 LEQ 在 AntMaze 上有效的核心组合。

## 6. 算法与伪代码

**算法名**：LEQ: Lower Expectile Q-learning with λ-returns。

Algorithm 1 可重写为：

1. 输入离线数据 $D_{env}$、expectile 参数 $\tau\le 0.5$、imagination length $H$、dataset expansion length $R$。
2. 初始化 world model ensemble $\{p_{\psi_i}\}_{i=1}^M$、策略 $\pi_\theta$ 和 Q 函数 $Q_\phi$。
3. 在 $D_{env}$ 上训练 world model，最大化 $\log p_\psi(s',r\mid s,a)$。
4. 用 BC 预训练策略，用 FQE 预训练 Q。
5. 循环直到收敛：
6. 从 $D_{env}$ 采样初始状态，用当前策略和随机选取的 world model rollout $R$ 步，扩展 $D_{model}$。
7. 从 $D_{model}$ 采样状态，rollout $H$ 步得到 imagined trajectory $T$。
8. 对模型轨迹计算 $Q_t^\lambda(T)$。
9. 用模型轨迹上的 lower expectile loss、真实数据上的 Bellman loss、EMA regularization 更新 critic。
10. 用 lower expectile λ-return 的 surrogate loss 更新 actor。

关键超参数：$\tau$ 控制保守程度，$H$ 控制模型想象长度，$R$ 控制数据扩展长度，$\beta$ 控制模型数据与真实数据 loss 比例。论文 Appendix Table 6/9 给出 task-specific $\tau$；主文强调 AntMaze 中使用真实 transition 训练 critic 对获得非零分数很关键。

## 7. 实验与消融

论文评测 D4RL AntMaze、D4RL MuJoCo Gym、NeoRL 和 V-D4RL。指标包括 AntMaze success rate 与 locomotion normalized score。

**AntMaze**：Figures 5 与 Table 11 显示 LEQ 在 umaze、medium、large、ultra 共 8 个任务上显著优于 prior model-based methods。主文指出 LEQ 是首个在 AntMaze 长时域任务上超过或匹配 model-free/sequence modeling baseline 的 model-based offline RL 方法。Table 1 的 ablation 总分显示完整 LEQ 为 461.8；把 λ-return 换成 H-step 或 1-step、把 policy objective 换成 Q(s,a) 或 AWR，都会明显下降。

**NeoRL 与 D4RL MuJoCo**：Figures 6 和 8 显示 LEQ 在 dense-reward locomotion 上与 SOTA model-based/model-free 方法相当，不只适用于稀疏 AntMaze。论文报告在 D4RL MuJoCo 12 个任务中有 6 个达到或接近 prior best。

**V-D4RL**：Figure 9 显示 LEQ 与 DreamerV3 结合后在像素输入任务上也能达到 SOTA 附近，说明方法不依赖低维 state-only 输入。

**关键消融**：

1. Table 1：lower expectile、λ-return、policy 使用 λ-return 三者都重要。
2. Table 2：单纯调 MOBILE 的惩罚超参数不能解释 LEQ 的 AntMaze 优势。
3. Table 3：更长 imagination length $H$ 往往有帮助，但过长会增加模型误差风险。
4. Appendix Table 18：移除 Q-learning loss、移除 policy expectile、或对真实 transition 也用 expectile，都会明显降低 AntMaze 总分。
5. Appendix Table 19：BC/FQE pretraining 对稳定训练重要。

实验能证明 LEQ 在多类 benchmark 上是强而简单的 model-based offline RL baseline。弱点是 AntMaze 个别 medium maze 失败与高训练方差仍存在，论文 Figure 7/10 也承认邻近状态 Q 值过近会造成失败和波动。

## 8. 展望

对 ORL 研究者的启发：

1. 保守性不一定必须来自 explicit uncertainty penalty，统计量选择本身也能带来保守估计。
2. Model-generated data 和 real offline data 不应被同等对待，LEQ 的分离 loss 是一个很实用的设计。
3. 长时域 model-based offline RL 的关键不是只缩短 rollout，而是降低多步 value target 的偏差。
4. Expectile 在 offline RL 中的方向很重要：upper expectile 服务于动作最大化，lower expectile 服务于模型数据保守化。

局限与开放问题：

1. $\tau$ 仍是任务相关超参数，保守程度没有自适应校准。
2. Lower expectile 主要处理过估计，对系统性模型偏差和 reward model 偏差不一定充分。
3. 多步 model rollout 在极端 OOD 状态仍可能产生伪低风险轨迹。
4. 对 stochastic policy、离线到在线微调和真实机器人任务的探索还有限。

可能的后续研究：

1. **Adaptive lower expectile**：用 ensemble disagreement 或 density ratio 动态调节 $\tau(s,a)$。
2. **LEQ + uncertainty gating**：只在模型可信区域使用长 λ-return，低可信区域回退到真实数据 Bellman target。
3. **Preference/visual LEQ**：将 lower expectile 用于 learned reward 或 human preference reward 下的 model-based offline RL。

## Links

- Project page: https://kwanyoungpark.github.io/LEQ/
- arXiv: https://arxiv.org/abs/2407.00699
- Code: https://github.com/kwanyoungpark/LEQ
- First author homepage: https://kwanyoungpark.github.io/
- Source notes: Project page confirms authors, Yonsei affiliation, arXiv/code links and high-level claims; local PDF was used for Algorithm 1, equations, Figures 1-10 and Tables 1-3/11-23.

</article>
