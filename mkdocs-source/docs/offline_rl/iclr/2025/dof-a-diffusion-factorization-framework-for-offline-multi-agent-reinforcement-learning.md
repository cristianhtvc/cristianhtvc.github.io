---
title: "DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning"
description: "DoF 将 MARL 的 IGM 原则推广为 IGD 分布一致性原则，用噪声和数据因子化把集中式扩散模型拆成可去中心化执行的小模型。"
tags:
  - Offline MARL
  - Diffusion Models
  - Value Factorization
  - CTDE
  - Multi-Agent RL
---

# DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Chao Li, Ziwei Deng, Chenxing Lin, Wenqi Chen, Yongquan Fu, Weiquan Liu, Chenglu Wen, Cheng Wang, Siqi Shen · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline MARL</span><span>Diffusion Models</span><span>Value Factorization</span><span>CTDE</span><span>Multi-Agent RL</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster |
| Authors | Chao Li, Ziwei Deng, Chenxing Lin, Wenqi Chen, Yongquan Fu, Weiquan Liu, Chenglu Wen, Cheng Wang, Siqi Shen |
| Affiliations | Xiamen University; National University of Defense Technology; Jimei University; University of Electronic Science and Technology of China，见论文首页 |
| Tags | Offline MARL; diffusion factorization; IGD; CTDE; trajectory generation |

**一句话概括：** DoF 提出 IGD 原则，把一个集中式多智能体扩散模型因子化为多个去中心化小扩散模型，使各 agent 独立生成的结果联合起来与集中式模型生成分布一致。

**我对这篇论文的定位：** 这是 offline cooperative MARL 中 diffusion-based decision making 的理论原则和框架论文。它把 MARL 中成熟的 IGM/value factorization 思想推广到生成分布层面，解决 MADIFF 的扩展性问题和 Independent Diffusion 的协作不足问题。

## 1. 第一作者相关信息

共同第一作者为 **Chao Li** 和 **Ziwei Deng**。论文首页显示二者来自 Xiamen University 相关实验室，通信作者为 Siqi Shen。OpenReview 和 ICLR slides 可核验作者、单位、代码和 ICLR 2025 Poster 信息；本次检索没有可靠确认两位第一作者的个人主页或完整发表列表。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning | ICLR 2025 Poster | Offline MARL, diffusion | 本文 |

从本文看，该团队关注 multi-agent reinforcement learning、diffusion models、value factorization 和 3D/感知相关方向。作者信息以论文首页、OpenReview 与 ICLR slides 为准。

## 2. 研究问题

论文研究 **离线协作 MARL 中如何使用扩散模型，同时保持可扩展性和协作性**。

现有两类方法各有缺陷：

- **集中式扩散模型**，如 MADIFF：能建模联合轨迹，协作较好，但输入维度随 agent 数量增加，去中心化执行时复杂度高。
- **Independent Diffusion**：每个 agent 训练自己的扩散模型，输入复杂度 $O(1)$，但不显式考虑其他 agent，协作差。

MARL 通常采用 CTDE：训练时可用全局信息，执行时每个 agent 只能用局部观测。传统 value factorization 用 IGM 原则保证局部 greedy action 的组合等于全局最优 action。DoF 的问题是：对 diffusion planner/policy，能否有类似 IGM 的原则，保证“局部生成结果组合起来”等价于“集中式生成结果”？

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Dec-POMDP | 多智能体部分可观测决策过程 | MARL 问题设定 |
| CTDE | 集中训练、去中心化执行 | DoF 的执行约束 |
| IGM | 个体 argmax 组合等于全局 argmax | 被推广的经典原则 |
| IGD | 个体生成分布乘积等于全局生成分布 | DoF 提出的原则 |
| CDM | centralized diffusion model | 训练时的联合扩散模型 |
| DDM | decentralized diffusion model | 执行时每个 agent 的小模型 |
| Noise factorization | 把联合噪声分解为个体噪声 | 满足 IGD 的关键 |
| Data factorization | 混合个体生成数据以建模关系 | 提升生成质量 |

IGM 原则可以写成：

$$
\arg\max_{\mathbf{u}}Q_{\mathrm{jt}}(\tau_{\mathrm{tot}},\mathbf{u})
=
\left(
\arg\max_{u_1}Q_1(\tau_1,u_1),\ldots,
\arg\max_{u_N}Q_N(\tau_N,u_N)
\right).
$$

DoF 要处理的不是 Q 值 argmax，而是扩散生成分布。它提出 IGD：

$$
\prod_{i=1}^{N}p_{\theta_i}(x_i^0)=p_{\theta_{\mathrm{tot}}}(x_{\mathrm{tot}}^0),
\quad \theta_i\subset\theta_{\mathrm{tot}}.
$$

含义：每个 agent 独立生成 $x_i^0$，这些样本拼起来的联合分布应与集中式模型生成的 $x_{\mathrm{tot}}^0$ 同分布。

## 4. 问题分析

论文用 Landmark Covering Game 作为动机。三个 agent 要分别覆盖不同 landmark 并避免碰撞。Independent Diffusion 容易多个 agent 去同一目标或碰撞；MADIFF 有一定协作但轨迹扭曲且扩展性差；DoF 的生成轨迹最接近人工指定目标的 ground truth。

MADIFF 的问题来自两个方面：

1. **输入复杂度**：集中式模型输入包含所有 agent 的观测/动作，复杂度随 $N$ 增长。
2. **去中心化噪声条件**：执行时每个 agent 使用同一个集中式模型，但其他 agent 条件往往用随机噪声填充，导致条件污染。

Independent Diffusion 的问题则是每个模型不知道自己对应哪个全局协作角色，也不建模联合分布，因此不满足 IGD。

DoF 的诊断是：diffusion-based MARL 缺少类似 IGM 的设计原则。只说“用扩散模型生成轨迹”不够，必须保证集中式训练后可以分解为去中心化模型，并且分布层面保持一致。

## 5. 思想与方法

DoF 的核心是两个 factorization function。

**Noise factorization function $f$**：把每个 agent 的噪声和中间生成数据组合成联合噪声/联合数据：

$$
\epsilon_{\mathrm{tot}}^k=f(\epsilon_1^k,\ldots,\epsilon_N^k),
$$

$$
\epsilon_{\theta_{\mathrm{tot}}}(x_{\mathrm{tot}}^k,k)
=f(\epsilon_{\theta_1}(x_1^k,k),\ldots,\epsilon_{\theta_N}(x_N^k,k)),
$$

$$
x_{\mathrm{tot}}^k=f(x_1^k,\ldots,x_N^k).
$$

论文主要研究 Concat 和 WConcat。Concat 直接按维度拼接；WConcat 给每个 agent 的噪声预测一个可学习权重。Theorem 1/2 证明这些函数在对角高斯噪声和独立参数分解条件下满足 IGD。

**Data factorization function $h$**：最后一步把个体生成数据 $x_i^0$ 混合成联合数据：

$$
x_{\mathrm{tot}}^0=h(x_1^0,\ldots,x_N^0).
$$

如果 $x_i^0$ 是个体 Q 值，$h$ 就类似 VDN/QMIX/QPLEX 的 value mixing network；如果 $x_i^0$ 是轨迹或动作，$h$ 建模 agent 之间的生成关系。论文测试 Concat、WConcat、Attention 等形式。

DoF 有两个具体 agent：

- **DoF-Trajectory**：基于 Decision Diffuser，生成未来 observation/trajectory，再用 inverse dynamics 得到动作。
- **DoF-Policy**：基于 DiffusionQL，直接生成动作，并结合 critic 做 policy improvement。

## 6. 算法与伪代码

DoF-Trajectory 集中训练：

1. 从离线数据 $D$ 采样联合轨迹片段 $x_{\mathrm{tot}}^0$。
2. 采样扩散步 $k$ 和全局高斯噪声。
3. 构造加噪联合数据 $x_{\mathrm{tot}}^k$。
4. 按 agent 维度切分得到 $x_i^k$。
5. 每个 agent 的噪声网络预测 $\epsilon_{\theta_i}(x_i^k,k)$。
6. 用 $f$ 合成联合噪声预测 $\epsilon_{\theta_{\mathrm{tot}}}$。
7. 最小化联合噪声预测误差；同时训练 inverse dynamics $I_\varphi(o,o')$ 预测动作。

可概括为：

$$
L(\theta,\varphi)=
\mathbb{E}\left[\|\epsilon-\epsilon_{\theta_{\mathrm{tot}}}(x^k(\tau),y(\tau),k)\|^2\right]
+
\mathbb{E}_{(s,u,s')\sim D}\left[\|u-I_\varphi(o,o')\|^2\right].
$$

去中心化执行：

1. 每个 agent 初始化自己的 $x_i^K\sim\mathcal{N}(0,I)$。
2. 对 $k=K,\ldots,1$，每个 agent 用本地模型预测噪声。
3. 按 DDPM 反向更新生成 $x_i^{k-1}$。
4. 得到个体轨迹/动作 $x_i^0$。
5. 若是 trajectory agent，用 inverse dynamics 从本地当前/下一观测恢复动作并执行。

DoF-Policy 则在 actor loss 中使用 diffusion loss + policy gradient 项：

$$
L(\theta)=
\mathbb{E}_{\epsilon,(s,u)\sim D}\left[\|\epsilon-\epsilon_{\theta_{\mathrm{tot}}}\|^2\right]
-\alpha\mathbb{E}_{s\sim D,u'\sim\pi_\theta}[Q_\Phi(s,u')].
$$

训练时用全局信息，执行时每个 agent 只用自己的 diffusion model 和局部观测。

## 7. 实验与消融

论文实验覆盖四类问题：生成分布是否匹配 ground truth、IGD 是否重要、offline MARL 策略性能、扩展性。

**Illustrative examples**：

- Figure 3 的二维混合高斯生成任务中，DoF 在视觉分布和四象限概率上最接近 ground truth，MADIFF 次之，Independent Diffusion 最差。
- Figure 4 的 landmark covering 中，DoF 取得更高回报和更少碰撞；policy generation 任务中 DoF 也明显优于 ID。
- Table 1 的 payoff matrix game 显示 DoF 能重构 joint Q matrix，包括最优策略，说明 $h$ 可作为 value factorization 类函数。

**SMAC / SMACv2 / MPE / MA-MuJoCo**：

- Table 2：SMAC 多个地图和 Good/Medium/Poor 数据质量下，DoF 大多最好。例如 3m-good 为 19.8，3m-medium 为 18.6，2c_vs_64zg-good 为 16.1。
- MADIFF 在一些同质任务表现不错，但在 heterogeneous 环境如 3s5z_vs_3s6z 上弱；DoF 通过 factorization 更稳。
- Appendix 中 SMACv2、MPE 和 MA-MuJoCo 也显示 DoF 通常最好或接近最好；MA-MuJoCo HalfCheetah 中 DoF 在 Medium 最好，Good/Poor 第二。

**扩展性**：

- Table 3：在自定义 MAgent 环境中，agent 数从 4 到 64 增长。DoF 在 64 agents 下仍可运行，GPU memory 5924 MB、inference 24.3s；MADIFF 在 64 agents OOM，且 32 agents 已需 14981 MB。

**消融**：

- Table 4：WConcat 通常优于 Concat；Dec-Atten 去中心化时由于省略其他 agent 权重，表现很差；中心化 Atten 表现好但不能去中心化。
- 将 $f$ 引入 MADIFF 得到 DoF+MADIFF 后优于原 MADIFF，说明 factorization function 本身有迁移价值。
- data factorization $h$ 的 appendix 消融显示 WConcat 优于 Concat，但略弱于 Attention。

实验解读：DoF 有力展示了 IGD 原则带来的协作与扩展性收益。需要注意的是，它的理论证明依赖对角高斯和特定 factorization 条件；复杂函数如 attention/QMIX 是否严格满足 diffusion 分布假设，需要具体分析。

## 8. 展望

对 ORL/MARL 研究者的启发：

- diffusion-based MARL 需要分布层面的 factorization principle，不能直接套用单智能体 diffusion policy。
- IGM 的思想可推广到生成分布、风险敏感策略甚至 planner，而不局限于 Q-value argmax。
- 去中心化执行中的“条件噪声污染”是集中式生成模型部署到 MARL 时的关键问题。
- WConcat 这类简单结构可能比复杂 attention 更适合严格去中心化执行。

局限与开放问题：

- IGD 证明主要覆盖 Concat/WConcat 这类结构，对更复杂 mixing 的理论条件不够完整。
- DoF-Trajectory 依赖 inverse dynamics，部分环境中 observation-to-action 恢复可能不唯一。
- SMAC/MAgent/MPE 等仍是模拟环境，真实多机器人或交通场景需要验证。
- 扩散采样本身仍有推理成本，agent 数很多时虽然优于 MADIFF，但仍可能不够实时。

可能后续方向：

1. **IGD + communication**：允许低带宽通信，研究生成分布一致性和通信成本的权衡。
2. **安全约束 DoF**：将 collision/risk 作为条件或 data factorization 目标，服务安全协作任务。
3. **大规模 heterogeneous DoF**：为不同 agent 类型设计共享-专属混合参数，提升异构 swarm 的泛化能力。

## Links

- OpenReview: https://openreview.net/forum?id=OTFKVkxSlL
- ICLR slides: https://iclr.cc/media/iclr-2025/Slides/29825.pdf
- Code: https://github.com/xmu-rl-3dv/DoF

</article>
