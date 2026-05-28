---
title: "M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model"
description: "用同一个预训练掩码轨迹模型在测试时完成动作提案、未来预测、回报评估与目标到达规划，从而把 BTM 从单步行为克隆器变成可规划的决策模型。"
tags:
  - Offline RL
  - Offline-to-Online RL
  - Model Predictive Control
  - Masked Trajectory Model
  - Robot Learning
---

# M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model

| 字段 | 内容 |
|---|---|
| Title | M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model |
| Year | 2025 |
| Source | ICLR 2025 Poster / Published as a conference paper |
| Authors | Kehan Wen, Yutong Hu, Yao Mu, Lei Ke |
| Affiliations | ETH Zurich; KU Leuven; Hong Kong University; Carnegie Mellon University. 论文脚注说明 Kehan Wen 与 Yutong Hu 为共同一作，Yao Mu 与 Lei Ke 为通讯作者，工作完成于 ETH Zurich。 |
| Tags | Offline RL; Offline-to-Online RL; Masked Autoencoding; Model-based RL; MPC; Robot Learning |
| Local PDF | `[ICLR_2025] M^3PC Test-time Model Predictive Control using Pretrained Masked Trajectory Model.pdf` |

**一句话概括**：M^3PC 不重新训练模型，而是在测试时组合 BTM 的 `[RCBC]` 动作重构、`[FD]` 前向动力学、`[RP]` 回报预测、`[PI]/[ID]` 路径与逆动力学能力，用 MPC 式动作筛选提升离线、离线到在线和目标到达任务表现。

**我对这篇论文的定位**：这篇文章站在序列建模式 Offline RL 与 model-based planning 的交界处。它的关键不是提出新的 Transformer 预训练目标，而是指出已经预训练好的 Bidirectional Trajectory Model 其实同时包含“策略模型”和“世界模型”的功能；旧方法只用单一掩码做行为克隆，M^3PC 则把多种掩码模式编排成测试时规划流程。论文的工程味道很强：贡献主要体现在推理时的决策组织方式和 O2O 探索策略，而不是新的 Bellman 理论。

## 1. 第一作者相关信息

本文有两位共同一作：**Kehan Wen** 与 **Yutong Hu**。论文首页给出的单位是 Kehan Wen 在 ETH Zurich，Yutong Hu 同时标注 ETH Zurich 与 KU Leuven；OpenReview 与 ICLR 页面也确认作者顺序和 ICLR 2025 Poster 状态。公开检索中 Kehan Wen 的独立主页/完整 publication list 不如资深作者充分，因此这里采用“可核验来源优先”的保守写法。

Kehan Wen 的研究轨迹可从本文推断为：围绕离线强化学习、轨迹序列建模、测试时规划与机器人决策展开。M^3PC 的一个明显特点是把 NLP/CV 中的 masked modeling 思想转译为 RL 轨迹模态之间的条件补全，再进一步把补全能力用于 MPC 规划。Yutong Hu 的共同一作身份也很重要，因为本文的技术贡献不是单一模块，而是模型预训练、推理掩码设计、D4RL/RoboMimic 实验和 O2O 流程的组合。

| 年份 | 标题 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model | ICLR 2025 Poster | Masked Trajectory Model, MPC, Offline RL | 本文，提出用多掩码测试时 MPC 激活 BTM 的规划能力 |

主要研究方向可以概括为：**用统一轨迹模型学习多种 MDP 模态关系，并在部署阶段通过规划而非单步模仿来释放这些关系**。来源：论文 PDF 首页、OpenReview 页面、ICLR 2025 virtual poster、官方代码仓库。

## 2. 研究问题

本文解决的具体问题是：**预训练的掩码轨迹模型在离线 RL 部署时通常只被当作 return-conditioned behavior cloning 使用，导致模型已经学到的未来状态、奖励、回报和逆动力学信息没有参与决策**。

设轨迹为

$$
\tau=(s_1,g_1,a_1,r_1,\ldots,s_T,g_T,a_T,r_T),
$$

其中 $s_t$ 是状态，$a_t$ 是动作，$r_t$ 是奖励，$g_t$ 是 return-to-go 或长期价值提示。BTM 预训练时随机掩码轨迹 token，学习从未掩码部分重构被掩码部分。现有推理范式通常只使用 `[RCBC]`：给定历史状态和目标回报，预测当前动作。这会带来两个失败模式：

1. **没有显式前瞻**：动作由行为克隆式映射生成，缺少“候选动作会导致什么未来”的评估。
2. **数据上限明显**：单纯模仿数据中相似 RTG 的轨迹，很难超越离线数据中的行为质量，也不擅长 OOD 目标到达。

论文假设的设置是标准单智能体 MDP、固定离线数据集，主要评测 D4RL locomotion、RoboMimic manipulation、离线到在线微调和 goal-reaching。它不是纯 model-based offline RL 中“另训 world model + policy”的路线，而是把一个预训练 BTM 当作统一的 policy/world/reward/path 模型。

## 3. 背景知识

**序列建模式 Offline RL** 把强化学习轨迹看作 token 序列。Decision Transformer 用自回归 Transformer 学习 $P(a_t \mid s_{\le t}, \hat{R}_{\le t})$；Trajectory Transformer 用轨迹 likelihood 和 beam search 做规划；BTM/Uni[Mask] 则用 BERT/MAE 风格的随机掩码预训练，使模型能从任意上下文补全轨迹中的任意模态。

**BTM 的多种掩码能力** 是理解本文的核心。`[RCBC]` 给定状态和回报预测动作，相当于策略；`[FD]` 给定状态动作预测未来状态，相当于动力学；`[RP]` 预测奖励或长期回报，相当于评价器；`[PI]` 给定起点和终点补全中间状态；`[ID]` 给定相邻状态预测动作。M^3PC 的洞察是：这些能力不必分别作为下游任务使用，可以在同一个下游决策步骤中协同。

**MPC** 的基本思想是滚动规划：在当前状态生成多个候选动作序列，预测未来，评分，执行第一步，然后下个时间步重新规划。传统 MPC 需要显式动力学模型；M^3PC 的动力学和评分都来自同一个 masked trajectory model。

**Offline-to-Online RL** 关心离线预训练后的在线微调。离线 RL 常因保守性导致在线探索弱；序列模型又常因监督学习范式导致探索效率低。M^3PC 通过“按效用采样候选动作”把规划评分转化为探索分布。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| BTM | 双向掩码轨迹模型 | 统一承载策略、世界模型、奖励预测和逆动力学 |
| `[RCBC]` | Return-conditioned behavior cloning mask | 生成候选动作序列 |
| `[FD]` | Forward dynamics mask | 预测候选动作导致的未来状态 |
| `[RP]` | Reward/return prediction mask | 给候选轨迹打分 |
| Forward M^3PC | 面向回报最大化的测试时 MPC | 用于 D4RL offline RL 与 O2O |
| Backward M^3PC | 从目标状态反推路径和动作 | 用于 zero-shot goal reaching |
| TD($\lambda$) utility | 短期奖励和长期价值的加权评分 | 平衡 model rollout 误差和长期价值估计 |

这篇文章的动机成立，是因为 masked modeling 的训练目标本来就强迫模型理解不同轨迹模态之间的条件关系；如果部署时只用一个动作补全 mask，就像训练了一个多工具系统却只拿其中一个工具做所有事情。

## 4. 问题分析

论文的诊断路径集中在 Figures 1-3 和 Section 4。Figure 1 对比了传统 DT/BTM 只根据过去上下文生成动作，M^3PC 则用模型预测未来状态、奖励和回报来增强测试时决策。Section 2 进一步指出，Trajectory Transformer 的 beam search 随 horizon 增长有明显计算成本，而双向 BTM 可以并行补全未来 token，因此更适合做固定步数的快速 MPC。

核心变量有三个：

1. **候选动作数 $N$**：决定测试时搜索宽度。
2. **规划 horizon $T-t$**：决定前瞻长度，但 BTM 的并行补全降低了 horizon 上的线性成本。
3. **评分温度 $\xi$ 与 TD($\lambda$) 参数 $\lambda$**：前者控制动作选择分布的尖锐程度，后者平衡短期奖励与长期回报估计。

Forward M^3PC 的效用函数使用类似 TD($\lambda$) 的加权回报：

$$
U=(1-\lambda)\sum_{n=0}^{T-t-1}\lambda^n G_{t:t+n}+\lambda^{T-t}G_{t:T},
$$

其中

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n g_{t+n}.
$$

含义：$r_{t+k}$ 来自短期奖励预测，$g_{t+n}$ 来自长期 RTG/Q 估计；$\lambda$ 越大越依赖长期估计，越小越依赖短期 rollout。论文还提出 M^3PC-Q，在数据策略差异较大、MC RTG 噪声高时，用 IQL 风格训练出的 transition-wise Q 作为长期启发。

这套分析揭示了 prior BTM 的盲点：问题不是 BTM 不会预测未来，而是推理流程没有让“预测未来”参与动作选择。

## 5. 思想与方法

M^3PC 的指导原则很直接：**不要把 BTM 只当策略网络，而要把它当一个可被掩码查询的通用 MDP 模型；测试时通过多次查询把动作选择变成规划问题**。

方法由两条路径组成。

**Forward M^3PC** 用于回报最大化。它先用 `[RCBC]` 生成不确定性感知的候选动作分布，再从中采样 $N$ 条候选动作序列；对每条候选用 `[FD]` 预测未来状态，用 `[RP]` 预测奖励和回报；最后用效用 $U_i$ 转成 softmax 选择分布：

$$
P[i]=\frac{\exp(\xi U_i)}{\sum_j \exp(\xi U_j)}.
$$

离线推理时返回 $P$ 下动作的期望，在线微调时按 $P$ 采样以保留探索。这个设计把“高效用动作更可能执行”和“仍保留随机性”结合起来。

**Backward M^3PC** 用于目标到达。给定当前状态和目标状态，先用 `[PI]` 补全中间状态路径，再用 `[ID]` 对相邻状态反推动作。它不是枚举大量候选动作再 roll out，而是利用 BTM 的双向条件能力直接做路径 inpainting。

真正新的部分在于**测试时多 mask 编排**。BTM、MAE、MPC、TD($\lambda$)、IQL Q 估计都不是本文发明的，但把这些模块组织为“同一个预训练模型自我增强的推理流程”，是本文的主要贡献。证据层面，论文主要依靠实验和消融：Table 1/2/3、Figures 5-8 显示性能提升，Figure 6/7 支持统一预训练和不确定性感知动作重构的重要性。

## 6. 算法与伪代码

**算法名**：Forward M^3PC for Reward Maximization；扩展包括 M^3PC-M、M^3PC-Q、Backward M^3PC。

Forward M^3PC 可重写为：

1. 输入当前状态 $s_t$、历史轨迹 $\tau_{<t}$、折扣 $\gamma$、衰减参数 $\lambda$、候选数 $N$、softmax 温度 $\xi$。
2. 用 `[RCBC]` mask 预测从 $t$ 到 $T$ 的动作分布 $\alpha_{t:T}$，这里输出均值和方差以支持不确定性感知采样。
3. 重复 $N$ 次：从 $\alpha_{t:T}$ 采样候选动作序列 $a^i_{t:T}$。
4. 对每条候选，用 `[FD]` mask 预测未来状态 $s^i_{t+1:T}$。
5. 用 `[RP]` mask 预测中间奖励 $r^i_{t:T}$ 与长期回报/价值 $g^i_{t:T}$。
6. 计算 TD($\lambda$) 风格效用 $U_i$。
7. 用 softmax 将效用转成候选选择分布 $P$。
8. 离线评估时执行 $\mathbb{E}_{i\sim P}[a^i_t]$；在线微调时从 $P$ 中采样动作。

Backward M^3PC 可重写为：

1. 输入当前状态 $s_t$ 和目标状态 $s_T^\star$。
2. 用 `[PI]` mask 预测中间路径 $s_{t+1},\ldots,s_{T-1}$。
3. 用 `[ID]` mask 从 $(s_t,s_{t+1})$ 推断当前动作 $a_t$。
4. 执行动作后滚动重规划。

实现上最关键的超参数包括候选数 $N$、规划 horizon、$\lambda$、$\xi$、不确定性动作重构的熵约束。Appendix Table 4 给出了离线/在线超参数，Appendix Figure 12 显示 $\lambda$ 选择并不极端敏感。

## 7. 实验与消融

实验回答四个问题：Forward M^3PC 是否提升离线和 O2O 回报；Backward M^3PC 是否能做目标到达；哪些组件重要；方法能否迁移到操作任务。

**D4RL offline RL**：Table 1 使用 Hopper、Walker2d、HalfCheetah 的 medium 与 medium-replay 数据。M^3PC-M 与原 BTM 共享权重，但总分从 BTM 的 372.8 提升到 395.9；M^3PC-Q 总分 429.8，超过 BC、TD3+BC、IQL、DT、TT 和 BTM。这说明提升来自测试时规划，而非额外训练。

**Offline-to-Online**：Table 2 给出 200K online samples 后的结果。M^3PC 总分从 429.8 到 530.8，提升 +101.0；ODT 从 377.4 到 422.7，提升 +45.3；IQL 几乎不提升。Figure 4 显示 M^3PC 探索比高斯噪声更容易采集高回报轨迹，同时保留一定随机性。

**Goal Reaching**：Figure 5 展示 HalfCheetah flipping、Walker splits、Hopper wiggling 等目标，这些行为不直接由离线奖励定义。单一 goal-reaching mask 不能稳定完成目标，而 Backward M^3PC 能通过路径补全和逆动力学执行。

**RoboMimic manipulation**：Table 3 在 Can、Square、Lift 与真实 Can-Real 上评估。M^3PC 在 Can-Pair 达 0.98，在 Can-Real 达 0.70，高于 DT 的 0.50；Square-MH 和 Lift-MG 并非全面领先，说明复杂操作场景仍受数据质量和模型预测限制。

**消融**：Figure 6 表明统一 BTM 优于拆成专门 policy/world model；Figure 7 表明不确定性感知动作重构和 planning-based resampling 都对在线微调稳定性重要。论文的弱点是大部分证据仍是模拟或有限真实任务，且测试时 MPC 的计算成本虽优于 beam search，但仍随候选数 $N$ 增加。

## 8. 展望

对 ORL 研究者有三个启发。第一，序列模型的价值不只在“把 RL 做成监督学习”，还在于它可以作为多查询接口。第二，测试时计算可以成为离线模型能力释放的重要维度。第三，O2O 探索可以从“给动作加噪声”转向“按模型预测效用采样”。

局限也很清楚。BTM 的未来预测仍来自离线数据分布，极端 OOD 规划可能产生幻觉；Backward M^3PC 需要目标状态可表达且可被路径 inpainting 连接；M^3PC-Q 依赖额外 Q 估计器，部分削弱“单模型”叙事；真实机器人实验规模还不够大。

可追问的后续方向：

1. **不确定性校准的 M^3PC**：为 `[FD]/[RP]` 预测加入校准误差或 ensemble disagreement，降低测试时规划对错误未来的过度信任。
2. **长 horizon 分层 M^3PC**：把 Backward M^3PC 的子目标路径与 Forward M^3PC 的局部动作选择结合，处理更长任务。
3. **语言/视觉目标条件化**：用 VLM 或 goal encoder 把自然语言目标转成状态约束，使 M^3PC 更接近通用机器人策略。

## Links

- Paper page: https://openreview.net/forum?id=inOwd7hZC1
- ICLR virtual poster: https://iclr.cc/virtual/2025/poster/28673
- PDF: https://openreview.net/pdf?id=inOwd7hZC1
- Code: https://github.com/wkh923/m3pc
- Source notes: OpenReview confirms ICLR 2025 Poster status, author list, keywords and code link; the local PDF was used for equations, algorithms, figures and tables.
