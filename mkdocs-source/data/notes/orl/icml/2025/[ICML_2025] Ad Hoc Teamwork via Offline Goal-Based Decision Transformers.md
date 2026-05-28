---
title: "Ad Hoc Teamwork via Offline Goal-Based Decision Transformers"
description: "TAGET 将 ad hoc teamwork 推到离线多智能体设定，用队友感知的 RTG 与子目标预测增强 Decision Transformer。"
tags:
  - Offline RL
  - Multi-Agent RL
  - Ad Hoc Teamwork
  - Decision Transformer
---

# Ad Hoc Teamwork via Offline Goal-Based Decision Transformers

| 字段 | 内容 |
|---|---|
| Title | Ad Hoc Teamwork via Offline Goal-Based Decision Transformers |
| Year | 2025 |
| Source | ICML 2025, PMLR 267:74602-74616；OpenReview 显示 Published: 01 May 2025，Last Modified: 23 Jul 2025 |
| Authors | Xinzhi Zhang, Hohei Chan, Deheng Ye, Yi Cai, Mengchen Zhao |
| Affiliations | South China University of Technology；Tencent。论文首页标注 Xinzhi Zhang、Hohei Chan、Yi Cai、Mengchen Zhao 属于华南理工大学软件学院，Deheng Ye 属于腾讯深圳 |
| Tags | Offline AHT, Offline MARL, Decision Transformer, Goal-conditioned RL, Partial Observability |

**一句话概括**：TAGET 把 Ad Hoc Teamwork 从在线交互设定扩展到离线数据设定，通过轨迹镜像、队友感知表征、TA-RTG 和 TA-Goal，让 ego agent 在只看本地观测时也能适应未知队友。

**我对这篇论文的定位**：这篇文章的核心不是提出一个更强的通用 offline MARL 算法，而是定义并系统化了“offline AHT”这个更窄但更现实的问题。它把 Decision Transformer 的条件生成范式用于多智能体协作，但指出单纯 RTG 不足以表达队友策略变化，因此加入 team context 和 future sub-goal。它的贡献偏方法与任务设定，理论部分较少，证据主要来自 Predator-Prey、Level-Based Foraging、Overcooked 三类协作环境实验。

## 1. 第一作者相关信息

第一作者 **Xinzhi Zhang**。论文与 OpenReview 均将其列为第一作者，机构为华南理工大学软件学院；OpenReview 作者页可作为身份入口。当前公开来源中未能确认其个人主页或完整 DBLP/Semantic Scholar 论文清单，因此这里不声称“完整发表列表”。

从这篇论文看，Xinzhi Zhang 的研究轨迹聚焦在离线多智能体决策、ad hoc teamwork 和序列建模。该工作与 Deheng Ye、Mengchen Zhao 等合作，技术上贴近腾讯游戏 AI/多智能体决策与华南理工的强化学习方向。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Ad Hoc Teamwork via Offline Goal-Based Decision Transformers | ICML 2025 Poster / OpenReview | Offline AHT | 本文，提出 TAGET |

主要研究方向可概括为：离线多智能体强化学习、未知队友适应、基于 Transformer 的轨迹条件生成。来源：论文首页、ICML 2025 虚拟海报页、OpenReview 页面。

## 2. 研究问题

Ad Hoc Teamwork (AHT) 关心的是：一个 ego agent 如何与训练时没有充分协调过、测试时可能未知的队友协作。传统 AHT 多数依赖在线交互和预设队友池；但真实场景中模拟器可能昂贵或不可得，队友策略空间又极大，因此作者提出 **offline AHT**：只用固定的多智能体交互数据训练 ego agent，部署时面对未知队友。

论文形式化为带额外队友联合策略空间的 Dec-POMDP。每个 agent 只能获得局部观测，离线数据由多智能体轨迹组成，训练目标是学习一个 ego policy，使其在给定任意队友策略时最大化团队奖励。难点有三类：离线数据覆盖有限、局部可观测下难以建模队友、普通 RTG 不能刻画队友行为变化。

## 3. 背景知识

AHT 与一般 MARL 的差别在于测试队友不一定来自训练时的固定联合最优策略。普通 MARL 往往学习联合策略或中心化 critic，而 AHT 要求 ego agent 根据有限历史快速推断队友意图。

Offline RL 的核心风险是分布外动作或状态导致外推错误。在 offline AHT 中，这个风险变成“双重分布偏移”：一方面 ego agent 不能尝试新动作，另一方面队友策略也可能不在训练数据覆盖内。部分可观测性进一步放大问题，因为 ego agent 只能看到自己的观测，不能直接知道全局 team context。

Decision Transformer (DT) 将离线 RL 视为序列建模：给定历史状态、动作和 return-to-go，预测下一步动作。本文认为标准 RTG 是全局标量，不能反映“当前队友正打算如何协作”。因此 TAGET 引入 teammate-aware return-to-go (TA-RTG) 和 teammate-aware sub-goal (TA-Goal)。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| AHT | 与未知队友临时协作 | 任务本体 |
| Offline AHT | 不在线试错，只从历史多智能体轨迹学习 AHT | 本文新设定 |
| Trajectory mirroring | 轮流把每个 agent 当 ego，扩增同一条轨迹的训练视角 | 缓解数据有限 |
| Team context | 全体 agent 当前协作状态的潜变量 | 高层模块要让 ego 从局部观测近似它 |
| TA-RTG | 依赖当前团队上下文的 return-to-go | 替代普通 RTG |
| TA-Goal | 面向未来的队友感知子目标 | 低层 DT 的条件输入 |

**为什么这篇文章的动机成立**：如果测试队友行为变化，固定 RTG 可能相同但应采取的协作动作不同。例如同样要获得高回报，ego 在 Overcooked 中要根据队友是否去拿菜、是否占用锅附近位置来改变动作。把未来全局状态或队友观测编码成 sub-goal，比只给一个总回报数字更接近“协作计划”。

## 4. 问题分析

论文用 Figure 1 对比 online AHT 与 offline AHT：在线方法能边交互边修正队友模型，离线方法只能利用固定 buffer。Figure 2 给出 TAGET 训练流程，说明作者的诊断路径是：先通过 trajectory mirroring 提高数据效率，再用双编码器处理部分可观测，最后用高层目标预测补足 RTG 的信息不足。

关键观察是：AHT 中“队友变化”不只是噪声，而是影响最优动作的条件变量。若模型直接学习 $\pi(a_t | R_t, o_t, h_t)$，其中 $R_t$ 是普通 RTG，它可能把不同队友策略下的动作平均掉。TAGET 改成先预测 team context、TA-RTG 和 TA-Goal，再让低层策略执行 goal-conditioned action generation。

实验结果也支持这一诊断：附录 Table 1 显示 TAGET 在 PP、LBF、Overcooked 的 4/8 agent 设定上整体优于 DT、BC、MADT、ODITS、LIAM 等基线；论文正文声称平均提升 37.83%。附录 Table 2 还显示 goal step 不是越大越好，复杂环境中过长的未来目标会引入预测噪声。

## 5. 思想与方法

TAGET 的思想是：不要直接把“未知队友适应”压进一个动作预测器，而要在层级上先预测协作意图，再生成动作。方法包含三部分。

第一，**trajectory mirroring**：一条包含 $N$ 个 agent 的轨迹会被拆成 $N$ 个 ego 视角，每个 agent 轮流作为 ego，其余作为 teammates。这增加训练样本，也强迫模型学习“从不同角色看同一团队轨迹”。

第二，**队友感知高层模块**：论文使用 team context encoder 与 proxy encoder。team context encoder 可看全体观测，proxy encoder 只看 ego 局部观测；通过正则化让后者近似前者，使部署时 ego 只凭局部信息也能推断协作上下文。

第三，**TA-RTG 与 TA-Goal**：高层模块动态预测 teammate-aware return-to-go 和未来子目标，低层 goal-conditioned DT 基于 $(TA\text{-}Goal, o_t, a_{<t})$ 输出动作。真正新增的是将 AHT 的队友适应显式转化为未来团队目标预测，而不是只做隐式 teammate modeling。

证据强度：理论上没有给出强保证；实验和消融支持三个模块都有作用。我的判断是，该方法在结构上合理，但对可获得的全局训练信息、轨迹质量和队友分布覆盖仍较敏感。

## 6. 算法与伪代码

算法名：**TAGET**，即 Teammate-Aware Goal driven hiErarchical Decision Transformers。

1. 输入离线多智能体轨迹数据集 $D$。
2. 对每条轨迹执行 trajectory mirroring，轮流指定 ego agent，得到扩增数据集 $D'$。
3. 训练高层模块：用全局观测编码 team context，用 ego 局部观测编码 proxy context，并用正则项让二者对齐。
4. 基于 team/proxy context 预测 TA-RTG。
5. 用 TA-RTG 解码未来 TA-Goal，即若干步后的队友/全局目标信息。
6. 训练低层 goal-conditioned Decision Transformer，根据历史序列和 TA-Goal 预测 ego action。
7. 测试时，每一步只使用 ego 局部观测更新 proxy context，预测 TA-RTG/TA-Goal，再由低层网络生成动作。

重要公式可概括为：

$$
\hat{R}_t = \sum_{t'=t}^{T} r_{t'}
$$

普通 RTG 是从当前时刻到 episode 结束的累积回报；TAGET 将其扩展为依赖当前 team context 的 TA-RTG，使回报条件更贴近队友行为。

实现细节：附录说明 DT backbone 采用 embedding dimension 64、context window $K=30$、2 层 transformer、1 个 attention head、dropout 0.3，AdamW 学习率 0.01，batch size 2048，weight decay 0.0001。

## 7. 实验与消融

环境：Predator-Prey、Level-Based Foraging、Overcooked，包含 4-agent 和 8-agent 设定。评价指标为 average return。

基线：DT、BC、MADT、ODITS、LIAM 等，分别代表单智能体序列建模、行为克隆、多智能体 DT、在线 AHT/队友建模方法。

主要结果：附录 Table 1 中 TAGET 在六个设置中均为最优或接近最优。例如 PP-4 为 61.4 左右、LBF-4 为 0.148、Overcooked-4 为 0.49；相对基线提升在不同任务上差异较大，Overcooked-4 的相对提升尤其明显。

消融：goal steps 取 1、2、3、6 时，不同环境最优值不同。Predator-Prey 偏简单且需要较长协同，step=6 最好；LBF step=2 最好；Overcooked step=3 最好，step=6 会因环境快速变化造成目标预测噪声。这个结果说明 TA-Goal 的时间尺度是关键超参数。

实验能证明：在这些标准协作环境和作者构造的 offline AHT 数据上，层级 goal prediction 比普通 DT 更稳。实验还不能完全证明：TAGET 对真实开放队友分布、视觉观测、长尾队友策略都有强泛化。

## 8. 展望

研究启发：

1. Offline AHT 是 offline MARL 中值得单独建模的设定，不应简单等同于学习联合最优策略。
2. 对未知队友的适应可通过“未来团队状态/目标”间接建模，而不是预测每个队友策略。
3. DT 类方法在多智能体任务中需要更细粒度条件信号，普通 RTG 往往太弱。
4. 轨迹镜像是多智能体离线数据利用率很高的朴素技巧。

局限与开放问题：

1. 训练依赖全局状态或全体 agent 观测，真实数据中未必有。
2. 只在小中型协作 benchmark 验证，离真实 AHT 仍有距离。
3. goal step 需要调参，且与环境复杂度强相关。
4. 缺少理论保证，尤其是未知队友分布偏移下的性能界。

后续想法：

1. 引入不确定性估计，让 TA-Goal 在队友意图不确定时输出多模态目标。
2. 将 TAGET 与扩散式轨迹规划结合，用生成模型表达多个可行协作计划。
3. 构建真实交通或机器人协作的 offline AHT 数据集，验证局部观测与队友分布偏移下的鲁棒性。

## Links

- Paper page: [PMLR](https://proceedings.mlr.press/v267/zhang25h.html)
- ICML poster: [ICML virtual poster](https://icml.cc/virtual/2025/poster/43765)
- OpenReview: [Ad Hoc Teamwork via Offline Goal-Based Decision Transformers](https://openreview.net/forum?id=tl3FlgWScA)
- PDF: [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhang25h/zhang25h.pdf)
- Code: 未能从论文/PMLR/OpenReview 公开来源确认官方代码链接
- Local PDF: `[ICML_2025] Ad Hoc Teamwork via Offline Goal-Based Decision Transformers.pdf`
