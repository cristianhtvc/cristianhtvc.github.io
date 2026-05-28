---
title: "Data Center Cooling System Optimization Using Offline Reinforcement Learning"
description: "一篇将物理先验、T-symmetry 表征学习和离线 RL 结合起来并真实部署到数据中心冷却控制的工业 ORL 论文。"
tags:
  - Offline RL
  - Industrial Control
  - Data Center Cooling
  - Physics-informed RL
  - T-symmetry
---

# Data Center Cooling System Optimization Using Offline Reinforcement Learning

| Field | 内容 |
|---|---|
| Title | Data Center Cooling System Optimization Using Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published: 22 Jan 2025, Last Modified: 14 Feb 2025 |
| Authors | Xianyuan Zhan, Xiangyu Zhu, Peng Cheng, Xiao Hu, Ziteng He, Hanfei Geng, Jichao Leng, Huiwen Zheng, Chenhui Liu, Tianshun Hong, Yan Liang, Yunxin Liu, Feng Zhao |
| Affiliations | Institute for AI Industry Research, Tsinghua University；Shanghai Artificial Intelligence Laboratory；Global Data Solutions Co., Ltd. |
| Tags | Offline RL, industrial control, data center cooling, GNN, T-symmetry, real-world deployment |

**一句话概括**：本文提出一个物理信息驱动的离线强化学习框架，用 GNN 和时间反演对称性学习可泛化的热动力学潜在表示，再在潜在空间中做样本高效的离线策略优化，并在真实商业数据中心实现 14%-21% 的冷却节能且无安全违规。

**我对这篇论文的定位**：这不是一篇只在 D4RL 上调参的 ORL 论文，而是少见的真实生产系统部署论文。它的核心贡献在于把离线 RL 的分布外泛化问题转化为“学习一个带物理结构的潜在动力学空间”，再用该空间支撑安全、稳健的策略学习。它也延续了 TSRL 的思想：当数据覆盖窄、仿真器不可用时，物理先验可能比更强的数据保守正则更关键。

## 1. 第一作者相关信息

**Xianyuan Zhan** 是清华大学智能产业研究院和上海人工智能实验室相关研究者，论文中与 Xiangyu Zhu 标注共同一作。其研究路线长期围绕离线强化学习、数据驱动工业控制、安全/稳健决策展开，尤其关注如何把 ORL 从标准 benchmark 推向真实工业系统。

从公开论文脉络看，Xianyuan Zhan 的工作有两条线：一条是离线 RL 的算法与理论，例如基于数据几何、分布校正、T-symmetry、扩散模型的离线策略学习；另一条是真实工业控制应用，例如火电燃烧优化、数据中心冷却优化等。本文可以看作这两条线的交汇：用前期关于 T-symmetry regularized offline RL 的思想解决数据中心冷却中的小数据、强安全、无仿真器问题。

| Year | Title | Venue/Source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2022 | DeepThermal / offline RL for combustion optimization | AAAI / industrial RL line | Industrial Offline RL | 真实工业控制落地的前置路线 |
| 2022 | Model-based Offline Planning with Trajectory Pruning | IJCAI | Model-based Offline RL | 关注离线规划和模型误差控制 |
| 2023 | Generalizable Offline RL via Data Geometry / GOFAR | ICLR | Generalization in ORL | 处理数据覆盖和泛化问题 |
| 2023 | T-symmetry Regularized Offline RL | NeurIPS | Physics-informed ORL | 本文 TTDM 和 T-symmetry 正则的直接思想来源 |
| 2024 | Diffusion-DICE | NeurIPS / related line | Generative Offline RL | 与 in-sample / 分布校正路线相关 |
| 2025 | Data Center Cooling System Optimization Using Offline RL | ICLR | Real-world ORL deployment | 本文，将物理先验 ORL 部署到生产数据中心 |

**研究方向综合**：第一作者团队的特点是把 ORL 看作“受数据支持、受结构先验约束的真实控制问题”，而不是单纯的 benchmark 优化。这使得本文很重视部署系统、传感器预处理、动作平滑、安全约束和真实测试时长。

**信息来源**：论文首页署名与机构；OpenReview 页面；DBLP 条目。

## 2. 研究问题

本文解决的是**数据中心地板级空气侧冷却系统的离线控制优化**。具体地，数据中心机房内有多台 ACU（Air Cooling Unit），每台 ACU 可调节风扇转速和冷冻水阀门开度。目标是在保证冷通道温度不超过安全阈值、送风/出风温度满足运维要求的情况下，尽可能降低 ACU 冷却能耗。

这个问题重要有三点：

1. 数据中心能耗巨大，论文指出典型数据中心约 30%-40% 能源用于冷却系统，降低冷却能耗有明显经济和碳排价值。
2. 真实数据中心不能做危险在线探索，服务器过热可能导致生产事故，因此在线 RL 很难直接训练。
3. 高保真热仿真器很难建立，CFD 或多物理场仿真存在系统辨识难、校准成本高、sim-to-real gap 大的问题。

论文假设的设置是一个连续状态、连续动作、离线数据驱动的 MDP：

- 状态 $s=\{s_s,s_a,s_e\}$：传感器温湿度、服务器机架温度、ACU 工作状态、外部不可控变量等。
- 动作 $a$：所有 ACU 的风扇转速 $f_m$ 和阀门开度 $o_m$。
- 数据集 $\mathcal{D}$：来自真实数据中心历史 PID 控制日志，分布窄、数据量相对控制复杂度有限。
- 策略：离线训练后部署到真实机房闭环控制，不依赖仿真器。

安全感知奖励函数是论文的核心建模之一：

$$
r =
r_0
- \beta_1 \sum_{m=1}^{M} f_m^3
- \beta_2 \sum_{n=1}^{N}\log(1+\exp(T_n^c-\rho_T))
- \beta_3 \sum_{m=1}^{M} o_m
- \beta_4 \sum_{m=1}^{M}\log(1+\exp(T_m^l-\rho_L)).
$$

含义：第一项让奖励为正；风扇功耗按转速三次方惩罚；冷通道温度超过阈值 $\rho_T$ 会受到 softplus 惩罚；阀门开度也计入能耗；ACU 送风/出风温度超过阈值 $\rho_L$ 也被惩罚。这里的目标不是单纯省电，而是在省电和热安全之间做可部署的折中。

## 3. 背景知识

**数据中心空气侧冷却控制**：典型数据中心由冷机和冷却塔提供冷冻水，ACU 将冷空气送入机房冷通道，服务器排出的热空气进入热通道。空气侧控制主要调节 ACU 风扇转速和阀门开度。风扇功耗与转速近似三次方相关，因此稍微降低风扇转速就可能节省大量能耗，但如果降得过多会导致冷通道温度超标。

**离线强化学习**：离线 RL 只使用固定数据集 $\mathcal{D}$ 学习策略，不能与环境交互。它适合数据中心，因为历史运行日志可用，在线探索危险。但它有分布偏移问题：新策略可能选择历史 PID 很少执行的动作，价值函数在这些 OOD 区域容易外推错误。

**为什么普通 ORL 不够**：CQL、IQL、TD3+BC 等方法通过保守 Q 值、行为约束或 in-sample 学习减少 OOD 错误。但真实数据中心数据来自保守 PID，动作覆盖很窄；如果只靠行为约束，策略会被锁在原 PID 附近，很难发现更节能的控制方式。

**T-symmetry**：时间反演对称性来自物理系统的基本先验，直觉是物理规律在时间方向反转后仍应保持一致。在本文中，它被用于约束潜在前向动力学 $f$ 和潜在反向动力学 $g$：

$$
f(z_s,z_a) \approx -g(z_s + f(z_s,z_a), z_a).
$$

也就是说，如果潜在状态在动作 $z_a$ 下从 $z_s$ 前进了一步，那么从前进后的状态往回看，应当能得到一致的反向变化。这种约束不是直接说温度真的可逆，而是给学到的潜在动力学施加物理结构，使其在小数据和 OOD 区域更平滑、更可泛化。

**GNN 在本文中的作用**：机房不是一个无结构向量系统。传感器之间有空间邻近关系，ACU 对不同传感器有控制影响关系。GNN 把这些依赖显式编码进模型，使热动力学表示不完全依赖数据自己发现结构。

| Term | 朴素含义 | 在本文中的作用 |
|---|---|---|
| ACU | 机房空调单元 | 被策略控制的执行器 |
| CAT | Cold Aisle Temperature | 主要安全约束指标 |
| ACLF | ACU 能耗 / 服务器能耗 | 评估冷却效率，越低越好 |
| TTDM | T-symmetry enforced Thermal Dynamics Model | 学习带物理先验的潜在热动力学 |
| OOD generalization | 对数据外状态动作的合理外推 | 解决历史 PID 数据覆盖窄的问题 |
| Latent policy learning | 在潜在表示上学 Q 与策略 | 提高小样本和部署稳健性 |

**为什么这篇文章的动机成立**：数据中心冷却具备“有日志、不能探索、仿真难、安全要求高”的典型 ORL 条件，但也恰好暴露了标准 ORL 的弱点：仅靠数据支持会过度保守，仅靠深度模型会过拟合窄数据。论文的动机是用物理先验弥补数据覆盖不足，这比单纯换一个更强 baseline 更贴近真实问题。

## 4. 问题分析

论文的诊断路径不是从一个理论定理出发，而是从真实系统约束出发：

1. **传统 PID 的局限**：各 ACU 多为局部控制，缺乏对全局热场和服务器负载变化的协同感知。Figure 3 中 PID 阶段风扇转速和阀门开度有长时间固定或异常剧烈调整，而本文方法动作更动态、更平滑。
2. **MPC/仿真方法的局限**：高保真热模型很难建立，真实机房布局、服务器负载、冷热通道流动都高度复杂，仿真到真实的迁移不可靠。
3. **普通离线 RL 的局限**：历史数据来自安全保守的 PID 控制器，数据窄。如果算法强行最大化 Q，容易走出数据支持；如果强行为约束，又很难超过 PID。
4. **核心变量**：传感器空间依赖、ACU 控制依赖、服务器负载变化、温度安全阈值、动作平滑性和 OOD 表征泛化共同决定策略能否部署。

支撑分析的关键证据包括：

- Figure 1 展示数据中心地板级冷却系统和不同负载下温度场变化，说明问题具备复杂空间热耦合。
- Figure 2 给出整体框架，显示 TTDM 通过图结构和 T-symmetry 连接表征学习与策略学习。
- Table 1 和 Figure 3-5 展示真实数据中心实验，证明方法不只是离线评估有效。
- Figure 6 在小型真实 testbed 上对比 PID、MPC、CCA、IQL、CQL、FISOR 等，强调本文方法在能耗和安全之间更稳。
- Figure 7 消融 GNN 和 T-symmetry，说明核心结构不是装饰项，确实降低多步预测误差并提升控制性能。

这一路径排除了两个简单解释：第一，节能不是因为牺牲温度安全，实验中无 CAT 违规；第二，提升不是单纯来自更大网络，因为去掉 GNN 或 T-symmetry 后预测和策略效果都下降。

## 5. 思想与方法

本文的核心思想是：**先学习一个遵守机房物理结构的潜在热动力学模型，再在该潜在空间中做保守但不僵化的离线策略优化**。

方法包含三个主要组件。

**组件一：安全感知 MDP 和奖励设计**  
状态拆成传感器状态、ACU 状态和外部因素；动作是风扇转速与阀门开度；奖励同时惩罚能耗和温度超限风险。这个设计把现场运维经验直接写入优化目标。

**组件二：TTDM 表征学习**  
TTDM 使用状态动作编码器 $\phi(s,a)$ 得到 $z_s,z_a$，用 GNN 捕捉传感器和 ACU 的空间/控制依赖，并训练重构、前向 ODE、反向 ODE、状态变化解码和 T-symmetry 一致性损失：

$$
\mathcal{L}_{TTDM}
=
\sum_{(s,a,s')\in \mathcal{D}}
(\mathcal{L}_{rec}+\mathcal{L}_{fwd}+\mathcal{L}_{rvs}+\mathcal{L}_{ds}+\mathcal{L}_{T\text{-}sym}).
$$

这个组件的因果机制是：图结构减少模型学习热场关系的样本需求，T-symmetry 让潜在动力学在正反时间方向上自洽，从而降低小数据下的任意外推。

**组件三：潜在空间离线策略优化**  
Q 函数不是直接在原始状态动作空间学，而是在 $Q(z_s,z_a)$ 上学习：

$$
\min_Q\ \mathbb{E}_{(s,a,s')\sim \mathcal{D}}
\left[
r(s,a)+\gamma \bar{Q}(\phi(s',\pi(s')))-Q(\phi(s,a))
\right]^2.
$$

策略目标类似 TD3+BC，但加入潜在 Q 最大化、行为动作约束和 T-symmetry 约束：

$$
\max_{\pi}\ \mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[
\lambda_\alpha Q(\phi(s,\pi(s)))
-(\pi(s)-a)^2
-\mathcal{L}_{T\text{-}sym}(\phi(s,\pi(s)))
\right].
$$

这比普通 TD3+BC 多了一个关键点：策略产生的动作不仅要接近数据，还要在 TTDM 的潜在物理一致性空间内合理。这让策略可以适度离开 PID 行为以省电，同时避免无物理意义的 OOD 动作。

**新颖性判断**：GNN、TD3+BC、T-symmetry 都不是从零发明；真正的新意是把它们组合成一个可部署的物理信息 ORL 系统，并用 2000 小时真实闭环运行证明有效。

## 6. 算法与伪代码

算法名可以概括为 **Physics-informed Offline RL for DC Cooling**，核心模型是 **TTDM + latent-space offline policy optimization**。

**训练流程（对应 Appendix Algorithm 1）**：

1. 输入预处理后的历史数据集 $\mathcal{D}$，初始化 Q 网络、策略网络 $\pi$ 和 TTDM。
2. 训练 TTDM：
   1. 从 $\mathcal{D}$ 中采样小批量 $(s,a,s',a')$。
   2. 用编码器 $\phi(s,a)$ 得到 $z_s,z_a,z_{s'},z_{a'}$。
   3. 计算重构损失 $\mathcal{L}_{rec}$，确保潜在表示能还原状态和动作。
   4. 计算潜在前向/反向 ODE 损失 $\mathcal{L}_{fwd},\mathcal{L}_{rvs}$。
   5. 计算状态变化解码损失 $\mathcal{L}_{ds}$。
   6. 计算 T-symmetry 一致性损失 $\mathcal{L}_{T\text{-}sym}$。
   7. 最小化总损失 $\mathcal{L}_{TTDM}$ 更新 TTDM。
3. 训练离线策略：
   1. 从 $\mathcal{D}$ 中采样 $(s,a,r,s')$，其中 $r$ 由安全感知奖励函数计算。
   2. 在潜在空间中用 TD 目标更新 $Q(\phi(s,a))$。
   3. 用潜在 Q 最大化、行为约束和 T-symmetry 正则更新策略 $\pi$。
4. 部署时：
   1. 实时过滤异常传感器数据并重采样到统一时间间隔，商业数据中心为 5 分钟，小型 testbed 为 2 分钟。
   2. 策略输出动作后做时间平滑，执行当前输出和前 4 个时刻输出的平均值。
   3. 将平滑后的风扇转速和阀门开度发送给 ACU。

重要实现细节：

- TTDM 使用 Adam，学习率 $3\times 10^{-4}$，weight decay $10^{-5}$。
- GNN hidden units 为 256，前向/反向模型 hidden units 为 128。
- 折扣因子 $\gamma=0.99$，target update rate 为 0.005。
- 策略网络和 critic 网络宽度为 512。
- $\alpha$ 在 $[2.5,10]$ 调参，用于平衡价值最大化和行为正则。
- 策略噪声 0.2，policy noise clipping 0.5，policy update frequency 2。

训练和部署之间的关键差距是：训练时只用历史数据；部署时必须处理实时数据质量、动作平滑、现场阈值和人工运维约束。论文把这些系统工程细节写进附录，是它可信度较高的原因之一。

## 7. 实验与消融

**真实商业数据中心实验**：

- 场景：两个真实服务器房间 A/B，控制 4 台、6 台以及全部 ACU，真实生产环境闭环运行。
- 时间：2024 年 1 月到 12 月期间多轮实验，总计超过 2000 小时。
- 指标：ACLF，即 ACU 冷却能耗与服务器能耗之比；同时监控 CAT 安全违规。
- 主要结果：Table 1 显示相较 PID，本文方法在两个机房实现 14%-21% 的冷却能效提升，并且无热安全或操作约束违规。

**控制质量分析**：

- Figure 3：48 小时对比表明，在相似服务器负载下，本文方法 ACLF 更低；动作更动态，倾向于降低风扇转速、略增阀门开度，从而在保持冷却能力的同时降低风扇功耗。
- Figure 4：14 天长期实验显示，不同服务器负载下本文方法 ACLF 均低于 PID，且负载越高节能优势越明显；冷通道温度仍低于 25 摄氏度阈值。
- Figure 5：控制更多 ACU 时节能效果更强，说明联合控制规模越大，策略优化空间越大。

**小型真实 testbed 对比**：

- 环境：22 台服务器和 1 台 ACU，可做更充分探索。
- 负载：低、中、高三种服务器负载，平均功率约 4.9kW、7.4kW、8.0kW。
- Baselines：PID、MPC、CCA、IQL、CQL、FISOR。
- Figure 6：CCA 和 CQL 等激进方法可能降低能耗但出现温度安全问题；本文方法在所有负载下有最高能效且无 CAT violation。

**消融实验**：

- Figure 7a/b：加入 GNN 后，多步热状态预测误差显著降低，尤其预测步数变长时更明显。
- Figure 7a/b：加入 T-symmetry 后，TTDM 预测误差进一步下降。
- Figure 7c：离线策略优化中去掉 T-symmetry 会显著削弱低负载和高负载下的节能表现。
- Appendix C.2 / Table 2：奖励函数中温度惩罚权重 $\beta_2$ 改变时 ACLF 仍比较稳定，说明奖励设计有一定鲁棒性。

**实验解释边界**：

这些实验强有力证明了方法可部署、可节能、可安全运行。但它们不能证明方法在所有工业控制场景中通用，也不能单独分离每个系统工程细节的贡献。尤其是动作平滑、传感器过滤、现场阈值调参等部署细节本身也可能对安全性很关键。

## 8. 展望

**对 ORL 研究者的启发**：

1. 真实 ORL 系统的难点往往不只是算法，而是“数据窄、无仿真、安全阈值、部署闭环”共同出现。
2. 物理先验可以替代一部分数据覆盖，在数据稀缺时比单纯加保守正则更有价值。
3. 潜在空间策略学习不是为了抽象而抽象，而是为了得到更可泛化的价值估计。
4. 真实部署指标应同时报告性能和安全违规，单一 return 不足以评价工业控制。
5. 动作平滑和实时数据预处理应被视为控制算法的一部分，而不是附录工程细节。

**局限与开放问题**：

1. 安全约束主要通过奖励惩罚和部署监控实现，尚非形式化安全保证。
2. T-symmetry 是否适用于所有非平衡热系统或其他工业过程，还需要更系统验证。
3. 方法依赖领域知识构图，不同数据中心布局迁移时需要重新建图或适配。
4. 真实商业数据中心的实验不可完全复现，社区难以独立比较。
5. 只有空气侧 ACU 控制，水侧冷机/冷却塔联合优化仍是更大问题。

**后续研究想法**：

1. **可证明安全的 physics-informed ORL**：将 TTDM 与 reachability 或 control barrier function 结合，为温度约束提供形式化保证。
2. **跨数据中心迁移学习**：研究如何把一个机房学到的图结构和潜在热动力学迁移到不同布局、不同 ACU 数量的机房。
3. **空气侧与水侧联合离线优化**：把 ACU、冷机、冷却塔放入统一多层 MDP，探索端到端冷却系统节能。

## Links

- Paper page: https://openreview.net/forum?id=W8xukd70cU
- DBLP: https://dblp.org/rec/conf/iclr/ZhanZ0HHGLZLHLL25
- arXiv: https://arxiv.org/abs/2501.15085
- Code: 未能从论文和 OpenReview 页面确认官方公开代码链接
- First author / DBLP search: https://dblp.org/search?q=Xianyuan%20Zhan
