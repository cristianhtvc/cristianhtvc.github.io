---
title: "OHIO: Offline Hierarchical Reinforcement Learning via Inverse Optimization"
description: "提出 OHIO，用低层控制器结构反推出不可观测高层动作，把普通离线轨迹转化为可训练层级策略的数据集。"
tags:
  - Offline RL
  - Hierarchical RL
  - Inverse Optimization
  - Robotics
  - Network Optimization
---

# Offline Hierarchical Reinforcement Learning via Inverse Optimization

| 字段 | 内容 |
|---|---|
| Title | Offline Hierarchical Reinforcement Learning via Inverse Optimization |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published: 2025-01-22, Last Modified: 2025-03-14 |
| Authors | Carolin Schmidt, Daniele Gammelli, James Harrison, Marco Pavone, Filipe Rodrigues |
| Affiliations | Technical University of Denmark; Stanford University; Google DeepMind |
| Tags | Offline RL; Hierarchical RL; Inverse Optimization; Low-Level Controllers; Robotics; Network Optimization |

**一句话概括**：OHIO 把“低层动作轨迹中缺失的高层动作”视为逆优化问题来恢复，从而把任意来源的 flat/offline 数据转换成高层策略可直接使用的离线 RL 数据集。

**我对这篇论文的定位**：它不是一个新的 IQL/CQL 类离线 RL 算法，而是一个数据重构框架。其价值在于让标准 offline RL 可以训练层级策略，尤其适合真实系统中已有可靠低层控制器或优化器、但历史数据不是按目标层级结构收集的场景。它把 model-based 信息只用于单步 inverse problem，而把长时域优化留给 model-free/offline RL，避免多步模型误差累积。

## 1. 第一作者相关信息

**Carolin Schmidt** 是 Technical University of Denmark 相关研究者，论文 PDF 给出 DTU 邮箱，并与 Filipe Rodrigues 共同隶属 DTU。OpenReview 与项目页显示本文为其 ICLR 2025 Poster。公开资料中 Carolin Schmidt 的可核验强化学习相关记录主要集中在 offline/hierarchical RL 与交通/网络控制方向。

其研究轨迹可概括为：面向真实系统的层级决策与离线学习，尤其是把控制/优化结构作为 inductive bias，使 RL 能落到机器人、车辆路由和供应链这类有硬约束的大规模系统。公开可直接核验的 Carolin Schmidt 相关 ORL 条目较少，因此下表只列本文及其公开版本/项目材料，不扩展到无法确认作者身份的同名记录。

| 年份 | 论文 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Offline Hierarchical Reinforcement Learning via Inverse Optimization | ICLR 2025 Poster | Offline HRL / Inverse Optimization | 本文；提出 OHIO |
| 2024 | Offline Hierarchical Reinforcement Learning via Inverse Optimization | arXiv preprint, arXiv:2410.07933 | Offline HRL / Inverse Optimization | 本文预印本版本 |
| 2025 | OHIO project page, code and data release | Project/GitHub | Reproducible Offline HRL | 本文配套实现与数据 |

**主要方向概括**：层级 RL、离线 RL、逆优化、机器人控制、网络优化。作者同名记录较多，本节仅使用论文 PDF、OpenReview、项目页和相关可核验论文线索。

## 2. 研究问题

层级策略通常由高层策略 $\pi_u$ 和低层策略 $\pi_l$ 组成。高层以较低频率输出子目标、目标状态或优化目标 $u_t$；低层把 $u_t$ 转成实际控制动作 $a_t$。问题是：历史离线数据通常只有 $(s_t,a_t,r_t,s_{t+1})$，没有高层动作 $u_t$。如果直接训练 $\pi_u(u_t\mid s_t)$，训练标签不存在。

更麻烦的是，数据可能由完全 flat 的 end-to-end 策略收集，或者低层控制器配置与部署时不同。已有 hierarchical offline RL 往往假设数据就是同一层级结构生成的，或者用 autoencoder/latent primitives 学隐变量；这些隐变量不一定是部署低层控制器真正需要的高层动作。

OHIO 的目标是：给定目标层级结构和低层策略知识，反推出每个 transition 最可能对应的高层动作 $\hat u_t$，构造

$$
\tilde{\mathcal D}=\{(s_t,\hat u_t,\hat r_t,s_{t+1})\},
$$

然后用普通 BC/IQL/CQL/SAC 等算法训练高层策略。

## 3. 背景知识

**Hierarchical RL**：把决策拆成不同时间尺度。高层负责规划“要去哪/要达到什么目标”，低层负责“如何执行”。好处是降低动作维度、利用控制器结构、让长时域任务更容易。

**低层显式策略与隐式策略**：

- 显式策略：$a=f(s,u,\epsilon)$，如 PID、LQR、神经网络 policy。
- 隐式策略：$a=\arg\min_{a\in A}f(s,a,u)$，如 MPC、线性规划、扩散策略等优化型控制器。

**逆优化**：正向问题是给定目标 $u$，求最优低层动作 $a^\star$；逆问题是观察到 $a_t$ 或 $s_t\to s_{t+1}$，反推出最可能的 $u_t$。OHIO 的关键是低层策略结构已知，所以逆问题不是任意表征学习，而是有物理/优化含义的反推。

**为什么不是普通 model-based RL**：OHIO 需要近似单步动力学 $\tilde P$ 来判断某个 $u$ 是否能解释 $s_{t+1}$，但不会用 $\tilde P$ 做多步 planning。长时域优化仍由 offline RL 在重构数据上完成。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| High-level action $u$ | 子目标、目标分布、目标状态等抽象动作 | 高层策略要学习的动作 |
| Low-level action $a$ | 实际控制动作、流量、力矩等 | 原始离线数据中可观测 |
| Policy inversion | 从 $s,a,s'$ 反推 $u$ | OHIO 的数据重构核心 |
| Inverse optimization | 反解优化器目标或参数 | 处理 MPC/LP 等隐式低层策略 |
| Observed State baseline | 用下一个状态当作高层动作 | 论文中反例，通常失败 |
| End-to-End baseline | 直接在原始动作空间训练 | 高维动作下容易不稳定 |

**为什么动机成立**：现实系统常已有低层控制器，例如机械臂 operational-space controller、车辆调度 LP、供应链 MPC。离线数据来自历史操作，但高层目标没记录。OHIO 利用低层结构补齐这层标签，让 RL 学“该给控制器什么目标”，而不是从零学所有底层动作。

## 4. 问题分析

论文的诊断是：层级 offline RL 失败的根本原因不是 offline RL 算法不够强，而是数据的动作空间错了。高层策略需要 $u_t$，数据里只有 $a_t$；直接把 $s_{t+1}$ 当 $u_t$ 的 Observed State baseline 在 Reacher 和 RoboSuite 中几乎失败。

核心逆问题写作：

$$
\hat u_{0:T-1}
=
\arg\max_{u_{0:T-1}}
\log \tilde P(s_{1:T}\mid a_{0:T-1},s_{0:T-1})
+\sum_t \mathcal L(u_t)
$$

subject to

$$
a_t=\pi_l(s_t,u_t),\quad \forall t.
$$

实践中作者把跨时间联合问题分解成单步问题：

$$
\hat u_t
=
\arg\max_{u_t}
\log \tilde P(s_{t+1}\mid a_t,s_t)+\mathcal L(u_t),
\quad
\mathrm{s.t.}\ a_t=\pi_l(s_t,u_t).
$$

这个分解牺牲了全局最优性，但计算可行。作者在 Discussion 中也承认，未来可用 cross-timestep losses 或 filtering/smoothing 式方法改进。

对 LQR 低层策略，论文 Example 3.2 给出解析逆：

$$
\hat u=(BK)^\dagger\left(s'-(As+c+Bk)\right),
$$

其中 $K,k$ 是 LQR 反馈策略参数，$(\cdot)^\dagger$ 是 Moore-Penrose pseudo-inverse。对 LP/MPC 低层策略，逆问题可转化为 LP 或数值优化。

## 5. 思想与方法

方法名 **OHIO** = **O**ffline **HI**erarchical RL via Inverse **O**ptimization。

指导原则是：如果低层控制器是已知结构，那么观察到的低层行为可以反推出“它像是在追求什么高层目标”。这一步完成后，标准 offline RL 就能在高层动作空间中工作。

方法组件：

1. **低层策略建模**：明确 $\pi_l(s,u)$ 是显式策略还是隐式优化器。
2. **逆问题求解**：根据是否有解析结构，选择 LQR 解析逆、LP 逆优化、梯度优化、CEM、采样或枚举。
3. **数据集重构**：把原始轨迹 $\tau=(s_0,a_0,r_0,\dots,s_T)$ 转成 $\tilde\tau=(s_0,\hat u_0,\hat r_0,\dots,s_T)$。
4. **高层 offline RL**：用 BC/IQL/CQL/SAC 等训练 $\pi_u$。
5. **部署**：高层输出 $u_t$，低层控制器执行实际动作，通常连续执行若干步。

真正新的是“用低层策略结构做 label recovery”，而不是学习一个抽象 latent action。这样恢复出的 $u_t$ 有领域含义，例如机械臂目标姿态、车辆目标分布、供应链商品目标分布。

它解决问题的机制也很直接：高维低层动作空间被压缩到低维高层目标空间；硬约束和可行性由低层优化器/控制器保证；offline RL 只需学习较低维、结构化、可解释的决策。

## 6. 算法与伪代码

**Algorithm 1: OHIO**

1. 输入原始状态转移数据集 $D$；可选输入近似动力学 $\tilde P$ 和奖励函数 $r$。
2. 初始化高层数据集 $\tilde D=\emptyset$。
3. 对每条原始轨迹 $\tau\in D$：
4. 对每个时间步 $t$，求解单步逆问题，得到 $\hat u_t$。
5. 若原始奖励可用，保留奖励；否则用奖励函数计算 $\hat r_t=r(s_t,\hat a_t)$。
6. 构造高层轨迹 $\tilde\tau=(s_0,\hat u_0,\hat r_0,\dots,s_T)$。
7. 将 $\tilde\tau$ 加入 $\tilde D$。
8. 在 $\tilde D$ 上训练高层策略 $\pi_u$，算法可选 BC/IQL/CQL/SAC。
9. 推理时，$\pi_u$ 输出 $u_t$，低层 $\pi_l$ 将其转成实际动作。

**求解器选择**

| 类型 | 适用低层策略 | 求解方式 | 取舍 |
|---|---|---|---|
| 解析逆 | LQR/线性反馈 | pseudo-inverse 闭式解 | 快，但对模型误差敏感 |
| 逆 LP | 网络流、供应链、MPC/LP | LP 或 $L_1$ 投影 | 可编码约束，适合真实系统 |
| 数值优化 | 一般可微低层 | 梯度下降/CEM/采样 | 更鲁棒但离线预处理成本高 |
| 枚举 | 离散高层动作 | 穷举所有 $u$ | 简单但动作空间不能太大 |

重要实现点：OHIO 不需要把梯度穿过低层优化器传到 RL 训练中；逆问题只在数据预处理阶段求解，所以可以使用较重的优化方法。

## 7. 实验与消融

论文实验覆盖 robotics 和 network optimization。

**Reacher 目标控制**：Table 1 比较 End-to-End、Observed State 和 OHIO。OHIO 在 HR Dataset 上数值逆 97.1、解析逆 98.2；Observed State 只有 0.05。E2E Dataset 上 OHIO 数值逆 99.2，与 End-to-End 99.3 接近。对 LQR 参数错配（E2E-10C/E2E-10S），数值逆仍保持 95+，解析逆略降，说明数值逆更抗模型错配。

**RoboSuite Lift/Door**：Table 2 显示原始控制器下 IQL 与 OHIO 都好；修改 stiffness/damping 后 IQL 大幅下降，Door 上 modified damping 只有 2.9，而 OHIO 仍有 76.7。Table 3 显示 Observed State baseline 在 Lift/Door 几乎为零，HRL baseline 低于 OHIO；减少 Door 数据后 OHIO 仍 88.2，HRL 降到 72.7。

**动态车辆路由**：Table 4 在 NYC/SHZ 多个行为策略数据集上比较 E2E 与 OHIO。E2E 的 BC 在 NYC 上甚至为 -35.2，OHIO 的 BC/IQL/CQL 多数保持 80-98。论文强调低层动作维度在 NYC/SHZ 是 196/289，而 OHIO 高层动作约 14/17，降维解释了稳定性。

**供应链管理**：Table 5 显示在 1W3S 和 1W10S 上，OHIO 对 CQL 等需要在未见动作上估值的算法更稳定。更关键的是 transfer：在 1W10S-MPC 训练、1W10S-MPC-CAP 测试时，E2E 策略性能至少掉 50%，OHIO 最多只掉约 5%。Figure 2 进一步显示在线 fine-tuning 中 OHIO 更稳定，E2E 会因违反约束而跌到负分。

**实验解释**：OHIO 的强项是高维动作、有硬约束、有已知低层优化结构的系统。它把可行性和约束交给低层控制器，让高层 RL 学低维目标；这比直接在原始动作空间做 offline RL 更稳。

## 8. 展望

对 ORL 研究者的启发：

1. 数据重构有时比设计新 offline RL loss 更重要；动作空间不对，算法再强也难学。
2. 真实系统已有控制器/优化器是宝贵先验，应通过层级结构纳入 RL。
3. 低层约束可作为 safety mechanism，减少 E2E 策略在 offline-to-online 中的约束违反。
4. 逆优化为“从日志数据学习高层意图”提供了比 latent action 更可解释的路径。

局限与开放问题：

1. 需要知道或近似低层策略结构；若低层控制器黑箱且不可逆，OHIO 难用。
2. 单步逆问题忽略跨时间一致性，可能产生抖动或局部错误标签。
3. 逆问题求解可能计算昂贵，尤其在复杂 MPC 或高维连续目标中。
4. 对动力学近似误差敏感，论文虽做部分错配实验，但尚未系统给出误差界。

可能的后续研究：

1. **时序一致的 OHIO**：用 filtering/smoothing 或 sequence-level inverse optimization 让 $\hat u_t$ 在时间上更平滑。
2. **黑箱低层控制器反演**：学习 surrogate inverse model，扩展到无法显式写出 $\pi_l$ 的工业系统。
3. **安全 fine-tuning 框架**：把 OHIO 与 constrained offline-to-online RL 结合，系统评估约束违反和恢复能力。

## Links

- Paper page: [OpenReview](https://openreview.net/forum?id=dTPz4rEDok)
- PDF: [OpenReview PDF](https://openreview.net/pdf?id=dTPz4rEDok)
- arXiv: [arXiv:2410.07933](https://arxiv.org/abs/2410.07933)
- Project / Code / Data: [OHIO project page](https://ohio-offline-hierarchical-rl.github.io/)
- First author profile: [Carolin Schmidt on OpenReview](https://openreview.net/profile?id=~Carolin_Schmidt1)
