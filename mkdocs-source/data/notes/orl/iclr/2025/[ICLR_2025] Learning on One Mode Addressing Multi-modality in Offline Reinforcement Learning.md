# Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning

> **深度阅读笔记** | ICLR 2025 Accepted Paper
> 阅读日期：2026-05-27

---

## 文章元数据

- **标题**：Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning
- **作者**：Mianchu Wang*, Yue Jin*, Giovanni Montana* (华威大学 University of Warwick; Montana 同时隶属 Alan Turing Institute)
- **会议/年份**：ICLR 2025
- **领域**：Offline Reinforcement Learning, Multi-modal Action Distributions, Weighted Imitation Learning
- **代码**：未在论文中提供链接，未能从论文/公开来源确认

---

## 一句话总结

LOM 无需对整个多模态动作分布建模，而是通过 GMM 建模行为策略、用 Hyper Q-function 选择最优模式（mode），再在单一模式下做加权模仿学习——在简化学习过程的同时实现 SOTA 性能。

---

## 定位段落

LOM 属于**离线强化学习中面向多模态行为策略的加权模仿学习方法**，处于以下研究线的交汇点：

1. **加权模仿学习 (Weighted Imitation Learning)**：继承 AWR、AWAC、CRR 等方法的优势加权思路，但首次放弃 unimodal Gaussian 假设，改用 GMM 建模行为策略；
2. **多模态离线 RL**：区别于 PLAS (CVAE)、LAPO (latent advantage weighting)、WCGAN/WCVAE、Diffusion-QL 等尝试**完整建模**多模态分布的复杂方法，LOM 反其道而行——只学一个最优模式；
3. **模式选择**：借鉴 DAWOG、DMPO 的状态-动作空间分区思想，但用一个全新的 Hyper-MDP 框架和 Hyper Q-function 来做模式级别的决策。

论文的核心哲学洞察是：**完整建模多模态分布既复杂又没必要——只需找到回报最高的那个模式，在上面做 imitation learning 就够了。**

---

## 第一作者简介

**Mianchu Wang (王冕初)** 和 **Yue Jin (金悦)** 为共同第一作者，英国华威大学。通讯作者 **Giovanni Montana** 是华威大学统计学教授，同时隶属 Alan Turing Institute。该团队此前在 goal-conditioned offline RL 和状态空间分区方面有相关工作（Wang et al., 2024a Machine Learning; Wang et al., 2024b TMLR/GOPlan）。未能从论文/公开来源确认第一作者完整发表列表。

---

## 预备知识地图

### 1. 离线 RL 中的多模态问题

真实离线数据集常由多个策略/人类演示/不同探索策略混合收集而来。同一个状态可能对应多个不同但都有效的动作（如自动驾驶的保守 vs 激进风格、机械臂抓取的不同角度），形成**多模态动作分布 (multi-modal action distribution)**。

**核心挑战**：传统方法假设行为策略是 unimodal Gaussian，导致学到的策略对 conflicting actions 取平均——这个平均值可能既不在数据集中，也无效。

### 2. 现有处理多模态的方法

- **生成式建模**：PLAS (CVAE)、LAPO (latent VAE + advantage weighting)、WCGAN/WCVAE (GAN/VAE + advantage weighting)、Diffusion-QL (扩散模型) —— 试图完整建模多模态分布，计算量大
- **分区方法**：DAWOG (状态-动作空间分区 + 条件高斯策略)、DMPO (混合确定性策略)
- **模式寻求**：TD3+RKL (reverse KL 的模式寻求性质)、BAW (过滤低值动作)

### 3. 加权模仿学习 (Weighted Imitation Learning)

核心公式：J(pi) = E_{s~d_pib, a~pib}[exp(A_pib(s,a)/beta) * log pi(a|s)]

即：从行为策略中采样动作，按其优势 (advantage) 的指数加权来模仿。优势高的动作被赋予更高的模仿权重。这等价于在 KL 散度约束下优化策略。

- **AWR** (Peng et al., 2019)：模仿行为策略的动作
- **AWAC** (Nair et al., 2021)：模仿最新策略的动作
- **CRR** (Wang et al., 2020)：直接最大化 Q 值 + KL 正则
- **STR** (Mao et al., 2023)：在支持集内做 trust region 优化

**共同局限**：假设模仿的动作来自 unimodal Gaussian——在多模态数据上失效。

### 4. Gaussian Mixture Model (GMM) 与 Mixture Density Network (MDN)

- **GMM**：用 M 个高斯分布的加权和来建模复杂分布：p(a|s) = sum_i alpha_i(s) * N(mu_i(s), sigma_i^2(s))
- **MDN** (Bishop, 1994)：用神经网络输出 GMM 参数（混合系数 alpha_i、均值 mu_i、标准差 sigma_i），通过最小化负对数似然训练

---

## 0. 问题分析与动机解读

### 核心观察

多模态行为策略中，不同的模式 (mode) 对应不同的行为策略。关键洞察是：

并非所有模式都同等重要。只有少数模式（甚至一个模式）与高回报关联。完整建模所有模式既不必要，还会引入噪声。

论文的 motivating example (附录 A, 图5) 清晰展示：在 one-step MDP 中，Gaussian 策略无法分离模式；CVAE 和 CGAN 可以建模多模态但精度有限；而 MDN 能清晰分离模式，且通过 Hyper Q-function 排序后，top 模式精确对应高回报区域。

### 现有方法的缺陷

1. 加权模仿学习方法 (AWAC, CRR, STR 等) 假设 unimodal Gaussian，在多模态数据上取平均
2. 生成式建模方法 (WCGAN, WCVAE, Diffusion-QL) 完整建模多模态分布，计算量大且对不必要模式的建模有害
3. 分区方法 (DAWOG, DMPO) 需要额外的状态-动作分区设计，复杂

### LOM 的两步直觉

Step 1 (Mode Selection): 从 GMM 建模的行为策略中找出回报最高的模式——通过 Hyper Q-function 完成
Step 2 (Imitation on One Mode): 在这个单一 (unimodal) 高斯模式上做加权模仿学习——传统方法现在可以工作了

这种先选模式再在模式内优化的策略与人类决策相似：先决定大方向（模式），再微调执行（模式内优化）。

---

## 5. 思想与方法

### 指导原则

LOM 的核心思想是：将多模态问题分解为两层决策——在模式层面做离散选择（选哪个高斯分量），在动作层面做连续优化（在该高斯分量内做加权模仿学习）。

### 方法框架：Hyper-MDP

LOM 引入了一个新颖的**Hyper-Markov Decision Process (H-MDP)** 框架，在标准 MDP 之上增加一层抽象：

M_H = <S, A, P, r, gamma, Omega, A_H, P_H, r_H>

新组件：
- Omega = {phi_1, ..., phi_M}: M 个高斯模式（对应 GMM 的各分量）
- A_H = {1, ..., M}: 超动作空间，索引各模式
- P_H(s_{t+1} | s_t, u_t): 选择模式 u_t 后的期望转移：E_{a~phi_u}[P(s_{t+1}|s_t, a)]
- r_H(s_t, u_t): 选择模式 u_t 后的期望奖励：E_{a~phi_u}[r(s_t, a)]

在这个框架下，agent 的决策分为两步：
1. 超策略 zeta: S -> A_H 选择模式（离散决策）
2. 模式内策略 phi_u: S -> A 选择动作（连续决策）

组合策略：pi_zeta(a|s) = sum_{u in A_H} zeta(u|s) * phi_u(a|s)

### 核心组件一：Hyper Q-function

定义：Q_H^zeta(s, u) = 在状态 s 选择模式 u 然后遵循超策略 zeta 的期望回报

关键性质 (Proposition 1)：Q_H^zeta(s, u) = E_{a~phi_u(.|s)}[Q^{pi_zeta}(s, a)]

即：超 Q 值 = 从该高斯模式中采样动作的 Q 值期望。这建立了模式级评估与动作级 Q 值的桥梁。

### 核心组件二：Greedy Mode Selection

行为超策略 zeta_b：zeta_b(u|s) = alpha_u(s)（GMM 的混合系数，反映原始数据中的模式偏好）

贪婪超策略 zeta_g：zeta_g(s) = argmax_u Q_H^{zeta_b}(s, u)（选 Q 值最高的模式）

**Theorem 1 (模式选择保证改进)**：V^{pi_zeta_g}(s) >= V^{pi_b}(s)，对所有状态 s。即：选择最优模式得到的组合策略不差于原始行为策略。

### 核心组件三：模式内加权模仿学习

在选定的单一模式下，LOM 进一步做加权模仿学习。模仿对象不再是整个行为策略 pi_b，而是单一高斯模式 pi_zeta_g(即 phi_{zeta_g(s)})。

目标：最大化 pi 相对于 pi_zeta_g 的期望改进，受 KL 散度约束：

pi*(a|s) = (1/Z(s)) * pi_zeta_g(a|s) * exp(A^{pi_zeta_g}(s, a) / beta)

最终加权模仿学习目标：
argmax_pi E_{s~d_pib, a~pi_zeta_g}[exp(A^{pi_zeta_g}(s,a)/beta) * log pi(a|s)]

**关键**：这里的动作 a 是从 pi_zeta_g（单一高斯模式！）中采样的，而非从多模态行为策略中采样。这解决了传统加权模仿学习方法在多模态数据上的问题。

**Theorem 2 (两步改进保证)**：V^{pi_L}(s) >= V^{pi_zeta_g}(s) >= V^{pi_b}(s)。LOM 实现了两步改进：模式选择 + 模式内优化。

**Theorem 3 (改进下界)**：V^{pi_L}(s) - V^{pi_zeta_g}(s) >= (1/(1-gamma)) * eta_hat(pi_L) - (A_max/(1-gamma)) * sqrt(D_KL(d_piL || d_pi_zeta_g)/2)。改进由优势加权项驱动，由分布偏移惩罚约束。

### 因果机制总结

| 问题 | LOM 的解决机制 | 证据强度 |
|------|---------------|----------|
| 多模态导致 unimodal 策略取平均 | MDN+GMM 精确建模多模态，Hyper Q 选最优模式 | Fig.2, Table 1-2 |
| 生成式方法建模完整分布太复杂 | 只学一个模式，计算简单 | 消融实验, Fig.4 |
| 加权模仿学习在多模态上失效 | 在单一模式下做加权模仿 | Theorem 1-3, Table 1 |
| OOD 动作外推 | 在选定模式的支撑集内做 KL 约束优化 | Theorem 3 的下界分析 |

### 创新点判断

- 真正新颖的：Hyper-MDP 框架 + Hyper Q-function 用于模式选择；先选模式再在模式内优化的两阶段范式
- 借用的：MDN/GMM 行为策略建模（Bishop 1994）；加权模仿学习公式（AWR/AWAC）
- 重组的：将模式选择 (GMM 分解) 与加权模仿学习结合起来，首次证明只需要学一个模式

---

## 6. 算法与伪代码

### 算法名称

**LOM (Learning on One Mode)** - 加权模仿学习 + 模式选择

### 训练算法 (Algorithm 1)

`
输入：离线数据集 D, 高斯分量数 M
输出：目标策略 pi_theta

1.  # Phase 1: 学习 MDN 行为策略
2.  for i = 1 to I_M:
3.      最小化负对数似然 L(rho) = -E[log pi_rho(a_t | s_t)]
4.      更新 rho (MDN 参数，固定供后续使用)

5.  # Phase 2: 联合学习 Q 函数 + Hyper Q + 目标策略
6.  for i = 1 to I_G:
7.      采样 transitions tau = {s_t, a_t, r_t, s_{t+1}, a_{t+1}} ~ D
8.      # 学习行为策略的 Q 函数
9.      最小化 TD error: L(psi) = E[(Q_psi(s_t, a_t) - (r_t + gamma*Q_psi-(s_{t+1}, a_{t+1})))^2]
10.     # 学习行为超策略的 Hyper Q-function
11.     最小化: L(phi) = E_{s_t, u~Uniform(A_H)}[(Q_phi(s_t, u) - E_{a~phi_u}[Q_psi(s_t, a)])^2]
12.     # 学习目标策略（加权模仿学习）
13.     选最优模式: u = argmax_u Q_phi(s_t, u)
14.     从最优模式采样: a_hat_t ~ phi_u(s_t)
15.     获取模式均值: bar_a_t = phi_u_mu(s_t)
16.     估计优势: A(s_t, a_hat_t) = Q_psi(s_t, a_hat_t) - Q_psi(s_t, bar_a_t)
17.     更新目标策略: min -E[exp(A(s_t, a_hat_t)) * log pi_theta(a_hat_t | s_t)]
18.     # 软更新 target Q 网络
19.     if i mod update_delay == 0: psi- = rho*psi- + (1-rho)*psi
`

### 关键设计细节

**MDN 行为策略建模**：
- 网络输出 {z_mu_i, z_sigma_i, z_alpha_i} for i=1..M
- mu_i(s) = z_mu_i, sigma_i(s) = exp(z_sigma_i)
- alpha_i(s) = softmax(z_alpha_i)
- 训练到收敛后参数固定

**Hyper Q-function 学习**：
- Q_phi 输入: [state_dim + M] (state + one-hot 模式索引)
- 输出: M 维向量（每个模式的 Q 值）
- 目标：Q_phi(s, u) 接近该模式下动作 Q 值的期望 E_{a~phi_u}[Q_psi(s,a)]
- 使用从 uniform(A_H) 采样的模式索引训练（而非仅训练最优模式），确保覆盖所有模式

**优势估计**：A(s, a_hat) = Q_psi(s, a_hat) - Q_psi(s, bar_a)，其中 bar_a 是模式的均值（而非 V(s)），用作 baseline

### 重要超参数

| 超参数 | 含义 | 默认值/选择 |
|--------|------|-----------|
| M | GMM 高斯分量数 | {2,5,10,15,20}，通过实验选择 |
| beta | 优势权重温度 | 5 |
| C | 优势权重截断 | 50 |
| rho | Polyak 平均系数 | 0.995 |
| update_delay | Target 网络更新频率 | 2 |

### 理论与实践差距

- Theorem 1-3 假设能准确获得 Hyper Q-function 和 Q-function，实际中用神经网络近似，存在估计误差
- MDN 的模式分解质量直接影响后续所有步骤——如果 GMM 不能正确分离模式，模式选择将失效
- 行为策略建模 (MDN) 需要先训练到收敛并固定，无法在线适应数据分布变化

---

## 7. 实验与消融

### 实验设置

**环境与数据集**：
- D4RL MuJoCo: HalfCheetah, Hopper, Walker2d (medium, medium-replay, medium-expert)
- Fetch 机器人: FetchReach, FetchPush, FetchPickAndPlace (高度多模态数据集)

**Baselines**：

| 类别 | 方法 |
|------|------|
| 加权模仿学习 (unimodal) | OneStep, AWAC, TD3BC |
| 保守方法 | CQL |
| 多模态建模 | LAPO (VAE+advantage), DMPO (混合确定性策略), WCGAN, WCVAE |

**评估**：4 个随机种子，D4RL normalized score (0=random, 100=expert)。

### 主要结果

#### D4RL 基准 (Table 1)

- **LOM 在 15 个任务中的 12 个取得最优**
- 尤其在 medium 数据集上优势显著：Hopper-medium: LOM 100.8 vs AWAC 59.8, CQL 53.0, TD3BC 59.3
- 在多模态方法中 (WCGAN, WCVAE, LAPO, DMPO)，LOM 全面领先：Hopper-medium LOM 100.8 vs WCVAE 89.0 vs LAPO 51.6
- 在 medium-expert 上，LOM 与 WCGAN/WCVAE 接近（这些数据集多模态性较弱，模式选择优势不显著）

#### 高度多模态数据集 (Table 2, 图2)

在 Fetch 任务上（每个状态有 4 个有效动作，指向不同目标）：

| 任务 | BC | MDN | AWAC | WCGAN | WCVAE | LOM |
|------|-----|------|------|-------|-------|-----|
| FetchReach | 14.3 | 17.1 | 32.4 | 42.3 | 40.8 | **47.2** |
| FetchPush | 11.8 | 18.9 | 26.8 | 41.2 | 41.8 | **44.3** |
| FetchPick | 3.6 | 8.1 | 22.7 | 29.2 | 30.5 | **34.2** |

**LOM 平均优于 baselines 53%**。图2 直观展示：BC 学出平均动作（指向中心，命中率为0）；MDN 重建了4个模式；AWAC 倾向最优模式但生成 OOD 动作；LOM 精准聚焦最优模式。

### 消融与敏感性分析

**高斯分量数 M 的影响 (Fig.3)**：
- M=1 时 LOM 退化为 unimodal → 无法处理多模态
- medium-replay 上性能随 M 增加持续提升（更多模式 → 更精细的分解）
- full-replay 上 M=10 后饱和
- M 过大 (>=20) 时每个模式标准差过小 (Fig.4d-e)，导致学习域过窄，模式坍缩到单点

**对比图4**：M=2时模式太宽覆盖 OOD；M=10时分离合理；M=50时最优模式坍缩为一点。M 的合理范围是 {5, 10, 15}。

### 实验解读

**证明了什么**：
- LOM 的 one-mode 哲学在多模态数据上确实优于全分布建模方法
- 模式选择 + 模式内优化两步改进真实有效（而非仅靠复杂的生成式建模）

**仅暗示了什么**：
- M 的最优值与数据集多模态程度相关，论文讨论了 bump hunting 等自动选择方法但未实现
- LOM 在轻度多模态数据上优势减弱（medium-expert），但至少不差于 baselines

**弱点**：
- 未在 image-based 环境或随机延迟环境中测试
- MDN 的行为策略建模质量是木桶短板——若 GMM 不能正确分离模式，整个 pipeline 失效
- 未与扩散模型方法 (Diffusion-QL) 直接比较（仅在 related work 中讨论）
- Fetch 任务是作者自己构建的多模态数据集，非标准 benchmark

---

## 8. 展望

### 对 ORL 研究者的启发

1. Less is more 在多模态离线 RL 中是成立的：完整建模多模态分布既复杂又不必要，LOM 证明了只需找准最优模式即可，这对资源受限场景（如机器人端侧部署）尤为重要
2. 模式选择 (Mode Selection) 作为一个独立的子问题值得更多关注：LOM 的 Hyper-MDP 框架将模式选择形式化为一个离散决策问题，这个抽象层面的思考可能推广到其他场景（如分层 RL 的 subgoal selection）
3. GMM/MDN 在离线 RL 中作为行为策略建模工具的潜力被低估：大多数方法跳过了精确的行为策略建模，直接使用生成式模型或正则化。LOM 展示了精确的行为策略建模（通过 MDN）如何 enable 后续的模式选择
4. 两层优化的设计模式：先做离散的模式级决策，再做连续的动作级优化，这种分而治之的策略可以在不牺牲性能的前提下大幅降低复杂度
5. Hyper Q-function 满足 Bellman-like 方程（论文 Section 7 提及），这暗示可以直接用 Bellman backup 学习超 Q 值，可能进一步提升效率

### 局限性与开放问题

1. MDN 行为策略建模质量是关键瓶颈：如果 GMM 不能正确分离模式（模式数错误、模式重叠严重），后续的模式选择将不可靠。目前 M 仍需人工调节
2. Hyper Q-function 的学习依赖 MC 回归 (E_a[Q(s,a)]) 而非 Bellman backup，信息利用效率有限。论文承认直接 Bellman 学习可能更好，但有 extrapolation error 风险
3. 模式固定后无法在线调整：行为策略建模 (MDN) 先训练后固定，意味着如果某些模式在数据集中未被充分表示，LOM 永远无法发现它们
4. 仅在连续动作空间测试：LOM 的 GMM 假设要求动作空间是连续的，在离散动作空间上不适用
5. 未解决模式数量的自动选择：文中提到 bump hunting 和 peak finding 可以估计 M，但未实现

### 可追问题点

1. Hyper Q-function 的 Bellman 学习：能否为 Hyper Q-function 建立可靠的 Bellman backup（例如用 in-sample learning 或 conservative regularization），使其能从更少数据中学习？（动机：论文已指出方向但未探索，可大幅提升效率）
2. 模式数量的自适应选择：能否设计一种在线方法，根据数据多模态程度动态调整 M？（动机：M 是当前最大的人为超参，bump hunting 等方法可能实现自动化）
3. LOM 与扩散模型的结合：能否用扩散模型（而非 GMM）作为行为策略建模，再利用 mode-seeking 的性质做类似 one-mode 的选择？（动机：扩散模型表达能力更强，但 LOM 的哲学可能使之更高效）

---

## Links

- **论文页面**：ICLR 2025 接收论文，未能从论文/公开来源确认具体 OpenReview 链接
- **arXiv**：未能从论文/公开来源确认
- **代码**：论文未提供代码链接，未能从公开来源确认官方代码仓库
- **通讯作者**：Giovanni Montana (华威大学 / Alan Turing Institute), g.montana@warwick.ac.uk
- **项目主页**：未能从论文/公开来源确认

---

> **Disclaimer**: 本笔记基于论文全文（18页）深度阅读生成。作者信息受限于论文本身提供的内容，未能进行网络验证。部分分析（尤其是展望部分）包含笔者的研究判断，非论文原始主张。
