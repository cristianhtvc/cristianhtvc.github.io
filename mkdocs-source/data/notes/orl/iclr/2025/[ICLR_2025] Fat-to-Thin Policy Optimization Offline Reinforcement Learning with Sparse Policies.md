---
title: "Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies"
description: "一篇研究连续稀疏策略在离线 RL 中支撑不匹配问题，并提出 Fat-to-Thin 两阶段优化框架的论文。"
tags:
  - Offline RL
  - Sparse Policies
  - q-Gaussian
  - Safety-critical RL
  - Policy Optimization
---

# Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies

| Field | 内容 |
|---|---|
| Title | Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies |
| Year | 2025 |
| Source | ICLR 2025 Poster |
| Authors | Lingwei Zhu, Han Wang, Yukie Nagai |
| Affiliations | University of Tokyo；University of Alberta |
| Tags | Offline RL, sparse continuous policy, q-Gaussian, Tsallis RL, two-stage actor-critic, safety-critical control |

**一句话概括**：本文提出 Fat-to-Thin Policy Optimization，用无限支撑的“胖”重尾策略从离线数据学习，再把知识迁移给有限支撑的“瘦”稀疏策略，从而解决稀疏连续策略在离线 RL 中因数据动作落到当前策略支撑外而导致 log-likelihood 崩溃的问题。

**我对这篇论文的定位**：这篇论文切入了 ORL 中一个很少被系统讨论的策略分布问题：很多离线算法默认 Gaussian 策略有全空间非零概率，因此不会遇到 $\log \pi(a|s)=-\infty$；但在医疗剂量、安全控制等场景，我们恰恰希望策略能对危险动作给出严格零概率。FtTPO 的贡献在于把“从数据中学习”和“部署安全稀疏策略”分给两种策略，避免稀疏策略直接吃离线动作带来的支撑不匹配。

## 1. 第一作者相关信息

**Lingwei Zhu** 是 University of Tokyo 相关研究者，论文邮箱为 `lingwei4@ualberta.ca`，与 Han Wang 共同一作。其近期研究围绕 Tsallis regularization、q-exponential / q-Gaussian policy、重尾与稀疏策略优化展开。

从研究轨迹看，Lingwei Zhu 关注的是“非 Gaussian 策略分布如何改变强化学习的探索、安全和离线学习性质”。在这条线上，q-Gaussian 很关键，因为它用一个参数 $q$ 统一了 Gaussian、重尾和稀疏分布。本文将该分布族引入离线 RL，并指出稀疏策略会带来独特的 out-of-support action 问题。

| Year | Title | Venue/Source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2023 | Generalized Munchausen RL using Tsallis KL divergence | NeurIPS / related line | Tsallis RL | q/Tsallis 正则化的前置路线 |
| 2024 | Offline RL with Tsallis Regularization | TMLR / related line | Offline RL + Tsallis | 与 q-exponential weighting 有直接关系 |
| 2024 | q-Exponential Family for Policy Optimization | arXiv | q-Gaussian policy | 本文 q-Gaussian 策略族的技术背景 |
| 2025 | Fat-to-Thin Policy Optimization | ICLR | Sparse-policy Offline RL | 本文主贡献 |

**研究方向综合**：第一作者的研究不是简单替换策略分布，而是在问策略分布的支撑、尾部厚度和稀疏性如何影响 RL 算法稳定性与安全性。FtTPO 是这一路线在离线 RL 场景中的系统化推进。

**信息来源**：论文首页署名和代码链接、论文引用、OpenReview/公开 arXiv 线索；作者完整论文表需以个人主页或 DBLP 为准。

## 2. 研究问题

本文研究的是**离线 RL 中如何学习连续稀疏策略**。

稀疏连续策略指某些动作区域概率严格为零，而其他动作区域仍可随机采样。它与 Gaussian 策略的根本差异是：Gaussian 对任意动作都有非零概率；稀疏策略可以明确排除危险动作。

问题出现在离线学习中。许多 offline RL / actor-critic 方法要最大化数据动作或期望动作在当前策略下的 log-likelihood：

$$
\mathcal{L}_{ForwardKL}(\phi)
=
\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_D}
[-\log \pi_\phi(a|s)].
$$

如果 $\pi_\phi$ 是 Gaussian，则任意数据动作 $a$ 都有定义。但如果 $\pi_\phi$ 是稀疏策略，数据动作可能落在当前支撑外：

$$
\pi_\phi(a|s)=0,\quad \log \pi_\phi(a|s)=-\infty.
$$

这会导致训练数值崩溃。本文的具体目标是：在只使用固定离线数据 $\mathcal{D}$ 的条件下，学习一个可部署的稀疏连续策略，同时避免 out-of-support action 引起的 log-probability failure。

假设设置：

- 连续控制 MDP。
- 固定离线数据集 $\mathcal{D}$，行为策略为 $\pi_D$。
- 策略分布为 q-Gaussian 家族。
- 单智能体、model-free、offline-only。
- 目标任务包括安全关键 treatment simulation 和 D4RL MuJoCo。

## 3. 背景知识

**离线 RL 与行为约束**：离线 RL 只能使用历史数据，常见算法会让学习策略不要离开数据分布太远。AWR/AWAC/IQL/XQL 等方法常使用 advantage-weighted regression，核心形式是对数据动作做加权行为克隆。

**为什么 Gaussian 默认安全但不够安全**：Gaussian 有无限支撑，训练时不会遇到 $\log 0$，但部署时也永远不会对危险动作给严格零概率。在医疗剂量等任务中，“极小概率”也可能不够，因为危险动作最好永远不被选。

**稀疏策略**：有限支撑分布能把某些动作区域概率设为严格 0。它适合安全场景，但在离线训练中会遇到数据动作和当前支撑不重叠的问题。

**q-exponential 与 q-Gaussian**：

q-exponential 定义为：

$$
\exp_q(x)=
\begin{cases}
\exp(x), & q=1,\\
[1+(1-q)x]_+^{\frac{1}{1-q}}, & q\ne 1.
\end{cases}
$$

其中 $[u]_+=\max(u,0)$。当 $q<1$ 时，输入低于阈值会被截断为 0，产生稀疏性。

q-Gaussian 可写为：

$$
\pi_{\mathcal{N}_q}(a|s)
\propto
\exp_q\left(
-\frac{(a-\mu(s))^2}{2\sigma(s)^2}
\right).
$$

关系如下：

| $q$ | 分布类型 | 支撑性质 |
|---|---|---|
| $q=1$ | Gaussian | 无限支撑 |
| $q>1$ | Heavy-tailed q-Gaussian | 无限支撑，尾部更厚 |
| $q<1$ | Sparse q-Gaussian | 有限支撑，支撑外概率严格为 0 |

**Two-stage actor-critic**：一些方法将策略分成 proposal policy 和 actor policy。proposal 负责产生候选动作或吸收有偏奖励，actor 再学习更适合部署的策略。FtTPO 继承这个思想，但把两阶段赋予“胖策略学数据、瘦策略负责部署”的含义。

| Term | 朴素含义 | 在本文中的作用 |
|---|---|---|
| Sparse policy | 对部分动作严格零概率 | 用于安全部署 |
| Fat policy | 无限支撑重尾策略 | 负责从离线数据学习 |
| Thin policy | 有限支撑稀疏策略 | 负责最终交互/部署 |
| OOS action | 当前策略支撑外的数据动作 | 导致 $\log \pi=-\infty$ |
| RAR | 随机动作替换 | 既有 ad hoc workaround |
| q-exp weight | 可截断低优势动作的权重 | 让 proposal 更关注好动作 |

**为什么这篇文章的动机成立**：如果安全任务需要严格排除危险动作，Gaussian 策略天生不理想；但直接使用稀疏策略又会破坏离线训练。因此需要一个框架既利用无限支撑策略的训练稳定性，又输出有限支撑策略的部署安全性。

## 4. 问题分析

论文首先在 Section 3 指出稀疏策略离线学习的核心失败模式。

Forward KL / AWR 类目标通常需要评估数据动作：

$$
\mathcal{L}(\phi)=
\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_D}
[-w(s,a)\log \pi_\phi(a|s)].
$$

当 $\pi_\phi$ 是稀疏策略，数据动作可能落在策略支撑外。Figure 1 左图用红色行为分布和蓝色稀疏策略展示：红色数据分布的一部分不在蓝色支撑内，因此 log-likelihood 不可用。

论文讨论了两类临时解决方案。

1. **Random Action Replacement (RAR)**：把支撑外动作替换成当前策略支撑内最近的采样动作。Figure 1 中图展示了这个思路。但高维空间中很难采到靠近支撑边界的动作，维度灾难严重。
2. **Reverse KL**：改为从当前策略采样动作，再计算到行为策略的 KL：

$$
\mathcal{L}_{ReverseKL}(\phi)=
\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\phi}
[\log\pi_\phi(a|s)-\log\pi_D(a|s)].
$$

这样不会采到支撑外动作，但稀疏策略采样集中在自身 mode 附近，更新慢且容易性能差。

Figure 1 右图在 HalfCheetah medium-expert 上显示 FtTPO 优于 RAR 和 ReverseKL，支持作者关于 ad hoc trick 不足的判断。

这部分分析揭示了一个此前被 Gaussian 掩盖的问题：**离线 RL 策略分布的支撑性质会改变算法可训练性**。不是所有策略分布都能直接塞进现有 offline actor loss。

## 5. 思想与方法

FtTPO 的指导原则是：**不要让稀疏策略直接面对离线数据动作；先用无限支撑的 fat proposal 从数据中学，再让 sparse thin actor 从 proposal 中学**。

方法包含两种策略：

- Fat proposal policy $\pi_\phi$：重尾或 Gaussian，无限支撑，能安全计算所有离线动作的 log-likelihood。
- Thin actor policy $\pi_\theta$：稀疏 q-Gaussian，有限支撑，是最终部署策略。

Fat proposal 的训练目标是加权 log-likelihood：

$$
\mathcal{L}_{Fat}(\phi)=
\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_D}
[-w(s,a)\log\pi_\phi(a|s)].
$$

Thin actor 通过 reverse KL 从 proposal 学习：

$$
\mathcal{L}_{Thin}(\theta)=
\mathbb{E}_{s\sim\mathcal{D}}
\left[
D_{KL}(\pi_\theta(\cdot|s)\Vert \pi_\phi(\cdot|s))
\right].
$$

实际中作者使用低方差无偏 KL estimator，并在 actor 每次更新前把 proposal mean 拷贝给 actor mean，以减少稀疏 KL 的数值不稳定。

q-Gaussian 实例化：

- Fat policy：$q=2$，重尾、无限支撑。
- Thin policy：$q=0$，稀疏、有限支撑。

权重函数也使用 q-exponential advantage：

$$
w(s,a)=
\exp_q\left(\frac{Q(s,a)-V(s)}{\tau}\right),\quad q<1.
$$

因为 $q<1$ 时 q-exp 会截断低优势动作，所以 proposal 学数据时并不是无差别模仿所有动作，而是过滤掉明显差的动作。

方法的因果机制：

1. Fat policy 有无限支撑，避免数据动作 OOS。
2. Fat policy 重尾，能覆盖更广动作区域并吸收离线数据。
3. Thin policy 不直接评估离线动作，而是从 fat policy 采样学习，因此不会遇到 $\log 0$。
4. Thin policy 截断 fat policy 的尾部，得到更安全、更集中的稀疏行为。

新颖性在于把支撑不匹配明确识别为离线稀疏策略学习的核心问题，并提出 fat-to-thin 的结构化解决方案。

## 6. 算法与伪代码

算法名：**Fat-to-Thin Policy Optimization (FtTPO)**。

**Algorithm 1：q-Gaussian 初始化**

1. 输入 $q_f>1$ 和 $q_s<1$。
2. 初始化 fat proposal $\pi_\phi=\mathcal{N}_{q_f}(\mu_\phi,\Sigma_\phi)$。
3. 初始化 thin actor $\pi_\theta=\mathcal{N}_{q_s}(\mu_\theta,\Sigma_\theta)$。
4. 返回两种策略。

**Algorithm 2：q-Gaussian 采样**

1. 采样 $u_1,u_2\sim Uniform(0,1)^N$。
2. 用 Generalized Box-Muller Method 计算标准 q-Gaussian 样本。
3. 通过 $\mu+\Sigma^{1/2}z$ 得到目标 q-Gaussian 样本。

**Algorithm 3：FtTPO**

1. 输入离线数据 $\mathcal{D}$、训练步数 $T$、温度 $\tau>0$、权重参数 $q_w<1$。
2. 初始化 fat proposal 和 thin actor。
3. 对每个训练 step：
   1. 从数据集采样状态 $s$ 和行为动作 $a$。
   2. 用标准 critic 过程得到 $Q_{\omega_t}(s,a)$ 和 $V_{\nu_t}(s)$。
   3. 计算 q-exponential advantage weight：
      $w(s,a)=\exp_{q_w}((Q(s,a)-V(s))/\tau)$。
   4. 更新 fat proposal，最小化
      $-\mathbb{E}[w(s,a)\log\pi_\phi(a|s)]$。
   5. 从 thin actor $\pi_\theta$ 采样动作 $b$。
   6. 将 proposal mean 拷贝到 actor mean，稳定 KL 学习。
   7. 更新 thin actor，最小化 $\pi_\theta$ 到 $\pi_\phi$ 的 reverse KL 低方差估计。
4. 输出 thin actor 作为最终部署策略。

关键实现点：

- 论文默认 thin $q=0$，fat $q=2$。
- 使用 GBMM 采样 q-Gaussian。
- critic / value 的训练细节放在附录，整体类似优势加权离线 actor-critic。
- D4RL 实验中训练 1M steps，作者报告 FtTPO 因维护两个 policy 网络，平均训练成本约为普通单策略方法的两倍。

## 7. 实验与消融

**实验一：安全关键 treatment simulation**

- 目的：验证 FtTPO 是否能学习稀疏且安全的治疗策略。
- Baselines：IQL、XQL、SQL 等。
- Figure 2：FtTPO 获得最高 score；只有 FtTPO 学到了既稀疏又随机的策略，并集中在较小动作带上。SQL 因用 Gaussian 近似稀疏策略而塌缩成类似 delta 的行为，其他方法随机性过大。
- Figure 3：展示前 400 次更新中的策略演化，fat proposal 先学习宽分布，thin actor 逐步截断尾部形成稀疏集中策略。

**实验二：D4RL MuJoCo**

- 环境：HalfCheetah、Hopper、Walker2d。
- 数据：medium、medium-replay、medium-expert，共 9 个数据集。
- Baselines：IQL、InAC、TAWAC、AWAC、XQL、SQL，以及 TAWAC-HT 等相关变体。
- Figure 4：FtTPO 在 MuJoCo 上与强 Gaussian baseline 竞争并多处占优，说明稀疏策略并不必然牺牲性能。
- Figure 5：HalfCheetah medium-expert 上展示 actor 和 proposal 的策略演化，thin actor 通过去尾部形成更集中策略。

**消融实验**

- Figure 6：比较 FtTPO-SPOT、TAWAC-HT、FtTPO-SG 与 FtTPO。
- FtTPO-SPOT：用更复杂的 SPOT actor loss 替换简单 KL actor，整体没有明显优势，只在少数环境略好。
- TAWAC-HT / proposal only：重尾 proposal 本身已经强，但 thin actor 不劣于 proposal，说明稀疏输出不必损失性能。
- FtTPO-SG：把 proposal 换成 Squashed Gaussian 后整体明显更差，说明 heavy-tailed fat policy 是重要组件。
- Figure 7：展示 FtTPO actor 相比 IQL+Gaussian 更少触及危险动作边界，说明稀疏策略提供了更原则化的安全排除机制。
- Figure 8 / Appendix Figure 15：进一步展示不同变体学习曲线和策略演化。

**实验解释边界**：

FtTPO 的结果说明“稀疏策略可以离线学且性能不差”，但仍不是形式化安全保证。安全主要来自奖励和有限支撑策略，而支撑边界是否真正排除所有危险动作取决于状态表征、奖励设计和数据覆盖。

## 8. 展望

**对 ORL 研究者的启发**：

1. 策略分布的支撑性质是离线 RL 算法设计的一等公民，不应默认 Gaussian。
2. 稀疏策略有安全优势，但需要避免直接对离线动作做 log-likelihood。
3. 重尾 proposal 和稀疏 actor 的分工是处理探索覆盖与安全部署的自然方式。
4. q-Gaussian 为研究 heavy-tailed、Gaussian、sparse policy 提供统一参数化。
5. Advantage weighting 也可以稀疏化，用 q-exp 过滤低优势动作。

**局限与开放问题**：

1. 两个策略网络带来约两倍计算开销。
2. q 值默认选择 $q_f=2,q_s=0$，不同任务是否需要自适应选择仍不清楚。
3. 安全性不是形式化约束，只是通过稀疏支撑和奖励间接实现。
4. 高维动作空间下稀疏支撑的形状是否足够表达复杂安全集合，仍需研究。
5. 真实医疗或机器人安全数据上的验证还不足。

**后续研究想法**：

1. **状态自适应稀疏度**：让 $q_s(s)$ 随状态变化，在高风险状态更稀疏，在低风险状态更随机。
2. **稀疏策略与安全约束结合**：用 CBF/reachability 学习支撑边界，让有限支撑有形式化安全含义。
3. **多模态 fat-to-thin**：用 mixture q-Gaussian proposal 处理多模态离线数据，再蒸馏成多支撑稀疏 actor。

## Links

- Paper page: https://openreview.net/forum?id=SRjzerUpB2
- Code: https://github.com/lingweizhu/fat2thin
- arXiv: 未能从论文 PDF 正文确认本文 arXiv 链接
- Related q-exponential paper: https://arxiv.org/abs/2408.07245
- First author / DBLP search: https://dblp.org/search?q=Lingwei%20Zhu
