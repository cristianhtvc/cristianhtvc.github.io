# [ICLR 2025] Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning

- 论文: Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning
- 会议/年份: ICLR 2025 Poster。已由论文 PDF、[OpenReview](https://openreview.net/forum?id=5Y9NT6lW21)、[arXiv](https://arxiv.org/abs/2503.05306) 和 [SNU Pure](https://snu.elsevierpure.com/en/publications/adversarial-policy-optimization-for-offline-preference-based-rein) 交叉确认。
- 一句话概括: 这篇论文把 offline preference-based RL 写成策略和奖励/价值模型之间的对抗博弈，用 trajectory-pair L1 deviation 和价值函数重参数化替代显式 confidence set，从而给出既可实现又有样本复杂度保证的 APPO。
- 我对这篇论文的定位: 这是 offline PbRL 理论和实践之间的一篇桥接论文。它不主打新的 reward model 架构，而是问一个更基础的问题: 怎样在没有真实奖励、只有离线偏好对的情况下，既保持 pessimism/conservatism，又避免 prior provable methods 中难以求解的置信集优化。放在 ORL 里看，它接近 model-based pessimistic offline RL 和 adversarially trained actor-critic 的交叉点，但目标函数由 preference data 驱动。

## 前置背景知识：读懂这篇文章需要先知道什么？

### 1. Preference-based RL 和 offline PbRL

标准 RL 假设环境每一步会给出 reward，算法直接最大化累计 reward。Preference-based RL 的动机是: 很多真实任务里 reward 难写，但人类更容易比较两段行为哪段更好。于是数据不是 $(s,a,r,s')$ 中的显式 reward，而是轨迹段对和偏好标签。

这篇论文的偏好数据形式是:

$$
D_{\mathrm{pref}}=\{(\tau_m^0,\tau_m^1,y_m)\}_{m=1}^{M},
\qquad
\tau_m^i=(s_1^{m,i},a_1^{m,i},\ldots,s_H^{m,i},a_H^{m,i}).
$$

其中 $y_m=1$ 表示 $\tau_m^1$ 比 $\tau_m^0$ 更受偏好。

论文假设偏好由真实但不可观测的 reward $r^\star$ 诱导:

$$
\mathbb{P}(y=1\mid \tau^0,\tau^1)
=\Phi\!\left(r^\star(\tau^1)-r^\star(\tau^0)\right),
\qquad
r^\star(\tau)=\sum_{h=1}^{H}r_h^\star(s_h,a_h).
$$

$\Phi$ 是单调 link function。若 $\Phi$ 是 sigmoid，就得到常见 Bradley-Terry-Luce preference model。直觉是: 两条轨迹的真实回报差越大，人类或标签器越可能偏好回报高的那条。

offline PbRL 比 online PbRL 更难，因为训练时不能继续向人提问，也不能靠在线探索纠正错误。论文还给定一个无标签轨迹对数据集:

$$
D_{\mathrm{traj}}=\{(\tau_n^0,\tau_n^1)\}_{n=1}^{N}.
$$

$D_{\mathrm{pref}}$ 用来学 reward model，$D_{\mathrm{traj}}$ 用来约束策略和模型不要偏离离线数据分布。

### 2. Offline RL 中的 distribution shift 和 conservatism

offline RL 只能从行为策略 $\mu$ 收集的旧数据学习。若目标策略 $\pi$ 选择了数据里很少出现的动作，Q function 或 reward model 可能对这些 out-of-distribution 区域过度乐观，导致策略利用模型错误而不是学到真实好行为。

因此 offline RL 常引入 pessimism/conservatism: 对数据覆盖不足的位置降低估值，或者让策略贴近数据支持。本文的关键问题是: 在 offline PbRL 中，理论上可证明的 conservatism 往往通过 confidence set 做最坏情况优化，但这些 confidence set 在一般函数逼近，尤其是 neural networks 下很难实际求解。

### 3. Trajectory concentrability

这篇论文不只要求 step-wise coverage，而是要求轨迹级覆盖。直观上，最优策略可能产生的整条轨迹，在行为策略 $\mu$ 的离线数据分布下也要有足够概率。

$$
\sup_{\tau}\frac{d_{\pi^\star}(\tau)}{d_{\mu}(\tau)}
\le C_{\mathrm{TR}}.
$$

含义: 如果 $C_{\mathrm{TR}}$ 很大，说明最优策略会走到离线数据几乎没见过的完整轨迹，offline PbRL 就很难。论文强调，Zhan et al. (2024a) 的下界说明 trajectory concentrability 对 offline PbRL 是必要的；这比标准 offline RL 中常见的 step-wise concentrability 更强。

### 4. 为什么本文的动机成立？

已有 provable offline PbRL 方法的套路是先从偏好数据构造 reward confidence set，再在 confidence set 里找最坏 reward 下仍然好的策略。这个思路理论上自然，因为保守性来自“对所有可能 reward 都不太差”。问题是，一旦 reward/transition/value 都用一般函数类表示，优化就会变成多层 constrained optimization 或 confidence-set width penalty，实际训练非常重。

本文的动机可以理解成一句话: 不显式构造“可信 reward 集合”，而是让一个 adversary reward/value model 在训练中主动寻找会让当前策略表现差的解释；策略必须在这个 adversary 下仍然变好。这等价于用博弈过程实现保守性。

### 术语表

| term | 中文直觉 | 在本文中的角色 |
|---|---|---|
| $D_{\mathrm{pref}}$ | 有标签偏好轨迹对 | 用 MLE 学初始 reward model $\hat{r}$ |
| $D_{\mathrm{traj}}$ | 无标签离线轨迹对 | 估计 trajectory-pair deviation 和离线期望 |
| $\mu$ | reference / behavior policy | 生成离线轨迹数据的策略 |
| $\pi$ | target policy | 要学习并输出的策略 |
| $r^\star$ | 真实但不可见的 reward | 偏好标签背后的假设目标 |
| $\hat{r}$ | 从偏好数据学出的 reward model | adversarial reward 偏离时的参照 |
| conservatism / pessimism | 对不确定区域保守 | 防止策略利用 reward/model 错误 |
| confidence set | 与数据一致的一组可能模型 | prior provable methods 的核心，但计算困难 |
| Stackelberg game | leader-follower 博弈 | 本文把策略和 reward/value 写成两方博弈 |
| induced reward | 由 value function 和 Bellman equation 反推出的 reward | 让 APPO 不需要真实 transition rollout |
| $C_{\mathrm{TR}}$ | 轨迹级 concentrability 常数 | 样本复杂度中控制离线覆盖难度 |

## 0. 作者信息与第一作者研究脉络

### 作者与单位

论文作者为 Hyungkyu Kang 和 Min-hwan Oh。PDF 署名单位均为 Seoul National University。SNU Pure 页面进一步记录该论文关联 Department of Data Science / Graduate School of Data Science。OpenReview profile 显示 Hyungkyu Kang 曾为 SNU undergrad student，当前 profile 中还列出 2026 起 Upstage intern 和 SNU research assistant；这些是 profile 当前信息，不一定代表 ICLR 2025 投稿时身份。

### 第一作者近期论文脉络

| 年份 | 论文 | Venue/source | topic tag | 与本文关系 |
|---|---|---|---|---|
| 2024 | Provably Efficient Policy Optimization with Rare Policy Switches | [OpenReview, ICLR 2024 withdrawn submission](https://openreview.net/forum?id=CrCMEV6oOI) | policy optimization theory, general function approximation | 体现第一作者对 provable policy optimization 和一般函数逼近的兴趣 |
| 2025 | Neural Dynamic Pricing: Provable and Practical Efficiency | [OpenReview, ICLR 2025 withdrawn submission](https://openreview.net/forum?id=YsOndItIxV) | dynamic pricing, oracle-efficient learning | 同样是“理论保证 + 实用算法”的取向，但不是 ORL |
| 2025 | Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning | [OpenReview](https://openreview.net/forum?id=5Y9NT6lW21), [arXiv](https://arxiv.org/abs/2503.05306) | offline PbRL, adversarial policy optimization | 当前论文 |
| 2026 | Offline Preference-Based Value Optimization | [OpenReview, ICLR 2026 Poster](https://openreview.net/forum?id=9cUdn8GKId) | offline PbRL, value alignment loss | 可看作本文后续方向: 继续追求 offline PbRL 的简单、稳定和可证明算法 |

综合来看，第一作者公开资料中最清晰的研究线索是: provable RL / policy optimization，以及把理论上可证明但难实现的方法改写成可用的深度 RL 训练目标。这个判断主要来自 OpenReview profile 和上述论文记录；未能从公开来源确认其完整 publication list。

## 1. 这篇文章旨在解决的问题是什么？

论文解决的是 offline preference-based reinforcement learning 中“可证明保守性”和“实际可训练性”之间的矛盾。

具体 setting 如下:

- 环境是 episodic MDP，长度为 $H$。
- 真实 reward $r^\star$ 不可见，只能通过 trajectory-pair preference feedback 间接学习。
- 数据分两类: 有标签偏好数据 $D_{\mathrm{pref}}$ 和无标签轨迹对数据 $D_{\mathrm{traj}}$，都来自 reference policy $\mu$。
- 算法目标是在不能在线交互、不能获得真实 reward 的情况下，输出接近最优的 policy $\hat{\pi}$。
- 理论部分考虑 reward class $\mathcal{R}$、transition class $\mathcal{P}$、value class $\mathcal{F}$ 的 general function approximation。

已有方法的失败模式有两层:

1. 经验 offline PbRL 方法，如 MR、PT、DPPO、IPL，可以在深度 RL benchmark 上跑，但通常缺少强统计保证。
2. 理论方法，如 FREEHAND-transition、Pace et al. 的 uncertainty penalty、Zhu et al. 的线性设定，可以给出 sample complexity，但往往依赖 explicit confidence set。一般函数逼近下，这些集合的约束优化很难实际求解。

因此本文不是只想提高 benchmark 分数，而是想证明: offline PbRL 中可以有一个既有样本复杂度保证、又能用常规 gradient-based optimization 训练的算法。

## 2. 文章如何对这个问题进行详细分析？

### 2.1 从 confidence-set pessimism 出发

论文先回顾 Zhan et al. (2024a) 风格的 model-based PbRL 目标。简化写法:

$$
\hat{\pi}
=\arg\max_{\pi}\min_{r\in\widehat{\mathcal{R}}}
\left[
V_{1,r}^{\pi}(s_1)-V_{1,r}^{\mu}(s_1)
\right],
$$

$$
\widehat{\mathcal{R}}
=\left\{
r\in\mathcal{R}^{H}:
\widehat{L}_{R}(r)\le \widehat{L}_{R}(\hat{r})+\beta
\right\}.
$$

解释:

- $\widehat{\mathcal{R}}$ 是和偏好数据相容的 reward confidence set。
- 内层 $\min$ 让算法面对最坏 reward，因此带来 conservatism。
- 外层 $\max$ 学一个即使在最坏 reward 下也优于 reference policy $\mu$ 的策略。

这个分析非常自然，但瓶颈在 $\widehat{\mathcal{R}}$。如果 $\mathcal{R}$ 是 neural network reward class，约束优化和 confidence set width 都不易求。

### 2.2 改写成 policy-reward Stackelberg game

本文提出不用显式维护 $\widehat{\mathcal{R}}$，而是让 reward adversary 对当前 policy 做最坏响应:

$$
\hat{\pi}
=\arg\max_{\pi}
\left[
V_{1,r_{\pi}}^{\pi}(s_1)-V_{1,r_{\pi}}^{\mu}(s_1)
\right],
$$

$$
r_{\pi}
=\arg\min_{r\in\mathcal{R}^{H}}
\left[
V_{1,r}^{\pi}(s_1)-V_{1,r}^{\mu}(s_1)+E(r;\hat{r})
\right].
$$

这里 $E(r;\hat{r})$ 是 reward 偏离 $\hat{r}$ 的惩罚。若没有这个惩罚，adversary 可以选任意坏 reward，学习就没有意义；惩罚项限制 adversary 只能在与偏好数据学到的 reward 相近的范围内作恶。

### 2.3 为什么不用 likelihood difference？

一个直觉选择是:

$$
E(r;\hat{r})=\widehat{L}_{R}(r)-\widehat{L}_{R}(\hat{r}).
$$

论文指出这个选择不能保证统计效率，因为 Stackelberg game 中缺少对 likelihood constraint 的合适 Lagrange multiplier。本文改用 trajectory-pair L1 loss:

$$
E(r;\hat{r})
=\mathbb{E}_{\tau^0,\tau^1\sim\mu}
\left|
\left[r(\tau^0)-r(\tau^1)\right]
-\left[\hat{r}(\tau^0)-\hat{r}(\tau^1)\right]
\right|.
$$

含义: reward adversary 可以改变每条轨迹的绝对分数，但不能随意改变两条轨迹之间的 return difference。偏好数据本身只识别相对好坏，所以这个度量比逐点 reward deviation 更贴近 PbRL 的可辨识对象。

### 2.4 用 performance difference lemma 消掉 rollout

APPO-rollout 版本假设可以知道真实 transition 或执行当前 policy 收集 rollout。真正的 offline 算法不能这样做。论文的关键观察是 performance difference lemma:

$$
\mathbb{E}_{\tau\sim\pi^t}\!\left[r(\tau)\right]
-\mathbb{E}_{\tau\sim\mu}\!\left[r(\tau)\right]
=
\sum_{h=1}^{H}
\mathbb{E}_{(s_h,a_h)\sim d_h^{\mu}}
\left[
(Q_{h,r}^{\pi^t}\circ\pi_h^t)(s_h)
-Q_{h,r}^{\pi^t}(s_h,a_h)
\right].
$$

重要点: 右边的期望是在 $d_h^\mu$ 下，即离线数据分布下，所以可用 $D_{\mathrm{traj}}$ 估计。为了做到这一点，论文把 reward optimization 重参数化为 value function optimization。

### 2.5 Induced reward model

给定 value function $f$、policy $\pi$ 和 transition $P$，可由 Bellman equation 反推出 reward:

$$
r_{h,P,\pi,f}(s,a)
=f_h(s,a)-P_h\!\left(f_{h+1}\circ\pi_{h+1}\right)(s,a),
\qquad
f_{H+1}=0.
$$

这叫 induced reward model。它使内层 adversary 不再直接搜索 reward $r$，而是搜索 value function $f$。这样 policy update 也可以直接使用 $f$，不再需要额外 policy evaluation oracle。

重参数化后的目标可简写为:

$$
\min_{f\in\mathcal{F}^{H}}
\sum_{h=1}^{H}
\mathbb{E}_{(s_h,a_h)\sim D_{\mathrm{traj}}}
\left[
(f_h\circ\pi_h^t)(s_h)-f_h(s_h,a_h)
\right]
+\lambda\,\widehat{E}_{D_{\mathrm{traj}}}(f;\widehat{P},\hat{r}).
$$

其中:

$$
\widehat{E}_{D_{\mathrm{traj}}}(f;\widehat{P},\hat{r})
=
\mathbb{E}_{(\tau^0,\tau^1)\sim D_{\mathrm{traj}}}
\left|
\left[
r_{\widehat{P},\pi^t,f}(\tau^0)
-r_{\widehat{P},\pi^t,f}(\tau^1)
\right]
-
\left[
\hat{r}(\tau^0)-\hat{r}(\tau^1)
\right]
\right|.
$$

这个分析把“保守 reward adversary”变成了“保守 value adversary”，并且所有项都能从离线数据估计。

## 3. 基于分析，文章使用了什么思想来解决这一问题？

核心思想可以概括为: 用对抗训练实现隐式 confidence-set pessimism。

更具体地说:

- prior theory: 在 confidence set 中取最坏 reward，让策略在所有可行解释下都稳健。
- APPO: 不显式构造 confidence set，而是训练一个 adversarial reward/value function，对当前 policy 产生悲观估计。
- trajectory-pair L1 loss: 限制 adversary 不能偏离 preference-implied reward differences 太远。
- value reparameterization: 用 Bellman equation 把 reward adversary 变成 value adversary，避免 offline setting 中不可用的 rollout 或真实 transition。

真正新颖的部分不是“adversarial training”本身，RL 中已有 RAMBO、ATAC 等相关思想；本文新意在于把它嵌入 offline PbRL，并且设计了可证明的 trajectory-pair loss 和 induced reward reparameterization。

## 4. 这一思想发展出了什么方法，以及如何用算法实现？

### 4.1 APPO-rollout: 理论构造的中间版本

Algorithm 1 假设可以 rollout 当前策略，步骤是:

1. 用 $D_{\mathrm{pref}}$ 通过 MLE 学 $\hat{r}$。
2. 对每轮 $t$，执行当前 policy $\pi^t$ 收集 $K_1$ 条轨迹。
3. 解 adversarial reward:

$$
r_t
=\arg\min_{r}
\left[
\mathbb{E}_{\tau\sim\mathrm{rollout}(\pi^t)}[r(\tau)]
-\mathbb{E}_{\tau\sim D_{\mathrm{traj}}}[r(\tau)]
+\lambda\,\widehat{E}_{D_{\mathrm{traj}}}(r;\hat{r})
\right].
$$

4. 用 Monte Carlo policy evaluation 估计 $\widehat{Q}^t$。
5. 用 TRPO-style exponentiated update 更新策略:

$$
\pi_h^{t+1}(a\mid s)
\propto
\pi_h^t(a\mid s)\exp\!\left(\eta\,\widehat{Q}_h^t(s,a)\right).
$$

APPO-rollout 用来说明 Stackelberg idea，但它不是真正 offline，因为需要 rollout。

### 4.2 APPO: unknown transition 下的主算法

Algorithm 2 的主流程是:

1. 用 $D_{\mathrm{pref}}$ 学 reward model $\hat{r}$。
2. 用 $D_{\mathrm{traj}}$ 学 transition model $\widehat{P}$。
3. 每轮在离线数据上优化 adversarial value function $f^t$。
4. 用 $f^t$ 更新 policy。
5. 输出多轮 policy 的 mixture:

$$
\bar{\pi}=\frac{1}{T}\sum_{t=1}^{T}\pi^t.
$$

关键是第 3 步。$f^t$ 的目标由两部分组成:

- 第一项类似 advantage under behavior data，让 value adversary 在当前 policy 附近施加保守压力。
- 第二项 trajectory-pair L1 loss，让 induced reward 不要偏离 $\hat{r}$ 的轨迹偏好差。

### 4.3 Practical APPO

理论 Algorithm 2 仍偏抽象。论文 Section 5 和 Appendix Algorithm 4 给出 deep PbRL 版本，使用 discounted MDP 和长度为 $L$ 的 trajectory segments。

Q function loss 分两块:

$$
L_{Q_i}=\lambda L_{\mathrm{adv},i}+E_i.
$$

其中:

$$
L_{\mathrm{adv},i}(B_{\mathrm{tup}})
=
\mathbb{E}_{(s,a)\sim B_{\mathrm{tup}}}
\left[
Q_i(s,\pi_{\theta}(s))-Q_i(s,a)
\right].
$$

直觉: 让当前 policy action 附近的 Q 相对更保守，离线行为动作附近相对更高，从而抑制 OOD action exploitation。

trajectory-pair term:

$$
E_i(B_{\mathrm{traj}})
=
\mathbb{E}_{(\tau^0,\tau^1)\sim B_{\mathrm{traj}}}
\left|
\left[r_i(\tau^0)-r_i(\tau^1)\right]
-\left[\hat{r}(\tau^0)-\hat{r}(\tau^1)\right]
\right|.
$$

这里 induced segment reward 近似为 TD error 之和:

$$
r_i(\tau)
=
\sum_{\ell=1}^{L}
\left[
Q_i(s_{\ell},a_{\ell})-\gamma V(s_{\ell+1})
\right].
$$

$V$ 的训练类似 clipped double Q target:

$$
L_V
=
\mathbb{E}_{s}
\left[
V(s)-\min_{i\in\{1,2\}}Q_{\bar{\theta}_i}(s_{\mathrm{next}},\pi_{\theta}(s_{\mathrm{next}}))
\right]^2.
$$

策略 loss:

$$
L_{\theta}
=
\mathbb{E}_{s}
\left[
Q_i(s,\pi_{\theta}(s))
-\alpha\log\pi_{\theta}(\pi_{\theta}(s)\mid s)
\right],
\qquad
i\sim\mathrm{Unif}\{1,2\}.
$$

注意论文写的是直接 optimize policy loss，形式类似 SAC 的 entropy-regularized actor update，但这里 Q 来自 APPO 的 adversarial value training。

### 4.4 重要实现细节

论文实验中:

- reward model: 3 个 ensemble fully-connected nets，每个 3 层 hidden 128，hidden ReLU，final Tanh，Adam lr 1e-3。
- Q/V/policy: 3 层 hidden 256。
- APPO critic/value lr 3e-4，policy lr 3e-5。
- batch: 256 transitions 和 16 trajectory pairs。
- target soft update: 0.001。
- discount factor: 0.99。
- 训练 250,000 gradient steps，每次约 3-4 小时，硬件为 RTX 3090。
- 代码公开在 [oh-lab/APPO](https://github.com/oh-lab/APPO)。

## 5. 这一方法为什么能解决问题？

### 5.1 方法组件到问题的对应关系

| 问题 | APPO 组件 | 机制 |
|---|---|---|
| reward 不可见，只能从偏好学 | $\hat{r}$ from $D_{\mathrm{pref}}$ | 将偏好对转成 reward difference 的统计估计 |
| confidence set 难优化 | Stackelberg adversary | 用内层 adversarial optimization 替代显式集合 |
| reward adversary 可能乱改 reward | trajectory-pair L1 loss | 限制 adversary 保持和 $\hat{r}$ 相近的轨迹差分 |
| offline 无法 rollout 当前 policy | performance difference lemma | 把 on-policy return difference 改写成 behavior distribution expectation |
| 不知道真实 transition | induced reward + $\widehat{P}$ / TD approximation | 用 value function 和 Bellman equation 反推 reward |
| policy 利用 OOD Q 误差 | $L_{\mathrm{adv}}$ conservative Q regularizer | 降低当前 policy action 相对行为动作的虚高估值 |

### 5.2 理论保证

论文 Theorem 4.1 在以下假设下给出 APPO sample complexity:

- reward realizability: $r^\star$ 在 reward class $\mathcal{R}$ 中。
- transition realizability: $P^\star$ 在 transition class $\mathcal{P}$ 中。
- value function class 足够大: 对任意 $r$ 和 $\pi$，对应 $Q_{r}^{\pi}$ 在 $\mathcal{F}$ 中，且有界。
- trajectory concentrability: $\sup_{\tau}d_{\pi^\star}(\tau)/d_{\mu}(\tau)\le C_{\mathrm{TR}}$。

定理的主界可粗略读成三类误差:

$$
\mathrm{Suboptimality}(\hat{\pi})
\lesssim
\mathrm{Err}_{\mathrm{reward}}(D_{\mathrm{pref}})
+\mathrm{Err}_{\mathrm{transition/value}}(D_{\mathrm{traj}})
+\mathrm{Err}_{\mathrm{policy}}(T).
$$

更具体的 $\epsilon$-optimal 设定为:

$$
T=\Theta\!\left(
\frac{R^2H^2\log|\mathcal{A}|}{\epsilon^2}
\right),
$$

$$
M=\Theta\!\left(
\frac{C_{\mathrm{TR}}^2\kappa^2H\log(|\mathcal{R}|/\delta)}{\epsilon^2}
\right),
$$

$$
N=\Theta\!\left(
\max\left\{
\frac{R^4H^5\log|\mathcal{A}|\log(H|\mathcal{F}|/\delta)}{\epsilon^4},
\frac{R^2H^2\log(H|\mathcal{P}|/\delta)}{\epsilon^2}
\right\}
\right).
$$

解释:

- $M$ 是有偏好标签的轨迹对数，随 $C_{\mathrm{TR}}^2$ 和 $\epsilon^{-2}$ 增长。
- $N$ 是无标签轨迹对数，包含一个 $\epsilon^{-4}$ 项，代价比 labeled data 更重。
- $T$ 是 policy optimization rounds。
- $\kappa$ 来自 link function 的斜率下界，偏好模型越不敏感，学习越难。

论文认为 APPO 和 FREEHAND-transition 在 labeled sample complexity 上匹配，但 FREEHAND-transition 的 confidence-set nested optimization 不可行；APPO 的代价是 unlabeled data bound 更松，却换来实际可实现性。

### 5.3 实验支持

实验使用 Choi et al. (2024) 的 Meta-World medium-replay 和 medium-expert datasets。偏好标签不是人工标注，而是用 ground-truth reward 生成: segment length 25，若两段 reward 差超过 12.5，则给二元偏好，否则给 $(0.5,0.5)$ tie label。

主表 Table 1 在 medium-replay 上比较 Oracle、MR、PT、DPPO、IPL 和 APPO。APPO 的 average rank 为 2.125，是 learned baselines 中最好；MR 为 2.316，PT 为 3.125，IPL 为 3.063，DPPO 为 4.375。APPO 在 button-press-topdown、dial-turn、button-press-topdown-wall、drawer-open 等任务上很强，但在 sweep、sweep-into、box-close 等任务上并非总是最优。因此更准确的结论是: APPO 在平均排名和若干任务上表现强，同时保持理论保证；不是对所有任务都显著碾压。

Figure 1 研究 $\lambda$。结论是 APPO 在一系列 $\lambda$ 上可学习，但合适 $\lambda$ 会提高稳定性。Figure 2 研究 preference feedback 数量，范围为 100 到 2000；论文报告 APPO 对偏好样本量较稳健，在部分任务中 100 个 preference samples 就能超过 Oracle。

Appendix Table 2 在 medium-expert 上只和 MR 比，APPO 在 dial-turn 的 500/1000 feedback 均高于 MR，在 sweep-into 500 feedback 略低于 MR、1000 feedback 高于 MR。

### 5.4 我的判断: 强处与薄弱处

强处:

- 目标很清楚: 用 adversarial training 替代 explicit confidence set。
- 技术关键点明确: trajectory-pair L1 loss 和 induced reward reparameterization。
- 理论与实现之间有真实连接，Practical APPO 不是完全另起炉灶。
- 实验选择了无法靠 incorrect reward 轻易“survival instinct”过关的数据集，这是合理的 benchmark 设计。

薄弱处:

- 理论依赖 realizability 和 trajectory concentrability，现实中都不容易验证。
- practical implementation 使用 discounted segment、TD approximation 和 neural networks，与理论 episodic finite-horizon 设置仍有 gap。
- 偏好标签由 ground-truth reward 合成，不是真实人类偏好。
- benchmark 主要是 Meta-World，广泛性有限。
- APPO 虽有较好 average rank，但并非每个任务都强，特别是 sweep/sweep-into 这类任务上结果混合。

## 研究启发与可追问点

### 对 ORL 研究的启发

1. Offline PbRL 的“可辨识对象”更接近 trajectory return difference，而不是逐点 reward。围绕 difference 设计 loss 可能比强行恢复 Markov reward 更自然。
2. Confidence-set pessimism 可以通过 adversarial training 隐式实现，这给其他 provable-but-intractable ORL 方法提供了改写方向。
3. Value/reward reparameterization 是连接 theory 和 deep implementation 的有效工具。只要 Bellman consistency 可用，就可能把难以观测的对象换成可训练 critic。
4. APPO 的 trajectory-level coverage 假设提醒我们: offline PbRL 比标准 offline RL 更依赖完整轨迹支持，数据收集策略质量可能比单步覆盖更重要。
5. 论文的实验说明 MR 仍是强 baseline。做 offline PbRL 新方法时，不能只和弱 reward-model pipeline 比。

### 局限和开放问题

1. 如果 $D_{\mathrm{traj}}$ 没有高质量轨迹，$C_{\mathrm{TR}}$ 很大，APPO 是否只会学到保守但低水平的 policy？
2. trajectory-pair L1 loss 对 preference noise、inconsistent preferences、非 Bradley-Terry 偏好是否稳健？
3. practical APPO 中 $\lambda$ 虽然是唯一算法超参，但仍直接控制 conservatism。能否自适应调节？
4. 合成偏好和人类偏好之间差异很大。APPO 在真实 human feedback 或 LLM/RLHF setting 中是否稳定？
5. induced reward 依赖 critic TD structure。若 critic 学坏，adversarial objective 是否会放大误差？

### 后续研究想法

1. Adaptive-lambda APPO: 用离线 OPE uncertainty 或 reward-difference calibration 动态调节 $\lambda$，减少每个任务手调 conservatism 的成本。
2. Preference-noise robust APPO: 将 trajectory-pair L1 换成对 label noise 更稳健的 distributional 或 trimmed objective，测试不一致人类偏好下的稳定性。
3. Sequence-model APPO: 用 Decision Transformer 或 diffusion planner 表示 policy，同时保留 APPO 的 adversarial value/reward regularizer，研究 trajectory-level support 和生成式策略之间的关系。

## 证据与来源

- 本地 PDF: `C:\Users\chenwy\OneDrive\ORL_papers\历年会议论文\ORL_2025_Papers\[ICLR_2025] Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning.pdf`
- 论文页面: [OpenReview](https://openreview.net/forum?id=5Y9NT6lW21), [arXiv](https://arxiv.org/abs/2503.05306), [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3469b211b829b39d2b0cfd3b880a869c-Paper-Conference.pdf)
- 作者/单位: [SNU Pure](https://snu.elsevierpure.com/en/publications/adversarial-policy-optimization-for-offline-preference-based-rein), [Hyungkyu Kang OpenReview profile](https://openreview.net/profile?id=~Hyungkyu_Kang1)
- 代码: [oh-lab/APPO](https://github.com/oh-lab/APPO)
