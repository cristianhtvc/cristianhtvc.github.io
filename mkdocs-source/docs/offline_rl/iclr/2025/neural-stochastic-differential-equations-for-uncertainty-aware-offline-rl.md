---
title: "Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL"
description: "提出 NUNO，用 Neural SDE 的漂移项注入物理先验、扩散项估计距离感知不确定性，以缓解 model-based offline RL 的 model exploitation。"
tags:
  - Offline RL
  - Model-Based RL
  - Neural SDE
  - Uncertainty Estimation
  - Conservative Policy Learning
---

# Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL

<div class="paper-hero">
<p class="paper-meta">Cevahir Koprulu, Franck Djeumou, Ufuk Topcu · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based RL</span><span>Neural SDE</span><span>Uncertainty Estimation</span><span>Conservative Policy Learning</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Neural Stochastic Differential Equations for Uncertainty-Aware, Offline RL |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published: 2025-01-22, Last Modified: 2025-04-13 |
| Authors | Cevahir Koprulu, Franck Djeumou, Ufuk Topcu |
| Affiliations | The University of Texas at Austin; Rensselaer Polytechnic Institute |
| Tags | Offline Model-Based RL; Neural SDE; Epistemic Uncertainty; Distance-Aware Uncertainty; Rollout Truncation; MuJoCo |

**一句话概括**：NUNO 用一个 Neural SDE 同时学习动力学和不确定性：漂移项承载刚体动力学等物理先验，扩散项拆分出数据内随机性和数据外距离感知不确定性，再用该不确定性惩罚奖励并截断模型 rollout，从而缓解 offline model-based RL 的 model exploitation。

**我对这篇论文的定位**：它是 model-based offline RL 中“更可靠 world model + 更合理 uncertainty penalty”的工作。它不是单纯换一个 dynamics model，而是指出 ensemble uncertainty 在 OOD 区域可能 distance-unaware，导致 MOPO/TATU 的保守项不准；NUNO 用 Neural SDE 的扩散项构造可微、平滑、有界、距离感知的不确定性估计，在低质量数据集上优势尤其明显。

## 1. 第一作者相关信息

**Cevahir Koprulu** 是 UT Austin 博士生，论文中与 Franck Djeumou 共同通讯/共同一作，导师团队为 Ufuk Topcu 组。公开主页显示其研究兴趣包括 reinforcement learning、control、safe autonomy 和 learning-enabled systems。Franck Djeumou 现为 RPI faculty，本文继承了其此前关于 Neural SDE 进行不确定性建模的工作。

Cevahir Koprulu 的研究轨迹集中在安全控制、强化学习与形式化/动力学建模交叉：一条线是安全强化学习和 reachability/safety filter，另一条线是把物理结构和不确定性估计注入 learned dynamics。NUNO 是这条线在 offline RL 上的系统化应用。

| 年份 | 论文 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL | ICLR 2025 Poster | Offline Model-Based RL | 本文；Neural SDE + uncertainty penalty |
| 2025 | Safety-Prioritizing Curricula for Constrained Reinforcement Learning | ICLR 2025 Poster | Constrained / Safe RL | 同属安全强化学习与约束决策方向 |
| 2023 | Risk-Aware Curriculum Generation for Heavy-Tailed Task Distributions | UAI 2023 | Curriculum RL / Risk | 关注风险感知任务分布，与安全泛化相关 |
| 2023 | Reward-Machine-Guided, Self-Paced Reinforcement Learning | UAI 2023 | Reward Machine / Curriculum RL | 体现其用任务结构辅助 RL 的研究线 |

**主要方向概括**：安全 RL、控制理论、学习动力学模型、物理先验和 uncertainty-aware decision making。出版表以作者主页、OpenReview 与论文引用可核验信息为准。

## 2. 研究问题

Model-based offline RL 的基本流程是：从离线数据 $D$ 学一个 dynamics model，在模型里 rollout 生成 synthetic data，然后用 RL 优化策略。它比纯 model-free offline RL 更能泛化到数据外区域，但也更容易出现 **model exploitation**：策略主动跑到模型不准但预测回报偏高的区域。

已有方法如 MOPO、MOReL、COMBO、TATU+MOPO 通过 uncertainty penalty 或 rollout truncation 约束策略。但论文认为，主流 deep probabilistic ensemble 的不确定性估计并不天然距离感知：一个点离训练数据很远时，ensemble disagreement 不一定大；这会让保守项在真正 OOD 的地方失效，或在不该保守的地方过度保守。

本文的设置是连续控制 offline model-based RL，数据集是 D4RL/NeoRL MuJoCo。目标是学习一个 dynamics model $\hat T$ 和一个 uncertainty estimator，使策略训练时既能利用模型泛化，又能被拉回离线数据支持附近。

## 3. 背景知识

**Model exploitation**：如果 learned model 在某些 OOD 状态动作上预测错误，策略优化会把这些错误当成真实机会利用，结果在真实环境中性能崩溃。offline setting 中没有在线交互纠错，所以这个问题尤其严重。

**Aleatoric vs epistemic uncertainty**：aleatoric uncertainty 是环境本身随机性，即使无限数据也存在；epistemic uncertainty 是模型没见过足够数据导致的不确定性，随数据覆盖增加而减少。offline RL 中真正需要惩罚的是 epistemic uncertainty，因为它标识数据外区域。

**Neural SDE**：随机微分方程写作

$$
ds=f_\theta(s,a)dt+\Sigma_\phi(s,a)\circ dW.
$$

$f_\theta$ 是漂移项，表示确定性动力学趋势；$\Sigma_\phi$ 是扩散项，表示随机扰动强度；$dW$ 是 Brownian motion 增量。Neural SDE 用神经网络参数化这些项。

**距离感知不确定性**：一个合理 epistemic estimator 应在训练数据附近低，在远离数据时高且有界，并且最好可微，以便和连续控制模型、数值积分、策略优化结合。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Neural SDE | 用 SDE 建模连续时间随机动力学 | NUNO 的 world model |
| Drift $f_\theta$ | 状态变化的确定性趋势 | 注入 MuJoCo 刚体动力学先验 |
| Diffusion $\Sigma_\phi$ | 随机扰动/不确定性强度 | 同时承载 aleatoric 与 epistemic uncertainty |
| Distance-aware estimator $\eta_\phi$ | 距训练数据越远值越大 | 作为 reward penalty 和 rollout truncation 指标 |
| MOPO penalty | 用不确定性惩罚模型奖励 | NUNO 继承并替换 uncertainty source |
| TATU truncation | 不确定性累计过大则截断 rollout | NUNO 用 $\eta_\phi$ 自适应截断 |

**为什么动机成立**：低质量数据集如 random/low 数据包含大量次优行为，但覆盖可能较广；高质量 expert 数据回报高但覆盖窄。一个好的 offline model-based 方法应区分“数据内随机性”和“数据外未知性”。ensemble heuristic 往往混淆二者；Neural SDE 的扩散结构提供了更自然的分解接口。

## 4. 问题分析

论文首先在 Section 1-3 指出现有保守 model-based offline RL 的隐患：它们通常假设 uncertainty estimator 是 admissible error estimator，但实践中用 ensemble disagreement 替代，缺少对数据距离的可靠刻画。TATU 还用训练数据上的单步不确定性来设 rollout 阈值，而 policy rollout 更长，误差会累积。

Section 4.1 提出距离感知估计器 $\eta_\phi(s,a)$。一个简化理论目标是让 $\eta$ 逼近 query point 到数据中心/数据支持的距离。Lemma 1 说明该目标的最优解是凸函数，并且负梯度 $-\nabla_{s,a}\eta$ 指向训练数据凸包内部。直觉是：如果 reward penalty 使用 $\eta$，策略为了少受惩罚，会被推向数据凸包。

核心公式如下：

$$
\bar{\eta}_{\phi}
=
\arg\min_{\eta}
\mathbb{E}_{(s,a)\sim D}
\left[
\mathbb{E}_{(s',a')\sim \mathrm{Uniform}(S\times A)}
\left(\eta(s',a')-\|(s,a)-(s',a')\|^2\right)
\right].
$$

论文随后指出，单纯到中心点的距离不足以处理多簇数据，因此引入两个训练约束：局部强凸性损失 $L_{\mathrm{sc}}$ 让 $\eta$ 离开数据点时上升；零梯度损失 $L_{\mathrm{grad}}$ 让数据点附近成为局部低值区域。

Section 5.2 和 Figure 3b/3c 验证：训练过程中，NUNO 给 random actions 的不确定性最高，给 dataset actions 的不确定性最低，learned policy actions 随训练逐渐接近 dataset uncertainty。这支持了“估计器能区分 OOD 与 in-distribution”的诊断。

## 5. 思想与方法

方法名是 **NUNO: Neural SDE for UNcertainty-aware Offline RL**，另有 **NUNOR** 变体会同时学习 reward dynamics。

指导原则是：用一个结构化 dynamics model 解决两件事。第一，漂移项通过物理先验提高多步预测质量；第二，扩散项通过距离感知估计提供更可信的保守信号。

NUNO 的 Neural SDE 写作：

$$
ds=f_\theta(s,a)dt+
\left(\sigma_\phi(s,a)+h_\phi(\eta_\phi(s,a))\right)\circ dW.
$$

其中 $\sigma_\phi$ 学 aleatoric uncertainty；$\eta_\phi$ 是 distance-aware epistemic estimator；$h_\phi$ 是有界单调变换，论文使用 scaled sigmoid，使得离数据很远时 diffusion 高但不发散。

对 MuJoCo 刚体系统，作者把状态拆成位置和速度，并把漂移项设计为：

$$
f_\theta^{\mathrm{pos}}(s,a)=s_{\mathrm{vel}},
\quad
f_\theta^{\mathrm{vel}}(s,a)=G_\theta(s_{\mathrm{vel}})a+H_\theta(s)s_{\mathrm{vel}},
\quad
f_\theta=[f_\theta^{\mathrm{pos}},f_\theta^{\mathrm{vel}}]+f_\theta^{\mathrm{res}}.
$$

含义：位置变化等于速度，这是最小物理先验；加速度由可学习项和残差建模。

策略训练时，NUNO 继承 MOPO/TATU：

$$
\tilde R(s,a)=R(s,a)-\lambda_{\mathrm{pen}}\eta_\phi(s,a),
$$

并在 rollout 中计算累计不确定性

$$
T=\sum_{t=0}^{h}\eta_\phi(s_t,a_t).
$$

若 $T$ 超过阈值 $\xi$，截断这条 synthetic trajectory。阈值通过训练数据上所有长度 $h$ 序列的 CVaR 自适应设置。

真正新的是 uncertainty estimator 的几何设计和与 Neural SDE diffusion 的结合；保守训练机制本身借鉴 MOPO 和 TATU。

## 6. 算法与伪代码

**NUNO 训练流程**

1. 输入离线数据 $D=\{(s_t,a_t,r_t,s_{t+1})\}$，选择 rollout length $h$、真实数据比例 $\beta$、CVaR 系数 $\alpha$、惩罚系数 $\lambda_{\mathrm{pen}}$。
2. 初始化 Neural SDE：漂移项 $f_\theta$、aleatoric diffusion $\sigma_\phi$、distance-aware estimator $\eta_\phi$、单调映射 $h_\phi$。
3. 从数据集中采样长度 $H$ 的状态动作序列。
4. 用 Euler-Maruyama 对 Neural SDE 积分，近似多步序列负对数似然 $L_{\mathrm{data}}$。
5. 同时计算局部强凸性约束 $L_{\mathrm{sc}}$、零梯度约束 $L_{\mathrm{grad}}$ 和强凸系数正则 $L_\mu$。
6. 优化总目标：

$$
\min_{\theta,\phi}
\mathbb{E}_{D}
\left[
\lambda_{\mathrm{data}}L_{\mathrm{data}}
+\lambda_{\mathrm{sc}}L_{\mathrm{sc}}
+\lambda_{\mathrm{grad}}L_{\mathrm{grad}}
+\lambda_{\mu}L_{\mu}
\right].
$$

7. 训练策略时，在 Neural SDE 中 rollout，使用悲观奖励 $\tilde R$。
8. 若 rollout 累计 $\eta_\phi$ 超过 CVaR 阈值 $\xi$，提前截断。
9. 将 synthetic data 与真实数据按比例混合，用 SAC 式 policy optimization 更新策略。

**实现细节**：Appendix C.4 使用 JAX；Neural SDE 用 Euler-Maruyama；训练序列 horizon 为 2；$\eta_\phi$ 两层 64 hidden swish；$\sigma_\phi$ 两层 256 hidden tanh；drift 主要网络三层 256 swish；batch size 128。NUNO 的主要超参数为真实数据比例 $\beta$、rollout length $h$、CVaR 系数 $\alpha$、penalty $\lambda_{\mathrm{pen}}$；论文固定 $\beta=0.05$，搜索 $h\in\{5,10,15,20\}$、$\alpha\in\{0.9,0.95,0.98,0.99,1.0\}$、$\lambda_{\mathrm{pen}}\in\{0.001,0.1,1\}$。

## 7. 实验与消融

**D4RL MuJoCo**：12 个任务，HalfCheetah/Hopper/Walker2d 乘以 random、medium、medium-replay、medium-expert。对比 EDAC、MOBILE、MOPO、TATU+MOPO、COMBO、MOReL、RAMBO。Table 1 显示 NUNO 平均 human-normalized score 为 83.8，NUNOR 为 82.4，均高于 MOBILE 80.0 和 EDAC 76.0。random 数据上优势最大：例如 hopper-random NUNO 73.2，MOBILE 31.9，MOPO 16.7；walker2d-random NUNO 27.7，MOPO 4.2。

**NeoRL MuJoCo**：9 个任务，HalfCheetah/Hopper/Walker2d 乘以 low/medium/high，每个 1000 trajectories。Table 2 显示 NUNO 平均 70.6，NUNOR 68.0，MOBILE 60.7，CQL 56.1，EDAC 50.7。NUNO/NUNOR 在 low、medium、high 多数任务达到最高或接近最高。

**Uncertainty 是否有效**：Figure 3b/3c 显示 learned policy actions 的 uncertainty 随训练下降并接近 dataset actions，而 random actions 始终高；这支持 distance-aware estimator 的判别能力。

**Model exploitation 分析**：Figure 4 比较 NUNO、NUNOR、TATU+MOPO、MOPO 在 learned model rollout 中的模型奖励和真实奖励。NUNO 构造的 pessimistic MDP 相对不那么过度保守，同时模型预测更可靠。Figure 5 和 Appendix Figure 8 显示 Neural SDE 在长 horizon 预测误差上优于 probabilistic ensemble，尤其在 OOD rollout 中更明显。

**消融**：Appendix D 比较 NUNO 与 NUNOal，后者用 aleatoric uncertainty $\sigma_\phi$ 而不是 distance-aware epistemic $\eta_\phi$ 做 penalty/truncation。Table 4 显示，在 medium-expert 低覆盖数据上，NUNO 明显优于 NUNOal：halfcheetah-medium-expert 为 97.0 vs 10.5，walker2d-medium-expert 为 113.2 vs 48.3。这说明真正关键的是 epistemic/distance-aware uncertainty，而不是一般 diffusion noise。

**实验解读**：NUNO 在低质量数据集提升显著，论文报告最多 93%；高质量数据集也能匹配或部分超过 SOTA，最多 55%。但它需要精心训练 Neural SDE 和调 $h,\alpha,\lambda_{\mathrm{pen}}$，工程复杂度高于直接使用 IQL/CQL。

## 8. 展望

对 ORL 研究者的启发：

1. model-based offline RL 的瓶颈不只是模型预测误差，而是“预测误差是否能被可靠 uncertainty 捕获”。
2. epistemic 与 aleatoric uncertainty 在 offline RL 中必须分开；惩罚 aleatoric noise 可能惩罚错对象。
3. 物理先验并不一定要很强；像“位置积分速度”这种弱先验也能改善 world model。
4. CVaR-based rollout threshold 是一个很实用的自适应保守性调节方式。

局限与开放问题：

1. 方法依赖连续控制和可用的状态结构分解；离散任务、图任务或高维视觉状态下是否适用仍未知。
2. Neural SDE 训练成本和调参复杂度较高。
3. 理论保证主要解释 penalty 的几何方向，没有给出完整 offline RL 性能界。
4. 真实机器人中 SDE rollout 的数值稳定性和 sim-to-real 校准还需要额外验证。

可能的后续研究：

1. **视觉-状态 NUNO**：结合 representation learning，使 $\eta_\phi$ 在 latent space 中保持距离感知。
2. **ensemble + Neural SDE 混合不确定性**：把 epistemic distance estimator 与 ensemble posterior 结合，提升复杂环境校准。
3. **offline-to-online safe fine-tuning**：用 NUNO uncertainty 作为在线探索 safety shield，减少部署初期风险。

## Links

- Paper page: [OpenReview](https://openreview.net/forum?id=hxUMQ4fic3)
- PDF: [OpenReview PDF](https://openreview.net/pdf?id=hxUMQ4fic3)
- arXiv: 未能从论文或公开来源确认独立 arXiv 链接
- Code: 论文脚注说明 code 包含在 supplement；未能从公开来源确认独立官方 GitHub 仓库
- First author profile: [Cevahir Koprulu](https://cevahir-koprulu.github.io/)

</article>
