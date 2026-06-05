---
title: "ICLR 2025"
description: "ICLR 2025 离线强化学习论文笔记"
---

# ICLR 2025

<p class="lead">表示学习、强化学习、偏好学习与大模型方法交汇较多，是 ORL 新思想的重要来源。</p>

<section class="paper-list">
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="active-offline-reinforcement-learning-via-adaptive-imitation-and-in-sample-v-ensemble/">ACTIVE: Offline Reinforcement Learning via Adaptive Imitation and In-sample V-Ensemble</a></h2>
<p class="paper-meta">Tianyuan Chen, Ronglong Cai, Faguo Wu, Xiao Zhang · ICLR 2025</p>
<p>ACTIVE 用 V 函数集成抑制 in-sample offline RL 的初始价值误差传播，并用自适应克隆温度缓解过度保守。</p>
<div class="tag-row"><span>Offline RL</span><span>In-sample Learning</span><span>Implicit Q-Learning</span><span>Ensemble</span><span>Policy Extraction</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="adversarial-policy-optimization-for-offline-preference-based-reinforcement-learning/">Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning</a></h2>
<p class="paper-meta">详见笔记 · ICLR 2025</p>
<p>这篇论文把 offline preference-based RL 写成策略和奖励/价值模型之间的对抗博弈，用 trajectory-pair L1 deviation 和价值函数重参数化替代显式 confidence set，从而给出既可实现又有样本复杂度保证的 APPO。</p>
<div class="tag-row"><span>Offline RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="an-optimal-discriminator-weighted-imitation-perspective-for-reinforcement-learning/">An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning</a></h2>
<p class="paper-meta">Haoran Xu, Shuozhe Li, Harshit Sikchi, Scott Niekum, Amy Zhang · ICLR 2025</p>
<p>IDRL 将离线 RL 中的 Dual-RL 分布比学习解释为最优判别器加权模仿，并通过状态-动作访问分布修正与迭代数据过滤逼近专家分布。</p>
<div class="tag-row"><span>Offline RL</span><span>Dual RL</span><span>Imitation Learning</span><span>Distribution Ratio</span><span>Weighted BC</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="any-step-dynamics-model-improves-future-predictions-for-online-and-offline-reinforcement-l/">Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Haoxin Lin, Yu-Yan Xu, Yihao Sun, Zhilong Zhang, Yi-Chen Li, Chengxing Jia, Junyin Ye, Jiaji Zhang, Yang Yu · ICLR 2025</p>
<p>ADM 用任意步回溯的直接预测降低模型 rollout 的误差累积，并在离线 MBRL 中用单模型差异估计不确定性。</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based RL</span><span>Dynamics Model</span><span>Uncertainty</span><span>D4RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="contradiff-planning-towards-high-return-states-via-contrastive-learning/">ContraDiff: Planning Towards High Return States via Contrastive Learning</a></h2>
<p class="paper-meta">Yixiang Shan, Zhengbang Zhu, Ting Long, Qifan Liang, Yi Chang, Weinan Zhang, Liang Yin · ICLR 2025</p>
<p>ContraDiff 在扩散轨迹规划中引入按回报定义的对比约束，让低回报轨迹作为负样本参与训练，帮助策略远离低回报状态。</p>
<div class="tag-row"><span>Offline RL</span><span>Diffusion Planning</span><span>Contrastive Learning</span><span>Dataset Imbalance</span><span>Trajectory Generation</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="correcting-the-mythos-of-kl-regularization-direct-alignment-without-overoptimization-via-c/">Correcting the Mythos of KL-Regularization Direct Alignment without Overoptimization via Chi-Squared Preference Optimization</a></h2>
<p class="paper-meta">详见笔记 · ICLR 2025</p>
<p>> **论文标题**: Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization</p>
<div class="tag-row"><span>Offline RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="cross-domain-offline-policy-adaptation-with-optimal-transport-and-dataset-constraint/">Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint</a></h2>
<p class="paper-meta">Jiafei Lyu, Mengbei Yan, Zhongjian Qiao, Runze Liu, Xiaoteng Ma, Deheng Ye, Jingwen Yang, Zongqing Lu, Xiu Li · ICLR 2025</p>
<p>OTDF 用最优传输筛选源域转移，并用目标域数据集约束防止策略偏向错误动力学。</p>
<div class="tag-row"><span>Offline RL</span><span>Domain Adaptation</span><span>Optimal Transport</span><span>Dynamics Shift</span><span>Dataset Constraint</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="data-center-cooling-system-optimization-using-offline-reinforcement-learning/">Data Center Cooling System Optimization Using Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Xianyuan Zhan, Xiangyu Zhu, Peng Cheng, Xiao Hu, Ziteng He, Hanfei Geng, Jichao Leng, Huiwen Zheng, Chenhui Liu, Tianshun Hong, Yan Liang, Yunxin Liu, Feng Zhao · ICLR 2025</p>
<p>一篇将物理先验、T-symmetry 表征学习和离线 RL 结合起来并真实部署到数据中心冷却控制的工业 ORL 论文。</p>
<div class="tag-row"><span>Offline RL</span><span>Industrial Control</span><span>Data Center Cooling</span><span>Physics-informed RL</span><span>T-symmetry</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="diffusion-actor-critic-formulating-constrained-policy-iteration-as-diffusion-noise-regress/">Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Linjiajie Fang, Ruoxue Liu, Jing Zhang, Wenjia Wang, Bing-Yi Jing · ICLR 2025</p>
<p>DAC 将 KL 约束策略迭代转化为扩散噪声回归，用 soft Q-guidance 训练扩散目标策略并避免 OOD 动作。</p>
<div class="tag-row"><span>Offline RL</span><span>Diffusion Policy</span><span>Actor-Critic</span><span>Policy Regularization</span><span>Q-Ensemble</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="dof-a-diffusion-factorization-framework-for-offline-multi-agent-reinforcement-learning/">DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning</a></h2>
<p class="paper-meta">Chao Li, Ziwei Deng, Chenxing Lin, Wenqi Chen, Yongquan Fu, Weiquan Liu, Chenglu Wen, Cheng Wang, Siqi Shen · ICLR 2025</p>
<p>DoF 将 MARL 的 IGM 原则推广为 IGD 分布一致性原则，用噪声和数据因子化把集中式扩散模型拆成可去中心化执行的小模型。</p>
<div class="tag-row"><span>Offline MARL</span><span>Diffusion Models</span><span>Value Factorization</span><span>CTDE</span><span>Multi-Agent RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="efficient-online-reinforcement-learning-fine-tuning-need-not-retain-offline-data/">Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data</a></h2>
<p class="paper-meta">Zhiyuan Zhou, Andy Peng, Qiyang Li, Sergey Levine, Aviral Kumar · ICLR 2025</p>
<p>一篇系统研究 offline-to-online RL 中无离线数据保留微调问题，并提出 Warm-start RL 的论文。</p>
<div class="tag-row"><span>Offline-to-Online RL</span><span>Fine-tuning</span><span>No-retention RL</span><span>Catastrophic Forgetting</span><span>Warm-start RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="energy-weighted-flow-matching-for-offline-reinforcement-learning/">Energy-Weighted Flow Matching for Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Shiyuan Zhang, Weitong Zhang, Quanquan Gu · ICLR 2025</p>
<p>一篇将精确能量引导引入 flow matching 和 diffusion，并用于 Q 加权离线策略优化的论文。</p>
<div class="tag-row"><span>Offline RL</span><span>Flow Matching</span><span>Diffusion Models</span><span>Energy Guidance</span><span>QIPO</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="fat-to-thin-policy-optimization-offline-rl-with-sparse-policies/">Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies</a></h2>
<p class="paper-meta">Lingwei Zhu, Han Wang, Yukie Nagai · ICLR 2025</p>
<p>一篇研究连续稀疏策略在离线 RL 中支撑不匹配问题，并提出 Fat-to-Thin 两阶段优化框架的论文。</p>
<div class="tag-row"><span>Offline RL</span><span>Sparse Policies</span><span>q-Gaussian</span><span>Safety-critical RL</span><span>Policy Optimization</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="fewer-may-be-better-enhancing-offline-reinforcement-learning-with-reduced-dataset/">Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset</a></h2>
<p class="paper-meta">Yiqin Yang, Quanwei Wang, Chenghao Li, Hao Hu, Chengjie Wu, Yuhua Jiang, Dianyu Zhong, Ziyou Zhang, Qianchuan Zhao, Chongjie Zhang, Bo Xu · ICLR 2025</p>
<p>ReDOR 将离线 RL 数据子集选择建模为梯度近似与弱次模优化，用更少数据提升训练效率和策略表现。</p>
<div class="tag-row"><span>Offline RL</span><span>Dataset Selection</span><span>Coreset</span><span>Submodular Optimization</span><span>D4RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="ins-interaction-aware-synthesis-to-enhance-offline-multi-agent-reinforcement-learning/">INS: Interaction-aware Synthesis to Enhance Offline Multi-agent Reinforcement Learning</a></h2>
<p class="paper-meta">详见笔记 · ICLR 2025</p>
<p>title: "INS: Interaction-aware Synthesis to Enhance Offline Multi-agent Reinforcement Learning"</p>
<div class="tag-row"><span>Offline RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="latent-safety-constrained-policy-approach-for-safe-offline-reinforcement-learning/">Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Prajwal Koirala, Zhanhong Jiang, Soumik Sarkar, Cody Fleming · ICLR 2025 Poster</p>
<p>LSPC 先用 CVAE 从离线数据中学习一个保守安全策略与 latent safety constraint，再用 reward-Advantage Weighted Regression 在受限隐空间中寻找高回报动作，从而缓解 safe offline RL 中“过度保守低奖励”和“放松约束高风险”的矛盾。</p>
<div class="tag-row"><span>Safe Offline RL</span><span>Latent Safety Constraints</span><span>CVAE</span><span>Advantage Weighted Regression</span><span>Constrained MDP</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="learning-on-one-mode-addressing-multi-modality-in-offline-reinforcement-learning/">Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning</a></h2>
<p class="paper-meta">详见笔记 · ICLR 2025</p>
<p>> **深度阅读笔记** | ICLR 2025 Accepted Paper</p>
<div class="tag-row"><span>Offline RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="long-short-decision-transformer-bridging-global-and-local-dependencies-for-generalized-dec/">Long-Short Decision Transformer Bridging Global and Local Dependencies for Generalized Decision-Making</a></h2>
<p class="paper-meta">详见笔记 · ICLR 2025</p>
<p>> **论文标题**: Long-Short Decision Transformer: Bridging Global and Local Dependencies for Generalized Decision-Making</p>
<div class="tag-row"><span>Offline RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="m-3pc-test-time-model-predictive-control-using-pretrained-masked-trajectory-model/">M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model</a></h2>
<p class="paper-meta">Kehan Wen, Yutong Hu, Yao Mu, Lei Ke · ICLR 2025</p>
<p>用同一个预训练掩码轨迹模型在测试时完成动作提案、未来预测、回报评估与目标到达规划，从而把 BTM 从单步行为克隆器变成可规划的决策模型。</p>
<div class="tag-row"><span>Offline RL</span><span>Offline-to-Online RL</span><span>Model Predictive Control</span><span>Masked Trajectory Model</span><span>Robot Learning</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="model-risk-sensitive-offline-reinforcement-learning/">Model Risk-sensitive Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Gwangpyo Yoo, Honguk Woo · ICLR 2025</p>
<p>将金融中的模型风险引入风险敏感离线强化学习，用最坏可置信回报分布的风险替代单一估计风险，并用 critic ensemble 与 Fourier-IQN 实现。</p>
<div class="tag-row"><span>Offline RL</span><span>Risk-sensitive RL</span><span>Distributional RL</span><span>Model Risk</span><span>Spectral Risk Measures</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="model-based-offline-reinforcement-learning-with-lower-expectile-q-learning/">Model-based Offline Reinforcement Learning with Lower Expectile Q-Learning</a></h2>
<p class="paper-meta">Kwanyoung Park, Youngwoon Lee · ICLR 2025</p>
<p>LEQ 用 lower expectile regression 学习模型 rollout 的保守 λ-return，从而让 model-based offline RL 能在 AntMaze 等长时域任务上摆脱不可靠不确定性惩罚。</p>
<div class="tag-row"><span>Offline RL</span><span>Model-based RL</span><span>Expectile Regression</span><span>Lambda Return</span><span>AntMaze</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="model-based-rl-as-a-minimalist-approach-to-horizon-free-and-second-order-bounds/">Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds</a></h2>
<p class="paper-meta">Zhiyong Wang, Dongruo Zhou, John C. S. Lui, Wen Sun · ICLR 2025</p>
<p>证明最标准的 MLE 学模型加乐观/悲观规划，已经足以在在线和离线 RL 中得到近似 horizon-free 与 second-order 的理论界。</p>
<div class="tag-row"><span>Model-based RL</span><span>RL Theory</span><span>Offline RL</span><span>Horizon-free Bounds</span><span>Second-order Bounds</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="model-free-offline-reinforcement-learning-with-enhanced-robustness/">Model-Free Offline Reinforcement Learning with Enhanced Robustness</a></h2>
<p class="paper-meta">Chi Zhang, Zain Ulabedeen Farhat, George K. Atia, Yue Wang · ICLR 2025</p>
<p>提出 double-pessimism 原则，用 model-free Q-learning 同时处理离线数据不足和部署环境模型偏差。</p>
<div class="tag-row"><span>Offline RL</span><span>Robust RL</span><span>Model-Free RL</span><span>Pessimism</span><span>Sample Complexity</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="multi-level-certified-defense-against-poisoning-attacks-in-offline-reinforcement-learning/">Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein · ICLR 2025</p>
<p>用差分隐私的 outcomes guarantee 为离线 RL 数据投毒提供动作级与策略级双层认证防御。</p>
<div class="tag-row"><span>Offline RL</span><span>Certified Defense</span><span>Data Poisoning</span><span>Differential Privacy</span><span>Safety</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">精读 · 已整理</p>
<h2><a href="neural-stochastic-differential-equations-for-uncertainty-aware-offline-rl/">Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL</a></h2>
<p class="paper-meta">Cevahir Koprulu, Franck Djeumou, Ufuk Topcu · ICLR 2025</p>
<p>提出 NUNO，用 Neural SDE 的漂移项注入物理先验、扩散项估计距离感知不确定性，以缓解 model-based offline RL 的 model exploitation。</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based RL</span><span>Neural SDE</span><span>Uncertainty Estimation</span><span>Conservative Policy Learning</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="offline-rl-in-regular-decision-processes-sample-efficiency-via-language-metrics/">Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics</a></h2>
<p class="paper-meta">Ahana Deb, Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi · ICLR 2025</p>
<p>用形式语言度量和 Count-Min-Sketch 改进 RDP 离线强化学习的样本效率与空间复杂度。</p>
<div class="tag-row"><span>Offline RL</span><span>Regular Decision Processes</span><span>Non-Markovian RL</span><span>Automata Learning</span><span>Sample Complexity</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="offline-rl-with-smooth-ood-generalization-in-convex-hull-and-its-neighborhood/">Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood</a></h2>
<p class="paper-meta">Qingmao Yao, Zhichao Lei, Tianyuan Chen, Ziyue Yuan, Xuefan Chen, Jianxiang Liu, Faguo Wu, Xiao Zhang · ICLR 2025</p>
<p>用 CHN 安全区域和 Smooth Bellman Operator 缓解离线 RL 中对 OOD 动作的过度保守。</p>
<div class="tag-row"><span>Offline RL</span><span>OOD Generalization</span><span>Q Function</span><span>Smooth Bellman Operator</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="ohio-offline-hierarchical-reinforcement-learning-via-inverse-optimization/">OHIO: Offline Hierarchical Reinforcement Learning via Inverse Optimization</a></h2>
<p class="paper-meta">Carolin Schmidt, Daniele Gammelli, James Harrison, Marco Pavone, Filipe Rodrigues · ICLR 2025</p>
<p>提出 OHIO，用低层控制器结构反推出不可观测高层动作，把普通离线轨迹转化为可训练层级策略的数据集。</p>
<div class="tag-row"><span>Offline RL</span><span>Hierarchical RL</span><span>Inverse Optimization</span><span>Robotics</span><span>Network Optimization</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="online-to-offline-rl-for-agent-alignment/">Online-to-Offline RL for Agent Alignment</a></h2>
<p class="paper-meta">Xu Liu, Haobo Fu, Stefano V. Albrecht, Qiang Fu, Shuai Li · ICLR 2025</p>
<p>将已在线训练好的游戏智能体用少量离线人类行为数据对齐到目标偏好。</p>
<div class="tag-row"><span>Agent Alignment</span><span>Preference Learning</span><span>Online-to-Offline RL</span><span>Game AI</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="preference-elicitation-for-offline-reinforcement-learning/">Preference Elicitation for Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Alizée Pace, Bernhard Schölkopf, Gunnar Rätsch, Giorgia Ramponi · ICLR 2025</p>
<p>在完全离线设定下主动选择偏好查询轨迹，用 Sim-OPRL 结合悲观模型学习与乐观偏好采样。</p>
<div class="tag-row"><span>Offline RL</span><span>Preference Learning</span><span>Active Learning</span><span>Model-based RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="rtdiff-reverse-trajectory-synthesis-via-diffusion-for-offline-reinforcement-learning/">RTDiff: Reverse Trajectory Synthesis via Diffusion for Offline Reinforcement Learning</a></h2>
<p class="paper-meta">Qianlan Yang, Yu-Xiong Wang · ICLR 2025</p>
<p>RTDiff 通过反向合成轨迹进行离线数据增强，降低 forward synthesis 从已知区域走向 OOD 区域带来的过估计风险。</p>
<div class="tag-row"><span>Offline RL</span><span>Diffusion Models</span><span>Data Augmentation</span><span>Trajectory Synthesis</span><span>Distribution Shift</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="scalable-decision-making-in-stochastic-environments-through-learned-temporal-abstraction/">Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction</a></h2>
<p class="paper-meta">Baiting Luo, Ava Pettet, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay · ICLR 2025</p>
<p>L-MAP 通过 VQ-VAE 学习宏动作 latent，再用 MCTS 在离散隐空间中高效处理随机连续控制规划。</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based Planning</span><span>Temporal Abstraction</span><span>MCTS</span><span>VQ-VAE</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="scaling-offline-model-based-rl-via-jointly-optimized-world-action-model-pretraining/">Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining</a></h2>
<p class="paper-meta">Jie Cheng, Ruixi Qiao, Yingwei Ma, Binhua Li, Gang Xiong, Qinghai Miao, Yongbin Li, Yisheng Lv · ICLR 2025</p>
<p>JOWA 用共享 Transformer 同时预训练世界模型和动作价值模型，探索 Atari 多任务 offline model-based RL 的 scaling。</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based RL</span><span>World Model</span><span>Atari</span><span>Scaling</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="scrutinize-what-we-ignore-reining-in-task-representation-shift-of-context-based-offline-me/">Scrutinize What We Ignore: Reining In Task Representation Shift Of Context-Based Offline Meta Reinforcement Learning</a></h2>
<p class="paper-meta">Hai Zhang, Boyuan Zheng, Tianying Ji, Jinhang Liu, Anqi Guo, Junqiao Zhao, Lanqing Li · ICLR 2025</p>
<p>这篇 ICLR 2025 论文为 COMRL 交替优化建立性能改进解释，并指出任务表示漂移会破坏单调改进。</p>
<div class="tag-row"><span>Offline Meta RL</span><span>Context-Based Meta RL</span><span>Task Representation Shift</span><span>Theory</span><span>Mutual Information</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="semantic-temporal-abstraction-via-vision-language-model-guidance-for-efficient-reinforceme/">Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning</a></h2>
<p class="paper-meta">Tian-Shuo Liu, Xu-Hui Liu, Ruifeng Chen, Lixuan Jin, Pengyuan Wang, Zhilong Zhang, Yang Yu · ICLR 2025</p>
<p>VanTA 用 VLM 知识引导 VQ-VAE 技能发现，让离线层级 RL 的时序抽象更有语义且更高效。</p>
<div class="tag-row"><span>Offline RL</span><span>Temporal Abstraction</span><span>Vision-Language Models</span><span>Hierarchical RL</span><span>Skill Discovery</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="tackling-data-corruption-in-offline-reinforcement-learning-via-sequence-modeling/">Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling</a></h2>
<p class="paper-meta">Jiawei Xu, Rui Yang, Shuang Qiu, Feng Luo, Meng Fang, Baoxiang Wang, Lei Han · ICLR 2025</p>
<p>RDT 说明序列建模式 offline RL 在数据损坏下天然更稳，并用三种轻量技术增强 Decision Transformer。</p>
<div class="tag-row"><span>Offline RL</span><span>Robust RL</span><span>Decision Transformer</span><span>Data Corruption</span><span>Sequence Modeling</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="value-aligned-behavior-cloning-for-offline-reinforcement-learning-via-bi-level-optimizatio/">Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization</a></h2>
<p class="paper-meta">Xingyu Jiang, Ning Gao, Xiuhui Zhang, Hongkun Dou, Yue Deng · ICLR 2025</p>
<p>VACO 用双层优化学习样本权重，让行为克隆在保持数据支持的同时对齐预训练价值函数。</p>
<div class="tag-row"><span>Offline RL</span><span>Behavior Cloning</span><span>Bi-level Optimization</span><span>Value Alignment</span><span>D4RL</span></div>
</article>
<article class="paper-card">
<p class="eyebrow">速读 · 已整理</p>
<h2><a href="what-makes-a-good-diffusion-planner-for-decision-making/">What Makes a Good Diffusion Planner for Decision Making?</a></h2>
<p class="paper-meta">Haofei Lu, Dongqi Han, Yifei Shen, Dongsheng Li · ICLR 2025</p>
<p>一项系统拆解扩散规划器关键设计的 ICLR 2025 Spotlight 经验研究，并给出强基线 Diffusion Veteran。</p>
<div class="tag-row"><span>Offline RL</span><span>Diffusion Planning</span><span>Decision Making</span><span>Empirical Study</span><span>Diffusion Veteran</span></div>
</article>
</section>
