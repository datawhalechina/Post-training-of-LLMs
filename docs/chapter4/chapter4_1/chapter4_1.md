# 4.1 在线强化学习基础理论
在本节课中，你将学习关于 **在线强化学习（Online Reinforcement Learning, RL）** 的基础概念，包括：

* 方法概述
* 常见应用场景
* 高质量数据策划的原则


## 4.1.1 语言模型中的强化学习：在线 vs 离线

在强化学习中，针对语言模型有两种主要的学习方式：

### 在线学习（Online Learning）

模型在**实时生成新响应**的过程中不断学习：

1. 生成新的响应（Response）
2. 获取对应的奖励（Reward）
3. 使用这些响应与奖励来更新模型权重
4. 模型持续学习并优化生成的响应

### 离线学习（Offline Learning）

模型只从**预先收集的（prompt, response, reward）三元组**中学习：

* 不会在训练过程中生成新的响应。

> 因此，当我们提到 “在线强化学习（Online RL）”时，通常指的是 **在在线学习场景中应用的强化学习方法**。


## 4.1.2 在线强化学习的工作机制

在线强化学习通常让模型**自主探索更好的响应**。
其典型流程如下：

1. 准备一批 Prompt（输入提示）；
2. 将这些 Prompt 输入语言模型；
3. 模型生成对应的 Response；
4. 将 (prompt, response) 对送入 **奖励函数（Reward Function）**；
5. 奖励函数为每对 (prompt, response) 打分；
6. 获得 (prompt, response, reward) 三元组；
7. 使用这些数据来更新语言模型。

模型更新可采用不同方法，本课重点介绍两种：

* **PPO（Proximal Policy Optimization）**
* **GRPO（Group Relative Policy Optimization）**



## 4.1.3 奖励函数（Reward Function）的选择

在在线强化学习中，奖励函数的设计至关重要。常见有两种类型：

### 训练好的奖励模型（Reward Model）

* 收集多个模型响应，由人类进行偏好标注（选择更优的响应）。
* 使用这些人类偏好数据训练奖励模型。
* 奖励模型通过优化如下损失函数学习：

$$
  L = \log(\sigma(r_j - r_k))
$$

  其中：

  * 若人类认为响应 **j 优于 k**，则鼓励模型提升$r_j$，降低$r_k$。

**特点：**

* 通常基于已有的 Instruct 模型初始化；
* 通过大规模人类或机器生成偏好数据训练；
* 可应用于开放式任务，如聊天能力、安全性提升等；
* 但在“正确性导向”的任务（如代码、数学、函数调用）中可能不够精确。

---

### 可验证奖励（Verifiable Reward）

在“正确性导向”场景中，更推荐使用**可验证奖励**：

* **数学任务**：验证模型输出是否与标准答案匹配。
* **编程任务**：通过 **单元测试（Unit Tests）** 检验代码执行结果是否正确。

**特点：**

* 需提前准备真值（Ground Truth）或测试集；
* 准备成本较高，但奖励信号更精确可靠；
* 更适合训练**推理类模型**（Reasoning Models），如代码、数学领域。



## 4.1.4 两种主流的在线强化学习算法对比

### PPO（Proximal Policy Optimization）

PPO 是第一代 ChatGPT 所使用的在线强化学习算法。

#### 工作流程：

1. 输入一组查询（queries） ( $q$ )；

2. 通过 **策略模型（Policy Model）**（即语言模型本身）生成响应；

3. 响应被送入以下模块：

   * **参考模型（Reference Model）**：计算 KL 散度，限制模型不偏离原始分布；
   * **奖励模型（Reward Model）**：计算奖励；
   * **价值模型（Value Model）** 或 **评论者模型（Critic Model）**：为每个 Token 分配价值。

4. 使用 **广义优势估计（Generalized Advantage Estimation, GAE）**
   来计算每个 Token 的 **优势函数（Advantage）**，反映该 Token 的贡献。

#### PPO 的目标函数：

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\text{old}}}(O|q)} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})} A_t, \text{clip} \left( \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})}, 1 - \varepsilon, 1 + \varepsilon \right) A_t \right] \right]$$


**总结：**

* 每个 Token 拥有独立的优势值；
* 反馈粒度更细；
* 但需额外训练价值模型 → 占用更多 GPU 内存。



### GRPO（Group Relative Policy Optimization）

GRPO 由 DeepSeek 提出，用于优化大型语言模型的推理能力。

#### 工作流程：

1. 对每个 Prompt，模型生成多个响应 ( $O_1, O_2, ..., O_g$ )；
2. 对每个响应计算：

   * 奖励（Reward）
   * 与参考模型的 KL 散度；
3. 对同一组（Group）响应计算**相对奖励（Relative Reward）**；
4. 将相对奖励作为整个响应的优势值；
5. 使用此优势更新策略模型。

**主要区别：**

* 不再需要价值模型（Value Model）；
* 所有 Token 在同一响应中共享相同优势值；
* 更节省显存，但优势估计较粗糙。



## 4.1.5 PPO 与 GRPO 的比较总结

| 特征   | PPO                        | GRPO                         |
| ---- | -------------------------- | ---------------------------- |
| 优势估计 | 基于价值模型 (Value Model) 的精细估计 | 基于响应组的相对奖励 (Relative Reward) |
| 计算粒度 | 每个 Token 拥有独立优势            | 整个响应共享同一优势                   |
| 显存需求 | 较高（需训练 Critic）             | 较低（无 Critic）                 |
| 样本效率 | 高（样本利用率好）                  | 较低（需更多样本）                    |
| 奖励适配 | 适合连续或模型化奖励                 | 适合二元/可验证奖励                   |
| 应用场景 | 聊天、对齐、安全优化                 | 数学、代码、推理任务                   |



## 4.1.6 课程小结

在本课中，你学习了：

1. **在线强化学习（Online RL）** 与 **离线强化学习（Offline RL）** 的区别；
2. 奖励函数的两种设计方式：**奖励模型** 与 **可验证奖励**；
3. 两种关键算法的机制与对比：**PPO** 与 **GRPO**。

在下一节课中，我们将使用 **GRPO** 来提升 **指令模型（Instruct Model）** 的**数学能力**。
