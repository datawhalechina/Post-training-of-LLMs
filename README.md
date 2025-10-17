# Post-Training of LLMs 中文教程

## 项目简介
本项目围绕 DeepLearning.AI 出品的 Post-Training for LLMs 系列课程打造中文翻译与知识整理教程。我们提供课程内容翻译、知识点梳理和示例代码，旨在降低语言门槛，帮助学生、研究人员和开发者系统掌握大语言模型（LLM）后训练阶段的核心技术与实践方法。

**在线视频课程地址：** [DeepLearning.AI - Post-training of LLMs](https://www.deeplearning.ai/short-courses/post-training-of-llms/)

## 项目受众
- 对 LLM 优化与落地应用感兴趣的学习者。
- 希望深入理解并掌握模型后训练方法的同学与研究者。
- 计划结合后训练技术打造领域专用模型的团队与开发者。
- 正在寻找系统化学习资源的在校学生与入门者。

## 项目亮点
1. **权威课程，本土化翻译**：精准翻译 DeepLearning.AI 官方前沿课程，打破语言壁垒，为国内学习者提供原汁原味且易于理解的 LLM 后训练核心知识。
2. **系统梳理后训练核心技术**：聚焦 SFT、DPO、Online RL 等关键环节，将碎片知识系统化，帮助学习者构建从理论到实践的完整知识体系。
3. **理论与实践并重**：提供配套可运行的代码示例，强化动手能力，确保学习者不仅能“看懂”，更能“上手”，为开发领域专用模型打下基础。

## 项目规划
#### 1、目录
- [第1章](./docs/chapter1)
    - [1.1 课程介绍](./docs/chapter1/chapter1_1)
    - [1.2 后训练技术介绍](./docs/chapter1/chapter1_2)
- [第2章](./docs/chapter2)
    - [2.1 监督微调基础理论](./docs/chapter2/chapter2_1)
    - [2.2 监督微调实践](./docs/chapter2/chapter2_2)
- [第3章](./docs/chapter3)
    - [3.1 直接偏好优化基础理论](./docs/chapter3/chapter3_1)
    - [3.2 直接偏好优化实践](./docs/chapter3/chapter3_2)
- [第4章](./docs/chapter4)
    - [4.1 在线强化学习基础理论](./docs/chapter4/chapter4_1)
    - [4.2 在线强化学习实践](./docs/chapter4/chapter4_2/)
- [第5章](./docs/chapter5/)
#### 2、各章节负责人以及预估完成时间

| 章节             | 负责人     | 预估完成时间 |
| -------------- | ------- | ------ |
| 1.1 课程介绍       | 李柯辰     | 10.7   |
| 1.2 后训练技术介绍    | 李柯辰     | 10.7   |
| 2.1 监督微调基础理论   | 朱广恩     | 10.7   |
| 2.2 监督微调实践     | 王泽宇     | 10.7   |
| 3.1 直接偏好优化基础理论 | 王海洪     | 10.7   |
| 3.2 直接偏好优化实践   | 张宏历     | 10.7   |
| 4.1 在线强化学习基础理论 | 朱伯湘     | 10.7   |
| 4.2 在线强化学习实践   | 蔡煊琪，朱伯湘 | 10.7   |
| 5.1 总结         | 张宏历     | 10.7   |
#### 3、可预见的困难

- **技术理解与翻译准确性的平衡**
LLM后训练领域涉及大量前沿、晦涩的专业术语（如DPO、OnlineRL等）。在翻译和解释时，如何在保持原意准确的前提下，使其在中文语境中易于理解，是一大挑战。理解偏差或翻译生硬都会影响学习效果。
- **代码实践与环境的复现难题**
示例代码的成功运行严重依赖于特定的软件库版本、硬件环境（如GPU）和数据集。环境配置的微小差异都可能导致代码报错，极大增加学习者的实践门槛和挫败感。
- **课程迭代与更新压力**
 LLM领域技术迭代速度极快，原版课程内容可能会更新，新的算法和工具也会不断涌现。项目面临着需要持续跟进、同步更新翻译与代码的巨大压力，否则内容将迅速过时。




## 已完成的部分

| 章节             | 负责人     | 预估完成时间 | 状态  |
| -------------- | ------- | ------ | --- |
| 1.1 课程介绍       | 李柯辰     | 10.7   | ✅  |
| 1.2 后训练技术介绍    | 李柯辰     | 10.7   | ✅  |
| 2.1 监督微调基础理论   | 朱广恩     | 10.7   | ✅   |
| 2.2 监督微调实践     | 王泽宇     | 10.7   | ✅  |
| 3.1 直接偏好优化基础理论 | 王海洪     | 10.7   | ✅   |
| 3.2 直接偏好优化实践   | 张宏历     | 10.7   | ✅   |
| 4.1 在线强化学习基础理论 | 朱伯湘     | 10.7   | ✅  |
| 4.2 在线强化学习实践   | 蔡煊琪，朱伯湘 | 10.7   | ✅  |
| 5.1 总结         | 张宏历     | 10.7   | ✅   |


## 致谢



- 特别感谢 [@Datawhale](https://github.com/datawhalechina) 对本项目的支持
- 如果有任何想法可以联系我们，也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/Post-training-of-LLMs/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/Post-training-of-LLMs" />
  </a>
</div>

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/Post-training-of-LLMs&type=Date)](https://star-history.com/#datawhalechina/Post-training-of-LLMs&Date)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
