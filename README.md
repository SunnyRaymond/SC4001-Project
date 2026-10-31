# SC4001 Project（方向 I）：少样本 CLIP 适配何时有害？

更新：2026-10-31（按课程大作业的体量收缩了范围，见 3.5、3.6 与第 7 节）。已选定 handout 方向 I（Adapting Foundation Models to Unseen Domains）。本文是研究方案与执行计划，文中的"预期"都是待验证的假设，不是结果。

**进度（2026-10-31）：** 已在 TC1 上建好环境（4.2）、下载数据、生成固定划分、提取 ViT-B/32 特征，并跑通 B0/B1（见 4.5）；超参网格已固定在 `configs/adapt.yaml`，A1、A2、A4 的主矩阵已跑完（见 4.6）；help/harm、置信度分箱、门控与对照已完成，H1、H2 已判定，第 7 节图 1–4 与主结果表已生成（见 4.7）。图 5（被改错的示例图）与 3.6 的可选项尚未做。方括号编号 [n] 对应第 8 节的参考文献。

## 0. 一页摘要

- **问题：** 用每类 K 张目标域标注图把冻结的 CLIP 适配到新域时，哪些测试图被"改对"、哪些被"改错"？被改错的图能否由 zero-shot 置信度提前识别并避免？
- **做法：** Office-Home 四个域轮流作目标域。在缓存的 CLIP 特征上比较 zero-shot、两种线性探针与 Tip-Adapter；把每种适配的净增益拆成 help（错→对）与 harm（对→错）；按 zero-shot 置信度分箱分析；据此检验逐样本置信度门控，并与"全局混合"和"保留率相同的随机门控"对照。
- **防泄漏：** 所有超参只在另外三个来源域上模拟同样的任务来选，目标域测试标签只在最终评估时用一次。
- **与已有工作的关系：** VRF [11]、DaWin [12] 已在大数据量的鲁棒微调中逐样本组合 zero-shot 与微调模型。本组研究的是：每类只有 1–8 张标注、又没有目标域验证集时，这种组合是否仍然有效，以及原因。
- **成本：** GPU 只用于一次性特征提取；其余实验都在 512 维缓存特征上运行，单组配置秒级完成，因此能把 域 × K × seeds × 方法 全部跑满。
- **范围：** 正文只有 10 页、离截止约两周，因此先定下报告要放的图表（第 7 节），只跑填满这些图表所需的实验；其余列为可选（3.6）。

## 1. 课程要求（摘自 [project_handout.pdf](project_handout.pdf)，2026-10-10 版）

- 最多三人一组；截止 **2026-11-16 23:59**；迟交每天扣 5%，最多三天。
- 需要：聚焦的研究问题、合适的基线、有理由的修改/比较/假设检验；测试数据与模型开发分离；分析局限与失败案例。证据充分的负结果同样有价值。
- 需注明使用的现有代码与预训练模型，说清本组贡献；每位成员都要能讲解整个工作。只用现成工具拼应用不够。
- 评分：**实验与结果 40%、报告表达 30%、新颖性 20%、同伴评价 10%**。新颖性可以是"有理由的方法改动、对问题的适配，或产生有用见解的原创研究"；**更大的模型或更高的分数本身不构成贡献**。
- 方向 I 原文要求：使用有限或无标签的目标数据开发并评价适配策略，**同时考察改进与适配反而有害的情形**。
- 提交到 NTULearn：PDF 报告 + 代码 ZIP。报告 A4、单栏、Arial 10 pt，正文最多 10 页（封面、目录、参考文献、附录不计）；主要内容必须在正文。封面写组员姓名与组号；需包含引言、现有技术综述、方法、实验与结果、讨论、**AI Use Statement**。
- 代码需有注释，并写明复现主要结果的依赖、数据准备与命令。
- AI 使用：需验证生成内容与引用；出现幻觉内容会扣分；漏写 AI Use Statement 会扣分。

## 2. 研究问题

### 2.1 为什么不沿用原方案

原方案比较 zero-shot、纯 adapter（α=1）、残差混合（α=0.2）三者在 4-shot 下的准确率。问题在于：

1. 这基本就是 CLIP-Adapter [4] 原论文里残差比例 α 的消融，评分人很容易看成复现，新颖性分数难拿。
2. 只看平均准确率，看不到适配**伤害了哪些样本**，而这恰恰是方向 I 明确要求考察的内容。
3. 只有 K=4 一个点，看不到伤害如何随标注量变化。

### 2.2 新的研究问题

**主问题：** 在不使用目标域测试标签选参的前提下，用少量目标域标注把冻结 CLIP 适配到 Office-Home 的新域时，净增益由多少"改对"和多少"改错"组成？"改错"能否由 zero-shot 置信度预测，并通过逐样本门控避免？

| 子问题 | 类型 | 回答方式 |
| --- | --- | --- |
| RQ1 harm 有多大，随 K、域、适配方法如何变化？ | 测量 | help/harm 分解，覆盖 4 域 × 4 个 K × 3 种适配 |
| RQ2 适配在哪些 zero-shot 置信度区间划算、哪些不划算？ | 诊断 | 按置信度分箱统计净增益与条件 harm 率 |
| RQ3 逐样本置信度门控能否去掉 harm、保住 help？收益取决于什么？ | 干预 | 在同一份适配输出上比较门控、随机门控、全局混合 |

**动机依据：** CLIP 论文 [1] 发现 zero-shot 大致相当于 4-shot 线性探针，并把"更好地结合 zero-shot 与 few-shot"列为未来方向；LP-FT [7] 和 CLAP [8] 都报告过适配在分布偏移下可能不如原模型。

### 2.3 预先写定的假设与判定标准

先写定判定标准再看结果，避免事后挑结论。

**RQ1 只做测量，不设假设。** harm 几乎必然大于 0（K=1 时光是随机性就会产生翻转），设"harm 率 < 1%"这类否定条件实际上不可能被满足，没有检验意义。报告直接给出 harm 的大小，以及它随 K、域、适配方法的变化。

- **H1（高置信样本上，适配不划算）：** 按 zero-shot 最大 softmax 概率（MSP）[10] 把测试图分成 10 个等频箱，置信度最高的 20% 样本上，适配的净增益 ≤ 0。
  *不成立的条件：* 最高置信 20% 的净增益在多数 seeds 下 > 0。
  *不能当作发现的部分：* harm 只会发生在 zero-shot 正确的样本上，而 MSP 本来就能预测 zero-shot 是否正确 [10]，所以"harm 集中在高置信箱"几乎由定义决定。报告中有信息量的是各箱的净增益，以及**条件 harm 率**（箱内 zero-shot 正确的样本被改错的比例）怎样随置信度变化。
- **H2（门控有效，且收益取决于适配离 zero-shot 多远）：** 在同一份适配输出上，置信度门控 G1（阈值只在来源域上选）的准确率高于保留率相同的随机门控 C2；G1 相对未门控输出的提升，在不带 zero-shot 先验的 A1 上最大，在输出端保留 zero-shot 项的 A4 上最小。
  *不成立的条件：* G1 不优于 C2；或提升大小与上述顺序不符。
  *与全局混合 C1 的比较不预设方向：* 在概率空间做全局混合时，更自信的一方本来就会主导预测，相当于隐式的逐样本门控（见 3.3），因此 G1 与 C1 可能很接近。如实报告二者的配对差值，不把"门控优于全局混合"写成预期。

**决策点：**
- 若 H1 不成立（高置信样本上适配也有净增益），门控就没有动机，不硬做；改为如实报告负结果，并检查其他可能预测 harm 的信号，例如类别文本原型之间的间隔、支持集与测试图的特征距离（后者即 VRF [11] 的思路，见 3.6）。
- 若 harm 远小于 help（例如所有 K ≤ 4 的组合中条件 harm 率都 < 1%），主线改为"增益如何分布在各域、各类别"，门控仍作为一项检验保留。

### 2.4 贡献与边界（写报告时按此表述）

**我们主张：**

1. 一个严格的受控测量：在少样本、只用来源域选参的设定下，把 CLIP 适配的增益拆成 help/harm（harm 即 Yan et al. [13] 定义的 negative flip），并在域、K、适配方法上对照。文献检索中没有找到针对少样本 CLIP adapter 的同类翻转分析（检索范围见 8.3）。
2. 检验"按 zero-shot 置信度逐样本决定是否采用适配结果"在少样本设定下是否仍然有效，并用全局混合、随机门控、oracle 上界做受控对照；说明门控的收益如何随适配方法离 zero-shot 的远近而变化。

**我们不主张：**

- **门控机制不是本组首创。**
  - VRF [11] 在鲁棒微调中按测试样本到"zero-shot 失败集"的距离逐样本组合 zero-shot 与微调模型，并把基于 MSP 的逐样本选择作为对比基线。
  - DaWin [12] 用两个模型的预测熵逐样本决定插值系数。

  两者都基于大量训练数据的全量微调。本组的问题是：每类只有 1–8 张标注、没有目标域验证集时，这类逐样本组合是否仍然成立，以及原因。报告中必须引用二者并说明差别。
- **不提出新的 adapter 结构。** CLIP-Adapter [4] 和 Tip-Adapter [5] 用全局权重组合 zero-shot 与适配结果，WiSE-FT [6] 在权重空间插值。
- **"未见域"只指适配前的目标域。** CLIP 预训练数据可能包含相似图像，不能声称模型从未见过这些风格或物体。
- **属于 few-shot domain adaptation。** 实验用了少量目标域标签，不是 domain generalization，两者不能混称。
- **域层面只做描述。** 只有 4 个域，域层面的规律不做统计推断。

## 3. 实验设计

### 3.1 数据与划分

- **数据：** Office-Home [2]，65 类，四个域：Art 2,427 张、Clipart 4,365 张、Product 4,439 张、Real World 4,357 张，共 15,588 张。
- **来源：** 优先使用 Hugging Face 镜像 `flwrlabs/office-home`。它是 parquet 格式，约 1.16 GB，含 image/domain/label 三列，无需登录，数据卡注明来自官方页面与 [2]。官方入口是 Google Drive 上的 `OfficeHomeDataset_10072016.zip`，是否需要登录未确认。详见 8.1。
- **使用条款：** 官方只给出 Fair Use Notice：图片含受版权保护的内容，仅限非商业的研究与教学使用。报告中注明。
- **类名：** 镜像的标签名与原始文件夹名可能有出入，数据卡提到对 Clock 文件夹的特殊处理。拿到数据后先核对 65 个类名，再固定映射。
- **轮换：** 每轮选一个目标域 d，另外三个域为来源域。
- **嵌套支持集：** 对每个 seed（0–4），在 d 的每个类别中预先抽出 K_max 张作为候选支持池；K ∈ {1, 2, 4, 8} 时取池中前 K 张，因此各 K 的支持集逐级包含。d 中池以外的全部图像作为测试集，**同一 seed 下所有 K 和所有方法共用同一个测试集**，跨 K 的比较不会被测试集变化干扰。
- **不做 K=16：** Art 平均每类只有约 37 张，K=16 会让部分类别的测试图过少，多一个 K 对结论帮助也不大；K_max 固定为 8。拿到数据后仍先统计每类数量，确认每类抽走 8 张后还有测试图。
- **选参协议（leave-target-out，参照 DomainBed [9] 的训练域验证思路）：**
  - 目标域 d 的所有超参都在三个来源域上选：学习率、步数、α、β、门控阈值 τ、全局混合权重 w。网格保持小，每个超参 3–5 个取值。
  - 做法是在每个来源域上模拟同样的 K-shot 任务（3 个 seeds），取来源域平均准确率最高的配置。
  - d 的测试标签只在最终评估时使用一次。
- **预期偏低：** CLAP [8] 指出，CLIP-Adapter、Tip-Adapter 原文的超参是按数据集搜索的，部分甚至在测试集上搜索。因此本组适配器的数字可能低于原论文报告值，这是协议更严格的预期结果，不是实现错误，报告中需要说明。
- **固定在先：** 类名只做"下划线/连字符 → 空格、小写化"；prompt 模板、超参网格、seeds 都在看到任何测试结果前写入 `configs/` 并提交 git。

### 3.2 模型与特征

- **主 backbone：** CLIP ViT-B/32（OpenAI 权重）[1]，完全冻结；图像和文本特征均为 512 维，L2 归一化。
  - 加载方式：`open_clip_torch==3.3.0`，`open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')`。
  - 权重来自 Hugging Face 的 `timm/vit_base_patch32_clip_224.openai`。
- **必须使用 `-quickgelu` 版本**，或传入 `force_quick_gelu=True`。OpenAI 权重使用 QuickGELU，open_clip 3.3.0 不会自动切换；用错会出现警告并掉点。
- **复核 backbone（可选，见 3.6）：** CLIP ViT-B/16，使用 `ViT-B-16-quickgelu` 与 `timm/vit_base_patch16_clip_224.openai`，流程相同，只换特征。
- **预处理：** 使用 open_clip 返回的 CLIP 预处理（resize + center crop 到 224×224）。
- **缓存：** 每张图只提特征一次，连同域、类别、在镜像中的索引一起保存；之后所有实验只读缓存。
- **V100 注意：** FP16 推理即可；不需要 BF16 或 FlashAttention。

### 3.3 方法与对照

记 zero-shot 概率为 p_zs，适配后的概率为 p_ad，zero-shot 置信度 c(x) = max p_zs（使用 CLIP 默认 logit scale）。

| 编号 | 方法 | 可训练参数 | 在来源域上选的超参 | 作用 |
| --- | --- | --- | --- | --- |
| B0 | Zero-shot，模板 "a photo of a {c}." | 0 | 无 | 主参照；help/harm 都相对它计算 |
| B1 | Zero-shot + 域提示（如 "a clipart image of a {c}."，模板预先固定） | 0 | 无 | 不用标签的廉价适配 |
| A1 | 线性探针（L2 正则的 logistic regression，lbfgs 从零初始化；目标为凸，初始化不影响解） | 约 33K | 正则强度 | 经典基线 |
| A2 | 线性探针，以 zero-shot 文本原型初始化 | 约 33K | 学习率/步数、正则 | 从 zero-shot 出发的适配；相当于不加约束的 CLAP [8] |
| A3 | CLIP-Adapter [4]：f* = α·A(f) + (1 − α)·f，A 为 512→128→512 MLP | 约 0.13M | 学习率/步数、α | **可选**，原因见表下说明 |
| A4 | Tip-Adapter [5]：logits = α·exp(−β(1 − f·Fᵀ))·L + f·Wᵀ | 0 | α、β | 免训练 adapter |
| G1 | **硬门控：** c(x) ≥ τ 时用 B0 的预测，否则用适配预测 | 0 | τ | 本组检验的方法 |
| G2 | **软门控：** p = c(x)·p_zs + (1 − c(x))·p_ad | 0 | 无 | 无超参变体；机制与 C1 接近，结果放附录 |
| C1 | 全局混合：p = w·p_zs + (1 − w)·p_ad | 0 | w | 对照：去掉"逐样本" |
| C2 | 随机门控，保留 zero-shot 的样本比例与 G1 相同（重复 20 次取平均） | 0 | 无 | 对照：去掉"置信度信号" |
| C3 | Oracle：B0 与适配任一正确即算对 | — | — | 只作上界，不是方法 |

- **主线只做 A1、A2、A4。** 三者覆盖"离 zero-shot 有多远"这条轴：A1 不带 zero-shot 先验；A2 用 zero-shot 原型初始化，但之后自由训练；A4 在输出端显式保留 zero-shot 项。A3 在特征上保留残差，在这条轴上与 A4 重复，因此列为可选。
- G1、G2、C1、C2、C3 分别作用在 A1、A2、A4 每一种适配输出上。它们都是对已存的逐图概率做后处理，不需要重新训练。
- **全局混合本身已有逐样本效应：** C1 在概率空间混合，p_zs 很尖锐时由它主导 argmax，p_zs 平坦时由 p_ad 主导，相当于隐式的置信度加权；G2 只是把这一效应放大。因此 G1、G2 与 C1 的差距可能很小，见 H2。
- 主分析以 B0 为参照；如果 B1 明显更强，再以 B1 为参照复核一次。
- c(x) 未经校准。本组不做温度校准，在局限中说明；Kumar et al. [14] 讨论过组合前校准的作用。

### 3.4 评价指标

- **主指标：** 目标域测试集 top-1 accuracy。macro-F1 只放附录。
- **help 率** = B0 错、适配对的样本比例；**harm 率** = B0 对、适配错的比例，即 negative flip [13]；Δacc = help − harm。
- **条件 harm 率** = harm 数 / B0 正确数，即"原本对的有多少被改错"；**条件 help 率** = help 数 / B0 错误数。
- **置信度分箱：** 按 c(x) 分 10 个等频箱，每箱报告 help、harm、净增益与条件 harm 率。
- **门控曲线：** 对每种适配，把 τ 从高扫到低：横轴为保留 zero-shot 预测的样本比例，纵轴为准确率；画 G1 与 C2 两条曲线，把 C1（来源域选出的 w）画成水平线，并标出来源域选出的 τ 对应的点。整条曲线用测试标签画，只用于分析；报告的数字只取标出的那个点。
- **统计：** 每个（方法, 域, K）报告 5 seeds 的均值 ± 标准差。主要比较（G1 对未门控、对 C1、对 C2）报告跨 seeds 的配对差值和赢/平/输计数。McNemar 检验可选，放附录。
- **失败分析：** 找出 harm 最多的类别；每个域抽样展示被改错的图；分析门控仍然失败的样本，例如高置信但 zero-shot 错误的图。

### 3.5 实验矩阵与预算

- **主矩阵：** 4 个目标域 × K ∈ {1, 2, 4, 8} × 5 seeds × 3 种适配（A1、A2、A4）× {未门控, G1, G2, C1, C2, C3}，另加 B0、B1。只有适配需要训练，门控与对照都是后处理。
- **选参：** 每个（目标域, K）在 3 个来源域 × 3 seeds 上搜索预先固定的网格。
- **复核（可选）：** ViT-B/16 跑相同的主矩阵；报告中只写一段，汇总表放附录。
- **GPU：** 每个 backbone 对 15,588 张图做一次特征提取，主要受图片解码速度限制。粗估每次不到半小时，含调试预留 2–4 GPU 小时；这是估计，需在试跑中实测。其余实验在缓存特征上运行，GPU 或 CPU 都可以。集群卡住时的备选见第 4 节。

### 3.6 可选项（11/8 冻结主结果后才做，按顺序，时间不够就停）

1. **ViT-B/16 复核：** 只需多提一次特征，主矩阵代码不变。
2. **A3 CLIP-Adapter：** 补上常见的训练式 adapter。
3. **以下扩展最多挑一项：**
   - **VRF 式加权的少样本版本 [11]：** 把支持集中 zero-shot 判错的图作为"失败集"，按测试图到失败集的特征距离决定组合权重，检验失败集只有几十张时它是否比 MSP 门控更好。这是"把已有方法适配到新问题"，对新颖性帮助最大。
   - **CLAP [8] 作为强基线：** 它通过约束让类原型不偏离 zero-shot，是避免 harm 的另一条路线，可与门控直接比较。
   - **无标签目标数据：** 把 zero-shot 高置信的无标签目标图作为伪标签加入 Tip-Adapter 缓存，检验它是否同样带来 harm、门控是否仍然有效。handout 写的是"limited or unlabelled target data"，无标签数据不是必做。

**明确不做：** K=16、温度校准、第二个多域数据集。

## 4. 在 TC1 集群上运行

依据 [CCDSGPU-TC1-UG-UserGuide.pdf](CCDSGPU-TC1-UG-UserGuide.pdf)（2026-02-12 版）。

**不要被集群卡住：** 特征提取是一次性的小任务。如果 QoS、联网或环境问题拖过半天，就直接在 Colab 上提取特征，再把特征文件拷回来；后续实验只读缓存特征，在本机或集群上都能跑。最要紧的是尽快拿到 B0 在四个域上的准确率。

### 4.1 连接

- [01-SSH-TC1.cmd](01-SSH-TC1.cmd)：SSH 到头节点（校外需先连 NTU VPN）。
- [02-VSCode-TC1.cmd](02-VSCode-TC1.cmd)：用 VS Code Remote-SSH 打开远程目录。目前指向作业目录 `SC4001-Assignment`，建好项目目录后改成项目路径。
- 头节点只用来编辑、提交和查询作业。特征提取、训练、评价都通过 `sbatch` 放到计算节点上运行；不要用 `srun` 跑长任务。

### 4.2 首次设置（头节点）

`MyTCinfo` 查询结果（2026-10-31）：QoS `normal`，每用户 1 GPU、20 核、64G 内存，单作业最长 6 小时，最多同时提交 2 个作业；分区 `UGGPU-TC1`，计算节点 GPU 为 Tesla V100-PCIE-32GB。

**不用课程 conda 环境：**

- 指南里的 `SC4001.yml` 不存在，`/tc1apps/3_conda_yml/` 下只有 `CZ4042_v4/v5/v6.yml`。
- v5（Python 3.11、PyTorch 2.5.1、CUDA 11.8）固定了 `mkl=2025.0.0`，`import torch` 报 `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`，这是 conda 版 PyTorch 与 MKL ≥ 2024.1 的已知冲突；v4 固定了同一 MKL 版本。v6 是 PyTorch 2.7 + CUDA 12.8，未确认是否支持 V100。
- 因此用集群的 Python 3.11 模块建 venv，从 PyTorch 官方源装 `torch==2.5.1+cu118` 的 pip wheel。它自带 CUDA 库，仍支持 V100（sm_70）。

在项目根目录（集群上为 `~/SC4001-Project`）运行一次：

```bash
bash jobs/setup_env.sh
```

- 环境装在 `~/envs/sc4001`，作业里用 `source ~/envs/sc4001/bin/activate` 激活，不需要 `module load`。
- 安装其余包时用约束文件固定 torch，防止 pip 顺带升级；实际版本记录在 `requirements-lock.txt`。
- open_clip 3.3.0 要求 torch ≥ 2.0。不要从 GitHub main 分支安装：其 README 写的是 torch ≥ 2.6。

### 4.3 数据与模型权重

计算节点可以联网：2026-10-31 的探测作业能访问 huggingface.co 与 pypi.org。下载作业按 `configs/base.yaml` 中固定的 revision 下载数据与权重，然后生成划分：

```bash
mkdir -p logs && sbatch jobs/download.sbatch
```

- 数据放在 `data/office_home`（约 1.16 GB）；权重只取 `open_clip_model.safetensors`，放在 `~/.cache/huggingface/hub`。
- 没有网络的机器：在别处运行 `python scripts/download.py`，再把这两个位置拷过去。特征提取作业设置了 `HF_HUB_OFFLINE=1`，只读缓存。
- 数据与特征放在 home 目录下，每完成一个阶段就把结果拉回本机。集群**没有备份**。

### 4.4 作业

从项目根目录提交（SLURM 不会自动创建 `logs/`）：

```bash
sbatch jobs/extract_features.sbatch ViT-B-32-quickgelu
sbatch jobs/zero_shot.sbatch ViT-B-32-quickgelu
```

- 后一个作业可以加 `--dependency=afterok:<前一个 jobid>` 排在前一个之后；QoS 最多同时提交 2 个作业。
- 特征提取作业每 60 秒把 GPU 利用率和显存写入 `logs/gpu_<jobid>.csv`。作业结束后用 `seff <jobid>` 查看实际的 CPU 和内存利用率，据此调整申请量。
- zero-shot 作业不申请 GPU，几秒内完成。
- 只使用分配给自己的 QoS，申请量不超过 QoS 上限；不要访问自己 home 目录和共享目录以外的路径。
- 同步代码：在本机项目根目录运行 `tar cf - configs scripts jobs requirements.txt | ssh CCDS-TC1 "cd ~/SC4001-Project && tar xf -"`。

### 4.5 首次运行记录（2026-10-31，ViT-B/32）

| 作业 | 用时 | 说明 |
| --- | --- | --- |
| 下载 + 划分（CPU） | 29 s | 四个域图片数与 3.1 一致，共 15,588 张；每类最少的是 Art/drill，15 张，抽走 8 张后剩 7 张测试图 |
| 特征提取（1×V100） | 75 s | 图像部分 50 s；logit scale = 100；无 QuickGELU 警告 |
| B0/B1（CPU） | 1 s | 结果写入 `results/ViT-B-32-quickgelu/` |

测试集 top-1 准确率（%），5 个 seeds 的均值 ± 标准差：

| 方法 | Art | Clipart | Product | Real World | 平均 |
| --- | --- | --- | --- | --- | --- |
| B0 | 79.23 ± 0.43 | 64.96 ± 0.32 | 86.74 ± 0.14 | 87.60 ± 0.07 | 79.63 |
| B1 | 79.84 ± 0.43 | 67.26 ± 0.26 | 88.07 ± 0.13 | 88.00 ± 0.12 | 80.79 |

- B1 在四个域上都高于 B0，Clipart 上高 2.3 个点。按 3.3，主分析仍以 B0 为参照，另以 B1 复核一次。
- 原计划预留 2–4 GPU 小时；实际一次提取不到 2 分钟，ViT-B/16 复核的成本可以忽略。

### 4.6 适配主矩阵（2026-10-31，ViT-B/32）

在本机 CPU 上运行 `python scripts/run_experiments.py`，用时约 7 分钟（A1 112 s、A2 306 s、A4 15 s）；特征从集群拷回 `features/`。选参按 3.1 的 leave-target-out 协议，各配置的来源域平均准确率在 `adapt_selection.csv`。

测试集 top-1 准确率（%），5 个 seeds 的均值：

| 方法 | K | Art | Clipart | Product | Real World | 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0 | 79.23 | 64.96 | 86.74 | 87.60 | 79.63 |
| B1 | 0 | 79.84 | 67.26 | 88.07 | 88.00 | 80.79 |
| A1 | 1 | 42.77 | 36.58 | 66.81 | 57.31 | 50.87 |
| A1 | 2 | 56.73 | 48.56 | 79.60 | 69.13 | 63.50 |
| A1 | 4 | 66.88 | 60.23 | 86.26 | 79.15 | 73.13 |
| A1 | 8 | 73.45 | 68.33 | 90.32 | 84.91 | 79.25 |
| A2 | 1 | 79.92 | 66.42 | 87.62 | 88.25 | 80.55 |
| A2 | 2 | 80.51 | 67.82 | 89.00 | 88.47 | 81.45 |
| A2 | 4 | 81.85 | 70.38 | 90.55 | 89.17 | 82.99 |
| A2 | 8 | 82.56 | 73.15 | 91.74 | 89.87 | 84.33 |
| A4 | 1 | 80.06 | 65.74 | 87.64 | 88.16 | 80.40 |
| A4 | 2 | 80.55 | 66.38 | 88.53 | 88.21 | 80.92 |
| A4 | 4 | 81.05 | 67.07 | 89.21 | 88.45 | 81.44 |
| A4 | 8 | 82.18 | 68.91 | 90.04 | 88.97 | 82.52 |

- A1 不带 zero-shot 先验，K ≤ 4 时远低于 B0，K=8 时平均仍低于 B0；A2、A4 在所有 K 上平均都高于 B0。help/harm 分解尚未做，平均准确率高于 B0 不代表没有 harm。
- **选中值落在网格边缘：** A4 的 α 在全部 16 个（域, K）上都选中最大值 5.0，K=8 时 β 也选中最大值 8.0；A2 的 lr 全部选中最小值 1e-4，K=1 时步数也选中最小值 50；A1 的 λ 有 7 处选中最小值 1e-5。说明 A2 偏好"少偏离 zero-shot"，A4 偏好"更多依赖缓存"，真正的最优可能在网格之外。网格按协议已固定，是否扩展见下一条。
- **决定（2026-10-31）：不扩展网格。** 网格保持预先提交的版本，报告中把"选中值落在边缘、A2/A4 的数字可能偏低"写入局限。
- 运行顺序：`run_experiments.py` → `analyze.py` → `gating.py`（约 1 分钟，需 `adapt_probs.npz`）→ `plot.py`。
- K=1 时 sklearn 提示"类别数超过样本数的一半"，这是每类 1 张图的正常情况，不影响拟合；λ=1e-5 时 lbfgs 无收敛警告。

### 4.7 分析结果（2026-10-31，ViT-B/32）

**RQ1，help/harm**（相对 B0，占测试图的 %，四域 × 5 seeds 平均；图 2）：

| 方法 | K=1 help / harm | K=2 | K=4 | K=8 |
| --- | --- | --- | --- | --- |
| A1 | 4.36 / 33.13 | 5.85 / 21.98 | 7.26 / 13.77 | 8.76 / 9.14 |
| A2 | 4.69 / 3.77 | 5.90 / 4.08 | 7.09 / 3.74 | 8.05 / 3.35 |
| A4 | 2.11 / 1.35 | 2.93 / 1.64 | 3.77 / 1.95 | 4.94 / 2.05 |

- A2 的净增益全部来自 help 增长，harm 基本不随 K 下降，稳定在 3–4%：原本对的图约 5% 被改错。A4 改动最少，harm 随 K 略升。
- 2.3 第二个决策点（条件 harm 率 < 1%）不触发：A2 约 5%，A4 约 2–3%。

**RQ2 / H1：成立。** 最高置信 20% 上的净增益在全部 48 个（方法, 域, K）中，多数 seeds 都 ≤ 0（`h1.csv`）。A2、A4 在这里几乎不改变预测，净增益约为 0；A1 仍为负。各箱的净增益随置信度单调下降：A2、A4 只在前 5–6 个箱有正增益（图 3）。

**RQ3 / H2：按预先写定的标准成立，但门控不如全局混合。** 配对差值为跨（域, K, seed）80 组的均值 ± 标准差，方括号为赢/平/输：

| 适配 | G1 − 未门控 | G1 − C2 | G1 − C1 | G2 − C1 |
| --- | --- | --- | --- | --- |
| A1 | +13.48 ± 10.85 [75/0/5] | +1.80 ± 1.11 [80/0/0] | −1.85 ± 0.91 [0/0/80] | −0.46 ± 0.48 [13/3/64] |
| A2 | +0.04 ± 0.29 [50/7/23] | +1.37 ± 0.81 [79/0/1] | −0.40 ± 0.43 [10/3/67] | −0.35 ± 0.50 [17/3/60] |
| A4 | +0.02 ± 0.08 [34/28/18] | +0.97 ± 0.56 [80/0/0] | +0.03 ± 0.08 [36/29/15] | −0.22 ± 0.34 [24/2/54] |

- G1 优于保留率相同的随机门控 C2，说明置信度确实带有信号；G1 相对未门控的提升按 A1 > A2 > A4 排序，与 H2 一致。但 A2 与 A4 的提升都在 0.05 pp 以内，排序的后半段没有实际意义。
- **全局混合 C1 在 A1、A2 上几乎每组都赢过 G1。** 这与 3.3 的预判一致：概率空间混合本身已有隐式的逐样本效应，而且还能利用适配输出在中等置信区间的信息。A4 上来源域选出的 w 大多为 0，C1 ≈ 未门控。
- 来源域选出的 τ：A1 偏低（0.3–0.7），A2、A4 偏高（0.7–0.9）。门控曲线（图 4）显示，A2、A4 把前约 50% 高置信图交回 B0 时准确率几乎不变：这些图上适配本来就不改动预测。
- K=4 主结果表见 `results/<backbone>/main_table_k4.csv`：B0 79.63；A2 82.99（harm 3.74）、A2+C1 83.24（harm 2.43）；A1+G1 80.39，仍低于未门控的 A2。
- **对报告的含义：** 主结论应写成"harm 真实存在且集中在中低置信区间；置信度能识别它，但简单的硬门控在少样本下不如全局混合"，而不是"门控有效"。3.6 的 VRF 式加权正好检验"比 MSP 更好的信号"能否让逐样本组合胜过全局混合，建议作为可选项的首选。

## 5. 代码结构

```
configs/base.yaml        # 数据/权重 revision、域、prompt 模板、K、seeds（看到结果前固定）
configs/adapt.yaml       # 适配方法、门控的超参网格（看到适配结果前固定，提交 2ccc21d）
requirements.txt         # 依赖；requirements-lock.txt 为 TC1 上的实际版本
scripts/
  common.py              # 配置、全局图像索引、类名规范化、读缓存特征与划分
  download.py            # 下载固定 revision 的数据与权重
  make_splits.py         # 生成所有（域, seed）的支持池与测试集 → splits/
  extract_features.py    # 一次性：图像/文本特征 → features/<backbone>/
  zero_shot.py           # B0、B1 → results/<backbone>/
  run_experiments.py     # A1、A2、A4 与来源域选参，逐图预测写入 results/
  analyze.py             # help/harm、置信度分箱、H1 判定 → flips/conf_bins/h1.csv
  gating.py              # G1、G2、C1–C3，τ 与 w 在来源域上选，门控曲线、H2 配对差值
  plot.py                # 第 7 节图 1–4 → figures/<backbone>/，K=4 主结果表
jobs/                    # setup_env.sh 与 sbatch 脚本
splits/                  # 固定划分与每类数量（提交到 git，全组共用）
results/                 # 逐图预测与汇总表（只存小文件）
figures/                 # 报告用图（PNG）
```

- **全局图像索引：** 按文件名排序后 parquet 分片的行序。特征、划分、逐图结果都用这个索引指代图像。
- `data/`、`features/`、`logs/` 不进 git。特征可在一个 GPU 作业内重新生成。
- `results/<backbone>/adapt_probs.npz`（完整 p_ad，约 85 MB）不进 git，重跑 `run_experiments.py` 即可生成，本机 CPU 约 7 分钟。

要保存**逐图预测**而不只是准确率：help/harm、置信度分箱、门控曲线都依赖逐图结果，门控与对照也是在保存的逐图概率上做后处理。

## 6. 时间表与分工（2026-10-31 至 11-16）

| 日期 | 工作 | 检查点 |
| --- | --- | --- |
| 10/31–11/2 | 确认 QoS 与计算节点联网（卡住就改用 Colab）；下载数据；建环境；核对类名、统计每类数量；固定 configs；提取 ViT-B/32 特征；跑 B0、B1 | 四个域的 zero-shot 准确率 |
| 11/3–11/5 | 实现 A1、A2、A4 与来源域选参；跑 K × seeds；完成 help/harm 与置信度分箱 | 判定 H1，按决策点确定门控是否继续 |
| 11/6–11/8 | 实现 G1、G2、C1–C3 与门控曲线；跑全部主矩阵；画出第 7 节列出的图表 | **11/8 冻结主结果** |
| 11/3 起并行 | 写相关工作与方法章节 | — |
| 11/9–11/11 | 有余力才做 3.6 的可选项；不改动已冻结的主结果 | — |
| 11/9–11/14 | 写结果与讨论、写失败分析；整理代码与复现说明；由另一名组员按说明独立复现主结果 | 报告完整初稿 |
| 11/15–11/16 | 校对页数与格式，写 AI Use Statement，打包 ZIP，**11/16 白天提交**，不等到 23:59 | 已提交 |

**建议分工（三人）：**

- 成员 1：数据与集群（下载、环境、特征、划分）；之后负责复现说明和可选的 backbone 复核。
- 成员 2：方法与实验（适配方法、门控、选参、主矩阵）。
- 成员 3：分析与写作（help/harm 分析、图表、相关工作，从第一周就开始写）。

每个人都要能讲清全流程，同伴评价占 10%。

## 7. 报告结构（正文 ≤ 10 页）

| 节 | 内容 | 建议篇幅 |
| --- | --- | --- |
| 封面、目录 | 组员姓名、组号（不计页数） | — |
| 1 Introduction | 问题、动机、RQ1–RQ3、贡献 | 0.75 |
| 2 Review of existing techniques | CLIP 与 zero-shot [1]；提示学习 [3]；adapter 类方法 [4, 5]；鲁棒微调与逐样本组合 [6, 7, 11, 12]；少样本选参问题 [8, 9]；翻转分析 [13] | 1.25 |
| 3 Methods | 设定、基线、help/harm 分解、置信度门控及对照 | 1.5 |
| 4 Experimental setup | 数据划分、leave-target-out 选参、实现细节、算力 | 1 |
| 5 Results | RQ1、RQ2、RQ3 各一小节；若做了 backbone 复核，加一段 | 3.5 |
| 6 Discussion | 失败案例、局限、负结果的含义 | 1.25 |
| 7 Conclusion | — | 0.25 |
| AI Use Statement | 工具、用途、核实方式、团队决策与贡献 | 0.25 |
| References、Appendix | 不计页数 | — |

**主要图表**（先定下这些，再只跑填满它们所需的实验）：

1. 每个域的准确率–K 曲线（B0、B1、A1、A2、A4）。
2. help/harm 堆叠柱状图（按 K 与适配方法）。
3. 置信度分箱图：每箱的净增益与条件 harm 率。
4. 门控曲线（定义见 3.4）。
5. 被改错图像的示例。

另加一张主结果表：K=4 时各方法在四个域上的准确率与 harm 率。

## 8. 数据来源与参考文献

以下信息于 2026-10-30 对照 arXiv 页面、官方页面和 Hugging Face 页面核对。写报告时，引用到的具体结论仍需组员对照原文再读一遍。

### 8.1 数据

- **Office-Home 官方页面：** <https://www.hemanthdv.org/officeHomeDataset.html>
  - 页面写明约 15,500 张、65 类、4 个域。
  - 下载入口为 Google Drive 上的 `OfficeHomeDataset_10072016.zip`。
  - 使用条款为 Fair Use Notice。
- **Hugging Face 镜像：** <https://huggingface.co/datasets/flwrlabs/office-home>
  - 3 个 parquet 分片，共约 1.16 GB；单一 `train` split；列为 `image`、`domain`、`label`。
  - 未设访问限制；数据卡引用官方页面与 [2]。
  - 各域图片数来自其 datasets-server 统计。
- **未确认：** 官方压缩包的大小、是否需要登录 Google；原始压缩包中 Real World 文件夹的确切名称（有的代码库改名为 `real_world`）。

### 8.2 模型与代码

- open_clip：<https://github.com/mlfoundations/open_clip>。PyPI 包名为 `open_clip_torch`，最新发布 3.3.0，要求 torch ≥ 2.0。
- OpenAI CLIP 权重（open_clip 默认从这里下载）：
  - <https://huggingface.co/timm/vit_base_patch32_clip_224.openai>
  - <https://huggingface.co/timm/vit_base_patch16_clip_224.openai>

### 8.3 参考文献

1. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
   - CLIP；§3.1.5 与图 6：在 20 个数据集上平均，zero-shot 与 4-shot 线性探针相当。
2. Venkateswara et al., "Deep Hashing Network for Unsupervised Domain Adaptation," CVPR 2017. [arXiv:1706.07522](https://arxiv.org/abs/1706.07522)
   - Office-Home 数据集。
3. Zhou et al., "Learning to Prompt for Vision-Language Models," IJCV 2022. [arXiv:2109.01134](https://arxiv.org/abs/2109.01134)
   - CoOp。
4. Gao et al., "CLIP-Adapter: Better Vision-Language Models with Feature Adapters," IJCV 2024. [arXiv:2110.04544](https://arxiv.org/abs/2110.04544)
   - 残差比例 α 见 §3.2 式 6。
5. Zhang et al., "Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification," ECCV 2022. [arXiv:2207.09519](https://arxiv.org/abs/2207.09519)
   - α 是加性权重，β 控制相似度锐度。
   - 注意：不要引用标题不同的早期预印本 2111.03930。
6. Wortsman et al., "Robust fine-tuning of zero-shot models," CVPR 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903)
   - WiSE-FT。
7. Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," ICLR 2022. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
   - 在 10 个分布偏移数据集上，全量微调的分布外准确率比线性探针低约 7%。
8. Silva-Rodríguez et al., "A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models," CVPR 2024. [arXiv:2312.12730](https://arxiv.org/abs/2312.12730)
   - CLAP。指出既有方法按任务甚至在测试集上调参；分布偏移下 adapter 常低于 zero-shot；提出以 zero-shot 原型为锚的约束线性探针。
9. Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization," ICLR 2021. [arXiv:2007.01434](https://arxiv.org/abs/2007.01434)
   - DomainBed；模型选择协议。
10. Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," ICLR 2017. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
    - 用 MSP 识别错误预测。
11. Zhu, Cui & Zhang, "Robust Fine-tuning of Zero-shot Models via Variance Reduction," NeurIPS 2024. [arXiv:2411.06966](https://arxiv.org/abs/2411.06966)
    - VRF；**最接近的已有工作**。
12. Oh et al., "DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation," ICLR 2025. [arXiv:2410.03782](https://arxiv.org/abs/2410.03782)
    - 按预测熵逐样本插值。
13. Yan et al., "Positive-Congruent Training: Towards Regression-Free Model Updates," CVPR 2021. [arXiv:2011.09161](https://arxiv.org/abs/2011.09161)
    - negative flip 的定义。
14. Kumar, Ma, Liang & Raghunathan, "Calibrated ensembles can mitigate accuracy tradeoffs under distribution shift," UAI 2022. [arXiv:2207.08977](https://arxiv.org/abs/2207.08977)
    - 组合前校准。

**相近工作的检索范围：** 逐样本或基于熵的融合、zero-shot 回退的置信度门控、CLIP 的 negative flip、无退化的模型更新。没有找到在少样本 CLIP adapter 上做 MSP 门控或翻转分析的论文，但可能有遗漏；写报告前再检索一次。

## 9. AI 使用记录（供最终 AI Use Statement 使用）

- **2026-10-13：** 使用 Codex 阅读 handout 与集群指南，生成 A–J 十个方向的选题草案（见 git 提交 `f92c718` 中的 README）。
- **2026-10-29/30：** 使用 Claude Code（Claude Opus 5.5）对比方向 C 与 I，细化方向 I 的研究问题，并重写本 README。第 8 节的数据、代码与文献信息由其检索后，对照 arXiv、官方页面与 Hugging Face 页面逐条核对。
- **2026-10-31：** 使用 Claude Code（Claude Opus 5.5）评审设计。它指出：范围超出 10 页正文能容纳的量；原 H2 的一部分由定义决定；概率空间的全局混合本身已有逐样本效应；原 H1 的否定条件过松。据此收缩了范围、改写了假设，并修改本 README。
- **2026-10-31：** 使用 Claude Code（Claude Opus 5.5）在 TC1 上搭建环境，编写 `configs/`、`scripts/`、`jobs/`，下载数据并跑通 B0/B1。核实方式：类名与各域图片数对照 Hugging Face 数据卡与 3.1；检查 CLIP 预处理与 logit scale；确认各 seed 的支持池与测试集不重叠。
- **2026-10-31：** 使用 Claude Code（Claude Opus 5.5）编写 `configs/adapt.yaml` 与 `scripts/run_experiments.py`，跑完 A1、A2、A4 主矩阵。网格先于任何适配结果提交（2ccc21d）。核实方式：断言支持集标签与类别一致；检查 λ 最小时 lbfgs 收敛；逐格检查选中值是否落在网格边缘。
- **2026-10-31：** 使用 Claude Code（Claude Opus 5.5）编写 `analyze.py`、`gating.py`、`plot.py`，完成 help/harm、置信度分箱、门控与对照并出图。核实方式：G1 在保留率 0 与 100% 时分别等于未门控与 B0（图 4 两端）；逐图检查图例与标注不遮挡数据。
- **团队决策：** 2026-10-29 选定方向 I；2026-10-31 决定按课程大作业的体量收缩范围。研究问题、假设与实验设计将在组内讨论后确认。
- **待补充：** 后续编码、调试、写作中的 AI 使用情况与核实方式。
