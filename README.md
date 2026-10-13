# SC4001 Project：单张 V100 下的选题与实验方案

更新时间：2026-10-13。本文是选题建议，尚未下载研究数据、运行模型或提交集群作业。

## 1. 项目要求与选题原则

依据 [project_handout.pdf](<D:/Downloads/SC4001 Project/project_handout.pdf>)（2026-10-10 更新，PDF 第 1–4 页）：

- 最多三人一组，截止时间为 **2026-11-16 23:59，新加坡时间**。
- 从 A–J 选择一个方向，或提出其他相关题目；不需要把十个方向全部实现。
- 工作应包含一个明确的研究问题、可运行的基线，以及有理由的修改、比较或假设检验。仅把现成工具拼成应用不够；有充分证据的负结果也有价值。
- 评分：实验与结果 40%、报告表达 30%、新颖性 20%、同伴评价 10%。因此应优先保证对比实验完整、结论可信，再考虑增加模型规模。
- 提交 PDF 报告和代码 ZIP。正文最多 10 页，A4、单栏、Arial 10 pt；参考文献、目录、封面、附录不计入正文页数。主要结果仍应放在正文。
- 代码需有复现说明，报告需包含 **AI Use Statement**，说明使用的工具、用途、核实方式以及团队自己的决策与贡献。

适合本项目的范围是：**一个主数据集 + 一个主模型 + 一个核心改进 + 两三个必要对照**。不要从零预训练基础模型，也不要同时开展多个独立研究问题。

## 2. 我们可以怎样使用 V100

### 2.1 指南确认的资源，以及尚未确认的额度

依据 [CCDSGPU-TC1-UG-UserGuide.pdf](<D:/Downloads/SC4001 Project/CCDSGPU-TC1-UG-UserGuide.pdf>)（2026-02-12 更新）：

| 项目 | 指南中的信息 | 本文采用的规划 |
| --- | --- | --- |
| GPU | 第 18 页：Tesla V100，每张固定 32GB 显存 | 所有主方案按 **单卡 32GB** 设计 |
| 调度 | 第 14、17–19 页：SLURM，本科分区示例为 `UGGPU-TC1` | 用 `sbatch` 提交可恢复的作业 |
| 普通 QoS | 第 14、16 页示例：1 GPU、20 CPU cores、64GB 主机内存、6 小时 MaxWall | 不假设能获得多卡或超过 6 小时的单次作业 |
| 作业数 | 指南示例允许两个作业，但总 GPU 额度仍为一张 | 不把“两作业”理解成“两张 GPU” |
| 存储 | 第 6 页：按账户分配磁盘额度，可能至少 100GB | 小数据优先；不默认有足够空间放百 GB 数据集 |
| 实际账户限制 | 指南明确说明 QoS 可随学期调整，可用 `MyTCinfo` 等查询 | **本轮未登录查询，当前 QoS、累计额度及排队时间均未确认** |

6 小时的 MaxWall 是单次作业时限，不代表整个项目只能使用 6 GPU 小时；累计资源限制需要另行确认。[SLURM 官方资源限制说明](https://slurm.schedmd.com/resource_limits.html)

### 2.2 模型选择的实际边界

- **小模型全量训练：** 1D CNN、小型序列模型、ResNet-18、DistilBERT 等通常适合。预训练分类模型优先控制在约 10M–100M 参数。
- **冻结大部分参数：** 蛋白质、视觉与文本编码器先提取特征，再训练小头，能够把重复实验的 GPU 成本降下来。
- **小语言模型：** 优先 0.5B–1.5B；确有需要再到 3B。使用 FP16 推理，适配采用 LoRA。7B 即使权重能装进显存，也会减少实验次数并增加实现风险，先不作为主方案。
- **图像生成：** Stable Diffusion 1.x、512×512、LoRA 是可考虑的上限路线；本项目不建议优先选择 SDXL、FLUX 或视频生成模型。
- **V100 是 Volta：** 使用 FP16/AMP；不要直接沿用 BF16 默认配置。BF16 Tensor Core 支持来自后续的 Ampere 架构。[NVIDIA 架构说明](https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/)
- **注意力实现：** 优先兼容的 PyTorch 普通注意力/SDPA 后端，必要时用 eager。FlashAttention-2 的官方 CUDA 支持列表不包含 V100，不能将其作为必要依赖。[FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
- **环境兼容性：** 选择包含 `sm_70` 支持的 PyTorch/CUDA 构建，不能无条件安装最新组合。例如 PyTorch 2.11 的 CUDA 12.8/12.9 预编译包已移除 Volta，官方建议该情况下使用 CUDA 12.6 构建。[PyTorch 发布说明](https://github.com/pytorch/pytorch/releases)
- **量化不是必需项：** 小模型先用 FP16，减少依赖。若确需 QLoRA，再验证 NF4、CUDA 和算子兼容性，并使用 FP16 计算；不能默认所有 8-bit/4-bit 实现都支持 V100，也不能把省显存直接当成更快。[bitsandbytes 硬件兼容说明](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

下面的 **显存是粗略估计，batch 是起步配置，GPU 小时是建议预留的实验预算，不是已测量耗时或完成保证**。M 表示百万参数，B 表示十亿参数。参数总量与可训练参数量分开列出；实际数量取决于分类头、词表、LoRA 插入位置等。预算均针对选中一个方向，包含有限开发、主对比与评价，不包含排队时间，不能把十行预算相加后当成必须开展的工作。

## 3. 十个方向总览

| 方向 | 建议主数据集 | 主模型与参数规模 | 显存估计 | 建议预留 GPU 小时 | 综合判断 |
| --- | --- | --- | --- | --- | --- |
| A ECG 心拍分类 | MIT-BIH Arrhythmia | 1D CNN，约 0.1–0.3M，全部训练 | 1–3GB | 2–6 | 算力最省，重点在患者划分与少数类 |
| B 蛋白质二级结构 | TAPE SS3 + CB513 | 冻结 ESM2-8M；头约 0.001–0.2M | 4–8GB | 6–12 | 可行，需理解逐残基标签与同源性 |
| C 情感分析 | IMDb | DistilBERT 约 66M；训练头、最后一层或全模型 | 4–10GB | 8–16 | **优先推荐，开发与实验都较稳妥** |
| D 图像描述 | COCO2017 固定 8k 图子集 | 冻结 ResNet-50 骨干约 23.5M + GRU/attention 约 8–10M | 4–8GB | 8–16 | 可行，评价与数据准备比分类复杂 |
| E VLM 幻觉与可靠性 | POPE + 对应 COCO 图像 | SmolVLM-500M，约 0.5B；主模型不训练 | 3–8GB | 4–12 | **生成式 AI 中较轻的选题** |
| F 可靠 RAG | HotpotQA distractor | Qwen2.5-1.5B，实际 1.54B；LoRA 约 2.18M | 12–22GB | 12–24 | 研究空间清楚，证据标注需认真设计 |
| G 小语言模型高效推理 | GSM8K + SVAMP 固定子集 | Qwen2.5-1.5B，实际 1.54B；不训练 | 6–12GB | 12–28 | 可行，生成次数决定成本 |
| H 可控/个性化图像生成 | DreamBooth 的 6 个主体 | SD1.5 管线约 1.0B；U-Net LoRA 约 1–5M | 12–24GB | 12–30 | 能做，但训练与人工评价都较费时 |
| I 基础模型领域适配 | Office-Home | 冻结 CLIP ViT-B/32 约 151M；adapter 约 0.13M | 4–8GB | 4–10 | **优先推荐，适合做充分的对比实验** |
| J 跨生成器 AI 图检测 | GenImage 按生成器取子集 | ResNet-18 二分类版约 11.2M，全部训练 | 6–12GB | 8–20 | 模型不大，但数据下载与来源偏差风险最高 |

**选题建议：** 如果主要目标是在约五周内完整交付，优先选 **C、I、A**；如果希望做生成式 AI，优先看 **E、F**。B 适合对生物序列有兴趣的组，D 适合想研究视觉到语言生成的组。G 要先确认推理吞吐，H 要愿意投入评价工作，J 要先通过数据取得与存储门槛。这是基于工作量和可解释性的建议，不是对得分或最终效果的保证。

## 4. 各方向的具体方案

### A. ECG heartbeat classification：少数类心拍识别

**建议题目：加入前一心拍的 RR 间隔，能否改善跨患者的少数类识别？**

- **数据：** MIT-BIH Arrhythmia 原始数据，48 条约半小时、双导联记录，来自 47 名受试者，360Hz，约 11 万条心拍标注。数据开放且体积小。使用一个固定导联和 R 峰附近约 300 个采样点。主任务采用 N/S/V/F 四类；Q 类及起搏记录的处理规则须事先固定，并报告实际纳入的记录、每类心拍数与排除数量。[PhysioNet 数据说明](https://physionet.org/content/mitdb/1.0.0/)
- **模型：** 三四层小型 1D CNN，约 0.1–0.3M 参数。基线只输入波形；改进在分类头拼接标准化的前一 RR 间隔。两者使用同一波形编码器，不需要 Transformer 或大模型。RR 归一化统计只从训练集取得。
- **最小对比：** 波形 CNN、波形 + RR、参数量近似匹配的波形分类头。三者采用相同损失和训练设置；如使用类别加权，各组都使用同样的权重。这样能区分“额外节律信息”与“多一点参数”的效果。
- **评价：** macro-F1、各类 precision/recall，重点查看 S/V/F 类；补充混淆矩阵、每位测试患者的表现及少数类支持数。Accuracy 不能作为唯一结论。[经典心拍分类研究](https://pubmed.ncbi.nlm.nih.gov/15248536/)
- **划分与边界：** 先按患者划分，再切心拍。**201 和 202 来自同一患者，必须同分区**；不能随机拆分心拍，也不能直接把记录编号当独立患者。本题使用标注 R 峰，结论只支持“已定位心拍的分类”，不支持端到端心拍检测。
- **起步与预算：** batch 128，最多约 20–40 epochs，按验证集早停；显存约 1–3GB，预留 2–6 GPU 小时。最终少数条件跑 3 seeds。可选的噪声扩展放到主实验完成之后。

**本组贡献可以是：** 在严格跨患者划分下，解释 RR 信息究竟改善哪些少数类、哪些患者仍然失败，而不只是报一个总体准确率。

### B. Protein secondary structure prediction：蛋白质逐残基分类

**建议题目：冻结蛋白语言模型后，局部卷积能否比逐位分类头更好地预测 SS3？**

- **数据：** TAPE secondary structure 的公开任务数据。论文给出 8,678 条训练蛋白、2,170 条验证蛋白，CB513 的 513 条作测试。官方提供 JSON/LMDB 下载，源码直接读取逐残基 `ss3` 标签，能预测 helix/strand/coil 三类，无需自行从 PDB 重建标签。[TAPE 数据入口](https://github.com/songlab-cal/tape)、[标签读取源码](https://github.com/songlab-cal/tape/blob/master/tape/datasets.py)、[原论文](https://arxiv.org/abs/1906.08230)
- **模型：** `facebook/esm2_t6_8M_UR50D`，约 8M 参数，冻结；每个残基获得 320 维表示。线性头约 1K 参数，浅层 1D 卷积头约 0.1–0.2M。若第一阶段顺利，再用冻结的 35M 版本做一次规模复核，先不全量微调。[ESM 官方模型规格](https://github.com/facebookresearch/esm)
- **最小对比：** 同一批缓存表示上的线性头、逐位 MLP、局部卷积头。MLP 与卷积头尽量匹配参数量，以判断收益来自邻域信息还是头的容量。
- **评价：** Q3 accuracy、三类 macro-F1；同时报告全体有效残基的汇总和逐蛋白平均表现。Padding 与无效标签不计入指标。同一蛋白的许多残基不能当成许多独立实验重复。
- **划分与边界：** 沿用任务的监督拆分，不重新随机拆残基。TAPE 报告训练与测试之间做了 25% 序列一致性过滤；这不能排除 ESM2 预训练语料与测试蛋白的重合。只主张固定预训练表示下的比较。
- **起步与预算：** 编码时 batch 4–8，按序列长度分桶；超长序列使用固定的重叠窗口并拼回原序列，不能默默丢掉尾部标签。先缓存残基表示，后续只训练小头。显存约 4–8GB，预留 6–12 GPU 小时；最终主要两三个头跑 3 seeds。

**本组贡献可以是：** 在冻结表示下研究局部结构信息的剩余价值，并按蛋白长度、二级结构类别分析失败。难点主要在数据与评价语义，算力并不是主要瓶颈。

### C. Sentiment analysis：少标签条件下的情感分类

**建议题目：训练标签较少时，局部解冻能否比全量微调更稳定，并降低训练成本？**

- **数据：** Stanford IMDb Large Movie Review：25,000 条有标签训练评论、25,000 条测试评论，另有 50,000 条无标签评论；原始下载包约 84MB。先只做二分类与少标签问题，暂不叠加领域迁移。[数据作者页面](https://ai.stanford.edu/~amaas/data/sentiment/)
- **模型：** 同一个 `distilbert/distilbert-base-uncased`，约 66M 参数。比较冻结编码器 + 约 1.5K 参数的线性头、只解冻最后一个 Transformer block + 分类头（约 7M 可训练）、全量微调（约 66M 可训练）。传统 TF-IDF + logistic regression 可作为廉价参考。[DistilBERT 原论文](https://arxiv.org/abs/1910.01108)、[官方规格](https://github.com/huggingface/transformers-research-projects/blob/main/distillation/README.md)
- **最小对比：** 固定分层验证集，例如 500 条，再从剩余训练候选中构造嵌套的 250/1,000/4,000 条训练标签子集。同一 seed 下三个方法使用同一批评论。**训练标签预算与额外的 500 条验证标签分开报告**，不把验证标签成本隐去。先单 seed 开发，最终选两个代表预算跑 3 seeds。
- **评价：** Accuracy、macro-F1、种子间标准差、实际训练 GPU 时间、峰值显存；画出性能与训练成本的关系。分析否定表达、长评论被截断等错误。
- **起步与预算：** 最大长度 512 tokens、batch 16、约 3–5 epochs，验证集选 checkpoint。显存约 4–10GB，预留 8–16 GPU 小时。
- **边界：** 这是训练标签有限时的受控适配比较，不应仅凭更高分声称提出新架构。无标签伪标签可作为后续扩展，先完成监督主实验。

**推荐理由：** 数据简单、成本低、三种训练策略的区别容易解释，也容易得到有价值的负结果，例如全量微调在极少标签时反而波动更大。

### D. Image captioning：小型图像描述模型

**建议题目：冻结视觉编码器时，空间注意力能否改善物体与动作描述？**

- **数据：** 主跑官方 COCO2017 captions 中固定的 8,000 张训练来源图像及对应参考描述，按 image ID 自建 6,000/1,000/1,000 的 train/dev/test。先下载标注，再按选定 ID 的 `coco_url` 取得图片，不必下载整个训练图片包。图像及其全部描述必须同分区；这是本项目子集，**不是 Karpathy 划分或完整 COCO benchmark**。[COCO 官方数据](https://cocodataset.org/#download)、[官方按 ID 下载接口](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
- **模型：** ImageNet 预训练 ResNet-50，去掉最终分类层后骨干约 23.5M 参数，冻结。接 512 hidden 的小型 GRU decoder，词表限制在约 8k，训练参数约 8–10M；总模型约 32–34M。基线使用全局平均特征，改进使用 7×7 空间特征上的加性注意力。[ResNet-50 官方规格](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- **最小对比：** 全局池化、可学习空间 attention、均匀 attention；decoder、词表、训练图片、训练步数和解码方式保持一致。均匀 attention 用来判断可学习的空间选择是否真的有用。
- **评价：** CIDEr 与 METEOR，条件允许再加 SPICE；固定抽取测试集 100 张图，盲评物体/动作正确性和无依据的描述。不能只报告 BLEU。[COCO Caption 论文](https://arxiv.org/abs/1504.00325)、[SPICE 论文](https://arxiv.org/abs/1607.08822)
- **起步与预算：** 缩放/填充到 224×224，保留完整图像，避免裁掉参考描述中的物体；先缓存冻结的 7×7×2048 特征，词表只从训练描述构建。decoder batch 64、句长最多约 32 tokens。显存约 4–8GB，预留 8–16 GPU 小时。主对比 3 seeds，辅助消融可先 1 seed。
- **可选数据与门槛：** Flickr8k（8,000 图、每图五句）规模合适，但本轮原作者下载入口访问受限，不能假设现成链接一定可下载。Flickr30k 作者提供 caption 和图片下载入口，也需确认访问及使用条款。因此两者作为取得许可后的替代或补充。[Flickr8k 作者页](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html)、[Flickr30k 作者页](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)

**本组贡献可以是：** 分析空间注意力是否减少物体/动作错误，还是只改变语言流畅度。小数据与预训练图像可能重合，不能声称完全排除了预训练污染。

### E. Hallucination and Reliability in VLMs：视觉问答的选择性拒答

**建议题目：图像分辨率下降时，校准后的拒答能否减少错误肯定，同时保留有用回答？**

- **数据：** POPE 的 COCO 物体存在性问题，包含 random/popular/adversarial 三种负例策略。官方默认 500 张图，每策略约 3,000 道 yes/no 问题。按 image ID 取 100 张作校准、400 张作最终测试，同图的所有问题及退化版本一起划分；因此最终报告本地 POPE 子集结果。[POPE 作者仓库](https://github.com/RUCAIBox/POPE)、[原论文](https://arxiv.org/abs/2305.10355)
- **模型：** `HuggingFaceTB/SmolVLM-500M-Instruct`，约 0.5B 参数；冻结做推理，不微调基础模型。V100 用 FP16 和兼容注意力后端。模型卡的低显存数字来自其特定配置，不能直接视为我们的 batch 峰值。[SmolVLM 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- **方法与对照：** 基线必须选择 Yes/No；改进对两个候选回答计算一致的序列分数，margin 小于校准阈值时输出 abstain。只在校准图上选阈值。对照为未校准的固定阈值，并加入 always-no/always-abstain 作为检查指标含义的参考。
- **评价：** 原始强制二分类的 precision/recall/F1 单列；选择性方法另列 coverage、已回答准确率、全样本错误肯定率、全体正例中被正确肯定的比例和 risk–coverage 曲线。**拒答不能映射成 No，也不能只靠少回答宣称改善。** 与基线尽量在相同覆盖率或相同错误风险下比较。
- **起步与预算：** 原图与预先定义的 256/128 像素长边退化配对比较，模型 processor 的设置保持一致；batch 从 4 起，提示尽量短。显存约 3–8GB，预留 4–12 GPU 小时。不需要把确定性推理重复三次充当独立训练重复。
- **边界：** POPE 主要支持物体存在性，不足以验证属性、关系或自由 caption 的全部幻觉。COCO 图像可能出现在预训练数据中，结论限定为此模型、此题型、此退化协议。

**推荐理由：** 可把主要资源投入指标设计和配对分析，适合想做 VLM、但不想承担大模型训练成本的组。

### F. Reliable RAG：证据不完整时的回答与拒答

**建议题目：加入支持句监督，能否让小语言模型在证据缺失时更可靠地拒答？**

- **数据：** HotpotQA distractor，官方 train/dev 为 90,447/7,405 题，下载约 535/44MB，提供候选段落、答案和支持句标注。使用作者给出的十个候选段落做句子检索，明确这是候选集内的可靠性研究，无需 fullwiki。可先从 train 取约 2,000 道训练问题、另取 500 道本地验证问题，最终在 dev 固定抽取约 1,000 道题。原始问题的所有改写必须同分区。[HotpotQA 官方数据与许可](https://hotpotqa.github.io/)、[字段与数量](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- **模型：** `Qwen/Qwen2.5-1.5B-Instruct`，实际 1.54B 参数。BM25 提取 top-12 候选句，并保留稳定的句 ID。基线冻结 LLM；改进仅训练 q/k/v/o 的 rank-8 LoRA，约 2.18M 可训练参数，生成短答案与支持句 ID，关键证据不足时输出 `INSUFFICIENT`。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **研究协议：** 对同一题配对构造完整支持、缺少一个关键支持、全部关键支持缺失三个条件；截断后必须重新确认给到模型的证据条件。缺少金标支持句不自动等于无法回答，还需检查剩余句子有无替代的证据链，再确定拒答标签。消融保持 LoRA、训练量与拒答监督一致，只去掉支持句 ID 监督，隔离“证据定位监督”的贡献。
- **评价：** 全样本答案 EM/F1，拒答在原问题答案指标中计为零；另列证据缺失条件下的正确拒答率、错误拒答率、拒答 precision/recall、引用支持句 precision/recall，以及 coverage 和已回答质量。依据金标支持句与抽样人工核查评价，不把“检索相关”直接当成“支持答案”。
- **起步与预算：** 输入最多 1,024 tokens，microbatch 1–2、梯度累积到约 16，生成最多 64 tokens，FP16 + gradient checkpointing，约 1–2 epochs。显存约 12–22GB，预留 12–24 GPU 小时。主方法可用 3 seeds，必要消融先 1 seed。
- **冲突与边界：** 可把实体/数值替换形成矛盾证据，作为小型、经人工检查的合成压力测试；仅凭相关度或支持句覆盖不能保证识别事实冲突。模型可能记住 Wikipedia/公开问题，因此重点比较对给定证据变化的敏感性，不主张无污染的新知识泛化。[HotpotQA 原论文](https://arxiv.org/abs/1809.09600)

**本组贡献可以是：** 把支持句监督与拒答效果联系起来，解释模型在部分证据下何时仍会编造答案。这超过了单纯搭建一个 RAG 聊天界面。

### G. Efficient Reasoning with Small LMs：固定预算下的推理策略

**建议题目：相同生成预算下，一条长推理与多条短推理投票，哪种更划算？**

- **数据：** GSM8K 作者数据有 7,473 道训练题、1,319 道测试题；SVAMP 有 1,000 道挑战题。开发和解析规则只用训练题；规划最终固定 GSM8K 测试中的 500 题和 SVAMP 的 200 题，两个子集在实验前选定，报告其 ID 和数量。这是预算受限的子集评价，不声称完整 benchmark 分数。[GSM8K 作者仓库](https://github.com/openai/grade-school-math)、[SVAMP 作者仓库](https://github.com/arkilpatel/SVAMP)
- **模型：** 固定 `Qwen/Qwen2.5-1.5B-Instruct`（1.54B），FP16，不训练。和 F 使用同一模型系列可以降低环境准备成本，但两个方向仍是替代选题。
- **最小对比：** 每题总输出上限 512 tokens：1×512、2×256、4×128，并分别对短推理答案进行多数投票。固定提示与采样参数，要求简短计算和明确的最终答案格式；平票规则预先固定。另给廉价 greedy 作为参考。[Self-consistency 原论文](https://arxiv.org/abs/2203.11171)
- **公平性：** 上限相同并不等于实际成本相同。累计所有候选的实际输出 tokens、重复输入 prefill、投票与额外调用的墙钟时间；同时画 accuracy–token 和 accuracy–GPU-time 曲线。截断且没有最终答案的输出计错，另报截断率，不能悄悄补生成。
- **起步与预算：** 输入最多约 512 tokens，batch 1–4，显存约 6–12GB，预留 12–28 GPU 小时。先用训练题做吞吐与截断率试跑，再冻结最终评测规模；最终随机生成条件用 3 sampling seeds。
- **边界：** 不调用昂贵闭源教师、不训练超大推理模型。公开数学题可能已被预训练模型见过，结果解释为固定小模型在公开题集上的预算分配表现。

**本组贡献可以是：** 找出短推理投票真正划算的预算区间，以及预算过紧、候选高度相关时为何失效。

### H. Controllable and Personalised Image Generation：少样本主体个性化

**建议题目：固定少量主体照片时，先验保持能否减少 LoRA 个性化的过拟合？**

- **范围：** 选择主体身份保持一条主线，本项目先不同时研究空间控制。避免把“加上 ControlNet 能生成图片”作为唯一贡献。
- **数据：** DreamBooth 作者数据包含 30 个主体，每个有约 4–6 张照片。固定选 6 个动物/物体主体，每个用 3 张训练，其余照片作为身份评价参考；训练与测试提示词分开。使用原数据的类别名，并核对逐图使用条款。[DreamBooth 作者数据](https://github.com/google/dreambooth)
- **模型：** SD1.5，完整管线约 1.0B 参数；冻结 VAE、text encoder 和原 U-Net 权重，只训练 rank 4 或 8 的 U-Net LoRA，估计约 1–5M 参数。当前模型入口是旧 Runway 仓库的镜像，需固定使用版本，不能当成从零训练的新模型。[SD1.5 模型卡](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)、[Diffusers 训练说明](https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/dreambooth.md)
- **最小对比：** 相同 LoRA、训练步数和主体图，比较 prior-preservation loss 关闭/开启。可另取 1–2 个开发主体选设置，最终六个主体不用于选择超参数。类图生成、训练和生成评价图的成本全部计入预算；先验保持组如果每步额外处理类图，需记录其额外计算量。
- **评价：** 与保留参考照片的 DINO/CLIP 图像相似度、prompt adherence、多样性和固定样本的盲评；同一 prompt 使用相同生成 seeds。额外查看复制训练图、背景不变和主体漂移。相似度只是身份保持的代理，少量生成图的 FID 不宜作主结论。
- **起步与预算：** FP16、512×512、batch 1，gradient checkpointing，每主体约 200–400 optimizer steps。每主体固定约 10 个新提示词、每提示词 4 个生成 seeds；主对比使用固定 3 个训练 seeds，按主体和 seed 拆作业。显存约 12–24GB，预留 12–30 GPU 小时。
- **边界：** 实际评价单位是主体和提示词；同一主体生成几百张图不等于几百个独立主体。只选择少量主体时，不宣称方法已泛化到所有人、动物和物体。

**主要风险：** 算力基本可控，但很容易出现“展示几张好图、没有系统评价”的项目。若选 H，应先确定评价协议再开始训练。

### I. Adapting Foundation Models to Unseen Domains：少标签领域适配

**建议题目：少量目标域标签下，保留原始 CLIP 特征能否减少适配带来的负迁移？**

- **数据：** Office-Home，约 15,500 张图、65 类，四域为 Art、Clipart、Product、Real-World；作者提供学术研究下载。每轮留出一个目标域，用另外三个域的开发数据确定训练设置。目标域每类只使用固定 4 张有标签支持图，共 260 张，剩余目标域图像只作最终测试；目标标签的这部分使用要明确报告。[Office-Home 作者页](https://www.hemanthdv.org/officeHomeDataset.html)
- **模型：** 冻结 CLIP ViT-B/32，完整图文模型约 151M 参数、图像表示为 512 维，224×224。接 512→128→512 的小型残差 adapter，约 0.13M 可训练参数；类别文本原型冻结。[CLIP 模型卡](https://huggingface.co/openai/clip-vit-base-patch32)、[CLIP-Adapter 原作者实现](https://github.com/gaopengcuhk/CLIP-Adapter)
- **最小对比：** CLIP zero-shot、只用适配特征（混合权重 α=1）、保留原始特征的残差混合（例如固定 α=0.2）。两个 adapter 条件使用相同 4-shot 支持集、初始化、训练步数和可训练参数。混合权重预先固定或仅从来源域开发数据选择，不能用目标测试标签调参。
- **评价：** 各目标域的 accuracy、balanced accuracy、macro-F1；相对 zero-shot 的增益，以及适配后下降的域和类别。报告四域分别的结果与等权平均，分析哪些原本正确的预测被适配改错。
- **起步与预算：** 先缓存冻结的 image/text embeddings，adapter batch 64–128，训练约几百步。显存约 4–8GB，预留 4–10 GPU 小时。四个目标域用固定 3 个支持集/初始化 seeds 作配对比较；无需全量微调 CLIP。
- **边界：** 这里实际用了少量目标域标签，属于 few-shot domain adaptation。只用来源域训练、完全不接触目标域数据则属于 domain generalization，二者不能混称。CLIP 预训练图文可能覆盖这些物体或图像，“未见域”指项目适配前的目标域，不保证基础模型从未见过它们。

**推荐理由：** GPU 成本低，能完成四个域的充分对比；“适配何时有害”也符合 handout 对局限和负结果的要求。采用已有 adapter 方法时，本组贡献在受控的领域适配研究与分析。

### J. Detecting AI Generated Images Across Generators：跨生成器检测

**建议题目：统一真假图像的压缩与缩放增强，能否改善未见生成器上的检测？**

- **数据：** GenImage，约 270 万张真实/生成图片，1,000 个类别、8 种生成器，真实图来源为 ImageNet。只在一个生成器（例如 SD1.4）的官方训练分区训练，其他生成器整体留出。可以预先按类别固定抽取约 20k real + 20k fake 训练，以及每个留出生成器约 2k real + 2k fake 测试；各方法共用相同 ID 与真实测试池。[GenImage 作者仓库](https://github.com/GenImage-Dataset/GenImage)、[NeurIPS 数据集论文](https://papers.neurips.cc/paper_files/paper/2023/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf)
- **下载门槛：** 官方链接按生成器给出压缩包入口；**本轮没有确认包大小或能否仅取选定文件**。训练只用 40k 图不代表只需下载 40k 图。先确认下载、解包后的容量，以及能否取得至少两个独立生成器的测试数据；不满足时应降低 J 的优先级，而不是默默用单生成器替代。
- **模型：** ResNet-18 从零训练，最终分类层改为二分类后约 11.2M 参数，全部训练。Torchvision 原始 1000 类模型是约 11.69M，二者参数量不能混记。无需引入大 VLM。[ResNet-18 规格](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- **最小对比：** 常规训练 vs 对真假两类同分布施加随机 JPEG 重编码和缩放的训练；增加只做 JPEG 的消融。模型、样本数和训练步数固定，测试同时看原图、统一重编码、统一缩放，检查检测器是否利用文件格式/尺寸捷径。[压缩与尺寸偏差的研究](https://arxiv.org/abs/2403.17608)
- **评价：** 每个未见生成器的 AUROC、balanced accuracy、固定阈值下 TPR/FPR，以及生成器间等权平均；阈值只在已见生成器验证集上选。同一 ImageNet 图像若在不同目录复制出现，需按原图 ID 保持分区，不必为此做全盘哈希。
- **起步与预算：** 224×224，batch 从 32 起，约 10–20 epochs；显存约 6–12GB，预留 8–20 GPU 小时。正式主对比使用 3 seeds。真实图与假图不是逐图配对生成，来源偏差应在讨论中保留。
- **轻量备用：** CIFAKE 为 32×32 的 CIFAR-10 real 与 SD1.4 fake，100k train、20k test，可快速试跑流程。但它本身不能回答跨生成器问题，只能作为准备阶段数据。[CIFAKE 作者仓库](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images)

**主要风险：** 不是模型装不进 V100，而是数据取得与真假来源偏差。跨生成器高分需要证明测试生成器确实留出，而非随机混合划分。

## 5. 选定方向后的最小执行计划

### 5.1 共同的实验规则

1. **先冻结问题和划分。** 固定样本 ID、主指标、基线、主改动与少量消融；训练、阈值选择和模型选择均不使用测试标签。同一原始样本的变体一起划分。
2. **做一次短试跑。** 在计算节点用约 100–200 个训练样本或少量 steps 测吞吐、显存与 checkpoint 恢复。根据实测一次性调整正式规模，再冻结正式协议。
3. **控制重复数量。** 少量超参数用单 seed 开发；最终核心比较再跑 3 seeds。确定性推理不机械重复三次。保留失败样本和失败条件，不只挑最好的 seed。
4. **计入全部成本。** 缓存特征、类图生成、LoRA 训练、多候选推理与评价模型的时间分别记录；缓存一次后的低成本不能冒充整个流程的总成本。
5. **保存可恢复的进度。** 训练保存权重、优化器、scheduler、随机状态与进度；长推理按样本及实验条件保存完成项。约每 20–30 分钟或每 epoch 保存，单作业尽量预留结束时间。
6. **按指南使用集群。** TC1 使用 SLURM。登录节点只连接、提交、查询和做轻量控制；数据下载、解包、依赖安装、训练和评价放在合适的计算 allocation，准备工作优先 CPU allocation，不在本机跑研究计算。
7. **记录真实 GPU 使用。** 作业期间约每 30–60 秒采集获分配 GPU 的利用率与显存使用，写入该作业 log；完成后据实际记录调整资源申请。不申请闲置 GPU。

这里只需要必要的数据划分检查与结果复现信息，不进行多余的全盘审计或哈希检查，也不使用账户里的 banked reset。

### 5.2 从现在到提交的安排

| 时间（新加坡） | 建议完成的工作 |
| --- | --- |
| 10 月 13–18 日 | 从十个方向选一个；取得数据与模型；确认账户 QoS；完成小试跑和基线 |
| 10 月 19–25 日 | 实现单一核心改动，冻结主实验与评价样本 |
| 10 月 26 日–11 月 1 日 | 运行主对比，查看失败案例并做必要消融 |
| 11 月 2–8 日 | 完成正式重复实验、成本记录和结果图表 |
| 11 月 9–15 日 | 写报告、整理代码与复现步骤、检查主要结果能否重现 |
| 11 月 16 日 | 留出上传缓冲，在 23:59 前提交 |

### 5.3 选题时最值得先确定的三件事

- **数据能否实际取得？** 尤其是 D 的 Flickr、I 的托管下载以及 J 的大压缩包；网页有入口不等于已成功下载。
- **一个变化是否能被单独解释？** 优先比较同一模型中的信息、监督或预算分配变化，避免同时换模型、换数据、换训练量。
- **结果没有提升时是否仍然有内容？** C 的微调波动、E 的覆盖率代价、F 的部分证据失败、I 的负迁移，都可以支撑完整讨论。

## 6. 来源与 AI 辅助说明

上面的精确数据规模、公开模型规格与硬件兼容信息已对照所链接的作者/官方来源；实验问题、模型组合、子集大小与预算是本项目的建议，不是这些来源报告的实验结果。数据下载成功、V100 实际显存和吞吐、当前账户额度仍待执行时确认。

本选题草案使用 Codex 协助阅读材料、检索来源与设计方案，采用 ponytail 技能控制实现范围，并参考 experimental-design 技能组织配对比较、划分和评价。正式报告的 AI Use Statement 应根据团队之后实际使用的工具与验证过程更新。

实验设计技能的参考来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。这条引用说明辅助工作流的来源，不作为任何模型效果或算力估算的证据。
