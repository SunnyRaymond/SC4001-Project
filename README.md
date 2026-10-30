# SC4001 Project（方向 I）：少样本 CLIP 适配何时有害？

更新：2026-10-30。已选定 handout 方向 I（Adapting Foundation Models to Unseen Domains）。本文是研究方案与执行计划：**尚未下载数据、运行模型或提交集群作业**，文中的"预期"都是待验证的假设，不是结果。方括号编号 [n] 对应第 8 节的参考文献。

## 0. 一页摘要

- **问题：** 用每类 K 张目标域标注图把冻结的 CLIP 适配到新域时，哪些测试图被"改对"、哪些被"改错"？被改错的图能否由 zero-shot 置信度提前识别并避免？
- **做法：** Office-Home 四个域轮流作目标域。在缓存的 CLIP 特征上比较 zero-shot、两种线性探针、CLIP-Adapter、Tip-Adapter；把每种适配的净增益拆成 help（错→对）与 harm（对→错）；按 zero-shot 置信度分箱分析；据此检验逐样本置信度门控，并与"全局混合"和"保留率相同的随机门控"对照。
- **防泄漏：** 所有超参只在另外三个来源域上模拟同样的任务来选，目标域测试标签只在最终评估时用一次。
- **与已有工作的关系：** VRF [11]、DaWin [12] 已在大数据量的鲁棒微调中逐样本组合 zero-shot 与微调模型。本组研究的是：每类只有 1–8 张标注、又没有目标域验证集时，这种组合是否仍然有效，以及原因。
- **成本：** GPU 只用于一次性特征提取；其余实验都在 512 维缓存特征上运行，单组配置秒级完成，因此能把 域 × K × seeds × 方法 全部跑满。

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
| RQ1 harm 有多大，随 K、域、适配方法如何变化？ | 测量 | help/harm 分解，覆盖 4 域 × 多个 K × 4 种适配 |
| RQ2 help 与 harm 在 zero-shot 置信度上是否可分？ | 诊断 | 按置信度分箱统计 help/harm/净增益 |
| RQ3 逐样本置信度门控能否比全局混合更好地去掉 harm、保住 help？ | 干预 | 在同一份适配输出上比较门控、全局混合、随机门控 |

**动机依据：** CLIP 论文 [1] 发现 zero-shot 大致相当于 4-shot 线性探针，并把"更好地结合 zero-shot 与 few-shot"列为未来方向；LP-FT [7] 和 CLAP [8] 都报告过适配在分布偏移下可能不如原模型。

### 2.3 预先写定的假设与判定标准

先写定判定标准再看结果，避免事后挑结论。

- **H1（harm 不可忽略）：** K ≤ 4 时，适配的净增益明显小于其 help 率，即相当一部分增益被 harm 抵消；在 zero-shot 已经较强的域上，净增益可能 ≤ 0。
  *不成立的条件：* 所有（方法, 域, K ≤ 4）组合的 harm 率都 < 1%，且净增益都 > 0。
- **H2（harm 可由置信度预测）：** 按 zero-shot 最大 softmax 概率（MSP，可作为预测是否出错的信号 [10]）把测试图分成 10 个等频箱，help/harm 比随置信度升高而下降；置信度最高的 20% 样本中，适配的净增益 ≤ 0。
  *不成立的条件：* 最高置信 20% 的净增益在多数 seeds 下 > 0，或分箱趋势在 seeds 间不稳定。
- **H3（门控有效）：** 在同一份适配输出上，置信度门控（阈值只在来源域上选）的平均准确率不低于全局混合（权重也在来源域上选），harm 率更低，并且优于保留率相同的随机门控。
  *不成立的条件：* 门控相对全局混合的配对平均差 ≤ 0，或不优于随机门控。

**决策点：**
- 若 H2 不成立，门控就没有动机，不硬做；改为如实报告负结果，并检查其他可能预测 harm 的信号，例如类别文本原型之间的间隔、支持集与测试图的特征距离（后者即 VRF [11] 的思路，见 3.6）。
- 若 H1 不成立（适配几乎不伤害），主线改为"增益如何分布在各域、各类别"，门控仍作为一项检验保留。

### 2.4 贡献与边界（写报告时按此表述）

**我们主张：**

1. 一个严格的受控测量：在少样本、只用来源域选参的设定下，把 CLIP 适配的增益拆成 help/harm（harm 即 Yan et al. [13] 定义的 negative flip），并在域、K、适配方法、backbone 上对照。文献检索中没有找到针对少样本 CLIP adapter 的同类翻转分析（检索范围见 8.3）。
2. 检验"按 zero-shot 置信度逐样本决定是否采用适配结果"在少样本设定下是否仍然有效，并用全局混合、随机门控、oracle 上界做受控对照。

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
- **K=16：** Art 平均每类只有约 37 张，因此仅当每个域每个类别都至少有约 20 张图时才加入（K_max 改为 16）；拿到数据后先统计每类数量再决定。
- **选参协议（leave-target-out，参照 DomainBed [9] 的训练域验证思路）：**
  - 目标域 d 的所有超参都在三个来源域上选：学习率、步数、α、β、门控阈值 τ、全局混合权重 w。
  - 做法是在每个来源域上模拟同样的 K-shot 任务（3 个 seeds），取来源域平均准确率最高的配置。
  - d 的测试标签只在最终评估时使用一次。
- **预期偏低：** CLAP [8] 指出，CLIP-Adapter、Tip-Adapter 原文的超参是按数据集搜索的，部分甚至在测试集上搜索。因此本组适配器的数字可能低于原论文报告值，这是协议更严格的预期结果，不是实现错误，报告中需要说明。
- **固定在先：** 类名只做"下划线/连字符 → 空格、小写化"；prompt 模板、超参网格、seeds 都在看到任何测试结果前写入 `configs/` 并提交 git。

### 3.2 模型与特征

- **主 backbone：** CLIP ViT-B/32（OpenAI 权重）[1]，完全冻结；图像和文本特征均为 512 维，L2 归一化。
  - 加载方式：`open_clip_torch==3.3.0`，`open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')`。
  - 权重来自 Hugging Face 的 `timm/vit_base_patch32_clip_224.openai`。
- **必须使用 `-quickgelu` 版本**，或传入 `force_quick_gelu=True`。OpenAI 权重使用 QuickGELU，open_clip 3.3.0 不会自动切换；用错会出现警告并掉点。
- **复核 backbone：** CLIP ViT-B/16，使用 `ViT-B-16-quickgelu` 与 `timm/vit_base_patch16_clip_224.openai`，流程相同，只换特征。
- **预处理：** 使用 open_clip 返回的 CLIP 预处理（resize + center crop 到 224×224）。
- **缓存：** 每张图只提特征一次，连同域、类别、在镜像中的索引一起保存；之后所有实验只读缓存。
- **V100 注意：** FP16 推理即可；不需要 BF16 或 FlashAttention。

### 3.3 方法与对照

记 zero-shot 概率为 p_zs，适配后的概率为 p_ad，zero-shot 置信度 c(x) = max p_zs（使用 CLIP 默认 logit scale）。

| 编号 | 方法 | 可训练参数 | 在来源域上选的超参 | 作用 |
| --- | --- | --- | --- | --- |
| B0 | Zero-shot，模板 "a photo of a {c}." | 0 | 无 | 主参照；help/harm 都相对它计算 |
| B1 | Zero-shot + 域提示（如 "a clipart image of a {c}."，模板预先固定） | 0 | 无 | 不用标签的廉价适配 |
| A1 | 线性探针（随机初始化、L2 正则的 logistic regression） | 约 33K | 正则强度 | 经典基线 |
| A2 | 线性探针，以 zero-shot 文本原型初始化 | 约 33K | 学习率/步数、正则 | 从 zero-shot 出发的适配；相当于不加约束的 CLAP [8] |
| A3 | CLIP-Adapter [4]：f* = α·A(f) + (1 − α)·f，A 为 512→128→512 MLP | 约 0.13M | 学习率/步数、α | 训练式 adapter |
| A4 | Tip-Adapter [5]：logits = α·exp(−β(1 − f·Fᵀ))·L + f·Wᵀ | 0 | α、β | 免训练 adapter |
| G1 | **硬门控：** c(x) ≥ τ 时用 B0 的预测，否则用适配预测 | 0 | τ | 本组检验的方法 |
| G2 | **软门控：** p = c(x)·p_zs + (1 − c(x))·p_ad | 0 | 无 | 无超参变体 |
| C1 | 全局混合：p = w·p_zs + (1 − w)·p_ad | 0 | w | 对照：去掉"逐样本" |
| C2 | 随机门控，保留 zero-shot 的样本比例与 G1 相同（重复 20 次取平均） | 0 | 无 | 对照：去掉"置信度信号" |
| C3 | Oracle：B0 与适配任一正确即算对 | — | — | 只作上界，不是方法 |

- G1、G2、C1、C2 分别作用在 A1–A4 每一种适配输出上，用来检验门控是否对不同的适配方法都起作用。
- 主分析以 B0 为参照；如果 B1 明显更强，再以 B1 为参照复核一次。
- c(x) 未经校准。Kumar et al. [14] 讨论过组合前校准的作用；若时间允许，在来源域上拟合温度后再组合，作为 G2 的变体。

### 3.4 评价指标

- **主指标：** 目标域测试集 top-1 accuracy；辅以 macro-F1，因为各类样本数不均衡。
- **help 率** = B0 错、适配对的样本比例；**harm 率** = B0 对、适配错的比例，即 negative flip [13]；Δacc = help − harm。
- **条件 harm 率** = harm 数 / B0 正确数，即"原本对的有多少被改错"。
- **置信度分箱：** 按 c(x) 分 10 个等频箱，每箱报告 help、harm 与净增益。
- **统计：** 每个（方法, 域, K）报告 5 seeds 的均值 ± 标准差。主要比较（G1 对未门控、对 C1、对 C2）报告配对差值和赢/平/输计数；代表性设置（如 K=4）对逐图预测做 McNemar 检验。
- **失败分析：** 找出 harm 最多的类别；每个域抽样展示被改错的图；分析门控仍然失败的样本，例如高置信但 zero-shot 错误的图。

### 3.5 实验矩阵与预算

- **主矩阵：** 4 个目标域 × K ∈ {1, 2, 4, 8} × 5 seeds × 4 种适配 × {未门控, G1, G2, C1, C2}，另加 B0、B1。
- **选参：** 每个（目标域, K）在 3 个来源域 × 3 seeds 上搜索预先固定的网格。
- **复核：** ViT-B/16 跑相同的主矩阵。
- **GPU：** 两个 backbone 各对 15,588 张图做一次特征提取，主要受图片解码速度限制。粗估每次不到半小时，含调试预留 2–4 GPU 小时；这是估计，需在试跑中实测。其余实验在缓存特征上运行，GPU 或 CPU 都可以。

### 3.6 可选扩展（主矩阵完成后才做，按优先级排序）

1. **VRF 式加权的少样本版本 [11]：** 把支持集中 zero-shot 判错的图作为"失败集"，按测试图到失败集的特征距离决定组合权重。检验在失败集只有几十张时，它是否比 MSP 门控更好。这是"把已有方法适配到新问题"，可增加新颖性。
2. **CLAP [8] 作为强基线：** 它通过约束让类原型不偏离 zero-shot，是避免 harm 的另一条路线，可与门控直接比较。
3. **无标签目标数据：** 把 zero-shot 高置信的无标签目标图作为伪标签加入 Tip-Adapter 缓存，检验它是否同样带来 harm、门控是否仍然有效。这对应 handout 中"unlabelled target data"的表述。
4. **第二个多域数据集：** 复核主要结论；必须先确认可以下载。

## 4. 在 TC1 集群上运行

依据 [CCDSGPU-TC1-UG-UserGuide.pdf](CCDSGPU-TC1-UG-UserGuide.pdf)（2026-02-12 版）。

### 4.1 连接

- [01-SSH-TC1.cmd](01-SSH-TC1.cmd)：SSH 到头节点（校外需先连 NTU VPN）。
- [02-VSCode-TC1.cmd](02-VSCode-TC1.cmd)：用 VS Code Remote-SSH 打开远程目录。目前指向作业目录 `SC4001-Assignment`，建好项目目录后改成项目路径。
- 头节点只用来编辑、提交和查询作业。特征提取、训练、评价都通过 `sbatch` 放到计算节点上运行；不要用 `srun` 跑长任务。

### 4.2 首次设置（头节点）

```bash
MyTCinfo
```

先确认本学期的 QoS。指南中的示例为：1 GPU、20 核、64G 内存、单作业最长 6 小时、最多 2 个作业；QoS 每学期可能调整，以实际查询为准。

```bash
module load anaconda
cp /tc1apps/3_conda_yml/SC4001.yml ~
conda env create -f ~/SC4001.yml -n mySC4001
```

课程提供的环境为 Python 3.12、PyTorch 2.5.1、CUDA 11.8，支持 V100，可以避开新版 PyTorch 不再支持 Volta 的问题。

```bash
conda activate mySC4001
pip install open_clip_torch==3.3.0 "torch==2.5.1" pyarrow pandas scikit-learn scipy matplotlib
```

- 固定 `torch==2.5.1`，防止 pip 顺带升级 torch；安装后用 `conda list | grep torch` 再确认一次版本。
- open_clip 3.3.0 要求 torch ≥ 2.0。不要从 GitHub main 分支安装：其 README 写的是 torch ≥ 2.6。

### 4.3 数据与模型权重

先提交一个不申请 GPU 的小作业，在计算节点上运行 `curl -sI https://huggingface.co`，确认计算节点能否联网：

- **能联网：** 在作业里下载数据与 CLIP 权重。
- **不能联网：** 在本机下载后用 WinSCP/scp 上传到 home 目录，作业里设置 `HF_HUB_OFFLINE=1`。

在作业中或本机下载数据：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('flwrlabs/office-home', repo_type='dataset', local_dir='data/office_home')"
```

权重同理用 `snapshot_download('timm/vit_base_patch32_clip_224.openai')` 和 `snapshot_download('timm/vit_base_patch16_clip_224.openai')` 下载到 HF 缓存；如果是在本机下载，就把缓存目录整体上传到集群的 `~/.cache/huggingface/hub/`。

数据与特征放在 home 目录下（磁盘额度至少 100GB），每完成一个阶段就把结果拉回本机。集群**没有备份**。

### 4.4 作业脚本模板（特征提取）

```bash
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=oh_feats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda
source activate mySC4001

# 每 60 秒记录一次分配到的 GPU 的利用率和显存
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > logs/gpu_${SLURM_JOB_ID}.csv &

python scripts/extract_features.py --backbone ViT-B-32-quickgelu --data data/office_home --out features/
```

- 提交前先 `mkdir -p logs`，因为 SLURM 不会自动创建日志目录。
- `scripts/extract_features.py` 尚未编写，见第 5 节的计划结构。
- 作业结束后用 `seff <jobid>` 查看实际的 CPU 和内存利用率，据此调整下一次的申请量。
- 只使用分配给自己的 QoS，申请量不超过 QoS 上限；不要访问自己 home 目录和共享目录以外的路径。

## 5. 计划中的代码结构

```
configs/                 # K、seeds、超参网格、prompt 模板、类名映射（看到结果前固定）
scripts/
  extract_features.py    # 一次性：图像/文本特征 → features/
  make_splits.py         # 生成并保存所有支持池与测试划分（按 seed）
  run_experiments.py     # 读缓存特征，跑全部方法、门控与来源域选参，逐图预测写入 results/
  analyze.py             # help/harm、置信度分箱、显著性检验、出图
jobs/                    # sbatch 脚本
results/                 # 逐图预测与汇总表（只存小文件）
```

要保存**逐图预测**而不只是准确率：help/harm、置信度分箱和 McNemar 检验都依赖逐图结果。

## 6. 时间表与分工（2026-10-30 至 11-16）

| 日期 | 工作 | 检查点 |
| --- | --- | --- |
| 10/30–11/1 | 确认 QoS 与计算节点联网；下载数据；建环境；核对类名、统计每类数量并定下 K_max；固定 configs；提取 ViT-B/32 特征；跑 B0、B1 | 四个域的 zero-shot 准确率 |
| 11/2–11/5 | 实现 A1–A4 与来源域选参；跑 K × seeds；完成 help/harm 与置信度分箱 | 判定 H1、H2，按决策点确定门控是否继续 |
| 11/6–11/9 | 实现 G1、G2、C1–C3；跑全部主矩阵；提取 ViT-B/16 特征并复核 | 冻结主结果表 |
| 11/3 起并行 | 写相关工作与方法章节 | — |
| 11/10–11/14 | 写结果与讨论、画图、写失败分析；整理代码与复现说明；由另一名组员按说明独立复现主结果 | 报告完整初稿 |
| 11/15–11/16 | 校对页数与格式，写 AI Use Statement，打包 ZIP，**11/16 白天提交**，不等到 23:59 | 已提交 |

**建议分工（三人）：**

- 成员 1：数据与集群（下载、环境、特征、划分）；之后负责 backbone 复核和复现说明。
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
| 5 Results | RQ1、RQ2、RQ3 各一小节，加 backbone 复核 | 3.5 |
| 6 Discussion | 失败案例、局限、负结果的含义 | 1.25 |
| 7 Conclusion | — | 0.25 |
| AI Use Statement | 工具、用途、核实方式、团队决策与贡献 | 0.25 |
| References、Appendix | 不计页数 | — |

主要图表：每个域的准确率–K 曲线；help/harm 堆叠柱状图；置信度分箱图；门控与对照的配对差值图；被改错图像的示例。

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
- **团队决策：** 2026-10-29 选定方向 I。研究问题、假设与实验设计将在组内讨论后确认。
- **待补充：** 后续编码、调试、写作中的 AI 使用情况与核实方式。
