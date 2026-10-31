# 报告初稿进度（2026-10-31）

本轮完成 `01-introduction.tex`、`05-results.tex`、`07-conclusion.tex` 的英文初稿，接入正文四张矢量图、K=4 综合表、门控与 VRF 配对差值表，以及附录 AUROC 图。正文引用的 CLIP、Office-Home、Tip-Adapter、negative flip、VRF、DaWin 六篇论文已查原始来源，BibTeX 位于 `references.bib`。这不代表完整文献综述已完成。

## 按分工继续

| 章节 | 状态 | 接下来要做 |
| --- | --- | --- |
| 01 Introduction | 首稿完成 | 合稿后检查 RQ、方法名和贡献范围与其他章节一致 |
| 02 Related Work | 待文献负责人写 | 补 CoOp、CLIP-Adapter、WiSE-FT、CLAP、DomainBed 等必要引用，明确研究差别 |
| 03 Methods | 待实验负责人写 | 用实际代码描述 A1/A2/A4、选参、G1/C1/C2、VRF/V0；已有 help/harm 定义 |
| 04 Setup | 待实验负责人写 | 数据与固定划分、seeds、source-only 超参选择、环境和复现命令 |
| 05 Results | 首稿完成 | B1 参照复核后补一句稳健性结论；实验负责人逐项核对 |
| 06 Discussion | 待文献/分析负责人写 | Figure 5 失败例、外观相近类别、局限；不要重复整节 Results |
| 07 Conclusion | 首稿完成 | 合稿后与最终结论保持一致 |
| 08 AI Use | 待全组确认 | 根据真实工具、实际模型、用途、验证和团队决策写，不能自动声称已独立复现 |

## 本轮核实和口径修正

- 数字来自提交的 CSV/逐图预测，没有重新训练、选择新超参或修改固定配置。重新生成表格：`python3 report/build_tables.py`；重新生成图：`python3 report/build_figures.py`。
- 检查了保存结果中的恒等式 `help - harm = adapted accuracy - B0 accuracy`，允许 CSV 六位小数舍入误差。
- A2 的 harm 随 K 为 3.77%、4.08%、3.74%、3.35%。因此不能照搬 README 中“标注增多 harm 完全不降、净增益全部来自 help”的绝对表述；应写“harm 未单调下降且持续存在，收益主要来自 help 增长”。
- H1 是 48 个条件均未触发“多数 seed 的 top-20% 净增益为正”的拒绝标准，不能写成每个条件严格净增益 ≤ 0。部分 A2/A4 条件有很小的正均值，但正 seed 最多一个。
- A2、K=4 最低置信 decile 的 pooled 净增益为 14.88 pp；低五个 decile 占其 harm 的 90.3%。这是按图像数池化的分箱诊断，与主表的域等权平均不同。
- A2 的 G1 增益 +0.04 pp 平均所有 K；K=4 为 82.97% 对未门控 82.99%。K=4 的 G1 保留率为 52.2%，所有 K 的均值约 55.7%，不能混用。
- 以提交的六位小数 CSV 核对 VRF W/T/L 时，绝对差 <0.001 pp 视为数值平局。A1/A2/A4 的 VRF 对 C1 分别为 34/2/44、27/6/47、14/54/12，和 README 用更细浮点差值计算的计数不同。均值和 H3 的“不获支持”结论不变；正文表注写明容差。
- 配对表中的 SD 描述 80 个域×K×seed 条件的差值变化。这些条件共享图像，不能当作 80 个独立观测做显著性推断。
- VRF 的距离 AUROC 约 0.62，弱于 MSP 的约 0.89；“更强的逐样本信号”应改为“另一种信号”。VRF 对随机失败集 V0 的均值优势极小，不能笼统称为显著更好。

## 待完成的证据

B1 作为参照的 adapter 翻转复核、失败案例人工检查、其余章节文献核查、独立复现、封面组号与姓名、全组对 AI 记录的确认仍待完成。正文没有把这些事项写成已完成。
