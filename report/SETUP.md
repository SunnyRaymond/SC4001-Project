# SC4001 报告工作区

这是报告工作区。首次骨架于 2026-10-31 在 Overleaf 通过 XeLaTeX 编译（0 errors / 0 warnings），下载 PDF 后核对了 A4、嵌入的 Arial 字体和表格。Introduction、Results、Conclusion 的首轮英文初稿现已完成，其余章节待组员补写；具体进度和数字口径见 [WRITING_STATUS.md](WRITING_STATUS.md)。课程要求以仓库根目录的 project_handout.pdf 为准：A4、单栏、Arial 10pt，正文最多 10 页。封面、目录、参考文献和附录不计。不能套用群聊中的 AAAI 双栏模板。

**Overleaf 项目：** https://www.overleaf.com/project/6ae58ae8c68c07d74ba7aefd

## Overleaf 设置

1. 在 Overleaf 首页选择 **New project → GitHub repo**，从现有 `SunnyRaymond/SC4001-Project` 导入。不要先建空项目再尝试绑定已有仓库。
2. 将根目录 `main.tex` 设为主文档，编译器选 **XeLaTeX**。Arial 列在 Overleaf 官方 fontspec 支持字体清单中。
3. 通过 **Share** 邀请两名组员；由拥有 Premium 的账号持有项目。不要把公开编辑链接当作默认分享方式。
4. GitHub 同步是手动的：**Integrations → GitHub → Pull / Push**。开始一次编辑前先同步；同一节同一时段由一人修改，完成后同步并交接。
5. 使用 GitHub 的分支/PR 审阅改动，合并到默认分支后再拉入 Overleaf。Overleaf 本身不提供 Git 分支工作流。

当前仓库体积较小，没有 LFS 或 submodule，可优先直接导入整个仓库；报告源文件放 `report/`，继续引用现有 `figures/` 与 `results/`。数据、特征缓存和完整适配概率仍按原仓库规则留在集群。

## 文件与数字

- `main.tex`：课程格式、封面、目录、章节入口，正文与参考文献分页。
- `report/sections/`：按 README 7.7 拆分的八个写作文件。TODO 必须在提交前清空。
- `report/tables/`：运行 `python3 report/build_tables.py` 从提交的预测/汇总生成 K=4 准确率与 harm 表及门控、VRF 配对差值表；包含 B1。**B1 这一行的 harm 以 B0 为参照，不是 README 7.8 要求的 B1 参照复核。**
- `report/figures/`：运行 `python3 report/build_figures.py` 从原始结果 CSV 重排单栏矢量图，不改变实验结果；原 `figures/` 的 PNG 保留。
- `report/references.bib`：已核对本轮正文引用的六篇论文的原始来源；后续文献作者沿用现有 key 并补充条目。
- 不手改生成表格。最终表应补上适当的 seed 不确定性，正文说明域等权平均与单位。
- 不声称新发明 adapter、门控或 negative flip。论文相关性与“首次”主张还要查原文。

## 现有证据与待办

主实验 A1/A2/A4、help/harm、置信度分箱、G1/G2/C1/C2/oracle、VRF/V0、六张图已有提交产物。本次只核对部分代码和结果表，没有重新训练模型，也没有独立验证全部引用。

提交前还需完成：

- B1 为参照的适配翻转复核；决定 Figure 5 的示例选择。
- 核查近邻相关工作、引用元数据和具体论断；谨慎使用“首次”。
- 团队确认 AI 使用记录中的工具、模型、分工和实际验证方式。
- 补封面姓名、组号；补正文、参考文献与复现说明；清空 TODO。
- 在 Overleaf 编译并检查 Arial、图表可读性、正文页数和最终 PDF。
- 由另一名组员按说明复现主结果；最终提交 PDF 和代码 ZIP。截止 2026-11-16 23:59（新加坡时间）。

数字口径：A2+G1 比 A2 多约 0.04 个百分点是所有 K、域、seed 的均值；K=4 单独看为 82.97% 对 82.99%。VRF 与 V0 几乎相同，不能笼统写成“所有逐样本方法明显优于各自随机对照”。

## 官方说明

- GitHub 连接与手动同步：https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization
- 字体清单：https://www.overleaf.com/learn/latex/Questions/Which_OTF_or_TTF_fonts_are_supported_via_fontspec%3F

搭建日期：2026-10-31。骨架提交 `28356e0` 已推送到 GitHub `main`，并从该仓库导入上述 Overleaf 项目；主文件 `main.tex`，编译器 XeLaTeX，TeX Live 2026。账号连接与项目绑定已经完成，后续仍需手动 Pull / Push 同步。
