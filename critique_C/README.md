# C 部分：① 数据收集 · ② 数据准备

所有数字都来自 Part I 实际使用的 94 张评分图和历史元数据（`../covid_severity_reproduction/`）。只依赖 numpy、pandas、matplotlib、pillow，几秒跑完。

```bash
cd critique_C
python partC_analysis.py
```

幻灯片：`DSA5206_Critique_C.pptx`（①② 共 3 页，演讲者备注里有讲稿）。

输出：`partC_summary.json`、`demographics_check.csv`、`image_properties.csv`、`mae_by_source.csv`，以及 `figures/` 下的来源图、分辨率直方图和 3 张示例图。`manual_view_review.csv` 是人工逐张查看 94 张图后记录的烧录文字（与 PA/AP 判断有关的部分）。

## ① 数据收集

**伦理**：论文 p.7 写 CERSES 批准号 #CERSES-20-058-D，“Consent was obtained or waived”。数据全部来自公开的论文插图和病例分享网站，作者不可能联系到患者，实际上只能是豁免知情同意。数据授权：元数据里 66/94 张没有记录 license，其余是 CC BY-NC-SA（22）和 CC BY-NC-ND（6）；期刊插图的再分发授权并不清楚。

**来源**（按 URL 域名统计，94 张 / 65 位患者）：

| 来源 | 图片 | 患者 |
| --- | --- | --- |
| 期刊论文插图（NEJM 10、ScienceDirect 7、RSNA 7、Lancet 4、KJR 4、OUP 3、Wiley 2、Nature 2） | 39 | 23 |
| SIRM 意大利放射学会病例 | 26 | 22 |
| Radiopaedia | 18 | 13 |
| 新加坡医学院 PDF | 8 | 4 |
| 其他病例网站（Figure1 2、RSNA Case 1） | 3 | 3 |

分工表写“期刊 40 张”，是把 RSNA Case 网站那 1 张算进了期刊。
**国家**：意大利 29、未记录 19、中国 12、西班牙 11、台湾 8、越南 6、美国 4、澳大利亚 2，加拿大、以色列、瑞典各 1。没有非洲、南美和南亚的图。
**时间**：2019 年 12 月到 2020 年 3 月，属于疫情最早期，病例是“值得发表”的典型或重症病例，存在发表偏倚。

**人口学核对**（`demographics_check.csv`）：

| | 论文 | 按图片 | 按患者 |
| --- | --- | --- | --- |
| 男 / 女 | 44 / 36 | 45 / 32 | 31 / 24 |
| 性别缺失 | 0（“no missing data”） | 17 | 10 |
| 年龄 | 56 ± 14.8 | 56.6 ± 15.2 | 57.0 ± 15.7 |
| 年龄缺失 | 0 | 25 | 15 |

论文的 44 + 36 = 80，既对不上已知性别的 77 张图，也对不上 55 位已知性别的患者。

**纳入标准“全部是 PA 片”不成立**：
- 当前版本的元数据已把 11 张改标为 AP 或 AP Supine，另有 1 张已被删除（`../covid_severity_reproduction/results/metadata_changes.csv`）。
- 人工逐张查看后，又发现 5 张图上烧录的文字直接说明不是 PA，但当前元数据仍标为 PA：`ryct…fig5-day4` 和 `day7`（“AP Sitting … I.D. Portable”）、Nature 图（“MOBILE ERECT AP”）、两张 SIRM 图（“CLINO”，意大利语卧位）。
- 合计 94 张里有 16 张被标为或改标为非 PA，另有 1 张已删除，共 17 张（18%）不是可靠的 PA。

## ② 数据准备

**标签**：三位专家（两位放射科医生和一位住院医师）盲评，只给单张图、没有临床信息。论文报告的 Fleiss κ 为 opacity 0.45（中等）、extent 0.71。公开的只有三人的均值，没有单个评分者的分数，所以 κ 无法复核，也无法估计标签噪声。没有 CT 或临床结局作参考标准。

**缺失数据**：论文说“there was no missing data”。实际 94 张图里，性别缺 17、年龄缺 25、地点缺 19、生存结局缺 59、是否进 ICU 缺 81。作者没有说明怎么处理这些缺失；它们不进模型，但用于描述队列和 Figure 3 的生存分析。

**图像来源质量**（`image_properties.csv`）：
- 最短边从 237 px 到 3520 px，中位数 963；24 张小于 600 px，4 张小于 300 px。新加坡 PDF 的图中位数只有 528 px。
- 格式混杂：JPEG RGB 66、PNG RGBA 20、JPEG 灰度 7、PNG RGB 1。预处理只取第一个通道。
- 很多图带烧录的面板字母（A/B、a/b）、箭头、日期、“R/L” 标记，有两张还印着“Lossy compressed”。预处理只做中心裁剪和缩放，没有去除这些标记，也没有做肺野分割或数据增强。

示例（`figures/`）：
1. `example_ap_sitting.png`：RSNA 论文插图，烧录“AP Sitting / I.D. Portable / Lossy compressed”，元数据标为 PA。
2. `example_pdf_arrow.png`：从新加坡 PDF 截出的 522 × 582 图，带白色箭头、“R”和面板字母“a”。专家评分时也看到了这个指向病灶的箭头。
3. `example_tiny_ap.png`：Nature Medicine 插图截出的 255 × 249 图，烧录“MOBILE ERECT AP”，元数据标为 PA（原图用最近邻放大，可见像素）。

**误差随来源变化**（`mae_by_source.csv`，Part I 单特征模型，50 次划分的测试预测合并）：extent MAE 在新加坡 PDF 上是 0.88，在 Radiopaedia 上是 1.46，“其他病例网站”（3 张）是 2.54。各组样本很少，只能说明来源异质，不能下结论。
