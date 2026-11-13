# C + D 演讲稿（批判部分，共约 5 分钟）

按仓库里的最终版幻灯片编写：C 用 `critique_C/DSA5206_Critique_C.pptx`（3 页），D 用 `critique_D/DSA5206_D_shared_template.pptx`（3 页，已按共享 PPT 模板制作）。

全组 15 分钟。建议分配：开场与论文介绍 1 分钟 · Part I 复现 3 分钟 · **批判 C+D 5 分钟（C 2 分钟，D 3 分钟）** · 替代方法与统计结论 4 分钟 · 总结 1 分钟 · 余 1 分钟缓冲。如果组里给的时间不同，按每页的“可删”提示压缩。

语速按每分钟约 130 个英文词。**加粗**的是这一页必须说出口的数字，其余数字留在幻灯片上让听众看。

---

## C：① 数据收集、② 数据准备（约 2 分钟）

### ① 1/2 · 94 images collected from papers and case websites（40 秒）

> We start with the data. All 94 images come from public websites: **39 are figures cut out of journal papers**, 26 from an Italian case-report site, 18 from Radiopaedia. Italy alone gives 29 images, and **19 images have no location at all**. These are cases someone chose to publish, so this is a convenience sample, not a clinical cohort: sampling, institutional and demographic bias. The study had ethics approval, but it was not registered.

可删：国家列表、授权情况。

### ① 2/2 · The cohort description does not match the data（45 秒）

> Next, does the paper describe its own data correctly? It says there is **no missing data**, but sex is missing for 17 images and age for 25. More importantly, the paper says all 94 images are PA films. The dataset maintainers later relabelled 11 as AP and removed one, and when we looked at every image ourselves, five more have “AP sitting” or “lying” printed on the film. So **17 of 94, 18 percent, fail the paper’s own inclusion criterion**.

可删：性别人数对比（44/36）。

### ② · Subjective labels on cropped figure images（35 秒）

> The labels are the average of three blinded readers, but agreement on opacity is only **kappa 0.45**, and only the average is released. The images are not clinical files: here is one stamped “lossy compressed”, one with an arrow pointing at the lesion, which both the readers and the model see, and one only **255 pixels wide**. Preprocessing keeps all of these marks: annotation and measurement bias.

过渡句（交给 D）：
> So the inputs are noisy. Now: what did the authors do with them?

---

## D：③ 模型选择、④ 模型训练、⑤ 模型部署（约 3 分钟；全念约 3 分 10 秒，删掉“可删”部分约 2 分 40 秒）

三页版式相同：左边是证据，右边依次是 Lecture 9 的问题、我们的发现（带论文页码）和结论。讲的时候先指左边的证据，最后落到右下角的 Verdict。

### ③ · Model selection: only OLS, only a constant baseline（45 秒）

> The lecture asks which models were used and whether they were compared. On the left is everything Table 1 compares. Every row is **ordinary least squares** on the same frozen DenseNet; only the input changes, from one logit to 1024 features. The only baseline is a constant that predicts the training mean. There is no regularised model, no non-linear model, no other encoder, and the error is never compared with how much the radiologists themselves disagree. That is the missing-baseline bias, so our verdict is **high concern**, and it is exactly the gap our Part II alternatives fill.

可删：“never compared with how much the radiologists disagree”这一句。

### ④ · Model training: feature set picked on the test set（1 分 30 秒）

> There is no validation set. The paper says the final model was chosen by **training error**. In the table we applied that rule on the same 50 splits: it picks the 1024-feature model every time, because that model memorises the roughly 47 training images, and it has the worst test error, **1.01**. So the single-logit model the paper reports must have been picked from test results. When we choose honestly, with cross-validation inside the training set only, error rises from **0.79 to 0.81**: about **2 to 4 percent worse**. The headline survives, but it is slightly optimistic.
>
> The chart below shows the second problem. Mild cases are over-predicted and severe cases under-predicted, by up to about **one point**. Careful: the calibration slope is about **1.0**, so this is regression to the mean, not miscalibration. The real issue is that only **7 of 94** images are severe, nothing corrects for that imbalance, and the sickest patients, the ICU use case, come out too low.
>
> In fairness, the split is done by patient, so there is no leakage, and code and data are open. Explainability, though, rests on four hand-picked saliency maps. Verdict: **some concern**.

可删：最后一段的可解释性；时间紧时只说 “Credit: no leakage.”

### ⑤ · Model deployment: proof of concept, not yet clinical（55 秒）

> Finally, could this be deployed? The histograms show every test prediction against the valid score range: **3.8 percent** of opacity and **5.9 percent** of extent predictions fall outside it, mostly as negative severity. Simply clipping the output would have cut extent error from 1.16 to 1.10. The intended setting is the ICU, which relies on bedside AP films, yet the study only claims PA films, and as we showed, 17 of those 94 are not reliably PA.
>
> The authors do admit the small sample, the missing external cohort and the lack of outcomes. They do not mention the test-set selection, the underestimated severe cases, or any plan for uncertainty, drift or automation bias. Verdict: **high concern**. A useful proof of concept, **not ready for clinical use**.

过渡句（交给替代方法部分）：
> Can a different model fix some of this? That is the next part.

可删：“Simply clipping the output…”一句。

---

## 问答准备

**Q: Why do you say regression to the mean, not miscalibration?**
> If you group by the true score, any model with R² below one looks biased toward the mean. If you group by the prediction, which is how calibration is defined, the slope is 1.02. So predictions are right on average; the range is just compressed.

**Q: How exactly did you do the nested cross-validation?**
> Inside each of the 50 training halves, we ran 5-fold cross-validation grouped by patient, picked the feature set with the lowest validation error, refit on the whole training half, and only then scored the test half.

**Q: Why is training only “some concern” when the model was picked on the test set?**
> Because we measured the damage: honest selection costs 2 to 4 percent MAE, so the conclusion still holds. The leakage-free patient split and open code also count in its favour.

**Q: How did you decide those five images are not PA?**
> We looked at all 94 images. Those five have the projection printed on the film itself, for example “AP Sitting” or the Italian “CLINO”, meaning lying down. The list is in our repository.

**Q: Why not conclude anything from subgroup results?**
> Each age group has only 19 to 29 images, and the 50 splits share data, so the differences are within the noise. We report them because the paper should have, not as a finding.

**Q: Isn’t 94 images just too small to judge anything?**
> Yes, and that is part of our critique: with this sample size, choices like feature-set selection and the PA criterion matter more, not less.

**Q: Is the ethics approval enough?**
> For retrospective public data, a waiver is acceptable. Our point is narrower: the study was not registered, and the image licences are not recorded for 66 of 94 images.
