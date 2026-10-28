# C + D 演讲稿（批判部分，共约 5 分钟）

全组 15 分钟。建议分配：开场与论文介绍 1 分钟 · Part I 复现 3 分钟 · **批判 C+D 5 分钟** · 替代方法与统计结论 4 分钟 · 总结 1 分钟 · 余 1 分钟缓冲。如果组里给的时间不同，按下面每页的“可删”提示压缩。

语速按每分钟约 130 个英文词。**加粗**的是这一页必须说出口的数字，其余数字留在幻灯片上。

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

## D：③ 模型选择、④ 模型训练、⑤ 模型部署（约 3 分钟）

### ③ · One model family, one trivial baseline（25 秒）

> Every result in the paper comes from ordinary least squares. The only baseline is a constant that predicts the average score. No regularised model, no non-linear model, no comparison with other methods. That gap is what our Part II tests.

### ④ 1/3 · The final feature set was picked on the test set（45 秒）

> There is no validation set. The paper says the final model was chosen by training error, but if you apply that rule, the 1024-feature model wins **every time**, because it memorises about 47 training images, and it has the worst test error. So the single-feature model must have been picked from test results. When we redo the choice honestly, with cross-validation inside the training set, error rises only **2 to 4 percent**. The main result survives, but it is slightly optimistic.

可删：右侧训练/测试 MSE 图的解释。

### ④ 2/3 · Calibrated on average, short on severe cases（35 秒）

> Mild cases are over-predicted by **0.72**, severe cases under-predicted by **0.87**. Careful: the calibration slope is about **1.0**, so this is not miscalibration, it is regression to the mean. The real problem is a compressed range: the sickest patients, exactly the ICU use case, are scored about one point too low.

### ④ 3/3 · No leakage, but imbalance and explainability left open（30 秒）

> One thing the paper does right: splits are by patient, so there is no leakage. But only **7 of 94** images are severe, and nothing is done about that imbalance. Explainability is just four hand-picked saliency maps. And subgroup performance is not reported; ours shows some differences by age, but the groups are too small to conclude.

可删：分组的具体数字。

### ⑤ · A proof of concept, not a deployable tool（30 秒）

> For deployment: **3.8 and 5.9 percent** of predictions fall outside the valid score range, mostly negative severity. The intended setting is the ICU, which uses bedside AP films, but the model was evaluated on PA only. And the authors don’t mention the test-set selection, the underestimation of severe cases, or any plan for uncertainty or automation bias.

### Checklist summary（15 秒）

> This table puts it all together against the Lecture 9 checklist. Open and reproducible, no leakage, but weak on data, model comparison and explainability. Our verdict: **a useful proof of concept, not ready for clinical use.**

过渡句（交给替代方法部分）：
> Can a better model fix some of this? That’s the next part.

---

## 问答准备

**Q: Why do you say regression to the mean, not miscalibration?**
> If you group by the true score, any model with R² below one looks biased toward the mean. If you group by the prediction, which is how calibration is defined, the slope is 1.02. So predictions are right on average; the range is just compressed.

**Q: How exactly did you do the nested cross-validation?**
> Inside each of the 50 training halves, we ran 5-fold cross-validation grouped by patient, picked the feature set with the lowest validation error, refit on the whole training half, and only then scored the test half.

**Q: How did you decide those five images are not PA?**
> We looked at all 94 images. Those five have the projection printed on the film itself, for example “AP Sitting” or the Italian “CLINO”, meaning lying down. The list is in our repository.

**Q: Why not conclude anything from the subgroup results?**
> Each age group has only 19 to 29 images, and the 50 splits share data, so the differences are within the noise. We report them because the paper should have, not as a finding.

**Q: Isn’t 94 images just too small to judge anything?**
> Yes, and that is part of our critique: with this sample size, choices like feature-set selection and the PA criterion matter more, not less.

**Q: Is the ethics approval enough?**
> For retrospective public data, a waiver is acceptable. Our point is narrower: the study was not registered, and the image licences are not recorded for 66 of 94 images.
