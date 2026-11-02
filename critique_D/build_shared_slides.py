"""按共享 PPT 模板（导航条、Presenter 标签、证据框 + 右栏）生成 D 部分的 3 页。

用法: python build_shared_slides.py  ->  DSA5206_D_shared_template.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FONT = "IBM Plex Sans"
BG, NAVY, ORANGE = RGBColor(0xF7, 0xF6, 0xF2), RGBColor(0x14, 0x21, 0x3D), RGBColor(0xB5, 0x52, 0x1B)
BODY, GREY, LIGHT = RGBColor(0x3E, 0x4A, 0x5C), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xD9, 0xD6, 0xCE)
PILL, BOX = RGBColor(0xE8, 0xE5, 0xDE), RGBColor(0xFA, 0xF9, 0xF6)
NAV = ["Background", "Part I Reproduction", "Part II Critique", "Part II Alternatives", "Conclusion"]
STEPS = ["1 Collection", "2 Preparation", "3 Selection", "4 Training", "5 Deployment"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def text(slide, x, y, w, h, runs, size=12, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """runs: str, or list of paragraphs; each paragraph is str or list of (text, overrides)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragraphs = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for part in ([(para, {})] if isinstance(para, str) else para):
            t, o = part if isinstance(part, tuple) else (part, {})
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = FONT, Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return box


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, dash=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb, s.line.width = line, Pt(1)
        if dash:
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s.shadow.inherit = False
    return s


def table(slide, x, y, w, rows, widths, size=11):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                               Inches(0.3 * len(rows))).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PILL if i == 0 else (RGBColor(0xF3, 0xE3, 0xD8) if "honest" in row[0] else BOX)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = val
            r.font.name, r.font.size, r.font.bold = FONT, Pt(size), i == 0
            r.font.color.rgb = NAVY if i == 0 else BODY
    return t


def base(step, title, asks, found, verdict, page):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    # 顶部导航
    for i, label in enumerate(NAV):
        x = 0.94 + i * 2.44
        active, done = i == 2, i < 2
        text(s, x, 0.42, 2.3, 0.25, label, size=10, bold=active, color=NAVY if active or done else GREY)
        rect(s, x, 0.72, 2.3, 0.04, ORANGE if active else (NAVY if done else LIGHT))
    # 步骤面包屑与 Presenter
    crumb = []
    for i, st in enumerate(STEPS):
        crumb.append((st, {"bold": i == step, "color": ORANGE if i == step else GREY}))
        if i < 4:
            crumb.append((" · ", {"color": GREY}))
    text(s, 0.94, 1.25, 8, 0.25, [crumb], size=11)
    pill = rect(s, 11.62, 1.2, 1.38, 0.32, PILL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    pill.adjustments[0] = 0.5
    text(s, 11.62, 1.2, 1.38, 0.32, "Presenter: D", size=10, bold=True, color=NAVY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.94, 1.62, 12, 0.6, title, size=28, bold=True, color=NAVY)
    # 证据框
    rect(s, 0.94, 2.4, 6.77, 4.27, BOX, line=LIGHT, dash=True)
    # 右栏
    y = 2.43
    text(s, 8.08, y, 4.9, 0.25, "LECTURE 9 ASKS", size=10, bold=True, color=ORANGE)
    text(s, 8.08, y + 0.3, 4.9, 1.2, ["•  " + a for a in asks], size=12, spacing=2)
    y += 1.55
    text(s, 8.08, y, 4.9, 0.25, "WHAT WE FOUND", size=10, bold=True, color=ORANGE)
    text(s, 8.08, y + 0.3, 4.9, 1.9, ["•  " + f for f in found], size=11, spacing=4)
    y += 2.05
    text(s, 8.08, y, 4.9, 0.25, "VERDICT", size=10, bold=True, color=ORANGE)
    chip = rect(s, 8.08, y + 0.3, 1.9, 0.34, PILL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    chip.adjustments[0] = 0.5
    text(s, 8.08, y + 0.3, 1.9, 0.34, verdict, size=11, bold=True, color=NAVY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # 页脚
    text(s, 0.94, 7.05, 8, 0.25, "Cohen et al., Cureus 2020 · DSA5206 Group [__]", size=10, color=GREY)
    text(s, 11.5, 7.05, 1.5, 0.25, str(page), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def notes(s, body):
    s.notes_slide.notes_text_frame.text = body


# ---------- ③ 模型选择 ----------
s = base(2, "Model selection: only OLS, only a constant baseline",
         ["Which models were used, and why?", "Were several models compared?",
          "Is there a baseline from existing methods?"],
         ["Only ordinary least squares on 4 input sets; the sole baseline predicts the training mean (p.3; Table 1, p.4)",
          "No regularised, non-linear or published comparator; model error never set against rater disagreement (p.4)",
          "Bias: missing baseline comparison. Part II tests the alternatives"],
         "High concern", 11)
text(s, 1.15, 2.6, 6.3, 0.3, "What Table 1 compares: all OLS on one frozen DenseNet-121", size=13, bold=True, color=NAVY)
table(s, 1.15, 3.0, 6.3, [["Input to linear regression", "Parameters"],
                          ["“Lung opacity” logit", "1 + 1"], ["4 pneumonia logits", "4 + 1"],
                          ["All 18 logits", "18 + 1"], ["1024 DenseNet features", "1024 + 1"],
                          ["No data (training mean)", "0 + 1"]], [4.4, 1.9], size=12)
text(s, 1.15, 5.05, 6.3, 1.4,
     [[("Never compared: ", {"bold": True, "color": NAVY}),
       ("Ridge / Lasso, random forest, SVR, gradient boosting, other pretrained encoders, "
        "any published CXR severity model, or radiologist disagreement as a reference error.", {})]], size=12)
notes(s, "Every result in the paper comes from ordinary least squares. The only baseline is a constant that "
         "predicts the average score. No regularised model, no non-linear model, no comparison with other methods. "
         "That gap is what our Part II tests.")

# ---------- ④ 模型训练 ----------
s = base(3, "Model training: feature set picked on the test set",
         ["How was data split; was the test set used only once?",
          "Are the metrics suitable; is calibration shown?",
          "Is it interpretable, reproducible and generalisable?"],
         ["No validation set; the stated rule (training MSE, p.3) would pick 1024 features every time, so “1 logit” was picked on test results. Honest CV: +2–4% MAE",
          "Severe cases under-predicted (~0.9); only 7 of 94 images score opacity ≥ 5; no calibration reported (p.4)",
          "Good: patient-grouped splits, no leakage (p.3); code and data open. Weak: 4 hand-picked saliency maps (p.6)"],
         "Some concern", 12)
text(s, 1.15, 2.55, 6.3, 0.3, "Re-running the selection rule on the same 50 splits (MAE)", size=13, bold=True, color=NAVY)
table(s, 1.15, 2.9, 6.3, [["Selection rule", "Opacity", "Extent"],
                          ["Paper: “1 logit”, picked from Table 1", "0.788", "1.161"],
                          ["Paper’s stated rule: training MSE", "1.012", "1.341"],
                          ["Nested grouped CV (honest)", "0.807", "1.209"],
                          ["Best on each test split (cheating)", "0.744", "1.118"]], [4.3, 1.0, 1.0], size=11)
s.shapes.add_picture(str(FIG / "slide_residual_by_band.png"), Inches(1.15), Inches(4.4), width=Inches(6.3))
notes(s, "There is no validation set. The paper says the final model was chosen by training error, but that rule "
         "picks the 1024-feature model every time, because it memorises about 47 training images, and it has the worst "
         "test error. So the single-feature model was picked from test results. Re-choosing honestly with cross-validation "
         "inside the training set costs only 2 to 4 percent MAE: the result survives but is slightly optimistic. "
         "The chart: mild cases are over-predicted and severe cases under-predicted, yet the calibration slope is about 1.0, "
         "so this is regression to the mean, not miscalibration. The real problem is that the sickest patients, the ICU use case, "
         "score about one point too low, and only 7 of 94 images are that severe. Credit: splits are by patient, so no leakage.")

# ---------- ⑤ 模型部署 ----------
s = base(4, "Model deployment: proof of concept, not yet clinical",
         ["Does the model fit the clinical setting?", "Are the limitations stated?",
          "Is it ready to deploy: out-of-range use, drift, automation bias?"],
         ["Stated: small sample, no external cohort, no link to outcomes, saliency limits (p.6)",
          "Unstated: test-set selection, out-of-range scores, severe cases underestimated, PA-only vs bedside AP in ICU",
          "No uncertainty output, drift monitoring or plan for automation and feedback-loop bias"],
         "High concern", 13)
s.shapes.add_picture(str(FIG / "slide_prediction_range.png"), Inches(1.15), Inches(2.55), width=Inches(6.3))
text(s, 1.15, 4.95, 6.3, 1.6,
     [[("Unbounded output: ", {"bold": True, "color": NAVY}),
       ("mostly negative severity; clipping alone cuts extent MAE 1.16 → 1.10.", {})],
      [("Off-label setting: ", {"bold": True, "color": NAVY}),
       ("17 of 94 “PA” images are not reliably PA (slide 11), while ICU films are bedside AP.", {})]],
     size=12, spacing=6)
notes(s, "For deployment: 3.8 and 5.9 percent of predictions fall outside the valid score range, mostly negative severity. "
         "Clipping alone would have improved extent MAE from 1.16 to 1.10. The intended setting is the ICU, which uses bedside AP "
         "films, but the model was evaluated on PA films only, and 17 of the 94 so-called PA images are not reliably PA. "
         "The authors admit the small sample and the missing external cohort, but not the test-set selection, the underestimation "
         "of severe cases, or any plan for uncertainty or automation bias. Verdict: a useful proof of concept, not ready for clinical use.")

prs.save(OUT / "DSA5206_D_shared_template.pptx")
print("saved")
