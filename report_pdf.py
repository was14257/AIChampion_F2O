import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

from glaucoma_cls.concepts import ALL_CONCEPTS, CONCEPT_META

_FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Malgun", str(_FONT_DIR / "malgun.ttf")))
pdfmetrics.registerFont(TTFont("Malgun-Bold", str(_FONT_DIR / "malgunbd.ttf")))

NAVY = colors.HexColor("#0f2b46")
SLATE = colors.HexColor("#3d4a57")
MUTED = colors.HexColor("#7a8793")
LINE = colors.HexColor("#dce4ec")
PANEL = colors.HexColor("#f8fbfd")
GRADE_COLORS = {"hero-green": colors.HexColor("#2e9e6b"),
                 "hero-amber": colors.HexColor("#e08a2b"),
                 "hero-red": colors.HexColor("#d6483f")}
GRADE_BG = {"hero-green": colors.HexColor("#eaf6f0"),
            "hero-amber": colors.HexColor("#fdf2e4"),
            "hero-red": colors.HexColor("#fbe9e8")}

_TITLE = ParagraphStyle("Title", fontName="Malgun-Bold", fontSize=17, leading=20, textColor=colors.white)
_TITLE_SUB = ParagraphStyle("TitleSub", fontName="Malgun", fontSize=9, leading=12, textColor=colors.white)
_H2 = ParagraphStyle("H2", fontName="Malgun-Bold", fontSize=11.5, leading=15, textColor=NAVY,
                      spaceBefore=2, spaceAfter=5)
_BODY = ParagraphStyle("Body", fontName="Malgun", fontSize=9.3, leading=14.5, textColor=SLATE)
_SMALL = ParagraphStyle("Small", fontName="Malgun", fontSize=7.6, leading=11.5, textColor=MUTED)
_META = ParagraphStyle("Meta", fontName="Malgun", fontSize=9, leading=13, textColor=SLATE)
_META_R = ParagraphStyle("MetaR", fontName="Malgun", fontSize=9, leading=13, textColor=SLATE, alignment=TA_RIGHT)
_GRADE = ParagraphStyle("Grade", fontName="Malgun-Bold", fontSize=15, leading=18)
_HEADLINE = ParagraphStyle("Headline", fontName="Malgun-Bold", fontSize=10.5, leading=14, textColor=NAVY)
_KPI_LABEL = ParagraphStyle("KpiLabel", fontName="Malgun", fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER)
_KPI_VALUE = ParagraphStyle("KpiValue", fontName="Malgun-Bold", fontSize=15, leading=18, alignment=TA_CENTER)
_CAP = ParagraphStyle("Cap", fontName="Malgun", fontSize=7.8, leading=11, textColor=MUTED, alignment=TA_CENTER)


def _np_to_rlimage(arr: np.ndarray, width_mm=52):
    """Convert a numpy image array into a reportlab Image flowable, scaled to width_mm."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    w = width_mm * mm
    h = w * im.height / im.width
    return RLImage(buf, width=w, height=h)


def _eye_logo_drawing(size_mm=9, color=colors.white):
    """Vector logo: almond-shaped eye + iris/pupil. Symbol placed next to the 'EYEON' text."""
    from reportlab.graphics.shapes import Drawing, Path, Circle
    s = size_mm * mm
    d = Drawing(s, s)
    cx, cy = s / 2, s / 2
    rx, ry = s / 2 - 0.4 * mm, s / 3.1
    p = Path(strokeColor=color, strokeWidth=1.1, fillColor=None)
    k = 0.5523
    p.moveTo(cx - rx, cy)
    p.curveTo(cx - rx, cy + ry * k, cx - rx * k, cy + ry, cx, cy + ry)
    p.curveTo(cx + rx * k, cy + ry, cx + rx, cy + ry * k, cx + rx, cy)
    p.curveTo(cx + rx, cy - ry * k, cx + rx * k, cy - ry, cx, cy - ry)
    p.curveTo(cx - rx * k, cy - ry, cx - rx, cy - ry * k, cx - rx, cy)
    d.add(p)
    d.add(Circle(cx, cy, s * 0.16, strokeColor=None, fillColor=color))
    return d


def _header(canvas, doc, patient_name, exam_date, report_id):
    """Navy header bar drawn at the top of every page (logo+title+patient meta)."""
    canvas.saveState()
    w, h = A4
    bar_h = 20 * mm
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - bar_h, w, bar_h, stroke=0, fill=1)

    logo = _eye_logo_drawing(8)
    logo.drawOn(canvas, 18 * mm, h - bar_h + 6.5 * mm)

    canvas.setFillColor(colors.white)
    canvas.setFont("Malgun-Bold", 15)
    canvas.drawString(30 * mm, h - bar_h + 10.5 * mm, "EYEON")
    canvas.setFont("Malgun", 7.8)
    canvas.drawString(30 * mm, h - bar_h + 5.6 * mm, "AI 기반 녹내장 조기 선별 결과 리포트")

    canvas.setFont("Malgun", 7.6)
    right_x = w - 18 * mm
    canvas.drawRightString(right_x, h - bar_h + 11.8 * mm, f"환자/피검자: {patient_name}")
    canvas.drawRightString(right_x, h - bar_h + 7.6 * mm, f"검사일: {exam_date}")
    canvas.drawRightString(right_x, h - bar_h + 3.4 * mm, f"리포트 번호: {report_id}")

    # Footer
    canvas.setFillColor(MUTED)
    canvas.setFont("Malgun", 6.8)
    canvas.drawString(18 * mm, 8 * mm,
                       "본 리포트는 선별(screening) 보조 목적이며 의사의 진단을 대체하지 않습니다.")
    canvas.drawRightString(w - 18 * mm, 8 * mm, f"{doc.page} page")
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 11 * mm, w - 18 * mm, 11 * mm)
    canvas.restoreState()


def _grade_badge_table(grade, cls, headline, risk_disp, ci_lo, ci_hi):
    """Two-column badge: verdict grade/headline on the left, risk score/CI on the right."""
    color = GRADE_COLORS[cls]
    bg = GRADE_BG[cls]
    label_style = ParagraphStyle("gradeLabel", fontName="Malgun", fontSize=8.5, textColor=MUTED)
    left = [
        Paragraph("종합 판정", label_style),
        Paragraph(f'<font color="{color}">{grade}</font>', _GRADE),
        Paragraph(headline, _HEADLINE),
    ]
    right = [
        Paragraph("AI 위험도 점수", _KPI_LABEL),
        Paragraph(f'<font color="{color}">{risk_disp*100:.1f}%</font>', _KPI_VALUE),
        Paragraph(f"95% CI {ci_lo*100:.0f}–{ci_hi*100:.0f}%", _CAP),
    ]
    t = Table([[left, right]], colWidths=[112 * mm, 47 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.8, color),
        ("LINEAFTER", (0, 0), (0, 0), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
    ]))
    return t


def _kpi_row(concept, conf_txt, thr_txt):
    """Three-cell KPI strip: vertical C/D ratio, prediction confidence, and the decision threshold."""
    cdr = concept.get("cdr", float("nan"))
    cdr_flag = "정상범위 초과" if cdr >= 0.6 else "정상범위"
    cdr_color = colors.HexColor("#d6483f") if cdr >= 0.6 else colors.HexColor("#2e9e6b")

    def cell(label, value, sub, sub_color=MUTED):
        return [
            Paragraph(label, _KPI_LABEL),
            Paragraph(f'<font color="{sub_color}">{value}</font>', _KPI_VALUE),
            Paragraph(sub, _CAP),
        ]

    row = [
        cell("수직 C/D 비율", f"{cdr:.3f}", cdr_flag, cdr_color),
        cell("예측 신뢰도", conf_txt, ""),
        cell("판정 기준", thr_txt, ""),
    ]
    t = Table([row], colWidths=[53 * mm, 53 * mm, 53 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build_report_pdf(result: dict, risk_grade_fn, platt_scale_fn,
                      patient_name: str = "홍길동", confidence_fn=None,
                      thr_suspect: float | None = None) -> bytes:
    """result: the dict returned by run_analysis(). risk_grade_fn/platt_scale_fn
    and confidence_fn are injected directly from app_streamlit.py's
    _risk_grade/_platt_scale and _confidence_from_ci, so the display logic
    isn't duplicated.

    thr_suspect: the raw-risk threshold at which a case first counts as
    "needs attention". Passed in (rather than hardcoded here) because it is
    recalibrated on every retrain — a stale copy in this file previously
    printed "raw risk >= 21%" long after the deployed threshold had moved."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18 * mm, rightMargin=18 * mm,
                             topMargin=28 * mm, bottomMargin=16 * mm)
    story = []

    risk = result["risk_prob"]
    ci = result["risk_ci"]
    concept = result["concept"]
    grade, cls, headline, detail, action = risk_grade_fn(risk)
    risk_disp = platt_scale_fn(risk)
    conf_txt = confidence_fn(ci)[0] if confidence_fn else "-"

    exam_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_id = "EY-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    story.append(_grade_badge_table(grade, cls, headline, risk_disp, ci["lo"], ci["hi"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(detail, _BODY))
    story.append(Paragraph(f"<b>권장 행동</b> — {action}", _BODY))
    story.append(Spacer(1, 2.5 * mm))

    thr_txt = (f"raw risk ≥ {thr_suspect*100:.0f}%"
               if thr_suspect is not None else "—")
    story.append(_kpi_row(concept, conf_txt, thr_txt))
    story.append(Spacer(1, 3.5 * mm))

    story.append(HRFlowable(width="100%", thickness=0.6, color=LINE))
    story.append(Paragraph("시신경/망막 정량 지표", _H2))
    header = ["지표", "값", "정상범위", "판정", "단위"]
    rows = [header]
    for c in ALL_CONCEPTS:
        v = concept.get(c, float("nan"))
        label, unit, desc, rng, direction = CONCEPT_META.get(c, (c, "", "", None, "high"))
        fmt = (lambda x: f"{x:,.0f}") if unit == "px" else (lambda x: f"{x:.3f}")
        if rng is not None:
            lo, hi = rng
            rng_txt = f"{fmt(lo)}~{fmt(hi)}"
            if v != v:
                verdict = "-"
            elif lo <= v <= hi:
                verdict = "정상범위"
            elif (direction == "high" and v > hi) or (direction == "low" and v < lo):
                verdict = "주의"
            else:
                verdict = "정상범위"
        else:
            rng_txt = "-"
            verdict = "-"
        rows.append([label, fmt(v), rng_txt, verdict, unit or "-"])

    tbl = Table(rows, colWidths=[46 * mm, 22 * mm, 34 * mm, 22 * mm, 18 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Malgun"),
        ("FONTNAME", (0, 0), (-1, 0), "Malgun-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
    ]
    for i, r in enumerate(rows[1:], start=1):
        if r[3] == "주의":
            style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#d6483f")))
            style.append(("FONTNAME", (3, i), (3, i), "Malgun-Bold"))
        elif r[3] == "정상범위":
            style.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#2e9e6b")))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 3.5 * mm))

    story.append(HRFlowable(width="100%", thickness=0.6, color=LINE))
    story.append(Paragraph("합성 OCT · 시신경 구조 영상", _H2))
    img_cell = lambda arr, cap: [
        _np_to_rlimage(arr, width_mm=44), Spacer(1, 1 * mm), Paragraph(cap, _CAP)]
    img_row = Table([[
        img_cell(result["oct_image"], "합성 OCT B-scan"),
        img_cell(result["seg_overlay"], "Disc(초록)/Cup(빨강)"),
        img_cell(result["gradcam_overlay"], "Attention Rollout(참고용)"),
    ]], colWidths=[57 * mm, 57 * mm, 57 * mm])
    img_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_row)
    story.append(Spacer(1, 3 * mm))

    story.append(HRFlowable(width="100%", thickness=0.6, color=LINE))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(
        "본 결과는 인공지능 기반 선별(screening) 보조 도구의 분석이며, 의사의 진단을 "
        "대체하지 않습니다. 최종 진단과 치료는 반드시 안과 전문의의 진료와 정밀 검사(OCT 등)를 "
        "통해 이루어져야 합니다. 학습 데이터가 동아시아인 안저 사진으로 구성되어 있어, "
        "다른 인종/지역에서는 정확도가 달라질 수 있습니다. Attention Rollout 히트맵은 공간적 "
        "판단 근거로서 신뢰도가 검증되지 않았으므로 참고용으로만 사용하십시오.",
        _SMALL))

    def _on_page(canvas, doc_):
        _header(canvas, doc_, patient_name, exam_date, report_id)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()
