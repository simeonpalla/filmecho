"""Formatted PDF export of a Filmecho Greenlight Memo.

This is a real, laid-out document (ReportLab Platypus, A4 page size),
built for printing or sharing as a team handout — not a browser
print-to-PDF screenshot of the web page. It takes the SAME JSON payload
the frontend already has after a pipeline run (entity, result, competitive,
marketing, cast, news) and renders it directly; it does not re-run any
agent, so generating the PDF costs zero additional Parallel/Gemini calls.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

VERDICT_COLORS = {
    "greenlight": colors.HexColor("#1f7a4d"),
    "greenlight_with_changes": colors.HexColor("#a85d0a"),
    "hold": colors.HexColor("#a63333"),
    "pass": colors.HexColor("#a63333"),
    "insufficient_data": colors.HexColor("#888888"),
}
VERDICT_LABELS = {
    "greenlight": "GREENLIGHT",
    "greenlight_with_changes": "GREENLIGHT WITH CHANGES",
    "hold": "HOLD",
    "pass": "PASS",
    "insufficient_data": "NOT ENOUGH DATA",
}
ROLE_LABELS = {
    "director": "The Director",
    "producer": "The Producer",
    "marketing_chief": "The Marketing Chief",
    "casting_executive": "The Casting Executive",
    "distribution_executive": "The Distribution Executive",
    "analyst": "The Analyst",
}
DIRECTION_LABELS = {
    "keep_current_window": "Keep current window",
    "consider_earlier": "Consider shifting earlier",
    "consider_later": "Consider shifting later",
}
SOURCE_LABELS = {
    "sentiment": "sentiment", "web": "web sentiment", "youtube": "YouTube",
    "competitive": "competitive landscape", "news": "production/cast news",
    "cast": "cast reception", "marketing": "marketing analysis",
}


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=22, spaceAfter=2, textColor=colors.HexColor("#1a1a1a")),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a1a1a")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#333333")),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10, leading=14),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=10, leading=14, leftIndent=14),
        "label": ParagraphStyle("label", parent=base["Normal"], fontSize=8, textColor=colors.HexColor("#888888"), spaceAfter=2, spaceBefore=6),
        "quote": ParagraphStyle("quote", parent=base["Normal"], fontSize=10, leading=14, leftIndent=10, textColor=colors.HexColor("#333333"), fontName="Helvetica-Oblique"),
        "footer": ParagraphStyle("footer", parent=base["Normal"], fontSize=8, textColor=colors.HexColor("#999999")),
        "vlabel": ParagraphStyle("vlabel", parent=base["Normal"], textColor=colors.white, fontSize=13, fontName="Helvetica-Bold"),
        "vconf": ParagraphStyle("vconf", parent=base["Normal"], textColor=colors.white, fontSize=10, alignment=TA_LEFT),
    }


def _bullets(items: list[str], style: ParagraphStyle) -> list:
    if not items:
        return [Paragraph("None surfaced.", style)]
    return [Paragraph(f"\u2022&nbsp;&nbsp;{_esc(item)}", style) for item in items]


def _esc(text) -> str:
    """Escape text for ReportLab's mini-HTML Paragraph markup — user/LLM
    generated text can contain characters (&, <, >) that would otherwise
    break the markup or silently swallow content."""
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(data: dict) -> bytes:
    """Build the full Greenlight Memo PDF from an already-computed payload.

    Args:
        data: The same dict shape the frontend receives in the SSE
            "result" event's `data` field — must contain (at minimum)
            "entity" and "result"; "competitive", "marketing", "cast",
            "news" are read if present and skipped gracefully if not.

    Returns:
        Raw PDF bytes, ready to write to a file or stream as a response.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Filmecho Greenlight Memo",
    )
    s = _styles()
    story: list = []

    entity = data.get("entity") or {}
    result = data.get("result") or {}
    competitive = data.get("competitive") or {}
    marketing = data.get("marketing") or {}
    cast = data.get("cast") or {}
    news = data.get("news") or {}
    sentiment_synthesis = data.get("sentiment_synthesis") or {}

    title = entity.get("title", "Untitled")
    release_status = entity.get("release_status", "unclear")

    # --- Header ---
    story.append(Paragraph("FILMECHO", s["title"]))
    meta_bits = [_esc(title)]
    if entity.get("director"):
        meta_bits.append(f"Dir. {_esc(entity['director'])}")
    date_bit = entity.get("release_date") or entity.get("release_year")
    if date_bit:
        meta_bits.append(_esc(date_bit))
    meta_bits.append(_esc(release_status))
    story.append(Paragraph(" &nbsp;|&nbsp; ".join(meta_bits), s["subtitle"]))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y')}", s["footer"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#dddddd"), spaceBefore=6, spaceAfter=14))

    # --- Verdict ---
    verdict = result.get("verdict") or "unclear"
    v_color = VERDICT_COLORS.get(verdict, colors.HexColor("#888888"))
    v_label = VERDICT_LABELS.get(verdict, "VERDICT UNCLEAR")
    confidence = result.get("confidence", 0)

    verdict_table = Table(
        [[Paragraph(v_label, s["vlabel"]), Paragraph(f"Confidence: {confidence}%", s["vconf"])]],
        colWidths=[None, 130],
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), v_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 6))

    sources_used = result.get("sources_used") or []
    confidence_rationale = result.get("confidence_rationale") or ""
    sources_label = ", ".join(SOURCE_LABELS.get(s, s) for s in sources_used) if sources_used else "no upstream sources recorded"
    basis_bits = [confidence_rationale] if confidence_rationale else []
    basis_bits.append(f"Based on {len(sources_used)}/5 sources: {sources_label}.")
    story.append(Paragraph(_esc(" ".join(basis_bits)), s["footer"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(_esc(result.get("headline", "")), s["h2"]))

    story.append(Paragraph("WHY", s["label"]))
    story.extend(_bullets(result.get("why", []), s["bullet"]))

    opp_risk = Table(
        [[Paragraph(f"<b>BIGGEST OPPORTUNITY</b><br/>{_esc(result.get('biggest_opportunity'))}", s["body"]),
          Paragraph(f"<b>BIGGEST RISK</b><br/>{_esc(result.get('biggest_risk'))}", s["body"])]],
        colWidths=[None, None],
    )
    opp_risk.setStyle(TableStyle([
        ("BOX", (0, 0), (0, 0), 0.75, colors.HexColor("#1f7a4d")),
        ("BOX", (1, 0), (1, 0), 0.75, colors.HexColor("#a63333")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Spacer(1, 6))
    story.append(opp_risk)
    story.append(Spacer(1, 10))

    story.append(Paragraph("RECOMMENDED ACTION", s["label"]))
    story.append(Paragraph(_esc(result.get("recommended_action")), s["body"]))

    change_items = result.get("lessons_learned") if release_status == "released" else result.get("what_we_would_change")
    if change_items:
        label = "LESSONS LEARNED" if release_status == "released" else "WHAT WE'D CHANGE"
        story.append(Paragraph(label, s["label"]))
        story.extend(_bullets(change_items, s["bullet"]))

    notable_news = result.get("notable_news") or []
    if notable_news:
        story.append(Paragraph("NOTABLE NEWS", s["label"]))
        story.extend(_bullets([n.get("claim", "") for n in notable_news], s["bullet"]))

    # --- War Room ---
    war_room = result.get("war_room") or []
    if war_room:
        story.append(Paragraph("THE WAR ROOM", s["h2"]))
        for voice in war_room:
            role_label = ROLE_LABELS.get(voice.get("role"), voice.get("role", ""))
            story.append(Paragraph(f"<b>{_esc(role_label)}</b>", s["h3"]))
            story.append(Paragraph(f"\u201c{_esc(voice.get('insight', ''))}\u201d", s["quote"]))

    # --- Reputation timeline (previously missing from this export
    # entirely, even though the frontend has always shown it) ---
    timeline = sentiment_synthesis.get("timeline") or []
    if timeline:
        story.append(Paragraph("THE LIFE OF ITS REPUTATION", s["h2"]))
        # Milestones are free-form now, not a fixed 5-phase enum — the
        # schema normalizes date to YYYY-MM-DD specifically so this
        # string sort is also a correct chronological sort.
        ordered = sorted(timeline, key=lambda e: e.get("date") or "")
        for entry in ordered:
            milestone_label = entry.get("milestone") or "Milestone"
            date_bit = f" — {_esc(entry['date'])}" if entry.get("date") else ""
            story.append(Paragraph(f"<b>{_esc(milestone_label)}{date_bit}</b>", s["h3"]))
            story.append(Paragraph(_esc(entry.get("note", "")), s["body"]))

    # --- Release window advisory ---
    rw = competitive.get("release_window") or {}
    if rw.get("suggested_direction") not in (None, "unclear") or rw.get("congestion") not in (None, "unclear"):
        story.append(Paragraph("RELEASE WINDOW ADVISORY", s["h3"]))
        story.append(Paragraph(f"Congestion: {_esc(rw.get('congestion', 'unclear'))}", s["body"]))
        if rw.get("reasoning"):
            story.append(Paragraph(_esc(rw["reasoning"]), s["body"]))
        direction_text = DIRECTION_LABELS.get(rw.get("suggested_direction"), "Direction unclear")
        story.append(Paragraph(f"<b>{direction_text}</b>", s["body"]))
        story.append(Paragraph(
            "Directional read based on real competing titles found this run — not a projection of what a different date would actually produce.",
            s["footer"],
        ))

    story.append(PageBreak())

    # --- Evidence: Marketing ---
    story.append(Paragraph("MARKETING", s["h2"]))
    strategies = marketing.get("strategies_observed") or []
    story.append(Paragraph("Strategies observed", s["h3"]))
    story.extend(_bullets([x.get("claim", "") for x in strategies], s["bullet"]))

    worked_label = "What worked" if release_status == "released" else "Positive signals so far"
    story.append(Paragraph(worked_label, s["h3"]))
    story.extend(_bullets([x.get("claim", "") for x in (marketing.get("what_worked") or [])], s["bullet"]))

    under_label = "What underperformed" if release_status == "released" else "Concerns so far"
    story.append(Paragraph(under_label, s["h3"]))
    story.extend(_bullets([x.get("claim", "") for x in (marketing.get("what_underperformed") or [])], s["bullet"]))

    mkt_lessons = marketing.get("lessons_learned") or []
    if mkt_lessons:
        story.append(Paragraph("Lessons learned", s["h3"]))
        story.extend(_bullets(mkt_lessons, s["bullet"]))

    # --- Evidence: Cast ---
    story.append(Paragraph("CAST RECEPTION", s["h2"]))
    if cast.get("standout_performance"):
        story.append(Paragraph(f"<b>Standout:</b> {_esc(cast['standout_performance'])}", s["body"]))
    if cast.get("overall_cast_reception"):
        story.append(Paragraph(_esc(cast["overall_cast_reception"]), s["body"]))
    performances = cast.get("performances") or []
    if performances:
        story.append(Paragraph("Performances", s["h3"]))
        for p in performances:
            story.append(Paragraph(f"<b>{_esc(p.get('actor',''))}</b> — {_esc(p.get('note',''))}", s["bullet"]))
    personal = cast.get("personal_updates") or []
    if personal:
        story.append(Paragraph("Behind the scenes", s["h3"]))
        for p in personal:
            story.append(Paragraph(f"<b>{_esc(p.get('actor',''))}</b> — {_esc(p.get('headline',''))}", s["bullet"]))

    # --- Evidence: News ---
    story.append(Paragraph("PRODUCTION &amp; CAST NEWS", s["h2"]))
    facts = news.get("facts") or []
    story.extend(_bullets([f.get("claim", "") for f in facts], s["bullet"]))
    rumors = news.get("rumors") or []
    if rumors:
        story.append(Paragraph("Rumors (unconfirmed)", s["h3"]))
        story.extend(_bullets([r.get("claim", "") for r in rumors], s["bullet"]))

    # --- Evidence: Collections (released titles only) ---
    if release_status == "released":
        bo = competitive.get("box_office") or {}
        has_bo_data = any(bo.get(k) for k in ("budget", "worldwide_gross", "domestic_gross", "opening_weekend"))
        competitors = competitive.get("competing_titles") or []
        if has_bo_data or (bo.get("verdict") and bo["verdict"] != "unclear") or competitors:
            story.append(Paragraph("COLLECTIONS &amp; PERFORMANCE", s["h2"]))
            for label, key in [
                ("Budget", "budget"), ("Worldwide gross", "worldwide_gross"),
                ("Domestic gross", "domestic_gross"), ("Opening weekend", "opening_weekend"),
            ]:
                if bo.get(key):
                    story.append(Paragraph(f"<b>{label}:</b> {_esc(bo[key])}", s["body"]))
            if bo.get("verdict") and bo["verdict"] != "unclear":
                story.append(Paragraph(f"<b>Commercial verdict:</b> {_esc(bo['verdict'])} — {_esc(bo.get('verdict_basis',''))}", s["body"]))
            if competitors:
                story.append(Paragraph("Vs. competing titles", s["h3"]))
                for c in competitors:
                    line = f"<b>{_esc(c.get('title',''))}</b> — {_esc(c.get('strength_vs_searched',''))}"
                    if c.get("gross_estimate"):
                        line += f" (their gross: {_esc(c['gross_estimate'])})"
                    story.append(Paragraph(line, s["bullet"]))

    # --- Franchise history — shown regardless of release_status, since
    # this is about how PRIOR installments did, not this title's own
    # outcome (relevant for an upcoming sequel too, not just a released one) ---
    franchise_history = competitive.get("franchise_history") or []
    if franchise_history:
        story.append(Paragraph("FRANCHISE HISTORY", s["h2"]))
        for f in franchise_history:
            year_bit = f" ({f['year']})" if f.get("year") else ""
            title_line = f"<b>{_esc(f.get('title',''))}{year_bit}</b>"
            if f.get("box_office"):
                title_line += f" — {_esc(f['box_office'])}"
            story.append(Paragraph(title_line, s["bullet"]))
            if f.get("reception_note"):
                story.append(Paragraph(_esc(f["reception_note"]), s["body"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
