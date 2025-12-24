def interpret_stone(count, avg_area):
    if count == 0:
        return "No calculi identified on this single axial image."
    if avg_area < 100:
        return "Calcifications appear small. Clinical correlation recommended."
    if avg_area < 1000:
        return "Moderate sized calculi. Consider clinical context for management."
    return "Larger calculus noted. Consider urology referral and confirm with DICOM CT."

def recommend(count, avg_area):
    if count == 0:
        return "- No acute intervention suggested based on this slice alone. Correlate clinically."
    if avg_area < 100:
        return "- Conservative management and follow-up imaging if symptomatic."
    if avg_area < 1000:
        return "- Consider urology follow-up and measurement on full DICOM CT."
    return "- Larger stone; consider formal CT with pixel spacing and surgical planning."

def generate_ct_report(stone_count, avg_area, mean_intensity, side):
    if stone_count == 0:
        findings = "No renal calculi were detected on the supplied axial image."
    elif stone_count == 1:
        findings = "One renal calculus was detected."
    else:
        findings = f"{stone_count} renal calculi were detected."
    side_text = "Kidney side could not be confidently determined."
    if side in ("Left", "Right"):
        side_text = f"The stone is located in the {side} kidney."
    report = f"""
CT KIDNEY STONE DIAGNOSTIC REPORT

Findings:
- {findings}
- Average detected stone area: {avg_area:.2f} px².
- Mean CT intensity (pixel units): {mean_intensity:.2f}.
- {side_text}

Impression:
{interpret_stone(stone_count, avg_area)}

Recommendations:
{recommend(stone_count, avg_area)}
"""
    return report.strip()
