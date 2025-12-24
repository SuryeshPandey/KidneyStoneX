def generate_report(m):
    if m["count"] == 0:
        finding = "No renal calculi detected."
    elif m["count"] == 1:
        finding = "One renal calculus detected."
    else:
        finding = f"{m['count']} renal calculi detected."

    return f"""
CT KIDNEY STONE DIAGNOSTIC REPORT

Findings:
- {finding}
- Average stone area: {m['avg_area']:.2f} px²
- Mean intensity: {m['mean_intensity']:.2f}
- Kidney side: {m['side']}

Impression:
Findings suggest nephrolithiasis. Correlate clinically.
""".strip()
