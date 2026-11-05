"""Generate three CVPR/LaTeX-compatible tables using measured output only."""
from __future__ import annotations

EOL = "\\" * 2   # a literal LaTeX row terminator (two backslashes)


def escape(value):
    """Escape labels from manifest.csv before writing LaTeX."""
    conversions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                   "_": r"\_", "#": r"\#", "$": r"\$"}
    return "".join(conversions.get(c, c) for c in str(value))


def fmt(value, digits=3):
    return "--" if value is None else f"{float(value):.{digits}f}"


def make_tables(directory, condition_rows, ablation_rows):
    baseline = 3 if any(r["confirmation_frames"] == 3 for r in condition_rows) else (
        min((r["confirmation_frames"] for r in condition_rows), default=3))
    positives = [r for r in condition_rows if r["confirmation_frames"] == baseline
                 and r["experiment"].lower() != "negative" and r["event_type"] == "ALL"]
    negatives = [r for r in condition_rows if r["confirmation_frames"] == baseline
                 and r["experiment"].lower() == "negative" and r["event_type"] == "PINCH_START"]

    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{Pinch recognition across conditions. Events matched to independently labelled thumb-index contact and separation.}",
            r"\label{tab:pinch_conditions}", r"\begin{tabular}{lrrrr}", r"\hline",
            "Condition & GT & Recall & F1 & $|\\Delta t|$ (ms) " + EOL,
            r"\hline"]
    for r in sorted(positives, key=lambda z: (z["experiment"], z["condition"])):
        body.append(f"{escape(r['experiment'] + ': ' + r['condition'])} & {r['annotated_events']} & "
                    f"{fmt(r['recall'])} & {fmt(r['f1'])} & "
                    f"{fmt(r['median_absolute_delay_ms'], 1)} " + EOL)
    body += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (directory / "table_conditions.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{False PINCH\_START activations during negative-control actions.}",
            r"\label{tab:pinch_negative}", r"\begin{tabular}{lrr}", r"\hline",
            "Control & False starts & False starts/min " + EOL, r"\hline"]
    for r in sorted(negatives, key=lambda z: z["condition"]):
        body.append(f"{escape(r['condition'])} & {r['predicted_events']} & "
                    f"{fmt(r['false_starts_per_min'])} " + EOL)
    body += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (directory / "table_negative.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{Temporal-confirmation ablation on the same recorded landmarks.}",
            r"\label{tab:pinch_ablation}", r"\begin{tabular}{rrrr}", r"\hline",
            "Frames & Positive F1 & $|\\Delta t|$ (ms) & False starts/min " + EOL,
            r"\hline"]
    for r in ablation_rows:
        body.append(f"{r['confirmation_frames']} & {fmt(r['positive_f1'])} & "
                    f"{fmt(r['median_absolute_delay_ms'], 1)} & "
                    f"{fmt(r['negative_false_starts_per_min'])} " + EOL)
    body += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (directory / "table_ablation.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
