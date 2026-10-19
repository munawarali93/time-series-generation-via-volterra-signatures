import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
notebook = json.loads((root / "learning_ddpm.ipynb").read_text())
streams = []
for cell in notebook["cells"]:
    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            value = output["text"]
            streams.append("".join(value) if isinstance(value, list) else value)
text = "\n".join(streams)
rows = {}
for label in ["Real test", "DDPM"]:
    match = re.search(r"^" + label + r"\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*$", text, re.M)
    if not match:
        raise ValueError("Run and save the complete notebook before refreshing the slides.")
    rows[label] = match.groups()
names = [r"Nearest $(-2,-2)$ (\%)", r"Nearest $(-2,2)$ (\%)",
         r"Nearest $(2,-2)$ (\%)", r"Nearest $(2,2)$ (\%)",
         "RMS distance to nearest center", r"Within distance 1 of a center (\%)"]
table = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Diagnostic & Fresh real & DDPM\\", r"\midrule"]
for i, name in enumerate(names):
    table.append(" & ".join([name, rows["Real test"][i], rows["DDPM"][i]]) + r"\\[1mm]")
table.extend([r"\bottomrule", r"\end{tabular}"])
(root / "results").mkdir(exist_ok=True)
(root / "results" / "comparison_table.tex").write_text("\n".join(table) + "\n")
match = re.search(r"Mean-only RMS radius: ([-0-9.]+)", text)
if not match:
    raise ValueError("The notebook must include the mean-only sampling experiment.")
(root / "results" / "ablation_metric.tex").write_text(
    f"RMS radius: real {rows['Real test'][4]}, DDPM {rows['DDPM'][4]}, mean-only {match[1]}.\n"
)
print("Updated slide statistics from the saved notebook outputs.")
