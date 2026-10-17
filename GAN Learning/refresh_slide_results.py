import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
notebook = json.loads((root / "learning_gans.ipynb").read_text())
streams = []
for cell in notebook["cells"]:
    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            value = output["text"]
            streams.append("".join(value) if isinstance(value, list) else value)
text = "\n".join(streams)
rows = {}
for label in ["Real test", "Generated"]:
    match = re.search(r"^" + label + r"\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*$", text, re.M)
    if not match:
        raise ValueError("Run and save the complete notebook before refreshing the slides.")
    rows[label] = match.groups()
table = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Statistic & Target & Real test & Generated\\", r"\midrule"]
names = ["Mean", "Standard deviation", "10th percentile", "Median", "90th percentile"]
targets = ["2.00000", "0.50000", "1.35922", "2.00000", "2.64078"]
for i, name in enumerate(names):
    table.append(" & ".join([name, targets[i], rows["Real test"][i], rows["Generated"][i]]) + r"\\[1.5mm]")
table.extend([r"\bottomrule", r"\end{tabular}"])
(root / "results").mkdir(exist_ok=True)
(root / "results" / "comparison_table.tex").write_text("\n".join(table) + "\n")
print("Updated the slide table from the executed notebook.")
