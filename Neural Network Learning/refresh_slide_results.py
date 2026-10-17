import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
notebook = json.loads((root / "learning_sine.ipynb").read_text())
streams = []
for cell in notebook["cells"]:
    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            value = output["text"]
            streams.append("".join(value) if isinstance(value, list) else value)
text = "\n".join(streams)
train = re.search(r"Final training MSE: ([0-9.]+)", text)
test = re.search(r"Test MSE: ([0-9.]+)", text)
maximum = re.search(r"Maximum test absolute error: ([0-9.]+)", text)
rows = re.findall(r"^\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s*$", text, re.M)
if not (train and test and maximum and len(rows) == 6):
    raise ValueError("Run and save the complete notebook before refreshing slide results.")
results = root / "results"
results.mkdir(exist_ok=True)
(results / "training_metric.tex").write_text(
    f"Final training MSE after 10,000 updates: $ {train[1]} $.\n"
)
(results / "test_metric.tex").write_text(
    f"Test MSE: $ {test[1]} $\\\\[2mm]\n"
    f"Maximum absolute error at these six points: $ {maximum[1]} $.\n"
)
table = [
    r"\begin{tabular}{rrrr}", r"\toprule",
    r"$x$ (radians) & $\sin(x)$ & Prediction & Absolute error\\", r"\midrule"
]
table.extend(" & ".join(row) + r"\\[1mm]" for row in rows)
table.extend([r"\bottomrule", r"\end{tabular}"])
(results / "test_table.tex").write_text("\n".join(table) + "\n")
print("Updated slide results from the saved notebook outputs.")
