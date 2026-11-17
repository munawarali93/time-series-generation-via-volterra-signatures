"""Convert the readable percent-cell script to a notebook and optionally execute."""
import argparse
import re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
source = (root / 'learning_wasserstein.py').read_text()
cells = []
for block in re.split(r'^# %%', source, flags=re.M)[1:]:
    heading, _, body = block.partition('\n')
    if '[markdown]' in heading:
        body = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line
                         for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nb.metadata.language_info = dict(name='python')
if args.execute:
    NotebookClient(nb, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, root / 'learning_wasserstein.ipynb')
print('Saved', len(cells), 'cells', '(executed)' if args.execute else '(unexecuted)')
