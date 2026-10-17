NEURAL NETWORK LEARNING

Start with learning_sine.ipynb. The notebook contains explanations, simple
comment-free code cells, and saved outputs from a complete run.

neural_network_learning.pdf is the 18-slide Beamer presentation.
neural_network_learning.tex is its editable LaTeX source.
Keep figures/ and results/ beside the TeX file when copying the source.

RUN THE NOTEBOOK

Open a terminal in this folder. Create an environment if needed:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Open learning_sine.ipynb in VS Code or your existing Jupyter application.
Select this environment as the Python kernel. Start the notebook with this
folder as its working directory. Restart the kernel and run all cells.

If you do not already have a notebook application:

    python -m pip install jupyterlab
    jupyter lab learning_sine.ipynb

The example runs on the CPU. It uses 64 training examples, 16 hidden neurons,
49 trainable parameters, and 10,000 full-batch gradient descent updates.
The learning rate is 0.05. The seed is 7. The six test inputs are distinct
from all training inputs and lie inside [0, pi/2]. All angles are radians.

Verified with Python 3.13.3, PyTorch 2.14.1, and Matplotlib 3.11.2 on macOS.
Saved run: training MSE 0.00011814, test MSE 0.00006936, and maximum absolute
error 0.01053 across the six test points. Small version-dependent differences
are possible. Requirements are unpinned so the notebook remains easy to install.

UPDATE THE SLIDES AFTER RERUNNING

Save the executed notebook, then run:

    python refresh_slide_results.py
    pdflatex -interaction=nonstopmode -halt-on-error neural_network_learning.tex
    pdflatex -interaction=nonstopmode -halt-on-error neural_network_learning.tex

The notebook regenerates the four vector PDF plots in figures/.
The refresh script reads the saved numerical outputs into results/.
It uses only Python's standard library. LaTeX compilation needs Beamer,
listings, and Latin Modern (available in a normal TeX Live/MacTeX setup).

If you change the architecture, test points, or training settings for a new
lesson, update the corresponding slide text and code too. The refresh script
expects the original six-row test table and its existing output headings.

TEACHING SEQUENCE

Slides 1-5: a network as a parameterized function, one neuron, the 49
parameters, and the role of a nonlinear activation.
Slides 6-11: data, architecture, MSE, chain rule, gradient descent, and code.
Slides 12-16: training loss, learned function, testing, values, and errors.
Slides 17-18: what the experiment shows and suggested student experiments.

The notebook explains each operation in Markdown instead of code comments.
Training learns an approximation. The six test points provide an interpolation
check, not a guarantee across the interval or beyond it. The known sine
formula is used to create targets and check results, not inside the model.

REFERENCES

https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
