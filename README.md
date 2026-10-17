# Time Series Generation via Volterra Signatures

This is a Directed Reading Program (DRP) project with undergraduate students at Florida State University, based in the Department of Mathematics. Students bring backgrounds in mathematics, finance, and other areas. We are building the mathematical and computational foundations needed to explore time series generation via Volterra signatures.

The project begins with signatures, then introduces neural networks and generative adversarial networks through small examples. The notebooks favor short, readable code, with explanations in Markdown. Each computational lesson includes a Beamer presentation, editable LaTeX source, and plots from an executed notebook.

## Progress and milestones

The weeks below describe the teaching sequence, rather than calendar dates. “Materials ready” means the lesson is prepared; it does not mean the class has already completed it.

| Week | Milestone | Main activity | Progress |
| --- | --- | --- | --- |
| 1 | Introduction to signatures | Discuss signatures as the starting point for the project. | Discussed in the first meeting; materials are not yet included here. |
| 2 | Understanding neural networks | Learn how weights, biases, activations, loss, and gradient descent fit a model to data. Approximate `sin(x)` on `[0, pi]` and test on new points in `[0, pi/2]`. | Notebook and slides ready. |
| 3 | Understanding GANs | Train a generator and discriminator on a simple scalar distribution. Generate new samples and compare distributions. | Notebook and slides ready. |
| Later | Time series generation via Volterra signatures | Study the relevant theory, design a time series experiment, and investigate how Volterra signatures can be used for generation or evaluation. | Planned; method and experiments remain to be developed. |

The current examples teach the prerequisites. They do not yet implement Volterra signatures or a time series generator.

## Teaching materials

### Neural networks: learning a sinusoidal function

- [Executed notebook](Neural%20Network%20Learning/learning_sine.ipynb)
- [18-slide Beamer PDF](Neural%20Network%20Learning/neural_network_learning.pdf)
- [Editable LaTeX source](Neural%20Network%20Learning/neural_network_learning.tex)
- [Setup and teaching notes](Neural%20Network%20Learning/README.txt)

This lesson uses one hidden layer with 16 neurons and 49 trainable parameters. Students follow data generation, architecture, mean squared error, gradient descent, and testing.

### GANs: learning to generate a distribution

- [Executed notebook](GAN%20Learning/learning_gans.ipynb)
- [20-slide Beamer PDF](GAN%20Learning/gan_learning.pdf)
- [Editable LaTeX source](GAN%20Learning/gan_learning.tex)
- [Setup and teaching notes](GAN%20Learning/README.txt)

This lesson uses synthetic scalar data from a normal distribution with mean 2 and standard deviation 0.5. A small generator transforms random noise into samples, while a discriminator supplies the training signal. Histograms, summary statistics, and quantiles provide a first evaluation. No image dataset, external data download, or GPU is required.

## Running the notebooks

Create a Python environment in the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open a notebook in VS Code or your existing Jupyter application and select this environment as the kernel. Run with the lesson folder as the working directory so plots are saved in its `figures/` folder. Restart the kernel and run all cells to repeat an experiment from the beginning.

If you need a notebook application, install JupyterLab:

```bash
python -m pip install jupyterlab
jupyter lab
```

The notebooks include saved outputs, so they can also be read without rerunning them. Results can differ across seeds, package versions, and platforms. GAN training is especially sensitive to these choices; compare the generated distribution rather than relying only on the losses.

## Rebuilding a presentation

After running and saving a notebook, open a terminal in its lesson folder and run `python refresh_slide_results.py`. Then compile the corresponding `.tex` file twice with `pdflatex`. The lesson's `README.txt` gives the exact commands. Keep the `figures/` and `results/` folders with the LaTeX source.

## Next steps

- Add the introductory signature notes from the first meeting.
- Record students' observations from the neural network and GAN experiments.
- Develop the Volterra-signature reading list and choose a precise research question.
- Define a time series dataset, baselines, and evaluation measures that examine temporal dependence.

## References

- Goodfellow et al. (2014), [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).
- PyTorch, [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
- PyTorch, [DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

This repository records a teaching and research project in progress.
