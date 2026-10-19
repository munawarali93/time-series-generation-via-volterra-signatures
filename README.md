# Time Series Generation via Volterra Signatures

This is a Directed Reading Program (DRP) project with undergraduate students at Florida State University, based in the Department of Mathematics. Students bring backgrounds in mathematics, finance, and other areas. We are building the mathematical and computational foundations needed to explore time series generation via Volterra signatures.

The project begins with signatures, then introduces neural networks, generative adversarial networks, ODE-based generation, and DDPM through small examples. The notebooks favor short, readable code, with explanations in Markdown. Each computational lesson includes a Beamer presentation, editable LaTeX source, and plots from an executed notebook.

## Progress and milestones

The weeks below describe the teaching sequence, rather than calendar dates. “Materials ready” means the lesson is prepared; it does not mean the class has already completed it.

| Week | Milestone | Main activity | Progress |
| --- | --- | --- | --- |
| 1 | Introduction to signatures | Discuss signatures as the starting point for the project. | Discussed in the first meeting; materials are not yet included here. |
| 2 | Understanding neural networks | Learn how weights, biases, activations, loss, and gradient descent fit a model to data. Approximate `sin(x)` on `[0, pi]` and test on new points in `[0, pi/2]`. | Notebook and slides ready. |
| 3 | Understanding GANs | Train a generator and discriminator on a simple scalar distribution. Generate new samples and compare distributions. | Notebook and slides ready. |
| 4 | Diffusion models and ODE generation | Learn a velocity field by flow matching, then use Euler steps to turn Gaussian noise into a four-component Gaussian mixture. Explain the connection to diffusion's probability-flow ODE. | Notebook and slides ready. |
| 5 | Denoising diffusion probabilistic models (DDPM) | Generate the same four-component mixture using a fixed forward noise schedule, noise-prediction training, and stochastic reverse sampling. | Notebook and slides ready. |
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

### Diffusion models and ODE generation: four Gaussian components

- [Executed notebook](Diffusion%20and%20Flow%20Matching/learning_flow_matching.ipynb)
- [28-slide Beamer PDF](Diffusion%20and%20Flow%20Matching/flow_matching.pdf)
- [Editable LaTeX source](Diffusion%20and%20Flow%20Matching/flow_matching.tex)
- [Setup and teaching notes](Diffusion%20and%20Flow%20Matching/README.txt)

This lesson turns two-dimensional standard Gaussian noise into an equally weighted mixture centered at `(-2,-2)`, `(-2,2)`, `(2,-2)`, and `(2,2)`. It implements flow matching with straight training interpolations and a simple Euler ODE sampler. The notebook includes evolving point clouds, individual trajectories, cluster diagnostics, and a comparison of 1, 10, and 100 sampling steps.

The implemented method learns a velocity directly. It is not a DDPM or a stochastic diffusion simulation; the lesson explains its relationship to score-based diffusion and probability-flow ODEs. Generation time is an artificial interpolation parameter, not time in a time series.

### DDPM: learning to reverse Gaussian noise

- [Executed notebook](DDPM%20Learning/learning_ddpm.ipynb)
- [28-slide blue Beamer PDF](DDPM%20Learning/ddpm_learning.pdf)
- [Editable LaTeX source](DDPM%20Learning/ddpm_learning.tex)
- [Setup and teaching notes](DDPM%20Learning/README.txt)

This lesson uses the same four Gaussian components as the ODE lesson. A small network learns to predict the noise added at a randomly selected diffusion timestep. Sampling starts from standard Gaussian noise and follows 1,000 reverse transitions, with fresh noise at every step except the last. The notebook includes forward noising, training loss, reverse point clouds, stochastic trajectories, and evaluation against fresh real samples.

An experiment removes the reverse-step noise while keeping the same initial points. The resulting distorted clusters illustrate why a mean-only chain is a different sampler; it is not DDIM or a probability-flow ODE. The saved DDPM run covers all four modes but has somewhat tighter clusters than the target. These teaching examples are not a controlled comparison of methods.

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
- Record students' observations from the neural network, GAN, flow-matching, and DDPM experiments.
- Develop the Volterra-signature reading list and choose a precise research question.
- Define a time series dataset, baselines, and evaluation measures that examine temporal dependence.

## References

- Goodfellow et al. (2014), [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).
- PyTorch, [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
- PyTorch, [DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Liu, Gong, and Liu, [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003).
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).
- Ho, Jain, and Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Song, Meng, and Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).

This repository records a teaching and research project in progress.
