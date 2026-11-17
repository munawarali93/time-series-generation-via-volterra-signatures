DDPM LEARNING

Start with learning_ddpm.ipynb. It contains Markdown explanations, short
comment-free code cells, saved outputs, and a sampling experiment.

CLASS AND REFERENCE VERSIONS

ddpm_learning_short.pdf is the original 28-slide deck for class.
ddpm_learning_short.tex is its editable source.
ddpm_learning.pdf is the retained 48-slide reference deck with derivations.
ddpm_learning.tex is its editable source.
Both use the blue Madrid theme. Keep figures/ and results/ with either TeX
file when copying it.

THE MODEL

This lesson implements a denoising diffusion probabilistic model (DDPM) on the
same four-component Gaussian mixture used in the ODE lesson. The four equally
weighted centers are (-2,-2), (-2,2), (2,-2), and (2,2). Each component has
standard deviation 0.35 per coordinate and independent coordinate noise.

Data: 4,096 training points. Seed: 7.
Forward process: 1,000 timesteps, beta linearly spaced from 0.0001 to 0.02.
Terminal alpha_bar is approximately 0.00004036, giving a nearly Gaussian prior.
Network: 3 inputs, two width-64 SiLU hidden layers, 2 noise outputs,
4,546 parameters. The third input is normalized diffusion time.
Training: 10,000 iterations, batch size 256, noise MSE, Adam at 0.001.
Sampling: every reverse timestep, fixed posterior variance beta_tilde,
and no added noise at the last step. There is no clipping to image bounds.
Evaluation: seed 2026, 5,000 fresh Gaussian starts, and 5,000 fresh real samples.

The fixed-variance Gaussian reverse transitions are an approximation. The
notebook uses the simplified noise-prediction objective, not the full
variational likelihood bound or a learned reverse variance.

RUN THE NOTEBOOK

From the project root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Open the notebook in VS Code or Jupyter and select this environment as the
kernel. Use DDPM Learning as the working directory. Restart the kernel and
run all cells to repeat an experiment from the beginning.

If you need a notebook application:

    python -m pip install jupyterlab
    jupyter lab

No external dataset, image download, or GPU is needed. The supplied run uses
Python 3.13.3, PyTorch 2.14.1, and Matplotlib 3.11.2 on macOS. Requirements
are unpinned for easy installation. Other versions or seeds can change results.

REBUILD THE PRESENTATION

Run and save the complete notebook. From DDPM Learning:

    python refresh_slide_results.py
    pdflatex -interaction=nonstopmode -halt-on-error ddpm_learning.tex
    pdflatex -interaction=nonstopmode -halt-on-error ddpm_learning.tex

The notebook creates seven vector PDF plots in figures/. The refresh script
uses only the standard library and reads saved statistics into results/.
Compilation needs Beamer, listings, and Latin Modern (normal TeX Live/MacTeX).
To rebuild the class deck, compile ddpm_learning_short.tex twice instead.

The plots assume T=1000 and the original four centers. If you change T,
change the forward/reverse snapshot timesteps too. Update slide settings,
code excerpts, and evaluation descriptions when changing the experiment.

TEACHING SEQUENCE (48-SLIDE REFERENCE DECK)

Slides 1-5: notation, the goal, and training data.
Slides 6-13: the forward schedule, marginal derivation, and Gaussian start.
Slides 14-22: the network, MSE, its conditional-mean target, and training.
Slides 23-31: clean estimates, Bayes' rule, completing the square, posterior
variance and mean, noise substitution, and a worked numerical example.
Slides 32-34: fixed sampler variance, missing clean-data uncertainty, and t=1.
Slides 35-37: Gaussian KL, weighted noise MSE, and the simplified objective.
Slides 38-43: sampling code, reverse stages, trajectories, and samples.
Slides 44-48: diagnostic definitions, results, noise ablation, and exercises.

The detailed derivation is in the slides themselves. The existing notebook
and saved experiment outputs are unchanged. The numerical example on slide
31 uses an illustrative schedule step, not the notebook's actual beta values.

POINTS TO EMPHASIZE

In this lesson x_0 is data and x_T is noise. Sampling runs backward.
Python index k corresponds to mathematical timestep t=k+1.
Training uses aggregate noise in the closed-form conditional marginal.
Independent draws at different noise levels are not a Markov trajectory.
The network predicts noise, not the velocity used in the ODE lesson.
Training iteration count and diffusion timestep count are different.
The tilde in beta_tilde distinguishes posterior variance from forward beta.
The posterior q(x_{t-1} | x_t, x_0) is Gaussian for t>1. Averaging over
unknown x_0 generally gives a non-Gaussian true reverse conditional.
Ideal noise MSE predicts E[epsilon | x_t,t], so the learned-mean formula
then equals the true reverse mean. Fixed beta_tilde omits uncertainty
about the clean point and does not give the exact true reverse covariance.
At t=1 the known-clean-point posterior is a point mass. The sampler's
last mean is a clean-point estimate; the true q(x_0 | x_1) can be uncertain.
Only the final reverse step omits noise in the default DDPM sampler.
The same initial point can have different endpoints with new reverse noises.
The mean-only experiment is not DDIM or a probability-flow ODE.
Do not skip diffusion indices while keeping the original one-step update.

Compare component weights and within-cluster spread. Smaller radius alone
is not better: a collapsed cluster has radius zero. The model is approximate,
and these diagnostics do not certify a complete match. The previous ODE
lesson has different compute settings, so this is not a controlled benchmark.

In the supplied run, DDPM covers all four modes (about 24-27% each), but its
RMS distance to a target center is 0.4246 versus 0.4934 for fresh real data.
The mean-only chain shifts clusters inward and distorts their shapes. Its
similar RMS radius of 0.4292 is a useful example of a scalar diagnostic
missing a visible distribution mismatch. Reinterpret results after rerunning.

If evaluation guides tuning, use fresh samples for the final test. The
example generates 2D points, not time series. Volterra signatures remain
future project work.

REFERENCES

https://arxiv.org/abs/2006.11239
https://github.com/hojonathanho/diffusion
https://arxiv.org/abs/2010.02502
