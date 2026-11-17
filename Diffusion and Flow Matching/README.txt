DIFFUSION MODELS AND ODE GENERATION

Start with learning_flow_matching.ipynb. It contains explanations in Markdown,
comment-free code cells, executed outputs, and student experiments.

flow_matching.pdf is the 31-slide presentation in the blue Madrid theme.
flow_matching.tex is the complete editable Beamer source. Keep figures/ and
results/ beside it when copying the source.

METHOD AND SCOPE

The implemented method is flow matching with independent endpoint pairs and
straight interpolation paths, closely related to the first rectified-flow
training step. It is a simple generative ODE, not a DDPM or stochastic
diffusion simulation. The slides and notebook explain the connection to
score-based diffusion and its probability-flow ODE without conflating them.

Noise is a standard Gaussian in two dimensions. The target is an equally
weighted four-component Gaussian mixture with centers (-2,-2), (-2,2),
(2,-2), and (2,2). Each component has standard deviation 0.35 in each
coordinate and independent coordinate noise.

Training: 4,096 examples, 5,000 iterations, batch size 256, seed 7.
Architecture: 3 inputs, two width-64 SiLU hidden layers, 2 outputs,
4,546 parameters. The third input is artificial generation time.
Loss: mean squared velocity error, averaged over examples and coordinates.
Optimizer: Adam with learning rate 0.001 and PyTorch's default betas.
Sampling: forward Euler, 100 steps from t=0 (noise) to t=1 (data).
Evaluation: seed 2026, 5,000 fresh noise points and 5,000 fresh real points.

The target sampling formula creates the data, but the trained sampler uses
only the velocity network, a Gaussian starting point, and the time variable.
The experiment generates points in R^2, not time series. Generation time
is not physical time or a time-series index. Volterra signatures are future work.

RUNNING THE NOTEBOOK

From the project root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Open learning_flow_matching.ipynb in VS Code or your Jupyter application and
select the environment as the kernel. Use this lesson folder as the working
directory so figures are saved here. Restart the kernel and run all cells.

If needed, install a notebook application:

    python -m pip install jupyterlab
    jupyter lab

The notebook needs no external data or GPU. The supplied run uses Python
3.13.3, PyTorch 2.14.1, and Matplotlib 3.11.2 on macOS. Requirements are
unpinned for easier installation. Versions, hardware, and seeds can affect
results. The saved outputs can be read without running the notebook.

REBUILDING THE PRESENTATION

After running and saving the notebook, run these commands in this folder:

    python refresh_slide_results.py
    pdflatex -interaction=nonstopmode -halt-on-error flow_matching.tex
    pdflatex -interaction=nonstopmode -halt-on-error flow_matching.tex

The notebook creates six vector PDF plots in figures/.
The additional diffusion_noise_strip.png is an AI-generated conceptual
illustration of forward noising, not a model result from this notebook.
The arrows and explanations on slide 2 describe both directions. The refresh script
reads the saved evaluation statistics into results/comparison_table.tex.
It uses only Python's standard library. LaTeX needs Beamer, listings, and
Latin Modern, available in a normal TeX Live/MacTeX installation.

The generation-stages plot assumes the default 100 sampling steps. If you
change the sampler, change its snapshot indices as well. For architecture,
center, spread, or training changes, update the slide text and code snippets.
The refresh script expects the original four-center output format.

VIDEO

videos/gaussian_to_gmm.mp4 is a 13-second animation of 2,000 points moving
under the trained velocity field. It uses 300 Euler steps for smooth motion.
The orange reference cloud is sampled independently from the target GMM.
It does not supply endpoints to the learned sampler.

To reproduce the video, run from this lesson folder:

    python -m pip install imageio-ffmpeg
    python create_flow_video.py

On its first run, the script repeats the notebook's training with seed 7 and
5,000 updates, then saves videos/flow_matching_model.pt. Subsequent runs
load that checkpoint. The script also saves the generated trajectories and
video_metadata.json. It does not overwrite the notebook's plots or results.

videos/learned_trajectories.mp4 follows the first 24 samples from the saved
trajectories for 14 seconds. Hollow circles mark starting positions, filled
dots mark current positions, and growing trails show the paths already
traveled. Colors indicate the nearest final cluster, assigned afterward.
Rebuild it without retraining by running:

    python create_trajectory_video.py

TEACHING SEQUENCE

Slides 1-10: a diffusion noising illustration, the target mixture and its
sum, sampling, data, ODEs, Euler steps, and method names.
Slides 11-19: interpolation, velocity targets, conditional averaging, the
network, tensor shapes, loss, optimizer, and the training code.
Slides 20-24: training loss, ODE sampling, new samples, and trajectories.
Slides 25-28: distribution checks and a controlled Euler step-size experiment.
Slides 29-31: diffusion's probability-flow ODE, limitations, and exercises.

KEY DISTINCTIONS

The network predicts velocity, not position, a component label, or a score.
Straight training paths do not imply straight learned ODE trajectories.
The MSE target has conditional variability, so zero loss is not expected.
Training iterations update parameters. Sampling steps move positions.
The same initial noise and fixed model produce the same Euler trajectory.
New noise produces new samples; no paired correct endpoint is supplied.
Counts near the four centers do not establish correct within-cluster spread.
More Euler steps cannot repair a poorly learned velocity field.

If evaluation guides tuning, reserve fresh data for a final test. Report
both good and bad runs. The target sampling formula is already exact for
this synthetic example; learning it is an educational exercise.

REFERENCES

https://arxiv.org/abs/2210.02747
https://arxiv.org/abs/2209.03003
https://arxiv.org/abs/2011.13456
