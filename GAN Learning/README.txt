GAN LEARNING

Start with learning_gans.ipynb. It contains short, comment-free code cells,
Markdown explanations, and saved outputs from a full run.

gan_learning.pdf is the 20-slide Beamer presentation in the blue Madrid theme.
gan_learning.tex is its editable LaTeX source. Keep figures/ and results/
beside the TeX file when copying it.

THE EXAMPLE

Generate scalar samples resembling a normal distribution with mean 2 and
standard deviation 0.5. This synthetic example requires no images, downloaded
dataset, or GPU. It teaches GAN training, not a better sampler for a known
normal distribution. It does not yet generate time series or use signatures.

There are 2,048 training examples. The generator and discriminator each have
16 hidden neurons and 49 parameters. Training uses 6,000 iterations with one
discriminator update and one generator update per iteration, batch size 128,
and Adam with learning rate 0.001 and betas (0.5, 0.999).

Training seed: 7. Evaluation seed: 2026. Evaluation compares 5,000 newly
generated samples with 5,000 new real samples. There are no paired targets.

RUN THE NOTEBOOK

From the repository root, create a Python environment if needed:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Open learning_gans.ipynb in VS Code or an existing Jupyter application.
Select the environment as the kernel. Run with GAN Learning as the working
directory. Restart the kernel and run all cells to repeat the experiment.

If you need a notebook application:

    python -m pip install jupyterlab
    jupyter lab

The notebook runs on the CPU. The supplied run uses Python 3.13.3,
PyTorch 2.14.1, and Matplotlib 3.11.2 on macOS. Requirements are unpinned
for easy installation. Different seeds and environments can change the result.

UPDATE THE SLIDES

Run all cells and save the notebook. In the GAN Learning folder, run:

    python refresh_slide_results.py
    pdflatex -interaction=nonstopmode -halt-on-error gan_learning.tex
    pdflatex -interaction=nonstopmode -halt-on-error gan_learning.tex

The notebook regenerates four vector PDF figures. The refresh script reads
the saved statistics into results/comparison_table.tex using only Python's
standard library. Compile with a normal TeX Live/MacTeX setup containing
Beamer, listings, and Latin Modern.

The refresh script assumes the original normal target and output headings.
If you change the data distribution or training settings, update the target
statistics, slide text, and code snippets accordingly.

TEACHING SEQUENCE

Slides 1-6: generation versus regression, real data, and the two networks.
Slides 7-10: binary cross-entropy, discriminator and generator losses, Adam.
Slides 11-13: the alternating training loop, detach, and gradient flow.
Slides 14-18: loss curves, generation, histograms, statistics, and quantiles.
Slides 19-20: connection to time series and student experiments.

TEACHING NOTES

D returns logits. BCEWithLogitsLoss handles sigmoid internally.
The discriminator loss is the sum of two batch means, so its ideal 0.5-output
reference is 2 log(2), while the generator reference is log(2).
These reference values are not proof that generation has succeeded.

fake.detach() blocks generator gradients during the discriminator step.
D.requires_grad_(False) freezes discriminator parameters during the generator
step, while preserving derivatives with respect to its input.

The generator uses the non-saturating objective. Evaluation compares
distributions, not individual pairs of real and generated samples. Quantiles
show differences that a mean and standard deviation can miss. Finite samples,
GAN instability, and the bounded Tanh architecture limit the approximation.

In the supplied run, the generated mean and standard deviation are close to
the targets, but generated samples accumulate near 2.7 and the upper quantiles
flatten. Discuss this partial fit explicitly: matching moments is not the
same as matching the whole distribution.

After tuning on evaluation results, use new samples for a final test.

REFERENCES

https://arxiv.org/abs/1406.2661
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
