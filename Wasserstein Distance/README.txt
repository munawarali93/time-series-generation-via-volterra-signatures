WASSERSTEIN DISTANCE

A 35-slide Beamer lesson for the DRP, matching the existing blue Madrid theme.
Start with wasserstein_distance.pdf and learning_wasserstein.ipynb.
The notebook has saved outputs and figures so it can be read without execution.

FILES

wasserstein_distance.pdf        35-slide teaching presentation
wasserstein_distance.tex        Editable Beamer source
learning_wasserstein.ipynb      Executed, explained notebook
learning_wasserstein.py         The same lesson as a runnable percent-cell script
ot_tools.py                    Small mathematical routines, with docstrings
check_examples.py              Independent mathematical cross-checks
build_notebook.py              Convert the script to a notebook, optionally execute
refresh_slide_results.py       Refresh slide tables from results/metrics.json
figures/                      13 reproducible figures in vector PDF and PNG
results/                      Measured results, versions, and generated LaTeX tables
requirements.txt              Python dependencies for this lesson

Keep figures/ and results/ next to the LaTeX source when copying the lesson.
Keep ot_tools.py and refresh_slide_results.py beside the notebook and script.
No external dataset, API key, GPU, or neural-network training is needed.

SETUP AND RUN

From the DRP repository root, using the existing environment:

    source .venv/bin/activate
    python -m pip install -r 'Wasserstein Distance/requirements.txt'
    cd 'Wasserstein Distance'
    python learning_wasserstein.py
    python check_examples.py

For a new environment, first run python3 -m venv .venv in the repository root.
Open the notebook in VS Code or Jupyter and select this Python environment.
Use the lesson folder as the notebook working directory. Restart and Run All.
The notebook overwrites its generated figures and results with the new run.

To regenerate and execute the notebook from the script:

    python build_notebook.py --execute

Without --execute, build_notebook.py writes a notebook without saved outputs.
The .py lesson is the authoring source. Changes made only in the .ipynb will
be overwritten by this conversion, so transfer any edits back to the script.

To rebuild the slides after running the examples:

    python refresh_slide_results.py
    pdflatex -interaction=nonstopmode -halt-on-error wasserstein_distance.tex
    pdflatex -interaction=nonstopmode -halt-on-error wasserstein_distance.tex

The script and notebook already refresh the tables automatically. The separate
refresh command is useful when a results file has been supplied independently.
Mathematical examples and slide prose describe the default parameters. If you
change these parameters, update the corresponding formulas and prose too.

A normal TeX Live/MacTeX installation with Beamer, Latin Modern, and listings
suffices. Requirements are unpinned for portability. The exact versions for
the delivered run are recorded in results/metrics.json. Different package
versions can select different optimal plans when an optimum is nonunique.

TEACHING SEQUENCE

Suggested meeting 1 (60-75 minutes): slides 1-20 and notebook sections 1-6.
Slides 2-7 define point masses, compute Wasserstein, TV, and KL by hand,
and explain the comparison graph. Then introduce the general coupling definition.
Derive W2 for Gaussians using a shared standard normal variable. Pause for
students to calculate the three mean/variance cases before showing figures.
Run the sorting example and compare population formulas with sample results.

Suggested meeting 2 (60-75 minutes): slides 21-35 and notebook sections 7-13.
Run the Sinkhorn experiment. Ask students to describe what each heatmap entry
means. Next ask them to predict image and time-series distances before code.
Use representation choices as the bridge to signatures and MMD.

For a shorter meeting, slides 1-20 and 23-27 form a core introduction.
Sinkhorn, signatures, WGAN, and MMD can be a second reading session.
The notebook also contains displacement interpolation and the multivariate
Gaussian W2 formula, which are supplementary to the 35-slide deck.

MATHEMATICAL NOTES FOR THE INSTRUCTOR

1. Why the Gaussian formula is an optimum

For any coupling, E[(X-Y)^2] equals the squared mean difference plus
sigma1^2 + sigma2^2 - 2 Cov(X,Y). Cauchy-Schwarz bounds the covariance by
sigma1*sigma2. Coupling X=mu1+sigma1*Z and Y=mu2+sigma2*Z attains this bound.
This proves the W2 formula independently of the general quantile theorem.
With independent Z and Z', the cost instead contains sigma1^2+sigma2^2,
so independent pairing usually overestimates the optimal transport cost.
For W1, the monotone-quantile theorem justifies the same shared-Z coupling.

W1 is the expectation of |m+sZ|, where m is the absolute mean difference
and s is the absolute standard-deviation difference. Split the integral at
Z=-m/s. Integrating z*phi(z) gives -phi(z), producing the folded-normal
formula in the notebook. Use W1=m when s=0 to avoid division by zero.

For N(0,1) versus N(0,4), the answer is W2=1, not 3. The latter mistakenly
subtracts variances. W1=sqrt(2/pi), approximately 0.79788456.
For N(0,1) versus N(2,4), W2=sqrt(5) and W1 is approximately 2.01698141.

An analytic formula is still useful computationally when its assumptions
apply. In arbitrary data examples there is generally no Gaussian formula
unless we deliberately approximate the laws by Gaussians. Matching only
means and covariances can miss important differences such as multimodality.

2. Distance, map, plan, and interpolation

The Wasserstein distance is a scalar. An optimizer is a coupling/plan;
there may be several optimizers. A deterministic map is a special plan.
For a nondegenerate 1D Gaussian source, the optimal affine map is
T(x)=mu2+(sigma2/sigma1)*(x-mu1). In the discrete hand example, source mass
at x=0 splits, so a deterministic map cannot represent that plan.

The optional interpolation figure contrasts moving quantiles with mixing
densities. In Gaussian displacement interpolation, the standard deviation
interpolates linearly. The variance is its square, not a linear average.
Learning a transport through a neural ODE does not establish OT optimality.

3. What the numerical solvers return

SciPy wasserstein_distance computes 1D W1. It sorts supports and handles
unequal weights and sizes. Passing histogram heights as sample values is
usually a mistake: the heights are weights at bin locations.
Our empirical W2 shortcut only handles equal sample sizes with uniform mass.
The general LP works with arbitrary finite supports and weights in any
Euclidean dimension, but it is intended for very small examples.

The exact_ot helper normalizes each supplied weight vector to sum to one.
It therefore compares probability measures and discards differences in total
mass. If total mass matters, retain a separate total-mass statistic or study
unbalanced OT. Do not silently interpret normalized brightness as total energy.

Sinkhorn uses C_ij=distance^2 in this example and the entropy convention
sum pi_ij*(log pi_ij - 1). It reports the transport part, the full regularized
objective, and the debiased Sinkhorn divergence separately. The full objective
can be negative because of the entropy term. It is not a metric. Regularized
self-transport can have positive travel cost. S_epsilon subtracts self terms
using full objectives and the same convention; it is zero on self-comparison.
No square roots of negative regularized objectives are taken.

The helper works in log space, removes no positive mass, and raises an error
if its marginal tolerance is not met. Very small epsilon may require more
iterations or specialized stabilized algorithms. The examples use epsilon
0.2, 0.7, and 2.0, measured in squared coordinate units.
POT offers specialized exact and regularized OT routines for further study.
In POT, emd2 denotes optimal transport cost, not automatically W2 squared:
the ground cost matrix determines the meaning. sinkhorn2 returns the
transport cost of the regularized plan, not the full entropy objective.

4. Images and time series

Image intensity histograms ignore where pixels occur. Spatial mass OT uses
pixel coordinates and preserves their geometry but discards total brightness
after normalization. Normalizing coordinates by image size changes units.
Color and learned-feature distributions are further modeling choices.
For FID, each entire image contributes a learned feature vector; this is a
different sample space from the spatial mass in a single grayscale image.

The time-series example has exactly equal amplitude distributions.
Time-amplitude OT allows unrestricted pairings and does not enforce a
monotone time warp. Choose coordinate units before selecting a time weight.
Windows capture local dependence but do not identify an entire path law.
Whole-path vectors preserve the chosen time grid and can be high-dimensional.
A single time series supplies dependent windows, not independent whole paths.
Do not use an ordinary iid permutation test without considering that dependence.

The two noiseless paths also have identical signature levels 1 and 2 after
adding normalized time. Their S_tx values both equal 1/2. Their third-level
S_ttx values are 7/54 and 13/54, respectively. This deliberate collision is
a useful example of the information lost through signature truncation.
The actual Chen updates through level 3 appear in ot_tools.py. A scalar
ordinary signature without time augmentation depends only on total increment.
These are ordinary signatures. No Volterra-signature implementation is claimed.

5. MMD and statistical interpretation

MMD is a distance between kernel mean embeddings. Its square has the simple
three-expectation formula. A Gaussian kernel is characteristic on Euclidean
space, but this only identifies distributions of the features fed into it.
An inadequate representation remains inadequate with either MMD or OT.

The code reports the biased empirical MMD squared, including self-diagonal
terms. The unbiased version excludes these diagonals and can be negative.
Bandwidth changes sensitivity. As bandwidth tends to infinity, the Gaussian
kernel tends to the constant 1 and empirical MMD squared tends to zero.
For distinct finite points and very small bandwidth, the biased self-diagonal
terms dominate, roughly giving 1/n+1/m. This is not evidence of a perfect fit.

Wasserstein and MMD values are not on a common scale. A small descriptive
score is not a p-value or proof that a generator has learned the correct law.
Use matched sample sizes, real/real baselines, repeated seeds, and diagnostics
for the aspects of the data that matter. The shaded sampling plot shows
quantiles of repeated estimates, not uncertainty intervals from one dataset.

EXERCISE ANSWERS

1. N(0,1) vs N(0,9): standard deviations 1 and 3. W2=2 and
   W1=2*sqrt(2/pi), approximately 1.59577.
2. Translation by (2,1): spatial W1=W2=sqrt(5) pixels, provided the patch
   stays inside the grid. The intensity histogram is unchanged, so its W1=0.
3. Scaling all intensities by 10 leaves normalized spatial mass unchanged.
   Relative to the unscaled image, the intensity distribution does change.
4. Permutation preserves the empirical amplitude law, hence amplitude OT
   and MMD are zero. Windows and signatures may change but need not always
   distinguish permutations. Whole-path RMS can reveal changes on a fixed grid.
5. Larger epsilon generally spreads the plan and increases its travel cost
   while changing the entropy objective. The marginal error should still
   satisfy the chosen tolerance. Larger MMD bandwidth need not monotonically
   increase or decrease the score, and very large bandwidth suppresses it.

REFERENCES

Full links appear in the notebook and on the slide reference page. Primary
sources include Peyre and Cuturi (Computational Optimal Transport), Cuturi
(2013), Feydy et al. (2019), Gretton et al. (2012), Chevyrev and Kormilitzin
(signature primer), Arjovsky et al. (WGAN), Heusel et al. (FID), and the SciPy
and POT API documentation. All empirical numbers and plots in this lesson
come from the supplied synthetic examples.
