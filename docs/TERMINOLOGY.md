# Terminology — Precise Definitions

Reference for the poster, the talk, and Q&A. Section 4 covers the places where a term of yours differs
from the standard usage; those are the questions most likely to catch you out.

---

## 1. The terms carrying your main claim

### Group action (setup)
The symmetric group $S_n$ is the group of all permutations (bijections) of $\{1,\dots,n\}$. Here $n = 1500$,
the number of payload bytes. A permutation $\sigma \in S_{1500}$ **acts** on a byte vector
$x \in \{0,\dots,255\}^{1500}$ by relabelling positions:

$$(x \circ \sigma)_i = x_{\sigma(i)}$$

Your swap attack applies a product of about 60 transpositions, which is one particular element of
$S_{1500}$. Your result holds for **every** $\sigma$, so it is strictly stronger than what your attack needs.

### Invariance
A quantity $F$ is **invariant** under a group action if the action does not change it:

$$F(x \circ \sigma) = F(x) \quad \text{for all } \sigma.$$

Applying the transformation makes no difference to the output.

### Equivariance
A map $f$ is **equivariant** if the action passes *through* it rather than vanishing:

$$f(x \circ \sigma) = f(x) \circ \sigma.$$

The output transforms in the same way the input did. **This distinction is the heart of your argument, and
it is the single easiest thing to state incorrectly** — see §4.1.

### Multiset
A collection allowing repeated elements, with multiplicity but without order. The **byte multiset** of a
packet records how many bytes take each value $0,\dots,255$, discarding position. A permutation preserves
the byte multiset *exactly* — this is precisely what makes it a permutation.

### Permutation invariance (your sense)
The property that a feature computed from the packet is unchanged when byte positions are permuted.
Your specific instance: the count of foreground pixels after binarization.

**The argument in full.** Binarization $b : \{0,\dots,255\} \to \{0,1\}$ is a fixed pointwise threshold,
applied componentwise. Then

$$\big(b(x \circ \sigma)\big)_i = b(x_{\sigma(i)}) = \big(b(x)\big)_{\sigma(i)} = \big(b(x) \circ \sigma\big)_i,$$

so $b$ is **equivariant**. Summation is symmetric — reindexing by a bijection does not change a finite sum —
so composing an equivariant map with a symmetric functional yields an **invariant**:

$$\sum_i \big(b(x \circ \sigma)\big)_i = \sum_i \big(b(x)\big)_{\sigma(i)} = \sum_j \big(b(x)\big)_j.$$

In one sentence: *binarization is equivariant, summation is symmetric, therefore the foreground count is
permutation-invariant.*

### "Invariance structure" — your own umbrella term
**Not standard terminology.** Say so if asked. You are using it to mean the pairing of two things:

1. the group of transformations an attack's perturbations belong to (permutations, for swaps; arbitrary
   value changes, for noise); and
2. which of the detector's features are invariant under that group.

Detectability then depends on the *relationship* between the two, not on either alone. The nearest
established framework is the **GENEO** literature (Group Equivariant Non-Expansive Operators; Frosini,
Bergomi, Quercioli et al.), which exists precisely because persistence diagrams are often *too* invariant to
discriminate well. Cite it if you use the phrase.

### Attenuation (vs. blindness)
**Attenuated** = sensitivity reduced because only some pathways from data to feature can respond.
**Blind / invisible** = zero sensitivity. Your result establishes attenuation on the count pathway only, and
your own data rules out blindness: the highest-capture condition in the study (7.92%) was swap-only.
Never say "invisible," "evades," or "defeats" without qualification.

---

## 2. TDA machinery

### Filtration
A nested family of spaces indexed by a parameter $t$, with $K_s \subseteq K_t$ whenever $s \leq t$. Given a
filtering function $f$ on pixels, the sublevel sets $\{f \leq t\}$ form a filtration as $t$ increases.
Topological features are born and die as $t$ grows.

### Height filtration
Filtering function $f(p) = \langle p, d \rangle$ — the projection of pixel position $p$ onto a chosen
direction $d$. Pixels enter in order of "height" along $d$. **Depends only on position.**

### Radial filtration
Filtering function $f(p) = \lVert p - c \rVert$ — distance from a chosen center $c$. Pixels enter outward
from $c$. **Also depends only on position.**

Your pipeline uses five: two Height (directions $[0,1]$, $[1,0]$) and three Radial (centers $[0,50]$,
$[0,25]$, $[30,0]$). All five are position-dependent, which is exactly why swap attacks register at all.

### Cubical persistence
Persistent homology computed on a **cubical complex** — a grid of pixels — rather than on a simplicial
complex built from a point cloud. The natural choice for image-shaped data, and what your pipeline uses.

### Persistence diagram
A multiset of points $(b, d)$ in the plane, one per topological feature, recording the parameter value at
which the feature is **born** and at which it **dies**. Points far from the diagonal are long-lived
(robust) features; points near it are short-lived (noise-like).

### Homology dimensions $H_0$, $H_1$
$H_0$ counts **connected components**; $H_1$ counts **loops** (independent 1-dimensional cycles). Your
feature vector uses both, giving $5 \text{ filtrations} \times 6 \text{ metrics} \times 2 \text{ dimensions}
= 60$ features.

### Wasserstein and bottleneck distances
Metrics between persistence diagrams, defined by optimal matching of points across the two diagrams.
Bottleneck takes the largest matched displacement; $p$-Wasserstein takes a $p$-norm over all of them.
The **stability theorem** (Cohen-Steiner, Edelsbrunner & Harer, 2007) bounds these distances by the distance
between the underlying datasets — which is what licenses using topological change as a detection signal.

---

## 3. Operational terms specific to this project

### Detectability (your operational definition)
**Not** an abstract property of an attack. In your work it is a measured property of the *(attack, detector)
pair*:

$$\text{capture \%} = \frac{\text{poisoned samples landing in Red clusters}}{\text{total poisoned samples}} \times 100.$$

### Cluster colors
After clustering the 60-dim features, each cluster is labelled by its poison fraction:
**Green** = 0% poisoned (retained as clean), **Red** = 100% poisoned (removed as detected),
**Pink** = >80% but <100%, **Yellow** = mixed (unresolved).

**The criterion is strict.** Only perfectly pure clusters count as detection — a 99%-poison cluster
contributes nothing to capture. This zero-tolerance rule is a major reason the absolute numbers are low, and
it is worth stating when someone asks why 6% rather than 60%.

### Surrogate classifier
A stand-in NIDS model that the attack optimizes against. Required because the attacks cited by the source
paper are *model-relative* (they maximize a classifier's error), while the detection pipeline itself is
unsupervised — and the source paper never specifies which classifier it attacked.

### Guidance
Selecting perturbations by search against the surrogate's loss, rather than at random. "Guided swaps" =
a genetic search over swap sets maximizing the surrogate's benign-class probability; "random swaps" = the
same number of swaps chosen uniformly.

### Topological distortion
$W_p$ distance between the persistence diagram of clean data and that of poisoned data — Ferrara's
currency for measuring an attack's topological effect. **You have not yet computed this**; your results are
in capture-rate terms. Do not conflate the two.

---

## 4. Precision traps

### 4.1 Equivariance is not invariance
Your poster equation $b(x \circ \sigma) = b(x) \circ \sigma$ states **equivariance of binarization**.
The **invariance** is of the *count*, $\sum_i b(x_i)$, and follows from equivariance plus the symmetry of
summation. If you call the displayed equation "the invariance," a careful listener will correct you.
Correct phrasing: *"binarization is equivariant; the count it feeds is therefore invariant."*

### 4.2 Two different meanings of "permutation invariance"
In the TDA literature this phrase usually means a **persistence diagram is a multiset**, so its points carry
no order. That is a statement about the *diagram*. Yours is about permuting **source pixels** — a different
claim entirely. Distinguish them explicitly, since anyone who knows TDA will hear the standard meaning
first.

### 4.3 The invariance is elementary; the application is not
The mathematics is folklore — persistence diagrams are invariant under homeomorphisms of the domain, and
your result is a discrete special case. Claim novelty for the **observation that this creates a concrete
attack blind spot in this security pipeline**, which is unstated in that literature. Overclaiming
mathematical originality is the main risk at a mathematics venue.

### 4.4 Fixed vs. adaptive thresholds
The argument requires $b$ to be a **fixed** pointwise threshold (yours is $0.4$). An adaptive threshold
computed from permutation-invariant statistics (a mean, a quantile) would preserve the result; one depending
on pixel position would not. Worth a sentence if pressed.

### 4.5 Attenuation is channel-specific
Say "attenuated **on the foreground-count channel**," not "attenuated" alone. The position-dependent
filtrations remain fully responsive, and guided search demonstrably exploits them.

---

## 5. Pipeline mechanics — what is actually computed

Verified against giotto-tda documentation and source, not from memory.

### The algorithm: `CubicalPersistence` (GUDHI backend)
Persistent homology is computed on **filtered cubical complexes**, not point clouds. giotto-tda passes the
image to GUDHI as `top_dimensional_cells` and calls
`cubical_complex.persistence(homology_coeff_field=coeff, min_persistence=0)`. GUDHI is the C++ backend
(reference: P. Dlotko, "Cubical complex", GUDHI manual). Defaults in use: `coeff=2` (homology over
$\mathbb{F}_2$) and `homology_dimensions=(0,1)`.

### Step by step, for one packet
1. **Reshape.** 1500 bytes $\to$ a $30 \times 50$ grayscale image, one byte per pixel.
2. **Binarize.** `Binarizer(threshold=0.4)`. **The threshold is a *fraction of the maximum pixel value*
   `max_value_`**, computed over all pixels in all images during `fit`. With byte data `max_value_` is
   almost certainly 255, so the effective cutoff is $\approx 102$.
3. **Filtration** (converts binary back to grayscale). `HeightFiltration` assigns each *activated* pixel its
   distance from the hyperplane defined by a direction vector; `RadialFiltration` assigns distance from a
   chosen center. Deactivated pixels receive the maximum value. Five are used: directions $[0,1]$, $[1,0]$;
   centers $[0,50]$, $[0,25]$, $[30,0]$.
4. **Persistence.** GUDHI builds the sublevel-set filtration: as $t$ increases, cells with filtration value
   $\leq t$ enter the complex. Components appear and merge ($H_0$); loops open and close ($H_1$). Each
   feature's $(\text{birth}, \text{death})$ is recorded. The computation is the standard persistence
   algorithm — boundary-matrix reduction, with union-find handling $H_0$ efficiently.
5. **Vectorize.** `Scaler` normalizes the diagram; then `PersistenceEntropy` plus five `Amplitude` metrics
   (bottleneck, Wasserstein, landscape, Betti, heat) each emit one value per homology dimension:
   6 transformers $\times$ 2 dimensions = 12 features per filtration, $\times$ 5 filtrations = **60**.

### Two distinct persistence computations in this repo — do not conflate
- **`CubicalPersistence`** on individual packet images — the feature extractor. **This is what the
  invariance proof concerns.**
- **`VietorisRipsPersistence`** in `iterative_filter.py` — one diagram over the whole residual treated as a
  point cloud in 60-dim feature space, used only for descriptive Wasserstein convergence tracking. The proof
  says nothing about it.

### Why this matters for the proof
The binarization threshold is **data-dependent** ($0.4 \times$ fitted `max_value_`), so strictly $b$ is not
a fixed constant map. The proof survives because `max_value_` is a maximum over pixel values, and the
maximum of a multiset is itself permutation-invariant — permuting bytes within packets cannot change it.
So the threshold is unchanged, $b$ remains equivariant, and the count invariance holds. **State this step
explicitly**; "your threshold is data-dependent, so your map isn't fixed" is a sharp question someone could
reasonably ask.

### Methodological ancestor worth citing
Both filtration classes cite A. Garin and G. Tauzin, "A topological reading lesson: Classification of MNIST
using TDA," ICMLA 2020 (arXiv:1910.08345) — the paper that established this exact
binarize $\to$ filtration $\to$ cubical-persistence $\to$ amplitude pattern. It is the direct methodological
ancestor of the Monkam pipeline and belongs in Related Work.

### Feature-count arithmetic
Monkam's Algorithm 1 as printed: 2 directions + 3 centers = 5 filtrations; entropy + 5 amplitude metrics =
6 transformers; 2 homology dimensions. $5 \times 6 \times 2 = 60$ — **exactly what this project produces**.
Their claimed 72 would require six filtrations. Verify against `tda_pipeline.py`; if it holds, this is a
second internal inconsistency in the source paper (alongside the binarizer 0.4-vs-0.3 discrepancy), and the
project's "divergence" is actually faithfulness to their own pseudocode.
