# Level 2 Experiment Specification: Density Gap (Interleaved Spirals)

## 1. Objective

Implement **Level 2: The Density Gap (Interleaved Spirals)** to evaluate the **Fisher Information Stabilization / Temperature Softening** mechanism.

The experiment must test the following claim:

> When the classifier is trained only on a dense central region and the unlabeled pool lies entirely in unsupported outer spiral tails, the classifier may become highly overconfident on out-of-support samples. This causes the softmax/Fisher Jacobian to collapse, weakening the raw Fisher-weighted IPM signal. Temperature softening should restore the Jacobian signal, allow the critic to detect the geometric shift, and produce a substantially better bound/proxy for the true risk gap.

The experiment is specifically designed to separate:

1. **True predictive failure on unsupported regions**.
2. **Jacobian/Fisher collapse caused by overconfidence**.
3. **Recovery of the critic signal through temperature softening**.

---

## 2. Core Experimental Hypothesis

For classifier logits \(z(x)\),

\[
p_T(x)=\operatorname{softmax}\left(\frac{z(x)}{T}\right).
\]

Compare:

- **Raw critic:** \(T=1\)
- **Softened critic:** \(T>1\), default \(T=5\)

The classifier itself must remain unchanged. Temperature softening is used **only inside the Fisher/IPM proxy or critic input/weighting**.

For binary classification, the softmax Jacobian contains the factor

\[
p(1-p).
\]

When the classifier becomes overconfident,

\[
p\to 0 \quad \text{or} \quad p\to 1,
\]

then

\[
p(1-p)\to 0,
\]

so the Fisher/Jacobian signal collapses.

The desired qualitative behavior is:

\[
\text{support gap}\uparrow
\Rightarrow
\text{true risk gap}\uparrow,
\]

while for \(T=1\),

\[
\text{Jacobian magnitude}\downarrow,
\qquad
\text{raw IPM signal becomes too small},
\]

but for \(T>1\),

\[
\text{Jacobian magnitude recovers},
\qquad
\text{softened IPM tracks the risk gap better}.
\]

---

## 3. Dataset: 2D Interleaved Spirals

Generate a synthetic binary classification population in 2D.

For class \(y\in\{0,1\}\), define

\[
\theta(r,y)=4\pi r+\pi y,
\]

and

\[
x(r,y)
=
r
\begin{bmatrix}
\cos\theta(r,y)\\
\sin\theta(r,y)
\end{bmatrix}.
\]

Optionally add a **small bounded perturbation**, but do not use large Gaussian noise that causes labeled and unlabeled supports to overlap.

### Required support split

The labeled population \(S\) must be restricted to the center:

\[
r \in [0.02, 0.25].
\]

The unlabeled population \(X\) must contain only outer tails:

\[
r \in [0.60, 1.00].
\]

Therefore there is an explicit empty radial gap:

\[
0.25 < r < 0.60.
\]

This is important: the labeled support should be effectively zero in the unlabeled region.

### Suggested population sizes

Use procedurally generated populations:

- Labeled/source population for evaluation: \(10^6\) samples if feasible.
- Unlabeled/target population for evaluation: \(10^6\) samples.
- A smaller labeled training subset is sampled from the center population.
- The hidden labels of the unlabeled population are retained only for evaluation/oracle use.

Use chunked computation if GPU/CPU memory is limited.

---

## 4. Classifier

Use a nonlinear classifier because the spiral boundary is not linear.

Recommended architecture:

```text
Input(2)
→ Linear(2, 64)
→ ReLU
→ Linear(64, 64)
→ ReLU
→ Linear(64, 2)
```

Train the classifier **only on labeled center samples**.

Do not train on outer-tail samples before they are selected by active learning.

The experiment should explicitly produce an early-stage regime in which:

- center training accuracy is high;
- outer-tail accuracy/risk is poor;
- outer-tail predictive confidence is nevertheless high.

This overconfident extrapolation is the failure mode the experiment is designed to study.

---

## 5. Active Learning Protocol

Use the same general active learning structure as Level 1 unless otherwise specified.

### Initial labeled set

Start with:

\[
|S_0|=10.
\]

The 10 samples must come exclusively from the dense center region.

Prefer a class-balanced initialization when possible:

- 5 samples from class 0
- 5 samples from class 1

This reduces failures caused only by accidental class imbalance.

### Episodes

Run:

\[
20
\]

active-learning episodes.

At each episode, select:

\[
10
\]

new samples from the current outer-tail unlabeled pool.

Therefore:

\[
|S_t|=10+10t,
\]

and after 20 acquisitions:

\[
|S_{20}|=210.
\]

Record episode 0 before the first acquisition, giving 21 checkpoints in total.

### Per-episode order

For each episode \(t\):

1. Train or retrain the classifier on current labeled set \(S_t\).
2. Compute classifier logits/probabilities on:
   - labeled set;
   - unlabeled pool;
   - fixed evaluation populations.
3. Train/evaluate the **raw critic** using \(T=1\).
4. Train/evaluate the **softened critic** using \(T=T_{\text{soft}}\), default \(5\).
5. Compute all metrics listed below.
6. Use the designated acquisition critic to score the unlabeled pool.
7. Select the top 10 samples.
8. Query their hidden labels.
9. Move them from unlabeled pool to labeled set.
10. Continue to next episode.

Unless a separate ablation is desired, use the **softened critic** for the main active-learning acquisition rule.

---

## 6. Temperature Softening

Let classifier logits be

\[
z(x).
\]

Raw probabilities:

\[
p_{\text{raw}}(x)
=
\operatorname{softmax}(z(x)).
\]

Softened probabilities:

\[
p_{\text{soft}}(x)
=
\operatorname{softmax}
\left(
\frac{z(x)}{T_{\text{soft}}}
\right).
\]

Default:

\[
T_{\text{soft}}=5.
\]

Important implementation rule:

> Temperature must not change the classifier decision rule used for accuracy or true risk evaluation. It is only used for the Fisher/Jacobian/IPM stabilization mechanism.

Thus:

- classifier accuracy uses raw logits / raw probabilities;
- true risk uses the actual classifier output;
- raw IPM uses \(T=1\);
- softened IPM uses \(T>1\).

---

## 7. Fisher / Jacobian Diagnostics

For each sample, define the softmax Jacobian

\[
J(p)
=
\operatorname{diag}(p)-pp^\top.
\]

For binary classification this is directly controlled by \(p(1-p)\).

Record at least the following diagnostics on the unlabeled population:

### Mean confidence

\[
C_t
=
\mathbb E_{X_t}
\left[
\max_c p_c(x)
\right].
\]

### Mean raw Jacobian norm

\[
J^{\text{raw}}_t
=
\mathbb E_{X_t}
\left[
\|J(p_{\text{raw}}(x))\|_F
\right].
\]

### Mean softened Jacobian norm

\[
J^{\text{soft}}_t
=
\mathbb E_{X_t}
\left[
\|J(p_{\text{soft}}(x))\|_F
\right].
\]

The intended failure signature is:

\[
C_t \text{ high},
\qquad
J^{\text{raw}}_t \text{ very small},
\]

while

\[
J^{\text{soft}}_t
>
J^{\text{raw}}_t.
\]

---

## 8. IPM Critics

Use two critic evaluations with identical architecture and optimization settings.

### Raw critic

Uses classifier probabilities or Fisher weighting with

\[
T=1.
\]

Denote its value:

\[
\operatorname{IPM}^{\text{raw}}_t.
\]

### Softened critic

Uses the same critic structure, but probabilities/Fisher terms are computed with

\[
T=T_{\text{soft}}>1.
\]

Denote:

\[
\operatorname{IPM}^{\text{soft}}_t.
\]

All other factors must be controlled:

- same network architecture;
- same hidden width;
- same spectral normalization;
- same optimizer;
- same number of steps;
- same batch size;
- same early-stopping rule;
- ideally paired/random-seed-controlled initialization.

The only intended difference should be temperature softening.

---

## 9. True Risk and True Risk Gap

Use a fixed evaluation population large enough to approximate population expectations accurately.

Let \(h_t\) be the classifier at episode \(t\).

Define:

\[
R_S(h_t)
=
\mathbb E_{(x,y)\sim S}
[\ell(h_t(x),y)],
\]

and

\[
R_X(h_t)
=
\mathbb E_{(x,y)\sim X}
[\ell(h_t(x),y)].
\]

Then define:

\[
\boxed{
\Delta R_t
=
R_X(h_t)-R_S(h_t)
}
\]

as the **True Risk Gap**.

Use the same loss as assumed by the theorem. If the theorem uses Brier loss, use Brier loss here.

Do not call a small minibatch empirical difference “True Risk Gap”. Compute this quantity on large fixed procedural evaluation populations.

---

## 10. Bound Evaluation

If the theorem has the form

\[
\Delta R_t
\le
\operatorname{IPM}_t
+
\epsilon_t,
\]

record the bound margin:

\[
B_t
=
\operatorname{IPM}_t
+
\epsilon_t
-
\Delta R_t.
\]

Compute this separately for raw and softened IPM:

\[
B_t^{\text{raw}}
=
\operatorname{IPM}^{\text{raw}}_t
+\epsilon_t
-\Delta R_t,
\]

\[
B_t^{\text{soft}}
=
\operatorname{IPM}^{\text{soft}}_t
+\epsilon_t
-\Delta R_t.
\]

If the theoretical experiment sets capacity slack to zero, use:

\[
\epsilon_t=0.
\]

Then:

\[
B_t^{\text{raw}}
=
\operatorname{IPM}^{\text{raw}}_t-\Delta R_t,
\]

\[
B_t^{\text{soft}}
=
\operatorname{IPM}^{\text{soft}}_t-\Delta R_t.
\]

Do not assume “perfectly bounds” means equality unless the theorem explicitly proves equality/tightness.

The implementation must use the theorem's exact inequality.

---

## 11. Required Metrics Per Episode

Record one row per episode with at least:

| Metric | Meaning |
|---|---|
| `episode` | 0 to 20 |
| `n_labeled` | 10 to 210 |
| `accuracy_center` | accuracy on center/source evaluation population |
| `accuracy_outer` | accuracy on outer-tail evaluation population |
| `risk_center` | \(R_S(h_t)\) |
| `risk_outer` | \(R_X(h_t)\) |
| `true_risk_gap` | \(R_X-R_S\) |
| `mean_confidence_outer` | mean max raw probability on outer tails |
| `jacobian_raw` | mean raw Fisher/Jacobian norm |
| `jacobian_soft` | mean softened Fisher/Jacobian norm |
| `ipm_raw` | critic/IPM value at \(T=1\) |
| `ipm_soft` | critic/IPM value at \(T>1\) |
| `bound_margin_raw` | raw IPM bound margin |
| `bound_margin_soft` | softened IPM bound margin |

Optional but useful:

- selected-sample radial distance;
- selected-sample class balance;
- selected-sample critic score;
- unlabeled-pool size;
- classifier entropy;
- expected calibration error.

---

## 12. Main Plots

Produce at least the following figures.

### Plot A — Geometry

Scatter plot of:

- labeled center samples;
- unlabeled outer-tail samples;
- class labels shown only for analysis;
- optional classifier decision regions.

This figure must visibly show the support gap.

### Plot B — Active Learning Accuracy

x-axis:

\[
n_{\text{labeled}}
\]

y-axis:

- outer-tail accuracy;
- optionally center accuracy.

Expected trend:

\[
\text{outer accuracy increases as AL proceeds}.
\]

### Plot C — True Risk Gap vs IPMs

Plot on the same axes:

\[
\Delta R_t,
\qquad
\operatorname{IPM}^{\text{raw}}_t,
\qquad
\operatorname{IPM}^{\text{soft}}_t.
\]

This is the primary Level 2 result.

Expected qualitative pattern at early episodes:

\[
\operatorname{IPM}^{\text{raw}}
\ll
\Delta R,
\]

while

\[
\operatorname{IPM}^{\text{soft}}
\]

should track or bound the risk substantially better.

### Plot D — Jacobian Collapse

Plot:

\[
J_t^{\text{raw}}
\quad\text{and}\quad
J_t^{\text{soft}}
\]

against episode or labeled-set size.

Also plot outer-tail confidence if useful.

The desired early-stage signature is:

\[
\text{confidence high},
\qquad
J^{\text{raw}}\approx0,
\qquad
J^{\text{soft}}>J^{\text{raw}}.
\]

### Plot E — Bound Margin

Plot:

\[
B_t^{\text{raw}}
\quad\text{and}\quad
B_t^{\text{soft}}.
\]

If the theorem predicts a valid upper bound, values should satisfy the theorem's required sign.

---

## 13. Density-Gap Severity Ablation

In addition to the 20-episode active-learning run, perform a controlled support-gap sweep.

Keep labeled support fixed:

\[
r\le0.25.
\]

Change the minimum target radius:

\[
r_{\min}^{X}
\in
\{0.35,0.45,0.55,0.65,0.75\}.
\]

For each setting, measure:

- outer true risk;
- true risk gap;
- raw Jacobian norm;
- softened Jacobian norm;
- raw IPM;
- softened IPM;
- raw/soft bound margins.

This ablation directly tests whether increasing geometric support separation causes:

\[
\Delta R\uparrow
\]

while raw Fisher signal collapses and softened signal remains informative.

---

## 14. Temperature Ablation

Test several temperatures:

\[
T\in\{1,2,3,5,10\}.
\]

For each temperature, measure:

- Jacobian norm;
- IPM;
- bound margin;
- acquisition performance.

The experiment should identify whether there is an intermediate temperature that stabilizes the Fisher signal without making the predictions nearly uniform.

Do not assume \(T=5\) is optimal; use it as the default main setting and report the ablation separately.

---

## 15. Random Seeds

Because the initial labeled set contains only 10 points, run multiple independent seeds.

Recommended:

\[
10\text{--}20 \text{ seeds}.
\]

For all primary curves report:

- mean;
- standard deviation or standard error;
- optionally 95% confidence intervals.

Use paired seeds when comparing \(T=1\) and \(T>1\).

---

## 16. Acceptance Criteria

The Level 2 experiment is considered successful only if the following behaviors are observed consistently across seeds.

### A. Support gap exists

Initial labeled data contain only center samples and unlabeled data contain only outer-tail samples.

### B. Early-stage overconfidence exists

At early episodes:

- center accuracy is high;
- outer risk is substantially higher;
- outer predictive confidence is still high.

### C. Raw Fisher/Jacobian collapse occurs

At early episodes:

\[
J^{\text{raw}}
\]

is substantially smaller than the softened version.

### D. Raw IPM loses signal

At early episodes, raw IPM substantially under-represents the true risk gap or is materially less informative than the softened IPM.

### E. Temperature softening restores signal

The softened IPM should:

- increase when outer-support mismatch/risk increases;
- correlate better with true risk gap;
- satisfy the theorem's bound more reliably/tightly, if the theorem applies.

### F. Active learning improves target performance

As outer-tail samples are acquired:

\[
\text{outer accuracy}\uparrow,
\qquad
\Delta R\downarrow.
\]

---

## 17. Important Controls

The following controls are mandatory.

1. **Same classifier** for raw and softened critic comparisons.
2. **Same dataset and same seeds** for paired comparisons.
3. Temperature softening must affect only the Fisher/IPM mechanism.
4. Do not change labels under softening.
5. Do not evaluate “true risk” on only the 10–210 labeled samples.
6. Use large fixed evaluation populations.
7. Do not claim a theoretical bound unless the exact theorem inequality is checked.
8. Do not interpret spectral normalization alone as Fisher weighting unless the theorem's Fisher metric is explicitly implemented elsewhere.
9. Keep critic architecture/optimizer fixed across \(T=1\) and \(T>1\).
10. Save all raw per-seed per-episode metrics before aggregation.

---

## 18. Output Files

The implementation should produce at least:

```text
results/
├── level2_metrics_all_seeds.csv
├── level2_metrics_summary.csv
├── level2_density_gap_ablation.csv
├── level2_temperature_ablation.csv
├── level2_geometry.png
├── level2_accuracy.png
├── level2_risk_ipm.png
├── level2_jacobian.png
├── level2_bound_margin.png
└── config.json
```

The CSV should contain one row per seed × episode for the main experiment.

---

## 19. Minimal Main Comparison

The central comparison that must not be omitted is:

\[
\boxed{
\Delta R_t
\quad\text{vs}\quad
\operatorname{IPM}^{\text{raw}}_t
\quad\text{vs}\quad
\operatorname{IPM}^{\text{soft}}_t
}
\]

together with:

\[
\boxed{
J_t^{\text{raw}}
\quad\text{vs}\quad
J_t^{\text{soft}}.
}
\]

The intended causal story is:

\[
\text{out-of-support extrapolation}
\rightarrow
\text{overconfidence}
\rightarrow
\text{Jacobian collapse}
\rightarrow
\text{raw Fisher proxy loses signal},
\]

while:

\[
\text{temperature softening}
\rightarrow
\text{non-degenerate Fisher/Jacobian}
\rightarrow
\text{critic recovers geometric shift}
\rightarrow
\text{IPM better tracks/bounds true risk}.
\]

---

## 20. Implementation Priority

Implement in this order:

1. Spiral generator and strict support split.
2. Classifier and large-population risk evaluation.
3. Raw Jacobian diagnostics.
4. Temperature-softened Jacobian diagnostics.
5. Raw and softened critics.
6. Main 20-episode active-learning loop.
7. Main metrics and plots.
8. Density-gap ablation.
9. Temperature ablation.
10. Multi-seed aggregation.

Do not optimize or complicate the implementation before the required qualitative failure mode is verified on a single seed.
