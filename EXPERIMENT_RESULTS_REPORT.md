# fMRI Responsiveness Classification: Experiment Results Report

**Analysis date:** November 3, 2026  
**Scope:** All saved experiment results under `plots/GRU`, `plots/LR`, `plots/MLP`, and `plots/motion_classifier`.

## Executive summary

The experiments show that fMRI contains a reproducible signal for distinguishing responsive from non-responsive segments, but performance depends strongly on the held-out unit and temporal segment length.

- **GRU is the strongest model under leave-one-participant-out (LOPO) validation.** In the directly comparable canonical 40-TR results, hidden 64, dropout 0.2, and WD 0.001 ranks first by participant-weighted balanced accuracy at **0.804 ± 0.033** across five seeds. Hidden 64/WD 0.01 (**0.802 ± 0.021**) and hidden 64/dropout 0.4/WD 0.001 (**0.801 ± 0.022**) are close behind.
- **A compact MLP is the strongest simple, well-replicated alternative.** The 16-unit MLP reached mean LOPO balanced accuracy **0.717 ± 0.025** at 60 TRs across five seeds. Logistic regression reached **0.702** at 40 TRs and **0.701** at 60 TRs in its single deterministic run.
- **Motion alone is not a sufficient explanation for the fMRI results.** Motion logistic regression peaked at only **0.594** LOPO balanced accuracy and overwhelmingly favored the responsive class (responsive recall 0.972 versus non-responsive recall 0.216 at 60 TRs).
- **Longer segments generally help.** Performance rises substantially from 10 to 30–40 TRs. Gains beyond 40 TRs are model-dependent, and the number of available examples falls from 11,928 at 10 TRs to 1,488 at 60 TRs.
- **LORO results are consistently higher than LOPO results.** Because leave-one-run-out (LORO) training can include other runs from the test participant, LORO measures across-run generalization within a partly familiar participant population. LOPO is the appropriate primary estimate for generalization to unseen participants.
- **The main limitation is experimental unevenness outside the principal GRU sweeps.** LR and motion have one run per condition, MLP has five seeds for each of two configurations, and the hidden-16 GRU variants still have one seed. Seven canonical 40-TR GRU configurations now have five LOPO seeds, including hidden 32/dropout 0.4. LORO replication remains less complete for some variants.

## 1. Experimental setup represented in the artifacts

### 1.1 Classification target and segmentation

All main experiments use the same responsiveness labeling family:

- Responsive intervals: `W=30`, `S=1`, `N_min=8`, `G_max=4`
- Non-responsive intervals: `T_min=20`
- Non-overlapping model segments with segment length equal to stride: 10, 20, 30, 40, 50, or 60 TRs
- All-ROI fMRI input: 42 ROIs
- Binary target: responsive versus non-responsive

The number of segments decreases sharply as the segment duration increases:

| Segment length | Number of segments | Relative to 10 TR |
|---:|---:|---:|
| 10 TR | 11,928 | 100.0% |
| 20 TR | 5,536 | 46.4% |
| 30 TR | 3,478 | 29.2% |
| 40 TR | 2,474 | 20.7% |
| 50 TR | 1,872 | 15.7% |
| 60 TR | 1,488 | 12.5% |

Thus, longer windows provide more temporal context but substantially fewer independent training examples. The improvement with longer windows is meaningful because it occurs despite this reduction in sample count.

### 1.2 Models evaluated

| Model family | Main representation/configuration | Validation | Replication |
|---|---|---|---|
| Logistic regression (LR) | Per-ROI mean/SD/slope features, `C=0.01` | LOPO and LORO, 10–60 TR | One deterministic run per condition |
| MLP | Same per-ROI summary features; hidden 16/32; dropout 0.2 | LOPO and LORO, 10–60 TR | Five seeds per configuration and condition |
| GRU | Sequence input; final-state pooling; several hidden sizes, dropout, weight decay, and inner-CV settings | Primarily LOPO/LORO | Uneven exploratory replication |
| Motion LR | Motion-derived features, `FD=0.25`, `C=0.01` | LOPO and LORO, 10–60 TR | One deterministic run per condition |

### 1.3 Cross-validation and training procedure

The complete procedure is repeated independently for every model configuration, segment length, validation scheme, outer fold, and (for neural networks) random seed:

1. **Construct labeled segments.** Responsive windows are identified using the criteria in Section 1.1, overlapping qualifying windows are merged into responsive intervals, and non-responsive intervals are defined from sufficiently long button-press-free periods. Fixed-length segments are then placed within each interval. In the reported canonical experiments, stride equals segment length, so segments do not overlap. Labels are encoded as responsive = 0 and non-responsive = 1.
2. **Define the outer split before fitting preprocessing or the model.** LOPO holds out all segments from one participant. LORO holds out one participant-run and may retain other runs from that participant in training. Every segment from a held-out unit remains together in the test fold.
3. **Derive the model input.** LR and MLP convert each ROI sequence to mean, SD, and linear-slope features. GRU retains the ordered ROI sequence. The underlying ROI series were z-scored run-wise before segmentation; this is distinct from the fold-specific feature standardization used by LR and MLP.
4. **Fit fold-specific preprocessing on the outer-training subset only.** For LR and MLP, each of the 126 summary features is centered and scaled using a weighted `StandardScaler`. Neither held-out features nor held-out labels enter this fit. The GRU has no additional fold-fitted scaler because it consumes the cached run-wise z-scored sequences directly.
5. **Balance the training contribution of participants and classes.** The exact method depends on the estimator and is detailed below. No test observations are resampled or reweighted during prediction.
6. **Fit a fresh model for the fold.** Fixed hyperparameters are used unless the run is explicitly marked as inner CV. Neural-network parameters, optimizer state, and sampling RNG are reinitialized for each outer fold. LR is effectively deterministic for a fixed fold.
7. **Generate one held-out prediction per segment.** A probability of at least 0.5 is classified as non-responsive; otherwise it is responsive. The threshold is fixed rather than selected on test data.
8. **Aggregate out-of-fold predictions.** After every outer unit has been held out once, pooled, participant-level, class-recall, and participant-weighted metrics are computed as described in Section 1.5. Neural-network results are subsequently summarized across seeds; folds are not treated as independent replicates.

#### Participant and class balancing

Balancing is applied **within each outer-training fold**, after the held-out participant or run has been removed. Let \(n_{pc}\) be the number of training segments for participant \(p\) and class \(c\), and let \(K_p\) be the number of classes actually available for that participant in the training fold. The deterministic weight assigned to each segment in cell \((p,c)\) is proportional to

\[
w_{i\in(p,c)} \propto \frac{1}{K_p n_{pc}}.
\]

The weights are normalized to have mean one. Consequently, every participant has equal total weight; within a participant, each available class has equal total weight; and segments in a large participant/class cell receive less weight than segments in a small cell. If a participant has only one class, that available class receives the participant's full allocation—missing classes are not synthesized.

- **LR:** these deterministic weights are passed to both `StandardScaler.fit(..., sample_weight=weights)` and `LogisticRegression.fit(..., sample_weight=weights)`. Thus LR's likelihood, but not its fixed 0.5 decision threshold, is participant/class weighted. The implementation does not additionally set scikit-learn's `class_weight`; doing so would duplicate the correction.
- **MLP:** the same deterministic weights are used only to estimate the train-fold scaler. Optimization uses a stochastic hierarchical sampler: for each draw it chooses a participant uniformly, chooses uniformly among that participant's available classes, and then chooses a segment uniformly from that participant/class cell. Each epoch contains exactly as many draws as the unresampled training fold, so minority cells can be sampled repeatedly and majority cells can be omitted on a particular epoch. Binary cross-entropy itself is unweighted.
- **GRU:** optimization uses the same participant-then-class hierarchical sampler as the MLP and unweighted binary cross-entropy. There is no fold-fitted summary-feature scaler. Most reported runs use this `participant_then_class` order; the code also supports `class_then_participant`, but those are a different balancing design and should not be pooled with the canonical runs.
- **Motion LR:** it uses the same deterministic participant/class weights as fMRI LR for both train-fold scaling and model fitting.

This procedure balances influence during model fitting; it does **not** alter the composition of the held-out folds. Reported pooled metrics therefore retain the observed test-set imbalance, while participant-weighted balanced accuracy corrects for unequal participant and class contributions at the metric stage.

### 1.4 Model architectures

#### Logistic regression

The fMRI LR is not trained directly on every ROI-by-TR value. Each segment is reduced to three temporal summaries for each of the 42 ROIs:

1. temporal mean;
2. population standard deviation (`ddof=0`);
3. ordinary-least-squares slope per TR, calculated against centered time.

This produces **126 input features** (`42 ROIs × 3 statistics`). Within each outer fold, a `StandardScaler` is fit only on the training data using participant/class weights. A binary scikit-learn logistic regression then maps the standardized feature vector directly to the non-responsive probability:

`126 standardized features → affine logit → sigmoid probability`

The model has 126 coefficients plus one intercept. It uses the `lbfgs` solver, L2 regularization with fixed `C=0.01`, a maximum of 5,000 iterations, and a 0.5 decision threshold. There is no hidden layer or temporal recurrence. The participant/class weights in Section 1.3 enter both the scaler and the penalized likelihood, and there is no inner hyperparameter selection.

#### Multilayer perceptron

The MLP receives the **same 126 standardized mean/SD/slope features** as LR. Its architecture is:

`126 features → fully connected hidden layer → ReLU → dropout → 1 output logit`

The first affine layer computes \(h=\operatorname{ReLU}(W_1x+b_1)\); dropout is applied to this hidden representation during training, and a second affine layer produces one non-responsiveness logit \(z=W_2h+b_2\). At evaluation, dropout is disabled and \(\sigma(z)\) is interpreted as the probability of the non-responsive class.

The main compact configuration has 16 hidden units and **2,049 trainable parameters** (`126×16 + 16` in the hidden layer and `16×1 + 1` in the output layer); the larger configuration has 32 hidden units and **4,097 trainable parameters**. Both use dropout 0.2. The compact model uses 10 epochs and weight decay 0.01, while the larger model uses 30 epochs and weight decay 0.001. Both use Adam with learning rate 0.001, batch size 32, and unweighted binary cross-entropy with logits.

Class balance is handled by the training sampler rather than the loss: each draw first samples a participant uniformly, then one of that participant's available classes uniformly, then a segment from that participant/class cell. Each epoch contains as many draws as the original training subset. The scaler is fit using the corresponding deterministic participant/class weights. The final epoch is evaluated; there is no early stopping or test-based checkpoint selection.

#### Gated recurrent unit

The GRU operates on the full temporal sequence rather than summary features. Each input has shape **`segment length × 42 ROIs`** and comes from the cached z-scored ROI time series. The architecture is:

`ROI sequence → single-layer unidirectional GRU → temporal pooling → dropout → linear output logit`

For the reported experiments, pooling is `final`: the final hidden state is used as the segment representation. The implementation also supports mean pooling over all GRU outputs, but the summarized result folders use final-state pooling. The GRU is single-layer and unidirectional; there is no attention mechanism, bidirectionality, normalization layer, or additional recurrent stack.

At time \(t\), the GRU updates its hidden state with learned reset and update gates, allowing the representation to retain or replace information from earlier TRs. For final pooling, only the last hidden state is passed through dropout and the output affine layer; sigmoid converts that logit to the non-responsive probability. Because the sequence is unidirectional, the state at each TR uses only the current and preceding TRs, although the final state summarizes the complete segment.

With 32 hidden units the model has **7,329 trainable parameters** (`3(42×32 + 32×32 + 2×32)` in the GRU and `32 + 1` in the output layer; PyTorch stores separate input and recurrent biases); with 64 hidden units it has **20,801**. The standard training setup uses dropout 0.2 on the pooled representation, Adam with learning rate 0.001, batch size 32, 30 epochs, and binary cross-entropy with logits. Weight decay varies across experiments from 0.0001 to 0.01. The participant-then-class sampler in Section 1.3 supplies training batches. Unless an experiment explicitly uses inner CV, the final epoch is evaluated without early stopping. Inner-CV runs use grouped splits confined to the outer-training data, select among epochs 10, 15, 20, 25, and 30 by pooled inner-validation balanced accuracy (earlier epoch wins an exact tie), and then retrain on the full outer-training set for the selected number of epochs.

#### Motion logistic-regression architecture

The motion control uses five segment-level framewise-displacement features: mean, maximum, median, population SD, and the fraction of TRs above FD = 0.25. These five features are train-fold standardized and mapped through a single affine logistic-regression output. It uses the same `lbfgs` solver, L2 penalty (`C=0.01`), participant/class weighting, and 0.5 threshold as the fMRI LR. The FD threshold defines one input feature; it is not used to censor segments.

### 1.5 Metrics and interpretation

The updated canonical GRU ranking uses **participant-weighted balanced accuracy** as the primary metric. The two class recalls are first calculated separately within every participant, each class recall is averaged equally across participants for whom that class exists, and those two participant-averaged class recalls are then averaged:

\[
\text{participant-weighted balanced accuracy}
= \frac{\operatorname{mean}_{p \in P_R}(R_p) +
\operatorname{mean}_{p \in P_N}(N_p)}{2}.
\]

Here, \(R_p\) and \(N_p\) are participant \(p\)'s responsive and non-responsive recalls, and \(P_R\) and \(P_N\) include only participants who have the corresponding class. In the canonical 40-TR LOPO results, 19 participants contribute responsive recall and 13 contribute non-responsive recall. This metric prevents participants with more segments from dominating either class recall.

The exported metrics mean:

- **Pooled train accuracy:** accuracy across training predictions, aggregated across folds with fold-training sizes as weights. It describes model fit, not held-out generalization.
- **Pooled test accuracy:** raw held-out accuracy after pooling every test segment; participants with more segments receive more weight.
- **Participant test accuracy:** raw accuracy is calculated within each participant and then averaged equally across participants. It is participant-balanced but not class-balanced.
- **Responsive recall:** fraction of all pooled responsive test segments predicted responsive.
- **Non-responsive recall:** fraction of all pooled non-responsive test segments predicted non-responsive.
- **Balanced test accuracy:** the mean of the two pooled class recalls:

\[
\text{balanced accuracy} = \frac{\text{responsive recall} + \text{non-responsive recall}}{2}.
\]

- **Participant-weighted balanced accuracy:** the mean of the two class recalls after first averaging each recall equally across eligible participants; this is the primary ranking metric for the latest canonical GRU comparison.
- **Class-recall participant counts:** the number of participants contributing to each participant-averaged class recall. A participant contributes only when that class exists in their held-out data.

Balanced accuracy is preferable to raw accuracy when the class counts are unequal. Participant-weighted balanced accuracy additionally addresses unequal segment counts across participants. It does not force the same participants to contribute to both classes; the accompanying counts therefore remain essential context.

## 2. Cross-model comparison

### 2.1 Canonical 40-TR GRU ranking across all participants

Configurations are ranked by mean participant-weighted balanced accuracy across seeds. All rows use final pooling, learning rate 0.001, batch size 32, 30 epochs, no inner CV, and the canonical untrimmed 40-TR segmentation. The ± value is the sample standard deviation across seeds.

| Rank | Configuration | Participant-weighted balanced accuracy | Participant test accuracy | Pooled balanced accuracy | Seeds |
|---:|---|---:|---:|---:|---:|
| 1 | Hidden 64, dropout 0.2, WD 0.001 | **0.804 ± 0.033** | 0.792 | 0.746 | 5 |
| 2 | Hidden 64, dropout 0.2, WD 0.01 | **0.802 ± 0.021** | 0.801 | 0.749 | 5 |
| 3 | Hidden 64, dropout 0.4, WD 0.001 | **0.801 ± 0.022** | 0.791 | 0.743 | 5 |
| 4 | Hidden 32, dropout 0.2, WD 0.001 | **0.798 ± 0.016** | 0.790 | 0.737 | 5 |
| 5 | Hidden 32, dropout 0.4, WD 0.001 | **0.796 ± 0.026** | 0.791 | 0.738 | 5 |
| 6 | Hidden 32, dropout 0.2, WD 0.01 | **0.794 ± 0.030** | 0.795 | 0.747 | 5 |
| 7 | Hidden 32, dropout 0.2, WD 0.003 | **0.789 ± 0.028** | 0.785 | 0.729 | 5 |
| 8 | Hidden 16, dropout 0.2, WD 0.001 | **0.784** | 0.790 | 0.732 | 1 |
| 9 | Hidden 16, dropout 0.4, WD 0.001 | **0.773** | 0.771 | 0.719 | 1 |

The top three means are separated by only 0.003 and have overlapping seed variability, so this table does not establish a unique winning configuration. The participant-weighted values are also systematically higher than pooled balanced accuracy because small participants receive equal weight rather than being overwhelmed by participants with many segments.

### 2.2 Cross-model LOPO comparison using pooled metrics

The following table uses the best segment length for each fixed, reasonably replicated configuration. Standard deviations are across seeds where multiple seeds exist.

| Model/configuration | Segment length | LOPO balanced accuracy | Responsive recall | Non-responsive recall | Pooled test accuracy | Pooled train accuracy |
|---|---:|---:|---:|---:|---:|---:|
| GRU, hidden 64, dropout 0.2, WD 0.01 | 50 | **0.753 ± 0.013** | 0.754 | 0.752 | 0.753 | 0.935 |
| GRU, hidden 64, dropout 0.2, WD 0.001 | 40 | **0.746 ± 0.021** | 0.729 | 0.763 | 0.748 | 0.982 |
| GRU, hidden 32, dropout 0.4, WD 0.001 | 40 | **0.738 ± 0.021** | 0.724 | 0.753 | 0.740 | 0.976 |
| GRU, hidden 32, dropout 0.2, WD 0.01 | 40 | **0.747 ± 0.023** | 0.761 | 0.732 | 0.745 | 0.930 |
| MLP, hidden 16, WD 0.01 | 60 | **0.717 ± 0.025** | 0.716 | 0.717 | 0.717 | 0.895 |
| Logistic regression, `C=0.01` | 40 | **0.702** | 0.775 | 0.629 | 0.692 | 0.829 |
| Motion logistic regression | 60 | **0.594** | 0.972 | 0.216 | 0.544 | 0.548 |

Important qualifications:

1. Both hidden-64 LOPO weight-decay sweeps now have all five planned seeds at every length. The hidden-32/dropout-0.4 canonical 40-TR result also has five LOPO seeds; its pooled balanced mean is 0.738, showing that the earlier 0.769 seed was optimistic relative to the completed set.
2. The WD=0.01 GRU has five canonical seeds. Additional rows for the same architecture in the consolidated file belong to alternate labeling or trimming conditions and must not be pooled with the canonical experiment.
3. LR and motion LR are deterministic single runs. Their lack of a standard deviation is not evidence of greater statistical certainty.
4. The selected window differs by model in this table. At the directly comparable 40-TR condition, LR is 0.702, the compact MLP is 0.694, and replicated GRUs are generally about 0.74–0.75.

### 2.3 LORO comparison

LORO performance is higher for every fMRI model tested in both schemes:

| Model/configuration | Segment length | LORO balanced accuracy | LOPO balanced accuracy | LORO–LOPO difference |
|---|---:|---:|---:|---:|
| LR | 60 | 0.783 | 0.701 | +0.082 |
| MLP, hidden 16, WD 0.01 | 60 | 0.791 | 0.717 | +0.074 |
| GRU, hidden 32, WD 0.001 | 40 | 0.817 | 0.737 | +0.080 |
| GRU, hidden 32, WD 0.003 | 40 | 0.817 | 0.729 | +0.088 |
| GRU, hidden 32, dropout 0.4, seed 42 | 40 | 0.825 | 0.769 | +0.056 |
| GRU, hidden 64, WD 0.001 | 50 | 0.822 | 0.735 | +0.087 |
| GRU, hidden 64, WD 0.01 | 50 | 0.811 | 0.753 | +0.058 |
| Motion LR | 60 | 0.598 | 0.594 | +0.003 |

The approximately 7–9 percentage-point gap for the fMRI models is evidence of participant-specific distribution shift. A model can generalize to another run from a participant more easily than to a completely unseen participant. The hidden-32/dropout-0.4 row is explicitly the paired seed-42 comparison because only that LORO seed is currently available; it should not be compared with the newer five-seed LOPO mean. The almost absent LORO advantage for motion suggests its weak signal is broadly similar across these split definitions rather than being driven by useful participant-specific structure.

## 3. Logistic regression results

### 3.1 Performance by temporal window

| Length | LOPO balanced accuracy | LOPO resp. recall | LOPO non-resp. recall | LORO balanced accuracy |
|---:|---:|---:|---:|---:|
| 10 | 0.646 | 0.686 | 0.606 | 0.696 |
| 20 | 0.677 | 0.723 | 0.631 | 0.732 |
| 30 | 0.698 | 0.765 | 0.630 | 0.757 |
| 40 | **0.702** | 0.775 | 0.629 | 0.763 |
| 50 | 0.688 | 0.741 | 0.635 | 0.763 |
| 60 | 0.701 | 0.749 | **0.653** | **0.783** |

### 3.2 Interpretation

LR improves quickly from 10 to 30 TRs and then plateaus around 0.70 LOPO balanced accuracy. This is a strong baseline: it shows that much of the discriminative information can be represented by a linear boundary over per-ROI temporal summary features.

The model consistently recalls responsive segments better than non-responsive segments. At 40 TRs the gap is 0.146 (0.775 versus 0.629). This bias is moderate compared with the motion model but shows that the classification problem is not equally difficult in both directions.

Training accuracy rises from 0.759 at 10 TRs to 0.851 at 60 TRs while LOPO test performance plateaus. This widening train–test gap suggests that the longer-window summaries increasingly permit participant-specific fitting.

## 4. MLP results

Two MLP configurations were run across all six lengths and both validation schemes, with five seeds each:

- **Compact/regularized:** hidden 16, dropout 0.2, 10 epochs, WD 0.01
- **Larger/less regularized:** hidden 32, dropout 0.2, 30 epochs, WD 0.001

### 4.1 LOPO balanced accuracy

| Length | Hidden 16, WD 0.01 | Hidden 32, WD 0.001 |
|---:|---:|---:|
| 10 | 0.669 | 0.660 |
| 20 | 0.681 | 0.681 |
| 30 | **0.693** | 0.670 |
| 40 | **0.694** | 0.676 |
| 50 | **0.687** | 0.660 |
| 60 | **0.717** | 0.672 |

The compact, more strongly regularized MLP is better at every length except the effective tie at 20 TRs. Its advantage is especially large at 60 TRs: approximately 4.5 percentage points.

### 4.2 LORO balanced accuracy

| Length | Hidden 16, WD 0.01 | Hidden 32, WD 0.001 |
|---:|---:|---:|
| 10 | 0.747 | **0.757** |
| 20 | 0.759 | **0.778** |
| 30 | 0.775 | **0.778** |
| 40 | **0.784** | 0.782 |
| 50 | **0.779** | 0.776 |
| 60 | **0.791** | 0.776 |

The larger MLP is competitive under LORO but worse under LOPO. This is consistent with participant-specific overfitting: its mean training accuracy is about 0.97–0.98 for lengths of 20 TR or more, compared with about 0.89–0.90 for the compact model.

### 4.3 Seed stability

For the compact MLP, LOPO standard deviations across seeds are approximately 0.005–0.025 depending on length. Variability is largest at 60 TRs, where balanced accuracy ranges from **0.666 to 0.727**. The 60-TR mean is best, but its smaller dataset and larger seed variance make 30–40 TRs a more conservative operating region.

## 5. GRU results

The GRU directory contains the broadest experiment set: temporal lengths, trimming variants, alternative responsiveness settings, weight-decay sweeps, dropout and hidden-size exploration, inner cross-validation, and LOPO/LORO evaluations. The consolidated `all_rois_summary.csv` contains **388 rows**, but replication remains uneven outside the principal sweeps, so this section separates stable trends from exploratory maxima.

### 5.1 Temporal context

For the common hidden-32, dropout-0.2, WD=0.001 configuration, mean LOPO balanced accuracy changes as follows:

| Length | Mean balanced accuracy | Mean pooled test accuracy | Mean training accuracy |
|---:|---:|---:|---:|
| 10 | 0.604 | 0.608 | 0.967 |
| 20 | 0.670 | 0.675 | 0.981 |
| 30 | 0.727 | 0.731 | 0.983 |
| 40 | 0.737 | 0.740 | 0.978 |
| 50 | **0.741** | **0.745** | 0.978 |
| 60 | 0.737 | 0.741 | 0.960 |

The GRU needs at least about 30 TRs to become clearly competitive. Its 10-TR performance is poor despite near-perfect training accuracy, indicating severe overfitting when sequences contain little temporal context.

### 5.2 Regularization and architecture exploration at 40 TRs

| Configuration | LOPO balanced accuracy | Seeds/rows | Train accuracy | Interpretation |
|---|---:|---:|---:|---|
| Hidden 64, dropout 0.2, WD 0.001 | **0.746 ± 0.021** | 5 | 0.982 | Complete five-seed LOPO result |
| Hidden 64, dropout 0.2, WD 0.01 | **0.749 ± 0.024** | 5 | 0.934 | Stronger regularization; best mean occurs at 50 TR |
| Hidden 32, dropout 0.4, WD 0.001 | **0.738 ± 0.021** | 5 | 0.976 | Completed replication is below the earlier seed-42 estimate |
| Hidden 32, dropout 0.2, WD 0.01 | **0.747 ± 0.023** | 5 | 0.931 | Best replicated evidence that stronger weight decay improves generalization |
| Hidden 32, dropout 0.2, inner CV=5, WD 0.001 | 0.739 ± 0.021 | 5 | 0.961 | No clear gain from inner CV |
| Hidden 32, dropout 0.2, WD 0.003 | 0.729 ± 0.015 | 5 | 0.964 | Intermediate regularization did not improve the mean |
| Hidden 16, dropout 0.4, WD 0.001 | 0.719 | 1 | 0.937 | Lower capacity reduces fit but not enough evidence for ranking |

The weight-decay sweep is informative. Increasing WD from 0.001 to 0.01 sharply reduces training accuracy while maintaining or improving LOPO test performance. For hidden 64, the advantage of WD 0.01 is clearest at 30 and 50 TRs; the two weight decays are close at 40 and 60 TRs. Inner CV does not currently show a reliable benefit, and its class recalls shift: at WD 0.01 with inner CV, responsive recall is 0.760 but non-responsive recall falls to 0.717.

### 5.3 Class balance

The better GRU configurations are more balanced across classes than LR. For example, the five-seed hidden-64, WD=0.001 model has responsive/non-responsive recall of 0.729/0.763, while the hidden-32, WD=0.01 model averages 0.761/0.732. This is an advantage over LR’s 0.775/0.629 at 40 TRs.

### 5.4 Evidence strength

The GRU results support the conclusion that sequence modeling can improve on summary-feature linear and MLP models, particularly at 40–60 TRs. However, the precise best architecture is not settled: the top three participant-weighted means at 40 TR differ by only 0.003 and their seed distributions overlap. The completed hidden-32/dropout-0.4 runs do not reproduce the initially striking seed-42 result as a mean. This illustrates why model comparisons should use all planned seeds.

### 5.5 Completed hidden-64 sweeps

The hidden-64, dropout-0.2, WD=0.001 LOPO sweep now has five completed seeds at every tested length:

| Length | Balanced accuracy | Responsive recall | Non-responsive recall | Pooled test accuracy | Train accuracy |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.617 ± 0.008 | 0.572 | 0.661 | 0.621 | 0.973 |
| 20 | 0.684 ± 0.012 | 0.643 | 0.726 | 0.689 | 0.982 |
| 30 | 0.727 ± 0.007 | 0.709 | 0.745 | 0.729 | 0.986 |
| 40 | **0.746 ± 0.021** | 0.729 | 0.763 | 0.748 | 0.982 |
| 50 | 0.735 ± 0.017 | 0.708 | 0.763 | 0.739 | 0.984 |
| 60 | **0.745 ± 0.016** | 0.728 | 0.762 | 0.748 | 0.978 |

The hidden-64, WD 0.01 LOPO sweep is also complete:

| Length | Balanced accuracy | Responsive recall | Non-responsive recall | Pooled test accuracy | Train accuracy |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.628 ± 0.011 | 0.629 | 0.627 | 0.628 | 0.890 |
| 20 | 0.688 ± 0.016 | 0.678 | 0.697 | 0.689 | 0.927 |
| 30 | 0.745 ± 0.016 | 0.753 | 0.737 | 0.744 | 0.938 |
| 40 | 0.749 ± 0.024 | 0.767 | 0.731 | 0.747 | 0.934 |
| 50 | **0.753 ± 0.013** | 0.754 | 0.752 | 0.753 | 0.935 |
| 60 | 0.742 ± 0.018 | 0.762 | 0.723 | 0.740 | 0.921 |

The WD 0.001 results plateau from 40 to 60 TRs, while stronger regularization produces its best mean at 50 TRs. Forty TRs remains a defensible data-efficient default because it retains 66% more segments than 60 TRs and performs within 0.004 of the best hidden-64/WD 0.01 mean.

The hidden-64, WD 0.001 LORO sweep is now complete at five seeds per length. It rises from 0.712 ± 0.009 at 10 TRs to a maximum of **0.822 ± 0.005 at 50 TRs**. The WD 0.01 LORO sweep has five seeds at 20–60 TRs and peaks at **0.811 ± 0.015 at 50 TRs**; its 10-TR condition is still absent. These replicated LORO results reinforce the approximately 6–9 point advantage over LOPO at longer lengths.

## 6. Motion-classifier results

### 6.1 Performance

| Length | LOPO balanced accuracy | Responsive recall | Non-responsive recall | LORO balanced accuracy |
|---:|---:|---:|---:|---:|
| 10 | 0.571 | 0.858 | 0.283 | 0.583 |
| 20 | 0.585 | 0.911 | 0.259 | 0.595 |
| 30 | 0.590 | 0.924 | 0.256 | 0.596 |
| 40 | 0.592 | 0.935 | 0.249 | 0.597 |
| 50 | 0.590 | 0.923 | 0.256 | 0.596 |
| 60 | **0.594** | **0.972** | 0.216 | **0.598** |

### 6.2 Interpretation as a confound check

Motion is modestly predictive above chance, but its predictions are extremely asymmetric. The model largely predicts the responsive class, producing very high responsive recall and very low non-responsive recall. Raw accuracy is only about 0.54–0.56.

The fMRI models exceed motion by approximately 11–17 balanced-accuracy points under LOPO and exhibit substantially healthier class balance. Therefore, the current evidence does **not** support the claim that the fMRI classifiers merely decode motion.

This is not a complete confound analysis. A stronger test would evaluate fMRI after motion residualization, include motion features jointly with fMRI, and quantify incremental performance using identical held-out predictions.

## 7. Generalization, overfitting, and participant heterogeneity

### 7.1 Train–test gaps

Approximate gaps for representative best conditions are:

| Model | Train accuracy | Test accuracy | Gap |
|---|---:|---:|---:|
| LR, 40 TR | 0.829 | 0.692 | 0.137 |
| MLP hidden 16, 60 TR | 0.895 | 0.717 | 0.178 |
| GRU hidden 32, WD 0.01, 40 TR | 0.930 | 0.745 | 0.185 |
| GRU hidden 64, 40 TR | 0.982 | 0.748 | 0.234 |
| GRU hidden 64, WD 0.01, 50 TR | 0.935 | 0.753 | 0.182 |

All nonlinear fMRI models overfit to some degree. Stronger GRU weight decay produces the best tradeoff between fit and replication.

### 7.2 Fold variability

The saved fold-level results show substantial participant-to-participant variation, including some participants with near-perfect accuracy and others near or below chance. Some held-out participants have very small or single-class segment sets at longer durations, making individual participant balanced accuracies unstable or partially undefined. This explains why participant-weighted accuracy can look much higher than pooled balanced accuracy.

Consequently:

- participant-weighted balanced accuracy should be the headline measure when the goal is performance across participants;
- pooled balanced accuracy should remain visible as a complementary segment-weighted measure;
- fold-level plots should always include held-out class counts;
- uncertainty should be estimated at the participant level, not by treating segments as independent observations.

## 8. What can and cannot be concluded

### Supported conclusions

1. **There is useful fMRI signal for responsiveness classification.** Multiple model families exceed the motion baseline and chance-level balanced accuracy under LOPO.
2. **Temporal context matters.** Windows of 30–60 TRs outperform 10-TR windows for LR, MLP, and GRU.
3. **Sequence modeling is promising.** Properly regularized GRUs outperform LR and MLP in the current results.
4. **Unseen-participant generalization is harder than unseen-run generalization.** The LOPO–LORO gap is consistent and material.
5. **Motion is a weak, class-biased predictor rather than a sufficient explanation.**

### Not yet supported

1. **A definitive winning GRU architecture.** Hidden 64/WD 0.001 ranks first at 40 TRs by participant-weighted balanced accuracy, but its 0.804 mean is effectively tied with the next two configurations given overlapping seed variability. Across temporal lengths, hidden-64/WD 0.01 remains numerically strongest at 50 TRs by the older pooled metric; participant-weighted summaries are not yet exported for every length.
2. **Formal statistical superiority of GRU over MLP or LR.** No participant-level paired confidence intervals or corrected hypothesis tests are included yet.
3. **Independence from all nuisance signals.** Motion-only classification is informative but is not equivalent to residualizing or conditioning the fMRI model on motion.
4. **Clinical or prospective generalization.** All results are internal cross-validation on the available participant cohort.
5. **An exact optimal segment duration.** Performance and uncertainty trade off against rapidly falling sample counts, and durations were selected on the same experimental collection.

## 9. Recommended next experiments

### Highest priority

1. **Run a focused confirmation study at 40 and 50 TRs.** Pre-specify the leading hidden-64 configurations, including WD 0.001 and 0.01. Treat the completed principal sweeps as exploratory evidence used to select these conditions.
2. **Use LOPO as the primary endpoint.** Report LORO only as a secondary within-participant/across-run analysis.
3. **Add participant-level uncertainty.** Bootstrap participants or use paired permutation tests on held-out participant predictions. Avoid segment-level confidence intervals that assume overlapping or within-participant samples are independent.
4. **Perform a matched motion-confound analysis.** Compare motion-only, fMRI-only, and fMRI-plus-motion models on identical folds; also test motion-residualized fMRI.

### Secondary priority

5. **Keep LR and compact MLP as mandatory baselines.** They are competitive, interpretable, and useful for detecting when a deep model is merely overfitting.
6. **Investigate participant normalization/domain shift.** The LOPO–LORO gap suggests that participant-specific scaling, harmonization, or domain-adversarial approaches may help.
7. **Inspect errors by participant and class count.** Focus on consistently difficult participants and ensure results are not dominated by a few large folds.

## 10. Reporting recommendations

For future summary figures and manuscripts:

- Lead with LOPO participant-weighted balanced accuracy and its two participant-averaged class recalls; retain pooled metrics for transparency.
- Show seed distributions rather than only the best seed.
- State the number of participants, runs, segments, and seeds for every result.
- Separate exploratory tuning from confirmatory evaluation.
- Display the motion baseline beside every fMRI model.
- Do not compare participant-weighted raw accuracy directly with pooled or participant-weighted balanced accuracy.
- Label LORO clearly as allowing other runs from the held-out run’s participant in training.

## 11. Overall assessment

The experiments provide encouraging evidence that responsiveness can be decoded from ROI-level fMRI time series in unseen participants. At the canonical 40-TR condition, well-replicated GRUs achieve approximately **0.79–0.80 participant-weighted balanced accuracy**, with hidden 64/WD 0.001 ranking first at **0.804 ± 0.033**. The top configurations are too close to declare a unique winner. Their corresponding pooled balanced accuracies remain approximately **0.74–0.75**, highlighting the practical effect of weighting participants equally. Logistic regression at approximately 0.70 pooled balanced accuracy remains a strong baseline, while motion at approximately 0.59 is clearly weaker.

The next phase should be narrower and confirmatory: equalize seeds, pre-specify a few GRU configurations, preserve LOPO as the main evaluation, and quantify participant-level uncertainty and motion-adjusted performance.

