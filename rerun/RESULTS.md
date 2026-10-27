# Selective History Lab: rerun with the three fixes

27 October 2026. This rerun extends the pilot in this repository without changing its model: the scripts import `history_model.py` from the repository root, whose only change is one word in its docstring. `history_model_ext.py` adds per-learner random-number streams (so conditions can be compared on identical simulated learners), variable probe length, a skill-specific sensor, a sensor read on every practice trial and a yoked arm. Eight checks in `test_ext.py` confirm that the extension reproduces the original likelihood and fits exactly where they overlap, and test what it adds.

**Terminology.** The adaptive system is a *personalization algorithm*, not a tutor, and what it stores is its *retained history*. This version of the repository uses those terms throughout.

Each experiment asks one question: does the proposed fix give the hypothesis something real to detect?

## Experiment 1 · Does the retained-history rule change what learners practise or learn? (Hypothesis 1)

The same 2,000 simulated learners (same persistent type and starting state) went through every condition. Differences are paired by learner, so their standard errors are small. About 90 practice trials per learner.

| Comparison (paired, same learners) | Practice A / B / AB (trials) | Both skills mastered | Delayed-test accuracy |
|---|---|---:|---:|
| Balanced − recency, capacity 3 | −0.12 / −0.03 / +0.15 | +0.3 pts (SE 0.3) | +0.0 pts (SE 0.1) |
| Balanced − recency, capacity 6 | −0.15 / −0.06 / +0.22 | +0.1 (0.3) | +0.2 (0.1) |
| Balanced − recency, capacity 9 | −0.26 / +0.01 / +0.25 | +0.5 (0.3) | +0.2 (0.1) |
| Balanced − recency, capacity 18 (pilot) | −0.34 / +0.05 / +0.29 | −0.4 (0.3) | −0.1 (0.1) |
| Balanced − recency, capacity 3, no exploration | −0.13 / −0.03 / +0.16 | +0.5 (0.3) | +0.1 (0.1) |
| Balanced − recency, capacity 18, no exploration | −0.45 / +0.08 / +0.37 | +0.2 (0.3) | +0.1 (0.1) |
| Uniform random − recency | +1.96 / −0.23 / −1.74 | −3.1 (0.45) | −1.1 (0.2) |
| Yoked − recency (another person's sequence) | +0.04 / −0.17 / +0.13 | −2.4 (0.7) | −0.8 (0.3) |

- **Smaller capacity does not fix the manipulation.** At every capacity, and with exploration removed, the two retention rules differ by under half a practice trial per activity and under half a point in mastery. With three activities and a rule that favours the least successful one, the retained history hardly changes the algorithm's ranking.
- **Contingency matters, a little.** Learners whose sequence was tailored to their own responses master both skills 2.4 points more often than learners given someone else's sequence with the same average composition (the yoked arm).
- **None of these effects is detectable in a human between-subjects study.** Using the simulated between-person SDs (0.48 for mastery of both skills, 0.26 for delayed accuracy), 80% power at α = .05 needs about 6,700 per arm for the yoked mastery effect and about 19,000 per arm for its delayed-accuracy effect; the retention-rule effect would need hundreds of thousands. These numbers are bound to this model's illustrative parameters.

**Answer:** no. In a three-activity environment the retained-history rule cannot be made to matter. Hypothesis 1 needs a richer environment in which what the algorithm retains decides which items or features a person ever encounters, as in Bahg et al.'s personalization task (sampled dimensions fell from 6.0 under control to roughly 3–5 under personalization).

## Experiment 2 · When does ignoring persistent learner differences distort the contrasts? (Hypothesis 2)

40 replications per cell, N = 120, recency-retained history. Each dataset was fitted with the homogeneous model (learner types ignored) and the specified-mixture model (types modelled, with their separation known to the fitter, an optimistic benchmark). *h* sets how far apart slow and fast learners are; at the pilot's gains, a slow/fast learner gains skill a from an A activity with probability .038/.162 at h = 0.8 (the pilot), .017/.301 at h = 1.6 and .008/.489 at h = 2.4.

| Gains | h | Contrast (truth) | Homogeneous: shrinkage toward zero | Homogeneous: coverage | Mixture: bias | Mixture: coverage | Mixture: interval width |
|---|---|---|---:|---:|---:|---:|---:|
| pilot | 0.8 | a (−0.042) | 17% | 90% | +0.007 | 93% | 0.096 |
| pilot | 0.8 | b (−0.018) | 36% | 95% | +0.002 | 98% | 0.078 |
| pilot | 1.6 | a (−0.060) | 47% | 22% (of 36) | +0.007 | 98% | 0.160 |
| pilot | 1.6 | b (−0.028) | 76% | 61% (of 36) | +0.005 | 95% | 0.133 |
| pilot | 2.4 | a (−0.076) | 75% | 0% (of 31) | +0.007 | 98% | 0.242 |
| pilot | 2.4 | b (−0.039) | 93% | 3% (of 31) | +0.010 | 93% | 0.211 |
| 3 × pilot | 0.8 | a (−0.107) | 24% | 93% | +0.018 | 95% | 0.222 |
| 3 × pilot | 1.6 | a (−0.102) | 45% | 43% | +0.007 | 93% | 0.233 |
| 3 × pilot | 2.4 | a (−0.079) | 63% | 3% (of 38) | −0.005 | 90% | 0.241 |
| 3 × pilot | 2.4 | b (−0.045) | 78% | 26% (of 38) | −0.004 | 93% | 0.230 |

Coverage is for nominal 95% model-based intervals; "of n" marks cells where the fitter refused some intervals at a parameter boundary. Full results, including bias Monte Carlo SEs, exact coverage bounds and sandwich intervals, are in `results/e2_summary.csv`.

- **Ignoring learner differences always shrinks the contrasts toward zero, and the shrinkage grows with h.** At the pilot's setting it is already 17% for skill a and 36% for skill b. It looks harmless there only because, at N = 120, the sampling error is larger than the bias; at the sample sizes needed to detect the contrasts, the bias would dominate. At h ≥ 1.6 intervals cover the truth in 0–61% of datasets, and the skill-b estimate has the wrong sign in 25–40% of datasets.
- **Modelling the differences removes the bias but costs precision.** The mixture model's coverage stays at 85–98%, but its intervals are 1.4–8 times wider than the homogeneous model's (the gap grows with h), and at N = 120 they exclude zero in only 5–43% of datasets. This is with the type separation known to the fitter; estimating it would widen them further.

**Answer:** yes. Mapping the regime makes Hypothesis 2 testable and gives it a direction: omitted heterogeneity attenuates the contrast, increasingly so as learners differ more. Which regime real learners are in is an empirical question. Estimating the spread of individual learning rates in the public category-learning data (Mack; Bahg) would tell you before you design the study.

## Experiment 3 · How reliable must a second channel be to beat asking more questions? (Hypothesis 3)

40 replications per condition, N = 120, recency-retained history, heterogeneity 0.8, fitted with the specified-mixture model. Every condition uses the same simulated learners, practice and learning (common random numbers), so differences reflect measurement alone. Width ratios are paired by replication against the pilot design (behaviour only, 9 probe items per block).

| Measurement design | Interval width vs pilot, skill a | Interval width vs pilot, skill b | Coverage a / b |
|---|---:|---:|---:|
| Behaviour, 9 probe items per block (pilot) | 1.000 | 1.000 | 95% / 100% |
| Behaviour, 18 probe items | 0.983 | 0.979 | 98% / 98% |
| Behaviour, 27 probe items | 0.979 | 0.973 | 98% / 100% |
| Total-mastery sensor at each probe block, SD 0.8 (pilot) | 0.995 | 0.994 | 95% / 98% |
| Total-mastery sensor, SD 0.4 | 0.987 | 0.984 | 98% / 98% |
| Total-mastery sensor, SD 0.2 | 0.980 | 0.975 | 98% / 98% |
| Skill-specific sensor, SD 0.8 | 0.997 | 0.993 | 93% / 100% |
| Skill-specific sensor, SD 0.4 | 0.989 | 0.983 | 98% / 98% |
| Skill-specific sensor, SD 0.2 | 0.979 | 0.973 | 98% / 98% |
| **Perfect measurement** (latent state on every trial, types known) | 0.54 | 0.54 | 95% / 98% |

Monte Carlo SEs of the width ratios are 0.002–0.005. Full results are in `results/e3_summary.csv` and `results/e3b_oracle.csv`.

- **Neither more questions nor a better sensor helps.** Tripling the probe items, or adding a near-perfect skill-specific sensor, narrows the intervals by 2–3%. The two are about equally (un)helpful.
- **Measurement could help in principle, but not at the probe blocks.** With the latent state observed on every trial, intervals would be about 46% narrower. Probes and probe-time sensors only see the state at the end of each 12-trial practice block, so they cannot tell which activity produced a transition, which is exactly what the contrasts ask.

### Experiment 3c · The same channel, measured on every practice trial

20 replications per condition, the same learners as Experiment 3. One reading on each of the 90 practice trials, taken before that trial's learning (skill-specific version: skill a on A trials, skill b on B trials, total mastery on AB trials). This is not an equal-burden comparison: 90 readings per person instead of 7.

| Trial-level channel | Interval width vs pilot, skill a | Skill b | Coverage a / b | Intervals excluding zero, a / b |
|---|---:|---:|---:|---:|
| Pilot design (behaviour, 9 probe items) | 1.00 | 1.00 | 95% / 100% | 45% / 5% |
| Total-mastery signal, SD 0.8 | 0.80 | 0.76 | 95% / 100% | 50% / 20% |
| Total-mastery signal, SD 0.4 | 0.64 | 0.61 | 90% / 100% | 75% / 25% |
| Skill-specific signal, SD 0.8 | 0.83 | 0.81 | 85% / 100% | 55% / 10% |
| Skill-specific signal, SD 0.4 | 0.66 | 0.64 | 90% / 100% | 80% / 20% |

Monte Carlo SEs of the width ratios are 0.005–0.009. With 20 replications, coverage of 85–90% is compatible with 95% (exact bounds in `results/e3c_summary.csv`).

- **A trial-level channel helps substantially.** At SD 0.4 it narrows intervals by 34–39%, which is worth about 2.3–2.7 times as many participants (width shrinks with the square root of N). At SD 0.8 the gain is 17–24%, worth about 1.5–1.7 times as many.
- **Skill-specific is not better here.** A total-mastery signal on every trial does as well or slightly better, because it reflects the practised skill on A and B trials and both skills on AB trials.

**Answer to Hypothesis 3:** the pilot's version, a second channel sampled at the assessment blocks, cannot help this estimand however reliable it is. A channel measured during practice, trial by trial, can. Candidate trial-level signals are response time, confidence, eye tracking or a trial-wise neural signal, each with its own observation model to build and validate. Even with perfect measurement, the skill-b contrast stays hard to detect at N = 120 (truth/SE ≈ 1.7).

## What this means for the study design

1. **Hypothesis 1 needs a different environment, not a different capacity.** In a three-activity world the retained-history rule cannot matter. Build the human-learning arm on a task where what the algorithm retains decides what a person ever encounters, as in Bahg et al.'s personalization paradigm, and show in simulation that the manipulation moves exposure before recruiting.
2. **Hypothesis 2 has a direction and a dependency.** Ignoring persistent learner differences attenuates the contrast, more so as learners differ more. Estimate how much real learners differ, from the Mack or Bahg data, before choosing the model and N.
3. **Hypothesis 3 should be reframed.** The question is not whether a second modality helps but whether a channel measured during practice carries information about when learning happens. Compare it against the cheapest trial-level behavioural signals (response time, confidence) before any imaging.
4. **Limits of this rerun.** Parameters are the pilot's illustrative values, not estimates from data. The mixture model knows the type separation and the overnight loss. Experiment 2 was planned at 60 replications per cell and stopped at 40 to save time; Experiment 3c used 20.

## Files

| File | Contents |
|---|---|
| `history_model_ext.py` | Extension: per-learner random streams, probe length, skill-specific and trial-level sensors, yoked arm |
| `test_ext.py` | 8 checks: likelihood and fits identical to the original, brute-force path enumeration, common random numbers, yoking, trial-level sensor |
| `e1_manipulation.py`, `e1b_no_exploration.py` | Experiment 1 |
| `e2_heterogeneity.py` | Experiment 2 |
| `e3_sensor_vs_probes.py`, `e3b_oracle_ceiling.py`, `e3c_practice_sensor.py` | Experiment 3 |
| `summarize.py` | Summaries with Monte Carlo uncertainty |
| `results/` | Every fit (`*_fits.jsonl`), summaries (`*_summary.csv`) and Experiment 1 tables |

## Reproduce

```bash
pip install numpy scipy pandas numba
cd rerun
python test_ext.py
python e1_manipulation.py && python e1b_no_exploration.py
python e2_heterogeneity.py 2          # about 40 minutes on 2 cores; set REPS in the file
python e3_sensor_vs_probes.py 2       # about 55 minutes on 2 cores
python e3b_oracle_ceiling.py
python e3c_practice_sensor.py         # about 20 minutes on 2 cores
python summarize.py e2 40 && python summarize.py e3 && python summarize.py e3c
```
