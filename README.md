# Selective History Lab

A runnable research pilot for **Selective Histories in Adaptive Learning and Multimodal Inference** (25 October 2026). The project asks two distinct questions: how personalization changes human learning and confidence, and how the restricted history a personalization algorithm retains affects a researcher's inference about learning. The simulation addresses only the second. It does not model human confidence or generalization to novel items; those remain prospective cognitive aims.

## What has been built and run

- A continuing two-skill latent-state simulator whose personalization algorithm keeps either a recency-based or a requirement-balanced retained history. The researcher retains the complete log. Learner type persists across all practice, probes and tests.
- Four fitted models: homogeneous or specified two-type dynamics, each using behavior alone or behavior plus an abstract auxiliary sensor. The sensor intercept, loading and noise are estimated. Heterogeneity support and overnight retention are fixed benchmark assumptions.
- A pilot with 12 repetitions in each of four policy/sensor settings, 120 synthetic participants per dataset and 192 fits. Clean/noisy sensor settings share latent trajectories. This is not 48 independent scientific replications.
- Numerical Hessian and participant-score sandwich intervals; all fit, boundary and interval diagnostics are retained. Coverage summaries include exact binomial Monte Carlo bounds.
- An independent simulated cohort under uniform assignment for illustrative one-step prediction scores. Parameters remain frozen; past test responses update the filtering state.
- An audit and descriptive baseline of 8,832 public category-learning trials from 23 participants, with a complete timing join and a sensitivity analysis excluding subject 09.
- An audit and descriptive baseline of the released Bahg personalization aggregate: 9,600 test responses from the 300 participants in the release. The release covers three category structures (cluster, crosscutting and ring), one of which the published analysis excluded, so these 300 are not the published analysis sample. The aggregate holds test responses and exposure summaries, not learning sequences, so it cannot supply learning-rate estimates for the hidden Markov model. The 0–10 confidence ratings are treated as ordinal.
- Eight statistical implementation tests and an executable notebook.

The project does **not** establish a novel theorem, an adequate recruitment target, a causal effect of replaying alternative teaching policies, or an fMRI/PET finding. No imaging volumes have been processed.

## Run locally

Python 3.11 or later is recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python test_model.py
python public_category.py
python public_personalization.py
python run_pilot.py --reps 12 --n 120 --starts 2 --workers 4
python evaluate_predictions.py
```

`public_category.py` uses the included, attributed source archives. `public_personalization.py` downloads the small processed source aggregate from the authors' OSF project on first use. Downloaded content is parsed as data; author scripts are not executed. Network access is needed for that first download.

Open `Selective_History_Lab.ipynb` in Jupyter or Google Colab for a small executable demonstration. Its embedded modules allow it to run without the ZIP. The demonstration uses fewer synthetic participants than the supplied Monte Carlo results and does not rerun the full pilot automatically.

Numba accelerates the likelihood. The model has a correct but much slower fallback if Numba is unavailable. Parallel workers are ordinary local computation, not remote agents. On platforms with multithreaded numerical libraries, setting `OPENBLAS_NUM_THREADS=1` avoids unnecessary oversubscription.

## Inspect the completed evidence

| Location | Contents |
|---|---|
| `outputs/pilot/fits.json` | All 192 fits, parameters, optimizer starts and interval flags |
| `outputs/pilot/replicate_results.csv` | Per-replicate estimates for both gain contrasts |
| `outputs/pilot/summary.csv` | Bias, RMSE, interval width, coverage and Monte Carlo uncertainty |
| `outputs/prediction/independent_cohort.csv` | Independent-cohort Brier score, log loss and calibration-in-the-large |
| `outputs/public_category/` | Joined events, data audit, provenance, learning curves and figure |
| `outputs/public_personalization/` | Condition and exposure summaries, source hash and coding audit |
| `outputs/validation.txt` | Executed implementation checks |

## Modeling boundaries

The fitted auxiliary observation is a scalar Gaussian signal of total mastery. It cannot distinguish states 10 and 01 by itself. It is not an EEG electrode, BOLD response, PET signal, decoded confidence score or direct measurement of competence. A modality-specific observation model and independent validation are required before such an interpretation.

The simulator implements 18 initial practice trials, a 9-item probe, six blocks of 12 practice trials plus 9 probes, 24 immediate test responses, an overnight loss step and 24 delayed test responses. Each practice block has three scheduled exploration trials and nine adaptive trials. Probes/tests do not change state by assumption. The overnight loss is known to the fitter. Responses on the tests use the same observation law, so this is not a mechanistic simulation of novel-item generalization.

The two target parameters average gain rates over the same equal mixture of learner types under every policy: joint activity minus the corresponding single activity. They are not policy-specific averages among whichever learners remain unmastered. Homogeneous models are intentionally misspecified when the generated types differ.

The policy factor is omitted from the sequential likelihood because this fixed policy is a known function of completely logged past observations and has no unknown generative parameters. This is not ordinary conditioning on the entire future-selected action sequence, and does not justify ignoring unrecorded signals used by a real personalization algorithm.

## Rerun: does each hypothesis have something to detect?

`rerun/` extends the simulator without changing its model (the likelihood and fits match the original exactly) and compares conditions on the same simulated learners. Full tables, Monte Carlo uncertainty and limits are in [`rerun/RESULTS.md`](rerun/RESULTS.md).

- **Hypothesis 1, the retained-history rule changes learning: only small effects in the settings tested.** Across the parameter settings and capacities tested (3 to 18 trials), retention produced small average differences in exposure and learning: under one practice trial per activity and under half a point in mastery, even with exploration removed entirely. On particular constructed histories the two rules can assign activities very differently, so this is not evidence that retention cannot matter with three activities. Tailoring practice to a learner's own responses, rather than replaying another learner's sequence, raises mastery of both skills by 2.4 points, which would need about 6,700 people per arm to detect.
- **Hypothesis 2, ignoring persistent learner differences distorts inference: yes; under this model, toward zero.** With two equal-sized learner types, the homogeneous model shrinks the two contrasts by 17% and 36% at the pilot's heterogeneity and by up to 93% at the largest tested, where its intervals cover the truth in 0–61% of datasets. The direction depends on how learners differ: an earlier simulation with differently structured differences produced a reversal. The mixture benchmark, told the true type separation and weights, has little bias but intervals 1.4–8 times wider; it does not show that heterogeneity can be recovered when those quantities are estimated.
- **Hypothesis 3, a second channel improves inference: barely, when it is read at the assessment blocks.** Tripling the probe items, or adding a probe-time sensor at the best reliability tested, narrows intervals by 2–3%. The same channel read on every practice trial narrows them by 17–39%, depending on its noise, but that takes 90 readings per person instead of 7 or 14. With the number of readings held fixed, placing them in practice rather than at the probes narrows intervals by only 1.5–8%, so most of the gain comes from reading the channel often.

These results use the pilot's illustrative parameters, not estimates from data, and 20–40 replications per cell. The mixture benchmark is given the true learner-type separation and weights. To check the extension, run `python test_ext.py` from `rerun/`.

## Reproducibility and attribution

The public category source files are CC BY 4.0 and retain their original README; attribution is in `data/public_mack/ATTRIBUTION.md`. The Bahg source aggregate is fetched from its original repository rather than redistributed. Earlier private project materials are not included. Output manifests record source addresses and hashes. Numerical packages used for the completed runs are recorded in `outputs/environment.json`.

The new code is a research implementation supplied for review and extension. Dataset licenses and attribution remain those of their creators. No paper result is claimed as a new finding of this project.

## How this was made

Mansi Agarwal designed and built the pilot: the simulator, the fitted models, the prediction check, the public-data audits and the notebook. Claude (Anthropic) reviewed the pilot and wrote the extension, tests, experiments and write-up in `rerun/`.

## License

Code is released under the MIT License (see `LICENSE`). Text, figures and simulation results are released under CC BY 4.0. The files in `data/public_mack/` are © their original creators and remain under CC BY 4.0, with attribution in `data/public_mack/ATTRIBUTION.md`.
