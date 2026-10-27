# Implemented model and estimands

## Continuing state and persistent type

For participant i, let S_it = (a_it,b_it) belong to {00,10,01,11}. U_i is a persistent binary type, with equal population weights. It is not redrawn each trial. Activities E_it are A, B or AB. Responses are observed before the activity's learning transition.

The four base gains are ordered (a under A, a under AB, b under B, b under AB). Type-specific gains are

\[
g_{k,e}(u)=\operatorname{logit}^{-1}(\operatorname{logit}(g_{k,e})+u h),\quad u\in\{-1,+1\}.
\]

An unmastered skill can be acquired only when the activity practices it. A mastered skill can be lost during any practice activity with its skill-specific loss probability. Conditional on type and starting state, the two skill transitions are independent. This restriction is part of the prototype, not a general learning theory. No simultaneous loss and reacquisition occurs within one activity.

## Behavioral and auxiliary observations

For the activity's requirement vector q_e, define eta(s,e)=1 when all required skills are mastered. The response probability is

\[
P(Y_{it}=1\mid S_{it}=s,E_{it}=e)=g+(1-g-\ell)\eta(s,e),
\]

where g is guessing and ell is slipping. Both are estimated, constrained below .5. There are no item-specific effects in this pilot.

The fitted joint model uses

\[
Z_{it}\mid S_{it}=s \sim N\{c+L(a+b),\sigma^2\},\quad L>0,
\]

on the first event in each nine-response assessment block. The intercept, loading and standard deviation are jointly estimated. Responses and Z are conditionally independent given S. States 10 and 01 have the same Z distribution, so the sensor alone cannot distinguish the skills. In the misspecification condition, the generator adds a time trend .75 times normalized event position and a persistent participant offset with SD .6. Neither term is in the fitted observation model. The latent process and behavioral observations are paired across the clean and perturbed sensor conditions.

This signal is deliberately abstract. It has no haemodynamic convolution, tracer kinetics, spatial neural representation or physiological interpretation. No synthetic result is evidence that MRI, fMRI, EEG or PET has this observation law.

## History policy

Only practice responses enter the algorithm's retained history. Recency retains the most recent 18. The balanced rule retains up to six recent observations of each requirement and fills unused capacity with the most recent remaining observations. On adaptive trials, with retained counts n_e and correct counts c_e,

\[
w_e=1-\frac{1+c_e}{2+n_e},\qquad
\pi_t(e\mid H_t)=\frac{.10}{3}+.90\frac{w_e}{\sum_j w_j}.
\]

Each 12-trial block also has three scheduled exploration positions with one A, B and AB in randomized order. Initial/scheduled actions are conditioned on the pre-randomized schedule; their stored probability of one is conditional on that schedule, not their unconditional randomization probability. Actual adaptive-trial chosen-action probabilities are logged. Research records retain all events, including those no longer in the retained history.

## Sequential likelihood

The model sums over state paths and one persistent participant type. In abbreviated notation,

\[
L_i(\theta)=\sum_u p(u)\sum_{s_{1:T+1}}p_\theta(s_1)
\prod_{t=1}^T O_\theta(y_{it},z_{it}\mid s_t,e_{it})
T_\theta(s_{t+1}\mid s_t,e_{it},u).
\]

Assessment events have identity transitions. The overnight event has a loss transition but no response. The policy factor, a product of known probabilities determined by the fully observed past, is constant in the fitted parameters and can be omitted. This criterion is a sequential likelihood factor, not the ordinary response distribution conditional on all future adaptive actions. If real assignment depends on unrecorded signals, the omission need not be justified.

## Common target population

The two estimands are standardized over the same equal-type reference population:

\[
\Delta_a=\tfrac12\sum_{u\in\{-1,1\}}[g_{a,AB}(u)-g_{a,A}(u)],\qquad
\Delta_b=\tfrac12\sum_u[g_{b,AB}(u)-g_{b,B}(u)].
\]

The latent starting state is set to the relevant unmastered state conceptually, while U retains the reference distribution. This differs from averaging over the policy-dependent subset still unmastered at a particular time. The simulator's independent-skill transition restriction makes the gain independent of the other skill's mastery. That restriction must be revisited if transfer is scientifically important.

The specified-mixture fits fix h=.8 and equal weights; homogeneous fits fix h=0. Overnight losses are fixed at .15. Other listed parameters are fitted. Thus the present mixture comparison is an optimistic benchmark, not a recovery test for unknown population heterogeneity or retention. The starting mastery distribution factorizes across the two skills.

## Uncertainty and prediction

Optimization uses two starts for completed runs. A central-difference Hessian of the full negative log likelihood supplies observed information; the L-BFGS inverse approximation is not used for reported intervals. A delta method propagates uncertainty to both gain contrasts. Participant score vectors yield a sandwich alternative. Regular Wald intervals are refused if optimization fails, a parameter is within .02 of a numerical bound, the Hessian is nonpositive, or its condition number exceeds 1e9. A successful check is not a proof of identification or global convergence.

Monte Carlo results retain every completed fit and both valid-interval and valid-and-covered denominators. Exact binomial bounds accompany coverage because a zero plug-in MCSE at 12/12 coverage is misleading. The two sensor scenarios share behavior-only fits and are not independent replications for that estimator.

Held-out evaluation freezes all fitted parameters and filters a new cohort. Current responses are predicted before incorporating the current auxiliary value; previous held-out responses update the state distribution. This evaluates one-step prediction under an explicitly generated test policy, not an open-loop long-horizon forecast or transport to a real clinical population.
