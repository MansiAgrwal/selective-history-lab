"""Summaries for Experiments 2 and 3, with Monte Carlo uncertainty."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

RES = Path(__file__).resolve().parent / 'results'
CONTRASTS = ('a: AB minus A', 'b: AB minus B')


def exact_ci(k, n):
    lo = beta.ppf(.025, k, n - k + 1) if k > 0 else 0.
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.
    return float(lo), float(hi)


def flatten(rows, keys):
    flat = []
    for r in rows:
        for j, name in enumerate(CONTRASTS):
            q = {k: r[k] for k in keys}
            q.update(contrast=name, truth=r['truth'][j], estimate=r['estimate'][j], error=r['estimate'][j] - r['truth'][j],
                     valid=bool(r['interval_valid']))
            for kind in ('model', 'sandwich'):
                ci = r['ci_' + kind] if r['interval_valid'] else None
                q['cover_' + kind] = float(ci[j][0] <= r['truth'][j] <= ci[j][1]) if ci else np.nan
                q['width_' + kind] = ci[j][1] - ci[j][0] if ci else np.nan
                q['excludes_zero_' + kind] = float(ci[j][1] < 0 or ci[j][0] > 0) if ci else np.nan
            flat.append(q)
    return pd.DataFrame(flat)


def describe(g):
    n = len(g); nv = int(g.valid.sum()); k = int(g.cover_model.sum())
    lo, hi = exact_ci(k, nv) if nv else (np.nan, np.nan)
    return pd.Series(dict(reps=n, truth=g.truth.iloc[0], bias=g.error.mean(), bias_mcse=g.error.std(ddof=1) / np.sqrt(n),
                          rmse=np.sqrt(np.mean(g.error ** 2)), sd=g.estimate.std(ddof=1), valid_intervals=nv,
                          coverage_model=k / nv if nv else np.nan, coverage_model_mc95=f'[{lo:.2f}, {hi:.2f}]',
                          coverage_sandwich=g.cover_sandwich.mean(), mean_width_model=g.width_model.mean(),
                          share_excluding_zero=g.excludes_zero_model.mean(),
                          wrong_sign_share=float(np.mean(np.sign(g.estimate) != np.sign(g.truth)))))


def experiment2(max_reps):
    rows = [json.loads(l) for l in (RES / 'e2_fits.jsonl').read_text().splitlines()]
    df = flatten(rows, ('rep', 'h', 'gain_scale', 'model'))
    complete = df.groupby('rep').size()
    keep = [r for r, n in complete.items() if n == complete.max() and r < max_reps]
    df = df[df.rep.isin(keep)]
    out = df.groupby(['gain_scale', 'h', 'model', 'contrast']).apply(describe, include_groups=False).reset_index()
    out.to_csv(RES / 'e2_summary.csv', index=False)
    return out, len(keep)


def experiment3():
    rows = [json.loads(l) for l in (RES / 'e3_fits.jsonl').read_text().splitlines()]
    df = flatten(rows, ('rep', 'condition', 'sensor', 'probe_items', 'sensor_sd'))
    complete = df.groupby('rep').condition.nunique()
    df = df[df.rep.isin(complete[complete == complete.max()].index)]
    out = df.groupby(['condition', 'contrast'], sort=False).apply(describe, include_groups=False).reset_index()
    # Paired precision relative to the pilot design (behaviour only, 9 probe items), same learners in every condition
    base = df[df.condition == 'behaviour, 9 probe items'].set_index(['rep', 'contrast']).width_model
    ratio = df.set_index(['rep', 'contrast']).assign(ratio=lambda d: d.width_model / base.reindex(d.index).values).reset_index()
    r = ratio.groupby(['condition', 'contrast'], sort=False).ratio.agg(['mean', lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())])
    r.columns = ['width_ratio_vs_pilot', 'width_ratio_mcse']
    out = out.merge(r.reset_index(), on=['condition', 'contrast'])
    # Information (1 / width^2) is close to linear in the number of probe items; invert that line for each sensor
    eq = []
    for c in CONTRASTS:
        probes = out[(out.contrast == c) & out.condition.str.startswith('behaviour')]
        items = probes.condition.str.extract(r'(\d+) probe')[0].astype(float).to_numpy()
        info = 1 / probes.mean_width_model.to_numpy() ** 2
        slope, intercept = np.polyfit(items, info, 1)
        for _, row in out[(out.contrast == c) & ~out.condition.str.startswith('behaviour')].iterrows():
            e = (1 / row.mean_width_model ** 2 - intercept) / slope
            eq.append(dict(condition=row.condition, contrast=c, equivalent_probe_items_per_block=e, extrapolated=bool(e > 27)))
    out = out.merge(pd.DataFrame(eq), on=['condition', 'contrast'], how='left')
    out.to_csv(RES / 'e3_summary.csv', index=False)
    return out, int(df.rep.nunique())


if __name__ == '__main__' and sys.argv[1] in ('e2', 'e3'):
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    which = sys.argv[1]
    if which == 'e2':
        out, n = experiment2(int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6)
        print(f'Experiment 2: {n} complete replications per cell')
        print(out[['gain_scale', 'h', 'model', 'contrast', 'truth', 'bias', 'bias_mcse', 'rmse', 'coverage_model', 'coverage_model_mc95',
                   'coverage_sandwich', 'mean_width_model', 'share_excluding_zero', 'wrong_sign_share']].round(4).to_string(index=False))
    else:
        out, n = experiment3()
        print(f'Experiment 3: {n} complete replications per condition')
        print(out[['condition', 'contrast', 'bias', 'bias_mcse', 'rmse', 'coverage_model', 'coverage_model_mc95', 'mean_width_model',
                   'width_ratio_vs_pilot', 'width_ratio_mcse', 'share_excluding_zero', 'equivalent_probe_items_per_block']].round(4).to_string(index=False))


def experiment3c():
    """Trial-level sensor vs the pilot design, paired on the same replications (Experiment 3 baseline rows)."""
    base_rows = [json.loads(l) for l in (RES / 'e3_fits.jsonl').read_text().splitlines()]
    rows = [json.loads(l) for l in (RES / 'e3c_fits.jsonl').read_text().splitlines()]
    reps = sorted({r['rep'] for r in rows})
    keep = [r for r in base_rows if r['rep'] in reps and r['condition'] in ('behaviour, 9 probe items', 'skill sensor, SD 0.4', 'total sensor, SD 0.8')]
    df = flatten(keep + rows, ('rep', 'condition', 'sensor', 'probe_items', 'sensor_sd'))
    complete = df.groupby('rep').condition.nunique(); df = df[df.rep.isin(complete[complete == complete.max()].index)]
    out = df.groupby(['condition', 'contrast'], sort=False).apply(describe, include_groups=False).reset_index()
    base = df[df.condition == 'behaviour, 9 probe items'].set_index(['rep', 'contrast']).width_model
    ratio = df.set_index(['rep', 'contrast']).assign(ratio=lambda d: d.width_model / base.reindex(d.index).values).reset_index()
    r = ratio.groupby(['condition', 'contrast'], sort=False).ratio.agg(['mean', lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())])
    r.columns = ['width_ratio_vs_pilot', 'width_ratio_mcse']
    out = out.merge(r.reset_index(), on=['condition', 'contrast'])
    out.to_csv(RES / 'e3c_summary.csv', index=False)
    return out, int(df.rep.nunique())


if __name__ == '__main__' and sys.argv[1] == 'e3c':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    out, n = experiment3c()
    print(f'Experiment 3c: {n} complete replications per condition')
    print(out[['condition', 'contrast', 'bias', 'bias_mcse', 'rmse', 'coverage_model', 'mean_width_model', 'width_ratio_vs_pilot',
               'width_ratio_mcse', 'share_excluding_zero']].round(4).to_string(index=False))
