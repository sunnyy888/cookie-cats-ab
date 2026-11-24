"""Cookie Cats A/B test: analysis functions (used by the notebook, the summary and the tests).

Data: data/cookie_cats.csv, 90,189 players who installed Cookie Cats during the test. Columns:
    userid, version (gate_30 = control, gate_40 = treatment), sum_gamerounds (rounds in the first 14 days),
    retention_1 / retention_7 (came back and played 1 / 7 days after installing).
"""
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026


def load(path=None):
    d = pd.read_csv(path or os.path.join(HERE, "data", "cookie_cats.csv"))
    for c in ("retention_1", "retention_7"):
        d[c] = d[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)
    return d


def norm_sf(z):
    """P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_sf_1df(x):
    """P(chi-square with 1 df > x) = 2 P(Z > sqrt(x))."""
    return 2 * norm_sf(math.sqrt(x))


def srm_check(n_a, n_b, expected_share=0.5):
    """Sample ratio mismatch: chi-square goodness of fit of the split against the planned share."""
    n = n_a + n_b
    ea, eb = n * expected_share, n * (1 - expected_share)
    chi2 = (n_a - ea) ** 2 / ea + (n_b - eb) ** 2 / eb
    return {"n_a": n_a, "n_b": n_b, "share_a": n_a / n, "chi2": chi2, "p": chi2_sf_1df(chi2)}


def two_prop(x_a, n_a, x_b, n_b, conf=0.95):
    """B minus A: pooled z-test (two-sided) and unpooled confidence interval for the difference."""
    pa, pb = x_a / n_a, x_b / n_b
    pool = (x_a + x_b) / (n_a + n_b)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pool
    se_un = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    zc = {0.95: 1.959963984540054, 0.9: 1.6448536269514722, 0.99: 2.5758293035489004}[conf]
    diff = pb - pa
    return {"p_a": pa, "p_b": pb, "diff": diff, "rel": diff / pa, "z": z, "p": 2 * norm_sf(abs(z)),
            "se_pool": se_pool, "se_unpooled": se_un, "ci": (diff - zc * se_un, diff + zc * se_un)}


def retention_tests(d):
    out = {}
    a, b = d[d.version == "gate_30"], d[d.version == "gate_40"]
    for c in ("retention_1", "retention_7"):
        out[c] = two_prop(int(a[c].sum()), len(a), int(b[c].sum()), len(b))
    return out


def bootstrap(d, col, reps=10_000, seed=SEED, rows=False):
    """Bootstrap the gate_40 - gate_30 difference in retention.

    rows=False draws each group's resampled retained count from Binomial(n, p_hat), which has exactly the
    distribution of the count in a with-replacement resample of n rows. rows=True resamples rows literally.
    """
    rng = np.random.default_rng(seed)
    ya = d.loc[d.version == "gate_30", col].to_numpy()
    yb = d.loc[d.version == "gate_40", col].to_numpy()
    if rows:
        ma = np.array([ya[rng.integers(0, len(ya), len(ya))].mean() for _ in range(reps)])
        mb = np.array([yb[rng.integers(0, len(yb), len(yb))].mean() for _ in range(reps)])
    else:
        ma = rng.binomial(len(ya), ya.mean(), reps) / len(ya)
        mb = rng.binomial(len(yb), yb.mean(), reps) / len(yb)
    diff = mb - ma
    return {"diff": diff, "ci": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
            "p_gate30_better": float((diff < 0).mean()), "mean_a": ma, "mean_b": mb}


def peeking_aa(d, col="retention_7", days=14, sims=2000, seed=SEED):
    """A/A peeking on this data: shuffle the version labels (so there is truly no effect), let players
    arrive in random order over `days` equal daily batches, and run the two-sided z-test after each day.
    Returns the false-positive rate when stopping at the first p < 0.05 vs checking once at the end."""
    rng = np.random.default_rng(seed)
    y = d[col].to_numpy()
    n_b = int((d.version == "gate_40").sum())
    n = len(y)
    cuts = np.linspace(0, n, days + 1).astype(int)[1:]
    first = np.full(sims, -1)
    final_sig = np.zeros(sims, bool)
    for s in range(sims):
        order = rng.permutation(n)
        lab = np.zeros(n, bool)
        lab[rng.choice(n, n_b, replace=False)] = True              # random gate_40 labels: an A/A test
        yy, ll = y[order], lab[order]
        cb, nb = np.cumsum(yy * ll)[cuts - 1], np.cumsum(ll)[cuts - 1]
        ca, na = np.cumsum(yy * ~ll)[cuts - 1], np.cumsum(~ll)[cuts - 1]
        pa, pb, pool = ca / na, cb / nb, (ca + cb) / (na + nb)
        z = (pb - pa) / np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
        sig = np.abs(z) > 1.959963984540054
        if sig.any():
            first[s] = int(np.argmax(sig))
        final_sig[s] = sig[-1]
    cum = np.array([(first >= 0) & (first <= k) for k in range(days)]).mean(axis=1)
    return {"peek": float((first >= 0).mean()), "once": float(final_sig.mean()), "cum": cum.tolist(), "days": days, "sims": sims}


def peeking_real(d, col="retention_7", days=14, orders=500, seed=SEED):
    """The real experiment replayed in random arrival orders: on which day would daily peeking first have
    shown p < 0.05, and how often would an early look have been inconclusive?"""
    rng = np.random.default_rng(seed)
    y = d[col].to_numpy()
    lab = (d.version == "gate_40").to_numpy()
    n = len(y)
    cuts = np.linspace(0, n, days + 1).astype(int)[1:]
    sig_by_day = np.zeros(days)
    paths = []
    for s in range(orders):
        o = rng.permutation(n)
        yy, ll = y[o], lab[o]
        cb, nb = np.cumsum(yy * ll)[cuts - 1], np.cumsum(ll)[cuts - 1]
        ca, na = np.cumsum(yy * ~ll)[cuts - 1], np.cumsum(~ll)[cuts - 1]
        pa, pb, pool = ca / na, cb / nb, (ca + cb) / (na + nb)
        z = (pb - pa) / np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
        p = np.array([2 * norm_sf(abs(v)) for v in z])
        sig_by_day += p < 0.05
        if s < 25:
            paths.append(p.tolist())
    return {"share_sig_by_day": (sig_by_day / orders).tolist(), "paths": paths, "days": days, "orders": orders}


def engagement(d, cap_outlier=True):
    """Rounds played in the first 14 days by group (median and trimmed mean); drops the single 49,854-round player."""
    x = d[d.sum_gamerounds < 40_000] if cap_outlier else d
    g = x.groupby("version").sum_gamerounds
    return {"median": g.median().to_dict(), "mean": g.mean().round(2).to_dict(), "zero_rounds": int((d.sum_gamerounds == 0).sum()),
            "outlier_rounds": int(d.sum_gamerounds.max())}
