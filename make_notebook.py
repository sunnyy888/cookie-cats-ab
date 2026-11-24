"""Build, execute and export the analysis notebook.

Run:  .venv/bin/python make_notebook.py      -> cookie_cats_analysis.ipynb (with outputs) + .html
"""
import os

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Cookie Cats: should the first gate be at level 30 or level 40?

**Cookie Cats** (Tactile Entertainment) is a puzzle game. As players progress they hit *gates* that make them wait, or pay, to continue. In this A/B test, new players were randomly assigned to a first gate at **level 30** (control, `gate_30`) or **level 40** (`gate_40`). We have 90,189 players and whether each came back **1 day** and **7 days** after installing.

**Question:** does moving the gate to level 40 change retention, and what should the studio do?

**Plan:**
1. Data check
2. Was the split clean?
3. Retention tests with CIs
4. Bootstrap
5. What daily peeking would have done
6. Recommendation

All functions live in `cookie.py`, and `test_cookie.py` tests them."""),
code("""import cookie as C
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
d = C.load()
print(f"{len(d):,} players, {d.userid.nunique():,} unique ids")
d.head()"""),
md("## 1. Data check"),
code("""summary = d.groupby("version").agg(players=("userid", "size"), day1=("retention_1", "mean"), day7=("retention_7", "mean"),
                                    median_rounds=("sum_gamerounds", "median"))
eng = C.engagement(d)
print(f"Players who installed but never played a round: {eng['zero_rounds']:,}")
print(f"Largest sum_gamerounds: {eng['outlier_rounds']:,} (one player; next largest is {int(d.sum_gamerounds.nlargest(2).iloc[1]):,})")
summary.style.format({"day1": "{:.2%}", "day7": "{:.2%}"})"""),
code("""fig, ax = plt.subplots(figsize=(7, 3.2))
x = d[d.sum_gamerounds < 200]
for v, c in (("gate_30", "#2f6fdb"), ("gate_40", "#e0742d")):
    ax.hist(x[x.version == v].sum_gamerounds, bins=100, alpha=.55, color=c, label=v)
ax.set(xlabel="game rounds in the first 14 days (players under 200 rounds)", ylabel="players", title="Most players play only a few rounds")
ax.legend(); plt.show()"""),
md("""One player logged 49,854 rounds in two weeks, which is almost surely a bot or a logging error. It doesn't affect the retention flags, but it would distort any average of `sum_gamerounds`, so engagement summaries below drop it. Rounds played are *after* assignment (the gate itself can change them), so they're an outcome, not a balance check."""),
md("## 2. Was the split clean? Sample ratio mismatch"),
code("""srm = C.srm_check(44700, 45489)
print(f"gate_30 share: {srm['share_a']:.2%} (planned 50%)")
print(f"chi-square = {srm['chi2']:.2f}, p = {srm['p']:.4f}")"""),
md("""The split is 49.56% / 50.44%. With 90,000 players that's a **statistically significant imbalance** (p = 0.009): a *sample ratio mismatch*. Random assignment should land much closer to 50/50.

An SRM doesn't prove the result is wrong, but it's a warning that assignment or logging may have dropped some players unevenly. The data has no pre-assignment covariates (device, country, install date) to check balance, so **this is a caveat to carry into the recommendation**, and something to ask the data team about."""),
md("## 3. Retention: two-proportion z-tests and 95% CIs (gate_40 − gate_30)"),
code("""rt = C.retention_tests(d)
pd.DataFrame({k: {"gate_30": v["p_a"], "gate_40": v["p_b"], "difference (pts)": 100 * v["diff"], "relative": v["rel"],
                  "z": v["z"], "p (two-sided)": v["p"], "95% CI low (pts)": 100 * v["ci"][0], "95% CI high (pts)": 100 * v["ci"][1]}
              for k, v in rt.items()}).T.style.format({"gate_30": "{:.2%}", "gate_40": "{:.2%}", "difference (pts)": "{:+.2f}",
              "relative": "{:+.1%}", "z": "{:.2f}", "p (two-sided)": "{:.4f}", "95% CI low (pts)": "{:+.2f}", "95% CI high (pts)": "{:+.2f}"})"""),
md("""- **Day 1:** gate_40 is 0.59 points lower (44.23% vs 44.82%), p = 0.074. The CI [−1.24, +0.06] includes zero: **not significant at 5%**.
- **Day 7:** gate_40 is **0.82 points lower** (18.20% vs 19.02%), a **4.3% relative drop**, p = 0.0016. The CI is [−1.33, −0.31], entirely below zero."""),
md("## 4. Bootstrap: the same question without the normal approximation"),
code("""B = {c: C.bootstrap(d, c, reps=10_000) for c in ("retention_1", "retention_7")}
check = C.bootstrap(d, "retention_7", reps=1000, rows=True)   # literal row resampling, as a check on the shortcut
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (c, b) in zip(axes, B.items()):
    ax.hist(100 * b["diff"], bins=60, color="#6a4bd6", alpha=.8)
    ax.axvline(0, color="k", lw=1); [ax.axvline(100 * v, color="#c9413a", ls="--", lw=1) for v in b["ci"]]
    ax.set(title=f"{c.replace('_', ' ')}: gate_40 − gate_30", xlabel="difference in retention (pts)")
plt.tight_layout(); plt.show()
for c, b in B.items():
    print(f"{c}: bootstrap 95% CI [{100*b['ci'][0]:+.2f}, {100*b['ci'][1]:+.2f}] pts; P(gate_30 retains better) = {b['p_gate30_better']:.1%}")
print(f"row-resampling check (1,000 reps), day 7: CI [{100*check['ci'][0]:+.2f}, {100*check['ci'][1]:+.2f}] pts")"""),
md("""The bootstrap agrees with the z-test. In **99.9%** of resamples gate_30 has the higher 7-day retention; for day 1 it's 96%. Resampling each group's retained count from Binomial(n, p̂) is exactly the distribution of a with-replacement row resample, and the literal row-resampling check gives the same interval."""),
md("""## 5. What if we had peeked every day?

The data has no timestamps, so we replay the test with players arriving in random order over 14 equal "days".

**(a) An A/A version.** Shuffle the gate labels so there's truly no effect, then run the day-7 retention test after every day."""),
code("""aa = C.peeking_aa(d, days=14, sims=2000)
print(f"A/A, stop at the first p < 0.05 across 14 daily looks: {aa['peek']:.1%} false positives")
print(f"A/A, check once at the end: {aa['once']:.1%}")
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(1, 15), 100 * np.array(aa["cum"]), "o-", color="#c9413a", label="peek daily, stop at first p < 0.05")
ax.axhline(100 * aa["once"], color="#2f6fdb", ls="--", label="check once at the end"); ax.axhline(5, color="gray", lw=.8)
ax.set(xlabel="day", ylabel="% of A/A tests declared significant", title="Peeking on Cookie Cats data with no real effect"); ax.legend(); plt.show()"""),
md("""**(b) The real experiment.** Replay it in 500 random arrival orders and ask how often each daily look would have shown p < 0.05."""),
code("""pr = C.peeking_real(d, days=14, orders=500)
fig, ax = plt.subplots(figsize=(7, 3))
for p in pr["paths"]: ax.plot(range(1, 15), p, color="#6a4bd6", alpha=.25, lw=1)
ax.axhline(0.05, color="#c9413a", ls="--"); ax.set_yscale("log")
ax.set(xlabel="day", ylabel="p-value (day-7 retention)", title="25 replays of the real test: the p-value wanders before settling"); plt.show()
print("share of replays significant on day 1, 3, 7, 14:", [f"{pr['share_sig_by_day'][k]:.0%}" for k in (0, 2, 6, 13)])"""),
md("""Two lessons:

- **(a) Peeking inflates false positives.** With no real effect, daily peeking "finds" one about 4× as often as a single planned look (about 20% vs about 5%).
- **(b) Early looks are unreliable either way.** Even though the effect here is real, an early look at this data is a coin flip: only about 14% of day-1 looks and about 60% of day-7 looks are significant. Deciding the sample size in advance (as this test did) is what makes the final answer trustworthy."""),
md("## 6. Recommendation"),
code("""r7 = rt["retention_7"]
lost_per_100k = -r7["diff"] * 100_000
print(f"Moving the gate to level 40 lowers 7-day retention by {-100*r7['diff']:.2f} pts ({-r7['rel']:.1%} relative)")
print(f"= about {lost_per_100k:,.0f} fewer players still playing after a week, per 100,000 installs (95% CI {-100_000*r7['ci'][1]:,.0f} to {-100_000*r7['ci'][0]:,.0f})")"""),
md("""**Keep the first gate at level 30.**

- **7-day retention:** moving the gate to level 40 cuts it by 0.82 points (4.3% relative, p = 0.002). That's about **820 fewer players still playing after a week per 100,000 installs** (95% CI: 310–1,330).
- **Day-1 retention:** it moves the same direction but isn't significant.
- **No benefit:** nothing in the data favors level 40.
- **Why it might work this way:** an earlier break may keep the game from being binged and burned through, a hedonic-adaptation story. The test shows the effect, not the mechanism.

**Caveats:**
1. **The split is imbalanced** (SRM, p = 0.009). Confirm that assignment and logging didn't drop players unevenly before treating the size of the effect as exact.
2. **Retention isn't revenue.** Gates also drive purchases, which this data doesn't include.
3. **Only one cohort of new players**, measured over 14 days.

**Next step:** if level 40 is still under consideration for monetization reasons, rerun with purchase data, a verified 50/50 split, and a pre-registered sample size and analysis date."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
nbf.write(nb, os.path.join(HERE, "cookie_cats_analysis.ipynb"))
html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
open(os.path.join(HERE, "cookie_cats_analysis.html"), "w").write(html)
print("wrote cookie_cats_analysis.ipynb and .html")
