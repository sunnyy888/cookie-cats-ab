"""One-page client summary (HTML -> PDF via headless Chrome). Every number is computed here from cookie.py.

Run:  .venv/bin/python make_summary.py      -> summary.html, summary.pdf, figures/summary_chart.png
"""
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cookie as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
d = C.load()
rt, srm = C.retention_tests(d), C.srm_check(44700, 45489)
b7 = C.bootstrap(d, "retention_7")
aa = C.peeking_aa(d, sims=2000)
r1, r7 = rt["retention_1"], rt["retention_7"]

# chart: retention by gate with 95% CIs
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), dpi=200)
for ax, (key, r, title) in zip(axes, (("retention_1", r1, "Day-1 retention"), ("retention_7", r7, "Day-7 retention"))):
    vals = [r["p_a"], r["p_b"]]
    ses = [((v * (1 - v)) / n) ** 0.5 for v, n in zip(vals, (44700, 45489))]
    ax.bar(["Gate at 30", "Gate at 40"], [100 * v for v in vals], yerr=[196 * s for s in ses], color=["#2f6fdb", "#e0742d"], capsize=4, width=0.6)
    lo = 100 * min(vals) - 2.2
    ax.set_ylim(lo, 100 * max(vals) + 1.4)
    for i, v in enumerate(vals):
        ax.text(i, 100 * v + 196 * ses[i] + 0.12, f"{100 * v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(f"{title}  (p = {r['p']:.3f})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("% of players", fontsize=8)
fig.text(0.5, 0.005, "Bars show 95% confidence intervals. Axes start above zero to make the gap visible.", ha="center", fontsize=6.5, color="#666")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(HERE, "figures", "summary_chart.png"))
plt.close(fig)

lost = -r7["diff"] * 100_000
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cookie Cats gate test: summary</title>
<style>
@page {{ size: letter; margin: 0.55in 0.6in; }}
body {{ font: 10.3pt/1.42 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1b1e24; margin: 0; }}
.kicker {{ font-size: 8pt; letter-spacing: .14em; text-transform: uppercase; color: #6a4bd6; font-weight: 700; }}
h1 {{ font-size: 19pt; margin: 2px 0 2px; line-height: 1.15; }}
.sub {{ color: #5f6570; margin: 0 0 10px; }}
.rec {{ background: #f1edff; border-left: 4px solid #6a4bd6; padding: 9px 12px; border-radius: 0 8px 8px 0; font-size: 11pt; margin: 8px 0 10px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0; }}
.kpi {{ background: #f4f5f8; border-radius: 8px; padding: 7px 9px; }}
.kpi b {{ display: block; font-size: 14.5pt; }} .kpi span {{ font-size: 7.8pt; color: #5f6570; }}
h2 {{ font-size: 11pt; margin: 10px 0 3px; color: #2b2f38; }}
ul {{ margin: 3px 0; padding-left: 16px; }} li {{ margin: 2px 0; }}
img {{ width: 100%; margin: 4px 0 0; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.foot {{ font-size: 7.5pt; color: #7a808a; border-top: 1px solid #e3e5ea; padding-top: 5px; margin-top: 8px; }}
</style></head><body>
<div class="kicker">A/B test readout · Cookie Cats · first gate placement</div>
<h1>Keep the first gate at level 30: moving it to 40 loses players</h1>
<p class="sub">90,189 new players were randomly assigned to a first gate at level 30 (control) or level 40, and tracked for 14 days.</p>
<div class="rec"><b>Recommendation:</b> keep the gate at level 30. Moving it to level 40 cut 7-day retention by <b>{-100 * r7['diff']:.2f} points</b> ({-r7['rel']:.1%} relative), about <b>{lost:,.0f} fewer active players per 100,000 installs</b>, with no offsetting gain anywhere in the data.</div>
<div class="kpis">
 <div class="kpi"><b>{100 * r7['p_a']:.1f}% → {100 * r7['p_b']:.1f}%</b><span>7-day retention, gate 30 → 40</span></div>
 <div class="kpi"><b>p = {r7['p']:.4f}</b><span>two-proportion z-test, day 7</span></div>
 <div class="kpi"><b>{b7['p_gate30_better']:.1%}</b><span>of 10,000 bootstrap resamples favor gate 30</span></div>
 <div class="kpi"><b>{-100_000 * r7['ci'][1]:,.0f}–{-100_000 * r7['ci'][0]:,.0f}</b><span>players lost per 100k installs (95% CI)</span></div>
</div>
<img src="figures/summary_chart.png" alt="Retention by gate">
<div class="cols">
<div>
<h2>What we found</h2>
<ul>
<li><b>Day 7:</b> {100 * r7['p_a']:.2f}% vs {100 * r7['p_b']:.2f}%, a {100 * r7['diff']:+.2f} pt difference, 95% CI [{100 * r7['ci'][0]:+.2f}, {100 * r7['ci'][1]:+.2f}]. Clearly negative.</li>
<li><b>Day 1:</b> {100 * r1['diff']:+.2f} pts (p = {r1['p']:.3f}): the same direction, not significant on its own.</li>
<li><b>Robust:</b> a bootstrap (no normal approximation) gives the same day-7 interval, [{100 * b7['ci'][0]:+.2f}, {100 * b7['ci'][1]:+.2f}] pts.</li>
<li><b>Engagement:</b> median rounds played are similar (17 vs 16). One 49,854-round account looks like a bot and was excluded from averages.</li>
</ul>
</div>
<div>
<h2>Caveats and next steps</h2>
<ul>
<li><b>Uneven split:</b> {100 * srm['share_a']:.2f}% / {100 * (1 - srm['share_a']):.2f}% instead of 50/50 (p = {srm['p']:.3f}). Ask the data team to confirm assignment and logging didn't drop players unevenly.</li>
<li><b>Retention ≠ revenue:</b> gates also drive purchases, which weren't in this data. If level 40 is still on the table for monetization, rerun with purchase data.</li>
<li><b>Don't peek:</b> replaying this data with daily checks and no real effect gave false "wins" in {aa['peek']:.0%} of tests vs {aa['once']:.0%} with one planned look. Fix the sample size and analysis date up front.</li>
</ul>
</div>
</div>
<div class="foot">Data: Cookie Cats A/B test (Tactile Entertainment; public Kaggle dataset), 90,189 players. Methods: two-proportion z-tests (pooled SE), unpooled 95% CIs, 10,000-sample bootstrap, chi-square sample-ratio test, permutation-based peeking simulation (2,000 A/A replays). Full analysis: cookie_cats_analysis.ipynb.</div>
</body></html>"""
open(os.path.join(HERE, "summary.html"), "w").write(html)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.path.join(HERE, 'summary.pdf')}",
                "file://" + os.path.join(HERE, "summary.html")], check=True, stderr=subprocess.DEVNULL)
print("wrote summary.html and summary.pdf")
