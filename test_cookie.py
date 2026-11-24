"""Tests for the Cookie Cats analysis. Run: .venv/bin/python test_cookie.py"""
import json
import math
import os

import numpy as np

import cookie as C

HERE = os.path.dirname(os.path.abspath(__file__))
ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, (": " + detail) if detail else ""))


d = C.load()
# 1. the data is the real public dataset (published figures)
counts = d.version.value_counts().to_dict()
check("90,189 unique players, split 44,700 / 45,489", len(d) == 90189 and d.userid.nunique() == 90189 and counts == {"gate_40": 45489, "gate_30": 44700})
r = C.retention_tests(d)
check("retention matches published figures (44.82/44.23%, 19.02/18.20%)",
      [round(100 * r[c][k], 2) for c in ("retention_1", "retention_7") for k in ("p_a", "p_b")] == [44.82, 44.23, 19.02, 18.2])

# 2. z-test by hand (day 7)
xa, na, xb, nb = int(d[d.version == "gate_30"].retention_7.sum()), 44700, int(d[d.version == "gate_40"].retention_7.sum()), 45489
pa, pb, pool = xa / na, xb / nb, (xa + xb) / (na + nb)
z = (pb - pa) / math.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
check("day-7 z-test matches a hand calculation", abs(r["retention_7"]["z"] - z) < 1e-12 and abs(r["retention_7"]["p"] - math.erfc(abs(z) / math.sqrt(2))) < 1e-12, f"z = {z:.4f}, p = {r['retention_7']['p']:.5f}")
check("day-7 effect significant, day-1 not", r["retention_7"]["p"] < 0.01 and r["retention_1"]["p"] > 0.05)

# 3. SRM
s = C.srm_check(44700, 45489)
check("SRM chi-square = 2 x 394.5^2 / 45,094.5 = 6.90, p ~ 0.0086", abs(s["chi2"] - 2 * 394.5 ** 2 / 45094.5) < 1e-9 and abs(s["p"] - 0.0086) < 2e-4, f"chi2 {s['chi2']:.3f}, p {s['p']:.4f}")
check("SRM test gives p ~ 1 for a perfect split", C.srm_check(45000, 45000)["p"] > 0.999)

# 4. bootstrap agrees with the normal CI; the binomial shortcut agrees with literal row resampling
b = C.bootstrap(d, "retention_7", reps=20000)
check("bootstrap 95% CI ≈ normal CI (within 0.05 pts)", all(abs(b["ci"][i] - r["retention_7"]["ci"][i]) < 5e-4 for i in (0, 1)), f"{[round(100 * v, 3) for v in b['ci']]} vs {[round(100 * v, 3) for v in r['retention_7']['ci']]}")
br = C.bootstrap(d, "retention_7", reps=1500, rows=True, seed=5)
check("row-resampling bootstrap has the same spread as the shortcut", abs(br["diff"].std() / b["diff"].std() - 1) < 0.08, f"sd {br['diff'].std():.5f} vs {b['diff'].std():.5f}")

# 5. peeking: A/A with one look holds 5%; daily peeking inflates it
aa = C.peeking_aa(d, sims=2000)
se = math.sqrt(0.05 * 0.95 / 2000)
check("A/A, one look at the end: ~5% false positives", abs(aa["once"] - 0.05) < 3 * se, f"{aa['once']:.3f}")
check("A/A, 14 daily peeks: much higher (15-30%)", 0.15 < aa["peek"] < 0.30, f"{aa['peek']:.3f}")
check("peeking curve is cumulative", all(a <= b for a, b in zip(aa["cum"], aa["cum"][1:])))
pr = C.peeking_real(d, orders=200)
check("real data: every replay is significant once all players are in", pr["share_sig_by_day"][-1] == 1.0 and pr["share_sig_by_day"][0] < 0.4)

# 6. deliverables exist, ran cleanly, and quote the computed numbers
nb = json.load(open(os.path.join(HERE, "cookie_cats_analysis.ipynb")))
errors = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
check("notebook executed with outputs and no errors", not errors and all(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"))
html = open(os.path.join(HERE, "summary.html")).read()
want = [f"{100 * r['retention_7']['p_a']:.1f}% → {100 * r['retention_7']['p_b']:.1f}%", f"p = {r['retention_7']['p']:.4f}", f"{-r['retention_7']['diff'] * 100000:,.0f} fewer active players"]
check("1-page summary quotes the computed results", all(w in html for w in want), " | ".join(want))
import re
check("summary.pdf is one page", re.search(r"^Pages:\s+1$", os.popen("pdfinfo " + os.path.join(HERE, "summary.pdf")).read(), re.M) is not None)
print("\n%d/%d PASS" % (sum(ok), len(ok)))
raise SystemExit(0 if all(ok) else 1)
