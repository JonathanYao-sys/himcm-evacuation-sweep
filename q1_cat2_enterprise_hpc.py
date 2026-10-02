"""Q1 Category 2: Enterprise + Supercomputers (U-PUE model)."""
# E_full = 8760 * sum(N * P_peak * PUE)
# E_avg = 8760 * sum(N * P_peak * PUE * [k_idle + (1-k_idle)*u])

def calc(it_mw, pue, u, k_idle=0.35):
    e_full = it_mw * 8760 / 1e6 * pue
    factor = k_idle + (1 - k_idle) * u
    return e_full, e_full * factor

SCENARIOS = {
    # 2a enterprise: N sites * MW/site = IT MW
    "2a-low":  dict(it_mw=7000*0.8,  pue=1.65, u=0.20, k_idle=0.40),
    "2a-base": dict(it_mw=8000*1.2,  pue=1.80, u=0.25, k_idle=0.40),
    "2a-high": dict(it_mw=9000*1.5,  pue=2.00, u=0.30, k_idle=0.45),
    # 2b supercomputers: Top500 + long tail
    "2b-low":  dict(it_mw=2500, pue=1.3, u=0.60, k_idle=0.30),
    "2b-base": dict(it_mw=4000, pue=1.4, u=0.70, k_idle=0.30),
    "2b-high": dict(it_mw=6000, pue=1.5, u=0.80, k_idle=0.30),
}

if __name__ == "__main__":
    res = {}
    for k, p in SCENARIOS.items():
        f, a = calc(**p)
        res[k] = (f, a)
        print(f"{k}: IT={p['it_mw']:.0f}MW PUE={p['pue']} u={p['u']} -> E_full={f:.1f} TWh E_avg={a:.1f} TWh")
    for level in ["low", "base", "high"]:
        f = res[f"2a-{level}"][0] + res[f"2b-{level}"][0]
        a = res[f"2a-{level}"][1] + res[f"2b-{level}"][1]
        print(f"TOTAL-{level}: E_full_2={f:.1f} TWh E_avg_2={a:.1f} TWh")
