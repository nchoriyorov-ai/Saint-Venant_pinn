"""
study.py - experiment framework (scenarios, training, evaluation, diagnostics)
"""
import json, time, os
import numpy as np
from scipy.stats import qmc
import sv_ref as sv
import pinn_np as pn

P = sv.PARAMS
S0 = sv.bed_slope(P["B"], P["n"], P["h0"], P["Q0"])
L, T, B = P["L"], P["T"], P["B"]
CASES = {
    "A": dict(dQ=8.0, tau=10.0, dh=0.2, label="gate opening: Q 20 -> 28 m3/s, tau = 10 s"),
    "B": dict(dQ=-6.0, tau=20.0, dh=0.2, label="gate closing: Q 20 -> 14 m3/s, tau = 20 s"),
}
ARCH = [2, 40, 40, 40, 40, 2]
T_TRAIN = 56.0           # data window of scenario S1 (70 % of the horizon)

_ref_cache = {}


def reference(case):
    if case not in _ref_cache:
        c = CASES[case]
        sol = sv.solve(N=2000, case=dict(dQ=c["dQ"], tau=c["tau"]))
        _ref_cache[case] = (sol, sv.make_interp(sol))
    return _ref_cache[case]


def scales(case):
    c = CASES[case]
    return pn.Scales(L, T, B, P["n"], S0, P["h0"], P["Q0"], dh=c["dh"], dQ=abs(c["dQ"]))


def q_in(case):
    c = CASES[case]
    return lambda t: sv.q_inflow(np.asarray(t), P["Q0"], c["dQ"], c["tau"])


def test_grid():
    x = np.linspace(0, L, 201)
    t = np.linspace(0, T, 161)
    XX, TT = np.meshgrid(x, t)           # (nt, nx)
    return x, t, XX.ravel(), TT.ravel()


# ----------------------------------------------------------------------------
def make_data(scenario, case, seed):
    """Returns (x,t,h,Q) training observations."""
    sol, f = reference(case)
    rng = np.random.default_rng(1000 + seed)
    if scenario == "S1":          # dense noise-free random samples, t <= T_TRAIN
        n = 3000
        x = rng.uniform(0, L, n)
        t = rng.uniform(0, T_TRAIN, n)
        h, Q = f(x, t)
        return x, t, h, Q
    if scenario == "S2":          # three gauges, 2 s sampling, noisy
        xs = np.array([125.0, 250.0, 375.0])
        ts = np.arange(0.0, T + 1e-9, 2.0)
        XX, TT = np.meshgrid(xs, ts)
        x, t = XX.ravel(), TT.ravel()
        h, Q = f(x, t)
        h = h + rng.normal(0, 0.01, h.shape)      # 1 cm level-sensor noise
        Q = Q + rng.normal(0, 0.20, Q.shape)      # 0.2 m3/s discharge noise
        return x, t, h, Q
    raise ValueError(scenario)


METHODS = {
    #            data  ic  bc  pde
    "NN":       dict(data=1.0, ic=0.0, bc=0.0, pde=0.0),
    "NN+BC":    dict(data=1.0, ic=1.0, bc=1.0, pde=0.0),
    "PINN":     dict(data=1.0, ic=1.0, bc=1.0, pde=1.0),
    "PINN-only": dict(data=0.0, ic=1.0, bc=1.0, pde=1.0),
}
N_COLLOC = 4096
N_IC = 251
N_BC = 201


def run(scenario, method, case="A", seed=0, iters=6000, log=print, w_override=None, n_colloc=N_COLLOC):
    sc = scales(case)
    w = dict(METHODS[method])
    if w_override:
        w.update(w_override)
    net = pn.MLP(ARCH, seed=seed)
    data = make_data(scenario, case, seed) if (scenario in ("S1", "S2") and w["data"] > 0) else None
    sob = qmc.Sobol(2, scramble=True, seed=2000 + seed).random(n_colloc)
    colloc = (sob[:, 0] * L, sob[:, 1] * T)
    ic = np.linspace(0, L, N_IC)
    tb = np.linspace(0, T, N_BC)
    prob = pn.Problem(sc, net, data=data, ic=ic, bc=(tb, tb), colloc=colloc, w=w, q_in=q_in(case))
    t0 = time.time()
    hist = pn.adam_train(prob, iters, log=log, tag=f"[{scenario}/{case}/{method}/s{seed}]")
    train_time = time.time() - t0
    return sc, net, dict(train_time=train_time, hist=[(h[0], float(h[1])) for h in hist],
                         n_data=0 if data is None else len(data[0]), n_params=net.n_params())


# ----------------------------------------------------------------------------
def evaluate(sc, net, case):
    sol, f = reference(case)
    x, t, X, Tm = test_grid()
    nx, nt = len(x), len(t)
    t0 = time.time()
    h, Q = pn.predict(sc, net, X, Tm)
    infer_time = time.time() - t0
    hr, Qr = f(X, Tm)
    out = dict(infer_time=infer_time)
    win = Tm <= T_TRAIN + 1e-9
    wext = ~win
    dh_ref = hr - P["h0"]
    dQ_ref = Qr - P["Q0"]

    def errs(mask, tag):
        e_h, e_Q = (h - hr)[mask], (Q - Qr)[mask]
        out[f"rmse_h_{tag}"] = float(np.sqrt(np.mean(e_h ** 2)))
        out[f"mae_h_{tag}"] = float(np.mean(np.abs(e_h)))
        out[f"rmse_Q_{tag}"] = float(np.sqrt(np.mean(e_Q ** 2)))
        out[f"mae_Q_{tag}"] = float(np.mean(np.abs(e_Q)))
        out[f"rel_h_{tag}"] = float(np.linalg.norm(e_h) / np.linalg.norm(dh_ref[mask]))
        out[f"rel_Q_{tag}"] = float(np.linalg.norm(e_Q) / np.linalg.norm(dQ_ref[mask]))
        out[f"max_h_{tag}"] = float(np.abs(e_h).max())
        out[f"max_Q_{tag}"] = float(np.abs(e_Q).max())
        out[f"r2_h_{tag}"] = float(1 - np.sum(e_h ** 2) / np.sum((hr[mask] - hr[mask].mean()) ** 2))
        out[f"r2_Q_{tag}"] = float(1 - np.sum(e_Q ** 2) / np.sum((Qr[mask] - Qr[mask].mean()) ** 2))

    errs(np.ones_like(win), "all")
    errs(win, "win")
    errs(wext, "ext")
    # ---- physics diagnostics ------------------------------------------------
    h_, Q_, R1, R2 = pn.predict_with_residuals(sc, net, X, Tm)
    out["pde_rms_R1"] = float(np.sqrt(np.mean(R1 ** 2)))
    out["pde_rms_R2"] = float(np.sqrt(np.mean(R2 ** 2)))
    H = h.reshape(nt, nx)
    QQ = Q.reshape(nt, nx)
    out.update(diagnostics_field(x, t, H, QQ, case))
    return out, (x, t, H, QQ)


def diagnostics_field(x, t, H, QQ, case):
    qi = q_in(case)
    out = {}
    # initial-condition error (t = 0 row)
    out["ic_rms_h"] = float(np.sqrt(np.mean((H[0] - P["h0"]) ** 2)))
    out["ic_rms_Q"] = float(np.sqrt(np.mean((QQ[0] - P["Q0"]) ** 2)))
    out["bc_rms_Qin"] = float(np.sqrt(np.mean((QQ[:, 0] - qi(t)) ** 2)))
    out["bc_rms_hout"] = float(np.sqrt(np.mean((H[:, -1] - P["h0"]) ** 2)))
    # global mass balance:  V(t)-V(0) = int (Qin - Qout) dt
    V = B * np.trapezoid(H, x, axis=1)
    flux = QQ[:, 0] - QQ[:, -1]
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(t))))
    err = (V - V[0]) - cum
    out["mass_err_max_m3"] = float(np.abs(err).max())
    ref_vol = float(np.abs(cum[-1]) + 1e-12)
    out["mass_err_rel"] = float(np.abs(err).max() / ref_vol)
    out["h_min"] = float(H.min())
    out["h_max"] = float(H.max())
    out["frac_h_below_h0"] = float(np.mean(H < P["h0"] - 1e-3))
    return out


def reference_diagnostics(case):
    sol, f = reference(case)
    x, t, X, Tm = test_grid()
    hr, Qr = f(X, Tm)
    return diagnostics_field(x, t, hr.reshape(len(t), len(x)), Qr.reshape(len(t), len(x)), case)


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
