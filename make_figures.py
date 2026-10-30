"""make_figures.py - aggregate result jsons into tables (json) and produce figures."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sv_ref as sv
import study as st

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
COL = {"NN": "#d95f02", "NN+BC": "#7570b3", "PINN": "#1b9e77", "PINN-only": "#1b9e77", "FVM": "k"}


def load():
    R = {}
    for fn in glob.glob("results/*.json"):
        r = json.load(open(fn))
        R.setdefault((r["scenario"], r["method"], r["case"]), []).append(r)
    return R


def agg(rs, key):
    v = np.array([r[key] for r in rs])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fld(scen, method, case, seed):
    tag = f"{scen}_{method.replace('+', 'p')}_{case}_s{seed}"
    d = np.load(f"fields/{tag}.npz")
    return d["x"], d["t"], d["H"].astype(float), d["Q"].astype(float)


def best_seed(rs, key="rmse_h_all"):
    return min(rs, key=lambda r: r[key])["seed"]


# ---------------------------------------------------------------------------
def fig_reference():
    sol, f = st.reference("A")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ex = [0, st.L, 0, st.T]
    im = ax[0].imshow(sol["h"], origin="lower", aspect="auto", extent=ex, cmap="viridis")
    ax[0].set(xlabel="x (m)", ylabel="t (s)", title="(a) h(x,t) (m)")
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(sol["Q"], origin="lower", aspect="auto", extent=ex, cmap="viridis")
    ax[1].set(xlabel="x (m)", title="(b) Q(x,t) (m$^3$/s)")
    plt.colorbar(im, ax=ax[1])
    for tt, ls in ((20, ":"), (40, "--"), (60, "-"), (80, "-.")):
        k = int(tt / 0.5)
        ax[2].plot(sol["x"], sol["h"][k], ls, color="k", label=f"t = {tt} s")
    ax[2].set(xlabel="x (m)", ylabel="h (m)", title="(c) depth profiles")
    ax[2].legend(frameon=False)
    for a in ax[:2]:
        a.grid(False)
    fig.tight_layout()
    fig.savefig("figs/fig2_reference.png")
    plt.close(fig)


def fig_convergence():
    # grid convergence (computed on the fly; cheap)
    Ns = (125, 250, 500, 1000)
    ref = sv.solve(N=2000)
    xs = np.linspace(0, st.L, 501)
    rows = []
    for N in Ns:
        s = sv.solve(N=N)
        e = {}
        for k, tn in ((120, 60),):
            hr = np.interp(xs, ref["x"], ref["h"][k])
            h = np.interp(xs, s["x"], s["h"][k])
            e["rmse_h"] = float(np.sqrt(np.mean((h - hr) ** 2)))
            qr = np.interp(xs, ref["x"], ref["Q"][k])
            q = np.interp(xs, s["x"], s["Q"][k])
            e["rmse_Q"] = float(np.sqrt(np.mean((q - qr) ** 2)))
        rows.append(dict(N=N, dx=st.L / N, wall=s["wall"], **e))
    json.dump(rows, open("results_summary_convergence.json", "w"), indent=1)
    return rows


def fig_snapshots(R, scen, case, method_list, fname, tsnap=(40, 60, 80), seed_pick=None):
    sol, f = st.reference(case)
    fig, ax = plt.subplots(2, len(tsnap), figsize=(10.5, 5.0), sharex=True)
    for j, tt in enumerate(tsnap):
        k = int(round(tt / 0.5))
        ax[0, j].plot(sol["x"], sol["h"][k], "k-", lw=2, label="Reference")
        ax[1, j].plot(sol["x"], sol["Q"][k], "k-", lw=2)
        for m in method_list:
            rs = R.get((scen, m, case))
            if not rs:
                continue
            s = best_seed(rs) if seed_pick is None else seed_pick
            x, t, H, Q = fld(scen, m, case, s)
            kk = int(np.argmin(np.abs(t - tt)))
            ax[0, j].plot(x, H[kk], "--", color=COL[m], lw=1.5, label=m)
            ax[1, j].plot(x, Q[kk], "--", color=COL[m], lw=1.5)
        ax[0, j].set(title=f"t = {tt} s", ylabel="h (m)" if j == 0 else "")
        ax[1, j].set(xlabel="x (m)", ylabel="Q (m$^3$/s)" if j == 0 else "")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_error_maps(R, scen, case, methods, fname):
    sol, f = st.reference(case)
    x, t, X, Tm = st.test_grid()
    hr, _ = f(X, Tm)
    hr = hr.reshape(len(t), len(x))
    fig, ax = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.0), sharey=True)
    ax = np.atleast_1d(ax)
    vmax = 0
    errs = []
    for m in methods:
        rs = R[(scen, m, case)]
        x_, t_, H, Q = fld(scen, m, case, best_seed(rs))
        errs.append((H - hr) * 1000.0)
        vmax = max(vmax, np.abs(errs[-1]).max())
    vmax = min(vmax, 60)
    for a, m, e in zip(ax, methods, errs):
        im = a.imshow(e, origin="lower", aspect="auto", extent=[0, st.L, 0, st.T], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set(xlabel="x (m)", title=f"{m}: h error (mm)")
        a.grid(False)
        if scen == "S1":
            a.axhline(st.T_TRAIN, color="k", lw=0.8, ls="--")
    ax[0].set_ylabel("t (s)")
    plt.colorbar(im, ax=ax.tolist(), shrink=0.9)
    fig.savefig(fname)
    plt.close(fig)


def fig_gauges(R, fname):
    sol, f = st.reference("A")
    xs = [125.0, 250.0, 375.0]
    ts = np.arange(0.0, st.T + 1e-9, 2.0)
    rng = np.random.default_rng(1000)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    for a, xg in zip(ax, xs):
        hh, _ = f(np.full_like(ts, xg), ts)
        a.plot(ts, hh, "k-", label="Reference")
        a.set(title=f"gauge x = {xg:.0f} m", xlabel="t (s)")
    # noisy observations for seed 0
    x, t, h, Q = st.make_data("S2", "A", 0)
    for a, xg in zip(ax, xs):
        m_ = np.isclose(x, xg)
        a.plot(t[m_], h[m_], "o", ms=3, color="gray", alpha=0.8, label="noisy observations")
        for m in ("NN", "NN+BC", "PINN"):
            rs = R.get(("S2", m, "A"))
            if rs:
                x_, t_, H, Q_ = fld("S2", m, "A", 0)
                ix = int(np.argmin(np.abs(x_ - xg)))
                a.plot(t_, H[:, ix], "--", color=COL[m], label=m)
    ax[0].set_ylabel("h (m)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_bar_physics(R, fname):
    import matplotlib.ticker as mt
    labels, vals = [], {}
    items = [("S2", "NN", "A"), ("S2", "NN+BC", "A"), ("S2", "PINN", "A"), ("S3", "PINN-only", "A"), ("S3", "PINN-only", "B")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    keys = [("pde_rms_R2", "PDE residual RMS (momentum, dimensionless)"),
            ("mass_err_rel", "max. mass-balance error (relative)"),
            ("bc_rms_Qin", "inlet-BC RMS error (m$^3$/s)")]
    names = []
    for it in items:
        if it in R:
            names.append(f"{it[1]}\n{it[0]}/{it[2]}")
    for a, (k, ttl) in zip(ax, keys):
        mu, sd, cols = [], [], []
        for it in items:
            if it in R:
                m_, s_ = agg(R[it], k)
                mu.append(m_); sd.append(s_); cols.append(COL[it[1]])
        a.bar(range(len(mu)), mu, yerr=sd, color=cols, capsize=3)
        a.set_yscale("log")
        a.set_xticks(range(len(mu)))
        a.set_xticklabels(names, fontsize=6.5)
        a.set_title(ttl, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_training(R, fname):
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0))
    for key, c, lab in ((("S3", "PINN-only", "A"), COL["PINN"], "case A"), (("S3", "PINN-only", "B"), "#66a61e", "case B")):
        if key in R:
            h = R[key][0]["hist"]
            ax.semilogy([a for a, b in h], [b for a, b in h], color=c, label=lab)
    ax.set(xlabel="iteration", ylabel="total loss", title="PINN-only training history")
    ax.legend(frameon=False)
    fig.savefig(fname)
    plt.close(fig)


def fig_arch(fname):
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    def box(x, y, w, h, txt, fc="#eef3fb", ec="#33507a", fs=8):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
    box(0.1, 1.4, 1.2, 1.2, "inputs\n$(x,t)$\nnormalised\nto [-1,1]")
    box(1.9, 1.0, 2.2, 2.0, "MLP: 4 hidden layers\n× 40 neurons, tanh\n(5122 parameters)\n$\\to(\\hat h,\\hat Q)$")
    box(4.7, 2.6, 2.5, 1.1, "automatic/analytic\nderivatives\n$h_x,h_t,Q_x,Q_t$", fc="#fdf2e0", ec="#a8741a")
    box(4.7, 0.3, 2.5, 1.1, "$h=h_0+\\Delta h\\,\\hat h$,\n$Q=Q_0+\\Delta Q\\,\\hat Q$\n(data, IC, BC terms)", fc="#fdf2e0", ec="#a8741a")
    box(7.8, 2.6, 2.0, 1.1, "$\\mathcal{L}_{pde}$:\nresiduals $R_1,R_2$\n(Eqs. 1–2)", fc="#e6f4ea", ec="#2e7d4f")
    box(7.8, 0.3, 2.0, 1.1, "$\\mathcal{L}_{data}$, $\\mathcal{L}_{ic}$,\n$\\mathcal{L}_{bc}$", fc="#e6f4ea", ec="#2e7d4f")
    for (a, b) in (((1.3, 2.0), (1.9, 2.0)), ((4.1, 2.2), (4.7, 3.1)), ((4.1, 1.8), (4.7, 0.9)), ((7.2, 3.15), (7.8, 3.15)), ((7.2, 0.85), (7.8, 0.85))):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(9.9, 2.0, "$\\mathcal{L}=w_d\\mathcal{L}_{data}+w_i\\mathcal{L}_{ic}$\n$+\\,w_b\\mathcal{L}_{bc}+w_p\\mathcal{L}_{pde}$\nAdam  → update $\\theta$", ha="right", va="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    fig.savefig(fname)
    plt.close(fig)


def fig_timing(rows, fname):
    pass


if __name__ == "__main__":
    R = load()
    fig_arch("figs/fig1_architecture.png")
    fig_reference()
    conv = fig_convergence()
    print("convergence:", conv)
    if ("S3", "PINN-only", "A") in R:
        fig_snapshots(R, "S3", "A", ["PINN-only"], "figs/fig_s3_case_A.png")
        fig_error_maps(R, "S3", "A", ["PINN-only"], "figs/fig_s3_err_A.png")
    if ("S3", "PINN-only", "B") in R:
        fig_snapshots(R, "S3", "B", ["PINN-only"], "figs/fig_s3_case_B.png")
        fig_error_maps(R, "S3", "B", ["PINN-only"], "figs/fig_s3_err_B.png")
    if ("S2", "PINN", "A") in R:
        fig_snapshots(R, "S2", "A", ["NN", "NN+BC", "PINN"], "figs/fig_s2_snapshots.png")
        fig_error_maps(R, "S2", "A", ["NN", "NN+BC", "PINN"], "figs/fig_s2_err.png")
        fig_gauges(R, "figs/fig_s2_gauges.png")
    if ("S1", "PINN", "A") in R:
        fig_snapshots(R, "S1", "A", ["NN", "NN+BC", "PINN"], "figs/fig_s1_snapshots.png", tsnap=(40, 60, 80))
        fig_error_maps(R, "S1", "A", ["NN", "NN+BC", "PINN"], "figs/fig_s1_err.png")
    fig_bar_physics(R, "figs/fig_physics_diag.png")
    fig_training(R, "figs/fig_training.png")
    print("figures done")
