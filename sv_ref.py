"""
sv_ref.py  -  Reference (numerical) solution of the 1-D Saint-Venant equations
for a prismatic rectangular channel (explicit 2nd-order finite-volume scheme).

Governing equations (per unit width q = Q/B, rectangular section A = B h):
    dh/dt + dq/dx                      = 0
    dq/dt + d/dx( q^2/h + g h^2/2 )    = g h (S0 - Sf)
    Sf = n^2 u|u| / R^(4/3),  u = q/h,  R = B h / (B + 2 h)      (Manning)

Boundary conditions (sub-critical flow, Fr < 1 -> ONE condition at each end):
    x = 0 : Q(0,t) = Q0 + dQ (1 - exp(-t/tau))        (gate operation)
    x = L : h(L,t) = h0                                (downstream control)
The missing variable at each boundary is obtained from the outgoing
characteristic (Riemann invariant u -/+ 2c carried by the incoming characteristic), which is the standard
well-posed treatment for sub-critical flow.
"""
import numpy as np
import time

G = 9.81

# ----------------------------------------------------------------------------
# channel / flow parameters (single source of truth for the whole study)
# ----------------------------------------------------------------------------
PARAMS = dict(
    L=500.0,      # channel length [m]
    B=10.0,       # bottom width [m]
    n=0.025,      # Manning roughness [s m^-1/3]
    h0=2.0,       # initial (uniform-flow) depth [m]
    Q0=20.0,      # initial discharge [m^3/s]
    T=80.0,       # simulated time [s]
)


def manning_sf(h, q, B, n):
    u = q / h
    R = B * h / (B + 2.0 * h)
    return n * n * u * np.abs(u) / R ** (4.0 / 3.0)


def bed_slope(B, n, h0, Q0):
    """Bed slope for which (h0,Q0) is exactly uniform flow."""
    return float(manning_sf(h0, Q0 / B, B, n))


def normal_depth(Q, B, n, S0):
    lo, hi = 1e-3, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        q = (1 / n) * (B * mid) * (B * mid / (B + 2 * mid)) ** (2 / 3) * np.sqrt(S0)
        if q < Q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def q_inflow(t, Q0, dQ, tau):
    return Q0 + dQ * (1.0 - np.exp(-t / tau))


# ----------------------------------------------------------------------------
def _hll(hl, ql, hr, qr):
    ul, ur = ql / hl, qr / hr
    cl, cr = np.sqrt(G * hl), np.sqrt(G * hr)
    sl = np.minimum(ul - cl, ur - cr)
    sr = np.maximum(ul + cl, ur + cr)
    f1l, f2l = ql, ql * ul + 0.5 * G * hl * hl
    f1r, f2r = qr, qr * ur + 0.5 * G * hr * hr
    den = sr - sl
    f1 = np.where(sl >= 0, f1l, np.where(sr <= 0, f1r, (sr * f1l - sl * f1r + sl * sr * (hr - hl)) / den))
    f2 = np.where(sl >= 0, f2l, np.where(sr <= 0, f2r, (sr * f2l - sl * f2r + sl * sr * (qr - ql)) / den))
    return f1, f2


def _minmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def _flux_phys(h, q):
    return q, q * q / h + 0.5 * G * h * h


def _bc_upstream(h1, q1, qin):
    """q prescribed, h from the incoming C- characteristic: u - 2c = const."""
    u1, c1 = q1 / h1, np.sqrt(G * h1)
    w = u1 - 2 * c1
    hb = h1
    for _ in range(30):
        f = qin / hb - 2 * np.sqrt(G * hb) - w
        df = -qin / hb ** 2 - np.sqrt(G / hb)
        hb = hb - f / df
    return hb


def _bc_downstream(hN, qN, hout):
    """h prescribed, q from the incoming C+ characteristic: u + 2c = const."""
    uN, cN = qN / hN, np.sqrt(G * hN)
    ub = uN + 2 * (cN - np.sqrt(G * hout))
    return hout * ub


def solve(N=500, case=None, params=None, t_out=0.5, cfl=0.45, verbose=False):
    """Returns dict with node-based fields h[nt,N+1], Q[nt,N+1] and metadata."""
    P = dict(PARAMS)
    if params:
        P.update(params)
    L, B, n, h0, Q0, T = (P[k] for k in ("L", "B", "n", "h0", "Q0", "T"))
    case = case or dict(dQ=8.0, tau=10.0)
    dQ, tau = case["dQ"], case["tau"]
    S0 = bed_slope(B, n, h0, Q0)
    dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    h = np.full(N, h0)
    q = np.full(N, Q0 / B)

    umax = abs(Q0 + dQ) / B / min(h0, h0 + 0.1) + np.sqrt(G * (h0 + 1.5))
    dt0 = cfl * dx / umax
    sub = int(np.ceil(t_out / dt0))
    dt = t_out / sub
    nt = int(round(T / t_out)) + 1

    def rhs(h, q, t):
        # reconstruction with ghost cells (zero-gradient ghosts for slopes)
        hg = np.concatenate(([h[0]], h, [h[-1]]))
        qg = np.concatenate(([q[0]], q, [q[-1]]))
        sh = _minmod(hg[1:-1] - hg[:-2], hg[2:] - hg[1:-1])
        sq = _minmod(qg[1:-1] - qg[:-2], qg[2:] - qg[1:-1])
        hL = h + 0.5 * sh   # right-face value of cell i (left state of face i+1/2)
        qL = q + 0.5 * sq
        hR = h - 0.5 * sh   # left-face value of cell i
        qR = q - 0.5 * sq
        f1 = np.empty(N + 1)
        f2 = np.empty(N + 1)
        f1[1:-1], f2[1:-1] = _hll(hL[:-1], qL[:-1], hR[1:], qR[1:])
        # boundary faces: physical flux of the characteristic boundary state
        qin = q_inflow(t, Q0, dQ, tau) / B
        hb0 = _bc_upstream(h[0], q[0], qin)
        f1[0], f2[0] = _flux_phys(hb0, qin)
        qb1 = _bc_downstream(h[-1], q[-1], h0)
        f1[-1], f2[-1] = _flux_phys(h0, qb1)
        Sf = manning_sf(h, q, B, n)
        dh = -(f1[1:] - f1[:-1]) / dx
        dq = -(f2[1:] - f2[:-1]) / dx + G * h * (S0 - Sf)
        return dh, dq, hb0, qin, qb1

    H = np.zeros((nt, N + 1))
    QQ = np.zeros((nt, N + 1))

    def record(k, h, q, hb0, qin, qb1):
        hn = np.empty(N + 1)
        qn = np.empty(N + 1)
        hn[1:-1] = 0.5 * (h[1:] + h[:-1])
        qn[1:-1] = 0.5 * (q[1:] + q[:-1])
        hn[0], qn[0] = hb0, qin
        hn[-1], qn[-1] = h0, qb1
        H[k], QQ[k] = hn, qn * B

    t = 0.0
    dh, dq, hb0, qin, qb1 = rhs(h, q, t)
    record(0, h, q, hb0, qin, qb1)
    t_wall = time.time()
    for k in range(1, nt):
        for _ in range(sub):
            dh1, dq1, *_ = rhs(h, q, t)
            h1, q1 = h + dt * dh1, q + dt * dq1
            dh2, dq2, *_ = rhs(h1, q1, t + dt)
            h = 0.5 * h + 0.5 * (h1 + dt * dh2)
            q = 0.5 * q + 0.5 * (q1 + dt * dq2)
            t += dt
        _, _, hb0, qin, qb1 = rhs(h, q, t)
        record(k, h, q, hb0, qin, qb1)
    wall = time.time() - t_wall
    x = np.linspace(0, L, N + 1)
    tt = np.arange(nt) * t_out
    if verbose:
        print(f"N={N} dx={dx:.3f} dt={dt:.4f} steps={(nt-1)*sub} wall={wall:.2f}s")
    return dict(x=x, t=tt, h=H, Q=QQ, S0=S0, dt=dt, dx=dx, wall=wall, params=P, case=case)


def make_interp(sol):
    from scipy.interpolate import RegularGridInterpolator
    fh = RegularGridInterpolator((sol["t"], sol["x"]), sol["h"], bounds_error=False, fill_value=None)
    fq = RegularGridInterpolator((sol["t"], sol["x"]), sol["Q"], bounds_error=False, fill_value=None)

    def f(x, t):
        pts = np.column_stack([np.asarray(t).ravel(), np.asarray(x).ravel()])
        return fh(pts), fq(pts)
    return f


if __name__ == "__main__":
    P = PARAMS
    S0 = bed_slope(P["B"], P["n"], P["h0"], P["Q0"])
    print("S0 =", S0)
    for Q in (20, 28, 14):
        print("normal depth for Q=%g:" % Q, normal_depth(Q, P["B"], P["n"], S0))
    u = P["Q0"] / (P["B"] * P["h0"])
    print("Fr0 =", u / np.sqrt(G * P["h0"]))
    sol = solve(N=500, verbose=True)
    print("h range", sol["h"].min(), sol["h"].max(), "Q range", sol["Q"].min(), sol["Q"].max())
    print("h(t=60) every 50 m:", np.round(sol["h"][120, ::50], 4))
