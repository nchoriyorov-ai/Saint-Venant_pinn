"""
pinn_np.py - Physics-informed neural network for the 1-D Saint-Venant
equations (rectangular channel, Manning friction) in plain NumPy.

The network maps normalised (x,t) -> normalised (h,Q).  First derivatives of
the outputs w.r.t. the inputs are propagated analytically (forward-mode
"tangent" pass) and the gradient of the *total* loss w.r.t. the weights is
obtained by an exact reverse pass through that tangent computation, so no
finite differences / no autodiff library are needed.  The implementation is
verified against finite differences in `grad_check()`.

Normalisation (all physical constants in `Scales`):
    xn = 2 x / L - 1 ,  tn = 2 t / T - 1                       in [-1, 1]
    h  = h0 + dh * yh ,  Q  = Q0 + dQ * yq
PDE residuals are made dimensionless with the characteristic magnitudes
    S1 = dQ / L              (continuity)
    S2 = g B h0 dh / L       (momentum, hydrostatic-pressure scale)
"""
import numpy as np

G = 9.81


class Scales:
    def __init__(self, L, T, B, n, S0, h0, Q0, dh, dQ):
        self.L, self.T, self.B, self.n, self.S0 = L, T, B, n, S0
        self.h0, self.Q0, self.dh, self.dQ = h0, Q0, dh, dQ
        self.S1 = dQ / L
        self.S2 = G * B * h0 * dh / L

    def xn(self, x):
        return 2.0 * np.asarray(x) / self.L - 1.0

    def tn(self, t):
        return 2.0 * np.asarray(t) / self.T - 1.0

    def to_phys(self, y):
        return self.h0 + self.dh * y[:, 0], self.Q0 + self.dQ * y[:, 1]

    def to_norm(self, h, Q):
        return (h - self.h0) / self.dh, (Q - self.Q0) / self.dQ


# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / (a + b)), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    # flat parameter helpers
    def get(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    def n_params(self):
        return sum(p.size for p in self.get())

    def forward(self, X, tang=True):
        Ws, bs = self.W, self.b
        H = len(Ws) - 1
        A = [X]
        S, ZX, ZT = [], [], []
        TX, TT = [None], [None]
        a = X
        for l in range(H):
            z = a @ Ws[l] + bs[l]
            an = np.tanh(z)
            A.append(an)
            if tang:
                if l == 0:
                    zx = np.broadcast_to(Ws[0][0], z.shape)
                    zt = np.broadcast_to(Ws[0][1], z.shape)
                else:
                    zx = TX[l] @ Ws[l]
                    zt = TT[l] @ Ws[l]
                s = 1.0 - an * an
                S.append(s); ZX.append(zx); ZT.append(zt)
                TX.append(s * zx); TT.append(s * zt)
            a = an
        y = a @ Ws[H] + bs[H]
        if tang:
            yx = TX[H] @ Ws[H]
            yt = TT[H] @ Ws[H]
        else:
            yx = yt = None
        return y, yx, yt, (A, S, ZX, ZT, TX, TT)

    def backward(self, cache, dy, dyx=None, dyt=None):
        A, S, ZX, ZT, TX, TT = cache
        Ws = self.W
        H = len(Ws) - 1
        gW = [None] * (H + 1)
        gb = [None] * (H + 1)
        gW[H] = A[H].T @ dy
        gb[H] = dy.sum(0)
        abar = dy @ Ws[H].T
        tang = dyx is not None
        if tang:
            gW[H] = gW[H] + TX[H].T @ dyx + TT[H].T @ dyt
            txbar = dyx @ Ws[H].T
            ttbar = dyt @ Ws[H].T
        for l in range(H - 1, -1, -1):
            if tang:
                sbar = txbar * ZX[l] + ttbar * ZT[l]
                zxbar = txbar * S[l]
                ztbar = ttbar * S[l]
                abar = abar - 2.0 * A[l + 1] * sbar
            else:
                s = 1.0 - A[l + 1] ** 2
                S_l = s
            s_l = S[l] if tang else S_l
            zbar = abar * s_l
            gW[l] = A[l].T @ zbar
            gb[l] = zbar.sum(0)
            if l == 0:
                if tang:
                    gW[0][0] += zxbar.sum(0)
                    gW[0][1] += ztbar.sum(0)
            else:
                if tang:
                    gW[l] = gW[l] + TX[l].T @ zxbar + TT[l].T @ ztbar
                    txbar = zxbar @ Ws[l].T
                    ttbar = ztbar @ Ws[l].T
                abar = zbar @ Ws[l].T
        return [g for pair in zip(gW, gb) for g in pair]


# ----------------------------------------------------------------------------
def residuals(sc, y, yx, yt, with_grad=False):
    """Dimensionless Saint-Venant residuals R1, R2 and (optionally) their
    partial derivatives w.r.t. (yh,yq,yhx,yht,yqx,yqt) [normalised units]."""
    B, n, S0, g = sc.B, sc.n, sc.S0, G
    h = sc.h0 + sc.dh * y[:, 0]
    Q = sc.Q0 + sc.dQ * y[:, 1]
    fx, ft = 2.0 / sc.L, 2.0 / sc.T
    hx = sc.dh * yx[:, 0] * fx
    ht = sc.dh * yt[:, 0] * ft
    Qx = sc.dQ * yx[:, 1] * fx
    Qt = sc.dQ * yt[:, 1] * ft
    A = B * h
    R = B * h / (B + 2.0 * h)
    R43 = R ** (4.0 / 3.0)
    F = g * n * n * Q * np.abs(Q) / (A * R43)          # = g A Sf
    R1 = (B * ht + Qx) / sc.S1
    R2 = (Qt + 2.0 * Q * Qx / A - Q * Q * hx / (B * h * h) + g * A * hx - g * A * S0 + F) / sc.S2
    if not with_grad:
        return R1, R2
    dFdQ = 2.0 * g * n * n * np.abs(Q) / (A * R43)
    dFdh = F * (-1.0 / h - (4.0 / 3.0) * B / (h * (B + 2.0 * h)))
    # R1 partials
    d1_ht = B / sc.S1
    d1_Qx = 1.0 / sc.S1
    # R2 partials
    d2_Qt = 1.0 / sc.S2
    d2_Qx = 2.0 * Q / A / sc.S2
    d2_hx = (-Q * Q / (B * h * h) + g * A) / sc.S2
    d2_Q = (2.0 * Qx / A - 2.0 * Q * hx / (B * h * h) + dFdQ) / sc.S2
    d2_h = (-2.0 * Q * Qx / (B * h * h) + 2.0 * Q * Q * hx / (B * h ** 3) + g * B * hx - g * B * S0 + dFdh) / sc.S2
    # chain rule to normalised network outputs
    j = dict(
        yh=(0.0, d2_h * sc.dh),
        yq=(0.0, d2_Q * sc.dQ),
        yhx=(0.0, d2_hx * sc.dh * fx),
        yht=(d1_ht * sc.dh * ft, 0.0),
        yqx=(d1_Qx * sc.dQ * fx, d2_Qx * sc.dQ * fx),
        yqt=(0.0, d2_Qt * sc.dQ * ft),
    )
    return R1, R2, j


# ----------------------------------------------------------------------------
class Problem:
    """Holds training sets and evaluates loss + gradient."""

    def __init__(self, sc, net, data=None, ic=None, bc=None, colloc=None, w=None, q_in=None):
        """
        data  : (x,t,h,Q)  physical coordinates / values of (possibly noisy) observations
        ic    : t=0 points  x_i
        bc    : (t_up, t_dn)  times where upstream Q / downstream h are imposed
        colloc: (x,t) collocation points
        w     : dict of weights  data, ic, bc, pde
        q_in  : callable t -> Q_in(t)
        """
        self.sc, self.net, self.w = sc, net, w
        self.q_in = q_in
        blocks, self.sl = [], {}
        pos = 0

        def add(name, x, t):
            nonlocal pos
            X = np.column_stack([sc.xn(x), sc.tn(t)])
            blocks.append(X)
            self.sl[name] = slice(pos, pos + len(X))
            pos += len(X)

        self.has = {}
        if data is not None and w.get("data", 0) > 0:
            x, t, h, Q = data
            add("data", x, t)
            yh, yq = sc.to_norm(h, Q)
            self.y_data = np.column_stack([yh, yq])
            self.has["data"] = True
        if ic is not None and w.get("ic", 0) > 0:
            add("ic", ic, np.zeros_like(ic))
            self.has["ic"] = True
        if bc is not None and w.get("bc", 0) > 0:
            tu, td = bc
            add("bu", np.zeros_like(tu), tu)
            add("bd", np.full_like(td, sc.L), td)
            self.yq_up = (q_in(tu) - sc.Q0) / sc.dQ
            self.has["bc"] = True
        if colloc is not None and w.get("pde", 0) > 0:
            add("pde", colloc[0], colloc[1])
            self.has["pde"] = True
        self.X = np.vstack(blocks)
        self.tang = self.has.get("pde", False)

    def loss_grad(self, want_grad=True):
        sc, net, w, sl = self.sc, self.net, self.w, self.sl
        y, yx, yt, cache = net.forward(self.X, tang=self.tang)
        dy = np.zeros_like(y)
        dyx = np.zeros_like(y) if self.tang else None
        dyt = np.zeros_like(y) if self.tang else None
        parts = {}
        if self.has.get("data"):
            r = y[sl["data"]] - self.y_data
            parts["data"] = np.mean(np.sum(r * r, 1))
            dy[sl["data"]] += w["data"] * 2.0 * r / len(r)
        if self.has.get("ic"):
            r = y[sl["ic"]]
            parts["ic"] = np.mean(np.sum(r * r, 1))
            dy[sl["ic"]] += w["ic"] * 2.0 * r / len(r)
        if self.has.get("bc"):
            ru = y[sl["bu"], 1] - self.yq_up
            rd = y[sl["bd"], 0]
            parts["bc"] = np.mean(ru ** 2) + np.mean(rd ** 2)
            dy[sl["bu"], 1] += w["bc"] * 2.0 * ru / len(ru)
            dy[sl["bd"], 0] += w["bc"] * 2.0 * rd / len(rd)
        if self.has.get("pde"):
            s = sl["pde"]
            R1, R2, j = residuals(sc, y[s], yx[s], yt[s], with_grad=True)
            N = len(R1)
            parts["pde"] = np.mean(R1 ** 2 + R2 ** 2)
            c1, c2 = w["pde"] * 2.0 * R1 / N, w["pde"] * 2.0 * R2 / N
            dy[s, 0] += c1 * j["yh"][0] + c2 * j["yh"][1]
            dy[s, 1] += c1 * j["yq"][0] + c2 * j["yq"][1]
            dyx[s, 0] += c1 * j["yhx"][0] + c2 * j["yhx"][1]
            dyt[s, 0] += c1 * j["yht"][0] + c2 * j["yht"][1]
            dyx[s, 1] += c1 * j["yqx"][0] + c2 * j["yqx"][1]
            dyt[s, 1] += c1 * j["yqt"][0] + c2 * j["yqt"][1]
        total = sum(w[k] * parts[k] for k in parts)
        if not want_grad:
            return total, parts
        grads = net.backward(cache, dy, dyx, dyt)
        return total, parts, grads


def adam_train(prob, iters, lr0=3e-3, lr1=1e-4, log_every=500, log=print, tag=""):
    net = prob.net
    P = net.get()
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(1, iters + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * (it - 1) / iters))
        total, parts, grads = prob.loss_grad()
        for k, (p, g) in enumerate(zip(P, grads)):
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            p -= lr * mh / (np.sqrt(vh) + eps)
        if it % log_every == 0 or it == 1:
            hist.append((it, total, dict(parts)))
            log(f"{tag} it={it:6d} loss={total:.3e} " + " ".join(f"{k}={v_:.2e}" for k, v_ in parts.items()))
    return hist


def predict(sc, net, x, t, chunk=20000):
    X = np.column_stack([sc.xn(x), sc.tn(t)])
    out = []
    for i in range(0, len(X), chunk):
        y, _, _, _ = net.forward(X[i:i + chunk], tang=False)
        out.append(y)
    y = np.vstack(out)
    return sc.to_phys(y)


def predict_with_residuals(sc, net, x, t):
    X = np.column_stack([sc.xn(x), sc.tn(t)])
    y, yx, yt, _ = net.forward(X, tang=True)
    R1, R2 = residuals(sc, y, yx, yt)
    h, Q = sc.to_phys(y)
    return h, Q, R1, R2


# ----------------------------------------------------------------------------
def grad_check(seed=1):
    """Finite-difference verification of the analytic gradient."""
    rng = np.random.default_rng(seed)
    sc = Scales(L=500, T=80, B=10, n=0.025, S0=3.88e-4, h0=2.0, Q0=20.0, dh=0.2, dQ=8.0)
    net = MLP([2, 7, 7, 2], seed=3)
    qin = lambda t: 20 + 8 * (1 - np.exp(-t / 10))
    xd, td = rng.uniform(0, 500, 15), rng.uniform(0, 80, 15)
    data = (xd, td, 2 + 0.1 * rng.random(15), 20 + 3 * rng.random(15))
    prob = Problem(sc, net, data=data, ic=rng.uniform(0, 500, 6), bc=(rng.uniform(0, 80, 5), rng.uniform(0, 80, 5)),
                   colloc=(rng.uniform(0, 500, 20), rng.uniform(0, 80, 20)),
                   w=dict(data=1, ic=1, bc=1, pde=1), q_in=qin)
    # make weights non-trivial so that residual derivatives matter
    for p in net.get():
        p += 0.3 * rng.normal(size=p.shape)
    tot, parts, grads = prob.loss_grad()
    worst = 0.0
    P = net.get()
    for k, (p, g) in enumerate(zip(P, grads)):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = p[idx]
            h = 1e-6
            p[idx] = old + h
            lp, _ = prob.loss_grad(want_grad=False)
            p[idx] = old - h
            lm, _ = prob.loss_grad(want_grad=False)
            p[idx] = old
            fd = (lp - lm) / (2 * h)
            err = abs(fd - g[idx]) / (abs(fd) + abs(g[idx]) + 1e-12)
            worst = max(worst, err)
    print("grad_check: loss parts", {k: float(v) for k, v in parts.items()})
    print("grad_check: worst relative error =", worst)
    return worst


if __name__ == "__main__":
    grad_check()
