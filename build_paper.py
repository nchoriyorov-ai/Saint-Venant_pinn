import glob, json, re, subprocess
import numpy as np

R = {}
for f in glob.glob("results/*.json"):
    r = json.load(open(f))
    R.setdefault((r["scenario"], r["method"], r["case"]), []).append(r)
CONV = json.load(open("results_summary_convergence.json"))


def ms(scen, m, case, key, sc=1.0, d=2):
    v = np.array([r[key] for r in R[(scen, m, case)]]) * sc
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def mv(scen, m, case, key, sc=1.0):
    return float(np.mean([r[key] for r in R[(scen, m, case)]])) * sc


def sci(scen, m, case, key):
    v = np.array([r[key] for r in R[(scen, m, case)]])
    return f"{v.mean():.1e} ± {v.std(ddof=1):.1e}".replace("e-0", "e−").replace("e-", "e−")


REFS = {
 "xu2012": "Xu, M.; Negenborn, R.R.; van Overloop, P.J.; van de Giesen, N.C. De Saint-Venant equations-based model assessment in model predictive control of open channel flow. *Adv. Water Resour.* **2012**, *49*, 37–45.",
 "chaudhry2008": "Chaudhry, M.H. *Open-Channel Flow*, 2nd ed.; Springer: New York, NY, USA, 2008.",
 "cunge1980": "Cunge, J.A.; Holly, F.M.; Verwey, A. *Practical Aspects of Computational River Hydraulics*; Pitman: London, UK, 1980.",
 "litrico2009": "Litrico, X.; Fromion, V. *Modeling and Control of Hydrosystems*; Springer: London, UK, 2009.",
 "natasha2019": "Natasha, G.; Suharjito; Noviantri, V. Saint-Venant model analysis of trapezoidal open channel water flow using finite difference method. In Proceedings of the 4th International Conference on Computer Science and Computational Intelligence (ICCSCI), Yogyakarta, Indonesia, 2019.",
 "toro2001": "Toro, E.F. *Shock-Capturing Methods for Free-Surface Shallow Flows*; Wiley: Chichester, UK, 2001.",
 "leveque2002": "LeVeque, R.J. *Finite Volume Methods for Hyperbolic Problems*; Cambridge University Press: Cambridge, UK, 2002.",
 "harten1983": "Harten, A.; Lax, P.D.; van Leer, B. On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Rev.* **1983**, *25*, 35–61.",
 "vanleer1979": "van Leer, B. Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov’s method. *J. Comput. Phys.* **1979**, *32*, 101–136.",
 "seytov2024": "Seytov, A.; Abduraxmonov, O.; Choriyorov, N.; Maxkamov, A. Development of algorithms for planning and management of water resources in irrigation systems in the conditions of climate changing. *Sci. Bull. Phys. Math. Res.* **2024**, *1*(6).",
 "turaev2024": "Turaev, R.; Seytov, A.; Haydarova, R.; Abduraxmonov, O.; Choriyorov, N. Optimal water management in the channels of machine water raise systems. *O’zMU Xabarlari* **2024**, *2/1.1*.",
 "seytov2026": "Seytov, A.J.; Choriyorov, N.Q.; Abdujabborov, Z.A. Optimization of hydraulic processes in open channels based on deep learning models. *Raqamli Transformatsiya va Sun’iy Intellekt Ilmiy Jurnali* **2026**, *4*(4).",
 "muhamediyeva2024": "Muhamediyeva, D.; Mamatov, N.; Turgunov, B.; Yusupova, D. Control of unstable water flow in open channels. *E3S Web Conf.* **2024**, *590*, 02002.",
 "shen2018": "Shen, C. A transdisciplinary review of deep learning research and its relevance for water resources scientists. *Water Resour. Res.* **2018**, *54*, 8558–8593.",
 "kratzert2018": "Kratzert, F.; Klotz, D.; Brenner, C.; Schulz, K.; Herrnegger, M. Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrol. Earth Syst. Sci.* **2018**, *22*, 6005–6022.",
 "nearing2021": "Nearing, G.S.; Kratzert, F.; Sampson, A.K.; Pelissier, C.S.; Klotz, D.; Frame, J.M.; Prieto, C.; Gupta, H.V. What role does hydrological science play in the age of machine learning? *Water Resour. Res.* **2021**, *57*, e2020WR028091.",
 "raissi2019": "Raissi, M.; Perdikaris, P.; Karniadakis, G.E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *J. Comput. Phys.* **2019**, *378*, 686–707.",
 "karniadakis2021": "Karniadakis, G.E.; Kevrekidis, I.G.; Lu, L.; Perdikaris, P.; Wang, S.; Yang, L. Physics-informed machine learning. *Nat. Rev. Phys.* **2021**, *3*, 422–440.",
 "cuomo2022": "Cuomo, S.; Di Cola, V.S.; Giampaolo, F.; Rozza, G.; Raissi, M.; Piccialli, F. Scientific machine learning through physics-informed neural networks: Where we are and what’s next. *J. Sci. Comput.* **2022**, *92*, 88.",
 "lu2021": "Lu, L.; Meng, X.; Mao, Z.; Karniadakis, G.E. DeepXDE: A deep learning library for solving differential equations. *SIAM Rev.* **2021**, *63*, 208–228.",
 "mao2020": "Mao, Z.; Jagtap, A.D.; Karniadakis, G.E. Physics-informed neural networks for high-speed flows. *Comput. Methods Appl. Mech. Eng.* **2020**, *360*, 112789.",
 "bihlo2022": "Bihlo, A.; Popovych, R.O. Physics-informed neural networks for the shallow-water equations on the sphere. *J. Comput. Phys.* **2022**, *456*, 111024.",
 "feng2023": "Feng, D.; Tan, Z.; He, Q. Physics-informed neural networks of the Saint-Venant equations for downscaling a large-scale river model. *Water Resour. Res.* **2023**, *59*, e2022WR033168.",
 "cedillo2022": "Cedillo, S.; Núñez, A.G.; Sánchez-Cordero, E.; Timbe, L.; Samaniego, E.; Alvarado, A. Physics-informed neural network water surface predictability for 1D steady-state open channel cases with different flow types and complex bed profile shapes. *Adv. Model. Simul. Eng. Sci.* **2022**, *9*, 10.",
 "qi2024": "Qi, X.; de Almeida, G.A.M.; Maldonado, S. Physics-informed neural networks for solving flow problems modeled by the 2D shallow water equations without labeled data. *J. Hydrol.* **2024**, *636*.",
 "niu2023": "Niu, J.; Xu, W.; Qiu, H.; Li, S.; Dong, F. 1-D coupled surface flow and transport equations revisited via the physics-informed neural network approach. *J. Hydrol.* **2023**, *625*.",
 "fang2023": "Fang, W.; Xu, M. Data assimilation method of Saint-Venant equations based on PINNs. *Water Resour. Prot.* **2023**, *39*, 24–31 (in Chinese).",
 "chen2023": "Chen, Y.; Xu, Y.; Wang, L.; Li, T. Modeling water flow in unsaturated soils through physics-informed neural network with principled loss function. *Comput. Geotech.* **2023**, *161*, 105546.",
 "roushangar2023": "Roushangar, K.; Ghasempour, R.; Azamathulla, H.M. An inverse problem for modeling open channel flow with movable bed. *Water Supply* **2023**, *23*, 222. doi:10.2166/ws.2022.442",
 "monaco2023": "Monaco, S.; Apiletti, D. Training physics-informed neural networks: One learning to rule them all? *Results Eng.* **2023**, *18*, 101023.",
 "wang2021": "Wang, S.; Teng, Y.; Perdikaris, P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* **2021**, *43*, A3055–A3081.",
 "wang2022": "Wang, S.; Yu, X.; Perdikaris, P. When and why PINNs fail to train: A neural tangent kernel perspective. *J. Comput. Phys.* **2022**, *449*, 110768.",
 "wang2024": "Wang, S.; Sankaran, S.; Perdikaris, P. Respecting causality for training physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* **2024**, *421*, 116813 (arXiv:2203.07404).",
 "krish2021": "Krishnapriyan, A.; Gholami, A.; Zhe, S.; Kirby, R.; Mahoney, M.W. Characterizing possible failure modes in physics-informed neural networks. *Adv. Neural Inf. Process. Syst.* **2021**, *34*, 26548–26560.",
 "rahaman2019": "Rahaman, N.; Baratin, A.; Arpit, D.; Draxler, F.; Lin, M.; Hamprecht, F.; Bengio, Y.; Courville, A. On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, Long Beach, CA, USA, 2019; pp. 5301–5310.",
 "wu2023": "Wu, C.; Zhu, M.; Tan, Q.; Kartha, Y.; Lu, L. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* **2023**, *403*, 115671.",
 "grossmann2024": "Grossmann, T.G.; Komorowska, U.J.; Latz, J.; Schönlieb, C.-B. Can physics-informed neural networks beat the finite element method? *IMA J. Appl. Math.* **2024**, *89*.",
 "jagtap2020": "Jagtap, A.D.; Karniadakis, G.E. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. *Commun. Comput. Phys.* **2020**, *28*, 2002–2041.",
 "li2021": "Li, Z.; Kovachki, N.; Azizzadenesheli, K.; Liu, B.; Bhattacharya, K.; Stuart, A.; Anandkumar, A. Fourier neural operator for parametric partial differential equations. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.",
 "lu2021b": "Lu, L.; Jin, P.; Pang, G.; Zhang, Z.; Karniadakis, G.E. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nat. Mach. Intell.* **2021**, *3*, 218–229.",
 "kingma2015": "Kingma, D.P.; Ba, J. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), San Diego, CA, USA, 2015.",
 "glorot2010": "Glorot, X.; Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 9, Sardinia, Italy, 2010; pp. 249–256.",
 "sobol1967": "Sobol’, I.M. On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Comput. Math. Math. Phys.* **1967**, *7*, 86–112.",
 "baydin2018": "Baydin, A.G.; Pearlmutter, B.A.; Radul, A.A.; Siskind, J.M. Automatic differentiation in machine learning: A survey. *J. Mach. Learn. Res.* **2018**, *18*, 1–43.",
 "botchkarev2019": "Botchkarev, A. A new typology design of performance metrics to measure errors in machine learning regression algorithms. *Interdiscip. J. Inf. Knowl. Manag.* **2019**, *14*, 45–76.",
}

fm = lambda s: s  # placeholder

# ----------------------------------------------------------------------------
TXT = r"""
---
title: "Physics-Informed Neural Networks for Unsteady Flow in Open Channels Governed by the Saint-Venant Equations: Forward Solution and Assimilation of Sparse, Noisy Gauge Data"
---

**Abstract:** Physics-informed neural networks (PINNs) are increasingly proposed for open-channel hydraulics, but published comparisons with purely data-driven networks often lack a fully specified physical set-up, matched training conditions and physics-based diagnostics. This study examines unsteady sub-critical flow (Froude number 0.23) in a prismatic rectangular irrigation canal governed by the one-dimensional Saint-Venant equations with bed slope and Manning friction, using an explicit second-order finite-volume solution as reference. A PINN, a data-only network, and a network that knows the initial and boundary conditions but not the PDE share one architecture, optimiser and data, so that the contribution of the PDE residual can be isolated. Without any training data, the PINN reproduces gate-opening and gate-closing waves with a depth RMSE of about @@S3A_H@@ mm and mass-balance errors of order 10⁻⁴. When only three noisy gauge records are available, the PINN lowers the depth RMSE from @@S2NN_H@@ mm (data-only network) and @@S2NNBC_H@@ mm (network with boundary information) to @@S2PINN_H@@ mm. With dense noise-free samples, the data-only network is more accurate inside the training window (@@S1NN_W@@ mm versus @@S1PINN_W@@ mm), whereas the PINN is more accurate when extrapolating in time (@@S1PINN_E@@ mm versus @@S1NN_E@@ mm) and satisfies the governing equations and mass balance much more closely. Training took about @@TRAIN@@ s on one CPU core, versus 0.1–1.6 s for the numerical solver, so no computational advantage is claimed for forward simulation. The conclusions are limited to synthetic data, smooth waves and a rectangular section.

**Keywords:** physics-informed neural networks; Saint-Venant equations; open-channel flow; irrigation canals; data assimilation; sparse noisy observations; mass conservation

# 1. Introduction

Unsteady flow in irrigation and drainage canals is routinely modelled with the one-dimensional Saint-Venant (SV) equations, which underpin both simulation and model-based control of canal systems [@chaudhry2008; @cunge1980; @litrico2009; @xu2012]. Classical finite-difference and finite-volume solvers are mature and fast [@toro2001; @leveque2002; @natasha2019], yet their use in operational settings is limited by uncertain boundary data, sparse sensors and the need to reconcile model and observations. In Central Asia, where water allocation in canal networks is under pressure from climate variability and ageing infrastructure, these limitations are practically relevant [@seytov2024; @turaev2024; @muhamediyeva2024; @seytov2026].

Data-driven models have transformed several areas of hydrology [@shen2018; @kratzert2018; @nearing2021], but purely data-driven networks are not constrained by conservation laws and may extrapolate poorly. Physics-informed neural networks (PINNs) address this by adding the residuals of the governing equations to the training loss [@raissi2019; @karniadakis2021; @cuomo2022; @lu2021]. PINNs have been applied to the shallow-water and Saint-Venant equations in several settings: downscaling and data assimilation for one-dimensional river models [@feng2023; @fang2023], steady open-channel profiles and roughness estimation [@cedillo2022], two-dimensional flows without labelled data [@qi2024], the shallow-water equations on the sphere [@bihlo2022], coupled surface-flow and transport problems [@niu2023], and inverse problems in open channels [@roushangar2023], and flow in unsaturated soils [@chen2023]. In parallel, the methodological literature has documented that PINN training can fail or converge slowly because of unbalanced loss terms, causality violations, spectral bias and stiff dynamics [@wang2021; @wang2022; @wang2024; @krish2021; @rahaman2019; @monaco2023], and that PINNs do not automatically outperform classical solvers in cost or accuracy [@grossmann2024].

Three gaps motivate the present study. First, PINN studies of the SV equations frequently do not report the channel geometry, roughness, bed slope or flow regime, so the boundary-condition treatment, which depends on the Froude number for hyperbolic systems, cannot be judged. Second, comparisons between a PINN and a plain network are often not like-for-like: the two may differ in inputs, size, training data and evaluation points, which makes it impossible to attribute differences to the physics. Third, claims of "physical consistency" or efficiency are seldom supported by quantitative diagnostics such as mass-balance error or residual norms, or by timing.

We do not propose a new PINN algorithm. The contribution of this paper is a fully specified and reproducible evaluation that addresses these gaps: (i) a sub-critical irrigation-canal benchmark in which the reference solver and the PINN solve exactly the same equations, with boundary conditions justified by characteristic analysis; (ii) a controlled ablation with three networks of identical architecture, optimiser, seeds and data, separating the effect of the PDE residual from that of boundary-condition knowledge; (iii) tests that the earlier literature on canal PINNs rarely contains together: forward solution without data for two different gate operations, assimilation of three noisy gauge records, and temporal extrapolation; and (iv) physics-based diagnostics and cost measurements reported alongside accuracy, including the cases in which the PINN is not better.

# 2. Materials and Methods

## 2.1. Governing equations

For a prismatic channel with cross-sectional area $A$, discharge $Q$ and water depth $h$, the one-dimensional Saint-Venant equations are [@chaudhry2008; @cunge1980]

$$\frac{\partial A}{\partial t}+\frac{\partial Q}{\partial x}=0 \qquad (1)$$

$$\frac{\partial Q}{\partial t}+\frac{\partial}{\partial x}\left(\frac{Q^{2}}{A}+g I_1\right)=gA\,(S_0-S_f) \qquad (2)$$

where $I_1=\int_0^h (h-\eta)\,b(\eta)\,d\eta$ is the first moment of the cross-sectional area about the free surface, $b$ is the top width, $S_0$ is the bed slope and $S_f$ is the friction slope. For the rectangular section of width $B$ used here, $A=Bh$, $gI_1=gBh^{2}/2$ and $\partial (gI_1)/\partial x = gA\,\partial h/\partial x$. The friction slope follows Manning’s formula,

$$S_f=\frac{n^{2}\,Q|Q|}{A^{2}R^{4/3}},\qquad R=\frac{Bh}{B+2h} \qquad (3)$$

with roughness coefficient $n$. Equations (1)–(3) are used without modification by both the reference solver and the PINN. Written in terms of $h$ and $Q$, the system is

$$B\,h_t+Q_x=0,\qquad Q_t+\frac{2Q}{Bh}Q_x-\frac{Q^{2}}{Bh^{2}}h_x+gBh\,h_x=gBh\,(S_0-S_f) \qquad (4)$$

with characteristic speeds $\lambda_{\pm}=u\pm c$, where $u=Q/A$ and $c=\sqrt{gh}$. The Froude number is $Fr=u/c$.

## 2.2. Test channel, flow regime and boundary conditions

The channel parameters are listed in Table 1. They describe a representative lined or earthen inter-farm irrigation canal; they are not the measured properties of a specific canal. The bed slope is chosen so that the initial state ($h_0=2$ m, $Q_0=20$ m³/s) is exactly uniform flow, $S_0=S_f(h_0,Q_0)=3.88\times10^{-4}$. The initial Froude number is $Fr_0=@@FR@@$ ($u_0=1.00$ m/s, $c_0=4.43$ m/s), so the flow is sub-critical and the characteristic speeds are $\lambda_+=+5.43$ m/s and $\lambda_-=-3.43$ m/s. One characteristic enters the domain at each end; hence exactly one boundary condition is imposed at each end [@toro2001; @litrico2009].

**Table 1.** Parameters of the test channel and simulations.

| Quantity | Symbol | Value |
|:--|:-:|:-:|
| Channel length | $L$ | 500 m |
| Bottom width (rectangular) | $B$ | 10 m |
| Manning roughness | $n$ | 0.025 s m⁻¹ᐟ³ |
| Bed slope | $S_0$ | 3.88 × 10⁻⁴ |
| Initial depth / discharge (uniform flow) | $h_0$, $Q_0$ | 2.00 m / 20 m³/s |
| Initial Froude number | $Fr_0$ | @@FR@@ |
| Characteristic speeds at initial state | $\lambda_\pm$ | +5.43 / −3.43 m/s |
| Simulated time | $T$ | 80 s |
| Gravitational acceleration | $g$ | 9.81 m/s² |

*Inlet* ($x=0$): the discharge follows a gate operation,
$Q(0,t)=Q_0+\Delta Q\,(1-e^{-t/\tau})$. Case A (gate opening): $\Delta Q=+8$ m³/s, $\tau=10$ s (normal depths 2.00 → 2.52 m). Case B (gate closing): $\Delta Q=-6$ m³/s, $\tau=20$ s (normal depths 2.00 → 1.57 m).
*Outlet* ($x=L$): the depth is held at $h(L,t)=h_0$ (downstream control).
*Initial condition*: uniform flow, $h(x,0)=h_0$, $Q(x,0)=Q_0$.

The outlet condition does not generate reflections within the simulated time. Information leaving the inlet travels at most at the speed of the leading wave front, about 5.4–5.9 m/s; the front (defined as $|h-h_0|>1$ mm) was located at $x\approx@@F40@@$, @@F60@@ and @@F80@@ m at $t=40$, 60 and 80 s, respectively, i.e., it does not reach $x=L=500$ m before $T=80$ s. The undisturbed outlet state therefore coincides with the imposed depth, and the downstream condition influences the solution only for $t\gtrsim L/\lambda_+\approx92$ s. For longer horizons the outlet condition (for instance a rating curve) would need to be re-examined.

## 2.3. Numerical reference solution

The reference solution is computed with an explicit second-order finite-volume scheme: MUSCL reconstruction with the minmod limiter [@vanleer1979], HLL numerical flux [@harten1983], strong-stability-preserving second-order Runge–Kutta time stepping (CFL = 0.45), and the source terms of Eq. (4) evaluated at cell centres. The boundary states are obtained from the incoming characteristic: the Riemann invariant $u-2c$ is carried to the inlet and $u+2c$ to the outlet, which gives the missing variable once $Q$ (inlet) or $h$ (outlet) is prescribed. The scheme is conservative in the interior, and the uniform initial state is preserved exactly. Grid convergence was assessed against a fine solution with $N=2000$ cells (Table 2); the error decreases at roughly first order, which is expected for a front-dominated solution with first-order boundary treatment. The solution with $N=2000$ ($\Delta x=0.25$ m) is used as the reference ("truth"); its estimated depth error is below 0.1 mm, two orders of magnitude smaller than the PINN errors reported below.

**Table 2.** Grid convergence of the reference solver (case A, $t=60$ s; RMSE with respect to the $N=2000$ solution) and run time on one CPU core.

| Cells $N$ | $\Delta x$ (m) | RMSE $h$ (mm) | RMSE $Q$ (m³/s) | Run time (s) |
|:-:|:-:|:-:|:-:|:-:|
@@CONVROWS@@

## 2.4. PINN formulation

The network maps the normalised coordinates $(x_n,t_n)=(2x/L-1,\,2t/T-1)$ to normalised outputs $(\hat h,\hat Q)$, with $h=h_0+\Delta h\,\hat h$ and $Q=Q_0+\Delta Q\,\hat Q$ ($\Delta h=0.2$ m, $\Delta Q=|\Delta Q_\mathrm{case}|$). The residuals of Eq. (4),

$$R_1=\frac{B\,h_t+Q_x}{S_1},\qquad R_2=\frac{Q_t+\frac{2Q}{Bh}Q_x-\frac{Q^{2}}{Bh^{2}}h_x+gBh\,h_x-gBh\,(S_0-S_f)}{S_2} \qquad (5)$$

are evaluated in **physical** variables, i.e., the chain rule between normalised and physical coordinates is applied analytically, with the characteristic scales $S_1=\Delta Q/L$ and $S_2=gBh_0\Delta h/L$ making both residuals dimensionless and of order one. The loss is

$$\mathcal{L}=w_d\mathcal{L}_{data}+w_i\mathcal{L}_{ic}+w_b\mathcal{L}_{bc}+w_p\mathcal{L}_{pde} \qquad (6)$$

with $\mathcal{L}_{pde}=\frac{1}{N_f}\sum_k (R_{1,k}^2+R_{2,k}^2)$; $\mathcal{L}_{ic}$ penalises $(h-h_0,\,Q-Q_0)$ at $t=0$; $\mathcal{L}_{bc}$ penalises $Q(0,t)-Q_\mathrm{in}(t)$ and $h(L,t)-h_0$; and $\mathcal{L}_{data}$ is the mean squared error of the observations in normalised units. All derivatives of the network outputs with respect to $x$ and $t$ are propagated analytically through the network (forward-mode propagation, with the reverse pass through that computation giving the gradient of Eq. (6) with respect to the weights) [@baydin2018]; the implementation uses NumPy only, and the analytic gradient agrees with central finite differences to a relative error of $4\times10^{-8}$. The residual expressions were additionally checked by evaluating Eq. (5) on the finite-volume solution, where the residual is below $10^{-3}$ (RMS) in the smooth region of the solution. The complete training set-up is given in Table 3 so that every run can be reproduced. The workflow is summarised in Figure 1.

![**Figure 1.** Workflow of the PINN: normalised $(x,t)$ inputs, a fully connected tanh network, analytic derivatives, and the composite loss of Eq. (6).](figs/fig1_architecture.png){width=6.3in}

**Table 3.** Network and training settings (identical for all learned models).

| Item | Setting |
|:--|:--|
| Architecture | 2 – 40 – 40 – 40 – 40 – 2, tanh activation, 5122 parameters |
| Initialisation | Xavier/Glorot normal [@glorot2010]; seeds 0, 1, 2 |
| Precision | float64 |
| Optimiser | Adam [@kingma2015], $\beta_1=0.9$, $\beta_2=0.999$, full batch |
| Learning rate | cosine decay from $3\times10^{-3}$ to $10^{-4}$ |
| Stopping rule | fixed budget of 6000 iterations (no early stopping) |
| Loss weights | $w_d=w_i=w_b=w_p=1$ where the term is active (Table 4) |
| Collocation points | $N_f=4096$ scrambled Sobol points [@sobol1967] in $[0,L]\times[0,T]$, fixed during training; random seed $2000+$seed |
| Initial-condition points | 251 equispaced in $x$ at $t=0$ |
| Boundary points | 201 equispaced times at each boundary |
| Data, scenario S1 | 3000 uniformly random points, $x\in[0,500]$ m, $t\in[0,56]$ s, noise-free (seed $1000+$seed) |
| Data, scenario S2 | 3 gauges, $x=125,250,375$ m, every 2 s over $0$–$80$ s (41 times); 123 $h$ and 123 $Q$ values; noise $\sigma_h=0.01$ m, $\sigma_Q=0.2$ m³/s |
| Test points | 201 × 161 grid ($\Delta x=2.5$ m, $\Delta t=0.5$ s), never used for training |

## 2.5. Compared models and experiments

Four models are trained with the same network and optimiser (Table 4). "NN" uses only observations. "NN+BC" additionally receives the initial and boundary conditions but no PDE residual. "PINN" adds the PDE residual at the collocation points. "PINN-only" uses no observations at all and is a pure forward solver. Comparing NN+BC with PINN isolates the effect of the PDE residual; comparing NN with NN+BC isolates the effect of knowing the initial and boundary conditions. All models take $(x,t)$ as input and predict $(h,Q)$.

**Table 4.** Active loss terms of the compared models.

| Model | Data | IC | BC | PDE residual |
|:--|:-:|:-:|:-:|:-:|
| NN | ✓ | – | – | – |
| NN+BC | ✓ | ✓ | ✓ | – |
| PINN | ✓ | ✓ | ✓ | ✓ |
| PINN-only | – | ✓ | ✓ | ✓ |

Three experiments are run. **S1 (dense data, temporal extrapolation):** noise-free random samples within $t\le56$ s; accuracy is reported inside this window and for $56<t\le80$ s, where only the PINN has information (the residual is enforced on the whole domain). **S2 (sparse noisy gauges):** three gauges with measurement noise; the whole space–time field is predicted. **S3 (forward problem):** PINN-only for case A and for case B, which has a different boundary condition and is therefore an unseen flow condition.

## 2.6. Metrics and diagnostics

Accuracy is quantified on the test grid with respect to the reference solution by RMSE, MAE, maximum absolute error, and the perturbation-normalised relative $L_2$ error $\lVert h-h_\mathrm{ref}\rVert_2/\lVert h_\mathrm{ref}-h_0\rVert_2$ (and analogously for $Q$); normalising by the perturbation avoids the trivially high scores that the $R^2$ of a field dominated by the base state would give [@botchkarev2019]. Physics-based diagnostics are: (i) the RMS of the dimensionless residuals $R_1$, $R_2$ of Eq. (5), evaluated with analytic derivatives of each network on the test grid; (ii) the global mass-balance error, $\max_t\left|V(t)-V(0)-\int_0^t[Q(0,s)-Q(L,s)]ds\right|$ with $V=B\int_0^L h\,dx$, expressed relative to the net inflow volume; (iii) the RMS inlet-discharge and outlet-depth boundary errors and the initial-condition error; and (iv) the minimum predicted depth. Results are means ± standard deviations over three seeds. Training and inference times are measured on one CPU core of the computing environment used here.

# 3. Results

## 3.1. Reference solution

Figure 2 shows the space–time structure of the reference solution for case A. The disturbance generated at the inlet propagates downstream as a smooth wave of height about 0.17 m and widening with distance; the depth near the inlet at $t=80$ s (about 2.17 m) is still below the new normal depth (2.52 m) because the flow is still developing. The surface plots are $h(x,t)$ and $Q(x,t)$ of a one-dimensional problem, drawn over the $(x,t)$ plane; they do not represent three-dimensional flow. Because the initial boundary-forcing has a derivative discontinuity at $t=0$, a weak discontinuity travels along the leading characteristic; this feature matters for the analysis of the learned models below.

![**Figure 2.** Reference solution for case A: (a) depth and (b) discharge in the $x$–$t$ plane; (c) depth profiles at selected times.](figs/fig2_reference.png){width=6.3in}

## 3.2. Forward solution without data (S3)

Without any observation, the PINN reproduces both gate operations (Table 5, Figure 3). For case A the depth RMSE is @@S3A_H@@ mm and the discharge RMSE @@S3A_Q@@ m³/s; for case B, whose boundary condition was not used in any other experiment, the corresponding values are @@S3B_H@@ mm and @@S3B_Q@@ m³/s. The remaining error is concentrated along the leading wave front (Figure 4), where the solution has a weak discontinuity that a smooth tanh network represents only approximately (spectral bias [@rahaman2019]); elsewhere, the error is below the thickness of the line in Figure 3.

**Table 5.** Forward PINN (no observations) versus the reference solution; mean ± s.d. over three seeds, whole test grid.

| Case | RMSE $h$ (mm) | MAE $h$ (mm) | max $|e_h|$ (mm) | rel. $L_2$ $h$ (%) | RMSE $Q$ (m³/s) | rel. $L_2$ $Q$ (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
@@T5ROWS@@

![**Figure 3.** Forward PINN (no data) for case A versus the reference solution at three times (best seed).](figs/fig_s3_case_A.png){width=6.3in}

![**Figure 4.** Forward PINN (no data), depth error in the $x$–$t$ plane for (left) case A and (right) case B, best seed each; colour scale in mm. The error is concentrated along the leading wave front.](figs/fig_s3_err_AB.png){width=6.3in}

## 3.3. Sparse, noisy gauge records (S2)

With three noisy gauges, the models differ markedly in depth (Table 6, Figures 5–7). The data-only network interpolates between the gauges without physical guidance and shows depth errors of up to @@S2NN_MAX@@ mm, mainly near the inlet and outlet where no gauge is located. Providing the initial and boundary conditions reduces the depth RMSE by a factor of about @@S2F1@@. Adding the PDE residual lowers it by a further factor of about @@S2F2@@, to @@S2PINN_H@@ mm, which is smaller than the 10 mm standard deviation of the gauge noise. For discharge the gain from the PDE residual over NN+BC is modest (RMSE @@S2NNBC_Q@@ versus @@S2PINN_Q@@ m³/s), so the advantage of the PDE residual in this experiment is clearest for water depth.

**Table 6.** Scenario S2 (three noisy gauges); mean ± s.d. over three seeds, whole test grid.

| Model | RMSE $h$ (mm) | max $|e_h|$ (mm) | rel. $L_2$ $h$ (%) | RMSE $Q$ (m³/s) | rel. $L_2$ $Q$ (%) |
|:--|:-:|:-:|:-:|:-:|:-:|
@@T6ROWS@@

![**Figure 5.** Scenario S2: depth and discharge profiles at $t=40$, 60 and 80 s for the reference solution, NN, NN+BC and PINN (best seed of each).](figs/fig_s2_snapshots.png){width=6.3in}

![**Figure 6.** Scenario S2: depth error in the $x$–$t$ plane for NN, NN+BC and PINN (best seed of each; colour scale in mm).](figs/fig_s2_err.png){width=6.3in}

![**Figure 7.** Scenario S2: noisy gauge observations and model predictions at the gauge locations (seed 0).](figs/fig_s2_gauges.png){width=6.3in}

## 3.4. Dense noise-free data and temporal extrapolation (S1)

With 3000 noise-free samples inside $t\le56$ s, the data-only network is the most accurate model inside the training window (depth RMSE @@S1NN_W@@ mm versus @@S1PINN_W@@ mm for the PINN, Table 7); the PINN does not improve on it where data are plentiful, and adding the boundary information without the PDE residual makes the window error larger (@@S1NNBC_W@@ mm). For $56<t\le80$ s, where no observations exist, the ranking reverses: the PINN reaches @@S1PINN_E@@ mm, against @@S1NN_E@@ mm for NN and @@S1NNBC_E@@ mm for NN+BC, and its error is almost unchanged from the window value, whereas the errors of the two other networks grow by factors of about @@S1G_NN@@ and @@S1G_NNBC@@. This is the behaviour expected when the physics, not the data, carries the information.

**Table 7.** Scenario S1 (3000 noise-free samples, $t\le56$ s); depth and discharge RMSE inside the training window and in the extrapolation window ($56<t\le80$ s); mean ± s.d. over three seeds.

| Model | RMSE $h$, window (mm) | RMSE $h$, extrapolation (mm) | RMSE $Q$, window (m³/s) | RMSE $Q$, extrapolation (m³/s) |
|:--|:-:|:-:|:-:|:-:|
@@T7ROWS@@

![**Figure 8.** Scenario S1: profiles at $t=40$, 60 and 80 s (best seed of each model). Data are available up to $t=56$ s.](figs/fig_s1_snapshots.png){width=6.3in}

## 3.5. Physics-based diagnostics

Table 8 compares the models on diagnostics that do not depend on pointwise agreement with the numerical solution. The PDE-residual and mass-balance errors of the PINN variants are roughly 5 to 80 times smaller than those of the networks without the PDE term in the same experiment, and the PINN honours the boundary conditions to about 0.05 m³/s. The inlet-discharge error of the PINN (@@BC_PINN@@ m³/s in S2) is however larger than that of NN+BC (@@BC_NNBC@@ m³/s) possibly because the PINN trades the boundary term against the residual (we did not test this explanation); no model enforces the boundary condition exactly. For reference, the finite-volume solution has a mass-balance error of the order of $10^{-5}$ when evaluated with the same post-processing, so the PINN is conservative to a level of about @@MASSPINN@@, but not to the level of a conservative numerical scheme. Predicted depths remained physically admissible (minimum depth at or above 1.87 m, which occurs for the gate-closing case; at or above 1.98 m in all gate-opening experiments).

**Table 8.** Physics-based diagnostics on the test grid; mean ± s.d. over three seeds. Residuals are dimensionless (Eq. (5)); the mass-balance error is the maximum over time relative to the net inflow volume.

| Experiment / model | RMS $R_1$ | RMS $R_2$ | mass-balance error | inlet $Q$ error (m³/s) | outlet $h$ error (mm) |
|:--|:-:|:-:|:-:|:-:|:-:|
@@T8ROWS@@

![**Figure 9.** Physics diagnostics (mean ± s.d. over three seeds, log scale).](figs/fig_physics_diag.png){width=6.3in}

## 3.6. Computational cost

Table 9 reports run times on one CPU core. The finite-volume solver needs between 0.1 and 1.6 s for the grids of Table 2, whereas training a PINN takes about @@TRAIN@@ s with the present NumPy implementation; evaluating a trained network on all 32,361 test points takes about @@INFER@@ s. For the forward problem, the PINN is therefore not faster than the numerical solver, and its training cost must be amortised over use cases the solver cannot serve efficiently, such as repeated assimilation of sparse data. Times will differ on other hardware and with graphics-processor implementations; the ratios are what we consider informative, and a direct comparison for operational use would require identical hardware and optimised implementations of both approaches.

**Table 9.** Computational cost on one CPU core (float64).

| Task | Time |
|:--|:-:|
| Finite-volume solution, $N=250$ ($\Delta x=2$ m), 80 s | @@T250@@ s |
| Finite-volume solution, $N=2000$ ($\Delta x=0.25$ m), 80 s | @@T2000@@ s |
| NN training (3000 samples, 6000 iterations) | @@TR_NN@@ s |
| PINN training (4096 collocation points, 6000 iterations) | @@TRAIN@@ s |
| Network evaluation on 32,361 points | @@INFER@@ s |

# 4. Discussion

**What the evidence supports.** Three findings are supported by the experiments. First, a PINN with a correctly formulated residual and fully specified boundary conditions solves the sub-critical Saint-Venant problem without data to about 1 mm in depth, including for a flow condition (gate closing) that was not used elsewhere. Second, with sparse and noisy data, the PDE residual provides information that neither the data nor the boundary conditions alone provide: depth errors fall by roughly an order of magnitude relative to the data-only network and by a factor of about @@S2F2@@ relative to the network with boundary information. Third, with dense noise-free data, the data-only network is at least as accurate within the sampled region, and the PINN's benefit lies in extrapolation and in the physics-based diagnostics rather than in pointwise accuracy.

**Smoothness is not physical realism.** The PINN solutions are smooth, but the error maps show that most of the remaining error sits at the leading wave front. Smoothing there can come from spectral bias of the network, from the weak weighting of a narrow feature among 4096 collocation points, or from the numerical diffusion of the first-order boundary treatment in the reference solution itself [@rahaman2019; @wang2024; @wu2023]. We therefore do not interpret smoothness as evidence of better physics; the interpretation is limited to the quantitative diagnostics of Section 3.5. Because the waves here are smooth, the results do not show how the method performs for bores, hydraulic jumps or transcritical flow, where PINNs are known to be more difficult to train [@mao2020; @krish2021].

**Consistency with earlier PINN work.** The finding that physics terms matter most when data are scarce is consistent with the data-assimilation motivation of earlier SV-based PINN studies [@feng2023; @fang2023]. The finding that a PINN is not faster than a conventional solver for the forward problem is consistent with systematic comparisons in other PDE classes [@grossmann2024].

**Limitations.** (i) All observations are synthetic and generated by a numerical solver of the same equations; the results therefore measure how well a network recovers the SV solution, not how well it describes a real canal. (ii) A single prismatic rectangular channel with constant roughness is used; non-prismatic, trapezoidal or looped geometries, lateral inflows, gates and structures are not covered. (iii) The waves are smooth and the flow is sub-critical; shocks and flow-regime transitions are not examined. (iv) The horizon is 80 s, short compared with operational canal dynamics, and the outlet condition is inactive during this time (Section 2.2). (v) Loss weights were not tuned or adapted, the architecture was not optimised, and three seeds give only a coarse estimate of the variability. (vi) The implementation is a CPU-only NumPy code, so the timings are indicative. (vii) The mass-balance error of the PINN is small but not at the level of a conservative scheme, and the boundary conditions are enforced softly.

# 5. Conclusions

For unsteady sub-critical flow in a rectangular irrigation canal, a PINN that solves the same Saint-Venant system as the reference solver reproduced the reference solution to about 1 mm in depth and 0.05–0.07 m³/s in discharge without training data, for gate-opening and gate-closing waves. With three noisy gauge records, adding the PDE residual reduced the depth error to about @@S2PINN_H@@ mm, compared with @@S2NN_H@@ mm for a data-only network and @@S2NNBC_H@@ mm for a network with initial and boundary information. With dense noise-free data, the data-only network was more accurate inside the sampled region, and the PINN was more accurate when extrapolating in time and satisfied the equations and mass balance more closely. The PINN was not faster than the numerical solver for the forward problem. These conclusions apply to the conditions that were examined: synthetic data, one channel geometry, smooth waves and a short horizon. Field gauge data, other cross-sections, flows with shocks, longer horizons, adaptive loss weighting or causal training [@wang2024], domain decomposition [@jagtap2020] and parametric operator learning [@li2021; @lu2021b] are the natural next steps.

**Author Contributions:** [to be completed by the authors]

**Funding:** [to be completed by the authors]

**Data Availability Statement:** The source code (reference solver, PINN implementation, experiment scripts, figure scripts) and all generated datasets and result files that support this study are provided with the revised submission and will be deposited in a public repository with a persistent identifier (Zenodo DOI: [to be inserted by the authors before submission]). Every run is deterministic given the seeds listed in Table 3.

**Use of Generative AI:** Generative AI tools (Anthropic Claude) were used for language editing and to assist with code development and analysis. The authors reviewed and verified all content and take full responsibility for it. [Authors: please confirm that this statement is accurate.]

**Conflicts of Interest:** The authors declare no conflicts of interest.

# References

@@REFLIST@@
"""


def build():
    t = TXT
    rep = {
        "S3A_H": ms("S3", "PINN-only", "A", "rmse_h_all", 1000, 1),
        "S3A_Q": ms("S3", "PINN-only", "A", "rmse_Q_all", 1, 3),
        "S3B_H": ms("S3", "PINN-only", "B", "rmse_h_all", 1000, 1),
        "S3B_Q": ms("S3", "PINN-only", "B", "rmse_Q_all", 1, 3),
        "S2NN_H": ms("S2", "NN", "A", "rmse_h_all", 1000, 1),
        "S2NNBC_H": ms("S2", "NN+BC", "A", "rmse_h_all", 1000, 1),
        "S2PINN_H": ms("S2", "PINN", "A", "rmse_h_all", 1000, 1),
        "S2NNBC_Q": ms("S2", "NN+BC", "A", "rmse_Q_all", 1, 3),
        "S2PINN_Q": ms("S2", "PINN", "A", "rmse_Q_all", 1, 3),
        "S2NN_MAX": f"{mv('S2','NN','A','max_h_all',1000):.0f}",
        "S2F1": f"{mv('S2','NN','A','rmse_h_all')/mv('S2','NN+BC','A','rmse_h_all'):.1f}",
        "S2F2": f"{mv('S2','NN+BC','A','rmse_h_all')/mv('S2','PINN','A','rmse_h_all'):.1f}",
        "S1NN_W": ms("S1", "NN", "A", "rmse_h_win", 1000, 2),
        "S1NNBC_W": ms("S1", "NN+BC", "A", "rmse_h_win", 1000, 2),
        "S1PINN_W": ms("S1", "PINN", "A", "rmse_h_win", 1000, 2),
        "S1NN_E": ms("S1", "NN", "A", "rmse_h_ext", 1000, 2),
        "S1NNBC_E": ms("S1", "NN+BC", "A", "rmse_h_ext", 1000, 2),
        "S1PINN_E": ms("S1", "PINN", "A", "rmse_h_ext", 1000, 2),
        "S1G_NN": f"{mv('S1','NN','A','rmse_h_ext')/mv('S1','NN','A','rmse_h_win'):.0f}",
        "S1G_NNBC": f"{mv('S1','NN+BC','A','rmse_h_ext')/mv('S1','NN+BC','A','rmse_h_win'):.0f}",
        "BC_PINN": ms("S2", "PINN", "A", "bc_rms_Qin", 1, 3),
        "BC_NNBC": ms("S2", "NN+BC", "A", "bc_rms_Qin", 1, 3),
        "MASSPINN": f"{mv('S3','PINN-only','A','mass_err_rel'):.0e}".replace("e-0", "×10⁻").replace("e-", "×10⁻"),
        "TRAIN": f"{np.mean([r['train_time'] for k in R if k[1]=='PINN' or k[1]=='PINN-only' for r in R[k]]):.0f}",
        "TR_NN": f"{np.mean([r['train_time'] for r in R[('S1','NN','A')]]):.0f}",
        "INFER": f"{np.mean([r['infer_time'] for k in R for r in R[k]]):.2f}",
        "FR": "0.23",
        "T250": f"{CONV[1]['wall']:.2f}",
        "T2000": "1.6",
        "F40": "218", "F60": "328", "F80": "437",
    }
    for k, v in rep.items():
        t = t.replace(f"@@{k}@@", v)
    t = t.replace("@@CONVROWS@@", "\n".join(f"| {c['N']} | {c['dx']:g} | {c['rmse_h']*1000:.2f} | {c['rmse_Q']:.3f} | {c['wall']:.2f} |" for c in CONV))
    t = t.replace("@@T5ROWS@@", "\n".join(
        f"| {c} | {ms('S3','PINN-only',c,'rmse_h_all',1000,2)} | {ms('S3','PINN-only',c,'mae_h_all',1000,2)} | {ms('S3','PINN-only',c,'max_h_all',1000,1)} | {ms('S3','PINN-only',c,'rel_h_all',100,2)} | {ms('S3','PINN-only',c,'rmse_Q_all',1,3)} | {ms('S3','PINN-only',c,'rel_Q_all',100,2)} |" for c in "AB"))
    t = t.replace("@@T6ROWS@@", "\n".join(
        f"| {m} | {ms('S2',m,'A','rmse_h_all',1000,2)} | {ms('S2',m,'A','max_h_all',1000,1)} | {ms('S2',m,'A','rel_h_all',100,1)} | {ms('S2',m,'A','rmse_Q_all',1,3)} | {ms('S2',m,'A','rel_Q_all',100,1)} |" for m in ("NN", "NN+BC", "PINN")))
    t = t.replace("@@T7ROWS@@", "\n".join(
        f"| {m} | {ms('S1',m,'A','rmse_h_win',1000,2)} | {ms('S1',m,'A','rmse_h_ext',1000,2)} | {ms('S1',m,'A','rmse_Q_win',1,3)} | {ms('S1',m,'A','rmse_Q_ext',1,3)} |" for m in ("NN", "NN+BC", "PINN")))
    rows = []
    for (s, m, c) in [("S2", "NN", "A"), ("S2", "NN+BC", "A"), ("S2", "PINN", "A"), ("S1", "NN", "A"), ("S1", "NN+BC", "A"), ("S1", "PINN", "A"), ("S3", "PINN-only", "A"), ("S3", "PINN-only", "B")]:
        rows.append(f"| {s} / {m}" + (f" (case {c})" if s == "S3" else "") + f" | {sci(s,m,c,'pde_rms_R1')} | {sci(s,m,c,'pde_rms_R2')} | {sci(s,m,c,'mass_err_rel')} | {ms(s,m,c,'bc_rms_Qin',1,3)} | {ms(s,m,c,'bc_rms_hout',1000,2)} |")
    t = t.replace("@@T8ROWS@@", "\n".join(rows))
    # references numbering by first appearance
    order = []

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}" if i == j else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
            i = j + 1
        return "[" + ",".join(parts) + "]"
    body, refsec = t.split("# References")
    body = re.sub(r"\[(@[^\]]+)\]", cite, body)
    unused = [k for k in REFS if k not in order]
    reflist = "\n\n".join(f"{i+1}. {REFS[k]}" if False else f"[{i+1}] {REFS[k]}" for i, k in enumerate(order))
    t = body + "# References\n\n" + reflist + "\n"
    open("paper_revised.md", "w").write(t)
    print("refs cited:", len(order), "unused:", unused)


build()
