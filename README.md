# Saint-Venant PINN study — reproducibility package
Requirements: Python 3, numpy, scipy, matplotlib (no TensorFlow/PyTorch needed).
1. `python sv_ref.py`       – reference solver self-test (S0, normal depth, Fr, convergence-ready)
2. `python pinn_np.py`      – finite-difference check of the analytic PINN gradient
3. `python run_all.py`      – all 24 runs (S1,S2,S3 x 3 seeds); resumable; writes results/*.json, fields/*.npz (~1.5 h on one CPU core)
4. `python make_figures.py` – figures + grid-convergence table
All runs are deterministic given the seeds (0,1,2). Timings in the paper were measured on one CPU core of a cloud container; re-measure on your hardware.
