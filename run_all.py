import os, sys, json, time, numpy as np
import study as st
ITERS = 6000
JOBS = []
for seed in (0, 1, 2):
    for m in ("NN", "NN+BC", "PINN"):
        JOBS.append(("S2", m, "A", seed))
for seed in (0, 1, 2):
    for case in ("A", "B"):
        JOBS.append(("S3", "PINN-only", case, seed))
for seed in (0, 1, 2):
    for m in ("NN", "NN+BC", "PINN"):
        JOBS.append(("S1", m, "A", seed))
def log(s):
    if "it=  1000" in s or "it=  3000" in s or "it=  6000" in s or "it=     1" in s:
        print(s, flush=True)
for scen, m, case, seed in JOBS:
    tag = f"{scen}_{m.replace('+','p')}_{case}_s{seed}"
    fn = f"results/{tag}.json"
    if os.path.exists(fn):
        continue
    sc, net, info = st.run(scen, m, case, seed=seed, iters=ITERS, log=log)
    res, (x, t, H, Q) = st.evaluate(sc, net, case)
    res.update(info); res.update(dict(scenario=scen, method=m, case=case, seed=seed, iters=ITERS))
    np.savez_compressed(f"fields/{tag}.npz", x=x, t=t, H=H.astype(np.float32), Q=Q.astype(np.float32))
    st.save_json(res, fn)
    print("DONE", tag, "rmse_h_all=%.4f rmse_Q_all=%.4f train=%.0fs" % (res["rmse_h_all"], res["rmse_Q_all"], res["train_time"]), flush=True)
print("ALL FINISHED", flush=True)
