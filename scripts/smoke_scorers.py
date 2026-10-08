import time, sys, numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.data.dataset import load_elliptic_dataset
from src.training.walkforward import WF, run
ds = load_elliptic_dataset()
names = sys.argv[1:] or ["lr","rf","xgb","mlp","mcd","sage","evolve","delayed"]
for s in names:
    cfg = WF(s, "F_all" if s not in ("sage","evolve") else "F_local", D=2, Wc=4, R=5, era=(1,34), seeds=(0,), origins=(30,), hp={"M":3} if s in("xgb","mlp","sage") else {})
    t=time.time(); torch.cuda.reset_peak_memory_stats()
    df = run(ds, cfg, verbose=False)
    d = df[df.period>=30]
    print(f"{s:8s} {time.time()-t:6.1f}s  gpuMB={torch.cuda.max_memory_allocated()/1e6:6.0f}  AUROC={roc_auc_score(d.y,d.p):.3f} AUPRC={average_precision_score(d.y,d.p):.3f}  mean_p={d.p.mean():.3f} mean_u={d.u.mean():.5f} rows={len(df)}", flush=True)
