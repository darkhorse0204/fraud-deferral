"""Neural scorers: batched deep-ensemble MLP, MC-dropout MLP, batched GraphSAGE ensemble, EvolveGCN-O.

All members of an ensemble are trained simultaneously as one batched model (weights carry a leading member
axis M) with independent initialisation and independent minibatch order. Early stopping uses an internal
TIME-ORDERED hold-out (the latest training periods), tracked per member.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------- helpers
def _bce_pos(logit, y, pw):
    """Per-member mean of pos-weighted BCE. logit,y: (M, n)."""
    w = torch.where(y > 0.5, pw, torch.ones_like(y))
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean(dim=1)


def _init(M, a, b, gen):
    w = torch.randn(M, a, b, generator=gen) * (2.0 / a) ** 0.5
    return nn.Parameter(w.to(DEV))


class _BestKeeper:
    """Per-member best-state tracking for batched models."""

    def __init__(self, params):
        self.params = list(params)
        self.best = [p.detach().clone() for p in self.params]
        self.best_loss = None

    def update(self, loss_m: torch.Tensor) -> None:  # loss_m: (M,)
        if self.best_loss is None:
            self.best_loss = torch.full_like(loss_m, float("inf"))
        imp = loss_m < self.best_loss - 1e-6
        self.best_loss = torch.where(imp, loss_m, self.best_loss)
        for p, b in zip(self.params, self.best):
            shape = (-1,) + (1,) * (p.dim() - 1)
            b.copy_(torch.where(imp.view(shape), p.detach(), b))
        self.n_improved = int(imp.sum())

    def restore(self) -> None:
        with torch.no_grad():
            for p, b in zip(self.params, self.best):
                p.copy_(b)


# ----------------------------------------------------------------------------------------------- MLP
class BMLP(nn.Module):
    def __init__(self, M, d_in, widths, drop, gen):
        super().__init__()
        dims = [d_in] + list(widths) + [1]
        self.W = nn.ParameterList([_init(M, a, b, gen) for a, b in zip(dims[:-1], dims[1:])])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(M, 1, b, device=DEV)) for b in dims[1:]])
        self.drop = drop

    def forward(self, x):  # x: (M,B,d) or (B,d)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.W[0].shape[0], -1, -1)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, W)
            if i < len(self.W) - 1:
                x = F.dropout(F.relu(x), self.drop, self.training)
        return x.squeeze(-1)  # (M,B)


def _split_holdout(periods: np.ndarray, y: np.ndarray, frac=0.15, min_pos=15):
    """Boolean mask of the latest-time hold-out rows inside the training set."""
    up = np.unique(periods)
    k = max(1, int(round(frac * len(up))))
    while k < len(up) - 1 and y[np.isin(periods, up[-k:])].sum() < min_pos:
        k += 1
    ho = np.isin(periods, up[-k:]) if len(up) > 1 else np.zeros(len(periods), bool)
    if ho.sum() == 0 or (~ho).sum() == 0:
        ho = np.zeros(len(periods), bool)
    return ho


def _fwd(net, X, chunk: int = 65536):
    """Forward pass in chunks for large inputs (keeps the ensemble's activations inside a 4 GB GPU).
    Inputs up to `chunk` rows take a single pass, so results on the small Study 1 tables are unchanged."""
    if X.shape[0] <= chunk:
        return net(X)
    return torch.cat([net(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)], dim=1)


def fit_predict_mlp(Xtr, ytr, ptr, Xte, seed: int, hp: dict, M: int = 5, mc: bool = False):
    from src.models.prep import NanStandardizer

    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    ho = _split_holdout(ptr, ytr)
    sc = NanStandardizer().fit(Xtr[~ho] if ho.any() else Xtr)
    Xt = torch.tensor(sc.transform(Xtr[~ho] if ho.any() else Xtr), device=DEV)
    yt = torch.tensor(ytr[~ho] if ho.any() else ytr, dtype=torch.float32, device=DEV)
    Xh = torch.tensor(sc.transform(Xtr[ho]), device=DEV) if ho.any() else None
    yh = torch.tensor(ytr[ho], dtype=torch.float32, device=DEV) if ho.any() else None
    Mm = 1 if mc else M
    net = BMLP(Mm, Xt.shape[1], hp.get("widths", (256, 128)), hp.get("drop", 0.2 if mc else 0.1), gen).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 1e-3), weight_decay=hp.get("wd", 1e-4))
    pos = max(float(yt.sum()), 1.0)
    pw = torch.tensor(((len(yt) - pos) / pos) ** hp.get("pos_weight_pow", 0.5), device=DEV)
    keeper, bs, n = _BestKeeper(net.parameters()), hp.get("bs", 1024), len(yt)
    pat, bad = hp.get("patience", 5), 0
    for ep in range(hp.get("epochs", 30)):
        net.train()
        order = torch.argsort(torch.rand(Mm, n, device=DEV), dim=1)  # independent minibatch order per member
        for k in range(0, n, bs):
            idx = order[:, k:k + bs]
            loss = _bce_pos(net(Xt[idx]), yt[idx], pw).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if Xh is not None:
            net.eval()
            with torch.no_grad():
                keeper.update(_bce_pos(_fwd(net, Xh), yh.unsqueeze(0).expand(Mm, -1), pw))
            bad = 0 if keeper.n_improved else bad + 1
            if bad >= pat:
                break
    if Xh is not None:
        keeper.restore()
    Z = torch.tensor(sc.transform(Xte), device=DEV)
    with torch.no_grad():
        if mc:
            net.train()
            ps = torch.stack([torch.sigmoid(_fwd(net, Z))[0] for _ in range(hp.get("T", 20))])
        else:
            net.eval()
            ps = torch.sigmoid(_fwd(net, Z))
    return ps.mean(0).cpu().numpy().astype(np.float32), ps.var(0).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------------------------- graphs
class PeriodGraphs:
    """Per-period symmetric graphs on GPU. Edges never cross periods in Elliptic (verified, M3)."""

    def __init__(self, ds, cols: np.ndarray, periods: list[int], shuffle_edges: bool = False, seed: int = 0):
        self.items: dict[int, dict] = {}
        ei = ds.edge_index
        ep = ds.period[ei[0]]
        rng = np.random.default_rng(12345 + seed)
        for p in periods:
            nodes = np.flatnonzero(ds.period == p)
            loc = np.full(ds.n, -1, dtype=np.int64)
            loc[nodes] = np.arange(len(nodes))
            e = ei[:, ep == p]
            s, d = loc[e[0]], loc[e[1]]
            if shuffle_edges:  # degree-preserving re-pairing of targets within the period (ablation A10)
                d = d[rng.permutation(len(d))]
            src = np.concatenate([s, d]); dst = np.concatenate([d, s])  # symmetrise
            self.items[p] = dict(nodes=nodes, src=torch.tensor(src, device=DEV), dst=torch.tensor(dst, device=DEV),
                                 n=len(nodes), X=torch.tensor(ds.X[nodes][:, cols], device=DEV),
                                 y=torch.tensor(ds.y[nodes], device=DEV))

    def mean_agg(self, p: int, H: torch.Tensor) -> torch.Tensor:
        """Mean over neighbours. H: (n, k) -> (n, k)."""
        it = self.items[p]
        out = torch.zeros_like(H)
        out.index_add_(0, it["dst"], H[it["src"]])
        deg = torch.zeros(it["n"], device=DEV).index_add_(0, it["dst"], torch.ones(len(it["dst"]), device=DEV))
        return out / deg.clamp(min=1).unsqueeze(1)

    def gcn_agg(self, p: int, H: torch.Tensor) -> torch.Tensor:
        """Symmetric-normalised aggregation with self-loops."""
        it = self.items[p]
        deg = torch.ones(it["n"], device=DEV).index_add_(0, it["dst"], torch.ones(len(it["dst"]), device=DEV))
        dinv = deg.pow(-0.5)
        out = H * (dinv * dinv).unsqueeze(1)
        msg = H[it["src"]] * dinv[it["src"]].unsqueeze(1) * dinv[it["dst"]].unsqueeze(1)
        return out.index_add(0, it["dst"], msg)


class BSage(nn.Module):
    def __init__(self, M, d_in, h, drop, gen):
        super().__init__()
        self.Ws1, self.Wn1 = _init(M, d_in, h, gen), _init(M, d_in, h, gen)
        self.Ws2, self.Wn2 = _init(M, h, h, gen), _init(M, h, h, gen)
        self.Wo = _init(M, h, 1, gen)
        self.b1, self.b2, self.bo = (nn.Parameter(torch.zeros(M, 1, k, device=DEV)) for k in (h, h, 1))
        self.drop, self.M, self.h = drop, M, h

    def forward(self, G: PeriodGraphs, p: int):
        X = G.items[p]["X"]
        AX = G.mean_agg(p, X)
        h1 = F.relu(torch.einsum("nd,mdh->mnh", X, self.Ws1) + torch.einsum("nd,mdh->mnh", AX, self.Wn1) + self.b1)
        h1 = F.dropout(h1, self.drop, self.training)
        flat = h1.permute(1, 0, 2).reshape(X.shape[0], self.M * self.h)
        Ah = G.mean_agg(p, flat).reshape(X.shape[0], self.M, self.h).permute(1, 0, 2)
        h2 = F.relu(torch.bmm(h1, self.Ws2) + torch.bmm(Ah, self.Wn2) + self.b2)
        h2 = F.dropout(h2, self.drop, self.training)
        return (torch.bmm(h2, self.Wo) + self.bo).squeeze(-1)  # (M, n)


def fit_predict_sage(ds, cols, tr_rows_periods, pr_periods, seed: int, hp: dict, M: int = 5, shuffle_edges=False):
    """tr_rows_periods: list of training periods (all nodes of those periods form the inductive training graph;
    loss only on labeled nodes). pr_periods: periods to predict (labeled nodes returned)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    G = PeriodGraphs(ds, cols, sorted(set(tr_rows_periods) | set(pr_periods)), shuffle_edges, seed)
    tr_p = sorted(tr_rows_periods)
    k = max(1, int(round(0.15 * len(tr_p))))
    ho_p, fit_p = tr_p[-k:], tr_p[:-k] or tr_p
    ho_p = ho_p if tr_p[:-k] else []
    allX = torch.cat([G.items[p]["X"] for p in fit_p])
    mu, sd = allX.mean(0), allX.std(0).clamp(min=1e-6)  # train-only standardisation (fit periods)
    for p in set(tr_p) | set(pr_periods):
        G.items[p]["Xs"] = ((G.items[p]["X"] - mu) / sd).clamp(-10, 10)
    for p in G.items:
        G.items[p]["X0"] = G.items[p]["X"]
        G.items[p]["X"] = G.items[p]["Xs"]
    net = BSage(M, len(cols), hp.get("hidden", 64), hp.get("drop", 0.2), gen).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 3e-3), weight_decay=hp.get("wd", 1e-4))
    ys = torch.cat([G.items[p]["y"][G.items[p]["y"] >= 0] for p in fit_p]).float()
    pos = max(float(ys.sum()), 1.0)
    pw = torch.tensor(((len(ys) - pos) / pos) ** hp.get("pos_weight_pow", 0.5), device=DEV)
    keeper, bad, pat = _BestKeeper(net.parameters()), 0, hp.get("patience", 8)

    def per_period_loss(p):
        it = G.items[p]
        lab = it["y"] >= 0
        if lab.sum() == 0:
            return None, 0
        lg = net(G, p)[:, lab]
        yy = it["y"][lab].float().unsqueeze(0).expand(net.M, -1)
        return _bce_pos(lg, yy, pw) * int(lab.sum()), int(lab.sum())

    rng = np.random.default_rng(seed)
    for ep in range(hp.get("epochs", 60)):
        net.train()
        order = list(rng.permutation(fit_p))
        for i in range(0, len(order), hp.get("periods_per_step", 4)):
            tot, cnt = 0, 0
            for p in order[i:i + hp.get("periods_per_step", 4)]:
                l, c = per_period_loss(p)
                if l is not None:
                    tot, cnt = tot + l, cnt + c
            if cnt:
                opt.zero_grad(set_to_none=True)
                (tot / cnt).sum().backward()
                opt.step()
        if ho_p:
            net.eval()
            with torch.no_grad():
                tot, cnt = 0, 0
                for p in ho_p:
                    l, c = per_period_loss(p)
                    if l is not None:
                        tot, cnt = tot + l, cnt + c
                keeper.update(tot / max(cnt, 1))
            bad = 0 if keeper.n_improved else bad + 1
            if bad >= pat:
                break
    if ho_p:
        keeper.restore()
    net.eval()
    rows, ps_all = [], []
    with torch.no_grad():
        for p in pr_periods:
            it = G.items[p]
            lab = it["y"] >= 0
            ps_all.append(torch.sigmoid(net(G, p)[:, lab]))
            rows.append(it["nodes"][lab.cpu().numpy()])
    ps = torch.cat(ps_all, dim=1)
    return np.concatenate(rows), ps.mean(0).cpu().numpy().astype(np.float32), ps.var(0).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------------------------- EvolveGCN-O
class EvolveGCNO(nn.Module):
    """EvolveGCN-O (Pareja et al., AAAI 2020): GCN weights evolve across snapshots by a GRU over the weights."""

    def __init__(self, d_in, h, drop):
        super().__init__()
        self.W1_0, self.W2_0 = nn.Parameter(torch.randn(d_in, h) * (2 / d_in) ** 0.5), nn.Parameter(torch.randn(h, h) * (2 / h) ** 0.5)
        self.g1, self.g2 = nn.GRUCell(d_in, d_in), nn.GRUCell(h, h)
        self.out = nn.Linear(h, 1)
        self.drop = drop

    def init_state(self):
        return self.W1_0, self.W2_0

    def step(self, G: PeriodGraphs, p: int, W1, W2):
        W1 = self.g1(W1.t(), W1.t()).t()
        W2 = self.g2(W2.t(), W2.t()).t()
        X = G.items[p]["X"]
        h = F.dropout(F.relu(G.gcn_agg(p, X @ W1)), self.drop, self.training)
        h = F.dropout(F.relu(G.gcn_agg(p, h @ W2)), self.drop, self.training)
        return self.out(h).squeeze(-1), W1, W2


def fit_predict_evolvegcn(ds, cols, tr_periods, pr_periods, seed: int, hp: dict, M: int = 3):
    torch.manual_seed(seed)
    tr_p = sorted(tr_periods)
    last_tr = tr_p[-1]
    seq_all = list(range(tr_p[0], max(pr_periods) + 1))
    G = PeriodGraphs(ds, cols, seq_all, False, seed)
    k = max(1, int(round(0.15 * len(tr_p))))
    fit_p = tr_p[:-k] or tr_p
    ho_p = tr_p[-k:] if tr_p[:-k] else []
    allX = torch.cat([G.items[p]["X"] for p in fit_p])
    mu, sd = allX.mean(0), allX.std(0).clamp(min=1e-6)
    for p in G.items:
        G.items[p]["X"] = ((G.items[p]["X"] - mu) / sd).clamp(-10, 10)
    ys = torch.cat([G.items[p]["y"][G.items[p]["y"] >= 0] for p in fit_p]).float()
    pos = max(float(ys.sum()), 1.0)
    pw = torch.tensor(((len(ys) - pos) / pos) ** hp.get("pos_weight_pow", 0.5), device=DEV)
    train_seq = list(range(tr_p[0], last_tr + 1))
    preds = []
    for m in range(M):
        torch.manual_seed(seed * 100 + m)
        net = EvolveGCNO(len(cols), hp.get("hidden", 64), hp.get("drop", 0.2)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=hp.get("lr", 3e-3), weight_decay=hp.get("wd", 1e-4))
        best, best_state, bad, chunk = float("inf"), None, 0, hp.get("chunk", 8)

        def run(seq, train: bool):
            W1, W2 = net.init_state()
            tot, cnt, out = 0.0, 0, {}
            for i in range(0, len(seq), chunk):
                loss_c, n_c = 0, 0
                for p in seq[i:i + chunk]:
                    lg, W1, W2 = net.step(G, p, W1, W2)
                    lab = G.items[p]["y"] >= 0
                    if p in (fit_p if train else ho_p) and lab.sum():
                        yy = G.items[p]["y"][lab].float().unsqueeze(0)
                        loss_c = loss_c + _bce_pos(lg[lab].unsqueeze(0), yy, pw).sum() * int(lab.sum())
                        n_c += int(lab.sum())
                    out[p] = lg
                if train and n_c:
                    opt.zero_grad(set_to_none=True)
                    (loss_c / n_c).backward()
                    opt.step()
                if n_c:
                    tot += float(loss_c) if not isinstance(loss_c, int) else 0.0
                    cnt += n_c
                W1, W2 = W1.detach(), W2.detach()
            return tot / max(cnt, 1), out

        for ep in range(hp.get("epochs", 40)):
            net.train()
            run(train_seq, True)
            if ho_p:
                net.eval()
                with torch.no_grad():
                    vl, _ = run(train_seq, False)
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= hp.get("patience", 8):
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            _, out = run(seq_all, False)
        preds.append(out)
    rows, ps = [], []
    for p in pr_periods:
        lab = G.items[p]["y"] >= 0
        rows.append(G.items[p]["nodes"][lab.cpu().numpy()])
        ps.append(torch.stack([torch.sigmoid(o[p][lab]) for o in preds]))
    ps = torch.cat(ps, dim=1)
    return np.concatenate(rows), ps.mean(0).cpu().numpy().astype(np.float32), ps.var(0).cpu().numpy().astype(np.float32)
