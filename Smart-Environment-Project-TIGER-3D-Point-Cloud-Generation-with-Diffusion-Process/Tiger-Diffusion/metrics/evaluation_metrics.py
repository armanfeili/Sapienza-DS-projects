# ────────────────────────────────────────────────────────────────────
#  NumPy-2 compatibility shim – restore numpy.rec & numpy.strings
# ────────────────────────────────────────────────────────────────────
import numpy as _np, types as _t

for _name, _module in [
    ("rec",     "numpy.core.records"),
    ("strings", "numpy.core.defchararray")
]:
    if not hasattr(_np, _name):
        try:
            m = __import__(_module, fromlist=["dummy"])
        except ModuleNotFoundError:
            m = _t.ModuleType(_name)
        setattr(_np, _name, m)
# ────────────────────────────────────────────────────────────────────

import torch, warnings, numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_euclidean_distance as _l2
from metrics.ChamferDistancePytorch.fscore import fscore   # pure-Torch helper

# ─────────────────────────────────────────────────────────────
#  Pure-PyTorch Chamfer-3D
# ─────────────────────────────────────────────────────────────
class _ChamferPure(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x, y : [B, N, 3]
        dist = torch.cdist(x, y, p=2) ** 2          # squared L2
        dl = dist.min(dim=2).values                 # x → y
        dr = dist.min(dim=1).values                 # y → x
        return dl, dr, None, None

cham3D = _ChamferPure()

# ─────────────────────────────────────────────────────────────
#  Lightweight EMD proxy (nearest-neighbour cost, symmetric)
# ─────────────────────────────────────────────────────────────
def EMD(x: torch.Tensor, y: torch.Tensor, transpose: bool = False):
    """
    Simple EMD proxy: for each batch, compute the minimal L2‐distance
    between points of x→y and y→x, then average.
    Returns a (B,) tensor.
    """
    # x, y : [B, N, 3]
    # compute full Euclidean distances [B, N, N]
    dist = torch.cdist(x, y, p=2)
    # minimal distance from each x to its nearest y, and vice versa
    d_xy = dist.min(dim=2).values   # [B, N]
    d_yx = dist.min(dim=1).values   # [B, N]
    # average per batch and take symmetric mean
    return 0.5 * (d_xy.mean(dim=1) + d_yx.mean(dim=1))   # [B]

# ------------------------------------------------------------------
#  Chamfer helper used by AtlasNet
# ------------------------------------------------------------------
def distChamfer(a: torch.Tensor, b: torch.Tensor):
    x, y = a, b
    bs, num_points, _ = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag = torch.arange(num_points, device=x.device).long()
    rx = xx[:, diag, diag].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag, diag].unsqueeze(1).expand_as(yy)
    P  = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]

# ------------------------------------------------------------------
#  Batch metrics
# ------------------------------------------------------------------
def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N = sample_pcs.shape[0]
    assert N == ref_pcs.shape[0], f"REF:{ref_pcs.shape[0]} SMP:{N}"
    cd_lst, emd_lst, fs_lst = [], [], []
    for b in range(0, N, batch_size):
        s, r = sample_pcs[b:b+batch_size], ref_pcs[b:b+batch_size]
        dl, dr, _, _ = cham3D(s.cuda(), r.cuda())
        cd_lst.append(dl.mean(1) + dr.mean(1))
        fs_lst.append(fscore(dl, dr)[0].cpu())
        emd_lst.append(EMD(s.cuda(), r.cuda()))
    cd  = torch.cat(cd_lst).mean() if reduced else torch.cat(cd_lst)
    emd = torch.cat(emd_lst).mean() if reduced else torch.cat(emd_lst)
    fs  = torch.cat(fs_lst).mean()
    return {'MMD-CD': cd, 'MMD-EMD': emd, 'fscore': fs}

def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size):
    N_s, N_r = sample_pcs.shape[0], ref_pcs.shape[0]
    all_cd, all_emd = [], []
    for i in tqdm(range(N_s)):
        s = sample_pcs[i]
        cd_row, emd_row = [], []
        for j in range(0, N_r, batch_size):
            r = ref_pcs[j:j+batch_size]
            bs = r.size(0)
            s_exp = s.view(1, -1, 3).expand(bs, -1, -1).contiguous()
            dl, dr, _, _ = cham3D(s_exp.cuda(), r.cuda())
            cd_row.append((dl.mean(1) + dr.mean(1)).view(1, -1).cpu())
            emd_row.append(EMD(s_exp.cuda(), r.cuda()).view(1, -1).cpu())
        all_cd.append(torch.cat(cd_row, 1))
        all_emd.append(torch.cat(emd_row, 1))
    return torch.cat(all_cd, 0), torch.cat(all_emd, 0)

# ------------------------------------------------------------------
#  K-NN helper
# ------------------------------------------------------------------
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0, n1 = Mxx.size(0), Myy.size(0)
    label  = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1),
                   torch.cat((Mxy.t(), Myy), 1)), 0)
    if sqrt: M = M.abs().sqrt()
    INF = float('inf')
    _, idx = (M + torch.diag(INF * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)
    count  = torch.zeros(n0 + n1, device=Mxx.device)
    for i in range(k):
        count += label.index_select(0, idx[i])
    pred = (count >= k / 2).float()
    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }
    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall'   : s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t'    : s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f'    : s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc'      : (pred == label).float().mean(),
    })
    return s

def lgan_mmd_cov(all_dist):
    N_s, N_r = all_dist.shape
    min_s, idx = all_dist.min(1)
    min_r, _   = all_dist.min(0)
    return {
        'lgan_mmd'     : min_r.mean(),
        'lgan_cov'     : torch.tensor(float(idx.unique().numel()) / N_r, device=all_dist.device),
        'lgan_mmd_smp' : min_s.mean(),
    }

def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    res = {}
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)
    res.update({f"{k}-CD" : v for k, v in lgan_mmd_cov(M_rs_cd.t()).items()})
    res.update({f"{k}-EMD": v for k, v in lgan_mmd_cov(M_rs_emd.t()).items()})
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)
    res.update({f"1-NN-CD-{k}": v for k, v in knn(M_rr_cd, M_rs_cd, M_ss_cd, 1).items() if 'acc' in k})
    res.update({f"1-NN-EMD-{k}": v for k, v in knn(M_rr_emd, M_rs_emd, M_ss_emd, 1).items() if 'acc' in k})
    return res

# ------------------------------------------------------------------
#  JSD helpers
# ------------------------------------------------------------------
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    grid = np.empty((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / (resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k] = (i * spacing - 0.5,
                                 j * spacing - 0.5,
                                 k * spacing - 0.5)
    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]
    return grid, spacing

def entropy_of_occupancy_grid(pcs, res, in_sphere=False):
    eps, bound = 1e-4, 0.5 + 1e-4
    if abs(pcs).max() > bound: warnings.warn('Point-clouds not in unit cube.')
    if in_sphere and np.sqrt((pcs**2).sum(2)).max() > bound:
        warnings.warn('Point-clouds not in unit sphere.')
    grid, _ = unit_cube_grid_point_cloud(res, in_sphere)
    grid = grid.reshape(-1, 3)
    counters  = np.zeros(len(grid))
    bern_vars = np.zeros(len(grid))
    nn = NearestNeighbors(n_neighbors=1).fit(grid)
    for pc in pcs:
        _, idx = nn.kneighbors(pc)
        idx = idx.squeeze()
        counters[idx] += 1
        bern_vars[np.unique(idx)] += 1
    acc_ent = sum(entropy([g/len(pcs), 1-g/len(pcs)]) for g in bern_vars if g > 0)
    return acc_ent / len(counters), counters

def jensen_shannon_divergence(P, Q):
    P_, Q_ = P / P.sum(), Q / Q.sum()
    M = 0.5 * (P_ + Q_)
    return 0.5 * (entropy(P_, M, base=2) + entropy(Q_, M, base=2))

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    sg = entropy_of_occupancy_grid(sample_pcs, resolution, True)[1]
    rg = entropy_of_occupancy_grid(ref_pcs,    resolution, True)[1]
    return jensen_shannon_divergence(sg, rg)

# ------------------------------------------------------------------
#  Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    B, N = 2, 10
    x, y = torch.rand(B, N, 3, device='cuda'), torch.rand(B, N, 3, device='cuda')
    dl, dr, _, _ = cham3D(x, y)
    print("Chamfer:", dl.mean().item(), dr.mean().item())
    print("EMD proxy:", EMD(x, y).mean().item())
    print("JSD:", jsd_between_point_cloud_sets(x.cpu().numpy(), y.cpu().numpy()))
