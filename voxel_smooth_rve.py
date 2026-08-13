# Smoothed-voxel C3D8R periodic RVE for the Xal_multi.for (HLIM) UMAT.
#
# The Xal_multi ecosystem needs C3D8R hexes with 6-face adjacency, so the
# conforming tet route is out. Instead: voxelize the SAME periodic Laguerre
# microstructure (seeds from rve.tess), then smooth the staircase grain
# boundaries while preserving periodicity exactly:
#   - node positions live on an N^3 torus grid -> opposite cube faces are
#     literally the same stored coordinates (+L), so they match bitwise
#   - nodes on the three cut planes only move within their plane -> the
#     domain stays a perfect cube
#   - GB nodes are smoothed hierarchically: 2-grain interface nodes with
#     same-interface neighbours, 3-grain triple-line nodes along the line,
#     >=4-grain junction nodes fixed -> interfaces relax to smooth surfaces
#     without shrinking the grain topology
#   - total displacement capped at CAP*h so hexes stay valid
#
# usage: python voxel_smooth_rve.py [N]        (default 32)
import sys
import numpy as np
from build_cube_rve import tess_read_seeds, read_orientations

L = 1.0
MAXDISP = 0.95      # max total GB displacement, element sizes
STEP = 0.2          # max GB-node move per pass, in element sizes
NPASS = 6           # projection/relaxation passes
RELAX_ITERS = 6     # interior laplacian iterations per pass
LAM = 0.5           # laplacian step (interior relaxation)
LAMT = 0.4          # tangential (in-plane) smoothing step for GB nodes
OUT = "xal"


def voxelize(N):
    seeds = tess_read_seeds("rve.tess")
    S, G, W = [], [], []
    for g, (x, y, z, w) in enumerate(seeds, start=1):
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    S.append([x + sx, y + sy, z + sz])
                    G.append(g)
                    W.append(w)
    S, G, W = np.array(S), np.array(G), np.array(W)
    h = L / N
    c = (np.arange(N) + 0.5) * h
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # Laguerre: owner = argmin |x-s|^2 - w^2   (chunked to bound memory)
    grain = np.empty(len(P), dtype=int)
    for k in range(0, len(P), 4096):
        d = ((P[k:k + 4096, None, :] - S[None, :, :]) ** 2).sum(-1) - W ** 2
        grain[k:k + 4096] = G[np.argmin(d, axis=1)]
    return grain.reshape(N, N, N), S, G, W


def classify_nodes(grain, N):
    """Per torus node: adjacent-cell grain set -> class size, key id, and
    the grain tuple for each key id."""
    adj = np.empty((8, N, N, N), dtype=int)
    n = 0
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                adj[n] = np.roll(grain, (-di, -dj, -dk), axis=(0, 1, 2))
                n += 1
    adj = np.sort(adj, axis=0)
    cls = (np.diff(adj, axis=0) != 0).sum(axis=0) + 1
    keys = np.zeros((N, N, N), dtype=np.int64)
    seen = {}
    flat = adj.reshape(8, -1)
    kflat = keys.reshape(-1)
    for idx in range(flat.shape[1]):
        t = tuple(np.unique(flat[:, idx]))
        kflat[idx] = seen.setdefault(t, len(seen))
    keylist = [None] * len(seen)
    for t, kid in seen.items():
        keylist[kid] = t
    return cls, keys, keylist


def smooth(grain, N, S, G, W):
    """Project GB nodes toward the analytic Laguerre interfaces (planes for
    2-grain nodes, plane intersections for triple lines), moving in bounded
    steps alternated with interior relaxation so hexes never invert. Total
    displacement capped at MAXDISP*h: exact planarity is topologically
    impossible on a fixed voxel grid, this is the validated compromise."""
    h = L / N
    g0 = np.stack(np.meshgrid(*([np.arange(N) * h] * 3), indexing="ij"),
                  axis=-1)
    p = g0.copy()
    cls, keys, keylist = classify_nodes(grain, N)

    imgs = {g: S[G == g] for g in np.unique(G)}
    wt = {g: W[G == g][0] for g in np.unique(G)}

    gbi = np.where((cls == 2) | (cls == 3))
    targets = np.zeros((len(gbi[0]), 3))
    for n, (i, j, k) in enumerate(zip(*gbi)):
        x0 = g0[i, j, k]
        gs = keylist[keys[i, j, k]]
        srefs, cs = [], []
        for g in gs:
            d = ((imgs[g] - x0) ** 2).sum(1) - wt[g] ** 2
            b = np.argmin(d)
            srefs.append(imgs[g][b])
            cs.append((imgs[g][b] ** 2).sum() - wt[g] ** 2)
        nplanes = min(len(gs) - 1, 3)
        A = np.array([2 * (srefs[m] - srefs[0]) for m in range(1, nplanes + 1)])
        b = np.array([cs[m] - cs[0] for m in range(1, nplanes + 1)])
        try:
            x = x0 + A.T @ np.linalg.solve(A @ A.T, b - A @ x0)
        except np.linalg.LinAlgError:
            x = x0 + np.linalg.lstsq(A, b - A @ x0, rcond=None)[0]
        dp = x - x0
        if i == 0:
            dp[0] = 0.0
        if j == 0:
            dp[1] = 0.0
        if k == 0:
            dp[2] = 0.0
        r = np.linalg.norm(dp)
        if r > MAXDISP * h:
            dp *= MAXDISP * h / r
        targets[n] = x0 + dp
    print(f"{len(targets)} GB nodes, median projection distance "
          f"{np.median(np.linalg.norm(targets - g0[gbi], axis=1)) / h:.2f} h")

    dirs = [(a, s) for a in range(3) for s in (1, -1)]

    def wrap_corr(a, s):
        corr = np.zeros((N, N, N, 3))
        idx = [slice(None)] * 3
        idx[a] = N - 1 if s == 1 else 0
        corr[tuple(idx) + (a,)] = L * s
        return corr

    corrs = {(a, s): wrap_corr(a, s) for a, s in dirs}
    interior = cls == 1
    for _ in range(NPASS):
        cur = p[gbi]
        d = targets - cur
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        step = np.minimum(dn, STEP * h)
        p[gbi] = cur + d * step / np.maximum(dn, 1e-30)
        for _ in range(RELAX_ITERS):
            ssum = np.zeros_like(p)
            for a, s in dirs:
                ssum += np.roll(p, -s, axis=a) + corrs[(a, s)]
            dp = LAM * (ssum / 6.0 - p)
            dp[0, :, :, 0] = 0.0
            dp[:, 0, :, 1] = 0.0
            dp[:, :, 0, 2] = 0.0
            p = np.where(interior[..., None], p + dp, p)
    return p


def hex_quality(nodes, conn):
    """(min scaled corner jacobian, mask of hexes below 0.05)."""
    # C3D8 corner triads (node, and its 3 edge-connected corners, RH order)
    triads = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
              (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
    X = nodes[conn]                       # (nel, 8, 3)
    per_el = np.full(len(conn), 1.0)
    for c, a, b, d in triads:
        e1, e2, e3 = (X[:, a] - X[:, c], X[:, b] - X[:, c], X[:, d] - X[:, c])
        det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        norm = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
                * np.linalg.norm(e3, axis=1))
        per_el = np.minimum(per_el, det / np.maximum(norm, 1e-300))
    return per_el.min(), per_el < 0.05


def build(N):
    import os
    os.makedirs(OUT, exist_ok=True)
    grain, S, G, W = voxelize(N)
    ng = len(np.unique(grain))
    print(f"voxelized {N}^3, {ng} grains")
    p = smooth(grain, N, S, G, W)
    print("smoothing done")

    # instantiate (N+1)^3 nodes from the torus grid: exact periodic match
    M = N + 1
    nid = lambda i, j, k: 1 + i + M * (j + M * k)
    torus_of = np.zeros((M ** 3, 3), dtype=int)

    def instantiate(p):
        coords = np.zeros((M ** 3, 3))
        for k in range(M):
            for j in range(M):
                for i in range(M):
                    q = p[i % N, j % N, k % N].copy()
                    q[0] += L if i == N else 0.0
                    q[1] += L if j == N else 0.0
                    q[2] += L if k == N else 0.0
                    coords[nid(i, j, k) - 1] = q
                    torus_of[nid(i, j, k) - 1] = (i % N, j % N, k % N)
        return coords

    conn, egrain = [], []
    for k in range(N):
        for j in range(N):
            for i in range(N):
                conn.append([nid(i, j, k), nid(i + 1, j, k),
                             nid(i + 1, j + 1, k), nid(i, j + 1, k),
                             nid(i, j, k + 1), nid(i + 1, j, k + 1),
                             nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
                egrain.append(grain[i, j, k])
    conn = np.array(conn)
    egrain = np.array(egrain)

    # untangle: pull nodes of ill-shaped hexes back toward the regular grid
    g0 = np.stack(np.meshgrid(*([np.arange(N) * (L / N)] * 3), indexing="ij"),
                  axis=-1)
    coords = instantiate(p)
    for it in range(25):
        mj, bad = hex_quality(coords, conn - 1)
        if mj > 0.05:
            break
        tor = torus_of[np.unique(conn[bad] - 1)]
        u = p - g0
        u[tor[:, 0], tor[:, 1], tor[:, 2]] *= 0.6
        p = g0 + u
        coords = instantiate(p)
    print(f"min scaled jacobian after smoothing/untangling: {mj:.3f} "
          f"({it} pullback iterations)")
    assert mj > 0.05, "smoothing produced degenerate hexes"

    with open(f"{OUT}/rve_nodes.inp", "w") as f:
        f.write("**\n*Node\n")
        for n, (x, y, z) in enumerate(coords, start=1):
            f.write(f"{n}\t,\t{x:.10f}\t,\t{y:.10f}\t,\t{z:.10f}\n")
        f.write("**\n")   # libs.readmesh needs a non-node terminator line
    with open(f"{OUT}/rve_elems.inp", "w") as f:
        f.write("**\n*Element, type=C3D8R\n")
        for e, c in enumerate(conn, start=1):
            f.write(f"   {e},  " + ",  ".join(map(str, c)) + "\n")
        f.write("**\n")
    with open(f"{OUT}/grainElset.inp", "w") as f:
        for g in np.unique(egrain):
            els = np.where(egrain == g)[0] + 1
            f.write(f"*Elset, elset=grain{g}\n")
            for k in range(0, len(els), 16):
                f.write("  " + ", ".join(map(str, els[k:k + 16])) + "\n")
    with open(f"{OUT}/elemGrain", "w") as f:
        f.write("elemId,grainId\n")
        for e, g in enumerate(egrain, start=1):
            f.write(f"{e},{g}\n")
    ori = read_orientations()
    with open(f"{OUT}/aeuler", "w") as f:
        f.write(" KFLAG: 1 --- SAME CRYSTAL OR SET OF CRYSTALS\n"
                "        2 --- DIFFERENT CRYSTALS AT INTEGRATION P\n 2\n"
                " NCRYS\n 1\n"
                " EULER ANGLES FOR EACH CRYSTAL AT AN INTEGRATION P\n"
                "    THETA       PHI      OMEGA\n")
        for g in egrain:
            e1, e2, e3 = ori[g]
            f.write(f"{e1} {e2} {e3}\n")

    # PBC nsets: structured numbering -> sorted ids already correspond
    sets = {"X0": [], "XL": [], "Y0": [], "YL": [], "Z0": [], "ZL": [],
            "X0Y0": [], "X0YL": [], "XLY0": [], "XLYL": [],
            "X0Z0": [], "X0ZL": [], "XLZ0": [], "XLZL": [],
            "Y0Z0": [], "Y0ZL": [], "YLZ0": [], "YLZL": [],
            "X0Y0Z0": [], "X0Y0ZL": [], "X0YLZ0": [], "X0YLZL": [],
            "XLY0Z0": [], "XLY0ZL": [], "XLYLZ0": [], "XLYLZL": []}
    for k in range(M):
        for j in range(M):
            for i in range(M):
                tags = []
                if i == 0:
                    tags.append("X0")
                elif i == N:
                    tags.append("XL")
                if j == 0:
                    tags.append("Y0")
                elif j == N:
                    tags.append("YL")
                if k == 0:
                    tags.append("Z0")
                elif k == N:
                    tags.append("ZL")
                if tags:
                    sets["".join(tags)].append(nid(i, j, k))
    with open(f"{OUT}/PBC_nset.inp", "w") as f:
        for name, ids in sets.items():
            f.write(f"*Nset, nset={name}, instance=PART-1-1\n")
            for k in range(0, len(ids), 8):
                f.write(",".join(map(str, ids[k:k + 8])) + ",\n")
    ngb = count_gb_pairs(egrain, N)
    print(f"wrote {OUT}/: {M ** 3} nodes, {N ** 3} C3D8R, {ng} grains, "
          f"{ngb} GB element pairs")
    print(f"umatcomm_p needs: MAXNOEL>={N ** 3}, numgb>={ngb}, "
          f"maxgrain>={ng}, mxelgrain>={np.bincount(egrain).max()}, "
          f"ch_L={L / N:g}")


def count_gb_pairs(egrain, N):
    # each unordered element pair shares one face -> counted once per axis roll
    g = egrain.reshape(N, N, N)   # axis labels irrelevant for the count
    return sum((np.roll(g, -1, axis=a) != g).sum() for a in range(3))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    build(N)
