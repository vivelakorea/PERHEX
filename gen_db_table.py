# Per-element, per-slip-system distance to the nearest *real* grain
# boundary along the slip direction (d_b), for the Haouala-type GB storage
# term rho_dot = (1/b) max(k1 sqrt(rho_f), K_s/d_b) - k2 rho |gdot|.
#
# Geometry source: the periodic Laguerre tessellation itself (exact planes),
# reconstructed from rve_cut.tess cell seeds (grain base seed = any cell
# seed wrapped mod 1; Neper Laguerre power = |x-s|^2 - w^2). A ray from the
# element centroid along +/- the slip direction is walked cell-to-cell;
# bisectors between periodic images of the SAME grain are not boundaries
# (the crystal continues), so the walk passes through them.
#
# Slip directions: OXFORD-UMAT FCC systems 1-12 (globalvariables.f dir2),
# rotated to the sample frame with the grain's Bunge angles (m_s = g^T m_c).
# d_b = min over the two senses, in micrometers with LPHYS the physical
# cube edge. Output db_table.dat: numel rows x 12 columns, element order =
# deck order.
import numpy as np
from build_cube_rve import tess_read_seeds, read_orientations

LPHYS = 25.0        # physical cube edge in um -> mean grain size ~9.2 um
NSYS = 12
EPS = 1e-12

# OXFORD FCC <110>{111}, rows 1-12 of dir2 (globalvariables.f)
DIRC = np.array([
    [1, -1, 0], [0, 1, -1], [1, 0, -1],
    [1, 1, 0], [0, 1, -1], [1, 0, 1],
    [1, 1, 0], [0, 1, 1], [1, 0, -1],
    [1, -1, 0], [0, 1, 1], [1, 0, 1]], float)
DIRC /= np.linalg.norm(DIRC, axis=1)[:, None]


def bunge_matrix(phi1, Phi, phi2):
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([
        [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s, -c1 * s, c]])


def build_generators():
    cs = tess_read_seeds("rve_cut.tess")            # (ncell, 4) x y z w
    cmap = np.loadtxt("seeds.map", dtype=int)       # cell -> grain
    grains = np.unique(cmap)
    base = {}
    for g in grains:
        s = cs[np.where(cmap == g)[0][0]]
        base[g] = (s[:3] % 1.0, s[3])
    offs = np.array([(i, j, k) for i in (-1, 0, 1)
                     for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    P, W, G = [], [], []
    for g in grains:
        p, w = base[g]
        for o in offs:
            P.append(p + o)
            W.append(w)
            G.append(g)
    return np.array(P), np.array(W), np.array(G), grains


def raytrace(X, D, P, c, gid):
    """min positive t to a different-grain Laguerre boundary, per ray."""
    n = len(X)
    x = X.copy()
    t_acc = np.zeros(n)
    active = np.ones(n, bool)
    # power argmin without the |x|^2 term: |P|^2 - w^2 - 2 x.P
    pp = np.sum(P * P, 1)
    cur = np.argmin((pp - c)[None, :] - 2 * x @ P.T, 1)
    g0 = gid[cur]
    for _ in range(12):
        if not active.any():
            break
        idx = np.where(active)[0]
        xa, da, ca = x[idx], D[idx], cur[idx]
        A = xa @ P.T
        B = da @ P.T
        numer = (pp - c)[None, :] - (pp - c)[ca][:, None] \
            - 2 * (A - A[np.arange(len(idx)), ca][:, None])
        denom = 2 * (B - B[np.arange(len(idx)), ca][:, None])
        t = np.where(denom > EPS, numer / denom, np.inf)
        t[t <= EPS] = np.inf
        j = np.argmin(t, 1)
        tj = t[np.arange(len(idx)), j]
        t_acc[idx] += tj
        hit = gid[j] != g0[idx]
        done = idx[hit]
        active[done] = False
        cont = idx[~hit]
        if len(cont):
            x[cont] = (x[cont] + (tj[~hit] + 1e-9)[:, None] * D[cont]) % 1.0
            cur[cont] = j[~hit]
    t_acc[active] = np.inf  # never exited (should not happen)
    return t_acc


def main():
    coords = np.load("mp_coords.npy")
    conn = np.load("mp_conn.npy")
    pid = np.load("mp_pid.npy")
    cmap = np.loadtxt("seeds.map", dtype=int)
    egrain = cmap[pid - 1]
    cent = coords[conn - 1].mean(1)
    P, W, G, grains = build_generators()
    c = W ** 2                      # power(x) = |x-p|^2 - w^2
    pp = np.sum(P * P, 1)

    # self-check: nearest-power generator grain == deck grain
    chk = G[np.argmin(pp[None, :] - c[None, :] - 2 * cent @ P.T, 1)]
    bad = np.sum(chk != egrain)
    print(f"centroid grain check: {bad}/{len(cent)} mismatches")
    assert bad == 0, "Laguerre reconstruction disagrees with mesh"

    ori = read_orientations()
    ms = np.zeros((len(cent), NSYS, 3))
    for g in grains:
        gmat = bunge_matrix(*np.radians(ori[g]))
        m = DIRC @ gmat            # rows: (g^T m_c)^T = m_c^T g
        ms[egrain == g] = m
    X = np.repeat(cent, NSYS, 0)
    D = ms.reshape(-1, 3)
    db = np.empty(len(X))
    CH = 30000
    for lo in range(0, len(X), CH):
        hi = min(lo + CH, len(X))
        tp = raytrace(X[lo:hi], D[lo:hi], P, c, G)
        tm = raytrace(X[lo:hi], -D[lo:hi], P, c, G)
        db[lo:hi] = np.minimum(tp, tm)
        print(f"  rays {hi}/{len(X)}")
    db = db.reshape(-1, NSYS) * LPHYS
    print(f"d_b (um): min {db.min():.4f}  median {np.median(db):.3f}  "
          f"max {db.max():.3f}")
    np.savetxt("oxford_mp/db_table.dat", db, fmt="%.6e")
    print(f"wrote oxford_mp/db_table.dat  ({db.shape[0]} elements)")


if __name__ == "__main__":
    main()
