# d_b table for the VOXEL deck, measured the only way a voxel-mesh user
# can: by marching rays through the staircase grain field. Also computes
# the EXACT Laguerre d_b at the same voxel centers, so the pointwise
# corruption of this model input by the staircase geometry is quantified
# directly (same points, same directions, only the geometry differs).
import numpy as np
from voxel_smooth_rve import voxelize
from build_cube_rve import read_orientations
from gen_db_table import LPHYS, NSYS, DIRC, bunge_matrix, raytrace

NV = 40
STEP = 1.0 / NV / 4.0        # DDA sampling step (unit cube)


def dda(X, D, gfield, g0):
    """distance along D until the sampled voxel grain differs from g0."""
    n = len(X)
    t = np.full(n, np.inf)
    active = np.arange(n)
    s = STEP
    for k in range(1, int(2.0 / STEP)):
        x = X[active] + k * s * D[active]
        idx = np.floor(x * NV).astype(int) % NV
        g = gfield[idx[:, 0], idx[:, 1], idx[:, 2]]
        hit = g != g0[active]
        t[active[hit]] = k * s
        active = active[~hit]
        if not len(active):
            break
    return t


def main():
    # use voxelize's own seed images as Laguerre generators: same frame
    # (rve.tess, original), same power convention, same grain labels
    gfield, P, G, W = voxelize(NV)
    h = 1.0 / NV
    ii, jj, kk = np.meshgrid(range(NV), range(NV), range(NV), indexing="ij")
    # element order in gen_vox_deck_oxford.py: k outer, j, i inner
    order = np.argsort((kk * NV + jj) * NV + ii, axis=None, kind="stable")
    cent = (np.stack([ii, jj, kk], -1).reshape(-1, 3)[order] + 0.5) * h
    eg = gfield.reshape(-1)[order]

    ori = read_orientations()
    ms = np.zeros((len(cent), NSYS, 3))
    for g in np.unique(eg):
        m = DIRC @ bunge_matrix(*np.radians(ori[g]))
        ms[eg == g] = m
    X = np.repeat(cent, NSYS, 0)
    D = ms.reshape(-1, 3)
    G0 = np.repeat(eg, NSYS)

    print("staircase DDA ...")
    dbs = np.minimum(dda(X, D, gfield, G0), dda(X, -D, gfield, G0))
    dbs = dbs.reshape(-1, NSYS) * LPHYS

    print("exact Laguerre ...")
    c = W ** 2
    dbe = np.empty(len(X))
    for lo in range(0, len(X), 30000):
        hi = min(lo + 30000, len(X))
        dbe[lo:hi] = np.minimum(raytrace(X[lo:hi], D[lo:hi], P, c, G),
                                raytrace(X[lo:hi], -D[lo:hi], P, c, G))
    dbe = dbe.reshape(-1, NSYS) * LPHYS

    # voxel centers whose voxel grain matches the Laguerre grain (bulk):
    pp = np.sum(P * P, 1)
    lg = G[np.argmin((pp - c)[None, :] - 2 * cent @ P.T, 1)]
    ok = (eg == lg)
    rel = (dbs[ok] - dbe[ok]) / dbe[ok]
    print(f"voxel-vs-Laguerre grain agreement: {ok.mean() * 100:.1f}%")
    print(f"d_b staircase vs exact (agreeing voxels, {ok.sum()} pts x 12):")
    print(f"  rel. error mean {np.mean(np.abs(rel)) * 100:.1f}%  "
          f"median {np.median(np.abs(rel)) * 100:.1f}%  "
          f"p90 {np.percentile(np.abs(rel), 90) * 100:.1f}%")
    np.savetxt("oxford_vox/db_table.dat", dbs, fmt="%.6e")
    np.save("dbvox_stair.npy", dbs)
    np.save("dbvox_exact.npy", dbe)
    print(f"wrote oxford_vox/db_table.dat ({dbs.shape[0]} elements)")


if __name__ == "__main__":
    main()
