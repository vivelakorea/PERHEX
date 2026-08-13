# Staircase-geometry d_b sampled at the MIDPOINT-mesh element centroids:
# the controlled comparison the paper needs. Same mesh, same material,
# same evaluation points as db_table.dat (exact) — only the geometry the
# rays march through is the N=40 voxel field. Prediction differences
# between the two runs are then attributable to d_b corruption alone.
#
# The voxel field is built in the CUT frame from the same generators as
# gen_db_table.py (grain of the power-argmin at voxel centers), which is
# exactly what voxelizing the cut tessellation would give.
import numpy as np
from build_cube_rve import read_orientations
from gen_db_table import LPHYS, NSYS, DIRC, bunge_matrix, build_generators
from gen_db_table_vox import dda, NV

coords = np.load("mp_coords.npy")
conn = np.load("mp_conn.npy")
pid = np.load("mp_pid.npy")
cmap = np.loadtxt("seeds.map", dtype=int)
eg = cmap[pid - 1]
cent = coords[conn - 1].mean(1)

P, W, G, grains = build_generators()
c = W ** 2
pp = np.sum(P * P, 1)

h = 1.0 / NV
vc = (np.stack(np.meshgrid(range(NV), range(NV), range(NV),
                           indexing="ij"), -1).reshape(-1, 3) + 0.5) * h
gfield = np.empty(len(vc), dtype=int)
for lo in range(0, len(vc), 8192):
    hi = min(lo + 8192, len(vc))
    gfield[lo:hi] = G[np.argmin((pp - c)[None, :] - 2 * vc[lo:hi] @ P.T, 1)]
gfield = gfield.reshape(NV, NV, NV)

ori = read_orientations()
ms = np.zeros((len(cent), NSYS, 3))
for g in grains:
    m = DIRC @ bunge_matrix(*np.radians(ori[g]))
    ms[eg == g] = m
X = np.repeat(cent, NSYS, 0)
D = ms.reshape(-1, 3)
G0 = np.repeat(eg, NSYS)

# rays start from the exact-geometry grain; where the voxel field already
# disagrees at the start point, d_b is the distance to where the field
# stops matching that grain — which is immediately. That IS the corruption.
db = np.minimum(dda(X, D, gfield, G0), dda(X, -D, gfield, G0))
db = db.reshape(-1, NSYS) * LPHYS

dbe = np.loadtxt("oxford_mp/db_table.dat")
rel = np.abs(db - dbe) / dbe
print(f"staircase-vs-exact d_b at midpoint centroids:")
print(f"  rel. error mean {np.nanmean(rel) * 100:.1f}%  "
      f"median {np.nanmedian(rel) * 100:.1f}%  "
      f"p90 {np.nanpercentile(rel, 90) * 100:.1f}%")
np.savetxt("db_table_stair_mp.dat", db, fmt="%.6e")
print(f"wrote db_table_stair_mp.dat ({db.shape[0]} elements)")
