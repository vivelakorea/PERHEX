# Web assets for the curved-GB results: (a) the curved GB quad network
# of the f3 n=2 projected mesh (grain-colored, per-quad unwrap), (b) the
# T4 percolating matrix grain's voxel shell (the topological limit
# case). -> docs/assets/mesh_pf.json for the three.js viewer.
import json
import numpy as np

OUT = "O:/prj_rve/mprve-public/docs/assets/mesh_pf.json"
FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]

# --- (a) curved GB network from the torus mesh ---
d = np.load("pf_snap_f3_mesh_n2p.npz")
coords, conn, grain = d["coords"], d["conn"], d["grain"]
facemap = {}
for e in range(len(conn)):
    for f in FACES:
        nid = conn[e, list(f)]
        facemap.setdefault(tuple(sorted(nid)), []).append((e, nid))
curved = {}
for v in facemap.values():
    if len(v) != 2 or grain[v[0][0]] == grain[v[1][0]]:
        continue
    e, nid = v[0]
    g = int(grain[e])
    pts = coords[nid].copy()
    # per-quad unwrap: bring all 4 vertices to the first vertex's image
    pts -= np.round(pts - pts[0])
    t = np.concatenate([pts[[0, 1, 2]], pts[[0, 2, 3]]]).ravel()
    curved.setdefault(g, []).append(np.round(t, 5))
curved = {g: np.concatenate(a).tolist() for g, a in curved.items()}
nq = sum(len(v) // 18 for v in curved.values())
print(f"curved GB quads: {nq}")

# --- (b) T4 percolating matrix grain shell (voxel faces) ---
d5 = np.load("gg_fewgrains_f5.npz")
gid = d5["gid"]
N = gid.shape[0]
gbig = np.bincount(gid.ravel()).argmax()
m = gid == gbig
print(f"T4 matrix grain {gbig}: {m.mean()*100:.0f}% of volume")
tris = []
h = 1.0 / N
for ax in range(3):
    nb = np.roll(m, -1, axis=ax)
    for i, j, k in np.argwhere(m & ~nb):
        base = np.array([i, j, k], float)
        base[ax] += 1
        o1, o2 = [a for a in range(3) if a != ax]
        q = []
        for da, db in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            p = base.copy()
            p[o1] += da
            p[o2] += db
            q.append(p * h)
        q = np.array(q)
        tris.append(np.concatenate([q[[0, 1, 2]], q[[0, 2, 3]]]).ravel())
    for i, j, k in np.argwhere(m & ~np.roll(m, 1, axis=ax)):
        base = np.array([i, j, k], float)
        o1, o2 = [a for a in range(3) if a != ax]
        q = []
        for da, db in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            p = base.copy()
            p[o1] += da
            p[o2] += db
            q.append(p * h)
        q = np.array(q)
        tris.append(np.concatenate([q[[0, 1, 2]], q[[0, 2, 3]]]).ravel())
perc = np.round(np.concatenate(tris), 5).tolist()
print(f"T4 shell tris: {len(perc)//9}")

with open(OUT, "w") as f:
    json.dump({"curved": curved, "perc": {str(gbig): perc}}, f,
              separators=(",", ":"))
import os
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} kB)")
