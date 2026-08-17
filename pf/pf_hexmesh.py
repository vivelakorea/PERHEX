# Midpoint hex mesh on the verified curved junction complex (pf_topology).
#   python pf_hexmesh.py pf_snap_f3.npz [n_refine] [ratio]
#
# v0 (n=1): one hex per (grain g, generic quad point q), corners
#   (q, m(e1), c(f12), m(e2), m(e3), c(f13), C_g, c(f23))
# per gen_midpoint_hex.py's pattern; handedness fix [0,3,2,1,4,7,6,5].
# Periodicity: one canonical wrapped position per construction point shared
# via entity key -> conforming on the torus; per-hex geometry lives in the
# grain's unwrapped frame (nearest image w.r.t. the grain center; validated
# against half-box).
#
# v1 apex fix: the grain-centroid apex folds far thin corners (r_qp/r_g>1).
# For each grain with an inverted hex, relocate its (shared) apex by
# maximizing the grain's worst Gauss scaled Jacobian (Nelder-Mead); all
# sibling hexes move consistently because they share the apex node. Grains
# where no apex position fixes the fold are reported as genuine
# topology-adaptation cases.
#
# n>1: n^3 graded trilinear subdivision in reference coords, one GLOBAL
# geometric schedule (ratio toward t=0; the three local coordinate planes
# through the q corner are exactly the GB quads, as in gen_midpoint_hex) ->
# shared faces get identical grids -> conforming + periodic preserved.
# Then GB-face-interior nodes are projected onto the OP iso-surface
# op_a = op_b of their face's pair (Newton along the gradient, displacement
# capped at CAPFRAC of the local grid spacing); quality reported with and
# without projection.
import sys
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

f = sys.argv[1] if len(sys.argv) > 1 else "pf_snap_f3.npz"
NREF = int(sys.argv[2]) if len(sys.argv) > 2 else 1
RATIO = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
CAPFRAC = 0.35   # ponytail: fixed projection cap; expose if projection folds
d = np.load(f.replace(".npz", "_complex.npz"), allow_pickle=True)
N = int(d["N"])
qp_pos, qp_gr, qp_ng = d["qp_pos"], d["qp_grains"], d["qp_ng"]
qp_tris = d["qp_tris"]
tri_mid, tri_trips, tri_faces = d["tri_mid"], d["tri_trips"], d["tri_faces"]
face_cent, face_pairs = d["face_cent"], d["face_pairs"]
grain_ids, grain_cent = d["grain_ids"], d["grain_cent"]
grain_nvox, grain_op = d["grain_nvox"], d["grain_op"]
grain_wrap = (d["grain_wrap"] if "grain_wrap" in d.files
              else np.zeros(len(grain_ids), int))
g2c = {int(g): grain_cent[i].copy() for i, g in enumerate(grain_ids)}
g2v = {int(g): int(grain_nvox[i]) for i, g in enumerate(grain_ids)}
g2o = {int(g): int(grain_op[i]) for i, g in enumerate(grain_ids)}
g2w = {int(g): int(grain_wrap[i]) for i, g in enumerate(grain_ids)}
wrapg = [g for g in g2w if g2w[g] > 0]
if wrapg:
    unm = sum(g2v[g] for g in wrapg) / N ** 3
    print(f"PERCOLATING grains {wrapg} (wrap rank > 0): single-apex "
          f"midpoint construction is topologically impossible for them -> "
          f"SKIPPED ({unm:.1%} of the volume unmeshed)")
nqp = len(qp_pos)
print(f"{f}: N={N}, {nqp} quad points, {len(tri_mid)} tri instances, "
      f"{len(face_cent)} face instances, {len(grain_ids)} grains, "
      f"n={NREF} ratio={RATIO}")

# ---------------- 2x2x2 Gauss machinery -------------------------------------
S = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
DN = []
for gp in S / np.sqrt(3.0):
    dN = np.empty((8, 3))
    for a in range(8):
        dN[a] = S[a] * [(1 + gp[1] * S[a, 1]) * (1 + gp[2] * S[a, 2]),
                        (1 + gp[0] * S[a, 0]) * (1 + gp[2] * S[a, 2]),
                        (1 + gp[0] * S[a, 0]) * (1 + gp[1] * S[a, 1])]
    DN.append(dN / 8.0)
DN = np.array(DN)                                   # (8gp, 8node, 3)


def quality(X):
    """X (m,8,3) -> min Gauss det, min scaled Jacobian, volume."""
    mindet = np.full(len(X), np.inf)
    minsj = np.full(len(X), np.inf)
    vol = np.zeros(len(X))
    for dN in DN:
        J = np.einsum("nai,aj->nji", X, dN)
        det = np.linalg.det(J)
        ln = np.linalg.norm(J, axis=2).prod(1)
        mindet = np.minimum(mindet, det)
        minsj = np.minimum(minsj, det / ln)
        vol += det
    return mindet, minsj, vol


def histo(sj, label):
    hist, edges = np.histogram(sj, bins=[-1, 0, .05, .1, .2, .3, .4, .5, 1])
    print(f"min-SJ histogram ({label}): " + "  ".join(
        f"[{lo:.2f},{hi:.2f}):{h}" for h, lo, hi in
        zip(hist, edges[:-1], edges[1:])))


def wd(p, ref):
    return ref + (np.asarray(p) - ref + N / 2.0) % N - N / 2.0


# ---------------- v0 assembly ------------------------------------------------
nodes, coords = {}, []


def nid(key, pos):
    i = nodes.get(key)
    if i is None:
        i = len(nodes)
        nodes[key] = i
        coords.append(np.asarray(pos) % N)
    return i


conn, egrain, hexX, hexF = [], [], [], []
skip_deg = skip_amb = 0
maxspan = 0.0
for q in range(nqp):
    if qp_ng[q] != 4:
        skip_deg += 1
        continue
    tmap, ok = {}, True
    for t in qp_tris[q]:
        key = frozenset(tri_trips[t])
        if key in tmap:
            ok = False
        tmap[key] = t
    if not ok or len(qp_tris[q]) != 4:
        skip_amb += 1
        continue
    gs = sorted(qp_gr[q])
    for g in gs:
        if g2w.get(int(g), 0) > 0:
            continue
        a, b, c = [x for x in gs if x != g]
        try:
            e1, e2, e3 = (tmap[frozenset((g, a, b))],
                          tmap[frozenset((g, a, c))],
                          tmap[frozenset((g, b, c))])
        except KeyError:
            skip_amb += 1
            continue

        def face_of(ex, ey, pair):
            cand = [fi for fi in set(tri_faces[ex]) & set(tri_faces[ey])
                    if set(face_pairs[fi]) == set(pair)]
            return cand[0] if len(cand) == 1 else None

        f12, f13, f23 = (face_of(e1, e2, (g, a)), face_of(e1, e3, (g, b)),
                         face_of(e2, e3, (g, c)))
        if None in (f12, f13, f23):
            skip_amb += 1
            continue
        keys = [("q", q), ("m", e1), ("f", f12), ("m", e2),
                ("m", e3), ("f", f13), ("c", g), ("f", f23)]
        raw = [qp_pos[q], tri_mid[e1], face_cent[f12], tri_mid[e2],
               tri_mid[e3], face_cent[f13], g2c[g], face_cent[f23]]
        C = g2c[g]
        X = np.array([wd(p, C) for p in raw])
        maxspan = max(maxspan, np.abs(X - C).max())
        ids = [nid(k, p) for k, p in zip(keys, raw)]
        conn.append(ids)
        egrain.append(g)
        hexX.append(X)
        hexF.append((f12, f13, f23))    # face instances at corners 2, 5, 7
coords = np.array(coords)
conn = np.array(conn)
egrain = np.array(egrain)
hexX = np.array(hexX)
hexF = np.array(hexF)
# handedness by CELL VOLUME, not the q-corner tetra: at thin far corners
# that tetra is a degenerate sliver whose sign can disagree with the cell
# (this exact misclassification produced all v0 "inversions" on f0/f5/f7:
# corner-0 det ~ +1 while the other 7 corners were ~ -100s)
_, _, v8 = quality(hexX)
flip = v8 < 0
perm = [0, 3, 2, 1, 4, 7, 6, 5]         # q stays 0, apex stays 6
conn[flip] = conn[flip][:, perm]
hexX[flip] = hexX[flip][:, perm]
hexF[flip] = hexF[flip][:, [0, 2, 1]]   # f13 <-> f23 swap with m1 <-> m2
print(f"{len(conn)} coarse hexes ({len(coords)} nodes); skipped {skip_deg} "
      f"degenerate vertices, {skip_amb} ambiguous corners; "
      f"max span {maxspan:.1f} vox (< {N // 2} ok); "
      f"{int(flip.sum())} cells volume-flipped")

# ---------------- v1: apex fix, then targeted untangling ---------------------
mindet, minsj, _ = quality(hexX)
bad_g = sorted(set(int(g) for g in egrain[mindet <= 0]))
print(f"after volume flip: {int((mindet <= 0).sum())} inverted "
      f"in grains {bad_g}")
for g in bad_g:
    idx = np.where(egrain == g)[0]
    Xg = hexX[idx].copy()

    def obj(C):
        Xg[:, 6, :] = C
        _, sj, _ = quality(Xg)
        return -sj.min()

    C0 = np.array(g2c[g])
    res = minimize(obj, C0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    Xg[:, 6, :] = res.x
    md1, sj1, _ = quality(Xg)
    if md1.min() > 0 and np.linalg.norm(res.x - C0) < 8:
        hexX[idx, 6, :] = res.x
        coords[conn[idx[0], 6]] = res.x % N
        g2c[g] = res.x
        print(f"  grain {g}: apex moved {np.linalg.norm(res.x - C0):.2f} vox"
              f" -> min SJ {sj1.min():.3f}")
    else:
        print(f"  grain {g}: apex move insufficient "
              f"(best min det {md1.min():.1f}) -> node untangler")

# targeted untangler: for hexes still inverted, relocate their nodes one at
# a time (bounded |move| <= UCAP vox) maximizing the min SJ of ALL hexes
# sharing the node; conformity survives because the node moves everywhere.
UCAP = 2.0   # ponytail: fidelity cap in voxels; raise only with a report
mindet, _, _ = quality(hexX)
if (mindet <= 0).any():
    n2h = defaultdict(list)
    for h in range(len(conn)):
        for s in range(8):
            n2h[int(conn[h, s])].append((h, s))
    moved = {}
    for sweep in range(4):
        mindet, msj, _ = quality(hexX)
        badh = np.where(mindet <= 0)[0]
        if not len(badh):
            break
        cand = sorted({int(conn[h, s]) for h in badh for s in range(8)})
        for nd in cand:
            hs = n2h[nd]
            hids = [h for h, _ in hs]
            X0 = hexX[hids].copy()

            def obj(delta):
                Xl = X0.copy()
                for k_, (h, s) in enumerate(hs):
                    Xl[k_, s] += delta
                _, sj, _ = quality(Xl)
                pen = max(0.0, np.linalg.norm(delta) - UCAP) * 100
                return -sj.min() + pen

            base = -obj(np.zeros(3))
            res = minimize(obj, np.zeros(3), method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-5,
                                    "maxiter": 200})
            if -res.fun > base + 1e-6:
                delta = res.x
                if np.linalg.norm(delta) > UCAP:
                    delta *= UCAP / np.linalg.norm(delta)
                for h, s in hs:
                    hexX[h, s] += delta
                coords[nd] = (coords[nd] + delta) % N
                moved[nd] = moved.get(nd, 0) + np.linalg.norm(delta)
    if moved:
        print(f"untangler: moved {len(moved)} nodes, "
              f"max total move {max(moved.values()):.2f} vox")
mindet, minsj, vol = quality(hexX)
print(f"v1: min SJ {minsj.min():.4f}, median {np.median(minsj):.4f}, "
      f"inverted {int((mindet <= 0).sum())}/{len(conn)}, "
      f"volume {vol.sum() / N ** 3:.4f}")
histo(minsj, "v1 coarse")
inv = mindet <= 0
if inv.any():
    print("remaining inverted (grain, r_qp/r_g, minSJ):")
    for i in np.where(inv)[0]:
        g = egrain[i]
        rg = (3 * g2v[g] / (4 * np.pi)) ** (1 / 3)
        rq = np.linalg.norm(hexX[i][0] - np.array(g2c[g]))
        print(f"   grain {g:3d}  r_qp/r_g={rq / rg:5.2f}  "
              f"minSJ={minsj[i]:7.3f}")


def conformity(cn):
    FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    fc = defaultdict(int)
    for e in cn:
        for fq in FACES:
            fc[frozenset(int(e[i]) for i in fq)] += 1
    un = sum(1 for v in fc.values() if v == 1)
    ov = sum(1 for v in fc.values() if v > 2)
    print(f"conformity: {len(fc)} quads, {un} unmatched (holes), "
          f"{ov} overshared (must be 0)")


conformity(conn)

if NREF == 1:
    out = f.replace(".npz", "_mesh.npz")
    np.savez_compressed(out, coords=coords / N, conn=conn, grain=egrain,
                        minsj=minsj, mindet=mindet, N=N)
    print("saved", out)
    sys.exit(0)

# ---------------- n^3 graded refinement --------------------------------------
n = NREF
if RATIO > 1.0:
    t = (RATIO ** np.arange(n + 1) - 1.0) / (RATIO ** n - 1.0)
else:
    t = np.arange(n + 1) / n
print(f"refining n={n}, layer fractions {np.round(np.diff(t), 3)}")

snap = np.load(f)
Nc = snap["ops"].shape[1] - 1                       # coarse OP grid (40)
OPS = snap["ops"].astype(np.float64)[:, :Nc, :Nc, :Nc]


def opsample(o, p80):
    """periodic trilinear sample of OP o at position in N-lattice units"""
    x = np.asarray(p80) * (Nc / N)
    i0 = np.floor(x).astype(int)
    fr = x - i0
    v = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[0] if dx else 1 - fr[0])
                     * (fr[1] if dy else 1 - fr[1])
                     * (fr[2] if dz else 1 - fr[2]))
                v += w * OPS[o, (i0[0] + dx) % Nc, (i0[1] + dy) % Nc,
                             (i0[2] + dz) % Nc]
    return v


fnodes, fcoords = {}, []
fconn, fX, fgrain = [], [], []
proj = {}          # node id -> (op_a, op_b, cap)
pair_clash = 0
loc = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)


def fnid(p):
    key = tuple(np.round(np.round(np.asarray(p) % N, 6) % N, 6))
    i = fnodes.get(key)
    if i is None:
        i = len(fnodes)
        fnodes[key] = i
        fcoords.append(np.asarray(p) % N)
    return i


for h in range(len(conn)):
    P = hexX[h]
    g = int(egrain[h])

    def tri(x, y, z):
        s = np.array([(1 - x) * (1 - y) * (1 - z), x * (1 - y) * (1 - z),
                      x * y * (1 - z), (1 - x) * y * (1 - z),
                      (1 - x) * (1 - y) * z, x * (1 - y) * z,
                      x * y * z, (1 - x) * y * z])
        return s @ P
    grid = {}
    gpos = {}
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                p = tri(t[i], t[j], t[k])
                gpos[(i, j, k)] = p
                grid[(i, j, k)] = fnid(p)
    # GB planes through corner 0: k=0 -> face at corner2, j=0 -> corner5,
    # i=0 -> corner7 (hexF recorded post-handedness)
    nid2pair = {}
    for plane, fi in zip(("k", "j", "i"), hexF[h]):
        pa, pb = face_pairs[fi]
        nid2pair[plane] = (g2o[int(pa)], g2o[int(pb)])
    for (i, j, k), nd in grid.items():
        nz = (i == 0, j == 0, k == 0)
        if sum(nz) != 1:
            continue
        plane = "i" if nz[0] else ("j" if nz[1] else "k")
        oa, ob = nid2pair[plane]
        # local spacing: min distance to adjacent grid nodes in the plane
        dirs = {"i": [(0, 1, 0), (0, 0, 1)], "j": [(1, 0, 0), (0, 0, 1)],
                "k": [(1, 0, 0), (0, 1, 0)]}[plane]
        dm = np.inf
        for dx_ in dirs:
            for sgn in (1, -1):
                nb = (i + sgn * dx_[0], j + sgn * dx_[1], k + sgn * dx_[2])
                if nb in gpos:
                    dm = min(dm, np.linalg.norm(gpos[(i, j, k)] - gpos[nb]))
        if nd in proj and proj[nd][:2] != (oa, ob) \
                and proj[nd][:2] != (ob, oa):
            pair_clash += 1
            continue
        proj[nd] = (oa, ob, min(proj.get(nd, (0, 0, np.inf))[2],
                                CAPFRAC * dm))
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cid = [grid[tuple(np.array([i, j, k]) + lc.astype(int))]
                       for lc in loc]
                cX = [gpos[tuple(np.array([i, j, k]) + lc.astype(int))]
                      for lc in loc]
                fconn.append(cid)
                fX.append(cX)
                fgrain.append(g)
fcoords = np.array(fcoords)
fconn = np.array(fconn)
fX = np.array(fX)
fgrain = np.array(fgrain)
print(f"refined: {len(fconn)} hexes, {len(fcoords)} nodes, "
      f"{len(proj)} GB-face nodes to project ({pair_clash} pair clashes)")
conformity(fconn)
mind0, msj0, vol0 = quality(fX)
print(f"unprojected: min SJ {msj0.min():.4f} median {np.median(msj0):.4f} "
      f"inverted {int((mind0 <= 0).sum())}/{len(fconn)} "
      f"volume {vol0.sum() / N ** 3:.4f}")
histo(msj0, f"n={n} unprojected")
np.savez_compressed(f.replace(".npz", f"_mesh_n{n}.npz"),
                    coords=fcoords / N, conn=fconn, grain=fgrain,
                    minsj=msj0, mindet=mind0, N=N)

# ---------------- OP projection ----------------------------------------------
disp = np.zeros((len(fcoords), 3))
nmoved = nclamp = 0
for nd, (oa, ob, cap) in proj.items():
    x = fcoords[nd].copy()
    x0 = x.copy()
    for _ in range(3):
        F = opsample(oa, x) - opsample(ob, x)
        h_ = 0.4
        gr = np.array([(opsample(oa, x + e) - opsample(ob, x + e)
                        - opsample(oa, x - e) + opsample(ob, x - e))
                       / (2 * h_)
                       for e in np.eye(3) * h_])
        g2 = gr @ gr
        if g2 < 1e-12:
            break
        x = x - F * gr / g2
    dtot = (x - x0 + N / 2) % N - N / 2
    L = np.linalg.norm(dtot)
    if L > cap:
        dtot *= cap / L
        nclamp += 1
    if L > 1e-9:
        disp[nd] = dtot
        nmoved += 1
dl = np.linalg.norm(disp[list(proj)], axis=1)
print(f"projection displacement: mean {dl.mean():.3f} max {dl.max():.3f} vox"
      f" (voxel = {1000 / N:.1f} nm-equiv units)")
fcoords_p = (fcoords + disp) % N
fXp = fX + disp[fconn]
mind1, msj1, vol1 = quality(fXp)
print(f"projected ({nmoved} nodes moved, {nclamp} clamped at "
      f"{CAPFRAC}*local spacing): min SJ {msj1.min():.4f} "
      f"median {np.median(msj1):.4f} "
      f"inverted {int((mind1 <= 0).sum())}/{len(fconn)} "
      f"volume {vol1.sum() / N ** 3:.4f}")
histo(msj1, f"n={n} projected")
out = f.replace(".npz", f"_mesh_n{n}p.npz")
np.savez_compressed(out, coords=fcoords_p / N, conn=fconn, grain=fgrain,
                    minsj=msj1, mindet=mind1, N=N)
print("saved", out, "(+ unprojected _mesh_n%d.npz)" % n)
