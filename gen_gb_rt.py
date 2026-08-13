# Precompute for TRUE per-increment d_b: ray-tracing against the deformed
# GB facets inside the solver. Extracts the midpoint mesh's GB facets
# (element faces between different grains, incl. periodic wrap), unwraps
# each grain into a contiguous frame, ray-traces every (element, slip
# system, +-sense) at t=0 (validated against the exact Laguerre table),
# and stores per-ray facet candidate lists (facets near the +-hit
# segment) so the in-solver rebuild only tests ~dozens of facets per ray.
#
# Facet vertices are NODE-tracked at runtime (user's design): the deck
# writes GB nodal displacements to the .fil every increment, URDFIL reads
# them, and the deformed quad vertex is x = X0(node) + u(node)
# + (F_macro - I).offset for periodic images. No affine reconstruction.
#
# Output gb_rt.bin (little-endian stream, all float64/int32):
#   header: numel, nfacet, nsys, ncand, nnode, lphys
#   o_e(numel,3)      element unwrap offset (F_macro-convected at runtime)
#   V0(numel)         reference volume (F_macro weights)
#   X0(nnode,3)       reference nodal coordinates (wrapped, as meshed)
#   fnode(nfacet,4)   facet vertex node ids (1-based)
#   foff(nfacet,4,3)  integer unwrap offset per facet vertex
#   csr_ptr(numel*nsys+1), csr_idx(ncand)  candidate facets per (el,sys)
# Everything in mesh units (unit cube) to match UMAT COORDS; the solver
# multiplies distances by lphys. gbnodes.txt lists the node ids the deck
# must put in the .fil output nset.
import numpy as np
from build_cube_rve import read_orientations
from gen_db_table import LPHYS, NSYS, DIRC, bunge_matrix

# candidate margin: absolute floor + fraction of the facet's distance
# from the ray origin (intersection drift scales with distance at fixed
# strain; 0.2 covers 10% tension with rotations twice over)
MARG0 = 0.02
MARGREL = 0.2
FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]

coords = np.load("mp_coords.npy")
conn = np.load("mp_conn.npy") - 1          # 0-based
pid = np.load("mp_pid.npy")
cmap = np.loadtxt("seeds.map", dtype=int)
eg = cmap[pid - 1]
nel = len(conn)

# --- face matching (interior exact, boundary via wrapped coordinates) ---
def wkey(nid):
    c = np.round(coords[nid] % 1.0, 8) % 1.0
    return tuple(sorted(map(tuple, c)))

facemap = {}
for e in range(nel):
    for fi, f in enumerate(FACES):
        nid = conn[e, list(f)]
        facemap.setdefault(wkey(nid), []).append((e, fi))
pairs = []
for k, v in facemap.items():
    if len(v) == 2:
        pairs.append(v)
    elif len(v) != 1:
        raise RuntimeError(f"face shared by {len(v)} elements")
print(f"{nel} elements, {len(pairs)} face pairs "
      f"({sum(len(v) == 1 for v in facemap.values())} unmatched)")
assert all(len(v) in (1, 2) for v in facemap.values())

# --- unwrap each grain (BFS over face adjacency incl. periodic jumps) ---
adj = [[] for _ in range(nel)]
for (e1, f1), (e2, f2) in pairs:
    c1 = coords[conn[e1, list(FACES[f1])]].mean(0)
    c2 = coords[conn[e2, list(FACES[f2])]].mean(0)
    off = np.round(c1 - c2)          # 0 interior, +-1 across the cube
    adj[e1].append((e2, off))
    adj[e2].append((e1, -off))
o_e = np.full((nel, 3), np.nan)
conflicts = 0
for seed_el in range(nel):
    if not np.isnan(o_e[seed_el, 0]):
        continue
    o_e[seed_el] = 0.0
    stack = [seed_el]
    while stack:
        e = stack.pop()
        for e2, off in adj[e]:
            if eg[e2] != eg[e]:
                continue
            want = o_e[e] + off
            if np.isnan(o_e[e2, 0]):
                o_e[e2] = want
                stack.append(e2)
            elif not np.allclose(o_e[e2], want):
                conflicts += 1
print(f"unwrap done, {conflicts} conflicting same-grain adjacencies")

cent = coords[conn].mean(1)
Xc = cent + o_e

# --- reference volumes (F_macro weights) ---
GP = np.array([[i, j, k] for k in (-1, 1) for j in (-1, 1)
               for i in (-1, 1)]) / np.sqrt(3.0)
def hexvol(nodes):
    # exact via 1-pt... use 8-pt Gauss determinant sum
    r = GP
    v = np.zeros(len(nodes))
    for g in r:
        dN = np.zeros((8, 3))
        s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        for a in range(8):
            dN[a] = s[a] * np.array(
                [(1 + g[1] * s[a, 1]) * (1 + g[2] * s[a, 2]),
                 (1 + g[0] * s[a, 0]) * (1 + g[2] * s[a, 2]),
                 (1 + g[0] * s[a, 0]) * (1 + g[1] * s[a, 1])]) / 8.0
        J = np.einsum("nai,aj->nij", nodes, dN)
        v += np.linalg.det(J)
    return v
V0 = hexvol(coords[conn])
assert V0.min() > 0
print(f"total volume {V0.sum():.6f}")

# --- GB facets, one copy per side (frame = same-grain element's) ---
# vertex node id + integer unwrap offset: the runtime position is
# X0(node) + u(node) + F_macro.offset (offset absorbs both the owner
# element's unwrap and the node's own periodic-image shift)
fverts, fnode, foff, fgrain = [], [], [], []
for (e1, f1), (e2, f2) in pairs:
    if eg[e1] == eg[e2]:
        continue
    for e, fi in ((e1, f1), (e2, f2)):
        nid = conn[e, list(FACES[fi])]
        v = coords[nid] + o_e[e]
        fverts.append(v)
        fnode.append(nid + 1)                     # 1-based
        foff.append(np.round(v - coords[nid]))    # = o_e[e], per vertex
        fgrain.append(eg[e])
fverts = np.array(fverts)
fnode = np.array(fnode, np.int32)
foff = np.array(foff)
fgrain = np.array(fgrain)
nfac = len(fverts)
gbnodes = np.unique(fnode)
print(f"{nfac} one-sided GB facets, {len(gbnodes)} GB nodes")

# --- slip directions (initial, sample frame) ---
ori = read_orientations()
ms = np.zeros((nel, NSYS, 3))
for g in np.unique(eg):
    ms[eg == g] = DIRC @ bunge_matrix(*np.radians(ori[g]))

# --- t=0 ray-trace (Moller-Trumbore, per grain, vectorized) ---
def mt(orig, dirs, tri):
    """orig (n,3), dirs (n,3), tri (m,3,3) -> t (n,m) inf where miss"""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(dirs[:, None, :], e2[None, :, :])
    det = np.einsum("nmi,mi->nm", p, e1)
    s = orig[:, None, :] - v0[None, :, :]
    u = np.einsum("nmi,nmi->nm", p, s) / det
    q = np.cross(s, e1[None, :, :])
    w = np.einsum("nmi,ni->nm", q, dirs) / det
    t = np.einsum("nmi,mi->nm", q, e2) / det
    eps = 1e-9
    bad = (np.abs(det) < 1e-14) | (u < -eps) | (w < -eps) \
        | (u + w > 1 + eps) | (t <= 1e-9)
    return np.where(bad, np.inf, t)

db0 = np.zeros((nel, NSYS, 2))
hitseg = np.zeros((nel, NSYS, 2, 3))
cands = [None] * (nel * NSYS)
for g in np.unique(eg):
    els = np.where(eg == g)[0]
    fi = np.where(fgrain == g)[0]
    tris = np.concatenate([fverts[fi][:, [0, 1, 2]],
                           fverts[fi][:, [0, 2, 3]]])   # (2m,3,3)
    m = len(fi)
    O = np.repeat(Xc[els], NSYS, 0)
    Dm = ms[els].reshape(-1, 3)
    for sense, dsign in ((0, 1.0), (1, -1.0)):
        t_all = np.full(len(O), np.inf)
        for lo in range(0, len(O), 4000):
            hi = min(lo + 4000, len(O))
            t = mt(O[lo:hi], dsign * Dm[lo:hi], tris)
            t_all[lo:hi] = t.min(1)
        db0[els, :, sense] = t_all.reshape(-1, NSYS)
        hitseg[els, :, sense, :] = (O + dsign * t_all[:, None] * Dm
                                    ).reshape(-1, NSYS, 3)
    # candidate lists: facet within MARGIN of the +- segment
    fc = fverts[fi].mean(1)                      # facet centers (m,3)
    frad = np.linalg.norm(fverts[fi] - fc[:, None], axis=2).max(1)
    for k, e in enumerate(els):
        dorig = np.linalg.norm(fc - Xc[e], axis=1)
        marg = MARG0 + MARGREL * dorig
        for s in range(NSYS):
            a = hitseg[e, s, 0]
            bpt = hitseg[e, s, 1]
            ab = bpt - a
            L2 = ab @ ab
            tproj = np.clip(((fc - a) @ ab) / max(L2, 1e-30), 0, 1)
            d = np.linalg.norm(fc - (a + tproj[:, None] * ab), axis=1)
            sel = np.where(d < frad + marg)[0]
            cands[e * NSYS + s] = fi[sel].astype(np.int32)
    print(f"  grain {g}: {len(els)} el, {m} facets, "
          f"avg cand {np.mean([len(cands[e * NSYS + s]) for e in els for s in range(NSYS)]):.0f}")

assert np.isfinite(db0).all(), "some rays never hit a facet"
# --- validate against the exact Laguerre table ---
dbe = np.loadtxt("oxford_mp/db_table.dat") / LPHYS
rel = np.abs(db0.min(2) - dbe) / dbe
print(f"t=0 facet-trace vs Laguerre: median rel {np.median(rel)*100:.2f}% "
      f"p99 {np.percentile(rel, 99)*100:.2f}% max {rel.max()*100:.2f}%")

# --- write binary ---
ptr = np.zeros(nel * NSYS + 1, np.int32)
for i, c in enumerate(cands):
    ptr[i + 1] = ptr[i] + len(c)
idx = np.concatenate(cands).astype(np.int32)
nnode = len(coords)
with open("gb_rt.bin", "wb") as f:
    np.array([nel, nfac, NSYS, len(idx), nnode], np.int32).tofile(f)
    np.array([LPHYS], np.float64).tofile(f)
    o_e.astype(np.float64).tofile(f)
    V0.astype(np.float64).tofile(f)
    coords.astype(np.float64).tofile(f)
    fnode.tofile(f)
    foff.astype(np.float64).tofile(f)
    ptr.tofile(f)
    (idx + 1).tofile(f)                          # 1-based
np.savetxt("gbnodes.txt", gbnodes, fmt="%d")
print(f"wrote gb_rt.bin: {nel} el, {nfac} facets, {nnode} nodes, "
      f"{len(idx)} candidates ({len(idx)/(nel*NSYS):.0f}/ray-pair); "
      f"gbnodes.txt {len(gbnodes)} ids")
