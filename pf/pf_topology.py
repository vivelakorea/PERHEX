# Junction-graph extraction (INSTANCE-resolved, periodic) from a MOOSE
# phase-field snapshot, seeded from the gid voxel partition.
#
# Why gid-seeded and not OP-threshold-seeded: gid is an exact partition of
# the torus, so faces/edges/vertices of the grain complex are exact voxel
# sub-complexes with no threshold parameter at all; the OP top-k masks need
# an interface-width-dependent threshold that provably merges clusters along
# short triple lines or drops weak quad points (pf_junctions_op.py, 5/24
# Euler). The smooth OP fields are still used: as an independent consistency
# check here, and as sub-voxel geometry refinement later.
#
# Entities (all connectivity periodic on the N^3 corner/voxel torus):
#   face instance : connected component of interface voxel-faces of one
#                   grain pair, joined across 2-grain voxel edges only
#   tri  instance : connected component of 3-grain voxel edges of one
#                   triplet, joined across 3-grain corners only
#   quad cluster  : connected component (26-conn) of >=4-grain corners with
#                   IDENTICAL grain set (adjacent corners with different
#                   4-sets stay distinct -> resolves the merge/miss tradeoff)
# Per-grain Euler gate: V - E + F == 2 with INSTANCE counts (V includes one
# fictitious vertex per closed-loop tri instance, per CW-complex rules).
#
#   python pf_topology.py pf_snap_f3.npz
import sys
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from scipy import ndimage

f = sys.argv[1] if len(sys.argv) > 1 else "pf_snap_f3.npz"
d = np.load(f)
gid_raw = d["gid"]
N = gid_raw.shape[0]
N3 = N ** 3

# ---------------- clean voxel partition from the OPs ------------------------
# GrainTracker's unique_grains is speckled inside the diffuse interfaces
# (thousands of tiny same-id islands). The smooth OPs define the partition:
# voxel grain = argmax of cell-averaged OPs; the raw gid only disambiguates
# OP reuse (GrainTracker maps distant grains to one OP) by majority vote
# over each periodic dominant-OP component.
SS = int(sys.argv[2]) if len(sys.argv) > 2 else 2   # partition supersampling
ops = d["ops"].astype(np.float32)[:, :N, :N, :N]
nop = ops.shape[0]
if SS > 1:
    # periodic trilinear refinement of the nodal OPs to a (SS*N)^3 grid:
    # resolves sub-voxel features (thin faces, short edges) that the raw
    # voxel argmax collapses into degenerate contacts
    for ax in (1, 2, 3):
        nx = list(ops.shape)
        nx[ax] *= SS
        fine = np.zeros(nx, np.float32)
        nb = np.roll(ops, -1, axis=ax)
        idx = [slice(None)] * 4
        for s in range(SS):
            w = s / SS
            idx[ax] = slice(s, None, SS)
            fine[tuple(idx)] = (1 - w) * ops + w * nb
        ops = fine
    gid_raw = np.repeat(np.repeat(np.repeat(gid_raw, SS, 0), SS, 1), SS, 2)
    N *= SS
    N3 = N ** 3
cellop = 0.125 * sum(np.roll(np.roll(np.roll(ops, -dx, 1), -dy, 2), -dz, 3)
                     for dx in (0, 1) for dy in (0, 1) for dz in (0, 1))
dom = np.argmax(cellop, axis=0)


def periodic_label(mask):
    lab, nl = ndimage.label(mask)
    if nl == 0:
        return lab, 0
    parent = np.arange(nl + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ax in range(3):
        a = np.take(lab, 0, axis=ax).ravel()
        b = np.take(lab, -1, axis=ax).ravel()
        for la, lb in zip(a, b):
            if la and lb:
                ra, rb = find(la), find(lb)
                if ra != rb:
                    parent[rb] = ra
    root = np.array([find(x) for x in range(nl + 1)])
    u, newlab = np.unique(root[lab], return_inverse=True)
    return newlab.reshape(mask.shape), len(u) - 1


gid = np.full(gid_raw.shape, -1, dtype=np.int64)
for o in range(nop):
    m = dom == o
    if not m.any():
        continue
    lab, nl = periodic_label(m)
    for l in range(1, nl + 1):
        comp = lab == l
        vals, cts = np.unique(gid_raw[comp], return_counts=True)
        gid[comp] = vals[np.argmax(cts)]
assert (gid >= 0).all()
nchanged = int((gid != gid_raw).sum())
grains0 = sorted(int(g) for g in np.unique(gid))
# sanity: every grain must now be ONE periodic component
multi = []
for g in grains0:
    _, nl = periodic_label(gid == g)
    if nl != 1:
        multi.append((g, nl))
print(f"{f}: N={N}, grains={len(grains0)} "
      f"(cleaned {nchanged} speckled voxels; "
      f"multi-component grains: {multi if multi else 'none'})")
gid = gid.astype(np.uint64)
assert int(gid.max()) < 64, "bitmask assumes grain ids < 64"

AR = np.arange(N3).reshape(N, N, N)


def roll(a, s):
    return np.roll(a, s, axis=(0, 1, 2))


def sh(ax_vals):  # {axis: shift} -> 3-tuple
    s = [0, 0, 0]
    for ax, v in ax_vals.items():
        s[ax] = v
    return tuple(s)


def bits_to_list(b):
    return [g for g in range(64) if (int(b) >> g) & 1]


one = np.uint64(1)
bit = one << gid                                    # (N,N,N)

# ---------------- raw voxel sub-complexes -----------------------------------
# corner grain sets (8 voxels at i-1..i, j-1..j, k-1..k)
Bc = np.zeros_like(bit)
for dx in (0, 1):
    for dy in (0, 1):
        for dz in (0, 1):
            Bc |= roll(bit, (dx, dy, dz))
Uc = np.bitwise_count(Bc).astype(np.int8)

# edge grain sets (4 voxels around the edge from corner c along axis a)
BE, UE = [], []
for a in range(3):
    b, c = [x for x in range(3) if x != a]
    m = np.zeros_like(bit)
    for db in (0, 1):
        for dc in (0, 1):
            m |= roll(bit, sh({b: db, c: dc}))
    BE.append(m)
    UE.append(np.bitwise_count(m).astype(np.int8))

# interface voxel faces (face of voxel v toward +ax)
FI = [gid != roll(gid, sh({a: -1})) for a in range(3)]
BF = [bit | (one << roll(gid, sh({a: -1}))) for a in range(3)]
nfacevox = sum(int(x.sum()) for x in FI)
print(f"{nfacevox} interface voxel-faces, "
      f"{sum(int((u == 3).sum()) for u in UE)} triple voxel-edges, "
      f"{int((Uc >= 4).sum())} corners with >=4 grains "
      f"({int((Uc > 4).sum())} degenerate >4)")


def components(nnodes, joins, members):
    """joins: list of (u,v) arrays; members: node ids that exist.
    Returns per-member instance index (0..ni-1) and ni."""
    if joins:
        u = np.concatenate([j[0] for j in joins])
        v = np.concatenate([j[1] for j in joins])
    else:
        u = v = np.zeros(0, int)
    g = sparse.csr_matrix((np.ones(len(u), np.int8), (u, v)),
                          shape=(nnodes, nnodes))
    _, lab = connected_components(g, directed=False)
    _, inst = np.unique(lab[members], return_inverse=True)
    return inst, inst.max() + 1 if len(inst) else 0


# ---------------- face instances --------------------------------------------
# join two interface faces of the same pair across a 2-grain edge. Around an
# edge the 4 cyclically ordered voxels give 4 candidate faces.
# Edge taxonomy by (unique grains, transitions around the 4-cycle):
#   (2,2) interior pair edge      -> join the 2 interface faces
#   (2,4) checkerboard a,b,a,b    -> non-manifold pair pinch (flagged)
#   (3,3) TRUE triple edge x,x,y,z (each pair shares a face here)
#   (3,4) PINCH a,b,a,c: b,c touch only diagonally -> NOT a triple line;
#         resolve the non-manifold ridge: join the two faces of each wedge
#         (same-pair) so each pair's surface continues smoothly through it
#   (>=4,*) quad material (endpoint corners have >=4 grains)
fjoins, ncheck, npinch = [], 0, 0
TE = [None] * 3
PW = [None] * 3
for a in range(3):
    b, c = [x for x in range(3) if x != a]
    # cyclic voxels v00(-1b,-1c) v01(-1b,0) v11(0,0) v10(0,-1c)
    spec = [(c, {b: 1, c: 1}), (b, {b: 1}), (c, {c: 1}), (b, {b: 1, c: 1})]
    flags = [roll(FI[fx], sh(s)) for fx, s in spec]
    ids = [roll(AR, sh(s)) + fx * N3 for fx, s in spec]
    g00 = roll(gid, sh({b: 1, c: 1}))
    g01 = roll(gid, sh({b: 1}))
    g11 = gid
    g10 = roll(gid, sh({c: 1}))
    ntr = ((g00 != g01).astype(np.int8) + (g01 != g11) + (g11 != g10)
           + (g10 != g00))
    TE[a] = (UE[a] == 3) & (ntr == 3)
    CB = roll(AR, sh({a: -1}))          # other endpoint corner of the edge
    m2 = UE[a] == 2
    mj = m2 & (ntr == 2)
    for p in range(4):
        for q in range(p + 1, 4):
            m = mj & flags[p] & flags[q]
            fjoins.append((ids[p][m], ids[q][m], AR[m], CB[m]))
    mcb = m2 & (ntr == 4)
    ncheck += int(mcb.sum())
    for p, q in ((0, 1), (1, 2), (2, 3)):
        fjoins.append((ids[p][mcb], ids[q][mcb], AR[mcb], CB[mcb]))
    # 3-grain pinch: doubled grain on a diagonal; join per wedge
    mp = (UE[a] == 3) & (ntr == 4)
    PW[a] = mp
    npinch += int(mp.sum())
    m1 = mp & (g00 == g11)      # wedges at v01 (fa,fb) and v10 (fc,fd)
    for p, q in ((0, 1), (2, 3)):
        fjoins.append((ids[p][m1], ids[q][m1], AR[m1], CB[m1]))
    m2d = mp & (g01 == g10)     # wedges at v00 (fa,fd) and v11 (fb,fc)
    for p, q in ((0, 3), (1, 2)):
        fjoins.append((ids[p][m2d], ids[q][m2d], AR[m2d], CB[m2d]))
fids = np.concatenate([np.flatnonzero(FI[a]) + a * N3 for a in range(3)])
face_inst, nfi = components(3 * N3, fjoins, fids)
fju = np.concatenate([j[0] for j in fjoins])
fjv = np.concatenate([j[1] for j in fjoins])
fjc1 = np.concatenate([j[2] for j in fjoins])
fjc2 = np.concatenate([j[3] for j in fjoins])
BFall = np.concatenate([BF[a].ravel() for a in range(3)])
# pair per instance (must be uniform within an instance)
o = np.argsort(face_inst, kind="stable")
si, sb = face_inst[o], BFall[fids[o]]
first = np.r_[True, si[1:] != si[:-1]]
fidx = np.maximum.accumulate(np.where(first, np.arange(len(si)), 0))
assert (sb == sb[fidx]).all(), "face instance with mixed grain pair"
face_pairB = sb[first]
print(f"{nfi} face instances from {len(np.unique(face_pairB))} distinct pairs"
      f" ({ncheck} checkerboard, {npinch} 3-grain pinch edges resolved)")

# ---------------- triple-line instances -------------------------------------
# join true triple edges sharing a corner whose full 8-voxel set is exactly
# the triplet (>=4-grain corners cut the curve there -> quad endpoints).
tjoins = []
inc_c, inc_e = [], []          # (corner, tri-edge) incidence at 3-grain corners
qinc_c, qinc_e = [], []        # (corner, tri-edge) incidence at >=4 corners
for a in range(3):
    for shift, idr in (({}, AR + a * N3),
                       ({a: 1}, roll(AR, sh({a: 1})) + a * N3)):
        fl = roll(TE[a], sh(shift))
        m = fl & (Uc == 3)
        inc_c.append(AR[m]); inc_e.append(idr[m])
        m = fl & (Uc >= 4)
        qinc_c.append(AR[m]); qinc_e.append(idr[m])
inc_c = np.concatenate(inc_c); inc_e = np.concatenate(inc_e)
o = np.argsort(inc_c, kind="stable")
inc_c, inc_e = inc_c[o], inc_e[o]
same = inc_c[1:] == inc_c[:-1]
tjoins.append((inc_e[:-1][same], inc_e[1:][same]))
# manifold check: each 3-grain corner should touch exactly 2 triple edges
cnt = np.bincount(inc_c, minlength=N3)
nm3 = int((cnt[Uc.ravel() == 3] > 2).sum())
teids = np.concatenate([np.flatnonzero(TE[a]) + a * N3 for a in range(3)])
tri_inst, nti = components(3 * N3, tjoins, teids)
BEall = np.concatenate([BE[a].ravel() for a in range(3)])
o = np.argsort(tri_inst, kind="stable")
si, sb = tri_inst[o], BEall[teids[o]]
first = np.r_[True, si[1:] != si[:-1]]
fidx = np.maximum.accumulate(np.where(first, np.arange(len(si)), 0))
assert (sb == sb[fidx]).all(), "tri instance with mixed triplet"
tri_tripB = sb[first]
print(f"{nti} triple-line instances from "
      f"{len(np.unique(tri_tripB))} distinct triplets"
      f" ({nm3} non-manifold 3-grain corners)")

# ---------------- quad clusters (junction blobs) ----------------------------
# A vertex of the complex = connected junction blob of >=4-grain corners:
#  - lattice-adjacent corners connect through the intervening edge iff it is
#    junction material (U>=4 edge or 3-grain pinch edge), or a TRUE triple
#    edge whose endpoint corner sets are NESTED (equal/subset: staircase of
#    one junction, the edge is a chord). A true triple edge between corners
#    with different non-nested sets is a legitimate length-1 bridge ->
#    vertices stay split (generic resolution of a near-degenerate junction).
#  - diagonal (26-conn) staircase corners connect iff their sets are NESTED
#    (two distinct 4-sets sharing a triplet are two vertices joined by a
#    sub-voxel bridge -> NOT merged; the bridge shows up as junction-edge
#    material or a short arc instead).
# Blob grain set = union over member corners; >4 grains = degenerate vertex
# (flagged: a real >4 junction or a sub-voxel collapsed feature).
mask4 = Uc >= 4
qjoins = []
for a in range(3):
    B2 = roll(Bc, sh({a: -1}))
    nested = ((Bc | B2) == Bc) | ((Bc | B2) == B2)
    m = (UE[a] >= 4) | ((PW[a] | (TE[a] & nested))
                        & mask4 & roll(mask4, sh({a: -1})))
    qjoins.append((AR[m], roll(AR, sh({a: -1}))[m]))
offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)
        and sum(abs(v) for v in (dx, dy, dz)) > 1]
for off in offs:
    s = sh(dict(enumerate(off)))
    B2 = roll(Bc, s)
    nested = ((Bc | B2) == Bc) | ((Bc | B2) == B2)
    m = mask4 & roll(mask4, s) & nested
    qjoins.append((AR[m], roll(AR, s)[m]))
qcids = np.flatnonzero(mask4)
qu = np.concatenate([j[0] for j in qjoins])
qv = np.concatenate([j[1] for j in qjoins])
qinc_c = np.concatenate(qinc_c); qinc_e = np.concatenate(qinc_e)
teids0, tri_inst0, tri_tripB0, nti0 = teids, tri_inst, tri_tripB, nti

# Iterative vertex building with a split repair: if a kept arc (>2 edges)
# begins AND ends on one blob at two DIFFERENT corners, that blob wrongly
# merged two genuine vertices (e.g. two equal-set quad points joined by two
# short triple lines) -> forbid the direct joins between those corners and
# rebuild. Chords (<=2-edge arcs swallowed by one blob) are deleted instead.
forbidden, forced = set(), set()
for it in range(4):
    if forbidden:
        keepj = np.array([(min(a, b), max(a, b)) not in forbidden
                          for a, b in zip(qu, qv)])
        u2, v2 = qu[keepj], qv[keepj]
    else:
        u2, v2 = qu, qv
    if forced:
        u2 = np.concatenate([u2, [p[0] for p in forced]])
        v2 = np.concatenate([v2, [p[1] for p in forced]])
    qp_inst, nqp = components(N3, [(u2, v2)], qcids)
    corner2qp = np.full(N3, -1)
    corner2qp[qcids] = qp_inst
    qp_B = np.zeros(nqp, np.uint64)
    np.bitwise_or.at(qp_B, qp_inst, Bc.ravel()[qcids])
    qp_ng = np.bitwise_count(qp_B)

    teids, tri_inst, tri_tripB, nti = teids0, tri_inst0, tri_tripB0, nti0
    edge2tri = np.full(3 * N3, -1)
    edge2tri[teids] = tri_inst
    tri_qps = defaultdict(set)
    qp_tris = defaultdict(set)
    tri_qcorners = defaultdict(lambda: defaultdict(set))
    for cflat, eid in zip(qinc_c, qinc_e):
        t = edge2tri[eid]
        q = corner2qp[cflat]
        tri_qps[t].add(q)
        qp_tris[q].add(t)
        tri_qcorners[t][q].add(cflat)

    # chord removal
    nedge_inst = np.bincount(tri_inst, minlength=nti)
    dead = np.array([len(tri_qps[t]) == 1 and nedge_inst[t] <= 2
                     for t in range(nti)])
    nchord = int(dead.sum())
    if nchord:
        keep = ~dead
        newid = np.cumsum(keep) - 1
        em = keep[tri_inst]
        teids, tri_inst = teids[em], newid[tri_inst[em]]
        tri_tripB = tri_tripB[keep]
        nti = int(keep.sum())
        edge2tri = np.full(3 * N3, -1)
        edge2tri[teids] = tri_inst
        old_tq, old_tc = tri_qps, tri_qcorners
        tri_qps = defaultdict(set)
        qp_tris = defaultdict(set)
        tri_qcorners = defaultdict(lambda: defaultdict(set))
        for told, qs in old_tq.items():
            if keep[told]:
                tri_qps[newid[told]] = qs
                tri_qcorners[newid[told]] = old_tc[told]
                for q in qs:
                    qp_tris[q].add(newid[told])

    # split detection
    nedge_inst = np.bincount(tri_inst, minlength=nti)
    newforb = set()
    for t in range(nti):
        if len(tri_qps[t]) == 1 and nedge_inst[t] > 2:
            q = next(iter(tri_qps[t]))
            cs = sorted(tri_qcorners[t][q])
            for i_ in range(len(cs)):
                for j_ in range(i_ + 1, len(cs)):
                    newforb.add((min(cs[i_], cs[j_]), max(cs[i_], cs[j_])))
    newforb -= forbidden
    # parallel bundles: >=2 short (<=2 edge) arcs between the SAME two blobs
    # cannot be generic -> the two vertices are one (degenerate) junction
    bund = defaultdict(int)
    for t in range(nti):
        if len(tri_qps[t]) == 2 and nedge_inst[t] <= 2:
            bund[frozenset(tri_qps[t])] += 1
    newforce = set()
    for pr_, k_ in bund.items():
        if k_ >= 2:
            qa, qb = sorted(pr_)
            newforce.add((int(qcids[qp_inst == qa][0]),
                          int(qcids[qp_inst == qb][0])))
    newforce -= forced
    if not newforb and not newforce:
        break
    forbidden |= newforb
    forced |= newforce
    if newforb:
        print(f"  split pass {it}: forbidding {len(newforb)} "
              f"blob-internal joins")
    if newforce:
        print(f"  bundle pass {it}: merging {len(newforce)} vertex pairs "
              f"joined by parallel short arcs")
else:
    print("WARNING: vertex split repair did not converge")
ndeg = int((qp_ng > 4).sum())
print(f"{nqp} quad clusters ({ndeg} degenerate with >4 grains)")
if nchord:
    print(f"removed {nchord} swallowed chord instances")
tri_nend = np.array([len(tri_qps[t]) for t in range(nti)])
loops = tri_nend == 0
# a chain must terminate at a >=4 corner; verify no free ends:
# free end = corner with exactly 1 incident triple edge and Uc==3
free_ends = int((cnt[Uc.ravel() == 3] == 1).sum())
print(f"tri instances: {int((tri_nend == 2).sum())} arcs, "
      f"{int(loops.sum())} closed loops, "
      f"{int((tri_nend == 1).sum())} single-cluster, "
      f"{free_ends} FREE ENDS (must be 0)")

grains = sorted(int(g) for g in np.unique(gid))
tri_trips = [bits_to_list(b) for b in tri_tripB]
qp_gr = [bits_to_list(b) for b in qp_B]

face_lab = np.full(3 * N3, -1)
face_lab[fids] = face_inst


def compute_borders():
    """tri instance -> bordering face instances via its edges."""
    tf = defaultdict(set)
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        spec = [(c, {b: 1, c: 1}), (b, {b: 1}), (c, {c: 1}),
                (b, {b: 1, c: 1})]
        m3 = TE[a]
        e3 = AR[m3] + a * N3
        for fx, s in spec:
            fl = roll(FI[fx], sh(s))[m3]
            fid = (roll(AR, sh(s)) + fx * N3)[m3]
            for eid, ok, ff in zip(e3, fl, fid):
                if ok and edge2tri[eid] >= 0:
                    tf[edge2tri[eid]].add(int(face_lab[ff]))
    return tf


def rim_stats(tri_faces):
    """(face -> arcs, non-disk faces (i, ncomp, deg dict), rim loop counts,
    bad-degree flags)."""
    ft = defaultdict(list)
    for t in range(nti):
        for fi_ in tri_faces[t]:
            ft[fi_].append(t)
    bad = []
    ncomps = np.zeros(nfi, int)
    degbad = np.zeros(nfi, bool)
    for i in range(nfi):
        ts = ft[i]
        deg = defaultdict(int)
        par = {t: t for t in ts}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        byq = defaultdict(list)
        for t in ts:
            eps = sorted(tri_qps[t])
            for q in eps:
                deg[q] += 2 if len(eps) == 1 else 1
                byq[q].append(t)
        for q, l in byq.items():
            for t2 in l[1:]:
                par[find(t2)] = find(l[0])
        ncomp = len({find(t) for t in ts}) if ts else 0
        ncomps[i] = ncomp
        degbad[i] = any(v % 2 for v in deg.values())   # odd = broken rim
        if ncomp != 1 or any(v != 2 for v in deg.values()):
            bad.append((i, ncomp, dict(deg)))
    return ft, bad, ncomps, degbad


# ribbon repair: a face whose rim passes a vertex more than twice is two
# disk patches glued through junction material (sub-voxel contact). Rebuild
# such instances without joins whose intervening edge lies entirely inside
# junction blobs; if that splits them, relabel and recompute.
for pass_ in range(3):
    tri_faces = compute_borders()
    face_tris, nondisk, rim_ncomp, rim_degbad = rim_stats(tri_faces)
    ribbons = [i for i, ncomp, deg in nondisk
               if ncomp == 1 and any(v >= 4 for v in deg.values())]
    if not ribbons:
        break
    did = 0
    for i in ribbons:
        jm = (face_lab[fju] == i) & \
            ~((corner2qp[fjc1] >= 0) & (corner2qp[fjc2] >= 0))
        mem = fids[face_inst == i]
        sub, nsub = components(3 * N3, [(fju[jm], fjv[jm])], mem)
        if nsub > 1:
            for snew in range(1, nsub):
                newi = nfi + snew - 1
                face_inst[np.isin(fids, mem[sub == snew])] = newi
            face_pairB = np.concatenate(
                [face_pairB, [face_pairB[i]] * (nsub - 1)])
            nfi += nsub - 1
            face_lab[fids] = face_inst
            did += nsub - 1
    if did:
        print(f"  ribbon repair pass {pass_}: split off {did} face patches")
    else:
        break
tri_nf = np.array([len(tri_faces[t]) for t in range(nti)])
print(f"tri instances bordering exactly 3 face instances: "
      f"{int((tri_nf == 3).sum())}/{nti}")
face_pairs = [bits_to_list(b) for b in face_pairB]

# quad cluster -> face instances (union over incident tri instances)
qp_faces = {q: set().union(*(tri_faces[t] for t in ts)) if ts else set()
            for q, ts in qp_tris.items()}
gen4 = [q for q in range(nqp) if qp_ng[q] == 4]
ok46 = sum(1 for q in gen4
           if len(qp_tris[q]) == 4 and len(qp_faces.get(q, ())) == 6)
print(f"generic quad clusters with 4 tri instances + 6 face instances: "
      f"{ok46}/{len(gen4)}")
nondisk = [(i, face_pairs[i], ncomp, deg) for i, ncomp, deg in nondisk]
# absorb rim-less micro-faces: a face instance with NO bordering arcs whose
# pair is contained in the grain set of the single junction blob that
# surrounds it is sub-voxel junction detail, not a 2-cell of the complex
absorb = []
for i in range(nfi):
    if face_tris[i]:
        continue
    mem = fids[face_inst == i]
    blobs = set()
    for fid_ in mem:
        a_, r_ = divmod(int(fid_), N3)
        ci = np.array(np.unravel_index(r_, (N, N, N)))
        ci[a_] = (ci[a_] + 1) % N        # 4 corners of the voxel face
        oth = [x for x in range(3) if x != a_]
        for da in (0, 1):
            for db in (0, 1):
                cc = ci.copy()
                cc[oth[0]] = (cc[oth[0]] + da) % N
                cc[oth[1]] = (cc[oth[1]] + db) % N
                q = corner2qp[np.ravel_multi_index(cc, (N, N, N))]
                if q >= 0:
                    blobs.add(int(q))
    b = blobs.pop() if len(blobs) == 1 else -1
    if b >= 0 and (face_pairB[i] | qp_B[b]) == qp_B[b]:
        absorb.append(i)
    else:
        print(f"   UNABSORBABLE rim-less face {i} pair {face_pairs[i]} "
              f"({len(mem)} voxel faces, blobs {sorted(blobs) + ([b] if b >= 0 else [])})")
if absorb:
    fkeep = np.ones(nfi, bool)
    fkeep[absorb] = False
    fnew = np.cumsum(fkeep) - 1
    em = fkeep[face_inst]
    fids, face_inst = fids[em], fnew[face_inst[em]]
    face_pairB = face_pairB[fkeep]
    face_pairs = [p for i, p in enumerate(face_pairs) if fkeep[i]]
    nfi = int(fkeep.sum())
    face_lab = np.full(3 * N3, -1)
    face_lab[fids] = face_inst
    tri_faces = defaultdict(set, {t: {int(fnew[x]) for x in s}
                                  for t, s in tri_faces.items()})
    nondisk = [nd for nd in nondisk if fkeep[nd[0]]]
    print(f"absorbed {len(absorb)} rim-less micro-faces into junction blobs")
print(f"non-disk face instances: {len(nondisk)}")
for nd in nondisk[:10]:
    print("   face", nd[0], "rim components", nd[2],
          "bad vertex degrees", {q: v for q, v in nd[3].items() if v != 2})

# ---------------- per-grain Euler gate (general form) ------------------------
# chi(boundary of g) = V - E + sum(chi_face); an n-loop-rim face has
# chi = 2 - n (disk 1, annulus 0). The reference is the INDEPENDENTLY
# computed voxel-surface Euler characteristic chi_vox(dg) (resolved
# non-manifold: per-edge g-run ridges, per-corner wedge sheets), so the
# gate holds for balls (chi=2), percolating matrix grains (chi=2-2*genus),
# and torus-minus-ball grains (chi=2) alike:
#   V - E + F - n_annuli == chi_vox(dg), no ODD-degree face rims
# (even rim degrees > 2 are legal pinched rims - reported, not failed).
_, _, rim_ncomp, rim_degbad = rim_stats(tri_faces)

BLOCK = [(bx, by, bz) for bx in (0, 1) for by in (0, 1) for bz in (0, 1)]
BIDX = {b: i for i, b in enumerate(BLOCK)}
FACEP = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if sum(abs(np.array(BLOCK[i]) - np.array(BLOCK[j]))) == 1]
FKEY = {frozenset(p): k for k, p in enumerate(FACEP)}
CEDGES = []
for a_ in range(3):
    o1, o2 = [x for x in range(3) if x != a_]
    for s_ in (0, 1):
        cyc = []
        for d1, d2 in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            bb = [0, 0, 0]
            bb[a_], bb[o1], bb[o2] = s_, d1, d2
            cyc.append(BIDX[tuple(bb)])
        CEDGES.append(cyc)


def chi_vox(g):
    m = gid == g
    Fs = sum(int((m ^ roll(m, sh({a_: -1}))).sum()) for a_ in range(3))
    Eruns = 0
    for a_ in range(3):
        b_, c_ = [x for x in range(3) if x != a_]
        m00 = roll(m, sh({b_: 1, c_: 1}))
        m01 = roll(m, sh({b_: 1}))
        m11 = m
        m10 = roll(m, sh({c_: 1}))
        Eruns += int(((m00 & ~m01).astype(np.int8) + (m01 & ~m11)
                      + (m11 & ~m10) + (m10 & ~m00)).sum())
    MB = np.stack([roll(m, tuple(1 - np.array(b))) for b in BLOCK])
    mix = MB.any(0) & ~MB.all(0)
    sheets = 0
    for ci in np.argwhere(mix):
        mb = MB[:, ci[0], ci[1], ci[2]]
        bf = [k for k, (i, j) in enumerate(FACEP) if mb[i] != mb[j]]
        par = {k: k for k in bf}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for cyc in CEDGES:
            vals = [mb[x] for x in cyc]
            for pos in range(4):
                if vals[pos] and not vals[(pos + 1) % 4]:
                    q_ = pos
                    while vals[(q_ - 1) % 4]:
                        q_ = (q_ - 1) % 4
                    fa = FKEY[frozenset((cyc[q_], cyc[(q_ - 1) % 4]))]
                    fb = FKEY[frozenset((cyc[pos], cyc[(pos + 1) % 4]))]
                    par[find(fa)] = find(fb)
        sheets += len({find(k) for k in bf})
    return Fs - Eruns + sheets


def wrap_rank(mask):
    """rank of the sub-lattice of Z^3 along which the region wraps."""
    lab, nl = ndimage.label(mask)
    if nl == 0:
        return 0
    edges = defaultdict(list)
    for ax in range(3):
        a = np.take(lab, -1, axis=ax).ravel()
        b = np.take(lab, 0, axis=ax).ravel()
        sh = np.zeros(3)
        sh[ax] = 1
        for la, lb in set(zip(a.tolist(), b.tolist())):
            if la and lb:
                edges[la].append((lb, sh))
                edges[lb].append((la, -sh))
    offc, vecs = {}, []
    for start in range(1, nl + 1):
        if start in offc:
            continue
        offc[start] = np.zeros(3)
        stack = [start]
        while stack:
            c = stack.pop()
            for c2, sh in edges[c]:
                want = offc[c] + sh
                if c2 not in offc:
                    offc[c2] = want
                    stack.append(c2)
                elif not np.allclose(offc[c2], want):
                    vecs.append(want - offc[c2])
    return int(np.linalg.matrix_rank(np.array(vecs))) if vecs else 0


gwrap = {g: wrap_rank(gid == g) for g in grains}
gV = defaultdict(int); gE = defaultdict(int); gF = defaultdict(int)
gAnn = defaultdict(int); gBad = defaultdict(int)
for i, pr in enumerate(face_pairs):
    for g in pr:
        gF[g] += 1
        gAnn[g] += max(0, rim_ncomp[i] - 1)
        gBad[g] += int(rim_degbad[i])
for t, tr in enumerate(tri_trips):
    for g in tr:
        gE[g] += 1
        if loops[t]:
            gV[g] += 1               # fictitious vertex on each closed loop
for gs in qp_gr:
    for g in gs:
        gV[g] += 1
npass = 0
fails = []
for g in grains:
    e = gV[g] - gE[g] + gF[g] - gAnn[g]
    tgt = chi_vox(g)
    if e == tgt and gBad[g] == 0:
        npass += 1
    else:
        fails.append((g, gV[g], gE[g], gF[g], gAnn[g], gwrap[g], e, tgt,
                      gBad[g]))
nball = sum(1 for g in grains if gwrap[g] == 0)
print(f"EULER (V-E+F-annuli == chi_vox): {npass}/{len(grains)} grains pass"
      f"  [{nball} ball-like, {len(grains) - nball} percolating; "
      f"{sum(gAnn.values()) // 2} annular faces]")
for g, V, E, F, A, w, e, tgt, nb in fails:
    print(f"  grain {g}: V={V} E={E} F={F} annuli={A} wrap={w} "
          f"chi={e} chi_vox={tgt} oddrims={nb}")

# ---------------- geometry samples (lattice units, wrapped) -----------------
def unwrap_mean(pts):
    ref = pts[0]
    dp = (pts - ref + N // 2) % N - N // 2
    return (ref + dp.mean(0)) % N


# quad positions: mean of member corners
qp_pos = np.zeros((nqp, 3))
for q in range(nqp):
    pts = np.array(np.unravel_index(qcids[qp_inst == q], (N, N, N))).T
    qp_pos[q] = unwrap_mean(pts.astype(float))

# face centroids: mean of member voxel-face centers
fax = fids // N3
fpos = np.array(np.unravel_index(fids % N3, (N, N, N))).T + 0.5
fpos[np.arange(len(fids)), fax] += 0.5
face_cent = np.zeros((nfi, 3))
for i in range(nfi):
    face_cent[i] = unwrap_mean(fpos[face_inst == i])

# tri chains -> arc-length midpoint
tri_mid = np.zeros((nti, 3))
nbranch = 0
tri_edges_by_inst = defaultdict(list)
for eid, t in zip(teids, tri_inst):
    a, r = divmod(eid, N3)
    tri_edges_by_inst[t].append((a,) + tuple(np.unravel_index(r, (N, N, N))))
for t in range(nti):
    edges = tri_edges_by_inst[t]
    adj = defaultdict(list)
    for a, i, j, k in edges:
        c1 = (i, j, k)
        c2 = list(c1); c2[a] = (c2[a] + 1) % N
        adj[c1].append(tuple(c2)); adj[tuple(c2)].append(c1)
    if any(len(v) > 2 for v in adj.values()):
        nbranch += 1
        mids = np.array([[i + 0.5 * (a == 0), j + 0.5 * (a == 1),
                          k + 0.5 * (a == 2)] for a, i, j, k in edges])
        tri_mid[t] = unwrap_mean(mids)
        continue
    ends = [cc for cc, v in adj.items() if len(v) == 1]
    start = ends[0] if ends else edges[0][1:]
    # walk, unwrapped
    path = [np.array(start, float)]
    prev, cur = None, start
    for _ in range(len(edges)):
        nxt = [cc for cc in adj[cur] if cc != prev]
        if not nxt:
            break
        nc = nxt[0]
        dpv = (np.array(nc) - np.array(cur) + N // 2) % N - N // 2
        path.append(path[-1] + dpv)
        prev, cur = cur, nc
    path = np.array(path)
    L = len(path) - 1
    half = L / 2.0
    i0 = int(half)
    frac = half - i0
    p = path[i0] if i0 == L else path[i0] * (1 - frac) + path[i0 + 1] * frac
    tri_mid[t] = p % N
if nbranch:
    print(f"  {nbranch} branched tri instances (midpoint from centroid)")

# grain centers: circular mean of voxel centers
grain_cent = {}
for g in grains:
    pts = np.array(np.nonzero(gid == g)).T + 0.5
    ang = pts * (2 * np.pi / N)
    m = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
    grain_cent[g] = (m * N / (2 * np.pi)) % N

# ---------------- OP consistency cross-check --------------------------------
g2op = {g: int(np.argmax(cellop[:, gid == g].mean(1))) for g in grains}
# sample: one corner node per face instance, check its top-2 OPs
agree = 0
for i in range(nfi):
    k = np.flatnonzero(face_inst == i)[0]
    a, r = divmod(fids[k], N3)
    ci = np.array(np.unravel_index(r, (N, N, N)))
    ci[a] = (ci[a] + 1) % N          # a corner of that voxel face
    top2 = set(np.argsort(-ops[:, ci[0], ci[1], ci[2]])[:2])
    pr = face_pairs[i]
    if top2 == {g2op[pr[0]], g2op[pr[1]]}:
        agree += 1
print(f"OP cross-check: top-2 OPs at a face-instance node match the gid pair "
      f"for {agree}/{nfi} instances")

# ---------------- save ------------------------------------------------------
out = f.replace(".npz", "_complex.npz")
np.savez_compressed(
    out, N=N,
    qp_pos=qp_pos, qp_grains=np.array(qp_gr, dtype=object),
    qp_ng=qp_ng,
    qp_tris=np.array([sorted(qp_tris.get(q, ())) for q in range(nqp)],
                     dtype=object),
    tri_mid=tri_mid, tri_trips=np.array(tri_trips, dtype=object),
    tri_faces=np.array([sorted(tri_faces[t]) for t in range(nti)],
                       dtype=object),
    tri_loop=loops, tri_nend=tri_nend,
    face_cent=face_cent, face_pairs=np.array(face_pairs, dtype=object),
    grain_ids=np.array(grains),
    grain_cent=np.array([grain_cent[g] for g in grains]),
    grain_nvox=np.array([int((gid == g).sum()) for g in grains]),
    grain_op=np.array([g2op[g] for g in grains]),
    grain_wrap=np.array([gwrap[g] for g in grains]),
    face_nloops=rim_ncomp,
    euler_pass=np.array([npass, len(grains)]),
    allow_pickle=True)
print("saved", out)
