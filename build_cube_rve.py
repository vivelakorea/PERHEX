# Build a CUBE-domain periodic polycrystal RVE with matching opposite-face
# node coordinates, smooth grain boundaries and C3D10 tets.
#
# Neper's own periodic meshes wrap around the box (image nodes outside the
# cube), which the downstream workflow does not accept. So instead:
#   1. read the periodic tessellation rve.tess, extract seeds (+weights)
#   2. add their periodic images (27 shifts), re-tessellate NON-periodically
#      on the unit cube -> cells cut at the box, boundary traces periodic
#   3. convert the cut .tess to gmsh .geo, add "Periodic Surface" constraints
#      pairing opposite boundary faces -> gmsh copies the surface mesh, so
#      opposite-face node coordinates match exactly
#   4. mesh with gmsh (C3D10), convert to an Abaqus deck in the
#      duplex_Ti_pipeline format: face/edge/corner UNSORTED nsets ordered for
#      set-based *Equation pairing + the original PBC.inp appended verbatim
#
# usage: python build_cube_rve.py [cl]      (default cl = 0.15)
import subprocess
import sys
import numpy as np

WSL = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--"]
CWD = r"O:\prj_rve\neper_rve"
TOL = 1e-6
MARGIN = 0.6          # keep image seeds within [-MARGIN, 1+MARGIN]^3


# ---------------------------------------------------------------- tess parsing
def tess_sections(path):
    lines = open(path).read().split("\n")
    idx = {l.strip(): k for k, l in enumerate(lines)
           if l.strip().startswith("*")}
    return lines, idx


def tess_read_seeds(path):
    lines, idx = tess_sections(path)
    # first *seed after **cell (the periodicity section has its own *seed)
    i = next(k for k, l in enumerate(lines)
             if l.strip() == "*seed" and k > idx["**cell"])
    ncell = int(lines[idx["**cell"] + 1])
    toks = " ".join(lines[i + 1:]).split()
    seeds = []
    for k in range(ncell):
        row = toks[k * 5:k * 5 + 5]
        seeds.append([float(row[1]), float(row[2]), float(row[3]),
                      float(row[4])])
    return np.array(seeds)


def tess_read_geometry(path):
    lines, idx = tess_sections(path)

    def block(name, stop):
        return " ".join(lines[idx[name] + 2:idx[stop]]).split()

    nver = int(lines[idx["**vertex"] + 1])
    toks = block("**vertex", "**edge")
    verts = {}
    for k in range(nver):
        r = toks[k * 5:k * 5 + 5]
        verts[int(r[0])] = np.array([float(r[1]), float(r[2]), float(r[3])])

    nedge = int(lines[idx["**edge"] + 1])
    toks = block("**edge", "**face")
    edges = {}
    for k in range(nedge):
        r = toks[k * 4:k * 4 + 4]
        edges[int(r[0])] = (int(r[1]), int(r[2]))

    nface = int(lines[idx["**face"] + 1])
    toks = block("**face", "**polyhedron")
    faces = {}
    p = 0
    for _ in range(nface):
        fid = int(toks[p]); p += 1
        nv = int(toks[p]); p += 1
        fverts = [int(x) for x in toks[p:p + nv]]; p += nv
        ne = int(toks[p]); p += 1
        fedges = [int(x) for x in toks[p:p + ne]]; p += ne
        p += 4 + 5     # plane eq + state/point
        faces[fid] = (fverts, fedges)
    assert p == len(toks), "face section parse error"

    npoly = int(lines[idx["**polyhedron"] + 1])
    toks = block("**polyhedron", "**domain")
    polys = {}
    p = 0
    for _ in range(npoly):
        pid = int(toks[p]); p += 1
        nf = int(toks[p]); p += 1
        polys[pid] = [int(x) for x in toks[p:p + nf]]; p += nf
    assert p == len(toks), "poly section parse error"
    return verts, edges, faces, polys


# ------------------------------------------------------------------- step 1+2
def best_cut_shift():
    """Place the cube's cut planes as far as possible from tessellation
    vertices (per axis: midpoint of the largest gap in vertex coords mod 1),
    so the cut does not slice off knife-thin pieces."""
    verts, _, _, _ = tess_read_geometry("rve.tess")
    xyz = np.array(list(verts.values())) % 1.0
    s = np.zeros(3)
    for ax in range(3):
        c = np.sort(xyz[:, ax])
        gaps = np.diff(np.concatenate([c, [c[0] + 1.0]]))
        k = int(np.argmax(gaps))
        s[ax] = (c[k] + gaps[k] / 2.0) % 1.0
        print(f"cut shift axis {'xyz'[ax]}: {s[ax]:.4f} "
              f"(clearance {gaps[k] / 2:.4f})")
    return s


def make_cut_tess(jitter=(0.0, 0.0, 0.0)):
    seeds = tess_read_seeds("rve.tess")
    print(f"rve.tess: {len(seeds)} primary seeds, "
          f"weights {seeds[:, 3].min():.3f}..{seeds[:, 3].max():.3f}")
    shift = (best_cut_shift() + np.array(jitter)) % 1.0
    seeds[:, :3] = (seeds[:, :3] - shift) % 1.0
    rows, mapping = [], []   # mapping[i] = primary grain of seed row i
    for g, (x, y, z, w) in enumerate(seeds, start=1):
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    c = np.array([x + sx, y + sy, z + sz])
                    if ((c > -MARGIN) & (c < 1 + MARGIN)).all():
                        rows.append(c)
                        mapping.append(g)
    print(f"{len(rows)} seeds after imaging/filtering")
    with open("seeds.coo", "w") as f:
        f.writelines(f"{c[0]:.15f} {c[1]:.15f} {c[2]:.15f}\n" for c in rows)
    with open("seeds.wt", "w") as f:
        f.writelines(f"{seeds[g - 1, 3]:.15f}\n" for g in mapping)
    np.savetxt("seeds.map", mapping, fmt="%d")
    # wsl.exe re-joins argv through the default shell, so parens must be
    # quoted inside one bash -c string (no double quotes: wsl mangles them)
    cmd = ("neper -T -n %d "
           "-domain 'cube(1,1,1)' -morpho voronoi "
           "-morphooptiini 'coo:file(seeds.coo),weight:file(seeds.wt)' "
           "-o rve_cut" % len(rows))
    r = subprocess.run(cmd, shell=True,
                       capture_output=True, text=True)
    if r.returncode != 0 or "rve_cut.tess" not in r.stdout:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("neper cut tessellation failed")
    print("rve_cut.tess written")
    return np.loadtxt("seeds.map", dtype=int)


# --------------------------------------------------------------------- step 3
def decimate(verts, faces, polys, tau):
    """Collapse tess edges shorter than tau (mini-regularization: Neper's
    -reg crashes with periodicity on this box). Cube-plane membership is
    preserved so boundary traces stay identical on opposite faces, keeping
    the mesh periodicity intact (write_geo's pairing check verifies)."""
    parent = {v: v for v in verts}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def planes_of(v):
        return {(ax, val) for ax in range(3) for val in (0.0, 1.0)
                if abs(verts[v][ax] - val) < TOL}

    # one collapse per round, candidates ordered by a periodic-invariant key,
    # merged position = plane-clamped midpoint -> both cube sides make
    # identical decisions and periodicity survives
    ncol = 0
    while True:
        cand = {}
        for fid, (vloop, _) in faces.items():
            loop = [find(v) for v in vloop]
            for a, b in zip(loop, loop[1:] + loop[:1]):
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in cand:
                    continue
                ln = np.linalg.norm(verts[a] - verts[b])
                if ln < tau:
                    mid = 0.5 * (verts[a] + verts[b])
                    skey = (round(ln * 1e9),) + tuple(
                        int(round((c % 1.0) * 1e9)) % 10 ** 9 for c in mid)
                    cand[key] = (skey, key)
                    break
        if not cand:
            break
        merged = False
        for _, (a, b) in sorted(cand.values()):
            pa, pb = planes_of(a), planes_of(b)
            if pa and pb and pa != pb:
                continue          # spans different cube planes: never collapse
            if pa and not pb:
                pos = verts[a].copy()      # boundary trace must not move
            elif pb and not pa:
                pos = verts[b].copy()
            else:                          # both interior, or same plane set
                pos = 0.5 * (verts[a] + verts[b])
                for ax, val in pa | pb:
                    pos[ax] = val
            parent[b] = a
            verts[a] = pos
            ncol += 1
            merged = True
            break
        if not merged:
            break
    # rebuild face loops through the collapse map
    newfaces = {}
    for fid, (vloop, _) in faces.items():
        loop = [find(v) for v in vloop]
        dedup = [v for m, v in enumerate(loop) if v != loop[m - 1]]
        if len(dedup) >= 3 and len(set(dedup)) == len(dedup):
            newfaces[fid] = (dedup, None)
    newpolys = {}
    for pid, fl in polys.items():
        kept = [f for f in fl if abs(f) in newfaces]
        if kept:
            newpolys[pid] = kept
    print(f"decimation: {ncol} short edges collapsed, "
          f"{len(faces) - len(newfaces)} faces removed")
    return newfaces, newpolys


def write_geo(cl, tau=0.0, gb_lc=None):
    """gb_lc = (lc_at_gb, dist_min, dist_max): refine near grain boundaries
    via a gmsh Distance+Threshold field (cl becomes the far-field size)."""
    verts, edges, faces, polys = tess_read_geometry("rve_cut.tess")
    # seeds whose Laguerre cell does not reach the cube leave empty polys
    polys = {k: v for k, v in polys.items() if v}
    print(f"{len(polys)} non-empty cells (cube pieces)")
    if tau > 0:
        # decimation invalidates the tess edge list: regenerate
        faces, polys = decimate(verts, faces, polys, tau)
        edges = {}
        edge_id = {}
        for fid, (vloop, _) in faces.items():
            for a, b in zip(vloop, vloop[1:] + vloop[:1]):
                key = (min(a, b), max(a, b))
                if key not in edge_id:
                    edge_id[key] = len(edge_id) + 1
                    edges[edge_id[key]] = key

        def signed_edges(vloop):
            out = []
            for a, b in zip(vloop, vloop[1:] + vloop[:1]):
                eid = edge_id[(min(a, b), max(a, b))]
                out.append(eid if a < b else -eid)
            return out
        for fid in faces:
            faces[fid] = (faces[fid][0], signed_edges(faces[fid][0]))
    else:
        # keep the tess-provided edges and signed loops untouched: gmsh's
        # periodic surface matching is sensitive to the curve orientation
        # pattern, and the tess-native one is the configuration that works
        edge_id = {(min(v1, v2), max(v1, v2)): eid
                   for eid, (v1, v2) in edges.items()}
    lo, hi = 0.0, 1.0
    xyz = np.array([verts[k] for k in sorted(verts)])
    assert xyz.min() > -1e-9 and xyz.max() < 1 + 1e-9, \
        "cut tess vertices outside unit cube"

    # boundary faces per axis/side: all vertices on that plane
    def face_on(fid, ax, val):
        return all(abs(verts[v][ax] - val) < TOL for v in faces[fid][0])

    pairs = []
    for ax in range(3):
        f0 = [f for f in faces if face_on(f, ax, lo)]
        fL = [f for f in faces if face_on(f, ax, hi)]
        oth = [k for k in range(3) if k != ax]

        def key(fid, ax=ax, oth=oth):
            vs = faces[fid][0]
            ks = sorted(tuple(int(round(verts[v][k] / TOL)) for k in oth)
                        for v in vs)
            return tuple(ks)
        d0 = {key(f): f for f in f0}
        if len(d0) != len(f0):
            sys.exit(f"duplicate face keys on axis {ax}")
        for f in fL:
            m = d0.pop(key(f), None)
            if m is None:
                sys.exit(f"axis {ax}: face {f} on max side has no translated "
                         f"match on min side -- traces not periodic")
            pairs.append((ax, m, f))    # master (min side), slave (max side)
        if d0:
            sys.exit(f"axis {ax}: unmatched faces on min side: {list(d0.values())}")
        print(f"axis {'xyz'[ax]}: {len(fL)} boundary face pairs matched")

    # uniform point sizes: varying per-point sizes desync gmsh's periodic
    # surface copy ("cannot find periodic counterpart"); sliver control is
    # done by decimation instead
    with open("cube.geo", "w") as g:
        g.write(f"Mesh.CharacteristicLengthMax = {cl};\n"
                "Mesh.Optimize = 1;\nMesh.OptimizeNetgen = 1;\n")
        for vid in sorted(verts):
            x, y, z = verts[vid]
            g.write(f"Point({vid}) = {{{x:.15f}, {y:.15f}, {z:.15f}, {cl}}};\n")
        for eid in sorted(edges):
            v1, v2 = edges[eid]
            g.write(f"Line({eid}) = {{{v1}, {v2}}};\n")
        # decimation can leave slightly non-planar interior loops; triangles
        # are always planar, so fan-triangulate those about their centroid.
        # (boundary faces keep their planar traces and stay whole, so the
        # periodic surface pairing below is unaffected.)
        emit = {}                       # fid -> list of emitted surface ids
        next_ln = max(edges) + 1
        next_sf = max(faces) + 1
        for fid in sorted(faces):
            vloop, fedges = faces[fid]
            vs = np.array([verts[v] for v in vloop])
            planar = np.linalg.svd(vs - vs.mean(axis=0))[1][2] < 1e-9
            if planar:
                g.write(f"Curve Loop({fid}) = "
                        f"{{{', '.join(map(str, fedges))}}};\n")
                g.write(f"Plane Surface({fid}) = {{{fid}}};\n")
                emit[fid] = [fid]
                continue
            # slightly non-planar after decimation: fan-triangulate from v0
            # (triangles are planar; convex loop -> no overlaps, no new points)
            v0 = vloop[0]
            diag = {}
            for v in vloop[2:-1]:
                diag[v] = next_ln
                g.write(f"Line({next_ln}) = {{{v0}, {v}}};\n")
                next_ln += 1
            emit[fid] = []
            for m in range(1, len(vloop) - 1):
                a, b = vloop[m], vloop[m + 1]
                eid = edge_id[(min(a, b), max(a, b))]
                seid = eid if a < b else -eid
                first = (edge_id[(min(v0, a), max(v0, a))]
                         * (1 if v0 < a else -1)) if m == 1 else diag[a]
                last = ((edge_id[(min(b, v0), max(b, v0))]
                         * (1 if b < v0 else -1))
                        if m == len(vloop) - 2 else -diag[b])
                g.write(f"Curve Loop({next_sf}) = {{{first}, {seid}, {last}}};\n")
                g.write(f"Plane Surface({next_sf}) = {{{next_sf}}};\n")
                emit[fid].append(next_sf)
                next_sf += 1
        # tess face signs follow the plane-equation normal, but gmsh surface
        # normals follow the curve-loop right-hand rule -- recompute signs
        # geometrically (cells are convex): outward if newell normal points
        # away from the cell centroid.
        def newell(fid):
            vs = [verts[v] for v in faces[fid][0]]
            n = np.zeros(3)
            for a, b in zip(vs, vs[1:] + vs[:1]):
                n += np.cross(a, b)
            return n, np.mean(vs, axis=0)

        for pid in sorted(polys):
            fl = [abs(f) for f in polys[pid]]
            cp = np.mean([newell(f)[1] for f in fl], axis=0)
            signed = []
            for f in fl:
                n, cf = newell(f)
                s = 1 if np.dot(n, cf - cp) > 0 else -1
                signed += [s * sub for sub in emit[f]]   # fan keeps orientation
            g.write(f"Surface Loop({pid}) = {{{', '.join(map(str, signed))}}};\n")
            g.write(f"Volume({pid}) = {{{pid}}};\n")
            g.write(f"Physical Volume({pid}) = {{{pid}}};\n")
        for ax, m, s in pairs:
            t = [0, 0, 0]
            t[ax] = 1
            g.write(f"Periodic Surface {{{s}}} = {{{m}}} "
                    f"Translate {{{t[0]}, {t[1]}, {t[2]}}};\n")
        if gb_lc:
            lcmin, dmin, dmax = gb_lc
            bset = set()
            for ax in range(3):
                for f in faces:
                    if face_on(f, ax, lo) or face_on(f, ax, hi):
                        bset.add(f)
            gb_surfs = [sid for fid in faces if fid not in bset
                        for sid in emit[fid]]
            g.write("Field[1] = Distance;\n")
            g.write(f"Field[1].FacesList = {{{', '.join(map(str, gb_surfs))}}};\n")
            g.write("Field[1].NNodesByEdge = 50;\n")
            g.write("Field[2] = Threshold;\nField[2].IField = 1;\n")
            g.write(f"Field[2].LcMin = {lcmin};\nField[2].LcMax = {cl};\n")
            g.write(f"Field[2].DistMin = {dmin};\nField[2].DistMax = {dmax};\n")
            g.write("Background Field = 2;\n"
                    "Mesh.CharacteristicLengthExtendFromBoundary = 0;\n")
    print(f"cube.geo written ({len(verts)} pts, {len(faces)} surfs, "
          f"{len(polys)} vols, {len(pairs)} periodic pairs)")


# --------------------------------------------------------------------- step 4
def run_gmsh():
    r = subprocess.run(WSL + ["gmsh", "-3", "-order", "2", "-format", "msh2",
                              "cube.geo", "-o", "cube.msh"],
                       capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-1500:]
    if r.returncode != 0 or "Error" in r.stdout or "Error" in r.stderr:
        print(tail)
        sys.exit("gmsh meshing failed")
    print("gmsh done")


# --------------------------------------------------------------------- step 5
def read_msh2(path):
    nodes, elems = {}, []
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("$Nodes"):
            n = int(lines[i + 1])
            for l in lines[i + 2:i + 2 + n]:
                p = l.split()
                nodes[int(p[0])] = np.array([float(x) for x in p[1:4]])
            i += n + 2
        elif lines[i].startswith("$Elements"):
            n = int(lines[i + 1])
            for l in lines[i + 2:i + 2 + n]:
                p = [int(x) for x in l.split()]
                etype, ntags = p[1], p[2]
                if etype in (4, 11, 5):   # tet4 / tet10 / hex8
                    elems.append((p[0], p[3], p[3 + ntags:]))  # physical tag
            i += n + 2
        else:
            i += 1
    return nodes, elems


def classify_and_order(nodes):
    """find_PBCset.py-style face/edge/corner sets, ordered by rounded free
    coords so parallel sets correspond entry by entry."""
    ids = np.array(sorted(nodes))
    xyz = np.array([nodes[k] for k in ids])
    on0 = np.abs(xyz - 0.0) < TOL
    onL = np.abs(xyz - 1.0) < TOL

    def pick(mask, order_axes):
        sel = np.where(mask)[0]
        keys = tuple(np.round(xyz[sel, ax] / TOL)
                     for ax in reversed(order_axes))
        if order_axes:
            sel = sel[np.lexsort(keys)]
        return [int(ids[k]) for k in sel]

    nb = {0: (on0[:, 0], onL[:, 0]), 1: (on0[:, 1], onL[:, 1]),
          2: (on0[:, 2], onL[:, 2])}
    onax = {ax: nb[ax][0] | nb[ax][1] for ax in range(3)}
    sets = {}
    for ax, name in enumerate("XYZ"):
        oth = [k for k in range(3) if k != ax]
        interior = ~(onax[oth[0]] | onax[oth[1]])
        sets[name + "0"] = pick(nb[ax][0] & interior, oth)
        sets[name + "L"] = pick(nb[ax][1] & interior, oth)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        free = [k for k in range(3) if k not in (a, b)]
        third = ~onax[free[0]]
        for sa, ma in (("0", nb[a][0]), ("L", nb[a][1])):
            for sb, mb in (("0", nb[b][0]), ("L", nb[b][1])):
                sets["XYZ"[a] + sa + "XYZ"[b] + sb] = pick(ma & mb & third, free)
    for sa, ma in (("0", nb[0][0]), ("L", nb[0][1])):
        for sb, mb in (("0", nb[1][0]), ("L", nb[1][1])):
            for sc, mc in (("0", nb[2][0]), ("L", nb[2][1])):
                sets[f"X{sa}Y{sb}Z{sc}"] = pick(ma & mb & mc, [])
    return sets


def verify_sets(sets, nodes):
    groups = [("X0", "XL", [1, 2]), ("Y0", "YL", [0, 2]), ("Z0", "ZL", [0, 1])]
    for a, b in (("X", "Y"), ("X", "Z"), ("Y", "Z")):
        free = [k for k in range(3) if k not in ("XYZ".index(a), "XYZ".index(b))]
        grp = [a + sa + b + sb for sa in "0L" for sb in "0L"]
        for o in grp[1:]:
            groups.append((grp[0], o, free))
    ok = True
    for a, b, free in groups:
        A, B = sets[a], sets[b]
        if len(A) != len(B):
            print(f"  FAIL {a}({len(A)}) vs {b}({len(B)}) count mismatch")
            ok = False
            continue
        d = 0.0 if not A else max(
            abs(nodes[i][k] - nodes[j][k])
            for i, j in zip(A, B) for k in free)
        print(f"  {a:6s} vs {b:6s}: {len(A):5d} nodes, "
              f"max in-plane offset {d:.2e}" + ("" if d < TOL else "  FAIL"))
        ok &= d < TOL
    if not ok:
        sys.exit("opposite-set coordinate verification FAILED")


def read_orientations(path="rve.tess"):
    """*ori section (rodrigues:passive) -> Bunge Euler angles in degrees."""
    lines, idx = tess_sections(path)
    i = idx["*ori"]
    assert "rodrigues" in lines[i + 1], "unexpected ori descriptor"
    ncell = int(lines[idx["**cell"] + 1])
    out = {}
    for g in range(1, ncell + 1):
        r = np.array([float(x) for x in lines[i + 1 + g].split()])
        th = 2.0 * np.arctan(np.linalg.norm(r))
        if th < 1e-12:
            out[g] = (0.0, 0.0, 0.0)
            continue
        n = r / np.linalg.norm(r)
        N = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        g_ = (np.cos(th) * np.eye(3) + (1 - np.cos(th)) * np.outer(n, n)
              - np.sin(th) * N)      # passive orientation matrix
        Phi = np.degrees(np.arccos(np.clip(g_[2, 2], -1, 1)))
        if abs(g_[2, 2]) > 1 - 1e-9:
            phi1 = np.degrees(np.arctan2(g_[0, 1], g_[0, 0]))
            phi2 = 0.0
        else:
            phi1 = np.degrees(np.arctan2(g_[2, 0], -g_[2, 1]))
            phi2 = np.degrees(np.arctan2(g_[0, 2], g_[1, 2]))
        out[g] = (phi1 % 360, Phi, phi2 % 360)
    return out


def write_inp(out, nodes, elems, sets, grain_of, cpfem=False):
    etype = "C3D4" if len(elems[0][2]) == 4 else "C3D10"
    grains = sorted({grain_of[pid] for _, pid, _ in elems})
    with open(out, "w") as f:
        f.write("** Cube-domain periodic RVE (build_cube_rve.py)\n")
        f.write("*Part, name=DREAM3D\n*NODE\n")
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write(f"{nid},\t{x:.6e},\t{y:.6e},\t{z:.6e}\n")
        f.write(f"*Element, type={etype}\n")
        for eid, _, conn in elems:
            if len(conn) == 10:
                conn = conn[:8] + [conn[9], conn[8]]  # gmsh -> abaqus tet10
            f.write(", ".join(str(v) for v in [eid] + conn) + "\n")
        for g in grains:
            f.write(f"*Elset, elset=GRAIN-{g}\n")
            eids = [str(eid) for eid, pid, _ in elems if grain_of[pid] == g]
            for k in range(0, len(eids), 9):
                f.write(", ".join(eids[k:k + 9]) + "\n")
        for g in grains:
            f.write(f"**Section: Section_Grain-{g}\n"
                    f"*Solid Section, elset=GRAIN-{g}, material=MATERIAL-GRAIN{g}\n,\n")
        f.write("*End Part\n")
        # dummy nodes just outside the cube so the odb view stays compact
        f.write("""*Part, name=DUMMY-1
*Node
100001, 1.2, 1.2, 0.
*End Part
*Part, name=DUMMY-2
*Node
100002, 1.2, 0., 1.2
*End Part
*Part, name=DUMMY-3
*Node
100003, 0., 1.2, 1.2
*End Part
**
*Assembly, name=Assembly
*Instance, name=DREAM3D-1, part=DREAM3D
*End Instance
*Instance, name=DUMMY-1-1, part=DUMMY-1
*End Instance
*Instance, name=DUMMY-2-2, part=DUMMY-2
*End Instance
*Instance, name=DUMMY-3-3, part=DUMMY-3
*End Instance
*Nset, nset=set-DUMMY-1, instance=DUMMY-1-1
100001,
*Nset, nset=set-DUMMY-2, instance=DUMMY-2-2
100002,
*Nset, nset=set-DUMMY-3, instance=DUMMY-3-3
100003,
""")
        f.write(f"*Nset, nset=FIX, instance=DREAM3D-1\n{sets['X0Y0Z0'][0]},\n")
        for name, nids in sets.items():
            if not nids:
                continue
            f.write(f"*Nset, nset={name}, instance=DREAM3D-1, unsorted\n")
            f.writelines(f"{n},\n" for n in nids)
        f.write(open("PBC.inp").read())
        f.write("*End Assembly\n")
        if cpfem:
            # OXFORD-UMAT props: euler1..3 (Bunge, deg), grain id, matID,
            # PROPS-file flag (0 = built-in materials of usermaterials.f)
            ori = read_orientations()
            for g in grains:
                e1, e2, e3 = ori[g]
                f.write(f"*Material, name=MATERIAL-GRAIN{g}\n"
                        f"*Depvar\n200,\n*User Material, constants=6\n"
                        f"{e1:.3f},{e2:.3f},{e3:.3f},{g},1,0\n")
            f.write("""**
** STEP: uniaxial x tension, 10% strain at 1e-3/s
*Step, name=Loading, nlgeom=YES, inc=1000000
*Static
0.05, 100., 1e-08, 1.
*Boundary
FIX, ENCASTRE
*Boundary
set-DUMMY-1, 1, 3, 0.
set-DUMMY-2, 1, 3, 0.
set-DUMMY-3, 1, 1, 1.e-01
*Restart, write, frequency=0
*Output, field, variable=PRESELECT
*Element Output, directions=YES
SDV,
*Output, history, variable=PRESELECT
*End Step
""")
        else:
            # elastic material for PBC validation
            for g in grains:
                f.write(f"*Material, name=MATERIAL-GRAIN{g}\n"
                        f"*Elastic\n110000., 0.34\n")
            f.write("""**
** STEP: uniaxial x tension, 0.1% strain
*Step, name=Loading, nlgeom=YES, inc=1000
*Static
0.1, 1., 1e-05, 0.1
*Boundary
FIX, ENCASTRE
*Boundary
set-DUMMY-1, 1, 3, 0.
set-DUMMY-2, 1, 3, 0.
set-DUMMY-3, 1, 1, 1.e-03
*Restart, write, frequency=0
*Output, field, variable=PRESELECT
*Output, history, variable=PRESELECT
*End Step
""")
    return etype, grains


if __name__ == "__main__":
    cpfem = "--cpfem" in sys.argv
    skip = "--skip-mesh" in sys.argv    # reuse existing cube.msh/seeds.map
    nums = [a for a in sys.argv[1:] if not a.startswith("--")]
    cl = float(nums[0]) if nums else 0.15
    if skip:
        mapping = np.loadtxt("seeds.map", dtype=int)
    else:
        mapping = make_cut_tess()
        write_geo(cl)
        run_gmsh()
    nodes, elems = read_msh2("cube.msh")
    print(f"cube.msh: {len(nodes)} nodes, {len(elems)} elements")
    grain_of = {i + 1: g for i, g in enumerate(mapping)}
    sets = classify_and_order(nodes)
    verify_sets(sets, nodes)
    out = "cube_cpfem.inp" if cpfem else "cube_final.inp"
    etype, grains = write_inp(out, nodes, elems, sets, grain_of, cpfem)
    print(f"wrote {out}: {etype}, {len(grains)} grains")

