# All-hex RVE with EXACTLY PLANAR grain boundaries via polyhedral midpoint
# subdivision:
#   - every vertex of a (generic) Laguerre cell touches exactly 3 cell edges
#     and 3 cell faces, so the cell splits into one hex per vertex:
#       (vertex, 3 edge midpoints, 3 face centroids, cell centroid)
#   - each GB face is covered by quads (vertex, edge mid, face centroid,
#     edge mid) whose 4 corners all lie IN the face plane -> GB exactly planar
#   - midpoints/centroids are shared between neighbouring cells -> conforming
#   - built on the trace-matched cut tessellation -> periodic + flat cube
#   - each coarse hex is refined n^3 by trilinear subdivision (planarity of
#     the GB quads is preserved)
#
# usage: python gen_midpoint_hex.py [n_refine]      (default 3)
import sys
import numpy as np
from build_cube_rve import tess_read_geometry, decimate

TOL = 1e-6


def cell_hexes(verts, faces, polys):
    """Yield coarse hexes (8 corner point keys) + owning poly."""
    # point keys: ('v', id) vertex, ('e', a, b) edge midpoint,
    #             ('f', fid) face centroid, ('c', pid) cell centroid
    hexes = []
    for pid, fl in polys.items():
        fids = [abs(f) for f in fl]
        # vertex -> incident faces (of this cell) and incident edges
        vfaces = {}
        vedges = {}
        for fid in fids:
            loop = faces[fid][0]
            nl = len(loop)
            for m, v in enumerate(loop):
                vfaces.setdefault(v, []).append(fid)
                a, b = loop[m - 1], loop[(m + 1) % nl]
                vedges.setdefault(v, set()).update(
                    [(min(v, a), max(v, a)), (min(v, b), max(v, b))])
        for v, fset in vfaces.items():
            eset = sorted(vedges[v])
            if len(fset) != 3 or len(eset) != 3:
                raise RuntimeError(
                    f"poly {pid} vertex {v}: {len(fset)} faces/"
                    f"{len(eset)} edges (non-trivalent; decimate more?)")
            # match edge i with the face NOT containing it (opposite), and
            # order (e1,e2,f12): face f12 contains edges e1 and e2
            fedges = {}
            for fid in fset:
                loop = faces[fid][0]
                es = set()
                nl = len(loop)
                for m, w in enumerate(loop):
                    es.add((min(w, loop[(m + 1) % nl]),
                            max(w, loop[(m + 1) % nl])))
                fedges[fid] = es
            e1, e2, e3 = eset
            # find face containing e1&e2, e1&e3, e2&e3
            def face_with(ea, eb):
                for fid in fset:
                    if ea in fedges[fid] and eb in fedges[fid]:
                        return fid
                raise RuntimeError("no face contains both edges")
            f12 = face_with(e1, e2)
            f13 = face_with(e1, e3)
            f23 = face_with(e2, e3)
            # hex: bottom quad in face f12: v, m(e1), c(f12), m(e2)
            #      top quad:               m(e3), c(f13), c(pid), c(f23)
            hexes.append((pid,
                          (('v', v), ('e',) + e1, ('f', f12), ('e',) + e2,
                           ('e',) + e3, ('f', f13), ('c', pid), ('f', f23))))
    return hexes


def build(nref):
    # no decimation: edge collapses merge triple points into non-trivalent
    # vertices, which break the one-hex-per-corner construction. Tiny tess
    # edges just become small (well-shaped) hexes here -- harmless.
    verts, edges, faces, polys = tess_read_geometry("rve_cut.tess")
    polys = {k: v for k, v in polys.items() if v}
    print(f"{len(polys)} cells")

    coarse = cell_hexes(verts, faces, polys)
    print(f"{len(coarse)} coarse hexes (one per cell corner)")

    # ---- constrained smoothing of the construction points ------------------
    # edge midpoints slide along their edge, face points move within their
    # face plane; this unfolds "dart" quads on skinny faces (which otherwise
    # make the bilinear cells overlap) and improves corner-kite angles.
    epos = {}
    for fid, (vloop, _) in faces.items():
        nl = len(vloop)
        for m, v in enumerate(vloop):
            a, b = v, vloop[(m + 1) % nl]
            epos[(min(a, b), max(a, b))] = 0.5  # parameter on the edge

    fpos = {fid: np.mean([verts[v] for v in faces[fid][0]], axis=0)
            for fid in faces}

    def edge_xyz(e):
        a, b = e
        return verts[a] + epos[e] * (verts[b] - verts[a])

    for _ in range(0):   # smoothing disabled: it created folds (negative SJ)
        # face points -> average of their fan ring (stays in plane: ring
        # points v and m(e) all lie in the face plane)
        for fid, (vloop, _) in faces.items():
            ring = [verts[v] for v in vloop]
            nl = len(vloop)
            ring += [edge_xyz((min(vloop[m], vloop[(m + 1) % nl]),
                               max(vloop[m], vloop[(m + 1) % nl])))
                     for m in range(nl)]
            fpos[fid] = np.mean(ring, axis=0)
        # edge midpoints -> projection of adjacent face points' midpoint
        # onto the edge, clamped away from the ends
        fadj = {}
        for fid, (vloop, _) in faces.items():
            nl = len(vloop)
            for m in range(nl):
                e = (min(vloop[m], vloop[(m + 1) % nl]),
                     max(vloop[m], vloop[(m + 1) % nl]))
                fadj.setdefault(e, []).append(fid)
        for e, fl in fadj.items():
            a, b = e
            target = np.mean([fpos[f] for f in fl], axis=0)
            d = verts[b] - verts[a]
            t = np.dot(target - verts[a], d) / np.dot(d, d)
            epos[e] = min(max(t, 0.25), 0.75)

    # point coordinates per key
    def pos(key):
        if key[0] == 'v':
            return verts[key[1]]
        if key[0] == 'e':
            return edge_xyz((key[1], key[2]))
        if key[0] == 'f':
            return fpos[key[1]]
        return cellcent[key[1]]

    cellcent = {}
    for pid, fl in polys.items():
        cellcent[pid] = np.mean([fpos[abs(f)] for f in fl], axis=0)

    # global refined nodes: key -> id, refined via trilinear on each coarse hex
    # subdivision nodes shared across hexes via rounded-coordinate keys
    nodes = {}
    coords = []

    def nid_of(p):
        key = tuple(np.round(p / (TOL * 0.01)).astype(np.int64))
        i = nodes.get(key)
        if i is None:
            i = len(coords) + 1
            nodes[key] = i
            coords.append(p)
        return i

    conn = []
    egrain_pid = []
    n = nref
    # graded spacing: thin layers near t=0 (the GB corner side: local planes
    # xi=0, eta=0, zeta=0 of every corner hex all lie ON cell faces), growing
    # geometrically by RATIO toward the cell center. One global schedule ->
    # shared faces get identical grids -> conforming + periodic preserved.
    if RATIO > 1.0:
        t = (RATIO ** np.arange(n + 1) - 1.0) / (RATIO ** n - 1.0)
    else:
        t = np.arange(n + 1) / n
    print(f"grading: n={n}, ratio={RATIO}, layer fractions "
          f"{np.round(np.diff(t), 3)}")
    for pid, corner_keys in coarse:
        P = np.array([pos(k) for k in corner_keys])
        # trilinear shape: order (bottom v,m1,c12,m2 / top m3,c13,C,c23)
        # standard hex node local coords:
        loc = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
        def tri(x, y, z):
            s = np.array([(1 - x) * (1 - y) * (1 - z), x * (1 - y) * (1 - z),
                          x * y * (1 - z), (1 - x) * y * (1 - z),
                          (1 - x) * (1 - y) * z, x * (1 - y) * z,
                          x * y * z, (1 - x) * y * z])
            return s @ P
        grid = {}
        for k in range(n + 1):
            for j in range(n + 1):
                for i in range(n + 1):
                    grid[(i, j, k)] = nid_of(tri(t[i], t[j], t[k]))
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    conn.append([grid[(i, j, k)], grid[(i + 1, j, k)],
                                 grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                                 grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)],
                                 grid[(i + 1, j + 1, k + 1)],
                                 grid[(i, j + 1, k + 1)]])
                    egrain_pid.append(pid)
    coords = np.array(coords)
    conn = np.array(conn)
    print(f"refined n={n}: {len(coords)} nodes, {len(conn)} hexes")

    # orientation fix: flip hexes with negative volume
    from voxel_smooth_rve import hex_quality
    X = coords[conn - 1]
    tr = [(0, 1, 3, 4)]
    e1 = X[:, 1] - X[:, 0]
    e2 = X[:, 3] - X[:, 0]
    e3 = X[:, 4] - X[:, 0]
    det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    flip = det < 0
    # true role reordering (reverse both quad loops), NOT a mirror: the
    # trilinear cell then parameterizes the same region right-handedly
    conn[flip] = conn[flip][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    print(f"reordered {flip.sum()} left-handed hexes")
    mj, bad = hex_quality(coords, conn - 1)
    print(f"min scaled jacobian {mj:.4f}, below 0.05: {bad.sum()}")

    np.save("mp_coords.npy", coords)
    np.save("mp_conn.npy", conn)
    np.save("mp_pid.npy", np.array(egrain_pid))
    print("saved mp_*.npy")
    return coords, conn


RATIO = 1.0

if __name__ == "__main__":
    nref = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    build(nref)
