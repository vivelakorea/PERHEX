# Rank candidate tessellations by minimum edge length (bigger = cleaner
# geometry = better midpoint hexes).
import glob
import numpy as np
from build_cube_rve import tess_read_geometry

rows = []
for path in sorted(glob.glob("cand_*.tess")):
    verts, edges, faces, polys = tess_read_geometry(path)
    ln = np.array([np.linalg.norm(verts[a] - verts[b])
                   for a, b in edges.values()])
    rows.append((ln.min(), (ln < 0.02).sum(), (ln < 0.05).sum(), path))
rows.sort(reverse=True)
print(f"{'minEdge':>8} {'<0.02':>6} {'<0.05':>6}  file")
for mn, c2, c5, path in rows:
    print(f"{mn:8.4f} {c2:6d} {c5:6d}  {path}")
