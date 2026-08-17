# Extract a MOOSE grain-growth snapshot from the Exodus output onto the
# structured periodic grid: per-element grain ids (unique_grains) and
# per-node order parameters (gr0..grN, for sub-voxel interface geometry).
#   python extract_pf.py [frame_index] [exodus_file]   (default: last frame,
#   grain_growth_3d_out.e -> pf_snap_f{i}.npz; other files -> {stem}_f{i}.npz)
import sys
import os
import numpy as np
from netCDF4 import Dataset

FILE = sys.argv[2] if len(sys.argv) > 2 else "grain_growth_3d_out.e"
STEM = ("pf_snap" if FILE == "grain_growth_3d_out.e"
        else os.path.basename(FILE).replace("_out.e", "").replace(".e", ""))

ds = Dataset(FILE)
times = ds.variables["time_whole"][:]
print(f"{len(times)} frames, t = {times[0]:.0f} .. {times[-1]:.0f}")
frame = int(sys.argv[1]) if len(sys.argv) > 1 else len(times) - 1

# variable name tables
def names(var):
    raw = ds.variables[var][:]
    return ["".join(c.decode() for c in row if c).strip() for row in raw.data]

nodal_names = names("name_nod_var")
elem_names = names("name_elem_var") if "name_elem_var" in ds.variables else []
print("nodal:", nodal_names[:5], "... total", len(nodal_names))
print("elemental:", elem_names)

x = ds.variables["coordx"][:]
y = ds.variables["coordy"][:]
z = ds.variables["coordz"][:]
n = len(x)
# structured mapping: GeneratedMesh nodes on a regular lattice
xs, ys, zs = (np.unique(np.round(c, 6)) for c in (x, y, z))
M = (len(xs), len(ys), len(zs))
print(f"node grid {M}")
ix = np.searchsorted(xs, np.round(x, 6))
iy = np.searchsorted(ys, np.round(y, 6))
iz = np.searchsorted(zs, np.round(z, 6))

ops = []
for k, nm in enumerate(nodal_names):
    if not nm.startswith("gr"):
        continue
    v = ds.variables[f"vals_nod_var{k+1}"][frame, :]
    a = np.zeros(M)
    a[ix, iy, iz] = v
    ops.append(a)
ops = np.array(ops)                      # (nop, nx+1, ny+1, nz+1)

# elemental unique_grains: connectivity -> element centroid grid index
gid = None
if "unique_grains" in elem_names:
    k = elem_names.index("unique_grains") + 1
    # single block assumed
    conn = ds.variables["connect1"][:] - 1          # (nel, 8)
    ex = x[conn].mean(1); ey = y[conn].mean(1); ez = z[conn].mean(1)
    hx = xs[1] - xs[0]
    jx = np.floor(ex / hx).astype(int)
    jy = np.floor(ey / hx).astype(int)
    jz = np.floor(ez / hx).astype(int)
    v = ds.variables[f"vals_elem_var{k}eb1"][frame, :]
    gid = np.full((M[0] - 1, M[1] - 1, M[2] - 1), -1.0)
    gid[jx, jy, jz] = v
    gid = gid.astype(int)
    assert (gid >= 0).all()
    print(f"grains present: {len(np.unique(gid))}")

out = f"{STEM}_f{frame}.npz"
np.savez_compressed(out, ops=ops.astype(np.float32),
                    gid=gid if gid is not None else np.zeros(0),
                    time=times[frame], xs=xs, ys=ys, zs=zs)
print(f"wrote {out}")
