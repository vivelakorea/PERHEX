# Self-contained wrap-style Abaqus CPFEM deck for a pf hex mesh
# (pf_snap_*_mesh*.npz). The mesh lives on the torus: elements crossing the
# cube get IMAGE nodes outside [0,1] and one *Equation per image dof
#   u_img - u_primary = shift . face jumps        (neper2abaqus.py scheme:
# dummy-3 = normal jumps, dummy-1/2 = shears, so the parent loading block
# works unchanged). Materials: per-grain OXFORD-UMAT 300-constant props
# (oxford_props.write_material) with fixed-seed random Bunge angles.
#   python pf_deck.py pf_snap_f3_mesh_n2p.npz pf_cpfem
import os
import sys
import numpy as np

sys.path.insert(0, "..")
from oxford_props import write_material                       # noqa: E402

mesh = sys.argv[1] if len(sys.argv) > 1 else "pf_snap_f3_mesh_n2p.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pf_cpfem"
STRAIN, RATE = 0.02, 1e-3
os.makedirs(OUT, exist_ok=True)

d = np.load(mesh)
coords, conn, grain = d["coords"], d["conn"], d["grain"]
nn = len(coords)
grains = np.unique(grain)
print(f"{mesh}: {nn} nodes, {len(conn)} hexes, {len(grains)} grains")

# per-element unwrap (elements are far smaller than the half box):
# offset of node a in element e = -round(x_a - x_a0)
rel = coords[conn] - coords[conn][:, :1, :]
off = -np.round(rel)                                   # (ne,8,3) in {-1,0,1}
assert np.abs(rel + off).max() < 0.5
# image nodes for (node, offset) pairs with offset != 0
img = {}
per = []                                               # (img_id, primary+1, shift)
econn = np.array(conn, dtype=int) + 1                  # 1-based, overwritten
for e in range(len(conn)):
    for a in range(8):
        o = tuple(int(x) for x in off[e, a])
        if o == (0, 0, 0):
            continue
        key = (int(conn[e, a]), o)
        if key not in img:
            iid = nn + len(img) + 1
            img[key] = iid
            per.append((iid, int(conn[e, a]) + 1, o))
        econn[e, a] = img[key]
print(f"{len(img)} image nodes, {3 * len(per)} PBC equations")

JUMP = {0: {1: ("set-DUMMY-3", 1), 2: ("set-DUMMY-2", 2), 3: ("set-DUMMY-1", 3)},
        1: {1: ("set-DUMMY-1", 1), 2: ("set-DUMMY-3", 2), 3: ("set-DUMMY-2", 3)},
        2: {1: ("set-DUMMY-2", 1), 2: ("set-DUMMY-1", 2), 3: ("set-DUMMY-3", 3)}}

rng = np.random.default_rng(42)
ori = {int(g): (rng.uniform(0, 360),
                np.degrees(np.arccos(rng.uniform(-1, 1))),
                rng.uniform(0, 360)) for g in grains}

T = STRAIN / RATE
with open(f"{OUT}/pf_cpfem.inp", "w") as f:
    f.write("*Heading\n** pf phase-field midpoint hex RVE, wrap-style PBC, "
            "OXFORD-UMAT\n*Preprint, echo=NO, model=NO, history=NO\n")
    f.write("*Part, name=RVE\n*NODE\n")
    for i, (x, y, z) in enumerate(coords, start=1):
        f.write(f"{i},\t{x:.10f},\t{y:.10f},\t{z:.10f}\n")
    for (nd, o), iid in img.items():
        x, y, z = coords[nd] + np.array(o)
        f.write(f"{iid},\t{x:.10f},\t{y:.10f},\t{z:.10f}\n")
    f.write("*Element, type=C3D8R\n")
    for e, c in enumerate(econn, start=1):
        f.write(", ".join(map(str, [e] + list(c))) + "\n")
    for g in grains:
        els = np.where(grain == g)[0] + 1
        f.write(f"*Elset, elset=GRAIN-{g}\n")
        for k in range(0, len(els), 16):
            f.write(", ".join(map(str, els[k:k + 16])) + ",\n")
    for g in grains:
        f.write(f"*Solid Section, elset=GRAIN-{g}, controls=EC-1, "
                f"material=MAT-GRAIN{g}\n,\n"
                "*Hourglass Stiffness\n250., , 0., 0.\n")
    f.write("*End Part\n")
    for dm, (x, y, z) in enumerate([(1.2, 1.2, 0.), (1.2, 0., 1.2),
                                    (0., 1.2, 1.2)], start=1):
        f.write(f"*Part, name=DUMMY-{dm}\n*Node\n"
                f"10000{dm}, {x}, {y}, {z}\n*End Part\n")
    f.write("*Assembly, name=Assembly\n"
            "*Instance, name=RVE-1, part=RVE\n*End Instance\n")
    for dm in (1, 2, 3):
        f.write(f"*Instance, name=DUMMY-{dm}-1, part=DUMMY-{dm}\n"
                f"*End Instance\n"
                f"*Nset, nset=set-DUMMY-{dm}, instance=DUMMY-{dm}-1\n"
                f"10000{dm},\n")
    f.write("*Nset, nset=FIX, instance=RVE-1\n1,\n")
    f.write("** wrap-style PBC: u_img - u_primary = shift . face jumps\n")
    for iid, pid, shift in per:
        for dof in (1, 2, 3):
            terms = [(f"RVE-1.{iid}", dof, 1.0), (f"RVE-1.{pid}", dof, -1.0)]
            for ax in range(3):
                if shift[ax]:
                    dset, ddof = JUMP[ax][dof]
                    terms.append((dset, ddof, -float(shift[ax])))
            f.write(f"*Equation\n{len(terms)}\n")
            f.write("\n".join(f"{n}, {dd}, {c:g}" for n, dd, c in terms)
                    + "\n")
    f.write("*End Assembly\n")
    f.write("*Section Controls, name=EC-1, hourglass=STIFFNESS\n1., 1., 1.\n")
    for g in grains:
        write_material(f, int(g), ori[int(g)])
    f.write(f"""**
** STEP: uniaxial x tension to {STRAIN * 100:.0f}% at {RATE:g}/s
*Step, name=Loading, nlgeom=YES, inc=1000000
*Static
{T / 200:.4g}, {T:.4g}, {T * 1e-7:.4g}, {T / 100:.4g}
*Boundary
FIX, ENCASTRE
*Boundary
set-DUMMY-1, 1, 3, 0.
set-DUMMY-2, 1, 3, 0.
set-DUMMY-3, 1, 1, {STRAIN:.6g}
*Restart, write, frequency=0
*Output, field, number interval=20, time marks=NO
*Node Output
RF, U
*Element Output, directions=YES
LE, S
*Output, history, variable=PRESELECT
*End Step
""")
np.savez(f"{OUT}/pbc_pairs.npz",
         img=np.array([p[0] for p in per]),
         prim=np.array([p[1] for p in per]),
         shift=np.array([p[2] for p in per]))
print(f"wrote {OUT}/pf_cpfem.inp + pbc_pairs.npz")
