# OXFORD-UMAT deck for the RAW VOXEL (staircase) mesh of the SAME id24
# microstructure — matrix cell M1 (comparison against the midpoint mesh).
# Same PBC scheme, same materials/orientations, same loading as
# gen_mp_deck_oxford.py.
import os
import numpy as np
from oxford_props import write_material
from build_cube_rve import classify_and_order, verify_sets, read_orientations
from voxel_smooth_rve import voxelize

OUT = "oxford_vox"
NV = 40
STRAIN = 0.10
RATE = 1e-3
MATID = 5
os.makedirs(OUT, exist_ok=True)

gfield, _, _, _ = voxelize(NV)
M = NV + 1
h = 1.0 / NV
nid = lambda i, j, k: i + M * (j + M * k)
coords = np.zeros((M ** 3, 3))
for k in range(M):
    for j in range(M):
        for i in range(M):
            coords[nid(i, j, k)] = (i * h, j * h, k * h)
conn = []
grain = []
for k in range(NV):
    for j in range(NV):
        for i in range(NV):
            conn.append([nid(i, j, k) + 1, nid(i + 1, j, k) + 1,
                         nid(i + 1, j + 1, k) + 1, nid(i, j + 1, k) + 1,
                         nid(i, j, k + 1) + 1, nid(i + 1, j, k + 1) + 1,
                         nid(i + 1, j + 1, k + 1) + 1,
                         nid(i, j + 1, k + 1) + 1])
            grain.append(gfield[i, j, k])
conn = np.array(conn)
grain = np.array(grain)
grains = np.unique(grain)
ori = read_orientations()
print(f"{len(coords)} nodes, {len(conn)} C3D8R, {len(grains)} grains")

nodes = {i + 1: coords[i] for i in range(len(coords))}
sets = classify_and_order(nodes)
verify_sets(sets, nodes)

EQS = []
def block(a, b, trip):
    for dof, dum, ddof in trip:
        EQS.append((a, b, dof, dum, ddof))
block("ZL", "Z0", [(2, 1, 2), (1, 2, 1), (3, 3, 3)])
block("XL", "X0", [(3, 1, 3), (2, 2, 2), (1, 3, 1)])
block("YL", "Y0", [(1, 1, 1), (3, 2, 3), (2, 3, 2)])
AX = {"X": [(3, 1, 3), (2, 2, 2), (1, 3, 1)],
      "Y": [(1, 1, 1), (3, 2, 3), (2, 3, 2)],
      "Z": [(2, 1, 2), (1, 2, 1), (3, 3, 3)]}
for grp, ref in [(["X0ZL", "XLZ0", "XLZL"], "X0Z0"),
                 (["X0YL", "XLY0", "XLYL"], "X0Y0"),
                 (["Y0ZL", "YLZ0", "YLZL"], "Y0Z0")]:
    for s in grp:
        jumps = [AX[a] for a in "XYZ" if a + "L" in s and a + "L" not in ref]
        for dof in (1, 2, 3):
            terms = [(s, dof, 1.0), (ref, dof, -1.0)]
            for axtrip in jumps:
                for d, dum, dd in axtrip:
                    if d == dof:
                        terms.append((f"set-DUMMY-{dum}", dd, -1.0))
            EQS.append(terms)
for s in ["XLY0Z0", "X0YLZ0", "X0Y0ZL", "XLYLZ0", "XLY0ZL", "X0YLZL",
          "XLYLZL"]:
    jumps = [AX[a] for a in "XYZ" if a + "L" in s]
    for dof in (1, 2, 3):
        terms = [(s, dof, 1.0), ("X0Y0Z0", dof, -1.0)]
        for axtrip in jumps:
            for d, dum, dd in axtrip:
                if d == dof:
                    terms.append((f"set-DUMMY-{dum}", dd, -1.0))
        EQS.append(terms)

with open(f"{OUT}/rve_oxford.inp", "w") as f:
    f.write("*Heading\n** Voxel (staircase) RVE + OXFORD-UMAT, matrix cell M1\n"
            "*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
    f.write("*Part, name=RVE\n*NODE\n")
    for n, (x, y, z) in enumerate(coords, start=1):
        f.write(f"{n},\t{x:.10f},\t{y:.10f},\t{z:.10f}\n")
    f.write("*Element, type=C3D8R\n")
    for e, c in enumerate(conn, start=1):
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
    for d, (x, y, z) in enumerate([(1.2, 1.2, 0.), (1.2, 0., 1.2),
                                   (0., 1.2, 1.2)], start=1):
        f.write(f"*Part, name=DUMMY-{d}\n*Node\n"
                f"10000{d}, {x}, {y}, {z}\n*End Part\n")
    f.write("*Assembly, name=Assembly\n"
            "*Instance, name=RVE-1, part=RVE\n*End Instance\n")
    for d in (1, 2, 3):
        f.write(f"*Instance, name=DUMMY-{d}-1, part=DUMMY-{d}\n"
                f"*End Instance\n"
                f"*Nset, nset=set-DUMMY-{d}, instance=DUMMY-{d}-1\n"
                f"10000{d},\n")
    f.write(f"*Nset, nset=FIX, instance=RVE-1\n{sets['X0Y0Z0'][0]},\n")
    for name, ids in sets.items():
        if not ids:
            continue
        f.write(f"*Nset, nset={name}, instance=RVE-1, unsorted\n")
        for k in range(0, len(ids), 8):
            f.write(", ".join(map(str, ids[k:k + 8])) + ",\n")
    for eq in EQS:
        if isinstance(eq, tuple):
            a, b, dof, dum, ddof = eq
            terms = [(a, dof, 1.0), (b, dof, -1.0),
                     (f"set-DUMMY-{dum}", ddof, -1.0)]
        else:
            terms = eq
        f.write(f"*Equation\n{len(terms)}\n")
        f.write("\n".join(f"{n}, {d}, {c:g}" for n, d, c in terms) + "\n")
    f.write("*End Assembly\n")
    f.write("*Section Controls, name=EC-1, hourglass=STIFFNESS\n1., 1., 1.\n")
    for g in grains:
        write_material(f, g, ori[g])
    T = STRAIN / RATE
    f.write(f"""**
** STEP: uniaxial x tension to {STRAIN * 100:.0f}% at {RATE:g}/s
*Step, name=Loading, nlgeom=YES, inc=1000000
*Static
0.25, {T:.4g}, {T * 1e-7:.4g}, 0.25
*Boundary
FIX, ENCASTRE
*Boundary
set-DUMMY-1, 1, 3, 0.
set-DUMMY-2, 1, 3, 0.
set-DUMMY-3, 1, 1, {STRAIN:.6g}
*Restart, write, frequency=0
*Output, field, number interval=50, time marks=NO
*Node Output
RF, U
*Element Output, directions=YES
LE, PEEQ, S, SDV
*Output, history, variable=PRESELECT
*End Step
""")
print(f"wrote {OUT}/rve_oxford.inp")

