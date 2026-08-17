# pf/ — curved grain boundaries from phase-field microstructures

MOOSE grain-growth (periodic Fan-Chen) -> conforming all-hex periodic
mesh with curved GBs.

1. `mpiexec -n 8 moose-opt -i grain_growth_3d.i` (MOOSE conda package;
   `gg_fewgrains.i` / `gg_bimodal.i` are the stress-test variants)
2. `python extract_pf.py <frame> [exodus]` -> npz snapshot
3. `python pf_topology.py <snap.npz>` -> junction complex + Euler gate
4. `python pf_hexmesh.py <snap.npz>` -> v1 mesh, n^3 refinement,
   OP projection, quality report
5. `python pf_deck.py` -> wrap-style PBC Abaqus deck (OXFORD-UMAT)

`pf_results.txt` holds the full experiment log (T1 baseline, bimodal,
domain-scale-curvature; quality tables and CPFEM outcomes).
