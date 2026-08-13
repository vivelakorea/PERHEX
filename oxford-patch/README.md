# OXFORD-UMAT patch: grain-boundary dislocation storage with live d_b

Adds the Haouala-type GB storage term to the Kocks–Mecking hardening
model (model 3) of [OXFORD-UMAT](https://github.com/TarletonGroup/CrystalPlasticity):

    rho_ssd_dot = ( max(k1 sqrt(rho_forest), K_s / d_b) / b - k2 rho_ssd ) |gammadot|

where `d_b` is the slip-direction distance to the nearest grain
boundary, per element and slip system.

## Applying

Start from a fresh clone of TarletonGroup/CrystalPlasticity and apply
the six patches to `OXFORD-UMAT/`, then drop in `dbtrace.f`:

```bash
cd CrystalPlasticity/OXFORD-UMAT
for p in globalvariables initializations cpsolver hardening userinputs OXFORD-UMAT; do
    patch -p1 < ../../mprve/oxford-patch/$p.patch
done
cp ../../mprve/oxford-patch/dbtrace.f .
```

`K_s` goes in `hardeningparam(7)` (= PROPS slot 62 in the 300-constant
custom-material layout). **With `K_s = 0` (default) every patched code
path is inert and the model is identical to upstream.**

## d_b sources (userinputs.f: `dbconvect`)

| mode | meaning | needs |
|------|---------|-------|
| 0 | frozen t=0 table (literature practice) | `db_table.dat` |
| 1 | chord convection `d_b(t) = d_b0 * abs(F.s0)` | `db_table.dat` |
| 2 | per-increment ray-trace of the deformed facets | `db_table.dat`, `gb_rt.bin`, `gbnodes.txt`, `*Node File` output, `mp_mode=threads` |

- `db_table.dat` — one row per element, 12 slip-system distances (um),
  from `gen_db_table.py` (exact Laguerre cell-walk ray-trace) or
  `gen_db_table_vox.py` / `gen_db_table_stair_mp.py` (staircase
  geometry, for corruption studies). Missing file = term inactive.
- `gb_rt.bin` + `gbnodes.txt` — facet/node tables and per-ray candidate
  lists from `gen_gb_rt.py`.

## How mode 2 works

The GB facet quads are tracked by their actual mesh nodes: the deck
requests `*Node File, nset=GBN, frequency=1`, so Abaqus writes the GB
nodal displacements to the `.fil` at every increment; `URDFIL`
(in `dbtrace.f`) reads them back, moves each facet vertex as
`x = X0 + u + F_macro . offset` (the offset term reconstructs periodic
images; the PBC `*EQUATION`s make it exact), and re-traces every
(element, slip system, +-sense) ray with Möller–Trumbore against a
precomputed candidate list. Distances updated at the end of increment
n are used during increment n+1 (explicit in geometry, consistent with
the explicit state update). The t=0 trace reproduces the exact
Laguerre distances to machine precision.

`.fil` node labels are internal numbers when the model is
assembly-form; the mapping is auto-detected against `gbnodes.txt` and
the run is stopped on any mismatch.

Diagnostics written to the job directory: `dbdrift.log` (per update:
mean and min of d_b(t)/d_b0) and `db_rt_last.bin` (latest full table).

## Caveats

- mode 2 requires thread-parallel Abaqus (`mp_mode=threads`): with MPI
  the per-element state arrays are distributed and the rebuild would
  see only one rank's data.
- ray-trace cost: with the distance-scaled candidate pruning of
  `gen_gb_rt.py` (~134 facets/ray), a 49,920-element mesh costs a few
  seconds per increment single-threaded.
