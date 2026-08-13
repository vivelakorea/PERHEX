# Aluminium material for OXFORD-UMAT via the 300-constant PROPS interface
# (custom FCC, material-id 2, Custom-Properties ON).
#
# Parameter set: Rubio, Haouala, LLorca, J. Mater. Res. 34 (2019) 2263
# (arXiv:1901.11145, Tables 1-2) = Haouala et al., IJP 126 (2020) 102600
# (Table 4). Their model: power-law slip (gdot0=1e-3, 1/m=20), tau0=0,
# CRSS emergent from dislocation density tau_c = mu*b*sqrt(sum a*rho)
# (Franciosi matrix), Kocks-Mecking evolution with K=9 (mean free path)
# and y_c=56 nm annihilation.
#
# Mapping to OXFORD (MPa-um units). OXFORD's model-3 update is
#   dssd = (k1/b*sqrt(rho_for) - k2*ssd)*|gammadot|   (k1 divided by b
# INTERNALLY), so against their rho_dot = (1/(bK)*sqrt(rho) - (2yc/b)*rho):
#   k1 = 1/K = 1/9 = 0.11111 (dimensionless)         -> hardeningparam(1)
#   k2 = 2*y_c/b = 2*0.056/2.86e-4 = 391.6 (-)       -> hardeningparam(2),
#        given directly so the parametric temperature branch is skipped
#   Taylor: OXFORD tau_c = tauc0 + G12*b*sqrt(gf^2*rho_forest); matching
#   mu*sqrt(a_avg) = 25000*0.35 with G12~C44=28500 -> gf = 0.306
#   (scalar approximation of the Franciosi interaction matrix - state this
#   simplification in the paper)
#   rho_0 = 0.1 um^-2 per slip system (1e11 m^-2), tauc0 = 0.
AL = dict(
    C11=108000.0, C12=61300.0, C44=28500.0,   # MPa (Rubio 2019 Table 1)
    gdot0=1e-3, n=20.0,                        # power law, 1/m = 20
    hardening_model=3,                         # Kocks-Mecking
    k1=0.11111, k2=391.6,                      # k1 = 1/K (see mapping above)
    geomfactor=0.306,                          # scalar Taylor (see note)
    burgers=2.86e-4,                           # 0.286 nm in micrometers
    tauc0=0.0,                                 # negligible friction stress
    rho0=0.1,                                  # um^-2 per slip system
    ks=5.0,                                    # GB storage K_s (Haouala 2020);
                                               # active only where the patched
                                               # UMAT finds db_table.dat
)


def build_props(euler, grain_id, p=AL):
    v = [0.0] * 300
    v[0:3] = [euler[0], euler[1], euler[2]]
    v[3] = float(grain_id)
    v[4] = 2.0          # custom FCC
    v[5] = 1.0          # use these PROPS
    v[6] = 2.0          # phase FCC
    v[7] = 12.0         # slip systems
    v[8] = 6.0          # screw systems
    v[9] = 0.0          # cubic slip off
    v[10] = p["geomfactor"]
    v[11] = p["C11"]
    v[12] = p["C12"]
    v[13] = p["C44"]
    v[20] = 1.0         # c/a (unused for FCC)
    v[24] = 3.0         # slip model: power law
    v[25] = p["gdot0"]
    v[26] = p["n"]
    v[54] = float(p["hardening_model"])
    v[55] = p["k1"]
    v[56] = p["k2"]
    v[61] = p.get("ks", 0.0)    # hardeningparam(7): GB storage K_s
    v[128] = p["rho0"]
    v[134] = p["burgers"]
    v[140] = p["tauc0"]
    return v


def write_material(f, grain_id, euler, depvar=300):
    f.write(f"*Material, name=MAT-GRAIN{grain_id}\n*Depvar\n{depvar},\n"
            f"*User Material, constants=300\n")
    v = build_props(euler, grain_id)
    for k in range(0, 300, 8):
        f.write(", ".join(f"{x:g}" for x in v[k:k + 8])
                + ("," if k < 292 else "") + "\n")
