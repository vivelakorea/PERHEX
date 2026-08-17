# 3D periodic polycrystal grain growth (Fan-Chen / GBEvolution).
# Purpose: a curved-boundary periodic microstructure snapshot for the
# PERHEX curved-GB meshing experiment. Physics constants are the classic
# Cu grain-growth example values; only the curvature-driven SHAPE matters.
#   moose-opt -i grain_growth_3d.i --n-threads=4  (or mpiexec -n N)

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 40
    ny = 40
    nz = 40
    xmax = 1000
    ymax = 1000
    zmax = 1000
  []
[]

[GlobalParams]
  op_num = 25
  var_name_base = gr
[]

[Variables]
  [PolycrystalVariables]
  []
[]

[UserObjects]
  [voronoi]
    type = PolycrystalVoronoi
    grain_num = 25
    file_name = bimodal_seeds.txt
    rand_seed = 24

    int_width = 40
  []
  [grain_tracker]
    type = GrainTracker
    threshold = 0.2
    connecting_threshold = 0.08
    compute_halo_maps = false
  []
[]

[ICs]
  [PolycrystalICs]
    [PolycrystalColoringIC]
      polycrystal_ic_uo = voronoi
    []
  []
[]

[BCs]
  [Periodic]
    [All]
      auto_direction = 'x y z'
      variable = 'gr0 gr1 gr2 gr3 gr4 gr5 gr6 gr7 gr8 gr9 gr10 gr11 gr12 gr13 gr14 gr15 gr16 gr17 gr18 gr19 gr20 gr21 gr22 gr23 gr24'
    []
  []
[]

[AuxVariables]
  [unique_grains]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [unique_grains]
    type = FeatureFloodCountAux
    variable = unique_grains
    flood_counter = grain_tracker
    field_display = UNIQUE_REGION
    execute_on = 'initial timestep_end'
  []
[]

[Kernels]
  [PolycrystalKernel]
  []
[]

[Materials]
  [GBEvo]
    type = GBEvolution
    T = 450
    wGB = 60
    GBmob0 = 2.5e-6
    Q = 0.23
    GBenergy = 0.708
  []
[]

[Postprocessors]
  [dt]
    type = TimestepSize
  []
  [n_grains]
    type = FeatureFloodCount
    variable = gr0
    threshold = 0.1
  []
[]

[Executioner]
  type = Transient
  scheme = bdf2
  solve_type = PJFNK
  petsc_options_iname = '-pc_type -pc_asm_overlap -sub_pc_type'
  petsc_options_value = 'asm 1 ilu'
  nl_abs_tol = 1e-9
  nl_rel_tol = 1e-7
  l_max_its = 30
  nl_max_its = 20
  [TimeStepper]
    type = IterationAdaptiveDT
    dt = 25
    growth_factor = 1.4
    cutback_factor = 0.6
    optimal_iterations = 8
  []
  end_time = 30000
  num_steps = 70
                    
[]

[Outputs]
  exodus = true
  time_step_interval = 5
  [console]
    type = Console
    max_rows = 8
  []
[]
