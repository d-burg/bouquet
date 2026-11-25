#!/usr/bin/env julia
"""
compute_transport_stiffness.jl

Optional Julia FUSE / TurbulentTransport.jl / TJLF.jl bridge and ground-truth
validation script for bouquet's standalone Python Miller + TGLF-NN module.

Usage:
    julia scripts/compute_transport_stiffness.jl <input_dd.json> <output_stiffness.json> [--model sat2_em_d3d_azf-1_withnegD]
"""

using IMAS
using JSON
const TurbulentTransport = try
    @eval using TurbulentTransport
    Main.TurbulentTransport
catch
    @eval using FUSE
    Main.FUSE.TurbulentTransport
end

function _parse_model_sat_and_em(model_name::AbstractString)
    lower = lowercase(String(model_name))
    sat_sym = if occursin("sat0quench", lower)
        :sat0quench
    elseif occursin("sat1geo", lower)
        :sat1geo
    elseif occursin("sat3", lower)
        :sat3
    elseif occursin("sat2", lower)
        :sat2
    elseif occursin("sat1", lower)
        :sat1
    else
        :sat0
    end
    electromagnetic = !occursin("_es_", lower)
    return sat_sym, electromagnetic
end

function main()
    if length(ARGS) < 2
        error("Usage: julia compute_transport_stiffness.jl <input_dd.json> <output_stiffness.json> [--model MODEL]")
    end
    dd_path = ARGS[1]
    out_path = ARGS[2]
    model_name = "sat2_em_d3d_azf-1_withnegD"
    req_time = nothing
    for i in 3:length(ARGS)-1
        if ARGS[i] == "--model"
            model_name = ARGS[i+1]
        elseif ARGS[i] == "--time"
            req_time = parse(Float64, ARGS[i+1])
        end
    end

    dd = IMAS.json2imas(dd_path)
    if req_time !== nothing
        IMAS.global_time(dd, req_time)
    elseif length(dd.equilibrium.time_slice) > 1
        IMAS.global_time(dd, dd.equilibrium.time_slice[cld(end, 2)].time)
    end
    rho_grid = collect(range(0.10, 0.88; length=18))

    sat_sym, em_flag = _parse_model_sat_and_em(model_name)
    input_tglfs = TurbulentTransport.InputTGLF(dd, rho_grid, sat_sym, em_flag, true).tglfs
    sols = TurbulentTransport.run_tglfnn(input_tglfs; model_filename=model_name, warn_nn_train_bounds=false)

    # Finite-difference stiffness dQ/d(a/L_T)
    h = 0.08
    in_te_p = deepcopy(input_tglfs)
    in_te_m = deepcopy(input_tglfs)
    in_ti_p = deepcopy(input_tglfs)
    in_ti_m = deepcopy(input_tglfs)
    for k in eachindex(rho_grid)
        in_te_p[k].RLTS_1 += h
        in_te_m[k].RLTS_1 -= h
        in_ti_p[k].RLTS_2 += h
        in_ti_m[k].RLTS_2 -= h
        if !ismissing(in_ti_p[k].RLTS_3)
            in_ti_p[k].RLTS_3 += h
            in_ti_m[k].RLTS_3 -= h
        end
    end
    sol_te_p = TurbulentTransport.run_tglfnn(in_te_p; model_filename=model_name, warn_nn_train_bounds=false)
    sol_te_m = TurbulentTransport.run_tglfnn(in_te_m; model_filename=model_name, warn_nn_train_bounds=false)
    sol_ti_p = TurbulentTransport.run_tglfnn(in_ti_p; model_filename=model_name, warn_nn_train_bounds=false)
    sol_ti_m = TurbulentTransport.run_tglfnn(in_ti_m; model_filename=model_name, warn_nn_train_bounds=false)

    out = Dict(
        "model" => model_name,
        "sat_rule" => String(sat_sym),
        "electromagnetic" => em_flag,
        "rho_tor_norm" => rho_grid,
        "RLTS_1" => [t.RLTS_1 for t in input_tglfs],
        "RLTS_2" => [t.RLTS_2 for t in input_tglfs],
        "RLNS_1" => [t.RLNS_1 for t in input_tglfs],
        "BETAE" => [t.BETAE for t in input_tglfs],
        "XNUE" => [t.XNUE for t in input_tglfs],
        "Q_LOC" => [t.Q_LOC for t in input_tglfs],
        "KAPPA_LOC" => [t.KAPPA_LOC for t in input_tglfs],
        "DELTA_LOC" => [t.DELTA_LOC for t in input_tglfs],
        "Qe_gB" => [s.ENERGY_FLUX_e for s in sols],
        "Qi_gB" => [s.ENERGY_FLUX_i for s in sols],
        "Ge_gB" => [s.PARTICLE_FLUX_e for s in sols],
        "dQe_daLTe_gB" => [(sol_te_p[k].ENERGY_FLUX_e - sol_te_m[k].ENERGY_FLUX_e) / (2h) for k in eachindex(rho_grid)],
        "dQi_daLTi_gB" => [(sol_ti_p[k].ENERGY_FLUX_i - sol_ti_m[k].ENERGY_FLUX_i) / (2h) for k in eachindex(rho_grid)],
    )
    open(out_path, "w") do io
        JSON.print(io, out, 2)
    end
    println("Wrote TurbulentTransport stiffness profile to $(out_path)")
end

main()
