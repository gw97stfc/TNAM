# This program will apply the absorption and flux corrections to the set of inputted .nxpse files.


using FileIO
using GeometryBasics
using LinearAlgebra
using MeshIO
using StaticArrays
using Base.Threads
using HDF5
using Random
include("nxspe.jl")
using .nxspe
include("sampling.jl")
using .sampling
include("gen_crystal.jl")
using .gen_crystal
include("sin_crystal.jl")
using .sin_crystal
include("projections.jl")
using .projections


# Defining the function to correct the data for neutron flux and absorption.


"""
Corrects the measured neutron signals by taking into consideration the variation of flux incident on the sample with the rotation angle, ψ, and the absorption of neutrons in the sample.
Corrected intensities are as if no neutrons had been absorbed and all .nxspe files had seen the same neutron flux.
Outputs corrected .nxspe files into a chosen directory.

Parameters
----------
nxspes (vector with integer elements): Variable part of each .nxspe file.
input_file_dir (string): Path to folder containing the input .nxspe files.
file_start (string): Constant part at the start of the .nxspe file.
file_end (string): Constant part at the end of the .nxspe file.
output_file_dir (string): Path to folder where output .nxspe files will be stored.
ψs (vector with float or integer elements): Sample rotation angles corresponding to each file, in degrees.
sample (string): Path to the .stl file (with units of mm).
complex (bool): Program that this function will use. true = general crystal morphology. false = single, convex crystal.
μ_ref (float or integer): Attenuation coefficent at the reference energy, in cm^-1.
en_ref (float or integer): (Optional) reference energy, in meV. Default of 25.3 meV.
n_abs (integer): (Optional) Number of MC sample points used in the absorption correction. Default of 1000.
n_flux (integer): (Optional) Number of MC sample points used in the flux correction. Default of 100000.
seed (integer): (Optional) Random seed.

"""
function correct_af(
    nxspes :: Vector{Int},
    input_file_dir :: String,
    file_start :: String,
    file_end :: String,
    output_file_dir :: String, 
    ψs :: AbstractVector{<:Real}, 
    sample :: String, 
    complex :: Bool, 
    μ_ref :: Real; 
    en_ref :: Real=25.3f0, 
    n_abs :: Integer=1000, 
    n_flux :: Integer=100000,
    seed :: Union{Int, Nothing}=nothing
    )
    # Ensuring that the number of .nxspe files is consistent.
    @assert length(nxspes) == length(ψs) "The vectors containing the angles and the variable part of the file name should be the same length."
    # Converting the inputs to Float32.
    ψs = Float32.(ψs)
    μ_ref = Float32(μ_ref)
    en_ref = Float32(en_ref)
    # Setting the random number seed.
    if isnothing(seed)
        Random.seed!()
    else
        Random.seed!(seed)
    end
    # Setting the complexity of the sample.
    # The crystal morphology affects the speed of program.
    # 'complex' means a sample with a concave surface or multiple crystals.
    if complex
        crystal = gen_crystal
    else
        crystal = sin_crystal
    end
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(sample)
    vertices = GeometryBasics.coordinates(stl)
    # Converting the units into cm from mm.
    vertices = vertices / 10
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating the number of .nxspe files.
    n_files = length(nxspes)
    # Creating the store of corrected data for each file.
    c_data = Vector{Matrix{Float64}}(undef, n_files)
    # Pre-allocating the vectors to store contents of .nxspe files.
    en_i = Vector{Float32}(undef, n_files)
    azi = Vector{Vector{Float32}}(undef, n_files)
    pol = Vector{Vector{Float32}}(undef, n_files)
    data = Vector{Matrix{Float32}}(undef, n_files)
    Δen = Vector{Vector{Float32}}(undef, n_files)
    # Pre-extracting the contents of each .nxspe file.
    for j in 1:n_files
        en_i[j], azi[j], pol[j], data[j], Δen[j] = nxspe.extract(input_file_dir * file_start * "$(nxspes[j])" * file_end)
    end
    # Setting the average proportion of MC sample points used for each file.
    av_acc_rate = 0
    # Parallelising the program to correct multiple files at once.
    @threads for i in 1:n_files
        # Finding the initial wavevector, in Angstrom^-1, from the initial energy.
        ki = zeros(3)
        ki[1] = nxspe.magk_calc(en_i[i])
        # Converting ki to a static array.
        ki = SVector{3, Float32}(ki)
        # Extracting the number of energy bins and number of detectors.
        n_bins = length(Δen[i]) - 1
        n_detectors = length(azi[i])
        # Calculating the final neutron energy, in meV, for each energy bin.
        ef_bins = nxspe.ef_calc(en_i[i], Δen[i], n_bins)
        # Calculating the final wavevector, in Angstrom^-1, for each energy bin and each detector.
        kx, ky, kz = nxspe.kf_calc(ef_bins, pol[i], azi[i])
        # Rotating the sample to the orientation of this specific file.
        # Assumes the .stl file describes the sample at 0 degrees.
        # Assumes ψ is anti-clockwise rotation angle.
        r_vertices = projections.rotate(ψs[i], vertices)
        # Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
        v1s, e2s, e3s = sampling.ve_calc(r_vertices, indices)
        # Pre-allocating vector of MC coordinates and pre-scattering path lengths.
        len_i = Vector{Float32}(undef, n_abs)
        mc_coords = Vector{SVector{3, Float32}}(undef, n_abs)
        # Calculating the extrema of the axis-aligned bounding box around the sample.
        ranges = sampling.aabb_3d(r_vertices)
        # Calculating the MC coordinates and initial path lengths.
        sampling.sample!(ranges, e2s, e3s, v1s, len_i, mc_coords, complex, n_faces, n_abs)
        # Calculating the pre-scattering attenuation coefficent.
        μi = crystal.μ_calc(μ_ref, en_ref, en_i[i])
        # Calculating this grid of attenuation factors.
        atten_grid, acc_rate = crystal.a_grid_calc(data[i], kx, ky, kz, ef_bins, v1s, e2s, e3s, mc_coords, len_i, n_bins, n_detectors, n_faces, n_abs, μ_ref, en_ref, μi)
        av_acc_rate += acc_rate
        # Correcting the data for absorption.
        ca_data = nxspe.abs_corr(data[i], atten_grid)
        # Correcting the (absorption-corrected) data for varying flux.
        cf_ca_data = projections.flux_corr(ca_data, r_vertices, indices, n_faces, n_flux)
        # Adding this corrected data to the vector containing that for all files.
        c_data[i] = cf_ca_data
    end
    # Creating and filling these corrected .nxspe files.
    for k in 1:n_files
        # Creating the .nxspe file outputted.
        corrected_nxspe = output_file_dir * "afC_" * file_start * "$(nxspes[k])" * file_end
        # Copying the .nxspe file into corrected, output file.
        cp(input_file_dir * file_start * "$(nxspes[k])" * file_end, corrected_nxspe)
        # Replacing the data with the corrected data.
        h5open(corrected_nxspe, "r+") do f
            write(f["ws_out/NXSPE_info/psi"], ψs[k])
            write(f["ws_out/data/data"], c_data[k])
        end
    end
    # Calculating the average acceptance rate across the files.
    av_acc_rate = av_acc_rate / n_files
    @assert av_acc_rate > 0.8 "The proportion of n_abs (MC sample points used in the absorption correction) being used is $av_acc_rate. If this is not enough accuracy, please increase n_abs."
end


# Defining the function to correct the data for absorption.


"""
Corrects the measured neutron signals by taking into consideration the absorption of neutrons in the sample.
Corrected intensities are as if no neutrons had been absorbed.
Outputs corrected nxspe files into a chosen directory.

Parameters
----------
nxspes (vector with integer elements): Variable part of each .nxspe file.
input_file_dir (string): Path to folder containing input .nxspe files.
file_start (string): Constant part at the start of the .nxspe file.
file_end (string): Constant part at the end of the .nxspe file.
output_file_dir (string): Path to folder where output .nxspe files will be stored.
ψs (vector with float or integer elements): Sample rotation angles corresponding to each file, in degrees.
sample (string): Path to the .stl file (with units of mm).
complex (bool): Program that this function will use. true = general crystal morphology. false = single crystal.
μ_ref (float or integer): Attenuation coefficent at the reference energy, in cm^-1.
en_ref (float or integer): (Optional) reference energy, in meV. Default of 25.3 meV.
n_abs (integer): (Optional) Number of MC sample points used in the absorption correction. Default of 1000.
seed (integer): (Optional) Random seed.

"""
function correct_a(
    nxspes :: Vector{Int},
    input_file_dir :: String, 
    file_start :: String, 
    file_end :: String, 
    output_file_dir :: String,  
    ψs :: AbstractVector{<:Real}, 
    sample :: String, 
    complex :: Bool, 
    μ_ref :: Real; 
    en_ref :: Real=25.3f0,
    n_abs :: Integer=1000,
    seed :: Union{Int, Nothing}=nothing
    )
    # Ensuring that the number of .nxspe files is consistent.
    @assert length(nxspes) == length(ψs) "The vectors containing the angles and the variable part of the file name should be the same length."
    # Converting the inputs to Float32.
    ψs = Float32.(ψs)
    μ_ref = Float32(μ_ref)
    en_ref = Float32(en_ref)
    # Setting the random number seed.
    if isnothing(seed)
        Random.seed!()
    else
        Random.seed!(seed)
    end
    # Setting the complexity of the sample.
    # The crystal morphology affects the speed of program.
    # 'complex' means a sample with a concave surface or multiple crystals.
    if complex
        crystal = gen_crystal
    else
        crystal = sin_crystal
    end
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(sample)
    vertices = GeometryBasics.coordinates(stl)
    # Converting the units into cm from mm.
    vertices = vertices / 10
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating the number of .nxspe files.
    n_files = length(nxspes)
    # Creating the store of corrected data for each .nxspe file.
    c_data = Vector{Matrix{Float64}}(undef, n_files)
    # Pre-allocating the vectors to store contents of .nxspe files.
    en_i = Vector{Float32}(undef, n_files)
    azi = Vector{Vector{Float32}}(undef, n_files)
    pol = Vector{Vector{Float32}}(undef, n_files)
    data = Vector{Matrix{Float32}}(undef, n_files)
    Δen = Vector{Vector{Float32}}(undef, n_files)
    # Pre-extracting the contents of each .nxspe file.
    for j in 1:n_files
        en_i[j], azi[j], pol[j], data[j], Δen[j] = nxspe.extract(input_file_dir * file_start * "$(nxspes[j])" * file_end)
    end
    # Setting the average proportion of MC sample points used for each file.
    av_acc_rate = 0
    # Parallelising the program to correct multiple files at once.
    @threads for i in 1:n_files
        # Finding the initial wavevector, in Angstrom^-1, from the initial energy.
        ki = zeros(3)
        ki[1] = nxspe.magk_calc(en_i[i])
        # Converting ki to a static array.
        ki = SVector{3, Float32}(ki)
        # Extracting the number of energy bins and number of detectors.
        n_bins = length(Δen[i]) - 1
        n_detectors = length(azi[i])
        # Calculating the final neutron energy, in meV, for each energy bin.
        ef_bins = nxspe.ef_calc(en_i[i], Δen[i], n_bins)
        # Calculating the final wavevector, in Angstrom^-1, for each energy bin and each detector.
        kx, ky, kz = nxspe.kf_calc(ef_bins, pol[i], azi[i])
        # Rotating the sample to the orientation of this specific file.
        # Assumes the .stl file describes the sample at 0 degrees.
        # Assumes ψ is anti-clockwise rotation angle.
        r_vertices = projections.rotate(ψs[i], vertices)
        # Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
        v1s, e2s, e3s = sampling.ve_calc(r_vertices, indices)
        # Pre-allocating vector of MC coordinates and pre-scattering path lengths.
        len_i = Vector{Float32}(undef, n_abs)
        mc_coords = Vector{SVector{3, Float32}}(undef, n_abs)
        # Calculating the extrema of the axis-aligned bounding box around the sample.
        ranges = sampling.aabb_3d(r_vertices)
        # Calculating the MC coordinates and initial path lengths.
        sampling.sample!(ranges, e2s, e3s, v1s, len_i, mc_coords, complex, n_faces, n_abs)
        # Calculating the pre-scattering attenuation coefficent.
        μi = crystal.μ_calc(μ_ref, en_ref, en_i[i])
        # Calculating this grid of attenuation factors.
        atten_grid, acc_rate = crystal.a_grid_calc(data[i], kx, ky, kz, ef_bins, v1s, e2s, e3s, mc_coords, len_i, n_bins, n_detectors, n_faces, n_abs, μ_ref, en_ref, μi)
        av_acc_rate += acc_rate
        # Correcting the data for absorption.
        ca_data = nxspe.abs_corr(data[i], atten_grid)
        # Adding this corrected data to the vector containing that for all files.
        c_data[i] = ca_data
    end
    # Creating and filling these corrected .nxspe files.
    for k in 1:n_files
        # Creating the .nxspe file outputted.
        corrected_nxspe = output_file_dir * "aC_" * file_start * "$(nxspes[k])" * file_end
        # Copying the .nxspe file into corrected, output file.
        cp(input_file_dir * file_start * "$(nxspes[k])" * file_end, corrected_nxspe)
        # Replacing the data with the corrected data.
        h5open(corrected_nxspe, "r+") do f
            write(f["ws_out/NXSPE_info/psi"], ψs[k])
            write(f["ws_out/data/data"], c_data[k])
        end
    end
    # Calculating the average acceptance rate across the files.
    av_acc_rate = av_acc_rate / n_files
    @assert av_acc_rate > 0.8 "The proportion of n_abs (MC sample points used in the absorption correction) being used is $av_acc_rate. If this is not enough accuracy, please increase n_abs."
end


"""
Corrects the measured neutron signals by taking into consideration the variation of flux incident on the sample with the rotation angle, ψ.
Corrected intensities are as if all .nxspe file had seen the same neutron flux.
Outputs corrected nxspe files into a chosen directory.

Parameters
----------
nxspes (vector with integer elements): Variable part of each .nxspe file.
input_file_dir (string): Path to folder containing input .nxspe files.
file_start (string): Constant part at the start of the .nxspe file.
file_end (string): Constant part at the end of the .nxspe file.
output_file_dir (string): Path to folder where output .nxspe files will be stored.
ψs (vector with float or integer elements): Sample rotation angles corresponding to each file, in degrees.
sample (string): Path to the .stl file (with units of mm).
n_flux (integer): (Optional) Number of MC sample points used in the flux correction. Default of 100000.
seed (integer): (Optional) Random seed.

"""
function correct_f(
    nxspes :: Vector{Int}, 
    input_file_dir :: String, 
    file_start :: String, 
    file_end :: String, 
    output_file_dir :: String, 
    ψs :: AbstractVector{<:Real}, 
    sample :: String; 
    n_flux :: Integer=100000,
    seed :: Union{Int, Nothing}=nothing
    )
    # Ensuring that the number of .nxspe files is consistent.
    @assert length(nxspes) == length(ψs) "The vectors containing the angles and the variable part of the file name should be the same length."
    # Converting the inputs to Float32.
    ψs = Float32.(ψs)
    # Setting the random number seed.
    if isnothing(seed)
        Random.seed!()
    else
        Random.seed!(seed)
    end
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(sample)
    vertices = GeometryBasics.coordinates(stl)
    # Converting the units into cm from mm.
    vertices = vertices / 10
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating the number of .nxspe files.
    n_files = length(nxspes)
    # Creating the store of corrected data for each .nxspe file.
    c_data = Vector{Matrix{Float64}}(undef, n_files)
    # Pre-allocating the vector to store contents of .nxspe files.
    data = Vector{Matrix{Float32}}(undef, n_files)
    # Pre-extracting the data from each .nxspe file.
    for j in 1:n_files
        data[j] = nxspe.extract_data(input_file_dir * file_start * "$(nxspes[j])" * file_end)
    end
    # Parallelising the program to correct multiple files at once.
    @threads for i in 1:n_files
        # Rotating the sample to the orientation of this specific file.
        # Assumes the .stl file describes the sample at 0 degrees.
        # Assumes ψ is anti-clockwise rotation angle.
        r_vertices = projections.rotate(ψs[i], vertices)
        # Correcting the data for varying flux.
        cf_data = projections.flux_corr(data[i], r_vertices, indices, n_faces, n_flux)
        # Adding this corrected data to the vector containing that for all files.
        c_data[i] = cf_data
    end
    # Creating and filling these corrected .nxspe files.
    for k in 1:n_files
        # Creating the .nxspe file outputted.
        corrected_nxspe = output_file_dir * "fC_" * file_start * "$(nxspes[k])" * file_end
        # Copying the .nxspe file into corrected, output file.
        cp(input_file_dir * file_start * "$(nxspes[k])" * file_end, corrected_nxspe)
        # Replacing the data with the corrected data.
        h5open(corrected_nxspe, "r+") do f
            write(f["ws_out/NXSPE_info/psi"], ψs[k])
            write(f["ws_out/data/data"], c_data[k])
        end
    end
end
