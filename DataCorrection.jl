# This program will apply the absorption and flux corrections to the set of inputted .nxpse files.
using FileIO
using GeometryBasics
using LinearAlgebra
using MeshIO
using StaticArrays
using Base.Threads
using BenchmarkTools
include("Modules/nxspe.jl")
using .nxspe
include("Modules/sampling.jl")
using .sampling
include("Modules/gen_crystal.jl")
using .gen_crystal
include("Modules/sin_crystal.jl")
using .sin_crystal
include("Modules/projections.jl")
using .projections


"""
Corrects the measured neutron signals by taking into consideration the variation of flux incident on the sample with the rotation angle, ψ, and the absorption of neutrons in the sample.

Parameters
----------
nxspes (vector with string elements): Path to each .nxspe file.
ψs (vector with float elements): Sample rotation angles corresponding to each file, in degrees.
sample (string): Path to the .stl file.
complex (bool): Program that this function will use. True = general crystal morphology. False = single crystal.
μ_ref (float): Attenuation coefficent at the reference energy, in cm^-1.
en_ref (float): Reference energy, in meV.
n_abs (integer): Number of MC sample points used in the absorption correction.
n_flux (integer): Number of MC sample points used in the flux correction.

Returns
-------
c_data (vector of matrix with float elements): Corrected data for each file.
"""
function correct(
    nxspes :: Vector{String}, 
    ψs :: Vector{Float32}, 
    sample :: String, 
    complex :: Bool, 
    μ_ref :: Float32, 
    en_ref :: Float32, 
    n_abs :: Integer, 
    n_flux :: Integer
    )
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
    # Creating the store of corrected data.
    c_data = Vector{Matrix{Float32}}(undef, n_files)
    # Pre-allocating the vectors to store contents of .nxspe files.
    en_i = Vector{Float32}(undef, n_files)
    azi = Vector{Vector{Float32}}(undef, n_files)
    pol = Vector{Vector{Float32}}(undef, n_files)
    data = Vector{Matrix{Float32}}(undef, n_files)
    Δen = Vector{Vector{Float32}}(undef, n_files)
    # Pre-extracting the contents of each .nxspe file.
    for j in 1:n_files
        en_i[j], azi[j], pol[j], data[j], Δen[j] = nxspe.extract(nxspes[j])
    end
    # Parallelising the program to correct multiple files at once.
    #@threads for i in 1:n_files
    for i in 1:n_files
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
        # Setting the desired number of MC sample points and creating vector for coordinates.
        mc_coords = Vector{SVector{3, Float32}}(undef, n_abs)
        # Calculating the extrema of the axis-aligned bounding box around the sample.
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_abs)
        # Calculating the pre-scattering attenuation coefficent.
        μi = crystal.μ_calc(μ_ref, en_ref, en_i[i])
        # The pre-scattering neutron path lengths are dependent only on the MC coordinates.
        # They can, therefore, be calculated and stored.
        len_i = crystal.len_i_calc(e2s, e3s, v1s, mc_coords, n_faces, n_abs)
        # Calculating this grid of attenuation factors.
        atten_grid = crystal.a_grid_calc(data[i], kx, ky, kz, ef_bins, v1s, e2s, e3s, mc_coords, len_i, n_bins, n_detectors, n_faces, n_abs, μ_ref, en_ref, μi)
        # Correcting the data for absorption.
        ca_data = nxspe.abs_corr(data[i], atten_grid)
        # Correcting the (absorption-corrected) data for varying flux.
        cf_ca_data = projections.flux_corr(ca_data, r_vertices, indices, n_faces, n_flux)
        # Adding this corrected data to the vector containing that for all files.
        c_data[i] = cf_ca_data
    end
    return c_data
end

c_data = correct(["test_nxspe_data/LET104215_3.7meV_1to1.nxspe", "test_nxspe_data/LET104216_3.7meV_1to1.nxspe"], [0f0, 0f0], "crystal.stl", false, 1f0, 25.3f0, 5, 10000)

#@benchmark correct(["test_nxspe_data/LET104215_3.7meV_1to1.nxspe", "test_nxspe_data/LET104216_3.7meV_1to1.nxspe", "test_nxspe_data/LET104217_3.7meV_1to1.nxspe", "test_nxspe_data/LET104218_3.7meV_1to1.nxspe", "test_nxspe_data/LET104219_3.7meV_1to1.nxspe", "test_nxspe_data/LET104220_3.7meV_1to1.nxspe", "test_nxspe_data/LET104221_3.7meV_1to1.nxspe", "test_nxspe_data/LET104222_3.7meV_1to1.nxspe", "test_nxspe_data/LET104223_3.7meV_1to1.nxspe", "test_nxspe_data/LET104224_3.7meV_1to1.nxspe"], [0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0], "crystal.stl", false, 5f0, 25.3f0, 5, 10000)

#@benchmark correct(["test_nxspe_data/LET104215_3.7meV_1to1.nxspe", "test_nxspe_data/LET104216_3.7meV_1to1.nxspe"], [0f0, 0f0], "crystal.stl", false, 5f0, 25.3f0, 1, 10000)