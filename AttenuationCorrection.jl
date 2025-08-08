using FileIO
using GeometryBasics
using LinearAlgebra
using MeshIO
using SparseArrays
using StaticArrays
include("Modules/nxspe.jl")
using .nxspe
include("Modules/sampling.jl")
using .sampling
include("Modules/gen_crystal.jl")
using .gen_crystal
include("Modules/sin_crystal.jl")
using .sin_crystal


# Setting the complexity of the sample.
# The crystal morphology affects the speed of program.
# 'complex' means a sample with a concave surface or multiple crystals.


const complex = false
# Ensuring the program works with either method.
if complex
    const crystal = gen_crystal
else
    const crystal = sin_crystal
end


# Extracting the contents of the .nxspe file.


en_i, azi, pol, data, Δen = nxspe.extract("test_nxspe_data/LET104215_3.7meV_1to1.nxspe")



# Calculating ki, n_bins, n_detectors from the extracted content of the .nxspe file.


# Finding the initial wavevector, in Angstrom^-1, from the initial energy.
ki = zeros(3)
ki[1] = nxspe.magk_calc(en_i)
# Converting ki to a static array.
ki = SVector{3, Float32}(ki)
# Extracting the number of energy bins and number of detectors.
const n_bins = length(Δen) - 1
const n_detectors = length(azi)


# Calculating the final neutron energy, in meV, for each energy bin.


ef_bins = nxspe.ef_calc(en_i, Δen, n_bins)


# Calculating the final wavevector, in Angstrom^-1, for each energy bin and each detector.


kx, ky, kz = nxspe.kf_calc(ef_bins, pol, azi)


# Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.


stl = load("crystal.stl")
vertices = GeometryBasics.coordinates(stl)
indices = GeometryBasics.faces(stl)
# Extracting the number of triangular faces used in the mesh.
const n_faces = length(indices)


# Setting the desired number of MC sample points and creating vector for coordinates.


const n_mc = 10
mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)


# Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
# The vertices of the triangular faces are labelled V1, V2, V3.

v1s, e2s, e3s = sampling.ve_calc(vertices, indices)


# Setting the (estimated) parameters of the sample.


# The reference attenuation coefficent at 25.3 meV in cm^-1.
const μ_ref = Float32(1)
const en_ref = Float32(25.3)


# Calculating the pre-scattering attenuation coefficent.


const μi = crystal.μ_calc(μ_ref, en_ref, en_i)


# Generating the sample points.


# Calculating the extrema of the axis-aligned bounding box around the sample.
ranges = sampling.aabb_3d(vertices)
sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)


# The pre-scattering neutron path lengths are dependent only on the MC coordinates.
# They can, therefore, be calculated and stored.


len_i = crystal.len_i_calc(e2s, e3s, v1s, mc_coords, n_faces, n_mc)


# Converting the grid of data and attenuation factors to sparse arrays.


# Replacing every NaN value with zero to allow conversion to sparse matrix.
data_copy = copy(data)
data_copy .= ifelse.(isnan.(data_copy), 0, data)
s_data = sparse(data_copy)


# Testing the time taken to output this grid of attenuation factors.


atten_grid = crystal.a_grid_calc(s_data, kx, ky, kz, ef_bins, v1s, e2s, e3s, mc_coords, len_i, n_bins, n_detectors, n_faces, n_mc, μ_ref, en_ref, μi)
# Converting the attenuation factors to a sparse matrix.
s_atten = sparse(atten_grid)
# Testing the same (known to be non-zero) datapoint.
println("The attenuation factor at this test point was $(s_atten[160, 6])")


# Calculating the corrected data/signal by dividing each measured signal by the corresponding attenuation factor.


c_data = nxspe.abs_corr(s_data, s_atten, n_bins, n_detectors)
# Converting this corrected data into a sparse matrix.
s_c_data = sparse(c_data)
# Testing the same (known to be non-zero) datapoint.
println("The signal at this test point was $(s_data[160,6])")
println("The corrected signal is therefore $(s_c_data[160,6])")
