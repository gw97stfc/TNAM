using FileIO
using MeshIO
using GeometryBasics
using StaticArrays
using LinearAlgebra
include("sampling.jl")
using .sampling

# Setting the desired number of MC sample points.
n_abs = 100000
# Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
stl = load("test/Test_STLs/Icosphere1280.stl")
vertices = GeometryBasics.coordinates(stl)
# Converting the units into cm from mm.
vertices = vertices / 10
indices = GeometryBasics.faces(stl)
# Extracting the number of triangular faces used in the mesh.
n_faces = length(indices)
# Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
# Pre-allocating vector of MC coordinates and pre-scattering path lengths.
len_i = Vector{Float32}(undef, n_abs)
mc_coords = Vector{SVector{3, Float32}}(undef, n_abs)
# Calculating the extrema of the axis-aligned bounding box around the sample.
ranges = sampling.aabb_3d(vertices)
# Calculating the MC coordinates and initial path lengths.
sampling.sample!(ranges, e2s, e3s, v1s, len_i, mc_coords, true, n_faces, n_abs)
# Choosing the coordinate on the surface of the sphere.
test_surface = SVector{3, Float32}(0, 0.1, 0)
# Storing the distance between coordinates within the sample and the chosen point on the surface.
distances = zeros(n_abs)
for i in 1:n_abs
    distances[i] = norm(test_surface - mc_coords[i])
end
# Averaging the initial and final path lengths.
avg_len_i = sum(len_i) / n_abs # Should be 3R/4.
avg_len_f = sum(distances) / n_abs # Should be 6R/5.
println("The average initial path length is: $avg_len_i.")
println("The average final path length is: $avg_len_f.")
# Choosing a suitable attenuation coefficient.
# Assuming approximately elastic scattering.
mu = 11.34f0
# Finding the attenuation factor due to these average path lengths.
avg_atten = exp(-mu*(avg_len_f+avg_len_i))
println("The attenuation factor due to these average path lengths is: $avg_atten.")

