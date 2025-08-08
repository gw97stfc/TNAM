# These tests will focus on ensuring an accurate path length is outputted.
include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
using .sin_crystal
include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
using .gen_crystal
include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
using .sampling
using FileIO
using MeshIO
using GeometryBasics
using StaticArrays
using TestItems
using Test
using TestItemRunner
using LinearAlgebra

@testitem "test" begin
    a = 1
    @test isapprox(a, 1, atol=1f-2)
end
# # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
#     stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
#     vertices = GeometryBasics.coordinates(stl)
#     indices = GeometryBasics.faces(stl)
#     # Extracting the number of triangular faces used in the mesh.
#     n_faces = length(indices)
#     # Calculating v1s, e2s, e3s needed for the MT algorithm.
#     v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
#     # Setting the centre of the circle.
#     centre = SVector{3, Float32}(0, 0, 0)
#     # Iterating through random direction vectors.
#     for i in 1:8
#         d_test = SVector{3, Float32}(randn(3))
#         d_test = d_test / norm(d_test)
#         # Pre-allocating path length stores.
#         λs = Vector{Float32}(undef, 10)
#         path_lengths = Vector{Float32}(undef, 10)
#         # Calculating p, det for MT algorithm.
#         p_test = Vector{SVector{3, Float32}}(undef, n_faces)
#         det_test = Vector{Float32}(undef, n_faces)
#         gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
#         # Testing if path length is approximately the radius, 1.
#         @test isapprox(gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, centre, v1s, λs, path_lengths, n_faces), 1f0; atol=1f-2)
#     end


# # Inputting a (ico)sphere .stl file and testing that all path lengths are approximately the radius when origin is at the centre.
# @testitem "Sphere Test - General Crystal(s)" begin
#     # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
#     stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
#     vertices = GeometryBasics.coordinates(stl)
#     indices = GeometryBasics.faces(stl)
#     # Extracting the number of triangular faces used in the mesh.
#     n_faces = length(indices)
#     # Calculating v1s, e2s, e3s needed for the MT algorithm.
#     v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
#     # Setting the centre of the circle.
#     centre = SVector{3, Float32}(0, 0, 0)
#     # Iterating through various direction vectors.
#     possible_d = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [-1, -1, -1]]
#     for i in 1:8
#         d_test = SVector{3, Float32}(possible_d[i])
#         d_test = d_test / norm(d_test)
#         # Pre-allocating path length stores.
#         λs = Vector{Float32}(undef, 10)
#         path_lengths = Vector{Float32}(undef, 10)
#         # Calculating p, det for MT algorithm.
#         p_test = Vector{SVector{3, Float32}}(undef, n_faces)
#         det_test = Vector{Float32}(undef, n_faces)
#         gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
#         # Testing if path length is approximately the radius, 1.
#         @test isapprox(gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, centre, v1s, λs, path_lengths, n_faces), 1f0; atol=1f-3)
#     end
# end

# @testitem "Sphere Test - Single Crystal" begin
#     # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
#     stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
#     vertices = GeometryBasics.coordinates(stl)
#     indices = GeometryBasics.faces(stl)
#     # Extracting the number of triangular faces used in the mesh.
#     n_faces = length(indices)
#     # Calculating v1s, e2s, e3s needed for the MT algorithm.
#     v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
#     # Setting the centre of the circle.
#     centre = SVector{3, Float32}(0, 0, 0)
#     # Iterating through various direction vectors.
#     possible_d = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [-1, -1, -1]]
#     for i in 1:8
#         d_test = SVector{3, Float32}(possible_d[i])
#         d_test = d_test / norm(d_test)
#         # Calculating p, det for MT algorithm.
#         p_test = Vector{SVector{3, Float32}}(undef, n_faces)
#         det_test = Vector{Float32}(undef, n_faces)
#         sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
#         # Testing if path length is approximately the radius, 1.
#         @test isapprox(sin_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, centre, v1s, n_faces), 1f0, atol=1f-3)
#     end
# end
