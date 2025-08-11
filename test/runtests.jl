using TestItemRunner
@run_package_tests


# Inputting a (ico)sphere .stl file and testing that all path lengths are approximately the radius when origin is at the centre.
@testitem "Sphere Test - Single Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
    vertices = GeometryBasics.coordinates(stl)
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating v1s, e2s, e3s needed for the MT algorithm.
    v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
    # Setting the centre of the circle.
    centre = SVector{3, Float32}(0, 0, 0)
    # Iterating through various direction vectors.
    for i in 1:100
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Testing if path length is approximately the radius, 1.
        radius = sin_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, centre, v1s, n_faces)
        @test isapprox(radius, 1f0, atol=1f-2)
    end
end


# Inputting a (ico)sphere .stl file and testing that all path lengths are approximately the radius when origin is at the centre.
@testitem "Sphere Test - General Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = FileIO.load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
    vertices = GeometryBasics.coordinates(stl)
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating v1s, e2s, e3s needed for the MT algorithm.
    v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
    # Setting the centre of the circle.
    centre = SVector{3, Float32}(0, 0, 0)
    # Iterating through various direction vectors.
    for i in 1:100
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Pre-allocating path length stores.
        λs = Vector{Float32}(undef, 10)
        path_lengths = Vector{Float32}(undef, 10)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Testing if path length is approximately the radius, 1.
        radius = gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, centre, v1s, λs, path_lengths, n_faces)
        @test isapprox(radius, 1f0; atol=1f-2)
    end
end


# Testing that a significant proportion of path lengths are calculated (as opposed to outputting nothing).
# Non-1 outputs arise due to floating point precision errors in duplicate sorting.
@testitem "Sufficient Output Test - Single Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Placing inside a let block to allow for local variable n_suc to work.
    let 
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "crystal.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # Determining the random coordinates within the sample.
        n_mc = 10000
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Tallying the number of successfully-calculated path lengths.
        n_suc = 0
        # Iterating through the sample points and testing if a path length is calculated for random directions.
        for i in 1:n_mc
            # Generating random direction vector.
            d_test = SVector{3, Float32}(randn(3))
            d_test = d_test / norm(d_test)
            # Calculating p, det for MT algorithm.
            p_test = Vector{SVector{3, Float32}}(undef, n_faces)
            det_test = Vector{Float32}(undef, n_faces)
            sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
            # Calculating path length.
            len = sin_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, mc_coords[i], v1s, n_faces)
            if typeof(len) == Float32
                n_suc += 1
            end 
        end
        # Testing whether at least 90% of path lengths are calculated.
        ratio = n_suc / n_mc
        println("The proportion of path length calculations correctly defined was $ratio for the single crystal program.")
        @test ratio > 0.9
    end
end


# Testing that a significant proportion of path lengths are calculated (as opposed to outputting nothing).
# Non-1 outputs arise due to floating point precision errors in duplicate sorting.
@testitem "Sufficient Output Test - General Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    # Placing inside a let block to allow for local variable n_suc to work.
    let 
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "7_Icospheres320.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # Determining the random coordinates within the sample.
        n_mc = 10000
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Tallying the number of successfully-calculated path lengths.
        n_suc = 0
        # Iterating through the sample points and testing if a path length is calculated for random directions.
        for i in 1:n_mc
            # Pre-allocating path length stores.
            λs = Vector{Float32}(undef, 10)
            path_lengths = Vector{Float32}(undef, 10)
            # Generating random direction vector.
            d_test = SVector{3, Float32}(randn(3))
            d_test = d_test / norm(d_test)
            # Calculating p, det for MT algorithm.
            p_test = Vector{SVector{3, Float32}}(undef, n_faces)
            det_test = Vector{Float32}(undef, n_faces)
            gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
            # Calculating path length.
            len = gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, mc_coords[i], v1s, λs, path_lengths, n_faces)
            if typeof(len) == Float32
                n_suc += 1
            end 
        end
        # Testing whether at least 90% of path lengths are calculated.
        ratio = n_suc / n_mc
        println("The proportion of path length calculations correctly defined was $ratio for the general crystal program.")
        @test ratio > 0.9
    end
end


# Testing if the outputs from the single and general crystal programs match.
# Trivial errors arise due to 'nothing' printed in some cases.
# Non-trivial errors arise due to small (perceived?) concave areas leading to an extra addition to the path length.
@testitem "Single and General Crystal Program Equivalence" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = FileIO.load(joinpath(@__DIR__, "..", "crystal.stl"))
    vertices = GeometryBasics.coordinates(stl)
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating v1s, e2s, e3s needed for the MT algorithm.
    v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
    # Determining the random coordinates within the sample.
    n_mc = 1000
    mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
    ranges = sampling.aabb_3d(vertices)
    sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
    # Iterating through the sample points and testing if the path lengths fom each program match.
    for i in 1:n_mc
        # Pre-allocating path length stores.
        λs = Vector{Float32}(undef, 10)
        path_lengths = Vector{Float32}(undef, 10)
        # Generating random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Calculating the path lengths from each program.
        sin_len = sin_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, mc_coords[i], v1s, n_faces)
        gen_len = gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, mc_coords[i], v1s, λs, path_lengths, n_faces)
        # Testing if path lengths are approximately equal.
        @test isapprox(sin_len, gen_len, atol=1f-2)
        # Printing the direction and origin of runs that failed to allow for further investigation.
        if typeof(sin_len) == Float32 && typeof(gen_len) == Float32
            if ~isapprox(sin_len, gen_len, atol=1f-2)
                println()
                println("The single and general crystal programs outputted path lengths $sin_len and $gen_len respectively.")
                println("The direction vector and position vector of this error were:")
                display(d_test)
                display(mc_coords[i])
                println()
            end
        end
    end
end


# Testing if the path length changes accordingly as the scattering site is moved.
@testitem "Path Length Variation with Origin - Single Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
    vertices = GeometryBasics.coordinates(stl)
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating v1s, e2s, e3s needed for the MT algorithm.
    v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
    for i in 1:1000
        # Setting the centre of the sphere.
        centre = SVector{3, Float32}(0, 0, 0)
        # Generating a random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Offsetting the origin by a random amount in the opposite direction to d_test.
        # As rand() ϵ [0, 1), d_test is normalised and the sphere is radius 1, origin should always lie within the sample.
        factor = rand()
        test = SVector{3, Float32}(centre - (factor * d_test))
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Testing if the path length is the radius, 1, plus this factor.
        len = sin_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, test, v1s, n_faces)
        @test isapprox(len, (1 + factor), atol=1f-2)
    end
end


# Testing if the path length changes accordingly as the scattering site is moved.
@testitem "Path Length Variation with Origin - General Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
    stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "Icosphere1280.stl"))
    vertices = GeometryBasics.coordinates(stl)
    indices = GeometryBasics.faces(stl)
    # Extracting the number of triangular faces used in the mesh.
    n_faces = length(indices)
    # Calculating v1s, e2s, e3s needed for the MT algorithm.
    v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
    for i in 1:1000
        # Pre-allocating path length stores.
        λs = Vector{Float32}(undef, 10)
        path_lengths = Vector{Float32}(undef, 10)
        # Setting the centre of the sphere.
        centre = SVector{3, Float32}(0, 0, 0)
        # Generating a random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Offsetting the origin by a random amount in the opposite direction to d_test.
        # As rand() ϵ [0, 1), d_test is normalised and the sphere is radius 1, origin should always lie within the sample.
        factor = rand()
        test = SVector{3, Float32}(centre - (factor * d_test))
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Testing if the path length is the radius, 1, plus this factor.
        len = gen_crystal.len_calc(e2s, e3s, d_test, p_test, det_test, test, v1s, λs, path_lengths, n_faces)
        @test isapprox(len, (1 + factor), atol=1f-2)
    end
end


# Testing if the attenuation increases with the post-scattering neutron energy.
@testitem "Attenuation Variation with Energy - Single Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Writing in a let block to allow for the local variable prev_atten.
    let
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "crystal.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # The reference attenuation coefficent at 25.3 meV in cm^-1.
        μ_ref = Float32(1)
        en_ref = Float32(25.3)
        # Setting initial attenuation coefficient in cm^-1.
        μi = Float32(2)
        # Determining the random coordinates within the sample.
        n_mc = 100
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Generating random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Creating fake vector of initial lengths.
        len_i = Float32.(ones(n_mc))
        # Storing the previous attenuation factor.
        prev_atten = 0
        for i in 1:10
            # Iterating through increasing final energies.
            en_test = Float32(i)
            # Calculating the attenuation factor for each different energy.
            attenuation = sin_crystal.atten_calc(d_test, en_test, v1s, e2s, e3s, mc_coords, len_i, p_test, det_test, n_faces, n_mc, μ_ref, en_ref, μi)
            # Checking if the attenuation factor has increased with this increased energy.
            @test attenuation > prev_atten
            # Assigning this attenuation factor to the previous attenuation factor.
            prev_atten = attenuation
        end
    end
end


# Testing if the attenuation increases with the post-scattering neutron energy.
@testitem "Attenuation Variation with Energy - General Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    # Writing in a let block to allow for the local variable prev_atten.
    let
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "7_Icospheres320.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # The reference attenuation coefficent at 25.3 meV in cm^-1.
        μ_ref = Float32(1)
        en_ref = Float32(25.3)
        # Setting initial attenuation coefficient in cm^-1.
        μi = Float32(2)
        # Determining the random coordinates within the sample.
        n_mc = 100
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Generating random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Creating fake vector of initial lengths.
        len_i = Float32.(ones(n_mc))
        # Pre-allocating path length stores.
        λs = Vector{Float32}(undef, 10)
        path_lengths = Vector{Float32}(undef, 10)
        # Storing the previous attenuation factor.
        prev_atten = 0
        for i in 1:10
            # Iterating through increasing final energies.
            en_test = Float32(i)
            # Calculating the attenuation factor for each different energy.
            attenuation = gen_crystal.atten_calc(d_test, en_test, v1s, e2s, e3s, mc_coords, len_i, p_test, det_test, λs, path_lengths, n_faces, n_mc, μ_ref, en_ref, μi)
            # Checking if the attenuation factor has increased with this increased energy.
            @test attenuation > prev_atten
            # Assigning this attenuation factor to the previous attenuation factor.
            prev_atten = attenuation
        end
    end
end


# Testing if the attenuation increases with decreasing path length.
@testitem "Attenuation Variation with Path Length - Single Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "sin_crystal.jl"))
    using .sin_crystal
    # Writing in a let block to allow for the local variable prev_atten.
    let
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "crystal.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # The reference attenuation coefficent at 25.3 meV in cm^-1.
        μ_ref = Float32(1)
        en_ref = Float32(25.3)
        # Setting initial attenuation coefficient in cm^-1.
        μi = Float32(2)
        # Determining the random coordinates within the sample.
        n_mc = 100
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Generating random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        sin_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Creating fake final energy.
        en_test = 1f0
        # Storing the previous attenuation factor.
        prev_atten = 0
        for i in 1:10
            # Iterating through decreasing initial path lengths.
            len_i = Float32.((10 - i) * ones(n_mc))
            # Calculating the attenuation factor for each different energy.
            attenuation = sin_crystal.atten_calc(d_test, en_test, v1s, e2s, e3s, mc_coords, len_i, p_test, det_test, n_faces, n_mc, μ_ref, en_ref, μi)
            # Checking if the attenuation factor has increased with this increased energy.
            @test attenuation > prev_atten
            # Assigning this attenuation factor to the previous attenuation factor.
            prev_atten = attenuation
        end
    end
end


# Testing if the attenuation increases with decreasing path length.
@testitem "Attenuation Variation with Path Length - General Crystal" begin
    using FileIO
    using GeometryBasics
    include(joinpath(@__DIR__, "..", "Modules", "sampling.jl"))
    using .sampling
    using StaticArrays
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "Modules", "gen_crystal.jl"))
    using .gen_crystal
    # Writing in a let block to allow for the local variable prev_atten.
    let
        # Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.
        stl = load(joinpath(@__DIR__, "..", "STL_FileExamples", "7_Icospheres320.stl"))
        vertices = GeometryBasics.coordinates(stl)
        indices = GeometryBasics.faces(stl)
        # Extracting the number of triangular faces used in the mesh.
        n_faces = length(indices)
        # Calculating v1s, e2s, e3s needed for the MT algorithm.
        v1s, e2s, e3s = sampling.ve_calc(vertices, indices)
        # The reference attenuation coefficent at 25.3 meV in cm^-1.
        μ_ref = Float32(1)
        en_ref = Float32(25.3)
        # Setting initial attenuation coefficient in cm^-1.
        μi = Float32(2)
        # Determining the random coordinates within the sample.
        n_mc = 100
        mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)
        ranges = sampling.aabb_3d(vertices)
        sampling.sample!(ranges, e2s, e3s, v1s, mc_coords, n_faces, n_mc)
        # Generating random direction vector.
        d_test = SVector{3, Float32}(randn(3))
        d_test = d_test / norm(d_test)
        # Calculating p, det for MT algorithm.
        p_test = Vector{SVector{3, Float32}}(undef, n_faces)
        det_test = Vector{Float32}(undef, n_faces)
        gen_crystal.pdet_calc!(d_test, e2s, e3s, p_test, det_test, n_faces)
        # Creating fake final energy.
        en_test = 1f0
        # Pre-allocating path length stores.
        λs = Vector{Float32}(undef, 10)
        path_lengths = Vector{Float32}(undef, 10)
        # Storing the previous attenuation factor.
        prev_atten = 0
        for i in 1:10
            # Iterating through decreasing initial path lengths.
            len_i = Float32.((10 - i) * ones(n_mc))
            # Calculating the attenuation factor for each different energy.
            attenuation = gen_crystal.atten_calc(d_test, en_test, v1s, e2s, e3s, mc_coords, len_i, p_test, det_test, λs, path_lengths, n_faces, n_mc, μ_ref, en_ref, μi)
            # Checking if the attenuation factor has increased with this increased energy.
            @test attenuation > prev_atten
            # Assigning this attenuation factor to the previous attenuation factor.
            prev_atten = attenuation
        end
    end
end