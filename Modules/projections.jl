# This module stores functions to aid in calculations of orthographic projection areas and rotating samples.
module projections


using GeometryBasics
using StaticArrays
using Distributions
using SparseArrays


# Defining a function to rotate the sample about the z-axis.


"""
Rotates the sample by θ degrees. This is accomplished by mutating each vertex.

Parameters
----------
θ (float): Rotation angle, in degrees.
vertices (vector with 3-vector elements with float elements): Vertices of sample, in units of .stl file.

Returns
-------
vertices (vector with 3-vector elements with float elements): Rotated vertices of sample, in units of .stl file.
"""
function rotate!(θ :: Float32, vertices :: Vector{Point{3, Float32}}) :: Vector{Point{3, Float32}}
    # Converting the angle to radians.
    θ = deg2rad(θ)
    # Calculating rotation matrix.
    R_z = Matrix{Float32}([[cos(θ), -sin(θ), 0] [sin(θ), cos(θ), 0] [0, 0, 1]])
    # Applying rotation matrix to each vertex.
    @inbounds for i in 1:length(vertices)
        vertices[i] = R_z * vertices[i]
    end
    return vertices
end


# Defining a function that calculates the vertices and triangles projected onto the y-z plane.


"""
Calculates the coordinates of the vertices and triangles projected onto the y-z plane (plane perpendicular to the neutron beam).
Accomplished by simply reading off the y and z coordinates.

Parameters
----------
vertices (vector with 3-vector elements with float elements): Vertices of sample, in units of .stl file.
indices (n_faces vector of 3-vectors with integer elements): Indices of the vertices that make up each face.
n_faces (integer): Number of faces.

Returns
-------
p_vertices (n_vert - vector of 2-vectors with float elements): 2D projected vertices, in units of the .stl file.
p_triangles (n_faces - vector of 3-vectors of 2-vectors with float elements): 2D projected vertices of each triangle, in units of the .stl file.
"""
function project(
    vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}, 
    n_faces :: Integer
    ) :: Tuple{Vector{SVector{2, Float32}}, Vector{Vector{SVector{2, Float32}}}}
    # Calculating the number of vertices.
    n_vert = length(vertices)
    # Projecting the sample onto the y-z plane.
    p_vertices = Vector{SVector{2, Float32}}(undef, n_vert)
    for i in 1:n_vert
        p_vertices[i] = [vertices[i][2], vertices[i][3]]
    end
    # Grouping these projected vertices by the triangles they create.
    p_triangles = Vector{Vector{SVector{2, Float32}}}(undef, n_faces)
    for i in 1:n_faces
        p_triangles[i] = [p_vertices[indices[i][1]], p_vertices[indices[i][2]], p_vertices[indices[i][3]]]
    end
    return p_vertices, p_triangles
end


# Defining a function that calculates the area of the 2D bounding box that surrounds the projected sample.


"""
Calculates the area of the 2D bounding box that surrounds the projected sample.

Parameters
----------
p_vertices (n_vert - vector of 2-vectors with float elements): 2D projected vertices, in units of the .stl file.

Returns
-------
min_coord (2-vector with float elements): Minimum value of y and z coordinate, in units of .stl file.
max_coord (2-vector with float elements): Maximum value of y and z coordinate, in units of .stl file.
rec_area (float): Area of bounding box, in unit of the .stl file squared.
"""
function aabb_2d(p_vertices :: Vector{SVector{2, Float32}}) :: Tuple{Vector{Float32}, Vector{Float32}, Float32}
    # Finding the maximum and minimum value of each projected coordinate.
    max_coord = [maximum(getindex.(p_vertices, 1)), maximum(getindex.(p_vertices, 2))]
    min_coord = [minimum(getindex.(p_vertices, 1)), minimum(getindex.(p_vertices, 2))]
    # Calculating the area of the bounding rectangle.
    rec_area = (max_coord[1] - min_coord[1]) * (max_coord[2] - min_coord[2])
    return min_coord, max_coord, rec_area
end


# Defining the function to calculate the area, in units of the .stl file squared, of the sample projected into the y-z plane.


"""
Calculates the area of the sample projected onto the y-z plane. This is the cross-sectional area visible to the neutron beam.
Adopts a Monte Carlo approach to achieve this.
Ratio of random coordinates in the sample projection to total is equal to the ratio of sample projection area to total bounding area.

Parameters
----------
min_coord (2-vector with float elements): Minimum value of y and z coordinate, in units of .stl file.
max_coord (2-vector with float elements): Maximum value of y and z coordinate, in units of .stl file.
p_triangles (n_faces - vector of 3-vectors of 2-vectors with float elements): Coordinates of projected triangles, in units of .stl file.
n_tot (integer): Total number of random coordinates to be generated.
n_faces (integer): Total number of faces.
rec_area (float): Area of bounding rectangle.

Returns
-------
area (float): Sample projection area, in units of .stl file squared.
"""
function area_calc(
    min_coord :: Vector{Float32}, 
    max_coord :: Vector{Float32}, 
    p_triangles :: Vector{Vector{SVector{2, Float32}}}, 
    n_tot :: Integer, 
    n_faces :: Integer, 
    rec_area :: Float32 
    ) :: Float32
    # Tallying the number of random coordinates that lie within the sample.
    n_in = 0
    # Pre-calculating the uniform distribution describing the bounding rectangle.
    y_range = Uniform(min_coord[1], max_coord[1])
    z_range = Uniform(min_coord[2], max_coord[2])
    # Pre-allocating vectors.
    p_e2 = Vector{Float32}(undef, 2)
    p_e3 = Vector{Float32}(undef, 2)
    p_t = Vector{Float32}(undef, 2)
    test = Vector{Float32}(undef, 2)
    for i in 1:n_tot
        # Generating a random 2D coordinate within the pre-defined ranges.
        y = rand(y_range)
        z = rand(z_range)
        test[1] = y
        test[2] = z
        for j in 1:n_faces
            triangle = p_triangles[j]
            # In barycentric coordinates, any point in a triangle can be expressed as (1-u-v) * V1 + u * V2 + v * V3 where u, v ≥ 0 and u + v ≤ 1.
            # Calculating p_e2 = V2 - V1, p_e3 = V3 - V1, p_t = test - V1, p_det = det(p_e2, p_e3) required for the algorithm.
            @. p_e2 = triangle[2] - triangle[1]
            @. p_e3 = triangle[3] - triangle[1]
            @. p_t = test - triangle[1]
            p_det = (p_e2[1] * p_e3[2]) - (p_e2[2] * p_e3[1])
            # Analytically calculating u and v.
            u = (1 / p_det) * ((p_t[1] * p_e3[2]) - (p_t[2] * p_e3[1]))
            v = (1 / p_det) * ((p_e2[1] * p_t[2]) - (p_e2[2] * p_t[1]))
            # Checking if the point lies within the triangle.
            if v ≥ 0 && u ≥ 0 && (u + v) ≤ 1
                n_in += 1
                break
            end
        end
    end
    area = (n_in / n_tot) * rec_area
    return area
end


# Defining a function to correct the measured data by taking into consideration the neutron flux it received.

"""
Corrects the measured data by taking into consideration the neutron flux the sample received considering the sample angle, ψ, of the .nxspe file.
Accomplishes this by dividing by the area of the sample projected onto the y-z plane.

Parameters
----------
s_data (n_bins x n_detectors sparse matrix with float elements): Neutron signal measured at different detectors for different energy bins.
r_vertices (vector with 3-vector elements with float elements): Vertices of sample (pre-rotated as needed for this .nxspe file), in units of .stl file.
indices (n_faces vector of 3-vectors with integer elements): Indices of the vertices that make up each face.
n_faces (integer): Number of faces.
n_tot (integer): Total number of random coordinates used in MC calculation of area.

Returns
-------
c_data (n_bins x n_detectors matrix with float elements): Corrected neutron signal measured at different detectors for different energy bins.
"""
function flux_corr(
    s_data :: SparseMatrixCSC{Float32, Int64}, 
    r_vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}, 
    n_faces :: Integer, 
    n_tot :: Integer
    ) :: SparseMatrixCSC{Float32, Int64}
    # Projecting the sample onto the y-z plane (the plane perpendicular to the neutron beam).
    # Grouping these projected vertices by the triangles they create.
    p_vertices, p_triangles = project(r_vertices, indices, n_faces)
    # Finding the maximum and minimum value of each projected coordinate.
    min_coord, max_coord, rec_area = aabb_2d(p_vertices)
    # Calculating the area of the sample projection.
    area = area_calc(min_coord, max_coord, p_triangles, n_tot, n_faces, rec_area)
    # Scaling each signal via the projection area.
    c_data = s_data / area
    return c_data
end


end