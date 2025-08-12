# This module stores functions that aid in determining random coordinates within the sample.
module sampling

using Distributions
using StaticArrays
using LinearAlgebra
using GeometryBasics


# Defining a function to pre-calculate p = d x e3 and determinant = p.e2 = (d x e3).e2 required for the MT Algorithm.


"""
Calculates p = d x e3 and determinant = p.e2 = (d x e3).e2 required for the MT Algorithm.

Parameters
----------
d (3-vector with float elements): Normalised direction vector.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
ps (n_faces-vector of 3-vectors with float elements): Pre-allocated vector.
dets (n_faces-vector with float elements): Pre-allocated vector.
n_faces (integer): Number of faces in .stl file.

Returns
-------
ps (n_faces-vector of 3-vectors with float elements): Array containing p = d x e3 for each face.
dets (n_faces-vector with float elements): Array containing det = p.e2 = (d x e3).e2 for each face.
"""
function pdet_calc!(
    d :: SVector{3, Float32}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}},
    ps :: Vector{SVector{3, Float32}},
    dets :: Vector{Float32}, 
    n_faces :: Integer
    ) :: Tuple{Vector{SVector{3, Float32}}, Vector{Float32}}
    # @inbounds is used to remove checks on the index i as we are sure of the sizes of our arrays.
    # @simd is used to vectorize and speed up the loop.
    @inbounds @simd for i in 1:n_faces
        # Calculating cross products, p = d x e3, for the direction vector, d, and for each face.
        ps[i] = cross(d, e3s[i])
        # Calculating the determinant = p.e2 = (d x e3).e2 for the direction vector, d, and for each face.
        dets[i] = dot(ps[i], e2s[i])
    end
    return ps, dets
end


# Defining the function to determine the number of times the neutron intersects the sample.


"""
Calculates how many faces a neutron intersects given its direction vector. Accomplishes this by considering ray-triangle intersections.
Iterates through each face to determine which one is intersected.
Exploits the method described in 'Fast, Minimum Storage Ray-Triangle Intersection' by Moller and Trumbore.

Parameters
----------
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
d (3-vector with float elements): Normalised direction vector.
ps (n_faces-vector of 3-vectors with float elements): Array containing p = d x e3 for each face.
dets (n_faces-vector with float elements): Array containing det = p.e2 = (d x e3).e2 for each face.
origin (3-vector with float elements): Coordinates of scattering sites.
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
λs (vector with float elements): Empty vector that will store distances between origin and intersection points, in units of the .stl file.
path_lengths (vector with float elements): Empty vector that will store non-duplicate distances, in units of the .stl file.
n_faces (integer): Number of faces of .stl file.

Returns
-------
n_int (integer): Number of intersections.
"""
function int_calc(
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    d :: SVector{3, Float32}, 
    ps :: Vector{SVector{3, Float32}}, 
    dets :: Vector{Float32}, 
    origin :: SVector{3, Float32}, 
    v1s :: Vector{SVector{3, Float32}},
    λs :: Vector{Float32}, 
    path_lengths :: Vector{Float32}, 
    n_faces :: Integer
    ) :: Integer
    # Emptying the pre-allocated path length stores.
    empty!(λs)
    empty!(path_lengths)
    # Iterating through all faces.
    @inbounds for j in 1:n_faces
        # If det = p.e2 = (d x e3).e2 = 0, the path is parallel to the triangular face, so it can never intersect it.
        # As we are working with multiple crystals, no culling of back- or front-facing triangles can be done.
        det = dets[j]
        if abs(det) > 1e-6
            inv_det = 1 / det
            # Calculating t = origin - V1 and q = t x e2 required for the MT algorithm.
            t = origin - v1s[j]
            q = cross(t, e2s[j])
            # Calculating the barycentric coordinates, (u,v), of the intersection.
            u = inv_det * (dot(ps[j], t))
            v = inv_det * (dot(q, d))
            # Determining whether the intersection point lies within the triangle.
            if v ≥ 0 && u ≥ 0 && (u + v) ≤ 1
                # Neutron's path described by r(λ) = origin + λd.
                λ = inv_det * dot(q, e3s[j])
                # Accepting only positive λ corresponding to forward direction.
                if λ > 0
                    # The path length is simply λ as the direction vector is normalised.
                    push!(λs, λ)
                end
            end
        end
    end
    if isempty(λs)
        # Returning 0 if no surface is intersected.
        return 0
    end
    # Ordering the path lengths.
    sort!(λs)
    # Filling the array with non-duplicate lengths in order.
    # Duplicate path lengths arise from paths near a vertex between faces.
    push!(path_lengths, λs[1])
    for i in λs
        if abs(i - last(path_lengths)) > 1e-6
            push!(path_lengths, i)
        end
    end
    # Finding the number of intersections.
    n_int = length(path_lengths)
    return n_int
end


# Defining a function to calculate V1, e2 = V2 - V1 and e3 = V3 - V1 for each face.


"""
Calculates V1, e2 = V2 - V1 and e3 = V3 - V1 for each face from the .stl file's vertices and indices.

Parameters
----------
vertices (n_vert-vector of 3-vectors with float elements): Vertices, in units of .stl file.
indices (n_faces vector of 3-vectors with integer elements): Indices of the vertices that make up each face.

Returns
-------
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
"""
function ve_calc(vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}
    ) :: Tuple{Vector{SVector{3, Float32}}, Vector{SVector{3, Float32}}, Vector{SVector{3, Float32}}}
    # Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
    # The vertices of the triangular faces are labelled V1, V2, V3.
    v1s = vertices[getindex.(indices, 1)]
    e2s = vertices[getindex.(indices, 2)] - vertices[getindex.(indices, 1)]
    e3s = vertices[getindex.(indices, 3)] - vertices[getindex.(indices, 1)]
    # Converting the 3-vectors within e2s and e3s to static arrays.
    v1s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in v1s]
    e2s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e2s]
    e3s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e3s]
    return v1s, e2s, e3s
end


# Defining a function that will calculate the extrema of the coordinates to build the axis-aligned bounding box (AABB).


"""
Builds the axis-aligned bounding box (AABB) which surrounds the sample. Required for MC sampling.

Parameters
----------
vertices (n_vert-vector of 3-vectors with float elements): Vertices, in units of .stl file.

Returns
-------
x_range (uniform distribution of floats): Range of x, in units of .stl file.
y_range (uniform distribution of floats): Range of y, in units of .stl file.
z_range (uniform distribution of floats): Range of z, in units of .stl file.

"""
function aabb_3d(vertices :: Vector{Point{3, Float32}}) :: Tuple{Uniform{Float32}, Uniform{Float32}, Uniform{Float32}}
    # Finding the maximum and minimum value of each coordinate.
    max_coord = [maximum(getindex.(vertices, 1)), maximum(getindex.(vertices, 2)), maximum(getindex.(vertices, 3))]
    min_coord = [minimum(getindex.(vertices, 1)), minimum(getindex.(vertices, 2)), minimum(getindex.(vertices, 3))]
    # Pre-calculating uniform distribution describing the volume our crystal(s) lies in.
    x_range = Uniform(min_coord[1], max_coord[1])
    y_range = Uniform(min_coord[2], max_coord[2])
    z_range = Uniform(min_coord[3], max_coord[3])
    return x_range, y_range, z_range
end


# Defining a function to generate the desired number of MC sample points.


"""
Generates the required number of MC sample points.

Parameters
----------
ranges (3-tuple of uniform distributions of floats): Extrema of the AABB, in units of the .stl file.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
coords (n_mc-vector of 3-vectors with float elements): Empty vector of coordinates of sample points.
n_faces (integer): Number of faces of .stl file.
n_mc (integer) : Number of sample points required.

Returns
-------
coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
"""

function sample!(
    ranges :: Tuple{Uniform{Float32}, Uniform{Float32}, Uniform{Float32}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    v1s :: Vector{SVector{3, Float32}}, 
    coords :: Vector{SVector{3, Float32}},
    n_faces :: Integer,
    n_mc :: Integer
    ) :: Vector{SVector{3, Float32}}
    # Pre-allocating the necessary vectors.
    λs = Vector{Float32}(undef, 10)
    path_lengths = Vector{Float32}(undef, 10)
    p = Vector{SVector{3, Float32}}(undef, n_faces)
    det = Vector{Float32}(undef, n_faces)
    # Sending a dummy neutron along the x direction, starting at this test coordinate.
    d = SVector{3, Float32}(-1, 0, 0)
    # Calculating p and det required for the MT algorithm.
    pdet_calc!(d, e2s, e3s, p, det, n_faces)
    # Tallying the number of accepted coordinates.
    n_acc = 0
    # Continuing this sample generation until there are n_mc coordinates inside the crytal(s).
    while n_acc < n_mc
        # Generating a random coordinate within the pre-defined sample range.
        x = rand(ranges[1])
        y = rand(ranges[2])
        z = rand(ranges[3])
        test = SVector{3, Float32}(x, y, z)
        # Calculating the number of intersections this theoretical neutron makes with the sample surfaces.
        n_int = int_calc(e2s, e3s, d, p, det, test, v1s, λs, path_lengths, n_faces)
        # If it makes an even number of intersections, it is outside a sample.
        # Odd number of intersections means it began in a sample.
        if isodd(n_int)
            # Accepting this test coordinate.
            coords[n_acc + 1] = test
            n_acc += 1
        end
    end
    return coords
end


end

