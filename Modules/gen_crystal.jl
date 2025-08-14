# This module contains the key functions for the attenuation calculation for a general crystal morphology.
module gen_crystal


using StaticArrays
using LinearAlgebra
using LoopVectorization


# Defining the function that calculates the attenuation coefficent for the inputted energy.


"""
Determines the attenuation coefficent (μ) for the inputted energy based on the absorption of the sample at a known, reference energy.

Parameters
----------
μ_ref (float): Attenuation coefficent at the reference energy, in cm^-1.
en_ref (float): Reference energy, in meV.
en (float): Energy in meV.

Returns
-------
μ (float): Attenuation coefficient in cm^-1.
"""
function μ_calc(μ_ref :: Float32, en_ref :: Float32, en :: Float32) :: Float32
    return μ_ref * sqrt(en_ref / en)
end


# Defining a function to output the net path length from a series of distances to intersections.


"""
Calculates the net path length through a general sample from the ordered collection of distinct distances between the origin and the intersection points.

Parameters
----------
path_lengths (odd-length vector with float elements): Ordered vector of non-duplicate distances, in units of the .stl file.
n_int (integer): Number of intersections.

Returns
-------
path_length (float): Net path length of neutron in sample, in units of the .stl file.
"""
function net_len_calc(path_lengths :: Vector{Float32}, n_int :: Integer) :: Float32
    # Calculating the path length via a recurrence relation.
    n_rr = Int((n_int - 1) / 2)
    path_length = path_lengths[1]
    for i in 1:n_rr
        path_length += path_lengths[2*i + 1] - path_lengths[2*i]
    end
    return path_length
end


# Defining the function to determine the net length of the paths the neutrons take within the sample.


"""
Calculates the net distance the neutron travels inside the sample. Accomplishes this by considering ray-triangle intersections.
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
n_faces (integer): Number of faces.

Returns
-------
path_length (float): Net distance travelled by the neutron in the sample, in units of the .stl file. nothing is outputted if there is no intersection.
"""
function len_calc(
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
    ) :: Union{Float32, Nothing}
    # Emptying the pre-allocated path length stores.
    empty!(λs)
    empty!(path_lengths)
    # Iterating through all faces.
    @inbounds for j in 1:n_faces
        # If det = p.e2 = (d x e3).e2 = 0, the path is parallel to the triangular face, so it can never intersect it.
        # As we are working with multiple crystals, no culling of back- or front-facing triangles can be done.
        det = dets[j]
        if abs(det) > 1f-6
            # Pre-computing the inverse determinant.
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
                # Only accepting positive λ as this indicates paths moving in positive direction of d.
                if λ > 0
                    # The path length is simply λ as the direction vector is normalised.
                    push!(λs, λ)
                end
            end
        end
    end
    if isempty(λs)
        # Returning nothing if no path is intersected.
        # Assuming the origin is within the sample, then this may occur due to floating point precision errors.
        # ie u+v = 1.00000001 > 1.
        return nothing
    end
    # Ordering the path lengths.
    sort!(λs)
    # Filling the array with non-duplicate lengths in order.
    # Duplicate path lengths arise from paths near a vertex between faces.
    push!(path_lengths, λs[1])
    for i in λs
        if abs(i - last(path_lengths)) > 1f-4
            push!(path_lengths, i)
        end
    end
    # Finding the number of intersections.
    n_int = length(path_lengths)
    # An odd number of intersections is expected as we start inside the sample.
    # This assumes no tangential intersections.
    if isodd(n_int)
        # Calculating the path length via a recurrence relation.
        path_length = net_len_calc(path_lengths, n_int)
        return path_length
    else
        # Returning nothing if there is an even number of intersections.
        # This suggests an error potentially from a sample outside the crystals or tangential intersection.
        return nothing
    end
end


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
n_faces (integer): Number of faces.

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


# Defining the function that will calculate the attenuation factor given a certain energy bin and wavevector.


"""
Calculates the attenuation factor given a set initial and final energy and wavevector.

Parameters
----------
df (3-vector with float elements): Normalised post-scattering neutron direction vector, in Angstrom^-1.
en_f (float): Post-scattering energy of neutron, in meV.
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
len_i (n_mc-vector of float elements): Pre-scattering path length of neutron, in units of .stl file.
p_f (n_faces-vector of 3-vectors with float elements): Pre-allocated vector.
det_f (n_faces-vector with float elements): Pre-allocated vector.
n_faces (integer): Number of faces.
n_mc (integer): Number of MC sample points
μ_ref (float): Attenuation coefficent at reference energy, in cm^-1.
en_ref (float): Reference energy, in meV.
μi (float): Pre-scattering attenuation coefficent, in cm^-1.

Returns
-------
atten_calc (float): Attenuation factor.
acc_pts (integer): Number of MC sample points used in calculation.
"""
function atten_calc(
    df :: SVector{3, Float32}, 
    en_f :: Float32, 
    v1s :: Vector{SVector{3, Float32}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    coords :: Vector{SVector{3, Float32}}, 
    len_i :: Vector{Float32},
    p_f :: Vector{SVector{3, Float32}},
    det_f :: Vector{Float32}, 
    λs :: Vector{Float32}, 
    path_lengths :: Vector{Float32}, 
    n_faces :: Integer, 
    n_mc :: Integer, 
    μ_ref :: Float32, 
    en_ref :: Float32, 
    μi :: Float32
    ) :: Tuple{Float32, Integer}
    # Calculating the absorption cross section after the neutron scatters.
    μf = μ_calc(μ_ref, en_ref, en_f)
    # Setting up the Moller-Trumbore algorithm for ray-triangle intersections.
    pdet_calc!(df, e2s, e3s, p_f, det_f, n_faces)
    # Tallying the number of accepted MC sample points where a path length could be calculated.
    acc_pts = 0
    atten = 0
    @inbounds for i in 1:n_mc
        # Calculating the path length, len_f, at this sample point.
        len_f = len_calc(e2s, e3s, df, p_f, det_f, coords[i], v1s, λs, path_lengths, n_faces)
        if typeof(len_f) == Float32
            # Adding the attenuation factor contribution from this sample point to A.
            atten += exp(-μi * len_i[i]) * exp(-μf * len_f)
            acc_pts += 1
        end 
    end
    # Dividing by the total number of contributing sample points.
    @assert acc_pts != 0 "The path length could not be calculated at any sample point (for a specific final wavevector). Try increasing n_abs."
    atten = atten / (acc_pts)
    return atten, acc_pts
end


# Defining the function that calculates the grid of attenuation factors.


"""
Calculates the attenuation factor for every non-zero, non-NaN, signal and stores in a grid of detector against energy bin.

Parameters
----------
data (n_bins x n_detectors matrix with float elements): Neutron signal measured at different detectors for different energy bins.
kx (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in x direction, in Angstrom^-1.
ky (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in y direction, in Angstrom^-1.
kz (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in z direction, in Angstrom^-1.
en_i (float): Pre-scattering neutron energy, in meV.
ef_bins (n_bins-vector with float elements): Post-scattering neutron energy of each bin, in meV.
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
len_i (n_mc-vector with float elements): Pre-scattering path length of neutron, in units of .stl file.
n_bins (integer): Number of bins.
n_detectors (integer): Number of detectors.
n_faces (integer): Number of faces.
n_mc (integer): Number of MC sample points.
μ_ref (float): Attenuation coefficent at reference energy, in cm^-1.
en_ref (float): Reference energy, in meV.
μi (float): Pre-scattering attenuation coefficent, in cm^-1.

Returns
-------
atten_grid (n_bins x n_detectors matrix with float elements): Attenuation factor for different detectors and energy bins.
acc_rate (float): Acceptance rate of MC sample points.
"""
function a_grid_calc(
    data :: Matrix{Float32}, 
    kx :: AbstractMatrix{Float32}, 
    ky :: AbstractMatrix{Float32}, 
    kz :: AbstractMatrix{Float32}, 
    ef_bins :: Vector{Float32}, 
    v1s :: Vector{SVector{3, Float32}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    coords :: Vector{SVector{3, Float32}}, 
    len_i :: Vector{Float32}, 
    n_bins :: Integer, 
    n_detectors :: Integer, 
    n_faces :: Integer, 
    n_mc :: Integer, 
    μ_ref :: Float32, 
    en_ref :: Float32, 
    μi :: Float32
    ) :: Tuple{Matrix{Float32}, Float32}
    atten_grid = zeros(Float32, n_bins, n_detectors)
    # Determining the locations in which the signal is either NaN or 0 as we don't want to calculate atten there.
    idx = findall(.~((data .== 0) .| (isnan.(data))))
    # Pre-allocating the vectors needed for the MT algorithm.
    p_f = Vector{SVector{3, Float32}}(undef, n_faces)
    det_f = Vector{Float32}(undef, n_faces)
    # Pre-allocating the path length stores.
    # The length of these vectors should be the maximum expected number of intersections.
    λs = Vector{Float32}(undef, 10)
    path_lengths = Vector{Float32}(undef, 10)
    # Setting the proportion of MC sample points used for this file.
    acc_rate = 0
    # Skipping checks on array lengths using @inbounds.
    @inbounds for I in idx
        kf = SVector{3, Float32}(kx[I], ky[I], kz[I])
        # Normalising the post-scattering direction vector.
        df = kf / norm(kf)
        atten_grid[I], acc_pts = atten_calc(df, ef_bins[I[1]], v1s, e2s, e3s, coords, len_i, p_f, det_f, λs, path_lengths, n_faces, n_mc, μ_ref, en_ref, μi)
        acc_rate += acc_pts
    end
    # Calculating this acceptance rate.
    acc_rate = acc_rate / (length(idx) * n_mc)
    return atten_grid, acc_rate
end


end