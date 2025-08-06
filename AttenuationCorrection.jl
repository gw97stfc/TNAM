using HDF5
using Unitful
import PhysicalConstants.CODATA2018: m_n, e, ħ
using FileIO
using GeometryBasics
using LinearAlgebra
using BenchmarkTools
using MeshIO
using SparseArrays
using StaticArrays
using LoopVectorization
using Distributions


# Extracting the contents of the .nxspe file.


en_i, azi, pol, data, Δen = h5open("test_nxspe_data/LET104215_3.7meV_1to1.nxspe", "r") do f
    # Initial energy in meV.
    en_i = read(f["ws_out/NXSPE_info/fixed_energy"])[1]
    # Azimuthal angles in degrees.
    azi = read(f["ws_out/data/azimuthal"])
    # Polar angles in degrees.
    pol = read(f["ws_out/data/polar"])
    # Measured signal for each energy bin and each detector.
    data = read(f["ws_out/data/data"])
    # Neutron energy changes, in meV.
    Δen = read(f["ws_out/data/energy"])
    # Converting the elements to Float32 since the vertices of the .stl file are also Float32.
    return Float32(en_i), Float32.(azi), Float32.(pol), Float32.(data), Float32.(Δen)
end


# Defining a function to calculate the magnitude of the wavevector, in Angstrom^-1, of the neutron from its energy.


"""
Calculates the magnitude of the wavevector, in Angstrom^-1, from an energy, in meV.

Parameters
----------
en (float): Energy, in meV.

Returns
-------
mag_k (float): Magnitude of wavevector, in Angstrom^-1.
"""
function magk_calc(en :: Float32) :: Float32
    mag_k = sqrt(2 * m_n * en * e * (1e-3)) / (ħ * (1e10))
    return ustrip(mag_k)
end


# Calculating ki, n_bins, n_detectors from the extracted content of the .nxspe file.


# Finding the initial wavevector, in Angstrom^-1, from the initial energy.
ki = zeros(3)
ki[1] = magk_calc(en_i)
# Converting ki to a static array.
ki = SVector{3, Float32}(ki)
# Extracting the number of energy bins and number of detectors.
const n_bins = length(Δen) - 1
const n_detectors = length(azi)


# Defining a function to calculate the final energy of neutrons from each bin, based on Ei and the energy change.


"""
Calculates the final neutron energy, in meV, for each energy bin.

Parameters
----------
en_i (float): Pre-scattering neutron energy, in meV.
Δen ((n_bins + 1)-vector with float elements): Neutron energy changes, in meV.

Returns
-------
ef_bins (n_bins-vector with float elements): Post-scattering neuton energy for each bin, in meV.
"""
function ef_calc(en_i :: Float32, Δen :: AbstractVector{Float32}) :: SVector{n_bins, Float32}
    ef_bins = zeros(n_bins)
    for i in 1:n_bins
        # Finding the bin centres by averaging the energies on each end of the bin.
        # Determining the final neutron energy, Ef = Ei - Δen based on which energy bin we are considering.
        ef_bins[i] = en_i - ((Δen[i] + Δen[i+1]) / 2)
    end
    # Returning the final energies in a static array.
    return SVector{n_bins, Float32}(ef_bins)
end


# Calculating the final neutron energy, in meV, for each energy bin.


ef_bins = ef_calc(en_i, Δen)


# Defining the function that calculates the final neutron wavevector from the detector angles and final neutron energies.


"""
Calculates the components of the post-scattering neutron wavevector, in Angstrom^-1, for each detector and for each energy bin.

Parameters
----------
ef_bins (n_bins-vector with float elements): Post-scattering neutron energy of each bin, in meV.
pol (n_detectors-vector with float elements): Polar angles of each detector, in degrees.
azi (n_detectors-vector with float elements): Azimuthal angles of each detector, in degrees.

Returns
-------
kx (n_bins x n_detectors matrix of floats): Post-scattering neutron wavevector components in x direction, in Angstrom^-1.
ky (n_bins x n_detectors matrix of floats): Post-scattering neutron wavevector components in y direction, in Angstrom^-1.
kz (n_bins x n_detectors matrix of floats): Post-scattering neutron wavevector components in z direction, in Angstrom^-1.
"""
function kf_calc(ef_bins :: SVector{n_bins, Float32}, pol :: Vector{Float32}, azi :: Vector{Float32})
    mag_kf = magk_calc.(ef_bins)
    # Reshaping the arrays to allow for broadcasting.
    pol_col = reshape(pol, :, 1)
    azi_col = reshape(azi, :, 1)
    mag_kf_row = reshape(mag_kf, 1, :)
    # Pre-calculating the azi and pol arrays in radians.
    pol_col_rad = deg2rad.(pol_col)
    azi_col_rad = deg2rad.(azi_col)
    # Determing the components of the final wavevector using broadcasting.
    kx = mag_kf_row .* (sin.(pol_col_rad) .* cos.(azi_col_rad))
    ky = mag_kf_row .* (sin.(pol_col_rad) .* sin.(azi_col_rad))
    kz = mag_kf_row .* cos.(pol_col_rad)
    # Reshaping these final wavevector grids to align with the grid of data.
    kx = transpose(kx)
    ky = transpose(ky)
    kz = transpose(kz)
    return kx, ky, kz
end


# Calculating the final wavevector, in Angstrom^-1, for each energy bin and each detector.


kx, ky, kz = kf_calc(ef_bins, pol, azi)


# Retrieving the vertices and indices of the triangular mesh of the sample surface from the .stl file.


stl = load("crystal.stl")
vertices = GeometryBasics.coordinates(stl)
indices = GeometryBasics.faces(stl)
# Extracting the number of triangular faces used in the mesh.
const n_faces = length(indices)


# Setting the desired number of MC sample points and creating vector for coordinates.


const n_mc = 100
mc_coords = Vector{SVector{3, Float32}}(undef, n_mc)


# Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, and V1s specifically in preparation for the Moller-Trumbore Algorithm.
# The vertices of the triangular faces are labelled V1, V2, V3.

v1s = vertices[getindex.(indices, 1)]
e2s = vertices[getindex.(indices, 2)] - vertices[getindex.(indices, 1)]
e3s = vertices[getindex.(indices, 3)] - vertices[getindex.(indices, 1)]
# Converting the 3-vectors within e2s and e3s to static arrays.
v1s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in v1s]
e2s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e2s]
e3s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e3s]


# Setting the (estimated) parameters of the sample.


# The number density of the sample in cm^-3.
const n = Float32(1e23)
# The reference absorption cross section at 25.3 meV in cm^2.
const axs_ref = Float32(1e-23)
const en_ref = Float32(25.3)
# The crystal morphology affects the speed of program.
# 'complex' means a sample with a concave surface or multiple crystals.
const complex = false


# Defining the function that calculates the absorption cross sections for the inputted energy.


"""
Determines the absorption cross section (axs) for the inputted energy based on the absorption of the sample at a known, reference energy.

Parameters
----------
en (float): Energy in meV.

Returns
-------
axs (float): Absorption cross section in cm^2.
"""
function axs_calc(en :: Float32) :: Float32
    return axs_ref * sqrt(en_ref / en)
end


# Calculating the pre-scattering absorption cross section.


const axsi = axs_calc(en_i)


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
function net_len_calc!(path_lengths :: Vector{Float32}, n_int :: Integer) :: Float32
    # Calculating the path length via a recurrence relation.
    n_rr = Int((n_int - 1) / 2)
    path_length = path_lengths[1]
    for i in 1:n_rr
        path_length += path_lengths[2*i + 1] - path_lengths[2*i]
    end
    return path_length
end


# Defining the function to determine the net length of the paths the neutrons take within the sample.
# For a general sample.


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

Returns
-------
path_length (float): Net distance travelled by the neutron in the sample, in units of the .stl file. nothing is outputted if there is no intersection.
"""
function gen_len_calc(
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    d :: SVector{3, Float32}, 
    ps :: Vector{SVector{3, Float32}}, 
    dets :: Vector{Float32}, 
    origin :: SVector{3, Float32}, 
    v1s :: Vector{SVector{3, Float32}},
    λs :: Vector{Float32}, 
    path_lengths :: Vector{Float32}
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
        # Assuming the origin is within the sample, then this only occurs due to floating point precision errors.
        # ie u+v = 1.00000001 > 1.
        return nothing
    end
    # Ordering the path lengths.
    sort!(λs)
    # Filling the array with non-duplicate lengths in order.
    # Duplicate path lengths arise from paths near a vertex between faces.
    push!(path_lengths, λs[1])
    for i in λs
        if abs(i - last(path_lengths)) > 1f-6
            push!(path_lengths, i)
        end
    end
    # Finding the number of intersections.
    n_int = length(path_lengths)
    # An odd number of intersections is expected as we start inside the sample.
    # This assumes no tangential intersections.
    if isodd(n_int)
        # Calculating the path length via a recurrence relation.
        path_length = net_len_calc!(path_lengths, n_int)
        return path_length
    else
        # Returning nothing if there is an even number of intersections.
        # This suggests an error potentially from a sample outside the crystals or tangential intersection.
        return nothing
    end
end


# Defining the function to determine the length of the paths the neutrons take within the sample.
# For a single, convex crystal.


"""
Calculates the distance between a given point in the sample (origin) and a triangular face of the mesh that describes the surface. 
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

Returns
-------
path_length (float): Distance between origin and the surface the neutron path intersects, in units of the .stl file. nothing is outputted if there is no intersection.
"""
function sin_len_calc(
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    d :: SVector{3, Float32}, 
    ps :: Vector{SVector{3, Float32}}, 
    dets :: Vector{Float32}, 
    origin :: SVector{3, Float32}, 
    v1s :: Vector{SVector{3, Float32}}
    )  :: Union{Float32, Nothing}
    # Iterating through all faces.
    @inbounds for j in 1:n_faces
        # If det = p.e2 = (d x e3).e2 = 0, the path is parallel to the triangular face, so it can never intersect it.
        # Keeping only negative determinants, culling front-facing triangles as we are inside the mesh.
        det = dets[j]
        if det < -1f-6
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
                    return Float32(λ)
                end
            end
        end
    end
    # Returning nothing if no path is intersected.
    # Assuming the origin is within the sample, then this only occurs due to floating point precision errors.
    # ie u+v = 1.00000001 > 1.
    return nothing
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
    dets :: Vector{Float32} 
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
    path_lengths :: Vector{Float32}
    ) :: Integer
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
                # Only accepting positive λ corresponding to forward direction.
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
        if abs(i - last(path_lengths)) > 1f-6
            push!(path_lengths, i)
        end
    end
    # Finding the number of intersections.
    n_int = length(path_lengths)
    return n_int
end


# Defining a function to generate the desired number of MC sample points.


"""
Generates the required number of MC sample points.

Parameters
----------
min_coord (3-vector with float elements): Minimum value of each x, y and z coordinate, in units of .stl file.
max_coord (3-vector with float elements): Maximum value of each x, y and z coordinate, in units of .stl file.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
v1s (n_faces-vector of 3-vectors with float elements): First vertex of each face, V1.
coords (n_mc-vector of 3-vectors with float elements): Empty vector of coordinates of sample points.

Returns
-------
coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
"""

function sampling!(
    min_coord :: Vector{Float32}, 
    max_coord :: Vector{Float32}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    v1s :: Vector{SVector{3, Float32}}, 
    coords :: Vector{SVector{3, Float32}}
    ) :: Vector{SVector{3, Float32}}
    # Pre-allocating the necessary vectors.
    λs = Vector{Float32}(undef, 10)
    path_lengths = Vector{Float32}(undef, 10)
    p = Vector{SVector{3, Float32}}(undef, n_faces)
    det = Vector{Float32}(undef, n_faces)
    # Pre-calculating uniform distribution describing the volume our crystal(s) lies in.
    x_range = Uniform(min_coord[1], max_coord[1])
    y_range = Uniform(min_coord[2], max_coord[2])
    z_range = Uniform(min_coord[3], max_coord[3])
    # Sending a dummy neutron along the x direction, starting at this test coordinate.
    d = SVector{3, Float32}(-1, 0, 0)
    # Calculating p and det required for the MT algorithm.
    pdet_calc!(d, e2s, e3s, p, det)
    # Tallying the number of accepted coordinates.
    n_acc = 0
    # Continuing this sample generation until there are n_mc coordinates inside the crytal(s).
    while n_acc < n_mc
        # Generating a random coordinate within the pre-defined sample range.
        x = rand(x_range)
        y = rand(y_range)
        z = rand(z_range)
        test = SVector{3, Float32}(x, y, z)
        # Calculating the number of intersections this theoretical neutron makes with the sample surfaces.
        n_int = int_calc(e2s, e3s, d, p, det, test, v1s, λs, path_lengths)
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


# Determining the coordinates of the points within our sample used for the Monte Carlo approximation of the volume integral.


# Finding the maximum and minimum value of each coordinate.
max_coord = [maximum(getindex.(vertices, 1)), maximum(getindex.(vertices, 2)), maximum(getindex.(vertices, 3))]
min_coord = [minimum(getindex.(vertices, 1)), minimum(getindex.(vertices, 2)), minimum(getindex.(vertices, 3))]
# Filling this vector with the randomly generated sample points.
sampling!(min_coord, max_coord, e2s, e3s, v1s, mc_coords)


# The pre-scattering neutron path lengths are dependent only on the MC coordinates.
# They can, therefore, be calculated and stored.


# Normalising the pre-scattering direction vector.
di = -ki / norm(-ki)
# Pre-allocating the path length stores.
# The length of these vectors should be the maximum expected number of intersections.
λs = Vector{Float32}(undef, 10)
path_lengths = Vector{Float32}(undef, 10)
# Pre-allocating these vectors.
p_i = Vector{SVector{3, Float32}}(undef, n_faces)
det_i = Vector{Float32}(undef, n_faces)
# Calculating p and det required for the MT algorithm.
pdet_calc!(di, e2s, e3s, p_i, det_i)
len_i = Float32.(zeros(n_mc))
# Iterating through the Monte Carlo sample points to find the pre-scattering path length of each.
if complex
    for i in 1:n_mc
        len_i[i] = gen_len_calc(e2s, e3s, di, p_i, det_i, mc_coords[i], v1s, λs, path_lengths)
    end
else
    for i in 1:n_mc
        len_i[i] = sin_len_calc(e2s, e3s, di, p_i, det_i, mc_coords[i], v1s)
    end
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

Returns
-------
atten_calc (float): Attenuation factor.
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
    path_lengths :: Vector{Float32}
    ) :: Float32
    # Calculating the absorption cross section after the neutron scatters.
    axsf = axs_calc(en_f)
    # Setting up the Moller-Trumbore algorithm for ray-triangle intersections.
    pdet_calc!(df, e2s, e3s, p_f, det_f)
    # Tallying the number of accepted MC sample points where a path length could be calculated.
    acc_pts = 0
    atten = 0
    if complex
        @inbounds for i in 1:n_mc
            # Calculating the path length, len_f, at this sample point.
            len_f = gen_len_calc(e2s, e3s, df, p_f, det_f, coords[i], v1s, λs, path_lengths)
            if typeof(len_f) == Float32
                # Adding the attenuation factor contribution from this sample point to A.
                atten += exp(-n * axsi * len_i[i]) * exp(-n * axsf * len_f)
                acc_pts += 1
            end
        end
    else
        @inbounds for i in 1:n_mc
            # Calculating the path length, len_f, at this sample point.
            len_f = sin_len_calc(e2s, e3s, df, p_f, det_f, coords[i], v1s)
            if typeof(len_f) == Float32
                # Adding the attenuation factor contribution from this sample point to A.
                atten += exp(-n * axsi * len_i[i]) * exp(-n * axsf * len_f)
                acc_pts += 1
            end
        end
    end
    # Dividing by the total number of contributing sample points.
    if acc_pts == 0
        error("The path length could not be calculated for any sample point. Are they all within the sample?")
    else
        atten = atten / (acc_pts)
        return atten
    end
end


# Defining the function that calculates the grid of attenuation factors.


"""
Calculates the attenuation factor for every non-zero, non-NaN, signal and stores in a grid of detector against energy bin.

Parameters
----------
s_data (n_bins x n_detectors sparse matrix with float elements): Neutron signal measured at different detectors for different energy bins.
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

Returns
-------
atten_grid (n_bins x n_detectors matrix with float elements): Attenuation factor for different detectors and energy bins.
"""
function a_grid_calc(
    s_data :: SparseMatrixCSC{Float32, Int64}, 
    kx :: AbstractMatrix{Float32}, 
    ky :: AbstractMatrix{Float32}, 
    kz :: AbstractMatrix{Float32}, 
    ef_bins :: SVector{n_bins, Float32}, 
    v1s :: Vector{SVector{3, Float32}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    coords :: Vector{SVector{3, Float32}}, 
    len_i :: Vector{Float32}
    ) :: Matrix{Float32}
    atten_grid = zeros(Float32, n_bins, n_detectors)
    # Determining the locations in which the signal is either NaN or 0 as we don't want to calculate atten there.
    idx = findnz(s_data)
    # Pre-allocating the vectors needed for the MT algorithm.
    p_f = Vector{SVector{3, Float32}}(undef, n_faces)
    det_f = Vector{Float32}(undef, n_faces)
    # Pre-allocating the path length stores.
    # The length of these vectors should be the maximum expected number of intersections.
    λs = Vector{Float32}(undef, 10)
    path_lengths = Vector{Float32}(undef, 10)
    # Skipping checks on array lengths using @inbounds.
    @inbounds for (i, j) in zip(idx[1], idx[2])
        kf = SVector{3, Float32}(kx[i, j], ky[i, j], kz[i, j])
        # Normalising the post-scattering direction vector.
        df = kf / norm(kf)
        atten_grid[i, j] = atten_calc(df, ef_bins[i], v1s, e2s, e3s, coords, len_i, p_f, det_f, λs, path_lengths)
    end
    return atten_grid
end


# Converting the grid of data and attenuation factors to sparse arrays.


# Replacing every NaN value with zero to allow conversion to sparse matrix.
data_copy = copy(data)
data_copy .= ifelse.(isnan.(data_copy), 0, data)
s_data = sparse(data_copy)


# Testing the time taken to output this grid of attenuation factors.


atten_grid = a_grid_calc(s_data, kx, ky, kz, ef_bins, v1s, e2s, e3s, mc_coords, len_i)
# Converting the attenuation factors to a sparse matrix.
s_atten = sparse(atten_grid)
# Testing the same (known to be non-zero) datapoint.
println("The attenuation factor at this test point was $(s_atten[160, 6])")


# Defining the function to correct the data/signal for absorption.


"""
Corrects the measured data by taking neutron absorption into consideration. The measured signal is divided by the corresponding attenuation factor.

Parameters
----------
s_data (n_bins x n_detectors sparse matrix with float elements): Neutron signal measured at different detectors for different energy bins.
s_atten (n_bins x n_detectors sparse matrix with float elements): Attenuation factor for different detectors and energy bins.

Returns
-------
c_data (n_bins x n_detectors matrix with float elements): Corrected neutron signal at different detectors for different energy bins.
"""
function data_corr(s_data :: SparseMatrixCSC{Float32, Int64}, s_atten :: SparseMatrixCSC{Float32, Int64}) :: Matrix{Float32}
    c_data = zeros(Float32, n_bins, n_detectors)
    # Finding the indices (the energy bin and detector) at which we have a signal.
    idx = findnz(s_data)
    # Correcting each signal by dividing by the corresponding attenuation factor.
    for (i, j, k) in zip(idx[1], idx[2], idx[3])
        c_data[i, j] = k / s_atten[i, j]
    end
    return c_data
end


# Calculating the corrected data/signal by dividing each measured signal by the corresponding attenuation factor.


c_data = data_corr(s_data, s_atten)
# Converting this corrected data into a sparse matrix.
s_c_data = sparse(c_data)
# Testing the same (known to be non-zero) datapoint.
println("The signal at this test point was $(s_data[160,6])")
println("The corrected signal is therefore $(s_c_data[160,6])")
