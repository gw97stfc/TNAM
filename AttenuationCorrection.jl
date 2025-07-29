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
using PyCall
trimesh = pyimport_conda("trimesh", "trimesh")


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


# Calculating and storing the vectors parallel to each face, e2 = V2 - V1 and e3 = V3 - V1, in preparation for hte Moller-Trumbore Algorithm.
# The vertices of the triangular faces are labelled V1, V2, V3.


e2s = vertices[getindex.(indices, 2)] - vertices[getindex.(indices, 1)]
e3s = vertices[getindex.(indices, 3)] - vertices[getindex.(indices, 1)]
# Converting the 3-vectors within e2s and e3s to static arrays.
e2s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e2s]
e3s = [SVector{3, Float32}(vec[1], vec[2] ,vec[3]) for vec in e3s]


# Determining the coordinates of the points within our sample used for the Monte Carlo approximation of the volume integral.
# User will have to run(`$(PyCall.python) -m pip install "trimesh[easy]"`) in their Julia REPL. Write in README or PyProject.toml.


const n_mc_max = 1000
mesh = trimesh.load_mesh("crystal.stl")
mc_coords = Float32.(trimesh.sample.volume_mesh(mesh, n_mc_max))
const n_mc = length(mc_coords[:, 1])



# Setting the (estimated) parameters of the sample.


# The number density of the sample in cm^-3.
const n = Float32(1e23)
# The reference absorption cross section at 25.3 meV in cm^2.
const axs_ref = Float32(1e-23)
const en_ref = Float32(25.3)


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


# Defining the function to determine the length of the paths the neutrons take within the sample.


"""
Calculates the distance between a given point in the sample (origin) and a triangular face of the mesh that describes the surface. 
Iterates through each face to determine which one is intersected.
Exploits the method described in 'Fast, Minimum Storage Ray-Triangle Intersection' by Moller and Trumbore.

Parameters
----------
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
d (3-vector with float elements): Direction vector.
ps (n_faces-vector of 3-vectors with float elements): Array containing p = d x e3 for each face.
dets (n_faces-vector with float elements): Array containing det = p.e2 = (d x e3).e2 for each face.
origin (3-vector with float elements): Coordinates of scattering sites.
vertices (n_faces-vector of 3-vectors with float elements): Vertices of triangular faces.
indices (n_faces-vector of 3-vectors with integer elements): Indices describing which vertices form which triangles.

Returns
-------
path_length (float): Distance between origin and the surface the neutron path intersects, in units of the .stl file.
"""
function len_calc(
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    d :: SVector{3, Float32}, 
    ps :: Vector{SVector{3, Float32}}, 
    dets :: Vector{Float32}, 
    origin :: Vector{Float32}, 
    vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}
    ) :: Float32
    # Iterating through all faces.
    @inbounds for j in 1:n_faces
        # If det = p.e2 = (d x e3).e2 = 0, the path is parallel to the triangular face, so it can never intersect it.
        # Keeping only positive determinants, equivalent to triangular faces in the forward direction.
        det = dets[j]
        if det > 1e-10
            # Calculating t = origin - V1 and q = t x e2 required for the MT algorithm.
            t = origin - vertices[indices[j][1]]
            q = cross(t, e2s[j])
            # Calculating the barycentric coordinates, (u,v), of the intersection.
            u = (1 / det) * (dot(ps[j], t))
            v = (1 / det) * (dot(q, d))
            # Determining whether the intersection point lies within the triangle.
            if v ≥ 0 && u ≥ 0 && (u + v) ≤ 1
                # Neutron's path described by r(λ) = origin + λd.
                λ = (1 / det) * dot(q, e3s[j])
                # Determining the path length based on λ and the magnitude of the inputted direction vector, d.
                path_length = abs(λ) * norm(d)
                return path_length
            end
        end
    end
    error("The neutron does not intersect a face. Is the origin within the sample?")
end


# Defining a function to pre-calculate p = d x e3 and determinant = p.e2 = (d x e3).e2 required for the MT Algorithm.


"""
Calculates p = d x e3 and determinant = p.e2 = (d x e3).e2 required for the MT Algorithm.

Parameters
----------
d (3-vector with float elements): Direction vector.
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


# The pre-scattering neutron path lengths are dependent only on the MC coordinates so can be pre-calculated.


# Pre-allocating these vectors.
p_i = Vector{SVector{3, Float32}}(undef, n_faces)
det_i = Vector{Float32}(undef, n_faces)
# Calculating p and det required for the MT algorithm.
p_i, det_i = pdet_calc!(-ki, e2s, e3s, p_i, det_i)
len_i = Float32.(zeros(n_mc))
# Iterating through the Monte Carlo sample points to find the pre-scattering path length of each.
for i in 1:n_mc
    len_i[i] = len_calc(e2s, e3s, -ki, p_i, det_i, mc_coords[i, :], vertices, indices)
end


# Defining the function that will calculate the attenuation factor given a certain energy bin and wavevector.


"""
Calculates the attenuation factor given a set initial and final energy and wavevector.

Parameters
----------
ki (3-vector with float elements): Pre-scattering wavevector of neutron, in Angstrom^-1.
kf (3-vector with float elements): Post-scattering wavevector of neutron, in Angstrom^-1.
en_f (float): Post-scattering energy of neutron, in meV.
vertices (n_faces-vector of 3-vectors with float elements): Vertices of the triangular faces.
indices (n_faces-vector of 3-vectors with integer elements): Indices describing which vertices correspond to which triangles.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
mc_coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
len_i (n_mc-vector of float elements): Pre-scattering path length of neutron, in units of .stl file.
p_f (n_faces-vector of 3-vectors with float elements): Pre-allocated vector.
det_f (n_faces-vector with float elements): Pre-allocated vector.

Returns
-------
atten_calc (float): Attenuation factor.
"""
function atten_calc(
    ki :: SVector{3, Float32}, 
    kf :: SVector{3, Float32}, 
    en_f :: Float32, 
    vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    mc_coords :: Matrix{Float32}, 
    len_i :: Vector{Float32},
    p_f :: Vector{SVector{3, Float32}},
    det_f :: Vector{Float32}
    ) :: Float32
    # Calculating the absorption cross section after the neutron scatters.
    axsf = axs_calc(en_f)
    # Setting up the Moller-Trumbore algorithm for ray-triangle intersections.
    p_f, det_f = pdet_calc!(kf, e2s, e3s, p_f, det_f)
    atten = 0
    for i in 1:n_mc
        # Calculating the path length, len_f, at this sample point.
        len_f = len_calc(e2s, e3s, kf, p_f, det_f, mc_coords[i, :], vertices, indices)
        # Adding the attenuation factor contribution from this sample point to A.
        atten += (1 / n_mc) * exp(-n * axsi * len_i[i]) * exp(-n * axsf * len_f)
    end
    return atten
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
ki (3-vector with float elements): Pre-scattering neutron wavevector, in Angstrom^-1.
en_i (float): Pre-scattering neutron energy, in meV.
ef_bins (n_bins-vector with float elements): Post-scattering neutron energy of each bin, in meV.
vertices (n_faces-vector of 3-vectors with float elements): Vertices of the triangular faces.
indices (n_faces-vector of 3-vectors with integer elements): Indices describing which vertices correspond to which triangles.
e2s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V2 - V1.
e3s (n_faces-vector of 3-vectors with float elements): Array containing vectors parallel to each face, equal to V3 - V1.
mc_coords (n_mc-vector of 3-vectors with float elements): Coordinates of sample points used in MC method.
len_i (n_mc-vector with float elements): Pre-scattering path length of neutron, in units of .stl file.

Returns
-------
A_grid (n_bins x n_detectors matrix of floats): Attenuation factor grid.
"""
function a_grid_calc(
    data :: Matrix{Float32}, 
    kx :: AbstractMatrix{Float32}, 
    ky :: AbstractMatrix{Float32}, 
    kz :: AbstractMatrix{Float32}, 
    ki :: SVector{3, Float32}, 
    ef_bins :: SVector{n_bins, Float32}, 
    vertices :: Vector{Point{3, Float32}}, 
    indices :: Vector{NgonFace{3, OffsetInteger{-1, UInt32}}}, 
    e2s :: Vector{SVector{3, Float32}}, 
    e3s :: Vector{SVector{3, Float32}}, 
    mc_coords :: Matrix{Float32}, 
    len_i :: Vector{Float32}
    ) :: Matrix{Float32}
    A_grid = zeros(n_bins, n_detectors)
    # Determining the locations in which the signal is either NaN or 0 as we don't want to calculate atten there.
    idx = findall(.~((data .== 0) .| (isnan.(data))))
    # Pre-allocating the vectors needed for the MT algorithm.
    p_f = Vector{SVector{3, Float32}}(undef, n_faces)
    det_f = Vector{Float32}(undef, n_faces)
    # Skipping checks on array lengths using @inbounds.
    @inbounds for I in idx
        kf = [kx[I], ky[I], kz[I]]
        A_grid[I] = atten_calc(ki, SVector{3}(kf), ef_bins[I[1]], vertices, indices, e2s, e3s, mc_coords, len_i, p_f, det_f)
    end
    return A_grid
end


# Testing the time taken to output this grid of attenuation factors.


A_grid = a_grid_calc(data, kx, ky, kz, ki, ef_bins, vertices, indices, e2s, e3s, mc_coords, len_i)
# Testing the same (known to be non-zero) datapoint.
display(A_grid[160,6])


# Converting the grid of data and attenuation factors to sparse arrays.


# Replacing every NaN value with zero to allow conversion to sparse matrix.
data_copy = copy(data)
data_copy .= ifelse.(isnan.(data_copy), 0, data)
s_data = sparse(data_copy)
# Converting the attenuation factors to a sparse matrix.
s_atten = sparse(A_grid)
display(s_atten)