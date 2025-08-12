# This module contains functions that extract and manipulate the data taken from the .nxspe files.
# These functions will be shared for both single and general crystal morphologies.
module nxspe


using HDF5
import PhysicalConstants.CODATA2018: m_n, e, ħ
using Unitful
using StaticArrays
using SparseArrays


# Defining a function to extract the useful parts of the .nxspe files.


"""
Extracts the initial energy, azimuthal angles, polar angles, measured data and energy changes for each bin from the .nxspe file.

Parameters
----------
file_name (String): Path to the .nxspe file.

Returns
-------
en_i (float): Initial energy, in meV.
azi (n_detectors - vector with float elements): Azimuthal angles of detectors, in degrees.
pol (n_detectors - vector with float elements): Polar angles of detectors, in degrees.
data (n_bins x n_detectors matrix with float elements): Measured signal at each detector for each bin.
Δen (n_bins - vector with float elements): Energy change for each bin, in meV.
"""
function extract(file_name :: String)
    # Extracting the contents of the .nxspe file.
    en_i, azi, pol, data, Δen = h5open(file_name, "r") do f
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
    return en_i, azi, pol, data, Δen
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


# Defining a function to calculate the final energy of neutrons from each bin, based on Ei and the energy change.


"""
Calculates the final neutron energy, in meV, for each energy bin.

Parameters
----------
en_i (float): Pre-scattering neutron energy, in meV.
Δen ((n_bins + 1)-vector with float elements): Neutron energy changes, in meV.
n_bins (integer): Number of bins

Returns
-------
ef_bins (n_bins-vector with float elements): Post-scattering neuton energy for each bin, in meV.
"""
function ef_calc(en_i :: Float32, Δen :: Vector{Float32}, n_bins :: Integer) :: Vector{Float32}
    ef_bins = zeros(n_bins)
    for i in 1:n_bins
        # Finding the bin centres by averaging the energies on each end of the bin.
        # Determining the final neutron energy, Ef = Ei - Δen based on which energy bin we are considering.
        ef_bins[i] = en_i - ((Δen[i] + Δen[i+1]) / 2)
    end
    # Returning the final energies in a static array.
    return ef_bins
end


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
function kf_calc(ef_bins :: Vector{Float32}, pol :: Vector{Float32}, azi :: Vector{Float32})
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


# Defining the function to correct the data/signal for absorption.


"""
Corrects the measured data by taking neutron absorption into consideration. The measured signal is divided by the corresponding attenuation factor.

Parameters
----------
data (n_bins x n_detectors matrix with float elements): Neutron signal measured at different detectors for different energy bins.
atten (n_bins x n_detectors matrix with float elements): Attenuation factor for different detectors and energy bins.

Returns
-------
c_data (n_bins x n_detectors matrix with float elements): Corrected neutron signal at different detectors for different energy bins.
"""
function abs_corr(
    data :: Matrix{Float32}, 
    atten :: Matrix{Float32}
    ) :: Matrix{Float32}
    # Copying the data to keep information as to location of NaN versus 0.
    c_data = copy(data)
    # Finding the indices (the energy bin and detector) at which we have a signal.
    idx = findall(.~((data .== 0) .| (isnan.(data))))
    # Correcting each signal by dividing by the corresponding attenuation factor.
    for I in idx
        c_data[I] = c_data[I] / atten[I]
    end
    return c_data
end

end