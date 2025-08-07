# This module contains tools to use to help out with debugging other programs.
module tools


using StaticArrays


# Defining a function to determine the indices of a desired wavevector.
# Usually used upon printing a (normalised) direction vector at which an error occurs and trying to locate it for further analysis.


"""
Determines the indices (ie the bin and detector) of a given normalised direction vector.

Parameters
----------
target_d (3-vector with float elements): Target direction vector.
s_data (n_bins x n_detectors sparse matrix with float elements): Neutron signal measured at different detectors for different energy bins.
kx (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in x direction, in Angstrom^-1.
ky (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in y direction, in Angstrom^-1.
kz (n_bins x n_detectors matrix with float elements): Post-scattering neutron wavevector component in z direction, in Angstrom^-1.

Returns
-------
I (vector with 2-vector elements with integer elements): Indices of wavevector corresponding to normalised direction vector.
"""
function d_locator(target_d, s_data, kx, ky, kz)
    # Creating empty array of indices.
    I = []
    # Iterating through each used wavevector.
    idx = findnz(s_data)
    @inbounds for (i, j) in zip(idx[1], idx[2])
        k_test = SVector(kx[i, j], ky[i, j], kz[i, j])
        # Normalising the wavevector as all direction vectors are.
        k_test = k_test / norm(k_test)
        if norm(k_test - target_d) <= 1f-6
            push!(I, [i, j])
        end
    end
    return I
end


end