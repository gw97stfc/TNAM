# This module contains the key functions in the attenuation calculation for the single, convex crystal.
module sin_crystal


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
function μ_calc(μ_ref, en_ref, en :: Float32) :: Float32
    return μ_ref * sqrt(en_ref / en)
end


end