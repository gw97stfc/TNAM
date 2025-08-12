include("Modules/nxspe.jl")
using .nxspe
using HDF5

# # Creating the .nxspe file outputted.
# corrected_nxspe = "output_nxspe_data/c_LET104215_3.7meV_1to1.nxspe"
# # Copying the .nxspe file it will correct.
# cp("test_nxspe_data/LET104215_3.7meV_1to1.nxspe", corrected_nxspe)
# # Replacing the data with the corrected data.
# c_data = zeros(Float64, (320, 98304))
# c_data[1, 1] = 80085
# h5open(corrected_nxspe, "r+") do f
#     write(f["ws_out/data/data"], c_data)
# end
string = "abcdsa"
string1 = "1234$string"
display(string1)