# Tomographic Neutron Absorption Models

This program provides a set of tools for applying tomographic absorption models to neutron scattering data for a general sample morphology.

## Basis for Implementation

The attenuation factor for a given $k_{i}\rightarrow k_{f}$ scattering is given by [Boothroyd, Chapter 10],

$\frac{1}{V}\int_{V} \exp(-\mu_{i}L_{i})\exp(-\mu_{f}L_{f})\mathrm{d}V$

This program approximates this volume integral through Monte-Carlo sampling of $N$ points within the sample, _i.e._

$\frac{1}{N} \sum_{i}^{N} \exp(-\mu_{i}L_{i})\exp(-\mu_{f}L_{f})$

## Practical Notes

The program has been implemented in Julia for syntactical ease and computational efficiency. In general the computation is demanding depending on the number of neturon scattering events which require attenuation correction.

To represent the 3D sample _.stl_ files are used. Neutron scattering files are stored in the _.nxspe_ (h5 format) for inelastic neutron scatteirng from direct geormetry spectrometers at ISIS Neutron and Muon Source.

When working with this repository on a computer, use [Julia environments](https://pkgdocs.julialang.org/v1/environments/) to ensure that exactly the same version of various packages are used for compatibility.

## References

[Boothroyd] A. T. Boothroyd, _Principles of Neutron Scattering from Condensed Matter_ (Oxford, 2020)

