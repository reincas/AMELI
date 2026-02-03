# AMELI 1.2.0 (Angular Matrix Elements of Lanthanide Ions)

This is a Python 3 package to calculate the angular matrix elements of spherical 
tensor operators of multi-electron systems. The main purpose of this software
is the preparation of a large set of symbolic state matrices of spherical tensor
operators, mainly perturbation Hamiltonians of lanthanide ions (rare-earths), for
publication in a Zenodo repository. 

AMELI calculates operator matrices in the product state space, but it is able to
transform them to LS coupling. All characteristic eigenvalues and irreducible
representations for each state are provided. Global signs are synchronized for
states inside a Stark group (same J). This allows to use reduced matrices for
intermediate coupling in amorphous materials.

For symbolic calculations AMELI uses SymPy. This is is a feasible approach, since
each matrix element can be expressed as signed square root of a rational number.
If you are interested in an alternative approach based on floating point numerical
operations, see the [Lanthanide](https://github.com/reincas/Lanthanide) package.

Operator matrices and other intermediate calculation results are stored as 
[SciDataContainer](https://scidatacontainer.readthedocs.io) files enriched with
plenty of meta data and descriptions. Following the FAIR data principles of
data management the container content is easily accessible to humans and machines.
A SciDataContainer file is a ZIP folder containing JSON files for the meta data
and HDF5 files for the numerical data.
