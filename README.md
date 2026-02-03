# AMELI 1.1.0 (Angular Matrix Elements of Lanthanide Ions)

This is a Python 3 package to calculate the angular matrix elements of spherical 
tensor operators of multi-electron systems. The main purpose of this software
is the preparation of a large set of state matrices of spherical tensor operators,
mainly perturbation Hamiltonians of lanthanide ions (rare-earths), for publication
in a Zenodo repository. 

AMELI calculates operator matrices in the product state space, but it is able to
transform them to LS coupling. All characteristic eigenvalues and irreducible
representations for each state are provided. Global signs are synchronized for
states inside a Stark group (same J). This allows to use reduced matrices for
intermediate coupling in amorphous materials.

For numerical calculation AMELI supports all floating point data types which
are also supported by numpy, namely 'float16', 'float32', and 'float64'. AMELI
is also able to use SymPy to calculate and store matrices with symbolic elements.
Each matrix element is the square root of a rational number.

**Note:** This is the final release which contains the code to calculate floating
point matrices, although it is not used any more. This code will be removed in
the next version. Floating point matrices can always be derived directly from
their exact symbolic counterparts. See [Lanthanide](https://github.com/reincas/Lanthanide)
if you are interested in pure floating point calculations. 

Operator matrices and other intermediate calculation results are stored as 
[SciDataContainer](https://scidatacontainer.readthedocs.io) files enriched with
plenty of meta data and descriptions. Following the FAIR data principles of
data management the container content is easily accessible to humans and machines.
A SciDataContainer file is a ZIP folder containing JSON files for the meta data
and HDF5 files for the numerical data.
