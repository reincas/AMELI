# AMELI 1.2.0 (Angular Matrix Elements of Lanthanide Ions)

This is a Python 3 package to calculate the angular matrix elements of spherical tensor operators of multi-electron
systems in exact arithmetic using SymPy.
Each matrix element is stored as signed square root of a rational number.
The main purpose of this software is the preparation of the comprehensive set of matrices (unit tensors, angular
momentum operators and Hamiltonians) in the [AMELI repository](https://doi.org/10.5281/zenodo.19144765) on Zenodo.

AMELI calculates operator matrices in the product state space, but it also generates a transformation matrix to
$LS$-coupling in exact arithmetic.
All characteristic eigenvalues and irreducible representations for each state are provided.
Global signs are synchronized for states inside each $J$-multiplet.
This allows to use reduced matrices for intermediate coupling in amorphous materials.

All angular matrices of a given electron configuration are constants and need to be calculated only once. 
This package therefore is intended mainly as reference.
Instead of integrating AMELI code in your own project, you should use the matrices from the AMELI repository on Zenodo.
An example representation will be the [Lanthanide](https://github.com/reincas/Lanthanide) package, which will soon
switch from self-generated matrices to AMELI.

All matrices and other intermediate calculation results are stored as
[SciDataContainer](https://scidatacontainer.readthedocs.io) files enriched with plenty of meta data and descriptions.
Following the [FAIR data principles](https://en.wikipedia.org/wiki/FAIR_data) of data management the container content
is easily accessible to humans and machines.
A Python implementation is available, but not necessary, because a SciDataContainer file is just a ZIP folder.
AMELI containers consist only of JSON files for the meta data and HDF5 files for the numerical data.
