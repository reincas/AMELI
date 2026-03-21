# AMELI 1.2.0 (Angular Matrix Elements of Lanthanide Ions)

This is a Python 3 package to calculate the angular matrix elements of spherical tensor operators of lanthanide ions
in exact arithmetic using SymPy.
Each matrix element is stored as signed square root of a rational number.
The basic functionality is ready for any many-electron configuration, but the implementation of most high-level
operators is specific for single-shell systems like the lanthanide ions. 

The main purpose of this software is the preparation of the comprehensive matrix datasets in the
[AMELI repository](https://zenodo.org/communities/ameli) on Zenodo.
Each lanthanide dataset for the configurations $f^1$ to $f^{13}$ contains unit tensor operators, angular momentum
operators and Hamiltonians.
All relevant perturbation operators are provided in first order (Coulomb, spin-orbit, spin-spin, spin-other-orbit) and
in second order (Coulomb, electrostatic spin-orbit).

The AMELI matrix datasets are intended as replacements for the printed incomplete tables published by B. R. Judd,
W. T. Carnall, C. W. Nielson, and G. F. Koster on the 1960s and 1970s.
In particular they are intended to replace all tables of reduced unit tensor matrix elements in intermediate coupling
used for Judd-Ofelt calculations up to now.
Such calculations should be based on a set of radial integrals serving as linear expansion coefficients for the
operator matrices.

## General Remarks

All angular matrices of a given electron configuration are constants and need to be calculated only once. 
This package therefore is intended mainly as reference.
Instead of integrating AMELI code in your own project, you should use the matrices from the AMELI repository on Zenodo.
An example representation will be the [Lanthanide](https://github.com/reincas/Lanthanide) package, which will soon
switch from self-generated matrices to AMELI.

AMELI calculates operator matrices in the product state space, but it also generates a transformation matrix to
$LS$-coupling in exact arithmetic.
All characteristic eigenvalues and irreducible representations for each state are provided.
Global signs are synchronized for states inside each $J$-multiplet.
This allows to use reduced matrices for intermediate coupling in amorphous materials.

All matrices and other intermediate calculation results are stored as
[SciDataContainer](https://scidatacontainer.readthedocs.io) files enriched with plenty of meta data and descriptions.
Following the [FAIR data principles](https://en.wikipedia.org/wiki/FAIR_data) of data management the container content
is easily accessible to humans and machines.
A Python implementation is available, but not necessary, because a SciDataContainer file is just a ZIP folder.
AMELI containers consist only of JSON files for the meta data and HDF5 files for the numerical data.

## Application 1: Energy-Level Fits

The matrices are used for the fit of lanthanide energy levels to measured absorption spectra resulting in radial
integrals and coefficients of all states in intermediate coupling.
For crystalline materials these calculations can take full advantage of the all spectral lines using the even-rank unit
tensor operators for modelling the crystal-field splitting.
This includes the mixing of states with different $J$ quantum number.
For amorphous materials the calculations can take advantage of the effective rotational site symmetry which allows to
perform them in a reduced $SLJ$ space with much smaller matrices.

## Application 2: Fit of Transition Intensities

Based on radial integrals from the literature or own energy-level fits, the coefficients of all states in intermediate
coupling can be determined as linear combination of $LS$-states.
This makes building all reduced matrix elements for Judd-Ofelt calculations very simple.
While the Judd-Ofelt theory is standard to predict the radiative transition intensities in amorphous hosts, it should
be avoided for crystalline hosts.
The AMELI repository contains all matrices for the Crystal Field Intensity (CFI) method, which uses the same
mathematical procedure as Judd-Ofelt, but with a larger set of parameters reflecting the richer structure of 
crystalline spectra.
Based on the odd-rank unit tensor operators the electric dipole operators for any site-symmetry can be modelled in
addition to the magnetic dipole operator, which is the same for amorphous and crystalline materials.

