# AMELI 1.3.0 (Angular Matrix Elements of Lanthanide Ions)

This is a Python 3 package to calculate the constant angular matrix elements of spherical tensor operators of
lanthanide ions in exact arithmetic using SymPy.
Each matrix element is stored as signed square root of a rational number.
The basic functions support any many-electron configuration, but the implementation of most high-level operators is
specific for single-shell systems like the lanthanide ions. 

The main purpose of this software is the preparation of the comprehensive matrix datasets in the
[AMELI community](https://zenodo.org/communities/ameli) on Zenodo.
Each lanthanide dataset for the electron configurations $f^1$ to $f^{13}$ contains unit tensor operators, angular
momentum operators and perturbation Hamiltonians.
All relevant perturbation operators are provided in first order (Coulomb, spin-orbit, spin-spin, spin-other-orbit) and
in second order (Coulomb, electrostatic spin-orbit).

The AMELI matrix datasets are intended as replacements for the printed and incomplete tables published mainly by
B. R. Judd, W. T. Carnall, C. W. Nielson, and G. F. Koster on the 1960s and 1970s.
In particular they are intended to replace all tables of reduced unit tensor matrix elements in intermediate coupling
still used for Judd-Ofelt calculations.
Such calculations should instead be based on a set of radial integrals serving as linear expansion coefficients for the
operator matrices.

## General Remarks

All angular matrices of a given electron configuration are constants and need to be calculated only once. 
This software package therefore is intended mainly as reference.
Instead of integrating AMELI code into your own project, you should use the matrices from the AMELI repository on
Zenodo.
A reference implementation will be provided by the [Lanthanide](https://github.com/reincas/Lanthanide) package,
which will soon be switching from self-generated matrices to AMELI.

AMELI calculates operator matrices in the product state space together with a transformation matrix to $LS$-coupling
in exact arithmetic.
All characteristic eigenvalues and irreducible representations for each state are provided.
Global signs are synchronized for states inside each $J$-multiplet.
This allows to use reduced matrices for intermediate coupling in amorphous materials.

All matrices and other intermediate calculation results are stored as
[SciDataContainer](https://scidatacontainer.readthedocs.io) files enriched with plenty of meta data and descriptions.
Following the [FAIR data principles](https://en.wikipedia.org/wiki/FAIR_data) of research data management the container
content is easily accessible to humans and machines.
A Python implementation is available, but not necessary, because a SciDataContainer file is just a ZIP folder.
AMELI containers consist only of JSON files for the meta data and HDF5 files for the numerical data.

## Documentation

A paper on the mathematical background of AMELI, its workflow, and a detailed comparison of the resulting tensor
matrices to the literature is [available on arXiv](https://doi.org/10.48550/arXiv.2603.21947) and will be submitted
to a scientific journal very soon.  

## Installation

You can install the AMELI software directly from PyPI using `pip`:

```
pip install ameli
```

## Application 1: Energy Levels of Lanthanides

The matrices can be used for the fit of lanthanide energy levels to measured absorption spectra resulting in radial
integrals and the coefficients of all states in intermediate coupling.
For crystalline materials these calculations can take full advantage of all spectral lines using the even-rank unit
tensor operators for modelling the crystal-field splitting.
This includes the mixing of states with different $J$ quantum number.
For amorphous materials the calculations can take advantage of the effective rotational site symmetry which allows to
perform them in a reduced $SLJ$ space with much smaller matrices.

## Application 2: Transition Intensities of Lanthanides

Based on radial integrals from the literature or own energy-level fits, the coefficients of all states in intermediate
coupling can be determined as linear combination of $LS$-states.
This makes the generation of all reduced matrix elements for Judd-Ofelt calculations a simple task.
While the Judd-Ofelt theory is standard to predict the radiative intensities of every emission and absorption
transition from measured absorption spectra in amorphous hosts, it is not intended for crystalline hosts.
The AMELI repository contains all matrices for the Crystal Field Intensity (CFI) method instead, which uses the same
mathematical procedure as Judd-Ofelt, but with a larger set of parameters reflecting the richer structure of 
crystalline spectra.
Based on odd-rank unit tensor operators the electric dipole operators for any site-symmetry can be modelled in
addition to the magnetic dipole operator, which is the same for amorphous and crystalline materials.

## Package Structure

The main code of the AMELI package is contained in the folder `ameli`.
The script `generate.py` in the folder `generate` is used to calculate the full set of matrices for all lanthanide
configurations.
Due to the exact arithmetic this is a time-consuming process.
Even though the script builds a dependency graph and schedules the computation tasks to all available CPU cores, it
takes several days to finish.
The folder `test`contains a set of test scripts orchestrated by the main script `test.py` which perform a large number
of mathematical tests and comparisons to values published in printed literature.
Its subfolder `results` contains the results from a test run and the subfolder `tables` contains the script used to
extract the comparison tables in the AMELI paper from the test results.
The folder `upload` is for documentation only.
It contains the scripts used to upload and update the matrix datasets on the Zenodo repository.

## Update History

* **1.2.1**: First public release
* **1.2.2**: Documentation update
* **1.3.0**: Fixes and supplements
    * Fixed splitting of tau-branches for f5-f9
    * Replaced sign correction algorithm by ladder operator
    * Added Coulomb and crystal field operators (C, Hcf, Dcf)

## Reference

**Reinhard Caspary (2026):** *AMELI: Angular Matrix Elements of Lanthanide Ions.* 
[arXiv preprint 2603.21947](https://arxiv.org/abs/2603.21947),
[DOI: 10.48550/arXiv.2603.21947](https://doi.org/10.48550/arXiv.2603.21947),
[pdf](https://arxiv.org/pdf/2603.21947)