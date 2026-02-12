##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from .. import desc_format
from ..matrix import MATRIX_INFO
from ..transform import SYM_INFO
from . import LANTHANIDE_IONS
from .content import get_matrix_heads

DESCRIPTION = """
<p>
This dataset provides exact symbolic angular matrices of spherical tensor operators for the {cfg} configuration
(lanthanide ion {element}<sup>3+</sup>).
The matrices are derived from the software package <a href="https://github.com/reincas/ameli">AMELI</a> 
(Angular Matrix Elements of Lanthanide Ions).
They are calculated in the determinantal product state space and transformed to LS coupling.
Each LS state is classified by a complete set of irreducible representations matching those published by C. W. Nielson 
and G. F. Koster.
The Hamiltonian matrices are matching the values published by B. R. Judd and W. T. Carnall.
</p>

<h2>Matrix Storage Format</h2>
<p>
The value $x$ of a non-zero matrix element is represented by three values $s$, $n$, and $d$ as signed square root of a 
rational number:
$$x=(-1)^s\\sqrt{{\\frac{{n}}{{d}}}}$$
The storage format leverages the sparsity of the matrices.
It is closely related to the COO standard format for sparse matrices with three vectors for the row index, column 
index, and value index of each non-zero matrix element.
For symmetric matrices only the lower triangle matrix is stored.
The actual values are stored in a separate list of unique values, the length of which is usually much smaller than the
total number of non-zero elements.
Storage follows the canonical COO order with row as primary and column as secondary index.
The list of unique values is sorted in ascending order.
If a matrix contains integer values $n$ or $d$ that exceed the 64-bit maximum of HDF5 data types, its values are split
bit-wise and stored in multiple parts.
</p><p>
Each matrix is stored in a <a href="https://scidatacontainer.readthedocs.io/en/latest/index.html">SciDataContainer</a>
file with filename extension <code>.zdc</code>.
This container file is a standard zip folder with a certain inner structure of metadata in JSON files following the
<a href="https://en.wikipedia.org/wiki/FAIR_data">FAIR</a> principles of modern research data management.
The matrix data is stored in the HDF5 file <code>data/matrix.hdf5</code> inside the container.
A detailed description of the data structures is given in the JSON file <code>meta.hdf5</code>.
</p>

<h2>Content of Dataset Record</h2>
<p>
All tensor operator matrices are grouped with respect to their state spaces.
The following zip files with matrix container files are included in this dataset record:
<table>
  <tr>
  <th>File</th><th>Description</th>
  </tr>
  <tr>
    <td><code>product.zip</code></td>
    <td>Tensor operator matrices in the product state space.</td>
  </tr>
  <tr>
    <td><code>sljm.zip</code></td>
    <td>Tensor operator matrices in LS coupling.</td>
  </tr>
  <tr>
    <td><code>slj.zip</code></td>
    <td>Tensor operator matrices of the Stark groups in LS coupling.
        These are matrices from <code>sljm.zip</code> collapsed to the stretched states $M_J=-J$.
        Intended for energy level fits for lanthanide ions in amorphous materials.</td>
  </tr>
  <tr>
    <td><code>slj_reduced.zip</code></td>
    <td>Matrices with reduced matrix elements of tensor operators.
        The state space of these matrices is the same as for those in <code>slj.zip</code>.
        Intended for Judd-Ofelt calculations for lanthanide ions in amorphous materials.</td>
  </tr>
  <tr>
    <td><code>support.zip</code></td>
    <td>This special file is only useful when the AMELI package is extended to generate new custom operator matrices.
        In this case the included data containers with intermediate results optimize the calculation speed within the 
        AMELI package.</td>
  </tr>
</table>
</p>

<h2>Matrix Files</h2>
<p>
Each set of tensor operator matrices contains the following files:
<table>
<th>Filename</th>
<th>Description</th>
<th>Tensor Operator</th>
<th>Radial Integral</th>
{matrix_rows}</table>  
</p>
<p>
The component part <code>_{{q}}</code> is missing in the filenames of the set of matrices containing reduced matrix 
elements.
Reduced matrix elements are an entity introduced by the 
<a href="https://en.wikipedia.org/wiki/Wigner%E2%80%93Eckart_theorem">Wigner-Eckart theorem</a>, which describes the 
angular behaviour of any spherical tensor operator $\\mathrm{{A}}^{{(k)}}$:
$$
\\langle J' M' | \\mathrm{{A}}^{{(k)}}_{{q}} | J M \\rangle = (-1)^{{J'-M'}}
\\begin{{pmatrix}} J' & k & J \\\\ -M' & q & M \\end{{pmatrix}}
\\langle J' || \\mathrm{{A}}^{{(k)}} || J \\rangle
$$ 
</p>

<h2>State Representation</h2>
<p>
Each matrix data container includes the quantum numbers of all states in the JSON file <code>data/matrix.json</code>.
For product states these are the quantum numbers $n$, $l$, $m_l$, $s$, and $m_s$ of each electron.
States in LS coupling are characterised by irreducible representations which in most cases are connected to eigenvalues 
of certain tensor operators. 
<table>
<th>Key</th>
<th>Irreducible Representation</th>
<th>Related Tensor Operator</th>
{ls_rows}</table>  
</p>

<h2>Version Policy</h2>
<p>
The dataset is published with a semantic version number in the scheme 'major.minor.patch', starting at '1.0.0'.
The <strong>patch number</strong> is incremented for every update of the datasets metadata without any modification of 
the included files.
The <strong>minor number</strong> identifies a dataset release with modified files.
However, these are only non-functional modifications of metadata in the data containers.
Modifications may include updated or corrected titles or descriptions in the metadata of data containers, or new 
metadata attributes.
The <strong>major number</strong> is incremented with functional modifications of data containers or additional tensor 
operators.
</p>
"""


def description(num_electrons):
    cfg = f"f<sup>{num_electrons}</sup>"
    element = LANTHANIDE_IONS[num_electrons]

    matrix_heads = get_matrix_heads(num_electrons)
    matrix_rows = []
    for name, info in MATRIX_INFO:
        if name not in matrix_heads:
            continue
        keys = info["keys"]
        if keys:
            name = name + "_" + "_".join([f"{{{key}}}" for key in keys])
        name = f"{name}.zdc"
        op = info["html_op"]
        desc = info["html_desc"]
        radial = info.get("html_radial", "")
        matrix_rows.append(f"<tr><td><code>{name}</code></td><td>{desc}</td><td>{op}</td><td>{radial}</td></tr>")
    matrix_rows = "\n".join(matrix_rows)

    ls_rows = []
    for name in ["S2", "C7", "C2", "L2", "J2", "Jz", "sen", "num", "tau"]:
        op = SYM_INFO[name]["html_op"]
        desc = SYM_INFO[name]["html_repr"]
        ls_rows.append(f"<tr><td><code>{name}</code></td><td>{desc}</td><td>{op}</td></tr>")
    ls_rows = "\n".join(ls_rows)

    kwargs = {
        "cfg": cfg,
        "element": element,
        "matrix_rows": matrix_rows,
        "ls_rows": ls_rows,
    }
    return desc_format(DESCRIPTION, kwargs)
