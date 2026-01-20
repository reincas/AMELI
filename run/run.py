##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging

import numpy as np
import sympy as sp

from ameli.datatype import DataType
from ameli.config import Config
from ameli.product import Product
from ameli.unit import Unit
from ameli.matrix import Matrix
from ameli.transform import Transform

from lanthanide import Lanthanide, Coupling


def num_diff(value_sym, value):
    assert isinstance(value_sym, sp.Expr)
    assert np.issubsctype(value, np.floating)
    dtype = value.dtype
    eps = num.finfo(dtype).eps
    value_ref = dtype(value_sym)
    return abs(value_ref - value) / (eps + eps * value_ref)


##########################################################################
# Log handlers
##########################################################################

def log_console(formatter, level):
    filename = "stdout"
    logger = logging.getLogger()
    handler = [handler for handler in logger.handlers if handler.name == filename]
    if handler:
        handler = handler[0]
    else:
        handler = logging.StreamHandler()
        handler.set_name = filename
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    handler.setLevel(level)


def log_file(filename, formatter, level):
    logger = logging.getLogger()
    handler = [handler for handler in logger.handlers if handler.name == filename]
    if handler:
        handler = handler[0]
    else:
        handler = logging.FileHandler(filename)
        handler.set_name = filename
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    handler.setLevel(level)


def check_orthonormal(dtype, config_name):
    transform = Transform(dtype, config_name)
    config = transform.config
    dtype = transform.dtype
    if dtype.is_symbolic:
        V = transform.matrix
        diff = (V.T * V - sp.eye(transform.num_states)).norm(sp.oo)
        success = diff == 0
        diff = "" if success else f" result = {diff} abs"
    else:
        V = transform.matrix
        diff = np.max(np.abs(V.T @ V - np.eye(V.shape[0])) / (dtype.eps + dtype.eps * np.eye(V.shape[0])))
        success = diff < 100
        diff = f" result = {diff:.0f} eps"
    res = "passed" if success else "FAILED"
    print(f"{config.name} ({dtype.name}) | Orthonormality test{diff}: {res}")
    return success


def check_transform(dtype, config_name):
    transform = Transform(dtype, config_name)
    fails = []
    for i, name in enumerate(transform.col_states.tensor_chain):
        if name in ("sen", "num", "tau"):
            continue
        matrix = Matrix(dtype, config_name, name, "Product").matrix
        eigenvalues = transform.eigenvalue_lists()[name]
        if dtype.is_symbolic:
            V = transform.matrix
            D = sp.diag(*eigenvalues)
            diff = (V.T * matrix * V - D).norm(sp.oo)
            is_equal = diff == 0
        else:
            V = transform.matrix
            D = np.diag([dtype.dtype(sp.S(x)) for x in eigenvalues])
            diff = np.max(np.abs(V.T @ matrix @ V - D) / (dtype.eps + dtype.eps * np.abs(D)))
            print(f"  {config_name} Transformation {name} diff = {diff:.0f} eps")
            is_equal = diff < 100
        if not is_equal:
            fails.append(name)
    success = len(fails) == 0
    if not success:
        fails = ", ".join(fails)
        print(f"  {config_name} Transformation failed for {fails}")
    res = "passed" if success else "FAILED"
    print(f"{config_name} ({dtype.name}) | {name} transformation test: {res}")
    return success


def compare_transform(dtype, config_name):
    dtype_ref = DataType("symbolic")
    transform_ref = Transform(dtype_ref, config_name)
    transform = Transform(dtype, config_name)
    assert transform_ref.col_states.names == transform.col_states.names
    diffs = []
    for i, state in enumerate(transform.col_states.names):
        eigenvector_ref = dtype.array(transform_ref.matrix)[:, i]
        eigenvector = transform.matrix[:, i]
        diffs.append((np.abs(np.dot(eigenvector_ref, eigenvector)) - 1.0) / dtype.eps)
    diff = max(diffs)
    success = diff < 100
    res = "passed" if success else "FAILED"
    print(f"{config_name} (symbolic <-> {dtype.name}) | Transformation vectors compare result = {diff:.0f} eps: {res}")
    return success


def check_matrix(config_name, name, space):
    dtype_sym = DataType("symbolic")
    obj_sym = Matrix(dtype_sym, config_name, name, space)
    dtype_num = DataType("float64")
    obj_num = Matrix(dtype_num, config_name, name, space)
    if space == "SLJM":
        assert obj_sym.states.names == obj_num.states.names

    matrix_sym = dtype_num.array(obj_sym.matrix)
    matrix_num = obj_num.matrix
    diff = np.max(np.abs(matrix_sym - matrix_num)) / dtype_num.eps
    success = np.allclose(matrix_sym, matrix_num, atol=1000 * dtype_num.eps)
    res = "passed" if success else "FAILED"
    print(f"{config_name} | matrix {name} test {space}: {res} ({diff:.0f} eps)")
    return success


def compare_old(names):
    for name_new, name_old in names:
        array_new = Matrix(config_name, name_new).array()
        with Lanthanide(num_electrons) as ion:
            array_old = ion.matrix(name_old, Coupling.Product).array
        equal = np.all(np.isclose(array_old, array_new, atol=1e-12))
        print(f"{config_name} | {name_new}: {equal}")
        if not equal:
            print(array_new[:5, :5])
            print(array_old[:5, :5])
            assert equal


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    # log_file("ameli.log", formatter, loglevel)
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    names = []
    names.extend([(f"U/{k},{q}", f"U/a/{k},{q}", 1) for k in range(7) for q in range(-k, k+1)])
    names.extend([(f"T/{k},{q}", f"T/a/{k},{q}", 1) for k in range(2) for q in range(-k, k+1)])
    names.extend([(f"UU/{k}", f"UU/{k}", 2) for k in (0, 1, 2, 3, 4, 5, 6)])
    names.extend([(f"TT/{k}", f"TT/{k}", 2) for k in (0, 1)])
    names.extend([(f"UT/{k}", f"UT/{k}", 2) for k in (0, 1)])
    names.extend([(f"L/{q}", f"L/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"S/{q}", f"S/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"J/{q}", f"J/{q}", 1) for q in range(-1, 2)])
    names.append(("L2", "L2", 1))
    names.append(("S2", "S2", 1))
    names.append(("J2", "J2", 1))
    names.append(("LS", "LS", 1))
    names.extend([(f"H1/{k}", f"H1/{k}", 2) for k in (2, 4, 6)])
    names.append(("H2", "H2", 1))
    names.extend([(f"H3/{i}", f"H3/{i}", 2) for i in (0, 1, 2)])
    names.extend([(f"H4/{c}", f"H4/{c}", 3) for c in (2, 3, 4, 6, 7, 8)])
    names.extend([(f"H5/{k}", f"H5/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H6/{k}", f"H6/{k}", 2) for k in (2, 4, 6)])

    # check_rational(config_name)
    # check_orthonormal(dtype_sym, config_name)
    # check_transform(dtype_sym, config_name)
    # print()
    # check_orthonormal(dtype_num, config_name)
    # check_transform(dtype_num, config_name)
    # print()
    # compare_transform(dtype_num, config_name)

    for num_electrons in range(1, 14):
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM"):
                for name, _, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue
                    matrix = Matrix(dtype, config_name, name, space)
