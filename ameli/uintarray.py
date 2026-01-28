##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# The internal module provides interface functions for SciDataContainer
# to store sequences or arrays containing non-negative integers of
# unlimited bit-length as numpy arrays with minimized uint datatypes.
# Arrays with very large elements are split bit-wise and stored in
# several parts, which are combined again in the decode functions.
# Available functions:
#
#   encode_uint_array(), encode_uint_arrays()
#   decode_uint_array(), decode_uint_arrays()
#
##########################################################################

import numpy as np


def max_bytes() -> int:
    """ Return number of bytes of the largest uint dtype supported by numpy. """

    dtypes = {np.dtype(dt).name for dt in np.sctypeDict.values()}
    dtypes = [getattr(np, name) for name in dtypes if name.startswith("uint")]
    dtypes.sort(key=lambda dt: np.iinfo(dt).bits)
    return dtypes[-1](0).itemsize


MAX_BYTES = max_bytes()


def get_dtype(max_val: int) -> tuple:
    """ Return the smallest uint dtype required for the storage of the given integer value and a boolean flag.
    If the flag is True, all array elements fit into the dtype array. If the flag is False, the higher bits of
    the  elements require another storage array. """

    # Number of bytes required to store the largest element
    max_val = int(max_val)
    bits = max_val.bit_length()
    bytes, remainder = divmod(bits, 8)
    if remainder:
        bytes += 1

    # A single numpy array is sufficient
    size = 1
    while size <= MAX_BYTES:
        if bytes <= size:
            return np.dtype(f"uint{8 * size}"), True
        size *= 2

    # Multiple numpy arrays are required
    return np.dtype(f"uint{8 * MAX_BYTES}"), False


def encode_uint_array(array, name: str) -> dict:
    """ Encode a sequence or array of non-negative integers into one or more numpy arrays with same shape, but
    different minimized uint dtypes and return them as values of a dictionary. If the input array fits into a
    single uint numpy array, the key is the given name. If the input array contains elements which are too large
    to fit in any available uint dtype, its elements are split bit-wise and stored in several uint arrays
    '<name>_part<i> with integer numbers i. The part array with i=0 contains the lowest order bits of the array
    elements. """

    # Initialize result dictionary
    result = {}

    # Empty array
    array = np.array(array)
    if 0 in array.shape:
        result[name] = array.astype(np.dtype("uint8"), copy=False)
        return result

    # Elements must be non-negative
    assert np.min(array) >= 0

    # Build arrays containing the lowest bits iteratively
    arrays = []
    finished = False
    while not finished:
        dtype, finished = get_dtype(np.max(array))
        if finished:
            arrays.append(array.astype(dtype, copy=False))
        else:
            bits = dtype.itemsize * 8
            mask = 2 ** bits - 1
            arrays.append((array & mask).astype(dtype, copy=False))
            array >>= bits

    # Store all arrays in the dictionary
    if len(arrays) < 2:
        result[name] = arrays[0]
    else:
        width = len(str(len(arrays) - 1))
        for i, array in enumerate(arrays):
            result[f"{name}_part{i:0{width}d}"] = array

    # Return the array dictionary
    return result


def encode_uint_arrays(arrays: dict) -> dict:
    """ Take a dictionary with several sequences or arrays and return a dictionary with all encoded arrays. The
    keys of the input array are taken as base-names for 'encode_large()'. """

    arrays = [encode_uint_array(values, name) for name, values in arrays.items()]
    return {key: value for d in arrays for key, value in d.items()}


def decode_uint_array(meta: dict, name: str) -> list:
    """ Decode an array encoded by 'encode_large()' with given base-name from the dictionary meta and return it as
    list. Each processed item is removed from meta. """

    # Single storage array
    if name in meta:
        array = meta[name].astype(object)

    # Combine large-element array stored in multiple parts
    else:
        keys = [key for key in meta if key.startswith(f"{name}_part")]
        keys.sort()
        array = meta.pop(keys[-1]).astype(object)
        for key in reversed(keys[:-1]):
            bits = meta[key].dtype.itemsize * 8
            array = (array << bits) | meta.pop(key).astype(object)

    # Return restored array as list
    return array.tolist()


def decode_uint_arrays(meta: dict, names: list) -> dict:
    """ Decode several arrays with given names from the dictionary meta and return them as lists in a dictionary
    with the names as keys. Each processed item is removed from meta. """

    arrays = {}
    for name in names:
        keys = [key for key in meta if key.startswith(f"{name}")]
        assert all(isinstance(value, np.ndarray) for value in meta[keys])
        assert len(set(value.shape for value in meta[keys])) == 1
        arrays[name] = decode_uint_array(meta, name)
    return arrays
