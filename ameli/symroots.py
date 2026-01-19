##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides classes representing SymPy square roots
# of rational numbers:
#
#    expr = (-1)^s * sqrt(n/d)
#
# with two integer values n and d and a sign flag s.
#
# The class RationalRadical represents a single expression. It may be
# initialised directly from a given SymPy expression or from a
# dictionary using the classmethod from_dict(). Its method to_dict()
# exports its content into such a dictionary.
#
# The class RationalRadicalList represents a list of expressions. It may
# be initialized directly from a list of SymPy expressions or from a
# dictionary using the class method from_dict(). Its method to_dict()
# exports its content into such a dictionary. The class uses uint
# encoding and decoding from the module uintarray to allow the storage
# of numerators and denominators with unlimited bit-length in numpy
# arrays.
#
##########################################################################

import sympy as sp

from .uintarray import encode_uint_arrays, decode_uint_array


##########################################################################
# Square root of a rational number
##########################################################################

class RationalRadical:
    """ Data object representing a SymPy expression which can be written as the square root of a rational number
    '(-1)^s * sqrt(n/d)' with a sign flag s and two integers n and d."""

    def __init__(self, expr: sp.Expr | str):
        """ Initialize a new object from the given SymPy expression. """

        # Expression must simplify to a number
        expr = sp.sympify(expr)
        assert expr.is_number

        # Split the expression into four integers
        self.sign, self.numerator, self.denominator = self.split_sqrt_fraction(expr)

    @classmethod
    def from_dict(cls, args: dict) -> "RationalRadical":
        """ Create a RationalRadical object from a dictionary. """

        value = cls.__new__(cls)
        value.sign = args["sign"]
        value.numerator = args["numerator"]
        value.denominator = args["denominator"]
        return value

    def __str__(self):
        """ Return a string representation of the object. """

        return str(self.expr)

    def __eq__(self, other):
        """ Determine the equality of two RationalRadical objects. """

        assert isinstance(other, RationalRadical)

        return (self.sign == other.sign and
                self.numerator == other.numerator and
                self.denominator == other.denominator)

    @property
    def expr(self) -> sp.Expr:
        """ Return the object as SymPy expression 'expr = (-1)^s * sqrt(c/d)'. """

        expr = sp.sqrt(sp.Rational(self.numerator, self.denominator))
        if self.sign:
            return -expr
        return expr

    @classmethod
    def split_sqrt_fraction(cls, expr: sp.Expr) -> tuple:
        """ Split the given SymPy expression into three integers s, n and d with '(-1)^s * sqrt(c/d)'. Return the
        integers if possible and raise an AssertionError otherwise."""

        # Expression must simplify to a number, no symbols allowed
        expr = sp.sympify(expr)
        assert expr.is_number

        # Expression is '0'
        if expr.is_zero:
            return 0, 0, 1

        # Store the sign of the expression
        sign = int(expr.is_negative)

        # Numerator and denominator of the squared expression
        expr = expr * expr
        assert expr.is_rational
        numerator, denominator = expr.as_numer_denom()

        # Return the integer arguments
        return sign, numerator, denominator

    def as_dict(self) -> dict:
        """ Return a dictionary representation of the object. """

        return {
            "sign": self.sign,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


##########################################################################
# List of square roots of rationals
##########################################################################

class RationalRadicalList:
    """ Data object representing a list of RationalRadical objects. It uses uint encoding and decoding from
    the module uintarray to allow the storage of numerators and denominators with unlimited bit-length in
    numpy arrays. """

    def __init__(self, expressions: list):
        """ Store given sequence of SymPy expressions as list of RationalRadical objects. """

        self.values = [RationalRadical(expr) for expr in expressions]

    @classmethod
    def from_dict(cls, meta: dict) -> "RationalRadicalList":
        """ Create a RationalRadicalList object from a uint-encoded dictionary representation. """

        obj = cls.__new__(cls)
        values = {
            "sign": decode_uint_array(meta, "sign"),
            "numerator": decode_uint_array(meta, "numerator"),
            "denominator": decode_uint_array(meta, "denominator"),
        }
        size = len(values["numerator"])
        values = [{key: int(values[key][i]) for key in values} for i in range(size)]
        obj.values = [RationalRadical.from_dict(value) for value in values]
        return obj

    def as_dict(self) -> dict:
        """ Return a uint-encoded dictionary representation of the object. """

        values = [value.as_dict() for value in self.values]
        values = {key: [value[key] for value in values] for key in values[0]}
        return encode_uint_arrays(values)
