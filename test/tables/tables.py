##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Generate the comparison tables for energy levels and reduced matrix
# elements for the publication.
#
##########################################################################

import re
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy import stats

sys.path.append("..")
from data_energy import RADIAL

DB_PATH = "../results/energy.db"
RED = "#{r:02X}{y:02X}{b:02X}".format(r=194, y=0, b=82)
BLUE = "#{r:02X}{y:02X}{b:02X}".format(r=0, y=80, b=155)
GRAY = "#{r:02X}{y:02X}{b:02X}".format(r=96, y=96, b=96)
BLACK = "#{r:02X}{y:02X}{b:02X}".format(r=0, y=0, b=0)

OPERATORS = {
    "E^1": 1, "E^2": 1, "E^3": 1,
    "H2": 2,
    "H3/0": 3, "H3/1": 3, "H3/2": 3,
    "H5/0": 5, "H5/2": 5, "H5/4": 5,
    "P_2": 6, "P_4": 6, "P_6": 6,
}

REFS = {
    "Pr3+:LaCl3": ("Carnall.1968", "$\\mathrm{Pr^{3+}} (f^{2})$", "$\\mathrm{LaCl_{3}}$"),
    "Pr3+:LaF3": ("Carnall.1968", "$\\mathrm{Pr^{3+}} (f^{2})$", "$\\mathrm{LaF_{3}}$"),
    "Pr3+:LaF3/alt": ("Carnall.1969", "$\\mathrm{Pr^{3+}} (f^{2})$", "$\\mathrm{LaF_{3}}$"),
    "Pr3+:LaF3/ext": ("Carnall.1970", "$\\mathrm{Pr^{3+}} (f^{2})$", "$\\mathrm{LaF_{3}}$"),
    "Pr3+:aq": ("Carnall.1968", "$\\mathrm{Pr^{3+}} (f^{2})$", "aq"),
    "Pr3+:free": ("Carnall.1968", "$\\mathrm{Pr^{3+}} (f^{2})$", "free ion"),
    "Nd3+:LaCl3": ("Carnall.1968", "$\\mathrm{Nd^{3+}} (f^{3})$", "$\\mathrm{LaCl_{3}}$"),
    "Nd3+:aq": ("Carnall.1968", "$\\mathrm{Nd^{3+}} (f^{3})$", "aq"),
    "Pm3+:aq": ("Carnall.1968", "$\\mathrm{Pm^{3+}} (f^{4})$", "aq"),
    "Sm3+:LaCl3": ("Carnall.1968", "$\\mathrm{Sm^{3+}} (f^{5})$", "$\\mathrm{LaCl_{3}}$"),
    "Sm3+:aq": ("Carnall.1968", "$\\mathrm{Sm^{3+}} (f^{5})$", "aq"),
    "Eu3+:LaCl3": ("Carnall.1968d", "$\\mathrm{Eu^{3+}} (f^{6})$", "$\\mathrm{LaCl_{3}}$"),
    "Eu3+:aq": ("Carnall.1968d", "$\\mathrm{Eu^{3+}} (f^{6})$", "aq"),
    "Gd3+:GdCl3": ("Carnall.1968b", "$\\mathrm{Gd^{3+}} (f^{7})$", "$\\mathrm{GdCl_{3}}$"),
    "Gd3+:aq": ("Carnall.1968b", "$\\mathrm{Gd^{3+}} (f^{7})$", "aq"),
    "Tb3+:LaCl3": ("Carnall.1968c", "$\\mathrm{Tb^{3+}} (f^{8})$", "$\\mathrm{LaCl_{3}}$"),
    "Tb3+:aq": ("Carnall.1968c", "$\\mathrm{Tb^{3+}} (f^{8})$", "aq"),
    "Dy3+:LaCl3": ("Carnall.1968", "$\\mathrm{Dy^{3+}} (f^{9})$", "$\\mathrm{LaCl_{3}}$"),
    "Dy3+:aq": ("Carnall.1968", "$\\mathrm{Dy^{3+}} (f^{9})$", "aq"),
    "Ho3+:LaCl3": ("Carnall.1968", "$\\mathrm{Ho^{3+}} (f^{10})$", "$\\mathrm{LaCl_{3}}$"),
    "Ho3+:aq": ("Carnall.1968", "$\\mathrm{Ho^{3+}} (f^{10})$", "aq"),
    "Er3+:LaCl3": ("Carnall.1968", "$\\mathrm{Er^{3+}} (f^{11})$", "$\\mathrm{LaCl_{3}}$"),
    "Er3+:LaF3": ("Carnall.1968", "$\\mathrm{Er^{3+}} (f^{11})$", "$\\mathrm{LaF_{3}}$"),
    "Er3+:aq": ("Carnall.1968", "$\\mathrm{Er^{3+}} (f^{11})$", "aq"),
    "Er3+:free": ("Carnall.1968", "$\\mathrm{Er^{3+}} (f^{11})$", "free"),
    "Tm3+:C2H5SO4": ("Carnall.1968", "$\\mathrm{Tm^{3+}} (f^{12})$", "$\\mathrm{C_{2}H_{5}SO_{4}}$"),
    "Tm3+:LaF3": ("Carnall.1970", "$\\mathrm{Tm^{3+}} (f^{12})$", "$\\mathrm{LaF_{3}}$"),
    "Tm3+:LaF3/ext": ("Carnall.1970", "$\\mathrm{Tm^{3+}} (f^{12})$", "$\\mathrm{LaF_{3}}$"),
    "Tm3+:aq": ("Carnall.1968", "$\\mathrm{Tm^{3+}} (f^{12})$", "aq"),
}


class Database:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.connection = None

        assert self.db_path.exists()

    def __enter__(self):
        db_uri = f"{self.db_path.absolute().as_uri()}?mode=ro"
        self.connection = sqlite3.connect(db_uri, uri=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    def names(self, table):
        """ Returns all names in the given table sorted by 'num' and then 'name'. """

        query = f"SELECT name FROM {table} GROUP BY name ORDER BY num ASC, name ASC"
        cursor = self.connection.execute(query)
        return [row[0] for row in cursor.fetchall()]

    def values(self, tables, name):
        if isinstance(tables, str):
            tables = [tables]
        dataset = []
        for table in tables:
            query = f"SELECT ref_value-calc_value FROM {table} WHERE name = ?"
            cursor = self.connection.execute(query, (name,))
            dataset += [row[0] for row in cursor.fetchall()]
        return dataset

    def stats(self, table, name):
        data = self.values(table, name)
        k2_stat, p_value = stats.normaltest(data)
        return k2_stat, p_value

    def count(self, table, name):
        query = f"SELECT COUNT(*) FROM {table} WHERE name = ?"
        cursor = self.connection.execute(query, (name,))
        return cursor.fetchone()[0]


def histogram(dataset, resolution, num_bins):
    half_range = (resolution * num_bins) / 2
    lower_bound = -half_range
    upper_bound = half_range

    bins = [0] * num_bins
    outliers = 0

    dataset = np.array(dataset)
    dataset -= np.median(dataset)
    for value in dataset:
        if value < lower_bound or value >= upper_bound:
            outliers += 1
        else:
            index = int((value - lower_bound) // resolution)
            if index == num_bins:
                index -= 1
            bins[index] += 1

    size = len(dataset)
    bins = [count / size for count in bins] if size > 0 else bins
    return bins, outliers


def table_row(name, size, outliers, hist):
    ion = REFS[name][1]
    host = REFS[name][2]

    image = ", ".join(f"{value:.3f}" for value in hist)
    image = f"\\drawhistogram{{{{{image}}}}}"

    cite = f"\\onlinecite{{{REFS[name][0]}}}"
    operators = {OPERATORS[key] for key in RADIAL[name]["radial"].keys()}
    if 6 in operators:
        cite = f"\\onlinecite{{{REFS[name][0]}}}$^\\ast$"

    return f"{ion} & {host} & {size} & {image} & {outliers} & {cite} \\\\"


def store_table(name, rows):
    col_name = "Levels" if name == "energy" else "Values"
    lines = []
    lines.append("  \\begin{tabular}{|l|l|c|c|c|c|}")
    lines.append("    \\hline")
    lines.append(f"    Ion & Host & {col_name} & Deviation & Outliers & Ref. \\\\")
    lines.append("    \\hline")
    last_num = 0
    for row in rows:
        num = int(re.search(r'\(f\^{(\d+)}\)', row).group(1))
        if name == "energy" and num != last_num:
            lines.append("    \\hline")
        last_num = num
        lines.append("    " + row)
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")

    with open(f"{name}.tex", "w") as fp:
        fp.write("\n".join(lines))


if __name__ == "__main__":
    resolution = 1.0
    rows = []
    with Database(DB_PATH) as db:
        for name in db.names("energy"):
            dataset = db.values("energy", name)
            hist, outliers = histogram(dataset, resolution, 7)
            long = f"{name} [{outliers}]" if outliers else name
            rows.append(table_row(name, len(dataset), outliers, hist))
    store_table("energy", rows)

    resolution = 0.0001
    rows = []
    with Database(DB_PATH) as db:
        for name in db.names("u2"):
            dataset = db.values(["u2", "u4", "u6"], name)
            hist, outliers = histogram(dataset, resolution, 7)
            long = f"{name} [{outliers}]" if outliers else name
            rows.append(table_row(name, len(dataset), outliers, hist))
    store_table("reduced", rows)
