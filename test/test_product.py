##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from ameli import Config, Product

def test_product():
    num_electrons = 3
    config_name = f"f{num_electrons}"

    product = Product(config_name)
    product.load_product(2)
    product.load_product(3)

    assert product.config.num_electrons == num_electrons
    assert product.num_states == 364
    assert product.num_indices == {1: 6370, 2: 36400, 3: 66430}
    assert product.num_elements == {1: 7098, 2: 172536, 3: 2391480}