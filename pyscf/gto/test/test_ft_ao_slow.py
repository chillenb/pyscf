#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.gto import ft_ao
from pyscf.dft import gen_grid

import pytest

libpbc = lib.load_library('libpbc')


@pytest.mark.parametrize('cart_orbs', [False, True])
@pytest.mark.parametrize('r_power', [2, 4])
@pytest.mark.parametrize('q', [np.array([0.1, 0.2, 0.3]), np.array([0., 0., 0.]), np.array([-2., 1., 0.5])])
def test_ft_aopair_r2(cart_orbs, r_power, q):
    suffix = '_cart' if cart_orbs else '_sph'
    mol = gto.M(atom='''
    C -1. 0. 0.
    C 1. 0. 0.
    ''', basis='cc-pvdz', cart=cart_orbs)
    nao = mol.nao

    # evaluate all AOs on a 3D grid.

    quad_grid = gen_grid.Grids(mol)
    quad_grid.level = 5
    quad_grid.build()
    coords = quad_grid.coords
    weights = quad_grid.weights

    ao_vals = mol.eval_gto('GTOval', coords, non0tab=quad_grid.non0tab)
    ng = ao_vals.shape[0]

    Gv = np.zeros((1,3))

    ft_intor_name = f'GTO_ft_r{r_power}_origi'

    ao_rn_exp_iqr = np.empty((ng, nao), dtype=np.complex128)

    ao_loc = mol.ao_loc

    for i in range(mol.nbas):
        center_i = mol.bas_coord(i)
        r2_from_i = np.sum((coords - center_i[None, :])**2, axis=1)
        rn_from_i = r2_from_i**(r_power/2)
        aoslice = slice(ao_loc[i], ao_loc[i+1])
        ao_rn_exp_iqr[:, aoslice] = (ao_vals[:, aoslice] * \
                                     np.exp(-1j * coords @ q).reshape(ng, 1)) \
                                     * rn_from_i[:, None]

    ref_ints = lib.einsum('gi, gj, g -> ij', ao_rn_exp_iqr, ao_vals, weights)

    ao_pair_ft = ft_ao.ft_aopair(mol, Gv=Gv, intor=ft_intor_name + suffix, q=q, comp=1).reshape(nao, nao)
    assert np.allclose(ao_pair_ft, ref_ints, atol=1e-8)



@pytest.mark.parametrize('cart_orbs', [False, True])
@pytest.mark.parametrize('r_power', [2, 4])
def test_ft_aopair_korigin(cart_orbs, r_power):
    suffix = '_cart' if cart_orbs else '_sph'

    ft_intor_name = f'GTO_ft_r{r_power}_origi' + suffix
    ref_intor_name = f'int1e_r{r_power}_origi' + suffix

    mol = gto.M(atom='''
    C -1. 0. 0.
    C 1. 0. 0.
    ''', basis='cc-pvqz', cart=cart_orbs)
    nao = mol.nao


    Gv = np.zeros((1,3))

    ref_ints = mol.intor(ref_intor_name, comp=1).reshape(nao, nao)

    ao_pair_ft = ft_ao.ft_aopair(mol, Gv=Gv, intor=ft_intor_name, q=Gv, comp=1).reshape(nao, nao)
    assert np.allclose(ao_pair_ft, ref_ints, atol=1e-8)

