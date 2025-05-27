#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
#

'''Analytic PP integrals for GTH/HGH PPs in velocity gauge.

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.gto import ft_ao
from pyscf.gto.mole import ATOM_OF

def get_gth_pp_velgauge(mol, A_over_c):
    return get_gth_pp_loc(mol) + get_gth_pp_nl_velgauge(mol, A_over_c)

def get_gth_pp_loc(mol):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore

    # Analytical integration for get_pp_loc_part1(cell).
    fakemol = pp_int.fake_cell_vloc(mol, 0)
    vpploc = 0
    if fakemol.nbas > 0:
        charges = fakemol.atom_charges()
        atmlst = fakemol._bas[:,ATOM_OF]
        v = incore.aux_e2(mol, fakemol, 'int3c2e', aosym='s2', comp=1)
        vpploc = np.einsum('...i,i->...', v, -charges[atmlst])

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    for cn in range(1, 5):
        fakemol = pp_int.fake_cell_vloc(mol, cn)
        if fakemol.nbas > 0:
            v = incore.aux_e2(mol, fakemol, intors[cn], aosym='s2', comp=1)
            vpploc += np.einsum('...i->...', v)

    if isinstance(vpploc, np.ndarray):
        vpploc = lib.unpack_tril(vpploc)

    return vpploc

def get_gth_pp_nl_velgauge(mol, A_over_c):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas
    ppnl_half = []
    intors = ('GTO_ft_ovlp', 'GTO_ft_r2_origi', 'GTO_ft_r4_origi')
    for i, intor in enumerate(intors):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(_ft_ao_cross(intor, fakemol, mol, Gv=A_over_c.reshape(1,3)))
        else:
            ppnl_half.append(None)
    fakemol._bas = _bas

    nao = mol.nao
    vppnl = np.zeros((nao, nao), dtype=np.complex128)
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao), dtype=np.complex128)
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        vppnl += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    return vppnl

def _ft_ao_cross(intor, fakemol, mol, Gv, q=np.zeros(3),
                comp=1):
    # Normally you only need one point in reciprocal space at a time.
    # (this corresponds to one value of the vector potential A)
    assert Gv.shape[0] == 1, "Gv must be a single vector"


    # make an auxiliary cell containing both the
    # original cell and the fakecell functions
    mol_conc_fakemol = gto.conc_mol(mol, fakemol)

    intor = mol_conc_fakemol._add_suffix(intor)
    nbas_conc = mol_conc_fakemol.nbas

    # This shls_slice selects all pairs of functions
    # with GTH projectors in the first index and
    # AO basis functions in the second index.
    shls_slice = (mol.nbas, nbas_conc, 0, mol.nbas)

    return ft_ao.ft_aopair(mol_conc_fakemol,
                            Gv,
                            q=q,
                            shls_slice=shls_slice,
                            aosym='s1',
                            intor=intor,
                            return_complex=True,
                            comp=comp)[0]
