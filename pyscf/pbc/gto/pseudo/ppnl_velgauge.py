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
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import ft_ao as pft_ao
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl

libpbc = lib.load_library('libpbc')


def get_pp_nl_velgauge(cell, A_over_c, kpts=None):
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)

    q = -A_over_c.reshape(1,3)

    ppnl_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, q)
    nao = cell.nao_nr()

    # ppnl_half could be complex, so _contract_ppnl will not work.
    # if gamma_point(kpts_lst):
    #     return _contract_ppnl(cell, fakecell, hl_blocks, ppnl_half, kpts=kpts)

    buf = np.empty((3*9*nao), dtype=np.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    ppnl = np.zeros((nkpts,nao,nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                offset[i] = p0 + nd
            ppnl[k] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)

    if kpts is None or np.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl

def get_gth_pp_nl_velgauge_commutator(cell, A_over_c, kpts=None, origin=(0,0,0)):
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)
    ppnl_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, A_over_c.reshape(1,3))
    ppnl_rc_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, A_over_c.reshape(1,3), intors=('GTO_ft_rc', 'GTO_ft_rc_r2_origi', 'GTO_ft_rc_r4_origi'), comp=3, origin=origin)

    nao = cell.nao_nr()

    # ppnl_half could be complex, so _contract_ppnl will not work.
    # if gamma_point(kpts_lst):
    #     return _contract_ppnl(cell, fakecell, hl_blocks, ppnl_half, kpts=kpts)

    buf = np.empty((3*9*nao), dtype=np.complex128)
    buf2 = np.empty((3*3*9*nao), dtype=np.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    vppnl_commutator = np.zeros((nkpts, 3, nao, nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = np.ndarray((hl_dim, nd, nao), dtype=np.complex128, buffer=buf)
            rc_ilp = np.ndarray((3, hl_dim, nd, nao), dtype=np.complex128, buffer=buf2)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                rc_ilp[:, i] = ppnl_rc_half[i][k][:, p0:p0+nd]
                offset[i] = p0 + nd
            vppnl_commutator[k] += np.einsum('xilp,ij,jlq->xpq', rc_ilp.conj(), hl, ilp)
            vppnl_commutator[k] -= np.einsum('ilp,ij,xjlq->xpq', ilp.conj(), hl, rc_ilp)
    if kpts is None or np.shape(kpts) == (3,):
        vppnl_commutator = vppnl_commutator[0]
    return vppnl_commutator

def get_gth_ppnl_rc(cell, A_over_c, kpts=None, origin=(0,0,0)):
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    fakecell, hl_blocks = fake_cell_vnl(cell)
    ppnl_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, A_over_c.reshape(1,3))
    ppnl_rc_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, A_over_c.reshape(1,3), intors=('GTO_ft_rc', 'GTO_ft_rc_r2_origi', 'GTO_ft_rc_r4_origi'), comp=3, origin=origin)
    return ppnl_half, ppnl_rc_half

# Modified version of _int_vnl in pyscf.pbc.gto.pseudo.pp_int
def _int_vnl_ft(cell, fakecell, hl_blocks, kpts, Gv, q=np.zeros(3),
                intors=None, comp=1, origin=(0,0,0)):
    if intors is None:
        intors = ['GTO_ft_ovlp', 'GTO_ft_r2_origi', 'GTO_ft_r4_origi']

    # Normally you only need one point in reciprocal space at a time.
    # (this corresponds to one value of the vector potential A)
    assert Gv.shape[0] == 1, "Gv must be a single vector"

    def int_ket(_bas, intor):
        if len(_bas) == 0:
            return []

        # make a copy of the fakecell including only the
        # basis functions that were passed in.
        fakecell_trunc = fakecell.copy(deep=True)
        fakecell_trunc._bas = _bas

        # make an auxiliary cell containing both the
        # original cell and the fakecell functions
        cell_conc_fakecell = pgto.conc_cell(cell, fakecell_trunc)

        intor = cell_conc_fakecell._add_suffix(intor)
        nbas_conc = cell_conc_fakecell.nbas

        # This shls_slice selects all pairs of functions
        # with GTH projectors in the first index and
        # AO basis functions in the second index.
        shls_slice = (cell.nbas, nbas_conc, 0, cell.nbas)

        with cell_conc_fakecell.with_common_origin(origin):
            retv = pft_ao.ft_aopair_kpts(cell_conc_fakecell,
                                Gv,
                                q=q,
                                shls_slice=shls_slice,
                                aosym='s1',
                                intor=intor,
                                comp=comp,
                                kptjs=kpts)
        # Gv is a single vector
        if comp == 1:
            retv = retv[:, 0]
        else:
            retv = retv[:, :, 0]
        return retv

    hl_dims = np.asarray([len(hl) for hl in hl_blocks])
    out = (int_ket(fakecell._bas[hl_dims>0], intors[0]),
           int_ket(fakecell._bas[hl_dims>1], intors[1]),
           int_ket(fakecell._bas[hl_dims>2], intors[2]))
    return out
