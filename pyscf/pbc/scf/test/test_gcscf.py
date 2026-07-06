#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import unittest

import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from pyscf.pbc import dft
from pyscf.pbc.scf import gcscf
from pyscf.pbc.scf import smearing


SIGMA = .1


def setUpModule():
    global cell, kpts
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [.5, 1.]]]
    cell.a = numpy.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()
    kpts = cell.make_kpts([2, 1, 1])


def tearDownModule():
    global cell, kpts
    cell.stdout.close()
    del cell, kpts


def _set_common_flags(mf):
    mf.verbose = 0
    mf.conv_tol = 1e-11
    return mf


def _run_reference(mf, **kwargs):
    mf = smearing.smearing_(mf, sigma=SIGMA, method='fermi', **kwargs)
    return _set_common_flags(mf).run()


def _run_gcscf(mf, **kwargs):
    mf = gcscf.gcscf(mf, sigma=SIGMA, **kwargs)
    mf.conv_tol_grad = 1e-10
    return _set_common_flags(mf).run()


class KnownValues(unittest.TestCase):
    def test_krhf_matches_smearing(self):
        ref = _run_reference(pscf.KRHF(cell, kpts=kpts))
        mf = _run_gcscf(pscf.KRHF(cell, kpts=kpts))

        self.assertTrue(ref.converged)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 10)
        self.assertAlmostEqual(mf.e_free, ref.e_free, 10)
        self.assertAlmostEqual(mf.entropy, ref.entropy, 9)
        self.assertAlmostEqual(mf.nelectron, cell.nelectron, 10)
        self.assertGreater(mf.n_haux_eval, 1)
        self.assertLess(mf.auxh_residual_norm, 1e-9)

    def test_krhf_fixed_mu0_matches_smearing(self):
        mu0 = .3
        ref = _run_reference(pscf.KRHF(cell, kpts=kpts), mu0=mu0)
        mf = _run_gcscf(pscf.KRHF(cell, kpts=kpts), mu0=mu0)
        nelectron = numpy.sum(ref.mo_occ) / len(kpts)
        e_grand = ref.e_free - mu0 * nelectron

        self.assertTrue(ref.converged)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 10)
        self.assertAlmostEqual(mf.e_free, ref.e_free, 10)
        self.assertAlmostEqual(mf.e_grand, e_grand, 10)
        self.assertAlmostEqual(mf.nelectron, nelectron, 10)
        self.assertGreater(mf.n_haux_eval, 1)
        self.assertLess(mf.auxh_residual_norm, 1e-9)

    def test_kuhf_fix_spin_matches_smearing(self):
        ref = _run_reference(pscf.KUHF(cell, kpts=kpts), fix_spin=True)
        mf = _run_gcscf(pscf.KUHF(cell, kpts=kpts), fix_spin=True)
        nelectron = numpy.sum(ref.mo_occ) / len(kpts)

        self.assertTrue(ref.converged)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 10)
        self.assertAlmostEqual(mf.e_free, ref.e_free, 10)
        self.assertAlmostEqual(mf.entropy, ref.entropy, 9)
        self.assertAlmostEqual(mf.nelectron, nelectron, 10)
        self.assertEqual(numpy.asarray(mf.mu).shape, (2,))
        self.assertGreater(mf.n_haux_eval, 1)
        self.assertLess(mf.auxh_residual_norm, 1e-9)

    def test_krks_matches_smearing(self):
        gamma = cell.make_kpts([1, 1, 1])
        ref = smearing.smearing_(
            dft.KRKS(cell, kpts=gamma), sigma=SIGMA, method='fermi')
        ref.xc = 'lda,vwn'
        ref = _set_common_flags(ref).run()

        mf = gcscf.gcscf(dft.KRKS(cell, kpts=gamma), sigma=SIGMA)
        mf.xc = 'lda,vwn'
        mf.conv_tol_grad = 1e-10
        mf = _set_common_flags(mf).run()

        self.assertTrue(ref.converged)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 10)
        self.assertAlmostEqual(mf.e_free, ref.e_free, 10)
        self.assertAlmostEqual(mf.entropy, ref.entropy, 9)
        self.assertGreater(mf.n_haux_eval, 1)
        self.assertLess(mf.auxh_residual_norm, 1e-9)


if __name__ == "__main__":
    print("Full Tests for PBC GC-SCF")
    unittest.main()
