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

from pyscf import dft
from pyscf import gto
from pyscf import scf
from pyscf.scf import gcscf
from pyscf.scf.smearing import smearing_


def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
        O  0.000000  0.000000  0.000000
        H  0.000000 -0.757000  0.587000
        H  0.000000  0.757000  0.587000
    '''
    mol.basis = 'cc-pvdz'
    mol.build()


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def _run_gcscf(mf, sigma=.01, mu0=-.2, fix_spin=False):
    mf = gcscf.gcscf(mf, sigma=sigma, mu0=mu0, fix_spin=fix_spin)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.verbose = 0
    mf.kernel()
    return mf


def _run_smearing_reference(mf, sigma=.01, mu0=-.2, fix_spin=False):
    mf = smearing_(mf, sigma=sigma, method='fermi', mu0=mu0,
                   fix_spin=fix_spin)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.verbose = 0
    mf.kernel()
    return mf


class KnownValues(unittest.TestCase):

    def test_rhf_gcscf_fixed_nelectron_default(self):
        mf = _run_gcscf(scf.RHF(mol), mu0=None)
        mf_ref = _run_smearing_reference(scf.RHF(mol), mu0=None)

        self.assertIsNone(mf.mu0)
        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 7)
        self.assertAlmostEqual(mf.nelectron, mol.nelectron, 7)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-7)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_rhf_gcscf(self):
        mf = _run_gcscf(scf.RHF(mol))
        mf_ref = _run_smearing_reference(scf.RHF(mol))
        e_grand_ref = mf_ref.e_free - mf.mu0 * mf_ref.mo_occ.sum()

        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 7)
        self.assertAlmostEqual(mf.e_grand, e_grand_ref, 7)
        self.assertLess(abs(mf.nelectron - mf_ref.mo_occ.sum()), 1e-6)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-7)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)
        self.assertGreater(mf.n_haux_eval, 0)

    def test_rks_gcscf(self):
        mf0 = dft.RKS(mol)
        mf0.xc = 'pbe0'
        mf = _run_gcscf(mf0)

        mf_ref = dft.RKS(mol)
        mf_ref.xc = 'pbe0'
        mf_ref = _run_smearing_reference(mf_ref)
        e_grand_ref = mf_ref.e_free - mf.mu0 * mf_ref.mo_occ.sum()

        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 6)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 6)
        self.assertAlmostEqual(mf.e_grand, e_grand_ref, 6)
        self.assertLess(abs(mf.nelectron - mf_ref.mo_occ.sum()), 1e-6)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-6)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_uhf_gcscf(self):
        mf = _run_gcscf(scf.UHF(mol))
        mf_ref = _run_smearing_reference(scf.UHF(mol))
        e_grand_ref = mf_ref.e_free - mf.mu0 * mf_ref.mo_occ.sum()

        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 7)
        self.assertAlmostEqual(mf.e_grand, e_grand_ref, 7)
        self.assertLess(abs(mf.nelectron - mf_ref.mo_occ.sum()), 1e-6)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-7)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_uhf_gcscf_fixed_nelectron_default(self):
        mf = _run_gcscf(scf.UHF(mol), mu0=None)
        mf_ref = _run_smearing_reference(scf.UHF(mol), mu0=None)

        self.assertIsNone(mf.mu0)
        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 7)
        self.assertAlmostEqual(mf.nelectron, mol.nelectron, 7)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-7)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_uhf_gcscf_fix_spin(self):
        mf = _run_gcscf(scf.UHF(mol), mu0=None, fix_spin=True)
        mf_ref = _run_smearing_reference(scf.UHF(mol), mu0=None,
                                         fix_spin=True)

        self.assertTrue(mf.fix_spin)
        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 7)
        self.assertLess(abs(mf.mo_occ[0].sum() - mf.nelec[0]), 1e-7)
        self.assertLess(abs(mf.mo_occ[1].sum() - mf.nelec[1]), 1e-7)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-7)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_uks_gcscf(self):
        mf0 = dft.UKS(mol)
        mf0.xc = 'pbe0'
        mf = _run_gcscf(mf0)

        mf_ref = dft.UKS(mol)
        mf_ref.xc = 'pbe0'
        mf_ref = _run_smearing_reference(mf_ref)
        e_grand_ref = mf_ref.e_free - mf.mu0 * mf_ref.mo_occ.sum()

        self.assertTrue(mf.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 6)
        self.assertAlmostEqual(mf.e_free, mf_ref.e_free, 6)
        self.assertAlmostEqual(mf.e_grand, e_grand_ref, 6)
        self.assertLess(abs(mf.nelectron - mf_ref.mo_occ.sum()), 1e-6)
        self.assertLess(abs(mf.entropy - mf_ref.entropy), 1e-6)
        self.assertLess(abs(mf.mo_occ - mf_ref.mo_occ).max(), 1e-6)
        self.assertLess(mf.auxh_residual_norm, mf.conv_tol_grad)

    def test_gcscf_decorator(self):
        mf = scf.RHF(mol)
        self.assertIs(gcscf.gcscf_(mf, sigma=.1, mu0=-.2), mf)
        self.assertTrue(isinstance(mf, gcscf._GCSCF))
        self.assertEqual(mf.sigma, .1)
        self.assertEqual(mf.mu0, -.2)

        mf1 = mf.undo_gcscf()
        self.assertFalse(isinstance(mf1, gcscf._GCSCF))
        self.assertFalse(hasattr(mf1, 'sigma'))
        self.assertFalse(hasattr(mf1, 'mu0'))

    def test_unsupported_rohf(self):
        with self.assertRaises(NotImplementedError):
            gcscf.gcscf(scf.ROHF(mol), sigma=.1, mu0=-.2)


if __name__ == "__main__":
    print("Full Tests for GC-SCF")
    unittest.main()
