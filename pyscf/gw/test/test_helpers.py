import unittest
import numpy as np

from pyscf.gw import helpers
from pyscf import lib

def setUpModule():
    pass

def tearDownModule():
    pass

def get_rho_response_old(omega, mo_energy, Lpq):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    eia = eia/(omega**2+eia*eia)
    Pia = lib.einsum('Pia,ia->Pia',Lpq,eia)
    # Response from both spin-up and spin-down density
    Pi = 4. * lib.einsum('Pia,Qia->PQ',Pia,Lpq)

    return Pi

class KnownValues(unittest.TestCase):
    def test_dmul_Lia_response(self):
        Lia = np.random.random((3, 10, 15))
        _, nocc, nvir = Lia.shape
        mo_energy = np.random.random(nocc + nvir)
        alpha = 0.1
        omega = 0.2
        Pia = helpers.dmul_Lia_response(Lia, mo_energy, omega, alpha=alpha,
                                       nocc_range=(3, 5))

        eia = mo_energy[3:5, None] - mo_energy[None, nocc:]
        expected = alpha * Lia[:, 3:5] * eia[None, ...] / (eia[None, ...]**2 + omega**2)
        self.assertAlmostEqual(np.linalg.norm(Pia - expected), 0.0, 12)

    def test_copy_Lia_nocc_slice(self):
        Lia = np.random.random((3, 4, 5))
        nocc_range = (1, 3)
        Pia = helpers.copy_Lia_nocc_slice(Lia, nocc_range=nocc_range)
        self.assertAlmostEqual(np.linalg.norm(Pia - Lia[:, 1:3, :]), 0.0, 12)

        Lia = Lia * 1j
        Pia = helpers.copy_Lia_nocc_slice(Lia, nocc_range=nocc_range, conj=True)
        self.assertAlmostEqual(np.linalg.norm(Pia.conj() - Lia[:, 1:3, :]), 0.0, 12)

    def test_rho_response_restricted(self):
        omega = 0.1
        mo_energy = np.random.random(9)
        Lia = np.random.random((3, 4, 5))
        Pi = helpers.rho_response_restricted(omega, mo_energy, Lia)
        Pi_ref = get_rho_response_old(omega, mo_energy, Lia)
        self.assertAlmostEqual(np.linalg.norm(Pi - Pi_ref), 0.0, 12)