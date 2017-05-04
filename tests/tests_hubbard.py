import unittest
import numpy as np


class TestHubbard(unittest.TestCase):

    def setUp(self):
        from pyscf import scf
        from tcc.hubbard import hubbard_from_scf

        rhf = hubbard_from_scf(scf.RHF, 6, 6, 3, 'y')
        rhf.damp = -4.0
        rhf.scf()
        self.assertEqual(np.allclose(rhf.e_tot, -3.5, 1e-5), True)
        self.rhf = rhf

    def test_cc_hubbard(self):
        from tcc.cc_solvers import residual_diis_solver
        from tcc.rccsd import RCCSD_UNIT

        cc = RCCSD_UNIT(self.rhf)
        converged, energy, _ = residual_diis_solver(
            cc, ndiis=5, conv_tol_res=1e-6, lam=5,
            max_cycle=100)
        self.assertEqual(np.allclose(energy, -0.93834495800469087, 1e-6), True)
