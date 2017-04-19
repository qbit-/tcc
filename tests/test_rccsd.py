import unittest
import numpy as np


class TestCCSolversModule(unittest.TestCase):

    def setUp(self):
        from pyscf import gto
        from pyscf import scf

        mol = gto.Mole()
        mol.verbose = 5
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = {'H': 'cc-pvdz',
                     'O': 'cc-pvdz', }
        mol.build(parse_arg=False)
        rhf = scf.RHF(mol)
        rhf.scf()  # -76.02676567
        self.rhf = rhf

    def test_rccsd_unit(self):
        from tcc.cc_solvers import classic_solver
        from tcc.rccsd import RCCSD, RCCSD_UNIT

        cc1 = RCCSD(self.rhf)
        cc2 = RCCSD_UNIT(self.rhf)
        converged1, energy1, amps = classic_solver(cc1)
        converged2, energy2, _ = classic_solver(cc2, amps=amps, conv_tol_energy=1e-10)

        self.assertEqual(converged1, converged2)
        self.assertEqual(np.allclose(energy1, -0.2133432609672395, 1e-5), True)
        self.assertEqual(np.allclose(energy1, energy2, 1e-5), True)
