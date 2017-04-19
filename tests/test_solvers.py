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

    def test_residual_solver(self):
        from tcc.cc_solvers import residual_diis_solver
        from tcc.rccsd import RCCSD

        cc = RCCSD(self.rhf)
        converged, energy, amps = residual_diis_solver(cc)
        self.assertEqual(converged, True)
        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)

        converged, energy, _ = residual_diis_solver(cc, amps, lam=3, conv_tol_energy=1e-10)
        self.assertEqual(converged, True)
        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)
        
    def test_classic_solver(self):
        from tcc.cc_solvers import classic_solver
        from tcc.rccsd import RCCSD

        cc = RCCSD(self.rhf)
        converged, energy, amps = classic_solver(cc)
        self.assertEqual(converged, True)
        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)

        converged, energy, _ = classic_solver(cc, amps, lam=3, conv_tol_energy=1e-10)
        self.assertEqual(converged, True)
        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)
        
if __name__ == '__main__':
    unittest.main()
