import unittest
import numpy as np

class TestRCCSD_MULModule(unittest.TestCase):

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

    def test_rccsd_mul(self):
        from tcc.cc_solvers import classic_solver, residual_diis_solver
        from tcc.rccsd_mul import RCCSD_MUL

        cc = RCCSD_MUL(self.rhf)
        converged, energy, amps = classic_solver(cc)

        self.assertEqual(converged, True)

        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)
        
        converged, energy1, _ = residual_diis_solver(cc, amps=amps,
                                                     conv_tol_energy=1e-10)
        
        self.assertEqual(np.allclose(energy, -0.2133432609672395, 1e-5), True)
