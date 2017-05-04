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

        mol.basis = {'H': 'sto-3g',
                     'O': 'sto-3g', }
        mol.build(parse_arg=False)
        rhf = scf.RHF(mol)
        rhf.scf()  # -74.963063129719558
        self.rhf = rhf

    def test_rccsd_mul(self):
        from tcc.cc_solvers import classic_solver, residual_diis_solver
        from tcc.rccsd_mul import RCCSD_MUL

        cc = RCCSD_MUL(self.rhf)
        converged, energy, amps = classic_solver(cc)

        self.assertEqual(converged, True)
        self.assertEqual(np.allclose(energy,
                                     -0.049466312886728606, 1e-5), True)

        converged, energy1, _ = residual_diis_solver(cc, amps=amps,
                                                     conv_tol_energy=1e-10)
        self.assertEqual(np.allclose(
            energy, -0.049466312886728606, 1e-5), True)

    def test_rccsd_mul_ri(self):
        from pyscf import gto
        from pyscf import scf
        mol = gto.Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = {'H': 'sto-3g',
                     'O': 'sto-3g', }
        mol.build(parse_arg=False)
        rhfri = scf.density_fit(scf.RHF(mol))
        rhfri.scf()  # -74.961181409648674

        from tcc.cc_solvers import classic_solver
        from tcc.rccsd_mul import RCCSD_MUL_RI
        cc = RCCSD_MUL_RI(rhfri)

        converged, energy, _ = classic_solver(cc)

        self.assertEqual(np.allclose(
            energy, -0.049398255827842984, 1e-6), True)

    def test_rccsd_mul_hub_ri(self):
        from pyscf import scf
        from tcc.hubbard import hubbard_from_scf
        rhf = hubbard_from_scf(scf.RHF, 6, 6, 3, 'y')
        rhf.damp = -4.0
        rhf.scf()  # -3.5

        from tcc.cc_solvers import residual_diis_solver
        from tcc.rccsd_mul import RCCSD_MUL_RI_HUB
        cc = RCCSD_MUL_RI_HUB(rhf)
        converged, energy, _ = residual_diis_solver(
            cc, ndiis=5, conv_tol_res=1e-6, lam=5,
            max_cycle=100)

        self.assertEqual(np.allclose(
            energy, -0.93833803927253978, 1e-6), True)
