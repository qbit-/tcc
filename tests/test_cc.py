import unittest
from tcc.cc import CC
from tcc.cc import concreter
from tcc.cc import kernel
from numpy import finfo


class TestCCAbstractClass(unittest.TestCase):

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

        self.CC_dummy_class = concreter(CC)
        self.CC_dummy = self.CC_dummy_class(rhf)

    def test_trivial_cc_calculation(self):
        kernel(self.CC_dummy)
        self.assertEqual(self.CC_dummy.energy_corr, 0)
        self.assertEqual((self.CC_dummy.energy_tot -
                          self.CC_dummy._scf.energy_tot()) < finfo(float).eps * 1000,
                         True)

if __name__ == '__main__':
    unittest.main()
