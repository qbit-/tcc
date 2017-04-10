import unittest
from tcc.cc import CC
from tcc.cc import concreter


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

        self.rhf = rhf
        self.CC_dummy_class = concreter(CC)
        self.CC_dummy = self.CC_dummy_class(rhf)

    def test_non_canonical_fock(self):
        from tcc.interaction import _calculate_noncanonical_fock
        import numpy

        fock_true = numpy.diag(self.rhf.mo_energy)
        fock_updated = _calculate_noncanonical_fock(self.rhf,
                                                    self.rhf.mo_coeff,
                                                    self.rhf.mo_occ)
        self.assertEqual(numpy.max(fock_true - fock_updated) < 1e-6,
                         True)

if __name__ == '__main__':
    unittest.main()
