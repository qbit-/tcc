import numpy as np
from tcc.cc_solvers import CC
from tcc.denom import cc_denom

from tcc.tensors import Tensors


class UCCSD(CC):
    """
    This class implements classic UCCSD method
    with vvoo ordered amplitudes and Dirac ordered integrals
    """

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None):
        """
        Initialize RCCSD
        """
        # Simply copy some parameters from RHF calculation
        super().__init__(mf)

        # Initialize molecular orbitals

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        from tcc.mos import UHF_MOS
        self._mos = UHF_MOS(mo_coeff, mo_energy, mo_occ, frozen)

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_UHF_FULL_CORE_DIR
        return HAM_UHF_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        tensor
        """

def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g', }
    mol.build()
    uhf = scf.UHF(mol)
    uhf.scf()  # -76.0267656731

    from tcc.uccsd_dir import UCCSD
    cc = UCCSD(uhf)
    h = cc.create_ham()
