import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from collections import namedtuple
from types import SimpleNamespace


from tcc._rccsdt import gen_energy, gen_residuals
class RCCSDT(CC):
    """
    This class implements classic RCCSDT method with
    vvvooo ordered amplitudes and Dirac ordered integrals
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays

    types = SimpleNamespace()

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

        from tcc.mos import SPINLESS_MOS
        self._mos = SPINLESS_MOS(mo_coeff, mo_energy, mo_occ, frozen)

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple('RCCSD_AMPLITUDES_FULL',
                                                field_names=('t1', 't2', 't3'))
        self.types.RHS_TYPE = namedtuple('RCCSD_RHS_FULL',
                                         field_names=('g1', 'g2', 'g3'))
        self.types.RESIDUALS_TYPE = namedtuple('RCCSD_RESIDUALS_FULL',
                                               field_names=('r1', 'r2', 'r3'))

    @property
    def method_name(self):
        return 'RCCSDT'

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_DIR
        return HAM_SPINLESS_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)
        t3 = np.zeros((nvir,) * 3 + (nocc,) * 3)

        return self.types.AMPLITUDES_TYPE(t1, t2, t3)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return gen_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """


        return self.types.RESIDUALS_TYPE(*gen_residuals(h, a))

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return self.types.RHS_TYPE(
            g1=r.r1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            g2=r.r2 - a.t2 / cc_denom(h.f, 4, 'dir', 'full'),
            g3=r.r3 - (a.t3 - a.t3.transpose([2, 1, 0, 3, 4, 5])) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        O3 = g.g3 / (- cc_denom(h.f, 6, 'dir', 'full'))
        O3 = 1 / 6 * (O3
                      - O3.transpose([1, 0, 2, 3, 4, 5])
                      + O3.transpose([1, 2, 0, 3, 4, 5])
                      - O3.transpose([2, 1, 0, 3, 4, 5])
                      + O3.transpose([2, 0, 1, 3, 4, 5])
                      - O3.transpose([0, 2, 1, 3, 4, 5])
        )
        O3 = 1 / 6 * (O3
                      - O3.transpose([0, 1, 2, 4, 3, 5])
                      + O3.transpose([0, 1, 2, 4, 5, 3])
                      - O3.transpose([0, 1, 2, 5, 4, 3])
                      + O3.transpose([0, 1, 2, 5, 3, 4])
                      - O3.transpose([0, 1, 2, 3, 5, 4])
        )

        return self.types.AMPLITUDES_TYPE(
            g[0] * (- cc_denom(h.f, g[0].ndim, 'dir', 'full')),
            g[1] * (- cc_denom(h.f, g[1].ndim, 'dir', 'full')),
            1 / 2 * O3
        )



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
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import root_solver, classic_solver
    from tcc.rccsdt import RCCSDT
    cc = RCCSDT(rhf)
#    converged1, energy1, _ = root_solver(cc)
    converged2, energy2, _ = classic_solver(cc, lam=10)


if __name__ == '__main__':
    test_cc()
