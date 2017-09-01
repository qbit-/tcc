import numpy as np
from numpy import einsum
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace
from tcc.cpd import cpd_initialize, cpd_rebuild
from tensorly.decomposition import parafac
from tensorly.kruskal import kruskal_to_tensor
from tcc._rccsd_cpd_ls import (calculate_energy_cpd, calc_residuals_cpd,
                               calc_r2dr2dx)
from tcc.cpd import cpd_symmetrize

class RCCSD_CPD_LS_T(CC):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes, where we calculate
    full residuals as in normal RCCSD, but taking advantage of the
    structure of T2. We then calculate CPD of T2 as a single
    shot ALS.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """
    types = SimpleNamespace()

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None, rankt=None):
        """
        Initialize RCCSD
        :param rankt: rank of the CPD decomposition of amplitudes
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

        # initialize sizes

        if rankt is None:
            self.rankt = np.min((self._mos.nocc, self._mos.nvir))
        else:
            self.rankt = rankt

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple(
            'RCCSD_AMPLITUDES',
            field_names=('t1',
                         'x1', 'x2', 'x3', 'x4'
                         ))

        self.types.RHS_TYPE = namedtuple(
            'RCCSD_RHS',
            field_names=('gt1',
                         'gt2'
                         ))

        self.types.RESIDUALS_TYPE = namedtuple(
            'RCCSD_RESIDUALS',
            field_names=('rt1',
                         'rt2'))

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD_CPD_LS_T'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = 2 * ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = 2 * v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        xs = parafac(t2_full, self.rankt)

        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2) : ('ident',)})

        return self.types.AMPLITUDES_TYPE(t1, *xs)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = calculate_energy_cpd(h, a)
        return energy

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        rt1, rt2 = calc_residuals_cpd(h, a)
        return self.types.RESIDUALS_TYPE(rt1, rt2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        t1 = g.gt1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full')

        t2_full = (2 * g.gt2 + g.gt2.transpose([0, 1, 3, 2])
                   ) / (- 6) * cc_denom(h.f, 4, 'dir', 'full')

        rankt = self.rankt
        xs = parafac(t2_full, rankt, n_iter_max=1, init='guess',
                     guess=[a.x1[:, :rankt], a.x2[:, :rankt],
                            a.x3[:, :rankt], a.x4[:, :rankt]])

        # xs = parafac(t2_full, rankt, n_iter_max=1, init='guess',
        #              guess=[2 * a.x1[:, :rankt], 2 * a.x2[:, :rankt],
        #                     2 * a.x3[:, :rankt], 2 * a.x4[:, :rankt]])

        # xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2) : ('ident',)})

        return self.types.AMPLITUDES_TYPE(t1, *xs)

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        return self.types.RHS_TYPE(
            gt1=r.rt1 - 2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            gt2=r.rt2 - 2 * (
                2 * kruskal_to_tensor((a.x1, a.x2, a.x3, a.x4))
                - kruskal_to_tensor((a.x1, a.x2, a.x3, a.x4)).transpose([0, 1, 3, 2])
            ) / cc_denom(h.f, 4, 'dir', 'full')
        )


class RCCSD_CPD_LS_T_HUB(RCCSD_CPD_LS_T):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes for Hubbard hamiltonian,
    where we calculate full residuals as in normal RCCSD,
    but taking advantage of the structure of T2.
    We then calculate CPD of T2 as a single
    shot ALS.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """
    @property
    def method_name(self):
        return 'RCCSD_CPD_LS_T_HUB'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


class RCCSD_CPD_LS_R2(CC):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes, where we attempt to
    minimize doubles residuals in a least squares sense with
    respect to factors in the CPD decomposition of T2.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """
    types = SimpleNamespace()

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None, rankt=None):
        """
        Initialize RCCSD
        :param rankt: rank of the CPD decomposition of amplitudes
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

        # initialize sizes

        if rankt is None:
            self.rankt = np.min((self._mos.nocc, self._mos.nvir))
        else:
            self.rankt = rankt

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple(
            'RCCSD_AMPLITUDES',
            field_names=('t1', 'x1', 'x2', 'x3', 'x4'
                         ))

        self.types.RESIDUALS_TYPE = namedtuple(
            'RCCSD_RESIDUALS',
            field_names=('rt1', 'rx1', 'rx2', 'rx3',
                         'rx4'
                         ))

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD_CPD_LS_R2'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = 2 * ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = 2 * v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t2_cpd = parafac(t2_full, self.rankt)

        return self.types.AMPLITUDES_TYPE(t1, *t2_cpd)

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        rt1, rt2 = calc_residuals_cpd(h, a)
        rx1, rx2, rx3, rx4 = calc_r2dr2dx(h, a, rt2)

        return self.types.RESIDUALS_TYPE(rt1, rx1, rx2,
                                         rx3, rx4)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = calculate_energy_cpd(h, a)
        return energy

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        raise NotImplementedError

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        raise NotImplementedError

class RCCSD_CPD_LS_R2_W(RCCSD_CPD_LS_R2):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes, where we attempt to
    minimize doubles residuals in a least squares sense with
    respect to factors in the CPD decomposition of T2.
    The difference is that we weight each residual with an
    inverse of the CC denominator
    D^-1_{abij} = 1 / (f_a + f_b - f_i - f_j)

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """
    @property
    def method_name(self):
        return 'RCCSD_CPD_LS_R2_W'

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        rt1, rt2 = calc_residuals_cpd(h, a)
        d2squared = cc_denom(h.f, 4, 'dir', 'full')**2
        rt2 = rt2 * d2squared
        rx1, rx2, rx3, rx4 = calc_r2dr2dx(h, a, rt2)

        return self.types.RESIDUALS_TYPE(rt1, rx1, rx2,
                                         rx3, rx4)

def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    from tcc.rccsd_mul import RCCSD_MUL_RI
    from tcc.rccsd_cpd import RCCSD_CPD_LS_T
    from tcc.rccsd_cpd import RCCSD_CPD_LS_R2
    from tcc.rccsd_cpd import RCCSD_CPD_LS_R2_W
    from tcc.cc_solvers import (residual_diis_solver,
                                classic_solver, root_solver)

    cc1 = RCCSD_CPD_LS_T(rhf, rankt=20)
    cc2 = RCCSD_MUL_RI(rhf)
    cc3 = RCCSD_CPD_LS_R2(rhf, rankt=20)
    cc4 = RCCSD_CPD_LS_R2_W(rhf, rankt=20)

    converged1, energy1, amps1 = classic_solver(
        cc1, max_cycle=150)

    converged2, energy2, amps2 = classic_solver(
        cc2, max_cycle=10)

    # converged3, energy3, amps3 = root_solver(
    #     cc3, amps=amps1)

    # converged4, energy4, amps4 = root_solver(
    #     cc4, amps=amps1)


def test_hubbard():   # pragma: nocover
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 6, 6, 1, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import (classic_solver, root_solver)
    from tcc.rccsd_mul import RCCSD_MUL_RI_HUB
    from tcc.rccsd_cpd import RCCSD_CPD_LS_T_HUB

    cc1 = RCCSD_MUL_RI_HUB(rhf)
    cc2 = RCCSD_CPD_LS_T_HUB(rhf, rankt=12)

    converged1, energy1, amps1 = classic_solver(
        cc1, lam=5, max_cycle=10)

    converged2, energy2, amps2 = classic_solver(
        cc2, lam=1, conv_tol_energy=1e-8, max_cycle=500)

    # converged3, energy3, amps3 = root_solver(
    #      cc2)   # does not work

if __name__ == '__main__':
    test_hubbard()
