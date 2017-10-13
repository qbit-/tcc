import numpy as np
from numpy import einsum
from tcc.tensors import Tensors
from tcc.cc_solvers import CC
from tcc.denom import cc_denom

from tcc.cpd import (cpd_initialize, cpd_rebuild,
                     als_dense, cpd_symmetrize)

from tcc._rccsd_cpd import (
    _rccsd_cpd_ls_t_calculate_energy,
    _rccsd_cpd_ls_t_calc_residuals,
    _rccsd_cpd_calc_r2dr2dx,
    _rccsd_cpd_ls_t_unf_calc_residuals,
    _rccsd_cpd_ls_t_unf_calculate_energy
)


class RCCSD_CPD_LS_T(CC):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes, where we calculate
    full residuals as in normal RCCSD, but taking advantage of the
    structure of T2. We then calculate CPD of full T2 as a single
    shot ALS.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """

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
        xs = cpd_initialize(t2_full.shape, self.rankt)
        xs = als_dense(xs, t2_full, max_cycle=100)

        names = ['x1', 'x2', 'x3', 'x4']

        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        return Tensors(t1=t1,
                       t2=Tensors(zip(names, xs)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = _rccsd_cpd_ls_t_calculate_energy(h, a)
        return energy

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        return _rccsd_cpd_ls_t_calc_residuals(h, a)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        t1 = g.t1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full')

        t2_full = (2 * g.t2 + g.t2.transpose([0, 1, 3, 2])
                   ) / (- 6) * cc_denom(h.f, 4, 'dir', 'full')

        xs = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                       t2_full, max_cycle=1)

        # xs = parafac(t2_full, rankt, n_iter_max=1, init='guess',
        #              guess=[2 * a.x1[:, :rankt], 2 * a.x2[:, :rankt],
        #                     2 * a.x3[:, :rankt], 2 * a.x4[:, :rankt]])

        # xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2) : ('ident',)})

        names = ['x1', 'x2', 'x3', 'x4']
        return Tensors(t1=t1, t2=Tensors(zip(names, xs)))

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        return Tensors(
            t1=r.t1 - 2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - 2 * (
                2 * cpd_rebuild((a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4))
                - cpd_rebuild((a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4)
                              ).transpose([0, 1, 3, 2])
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
        xs = cpd_initialize(t2_full.shape, self.rankt)
        t2_cpd = als_dense(xs, t2_full, max_cycle=100)

        names = ['x1', 'x2', 'x3', 'x4']
        return Tensors(t1=t1, t2=Tensors(zip(names, t2_cpd)))

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        full_resids = _rccsd_cpd_ls_t_calc_residuals(h, a)
        resids_t2 = _rccsd_cpd_calc_r2dr2dx(h, a, full_resids.t2)

        names = ['x1', 'x2', 'x3', 'x4']
        return Tensors(t1=full_resids.t1, t2=resids_t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = _rccsd_cpd_ls_t_calculate_energy(h, a)
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

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        rt1, rt2 = _rccsd_cpd_ls_t_calc_residuals(h, a)
        d2squared = cc_denom(h.f, 4, 'dir', 'full')**2
        rt2 = rt2 * d2squared
        resids = _rccsd_cpd_calc_r2dr2dx(h, a, rt2)

        names = ['x1', 'x2', 'x3', 'x4']
        return Tensors(t1=rt1, t2=Tensors(zip(names, resids)))


class RCCSD_CPD_LS_T_UNF(CC):
    """
    This implements RCCSD with nCPD decomposed amplitudes.
    We build nCPD factors to approximate amplitudes in the least squares sense.
    For the next iteration we build full T1 and T2 residuals,
    but use the structure of T2 and RI decomposed interaction. This
    results in an N^5 algorithm.
    """

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
        xs = cpd_initialize(t2_full.shape, self.rankt)
        xs = als_dense(xs, t2_full, max_cycle=100)
        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        names = ['x1', 'x2', 'x3', 'x4']

        return Tensors(t1=t1,
                       t2=Tensors(zip(names, xs)))

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        return _rccsd_cpd_ls_t_unf_calc_residuals(h, a)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = _rccsd_cpd_ls_t_unf_calculate_energy(h, a)
        return energy

    def calculate_update(self, h, a):
        """
        Calculate dt
        """

        xs = [a.t2[key] for key in sorted(a.t2.keys())]
        # symmetrize a before feeding into res
        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})
        names = ['x1', 'x2', 'x3', 'x4']
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(names, xs))))

        dt1 = r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        # Do one ALS step for each factor (with updates for every other)
        dt2 = als_dense(xs, r.t2 * (cc_denom(h.f, 4, 'dir', 'full')),
                        self.rankt, max_cycle=1)

        return Tensors(t1=dt1, t2=Tensors(zip(names, dt2)))

    def solve_amps(h, a):
        raise NotImplemented

    def update_rhs(h, a):
        raise NotImplemented

    def solve_amps(self, h, a, g):
        """
        Solve for new amplitudes using RHS and denominator
        """
        t1 = g.t1 / (- cc_denom(h.f, 2, 'dir', 'full'))
        t2_full = g.t2 / (- cc_denom(h.f, 4, 'dir', 'full'))

        xs = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                       t2_full, max_cycle=1)

        names = ['x1', 'x2', 'x3', 'x4']
        return Tensors(t1=t1, t2=Tensors(zip(names, xs)))

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - cpd_rebuild(
                (a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4)
            ) / cc_denom(h.f, 4, 'dir', 'full')
        )


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
    cc2 = RCCSD_CPD_LS_T_HUB(rhf, rankt=30)

    converged1, energy1, amps1 = classic_solver(
        cc1, lam=5, max_cycle=50)

    converged2, energy2, amps2 = classic_solver(
        cc2, lam=1, conv_tol_energy=1e-8, max_cycle=500)

    # converged3, energy3, amps3 = root_solver(
    #      cc2)   # does not work


def test_cpd_unf():
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
    from tcc.rccsd_cpd import RCCSD_CPD_LS_T_UNF
    from tcc.cc_solvers import (classic_solver,
                                step_solver)

    cc1 = RCCSD_MUL_RI(rhf)
    cc2 = RCCSD_CPD_LS_T_UNF(rhf, rankt=20)

    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-8,)
    # optimizer_kwargs=dict(alpha=1, beta=0),
    # use_optimizer='momentum', max_cycle=50)

    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-8,)
    # optimizer_kwargs=dict(alpha=1, beta=0.5),
    # use_optimizer='momentum', max_cycle=50)


if __name__ == '__main__':
    test_hubbard()
