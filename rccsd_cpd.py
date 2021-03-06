import numpy as np
from numpy import einsum
from tcc.tensors import Tensors
from tcc.cc_solvers import CC
from tcc.denom import cc_denom

from tcc.cpd import (cpd_initialize, cpd_rebuild,
                     ncpd_initialize, ncpd_rebuild,
                     als_dense, als_cpd, cpd_symmetrize,
                     ncpd_symmetrize, als_contract_cpd,
                     als_contract_dense, als_pseudo_inverse)

from tcc._rccsd_cpd import (
    _rccsd_cpd_ls_t_calculate_energy,
    _rccsd_cpd_ls_t_calc_residuals,
)

from tcc._rccsd_cpd import (
    _rccsd_cpd_ls_t_unf_calc_residuals,
    _rccsd_cpd_ls_t_unf_calculate_energy
)

from tcc._rccsd_cpd import (
    _rccsd_ncpd_ls_t_unf_calc_residuals,
    _rccsd_ncpd_ls_t_unf_calculate_energy
)

from tcc._rccsd_cpd import (
    _rccsd_cpd_ls_t_true_calculate_energy,
    _rccsd_cpd_ls_t_true_calc_r1,
    _rccsd_cpd_ls_t_true_calc_r21,
    _rccsd_cpd_ls_t_true_calc_r22,
    _rccsd_cpd_ls_t_true_calc_r23,
    _rccsd_cpd_ls_t_true_calc_r24,
)

from tcc._rccsd import (
    _rccsd_ri_calculate_energy,
    _rccsd_ri_calc_residuals
)


class RCCSD_CPD_LS_T_UNIT(CC):
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
            n = np.min((self._mos.nocc, self._mos.nvir))
            self.rankt = Tensors(t2=n)
        else:
            self.rankt = Tensors(rankt)

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
        xs = cpd_initialize(t2_full.shape, self.rankt.t2)
        xs = als_dense(xs, t2_full, max_cycle=100)

        names = ['x1', 'x2', 'x3', 'x4']

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
        # Symmetrize T2 before feeding into res
        names = ['x1', 'x2', 'x3', 'x4']
        xs_sym = cpd_symmetrize(
            [a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
            {(1, 0, 3, 2): ('ident',)})

        return _rccsd_cpd_ls_t_calc_residuals(
            h, Tensors(t1=a.t1,
                       t2=Tensors(zip(names, xs_sym))))

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        t1 = g.t1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full')

        # Build unit RHS
        g2 = (2 * g.t2 + g.t2.transpose([0, 1, 3, 2])
              ) / 6

        # Symmetrize RHS
        g2 = 1 / 2 * (g2 + g2.transpose([1, 0, 3, 2]))

        # Solve
        t2_full = g2 * (- cc_denom(h.f, 4, 'dir', 'full')
                        )
        xs = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                       t2_full, max_cycle=1)

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


class RCCSD_nCPD_LS_T(CC):
    """
    This implements RCCSD with nCPD decomposed amplitudes.
    We build nCPD factors to approximate amplitudes in
    the least squares sense.
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
            n = np.min((self._mos.nocc, self._mos.nvir))
            self.rankt = Tensors(t2=n)
        else:
            self.rankt = Tensors(rankt)

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

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        xs = ncpd_initialize(t2_full.shape, self.rankt.t2)
        xs = als_dense(xs, t2_full, max_cycle=100, tensor_format='ncpd')
        # xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        names = ['xlam', 'x1', 'x2', 'x3', 'x4']

        return Tensors(t1=t1,
                       t2=Tensors(zip(names, xs)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        return _rccsd_ncpd_ls_t_unf_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """

        # Symmetrize T2 before feeding into res
        xs_sym = ncpd_symmetrize(
            [a.t2.xlam, a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
            {(1, 0, 3, 2): ('ident',)})

        amps = Tensors(
            t1=a.t1,
            t2=ncpd_rebuild(xs_sym))

        # residuals are calculated with full amps for speed
        return _rccsd_ri_calc_residuals(h, amps)
        # return _rccsd_ncpd_ls_t_unf_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - ncpd_rebuild(
                (a.t2.xlam, a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4)
            ) / cc_denom(h.f, 4, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solve for new amplitudes using RHS and denominator
        """

        t1 = g.t1 * (- cc_denom(h.f, 2, 'dir', 'full'))

        # Symmetrize T2 RHS
        g2 = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))

        t2_full = g2 * (- cc_denom(h.f, 4, 'dir', 'full'))

        xs = als_dense([a.t2.xlam, a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                       t2_full, max_cycle=1, tensor_format='ncpd')

        return Tensors(t1=t1, t2=Tensors(xlam=xs[0], x1=xs[1],
                                         x2=xs[2], x3=xs[3], x4=xs[4]))

    def calculate_gradient(self, h, a):
        """
        Calculate dt
        """
        raise NotImplemented('RCCSD-nCPD can not be calculated'
                             ' as gradient update')

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        names_abij = ['xlam', 'x1', 'x2', 'x3', 'x4']
        xs = [a.t2[key] for key in names_abij]

        # symmetrize t2 before feeding into res
        xs_sym = ncpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        # Calculate residuals
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(names_abij, xs_sym))))

        # Calculate T1
        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        # Symmetrize T2 residuals
        r2 = 1 / 2 * (r.t2 + r.t2.transpose([1, 0, 3, 2]))

        # Solve
        r2_d = - r2 * cc_denom(h.f, 4, 'dir', 'full')

        t2 = [f for f in xs]
        for idx in range(len(t2)):
            g = (als_contract_dense(t2, r2_d, idx,
                                    tensor_format='ncpd')
                 + als_contract_cpd(t2, xs_sym, idx,
                                    tensor_format='ncpd'))
            s = als_pseudo_inverse(t2, t2, idx)
            f = np.dot(g, s)
            t2[idx] = f

        return Tensors(t1=t1, t2=Tensors(zip(names_abij, t2)))


class RCCSD_nCPD_LS_T_HUB(RCCSD_nCPD_LS_T):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes for Hubbard hamiltonian,
    where we calculate full residuals as in normal RCCSD,
    but taking advantage of the structure of T2.
    We then calculate CPD of T2 as a single
    shot ALS.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


class RCCSD_CPD_LS_T(CC):
    """
    This implements RCCSD with CPD decomposed amplitudes.
    We build CPD factors to approximate amplitudes in
    the least squares sense.
    For the next iteration we build full T1 and T2 residuals,
    but use the structure of T2 and RI decomposed interaction. This
    results in an N^5 algorithm.

    TODO: Unify this with RCCSD_nCPD_LS_T
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
            n = np.min((self._mos.nocc, self._mos.nvir))
            self.rankt = Tensors(t2=n)
        else:
            self.rankt = Tensors(rankt)

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

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        xs = cpd_initialize(t2_full.shape, self.rankt.t2)
        xs = als_dense(xs, t2_full, max_cycle=100, tensor_format='cpd')
        # xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        names = ['x1', 'x2', 'x3', 'x4']

        return Tensors(t1=t1,
                       t2=Tensors(zip(names, xs)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        return _rccsd_cpd_ls_t_unf_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """

        # Symmetrize T2 before feeding into res
        xs_sym = cpd_symmetrize(
            [a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
            {(1, 0, 3, 2): ('ident',)})

        amps = Tensors(
            t1=a.t1,
            t2=cpd_rebuild(xs_sym))

        # residuals are calculated with full amps for speed
        return _rccsd_ri_calc_residuals(h, amps)
        # return _rccsd_cpd_ls_t_unf_calc_residuals(h, a)

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

    def solve_amps(self, h, a, g):
        """
        Solve for new amplitudes using RHS and denominator
        """

        t1 = g.t1 * (- cc_denom(h.f, 2, 'dir', 'full'))

        # Symmetrize T2 HS
        g2 = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))

        # Solve T2
        t2_full = g2 * (- cc_denom(h.f, 4, 'dir', 'full'))

        xs = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                       t2_full, max_cycle=1, tensor_format='cpd')

        return Tensors(t1=t1, t2=Tensors(x1=xs[0],
                                         x2=xs[1], x3=xs[2], x4=xs[3]))

    def calculate_gradient(self, h, a):
        raise NotImplemented('RCCSD-CPD can not be solved using'
                             'gradient updates.')

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        names_abij = ['x1', 'x2', 'x3', 'x4']
        xs = [a.t2[key] for key in names_abij]

        # symmetrize t2 before feeding into res
        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})

        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(names_abij, xs_sym))))

        # Solve T1
        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        # Symmetrize T2 residuals
        r2 = 1 / 2 * (r.t2 + r.t2.transpose([1, 0, 3, 2]))

        # Solve T2
        r2_d = - r2 * cc_denom(h.f, 4, 'dir', 'full')

        t2 = [f for f in xs]
        for idx in range(len(t2)):
            g = (als_contract_dense(t2, r2_d, idx,
                                    tensor_format='cpd')
                 + als_contract_cpd(t2, xs_sym, idx,
                                    tensor_format='cpd'))
            s = als_pseudo_inverse(t2, t2, idx)
            f = np.dot(g, s)
            t2[idx] = f

        return Tensors(t1=t1, t2=Tensors(zip(names_abij, t2)))


class RCCSD_CPD_LS_T_HUB(RCCSD_CPD_LS_T):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes for Hubbard hamiltonian,
    where we calculate full residuals as in normal RCCSD,
    but taking advantage of the structure of T2.
    We then calculate CPD of T2 as a single
    shot ALS.

    Interaction is RI decomposed, T2 amplitudes are abij order.
    TODO: Unify this with RCCSD_nCPD_LS_T
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


class RCCSD_CPD_LS_T_TRUE(RCCSD_CPD_LS_T):
    """
    Final trial RCCSD code, where updates to the
    CPD factors are calculated with true N^4 scaling.
    It is, however, much much slower then N^6 code
    """

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """
        return _rccsd_cpd_ls_t_true_calculate_energy(h, a)

    def _calculate_r2d_projection(self, sel, h, a, b, d):
        """
        Calculate projection of R2 * D2 on dT/dX_{sel}
        sel - selects the component matrix X
        """
        if sel == 'x1':
            return _rccsd_cpd_ls_t_true_calc_r21(h, a, b, d)
        elif sel == 'x2':
            return _rccsd_cpd_ls_t_true_calc_r22(h, a, b, d)
        elif sel == 'x3':
            return _rccsd_cpd_ls_t_true_calc_r23(h, a, b, d)
        elif sel == 'x4':
            return _rccsd_cpd_ls_t_true_calc_r24(h, a, b, d)
        else:
            raise ValueError('Wrong index: {}'.format(sel))

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """
        names_abij = sorted(a.t2.keys())
        xs = [elem for elem in a.t2.to_generator()]

        # symmetrize t2 before feeding into res
        xs_sym = cpd_symmetrize(xs, {(1, 0, 3, 2): ('ident',)})
        a_sym = Tensors(t1=a.t1, t2=Tensors(zip(names_abij, xs_sym)))

        # Calculate residuals
        r1 = _rccsd_cpd_ls_t_true_calc_r1(
            h,
            a_sym)

        # Solve T1
        t1 = a.t1 - r1 * (cc_denom(h.f, 2, 'dir', 'full'))

        namesd_abij = ['d1', 'd2', 'd3', 'd4']
        d = Tensors(t2=Tensors(
            zip(namesd_abij, cc_denom(h.f, 4, 'dir', 'cpd'))))

        new_a = Tensors(t1=t1,
                        t2=Tensors(zip(names_abij, xs)))

        for idx, name in enumerate(sorted(a_sym.t2.keys())):
            g = (- self._calculate_r2d_projection(name, h, a_sym, new_a, d)
                 + als_contract_cpd(new_a.t2.to_list(), a_sym.t2.to_list(),
                                    idx,
                                    tensor_format='cpd'))
            s = als_pseudo_inverse(new_a.t2.to_list(),
                                   new_a.t2.to_list(), idx)
            f = np.dot(g, s)
            new_a.t2[name] = f

        return new_a


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
    from tcc.cc_solvers import (residual_diis_solver,
                                classic_solver, root_solver)

    cc1 = RCCSD_CPD_LS_T(rhf, rankt={'t2': 20})
    cc2 = RCCSD_MUL_RI(rhf)

    converged1, energy1, amps1 = classic_solver(
        cc1, max_cycle=150)

    converged2, energy2, amps2 = classic_solver(
        cc2, max_cycle=150)


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
    cc2 = RCCSD_CPD_LS_T_HUB(rhf, rankt={'t2': 30})

    converged1, energy1, amps1 = classic_solver(
        cc1, lam=5, max_cycle=50)

    converged2, energy2, amps2 = classic_solver(
        cc2, lam=1, conv_tol_energy=1e-8, max_cycle=500)


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
    rhf_ri = scf.density_fit(scf.RHF(mol))
    rhf_ri.scf()  # -76.0267656731

    from tcc.rccsd import RCCSD, RCCSD_DIR_RI
    from tcc.rccsd_cpd import (RCCSD_nCPD_LS_T,
                               RCCSD_CPD_LS_T)

    from tcc.cc_solvers import (classic_solver,
                                step_solver)

    cc1 = RCCSD_DIR_RI(rhf_ri)
    cc2 = RCCSD_nCPD_LS_T(rhf_ri, rankt={'t2': 30})
    cc3 = RCCSD_CPD_LS_T(rhf_ri, rankt={'t2': 30})

    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-8,)

    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-8,
        max_cycle=50)

    converged2, energy2, amps2 = step_solver(
        cc2, conv_tol_energy=1e-8,
        beta=0, max_cycle=150)

    converged3, energy3, amps3 = classic_solver(
        cc3, conv_tol_energy=1e-8,
        max_cycle=50)

    converged3, energy3, amps3 = step_solver(
        cc3, conv_tol_energy=1e-8,
        beta=0, max_cycle=150)


def test_update_diis_solver():
    from tcc.hubbard import hubbard_from_scf
    from pyscf import scf
    N = 6
    U = 4
    USE_PBC = 'y'
    rhf = hubbard_from_scf(scf.RHF, N, N, U, USE_PBC)
    rhf.scf()  # -76.0267656731

    from tcc.rccsd import RCCSD_UNIT
    from tcc.rccsd_cpd import (RCCSD_nCPD_LS_T_HUB)

    cc1 = RCCSD_UNIT(rhf)
    cc2 = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': 30})

    from tcc.cc_solvers import (residual_diis_solver,
                                update_diis_solver,
                                classic_solver)
    cc1._converged = False
    converged1, energy1, amps1 = residual_diis_solver(
        cc1, conv_tol_energy=1e-8, lam=3, max_cycle=100)

    import pickle
    with open('test_amps.p', 'rb') as fp:
        ampsi = pickle.load(fp)

    cc2._converged = False
    converged2, energy2, amps2 = update_diis_solver(
        cc2, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        beta=0.666,
        max_cycle=100,
        amps=ampsi)

    cc2._converged = False
    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        lam=3,
        max_cycle=100)


def test_cpd_equals_ncpd():
    from tcc.hubbard import hubbard_from_scf
    from pyscf import scf
    N = 6
    U = 4
    USE_PBC = 'y'
    rhf = hubbard_from_scf(scf.RHF, N, N, U, USE_PBC)
    rhf.scf()  # -76.0267656731

    from tcc.rccsd_cpd import (RCCSD_nCPD_LS_T_HUB,
                               RCCSD_CPD_LS_T_HUB)

    cc1 = RCCSD_CPD_LS_T_HUB(rhf, rankt={'t2': 30})
    cc2 = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': 30})

    from tcc.cc_solvers import (residual_diis_solver,
                                update_diis_solver,
                                classic_solver)
    cc1._converged = False
    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        lam=3,
        max_cycle=140)

    cc2._converged = False
    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        lam=3,
        max_cycle=140)

    print(energy2 - energy1)


def test_rccsd_cpd_true():
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = '3-21g'
    mol.build()
    rhf_ri = scf.density_fit(scf.RHF(mol))
    rhf_ri.scf()

    from tcc.rccsd_cpd import (RCCSD_CPD_LS_T,
                               RCCSD_CPD_LS_T_TRUE)

    cc1 = RCCSD_CPD_LS_T(rhf_ri, rankt={'t2': 30})
    cc2 = RCCSD_CPD_LS_T_TRUE(rhf_ri, rankt={'t2': 30})

    from tcc.cc_solvers import (update_diis_solver,
                                classic_solver)

    converged1, energy1, amps1 = update_diis_solver(
        cc1, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        beta=0.666,
        max_cycle=140)

    converged2, energy2, amps2 = update_diis_solver(
        cc2, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        beta=0.666,
        max_cycle=140)


def profile_cpd_true():
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = 'sto-3g'
    mol.build()
    rhf_ri = scf.density_fit(scf.RHF(mol))
    rhf_ri.scf()

    from tcc.rccsd_cpd import (RCCSD_CPD_LS_T,
                               RCCSD_CPD_LS_T_TRUE)

    cc1 = RCCSD_CPD_LS_T(rhf_ri, rankt={'t2': 30})
    cc2 = RCCSD_CPD_LS_T_TRUE(rhf_ri, rankt={'t2': 30})

    from tcc.cc_solvers import (update_diis_solver,
                                classic_solver)

    converged2, energy2, amps2 = update_diis_solver(
        cc2, conv_tol_energy=1e-8,
        conv_tol_res=1e-8,
        beta=0.666,
        max_cycle=140)


if __name__ == '__main__':
    profile_cpd_true()
    # test_cc()
