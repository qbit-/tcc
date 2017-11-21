import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from tcc.tensors import Tensors

from tcc.cpd import (cpd_initialize, cpd_rebuild,
                     ncpd_initialize, ncpd_rebuild,
                     als_dense, als_cpd, cpd_symmetrize,
                     ncpd_symmetrize, als_contract_cpd,
                     als_contract_dense, als_pseudo_inverse)

from tcc._rccsdt_cpd import (_rccsdt_cpd_ls_t_calculate_energy,
                             _rccsdt_cpd_ls_t_calc_residuals)

from tcc._rccsdt_cpd import (_rccsdt_ncpd_ls_t_calculate_energy,
                             _rccsdt_ncpd_ls_t_calc_residuals)

from tcc._rccsdt_mul import (_rccsdt_mul_ri_calc_residuals)

from tcc._rccsdt_cpd import (_rccsdt_ncpd_t2f_ls_t_calculate_energy,
                             _rccsdt_ncpd_t2f_ls_t_calc_residuals)


class RCCSDT_CPD_LS_T(CC):
    """
    This class implements classic RCCSDT method with
    CPD decomposed amplitudes, where we calculate
    full residuals as in normal RCCSDT, but taking advantage
    of the structure of T2 and T3.

    The order of amplitudes is
    t1: vo, t2: vvoo, t3: vvvooo
    and integrals are RI decomposed
    """

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None, rankt=None):
        """
        Initialize RCCSDT
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

        # initialize ranks

        if rankt is None:
            n = np.min((self._mos.nocc, self._mos.nvir))
            self.rankt = Tensors(t2=n, t3=n)
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
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t2names = ['x1', 'x2', 'x3', 'x4']
        t3names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        t2x = cpd_initialize(t2_full.shape, self.rankt.t2)
        t2x = als_dense(t2x, t2_full, max_cycle=100,
                        tensor_format='cpd')

        t3x = cpd_initialize((nvir,) * 3 + (nocc,) * 3,
                             self.rankt.t3, init_function=np.zeros)

        return Tensors(t1=t1, t2=Tensors(zip(t2names, t2x)),
                       t3=Tensors(zip(t3names, t3x)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_cpd_ls_t_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_cpd_ls_t_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - cpd_rebuild(
                (a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4)
            ) / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (cpd_rebuild(
                (a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, a.t3.x6)
            ) - cpd_rebuild(
                (a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x4, a.t3.x6)
            )) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        g3 = (+ g.t3
              + g.t3.transpose([0, 1, 2, 5, 3, 4])
              + g.t3.transpose([0, 1, 2, 4, 5, 3])
              + g.t3.transpose([0, 1, 2, 3, 4, 5])
              + g.t3.transpose([0, 2, 1, 3, 5, 4])
              + g.t3.transpose([2, 1, 0, 5, 4, 3])) / 6

        t2_full = 1 / 2 * ((g.t2 + g.t2.transpose([1, 0, 3, 2])) *
                           (- cc_denom(h.f, g.t2.ndim, 'dir', 'full')))
        t3_full = g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))

        t2x = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                        t2_full, max_cycle=1, tensor_format='cpd')
        t3x = als_dense([a.t3.x1, a.t3.x2, a.t3.x3,
                         a.t3.x4, a.t3.x5, a.t3.x6],
                        t3_full, max_cycle=1, tensor_format='cpd')
        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=Tensors(x1=t2x[0], x2=t2x[1],
                       x3=t2x[2], x4=t2x[3]),
            t3=Tensors(x1=t3x[0], x2=t3x[1],
                       x3=t3x[2], x4=t3x[3],
                       x5=t3x[4], x6=t3x[5]),
        )

    def calculate_gradient(self, h, a):
        """
        Solving for new T amlitudes using RHS and denominator
        """
        raise NotImplemented('CC-CPD can not be implemented'
                             ' as a gradient descent')

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        t2names = ['x1', 'x2', 'x3', 'x4']
        t2x = [a.t2[key] for key in t2names]

        t3names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        t3x = [a.t3[key] for key in t3names]

        # symmetrize t2 before feeding into res
        t2x_sym = cpd_symmetrize(t2x, {(1, 0, 3, 2): ('ident',)})

        # symmetrize t3 before feeding into res
        t3x_sym = cpd_symmetrize(t3x, {(0, 1, 2, 5, 3, 4): ('ident',),
                                       (0, 1, 2, 4, 5, 3): ('ident',),
                                       (0, 1, 2, 3, 4, 5): ('ident',),
                                       (0, 2, 1, 3, 5, 4): ('ident',),
                                       (2, 0, 1, 5, 3, 4): ('ident',)})

        # Running residuals with symmetrized amplitudes is much slower,
        # but convergence is more stable. Derive unsymm equations?
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(t2names, t2x_sym)),
                    t3=Tensors(zip(t3names, t3x_sym)))
        )

        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        r2_d = - r.t2 * cc_denom(h.f, 4, 'dir', 'full')

        t2 = [f for f in t2x]
        for idx in range(len(t2)):
            g = (als_contract_dense(t2, r2_d, idx,
                                    tensor_format='cpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t2, t2x_sym, idx,
                                    tensor_format='cpd'))
            s = als_pseudo_inverse(t2, t2, idx)
            f = np.dot(g, s)
            t2[idx] = f

        r3 = ((+ r.t3
               + r.t3.transpose([0, 1, 2, 5, 3, 4])
               + r.t3.transpose([0, 1, 2, 4, 5, 3])
               + r.t3
               + r.t3.transpose([0, 2, 1, 3, 5, 4])
               + r.t3.transpose([2, 0, 1, 5, 3, 4])) / 6 *
              cc_denom(h.f, 6, 'dir', 'full'))

        r3_d = - r3 * cc_denom(h.f, 6, 'dir', 'full')
        t3 = [f for f in t3x]
        for idx in range(len(t2)):
            g = (als_contract_dense(t3, r3_d, idx,
                                    tensor_format='cpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t3, t3x_sym, idx,
                                    tensor_format='cpd'))
            s = als_pseudo_inverse(t3, t3, idx)
            f = np.dot(g, s)
            t3[idx] = f

        return Tensors(t1=t1, t2=Tensors(zip(t2names, t2)),
                       t3=Tensors(zip(t3names, t3)))


class RCCSDT_nCPD_LS_T(RCCSDT_CPD_LS_T):
    """
    This class implements classic RCCSDT method with
    nCPD decomposed amplitudes, where we calculate
    full residuals as in normal RCCSDT, but taking advantage
    of the structure of T2 and T3.

    The order of amplitudes is
    t1: vo, t2: vvoo, t3: vvvooo
    and integrals are RI decomposed
    """

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
        t3names = ['xlam', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        t2x = ncpd_initialize(t2_full.shape, self.rankt.t2)
        t2x = als_dense(t2x, t2_full, max_cycle=100,
                        tensor_format='ncpd')

        t3x = ncpd_initialize(
            (nvir,) * 3 + (nocc,) * 3,
            self.rankt.t3,
            init_function=(lambda x: 0.001 * np.random.rand(*x)))

        return Tensors(t1=t1, t2=Tensors(zip(t2names, t2x)),
                       t3=Tensors(zip(t3names, t3x)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_ncpd_ls_t_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_ncpd_ls_t_calc_residuals(h, a)
        # return _rccsdt_mul_ri_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - ncpd_rebuild(
                (a.t2.xlam, a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4)
            ) / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (ncpd_rebuild(
                (a.t3.xlam, a.t3.x1, a.t3.x2,
                 a.t3.x3, a.t3.x4, a.t3.x5, a.t3.x6)
            ) - ncpd_rebuild(
                (a.t3.xlam, a.t3.x1, a.t3.x2, a.t3.x3,
                 a.t3.x5, a.t3.x4, a.t3.x6)
            )) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        g3 = (+ g.t3
              + g.t3.transpose([0, 1, 2, 5, 3, 4])
              + g.t3.transpose([0, 1, 2, 4, 5, 3])
              + g.t3.transpose([0, 1, 2, 3, 4, 5])
              + g.t3.transpose([0, 2, 1, 3, 5, 4])
              + g.t3.transpose([2, 1, 0, 5, 4, 3])) / 6

        t2_full = 1 / 2 * ((g.t2 + g.t2.transpose([1, 0, 3, 2])) *
                           (- cc_denom(h.f, g.t2.ndim, 'dir', 'full')))
        t3_full = g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))

        t2x = als_dense([a.t2.xlam, a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                        t2_full, max_cycle=1, tensor_format='ncpd')
        t3x = als_dense([a.t3.xlam, a.t3.x1, a.t3.x2, a.t3.x3,
                         a.t3.x4, a.t3.x5, a.t3.x6],
                        t3_full, max_cycle=1, tensor_format='ncpd')
        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=Tensors(xlam=t2x[0], x1=t2x[1], x2=t2x[2],
                       x3=t2x[3], x4=t2x[4]),
            t3=Tensors(xlam=t3x[0], x1=t3x[1], x2=t3x[2],
                       x3=t3x[3], x4=t3x[4],
                       x5=t3x[5], x6=t3x[6]),
        )

    def calculate_gradient(self, h, a):
        """
        Solving for new T amlitudes using RHS and denominator
        """
        raise NotImplemented('CC-CPD can not be implemented as a'
                             'gradient descent')

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
        t2x = [a.t2[key] for key in t2names]

        t3names = ['xlam', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        t3x = [a.t3[key] for key in t3names]

        # symmetrize t2 before feeding into res
        t2x_sym = ncpd_symmetrize(t2x, {(1, 0, 3, 2): ('ident',)})

        # symmetrize t3 before feeding into res
        t3x_sym = ncpd_symmetrize(t3x, {(0, 1, 2, 5, 3, 4): ('ident',),
                                        (0, 1, 2, 4, 5, 3): ('ident',),
                                        (0, 1, 2, 3, 4, 5): ('ident',),
                                        (0, 2, 1, 3, 5, 4): ('ident',),
                                        (2, 0, 1, 5, 3, 4): ('ident',)})

        # Running residuals with symmetrized amplitudes is much slower,
        # but convergence is more stable. Derive unsymm equations?
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(t2names, t2x_sym)),
                    t3=Tensors(zip(t3names, t3x_sym)))
        )
        # r = self.calc_residuals(
        #     h,
        #     Tensors(t1=a.t1, t2=ncpd_rebuild(t2x_sym),
        #             t3=ncpd_rebuild(t3x_sym))
        # )

        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        r2_d = - (1 / 2 * (+ r.t2
                           + r.t2.transpose([1, 0, 3, 2]))
                  * cc_denom(h.f, 4, 'dir', 'full'))

        t2 = [f for f in t2x]
        for idx in range(len(t2)):
            g = (als_contract_dense(t2, r2_d, idx,
                                    tensor_format='ncpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t2, t2x_sym, idx,
                                    tensor_format='ncpd'))
            s = als_pseudo_inverse(t2, t2, idx)
            f = np.dot(g, s)
            t2[idx] = f

        r3 = 1 / 6 * (+ r.t3
                      + r.t3.transpose([0, 1, 2, 5, 3, 4])
                      + r.t3.transpose([0, 1, 2, 4, 5, 3])
                      + r.t3
                      + r.t3.transpose([0, 2, 1, 3, 5, 4])
                      + r.t3.transpose([2, 0, 1, 5, 3, 4]))

        r3_d = - r3 * cc_denom(h.f, 6, 'dir', 'full')
        t3 = [f for f in t3x]
        for idx in range(len(t3)):
            g = (als_contract_dense(t3, r3_d, idx,
                                    tensor_format='ncpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t3, t3x_sym, idx,
                                    tensor_format='ncpd'))
            s = als_pseudo_inverse(t3, t3, idx)
            f = np.dot(g, s)
            t3[idx] = f

        return Tensors(t1=t1, t2=Tensors(zip(t2names, t2)),
                       t3=Tensors(zip(t3names, t3)))


class RCCSDT_nCPD_T_LS_T(RCCSDT_nCPD_LS_T):
    """
    This class implements classic RCCSDT method with
    nCPD decomposed T3 amplitudes and full T2 amplitudes,
    where we calculate full residuals as in normal
    RCCSDT, but taking advantage
    of the structure of T3.

    The order of amplitudes is
    t1: vo, t2: vvoo, t3: vvvooo
    and integrals are RI decomposed
    """

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2_full = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t3names = ['xlam', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        t3x = ncpd_initialize(
            (nvir,) * 3 + (nocc,) * 3,
            self.rankt.t3,
            init_function=(lambda x: 0.001 * np.random.rand(*x)))

        return Tensors(t1=t1, t2=t2_full,
                       t3=Tensors(zip(t3names, t3x)))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_ncpd_t2f_ls_t_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        t3 = ncpd_rebuild([a.t3.xlam, a.t3.x1, a.t3.x2, a.t3.x3,
                           a.t3.x4, a.t3.x5, a.t3.x6])

        return _rccsdt_mul_ri_calc_residuals(h,
                                             Tensors(
                                                 t1=a.t1,
                                                 t2=a.t2,
                                                 t3=t3
                                             ))

        # return _rccsdt_ncpd_t2f_ls_t_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - a.t2 / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (ncpd_rebuild(
                (a.t3.xlam, a.t3.x1, a.t3.x2,
                 a.t3.x3, a.t3.x4, a.t3.x5, a.t3.x6)
            ) - ncpd_rebuild(
                (a.t3.xlam, a.t3.x1, a.t3.x2, a.t3.x3,
                 a.t3.x5, a.t3.x4, a.t3.x6)
            )) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        t2_full = 1 / 2 * ((g.t2 + g.t2.transpose([1, 0, 3, 2])) *
                           (- cc_denom(h.f, g.t2.ndim, 'dir', 'full')))
        g3 = 1 / 6 * (+ g.t3
                      + g.t3.transpose([0, 1, 2, 5, 3, 4])
                      + g.t3.transpose([0, 1, 2, 4, 5, 3])
                      + g.t3.transpose([0, 1, 2, 3, 4, 5])
                      + g.t3.transpose([0, 2, 1, 3, 5, 4])
                      + g.t3.transpose([2, 1, 0, 5, 4, 3]))

        t3_full = g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))

        t3x = als_dense([a.t3.xlam, a.t3.x1, a.t3.x2, a.t3.x3,
                         a.t3.x4, a.t3.x5, a.t3.x6],
                        t3_full, max_cycle=1, tensor_format='ncpd')
        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=t2_full,
            t3=Tensors(xlam=t3x[0], x1=t3x[1], x2=t3x[2],
                       x3=t3x[3], x4=t3x[4],
                       x5=t3x[5], x6=t3x[6]),
        )

    def calculate_gradient(self, h, a):
        """
        Solving for new T amlitudes using RHS and denominator
        """
        raise NotImplemented('CC-CPD can not be implemented'
                             ' as a gradient descent')

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        t3names = ['xlam', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        t3x = [a.t3[key] for key in t3names]

        # symmetrize t3 before feeding into res
        t3x_sym = ncpd_symmetrize(t3x, {(0, 1, 2, 5, 3, 4): ('ident',),
                                        (0, 1, 2, 4, 5, 3): ('ident',),
                                        (0, 1, 2, 3, 4, 5): ('ident',),
                                        (0, 2, 1, 3, 5, 4): ('ident',),
                                        (2, 0, 1, 5, 3, 4): ('ident',)})

        # Running residuals with symmetrized amplitudes is much slower,
        # but convergence is more stable. Derive unsymm equations?
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=a.t2,
                    t3=Tensors(zip(t3names, t3x_sym)))
        )

        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        t2 = a.t2 - (1 / 2 * (+ r.t2
                              + r.t2.transpose([1, 0, 3, 2]))
                     * (cc_denom(h.f, 4, 'dir', 'full')))

        r3 = 1 / 6 * (+ r.t3
                      + r.t3.transpose([0, 1, 2, 5, 3, 4])
                      + r.t3.transpose([0, 1, 2, 4, 5, 3])
                      + r.t3
                      + r.t3.transpose([0, 2, 1, 3, 5, 4])
                      + r.t3.transpose([2, 0, 1, 5, 3, 4]))

        r3_d = - r3 * cc_denom(h.f, 6, 'dir', 'full')
        t3 = [f for f in t3x]
        for idx in range(len(t3)):
            g = (als_contract_dense(t3, r3_d, idx,
                                    tensor_format='ncpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t3, t3x_sym, idx,
                                    tensor_format='ncpd'))
            s = als_pseudo_inverse(t3, t3, idx)
            f = np.dot(g, s)
            t3[idx] = f

        return Tensors(t1=t1, t2=t2,
                       t3=Tensors(zip(t3names, t3)))


class RCCSDT_nCPD_T_LS_T_HUB(RCCSDT_nCPD_T_LS_T):
    """
    This class implements classic RCCSDT method
    for hubbard Hamiltonian. T3 amplitudes are
    nCPD decomposed and T2 amplitudes are full.
    We calculate full residuals as in normal
    RCCSDT, but taking advantage
    of the structure of T3.

    The order of amplitudes is
    t1: vo, t2: vvoo, t3: vvvooo
    and integrals are RI decomposed
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()

    from tcc.rccsdt_mul import RCCSDT_MUL_RI
    from tcc.rccsdt_cpd import RCCSDT_nCPD_LS_T
    from tcc.cc_solvers import (classic_solver, step_solver)
    cc1 = RCCSDT_MUL_RI(rhf)
    cc2 = RCCSDT_nCPD_LS_T(rhf, rankt={'t2': 20, 't3': 40})
    cc3 = RCCSDT_CPD_LS_T(rhf, rankt={'t2': 20, 't3': 40})

    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy1 - -1.304876e-01))

    import numpy as np
    np.seterr(all='raise')
    import warnings
    warnings.filterwarnings("error")

    converged2, energy2, amps2 = step_solver(
        cc2, conv_tol_energy=1e-8,
        max_cycle=100)

    converged3, energy3, amps3 = step_solver(
        cc3, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy2 - -0.12621546190311517))
    print('E(CPD)-E(nCPD): {}'.format(energy3 - energy2))


def test_cc_t2f():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()

    from tcc.rccsdt_cpd import (RCCSDT_nCPD_LS_T,
                                RCCSDT_nCPD_T_LS_T)
    from tcc.cc_solvers import (classic_solver, step_solver)

    cc1 = RCCSDT_nCPD_LS_T(rhf, rankt={'t2': 20, 't3': 40})
    cc2 = RCCSDT_nCPD_T_LS_T(rhf, rankt={'t3': 40})

    converged2, energy2, amps2 = step_solver(
        cc2, conv_tol_energy=1e-8,
        max_cycle=100)

    converged1, energy1, amps1 = step_solver(
        cc1, conv_tol_energy=1e-8,
        max_cycle=100)

    print('E(full) - E(T2 CPD): {}'.format(energy2 - energy1))


if __name__ == '__main__':
    test_cc_t2f()
