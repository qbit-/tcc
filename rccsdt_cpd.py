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

from tcc._rccsdt_mul import (_rccsdt_cpd_ls_t_calculate_energy,
                             _rccsdt_cpd_ls_t_calc_residuals)


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
    #  These are containers used by all  methods of this class
    # to pass numpy arrays

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
        t2x = als_dense(t2x, t2_full, max_cycle=100)

        t3x = ncpd_initialize((nvir,) * 3 + (nocc,) * 3,
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
        t2names = ['x1', 'x2', 'x3', 'x4']
        t3names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        r = self.calc_residuals(h, a)
        r3 = ((+ r.t3
               + r.t3.transpose([0, 1, 2, 5, 3, 4])
               + r.t3.transpose([0, 1, 2, 4, 5, 3])
               + r.t3
               + r.t3.transpose([0, 2, 1, 3, 5, 4])
               + r.t3.transpose([2, 0, 1, 5, 3, 4])) / 6 *
              cc_denom(h.f, 6, 'dir', 'full'))

        dt1 = r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))
        dt2x = als_dense([a.t2.x1, a.t2.x2, a.t2.x3, a.t2.x4],
                         r.t2 * (cc_denom(h.f, 4, 'dir', 'full')),
                         self.rankt.t2, max_cycle=1,
                         tensor_format='cpd')
        dt3x = als_dense([a.t3.x1, a.t3.x2, a.t3.x3,
                          a.t3.x4, a.t3.x5, a.t3.x6],
                         r3 * (cc_denom(h.f, 6, 'dir', 'full')),
                         self.rankt.t3, max_cycle=1,
                         tensor_format='cpd')

        return Tensors(
            t1=dt1, t2=Tensors(zip(t2names, dt2x)),
            t3=Tensors(zip(t3names, dt3x)))

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes
        """

        t2names = ['x1', 'x2', 'x3', 'x4']
        t2x = [a.t2[key] for key in t2names]

        t3names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        t2x = [a.t3[key] for key in t3names]

        # symmetrize t2 before feeding into res
        t2x_sym = ncpd_symmetrize(t2x, {(1, 0, 3, 2): ('ident',)})

        # symmetrize t3 before feeding into res
        t3x_sym = ncpd_symmetrize(t2x, {(1, 0, 3, 2): ('ident',)})

        # Running residuals with symmetrized amplitudes is much slower,
        # but convergence is more stable. Derive unsymm equations?
        r = self.calc_residuals(
            h,
            Tensors(t1=a.t1, t2=Tensors(zip(t2names, t2x_sym))))

        t1 = a.t1 - r.t1 * (cc_denom(h.f, 2, 'dir', 'full'))

        r_d = - r.t2 * cc_denom(h.f, 4, 'dir', 'full')

        t2 = [f for f in xs1]
        for idx in range(len(t2)):
            g = (als_contract_dense(t2, r_d, idx,
                                    tensor_format='ncpd')
                 # here we can use unsymmetried amps as well,
                 # giving lower energy and worse convergence
                 + als_contract_cpd(t2, xs_sym, idx,
                                    tensor_format='ncpd'))
            s = als_pseudo_inverse(t2, t2, idx)
            f = np.dot(g, s)
            t2[idx] = f

        return Tensors(t1=t1, t2=Tensors(zip(names_abij, t2)))


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
    from tcc.cc_solvers import classic_solver
    cc = RCCSDT_MUL_RI(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy - -1.304876e-01))
