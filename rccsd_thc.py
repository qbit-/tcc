import numpy as np
from numpy import dot
from numpy.linalg import norm, pinv
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace
from tcc.thc import thc_rebuild, thc_contract_thc, thc_initialize

from tcc._rccsd_thc_ls import (gen_energy, gen_R1,
                               gen_RY1, gen_RY2,
                               gen_RY3, gen_RY4,
                               gen_RZ, gen_A1,
                               gen_A2, gen_A3,
                               gen_A4, gen_AZl,
                               gen_AZr)

class RCCSD_THC_LS_T(CC):
    """
    This class implements classic RCCSD method
    with THC decomposed amplitudes, where we
    solve for factors in the THC decomposition of T in a
    least squares sense.
    We use a THC decomposed Hamiltonian and a CPD decomposed
    energy denominator.

    The order of amplitudes and interaction is Mulliken,
    the order of factors in the THC decomposition of both
    amplitudes and interaction is defined by:

    v_{pqrs} = \sum_{m, n} x1_{p,m} x2_{q,m} x5_{m, n} x3_{r,n} x4_{p,n}

    """
    types = SimpleNamespace()

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None, load_ham_from=None, rankt=None):
        """
        Initialize RCCSD
        :param rankt: rank of the THC decomposition of amplitudes
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

        # initialize everything

        self.load_ham_from = load_ham_from

        if rankt is None:
            self.rankt = np.min((self._mos.nocc, self._mos.nvir))
        else:
            self.rankt = rankt

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple(
            'RCCSD_AMPLITUDES',
            field_names=('t1',
                         'x1', 'x2', 'x3', 'x4', 'x5'
                         ))

        self.types.RHS_TYPE = namedtuple(
            'RCCSD_RHS',
            field_names=('gt1',
                         'gx1', 'gx2', 'gx3', 'gx4', 'gx5'
                         ))

        self.types.RESIDUALS_TYPE = namedtuple(
            'RCCSD_RESIDUALS',
            field_names=('rt1',
                         'rx1', 'rx2', 'rx3', 'rx4', 'rx5'))

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD_THC_LS_T'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import _HAM_SPINLESS_THC_CORE_MATFILE
        return _HAM_SPINLESS_THC_CORE_MATFILE(self,
                                              filename=self.load_ham_from)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'mul', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)

        no = self.mos.nocc
        nv = self.mos.nvir

        v_vovo_factors = (ham.v.x1.v, ham.v.x2.o,
                          ham.v.x3.v, ham.v.x4.o, ham.v.x5)

        v_vovo_norm = thc_contract_thc(v_vovo_factors, v_vovo_factors)

        x1, x2, x3, x4, x5 = thc_initialize((nv, no, nv, no), self.rankt,
                                            scale_to_norm=1.0)

        return self.types.AMPLITUDES_TYPE(t1, x1, x2, x3, x4, x5)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        energy = gen_energy(h.f, h.v.x1, h.v.x2, h.v.x3,
                            h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                            a.x3, a.x4, a.x5)
        return energy

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        rt1 = gen_R1(h.f, h.v.x1, h.v.x2, h.v.x3,
                     h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                     a.x3, a.x4, a.x5)

        d = cc_denom(h.f, 4, ordering='mul', kind='cpd')

        rx1 = gen_RY1(h.f, h.v.x1, h.v.x2, h.v.x3,
                      h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                      a.x3, a.x4, a.x5, d[0], d[1],
                      d[2], d[3])
        rx2 = gen_RY2(h.f, h.v.x1, h.v.x2, h.v.x3,
                      h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                      a.x3, a.x4, a.x5, d[0], d[1],
                      d[2], d[3])
        rx3 = gen_RY3(h.f, h.v.x1, h.v.x2, h.v.x3,
                      h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                      a.x3, a.x4, a.x5, d[0], d[1],
                      d[2], d[3])
        rx4 = gen_RY4(h.f, h.v.x1, h.v.x2, h.v.x3,
                      h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                      a.x3, a.x4, a.x5, d[0], d[1],
                      d[2], d[3])
        rx5 = gen_RZ(h.f, h.v.x1, h.v.x2, h.v.x3,
                     h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                     a.x3, a.x4, a.x5, d[0], d[1],
                     d[2], d[3])

        return self.types.RESIDUALS_TYPE(rt1, rx1, rx2, rx3, rx4, rx5)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        t1 = g.gt1 / (- cc_denom(h.f, 2, 'mul', 'full'))

        A1 = gen_A1(a.x1, a.x2, a.x3, a.x4, a.x5)
        A2 = gen_A2(a.x1, a.x2, a.x3, a.x4, a.x5)
        A3 = gen_A3(a.x1, a.x2, a.x3, a.x4, a.x5)
        A4 = gen_A4(a.x1, a.x2, a.x3, a.x4, a.x5)
        AZl = gen_AZl(a.x1, a.x2, a.x3, a.x4, a.x5)
        AZr = gen_AZr(a.x1, a.x2, a.x3, a.x4, a.x5)

        a_new = [
            -dot(g.gx1, pinv(A1, rcond=1e-10)) + a.x1,
            -dot(g.gx2, pinv(A2, rcond=1e-10)) + a.x2,
            -dot(g.gx3, pinv(A3, rcond=1e-10)) + a.x3,
            -dot(g.gx4, pinv(A4, rcond=1e-10)) + a.x4,
            -dot(dot(pinv(AZl, rcond=1e-10),
                     g.gx5), pinv(AZr, rcond=1e-10)) + a.x5
        ]
        # Normalize columns to 1 in X factors, Hermitize Z
        a_new = [factor / norm(factor, axis=0)
                 for factor
                 in a_new[:4]] + [
                     1/2 * (a_new[4] + a_new[4].T),
                     ]

        return self.types.AMPLITUDES_TYPE(
            t1, *a_new)

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        # Here we simply copy the residuals of T2 factors
        # to the RHS arrays

        return self.types.RHS_TYPE(
            gt1=r.rt1 - a.t1 / cc_denom(h.f, 2, 'mul', 'full'),
            gx1=r.rx1, gx2=r.rx2, gx3=r.rx3, gx4=r.rx4, gx5=r.rx5
        )


class RCCSD_THC_LS_T_HUB(RCCSD_THC_LS_T):
    """
    This class implements classic RCCSD method
    with THC decomposed amplitudes for Hubbard model, where we
    solve for factors in the THC decomposition of T in a
    least squares sense.
    We use a THC decomposed Hamiltonian built analytically
    and a CPD decomposed energy denominator.

    The order of amplitudes and interaction is Mulliken,
    the order of factors in the THC decomposition of both
    amplitudes and interaction is defined by:

    v_{pqrs} = \sum_{m, n} x1_{p,m} x2_{q,m} x5_{m, n} x3_{r,n} x4_{p,n}

    """

    @property
    def method_name(self):
        return 'RCCSD_THC_LS_T_HUB'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_THC_CORE_HUBBARD
        return HAM_SPINLESS_THC_CORE_HUBBARD(self)


def initial_test():
    import numpy as np
    from tcc._rccsd_thc_ls import (gen_energy, gen_R1,
                               gen_RY1, gen_RY2,
                               gen_RY3, gen_RY4,
                               gen_RZ, gen_A1,
                               gen_A2, gen_A3,
                               gen_A4, gen_AZl,
                               gen_AZr)
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
    rhf.scf()  # -76.0267656731

    from tcc.rccsd_thc import RCCSD_THC_LS_T
    cc = RCCSD_THC_LS_T(
        rhf, load_ham_from='reference_random_ham.mat')

    h = cc.create_ham()

    from scipy.io import loadmat
    ref = loadmat('reference_random_ham.mat',
                  matlab_compatible=True)
    t2l = ref['t2s'][0][0]
    t1 = ref['t1']

    a = cc.types.AMPLITUDES_TYPE(t1, *t2l)

    from tcc.denom import cc_denom
    d = cc_denom(h.f, 4, ordering='mul', kind='cpd')

    from tcc._rccsd_thc_ls import gen_energy, gen_R1

    energy = gen_energy(h.f, h.v.x1, h.v.x2, h.v.x3,
                        h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                        a.x3, a.x4, a.x5)
    e_ref = ref['energy'][0][0]
    print(energy - e_ref)

    rt1 = gen_R1(h.f, h.v.x1, h.v.x2, h.v.x3,
                 h.v.x4, h.v.x5, a.t1, a.x1, a.x2,
                 a.x3, a.x4, a.x5)

    gt1 = rt1 - a.t1 / cc_denom(h.f, 2, 'mul', 'full')
    t1n = gt1 * (- cc_denom(h.f, 2, 'mul', 'full'))

    t1n_ref = ref['t1n']
    print(t1n - t1n_ref)

    rz = gen_RZ(h.f, h.v.x1, h.v.x2, h.v.x3, h.v.x4, h.v.x5, a.t1,
                a.x1, a.x2, a.x3, a.x4, a.x5, d[0], d[1], d[2], d[3])
    ry1 = gen_RY1(h.f, h.v.x1, h.v.x2, h.v.x3, h.v.x4, h.v.x5, a.t1,
                  a.x1, a.x2, a.x3, a.x4, a.x5, d[0], d[1], d[2], d[3])
    ry2 = gen_RY2(h.f, h.v.x1, h.v.x2, h.v.x3, h.v.x4, h.v.x5, a.t1,
                  a.x1, a.x2, a.x3, a.x4, a.x5, d[0], d[1], d[2], d[3])
    ry3 = gen_RY3(h.f, h.v.x1, h.v.x2, h.v.x3, h.v.x4, h.v.x5, a.t1,
                  a.x1, a.x2, a.x3, a.x4, a.x5, d[0], d[1], d[2], d[3])
    ry4 = gen_RY4(h.f, h.v.x1, h.v.x2, h.v.x3, h.v.x4, h.v.x5, a.t1,
                  a.x1, a.x2, a.x3, a.x4, a.x5, d[0], d[1], d[2], d[3])

    print([np.max(r_new + r_ref)
           for (r_new, r_ref)
           in zip([ry1, ry2, ry3, ry4, rz], ref['r2n'][0][0])])

    a1 = gen_A1(a.x1, a.x2, a.x3, a.x4, a.x5)
    a2 = gen_A2(a.x1, a.x2, a.x3, a.x4, a.x5)
    a3 = gen_A3(a.x1, a.x2, a.x3, a.x4, a.x5)
    a4 = gen_A4(a.x1, a.x2, a.x3, a.x4, a.x5)
    azl = gen_AZl(a.x1, a.x2, a.x3, a.x4, a.x5)
    azr = gen_AZr(a.x1, a.x2, a.x3, a.x4, a.x5)

    print([np.max(a_new - a_ref)
           for (a_new, a_ref)
           in zip([a1, a2, a3, a4, azl, azr], ref['as'][0][0])])

    x1n = a.x1 - np.dot(ry1, np.linalg.pinv(a1))


def test_hubbard_iterations():
    from pyscf import gto
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 6, 6, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from scipy.io import loadmat
    ref = loadmat('reference_hub_rccsdthc.mat', matlab_compatible=True)
    mo_coeff = ref['orbA']
    mo_energy = ref['valsA'].flatten()

    from tcc.cc_solvers import classic_solver
    from tcc.rccsd_thc import RCCSD_THC_LS_T_HUB
    cc = RCCSD_THC_LS_T_HUB(rhf, rankt=3, mo_energy=mo_energy,
                            mo_coeff=mo_coeff)
    t1 = ref['t1']
    t2l = ref['t2s'][0][0]
    amps = cc.types.AMPLITUDES_TYPE(t1, *t2l)

    t1n = ref['t1n']
    t2ln = ref['t2sn'][0][0]
    ref_amps = cc.types.AMPLITUDES_TYPE(t1n, *t2ln)

    dt1 = ref['dt1']
    dt2l = ref['dt2s'][0][0]
    delta = cc.types.AMPLITUDES_TYPE(dt1, *dt2l)

    ham = cc.create_ham()
    res = cc.calc_residuals(ham, amps)
    rhs = cc.update_rhs(ham, amps, res)
    new_amps = cc.solve_amps(ham, amps, rhs)

    delta1 = cc.types.AMPLITUDES_TYPE(new_amps.t1 - amps.t1,
                                      new_amps.x1 - amps.x1,
                                      new_amps.t1 - amps.x2,
                                      new_amps.t1 - amps.x3,
                                      new_amps.t1 - amps.x4,
                                      new_amps.t1 - amps.x5
    ) 
    # cc = RCCSD_THC_LS_T_HUB(rhf, rankt=3)
    converged, energy, _ = classic_solver(
        cc, max_cycle=300, lam=3, amps=amps)
    print(converged, energy)

if __name__ == '__main__':
    test_hubbard_iterations()
