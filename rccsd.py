import numpy as np
from numpy import einsum
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple


class RCCSD(CC):
    """
    This class implements classic RCCSD method
    with vvoo ordered amplitudes
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays
    AMPLITUDES_TYPE = namedtuple('RCCSD_AMPLITUDES_FULL',
                                 field_names=('t1', 't2'))
    RHS_TYPE = namedtuple('RCCSD_RHS_FULL',
                          field_names=('g1', 'g2'))
    RESIDUALS_TYPE = namedtuple('RCCSD_RESIDUALS_FULL',
                                field_names=('r1', 'r2'))
    EXTENDED_AMPLITUDES_TYPE = namedtuple('RCCSD_EXT_AMPLITUDES',
                                          field_names=('t', 'z'))

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None):
        """
        Initialize RCCSD
        """
        # Simply copy some parameters from RHF calculation
        super(RCCSD, self).__init__(mf)

        # Initialize molecular orbitals

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        from tcc.mos import SPINLESS_MOS
        self._mos = SPINLESS_MOS(mo_coeff, mo_energy, mo_occ, frozen)

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD'

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

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)

        return self.AMPLITUDES_TYPE(t1, t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        tau0 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        energy = (
            einsum("ai,ia->", a.t1, tau1)
            + einsum("abij,jiba->", a.t2, tau0)
        )

        return energy

    def update_rhs(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        tau0 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        tau2 = (
            einsum("bj,jiba->ia", a.t1, tau0)
        )

        tau3 = (
            einsum("ia->ia", tau2)
            + einsum("ia->ia", h.f.ov)
        )

        tau4 = (
            - einsum("abji->ijab", a.t2)
            + 2 * einsum("baji->ijab", a.t2)
        )

        tau5 = (
            2 * einsum("abji->ijab", a.t2)
            - einsum("baji->ijab", a.t2)
        )

        tau6 = (
            2 * einsum("ijka->ijka", h.v.ooov)
            - einsum("jika->ijka", h.v.ooov)
        )

        tau7 = (
            einsum("aj,ia->ij", a.t1, tau3)
            + einsum("kjab,kiba->ij", tau5, h.v.oovv)
            + einsum("ij->ij", h.f.oo)
            + einsum("ak,ikja->ij", a.t1, tau6)
        )

        tau8 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau9 = (
            einsum("ijka->ijka", tau8)
            - 2 * einsum("ikja->ijka", tau8)
            - 2 * einsum("jkia->ijka", h.v.ooov)
            + einsum("kjia->ijka", h.v.ooov)
        )

        tau10 = (
            2 * einsum("bjia->ijab", h.v.voov.conj())
            - einsum("iajb->ijab", h.v.ovov)
        )

        tau11 = (
            - einsum("iabc->iabc", h.v.ovvv)
            + 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau12 = (
            einsum("ci,iabc->ab", a.t1, tau11)
            + einsum("ab->ab", h.f.vv)
        )

        tau13 = (
            einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau14 = (
            - 2 * einsum("abji->ijab", a.t2)
            + einsum("baji->ijab", a.t2)
        )

        tau15 = (
            einsum("abji->ijab", a.t2)
            - 2 * einsum("baji->ijab", a.t2)
        )

        tau16 = (
            einsum("jica,ijbc->ab", tau15, h.v.oovv)
        )

        tau17 = (
            einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau18 = (
            einsum("ai,jkla->ijkl", a.t1, tau8)
        )

        tau19 = (
            einsum("jilk->ijkl", h.v.oooo)
            + einsum("lkji->ijkl", tau17)
            + einsum("lkji->ijkl", tau18)
        )

        tau20 = (
            einsum("al,likj->ijka", a.t1, tau19)
        )

        tau21 = (
            einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau22 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau23 = (
            einsum("ijab->ijab", tau21)
            + 2 * einsum("caki,kjbc->ijab", a.t2, tau22)
        )

        tau24 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau25 = (
            einsum("acki,kjbc->ijab", a.t2, tau24)
        )

        tau26 = (
            einsum("di,badc->iabc", a.t1, h.v.vvvv)
        )

        tau27 = (
            einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau28 = (
            einsum("ai,ja->ij", a.t1, tau3)
        )

        tau29 = (
            einsum("ij->ij", tau28)
            + einsum("ji->ij", h.f.oo)
        )

        tau30 = (
            - einsum("ik,bakj->ijab", tau29, a.t2)
        )

        tau31 = (
            einsum("ac,cbji->ijab", h.f.vv, a.t2)
        )

        tau32 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau33 = (
            einsum("ikbc,kjca->ijab", tau32, tau4)
        )

        tau34 = (
            - einsum("ib,bakj->ijka", tau3, a.t2)
        )

        tau35 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau36 = (
            einsum("balj,likb->ijka", a.t2, tau35)
        )

        tau37 = (
            einsum("bi,jakb->ijka", a.t1, h.v.ovov)
        )

        tau38 = (
            einsum("bi,jkab->ijka", a.t1, tau21)
        )

        tau39 = (
            einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau40 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau41 = (
            einsum("al,ijlk->ijka", a.t1, tau40)
        )

        tau42 = (
            einsum("ijka->ijka", tau34)
            + einsum("ijka->ijka", tau36)
            - einsum("jika->ijka", tau37)
            + einsum("jkia->ijka", tau38)
            + einsum("jika->ijka", tau39)
            + einsum("jika->ijka", tau41)
        )

        tau43 = (
            einsum("ak,kijb->ijab", a.t1, tau42)
        )

        tau44 = (
            einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau45 = (
            einsum("ci,jabc->ijab", a.t1, tau44)
        )

        tau46 = (
            einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
        )

        tau47 = (
            einsum("ci,jabc->ijab", a.t1, tau46)
        )

        tau48 = (
            einsum("ci,jcab->ijab", a.t1, h.v.ovvv.conj())
        )

        tau49 = (
            - einsum("ijab->ijab", tau27)
            + einsum("ijab->ijab", tau30)
            + einsum("ijab->ijab", tau31)
            + einsum("ijab->ijab", tau33)
            + einsum("ijab->ijab", tau43)
            - einsum("ijab->ijab", tau45)
            + einsum("ijab->ijab", tau47)
            + einsum("ijab->ijab", tau48)
        )

        tau50 = (
            einsum("bi,jkab->ijka", a.t1, tau32)
        )

        tau51 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau52 = (
            einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau53 = (
            - einsum("ijka->ijka", tau8)
            + 2 * einsum("ikja->ijka", tau8)
        )

        tau54 = (
            einsum("balk,ijlb->ijka", a.t2, tau53)
        )

        tau55 = (
            einsum("bi,bkja->ijka", a.t1, h.v.voov.conj())
        )

        tau56 = (
            einsum("bi,jkab->ijka", a.t1, tau13)
        )

        tau57 = (
            einsum("ikja->ijka", tau50)
            - einsum("ijka->ijka", tau51)
            + einsum("ikja->ijka", tau52)
            + einsum("ijka->ijka", tau54)
            + einsum("ijka->ijka", tau55)
            + einsum("jaik->ijka", h.v.ovoo)
            - einsum("ikja->ijka", tau56)
        )

        tau58 = (
            einsum("ak,ikjb->ijab", a.t1, tau57)
        )

        tau59 = (
            einsum("bakj,kiab->ij", a.t2, tau22)
        )

        tau60 = (
            einsum("ak,ikja->ij", a.t1, tau6)
        )

        tau61 = (
            einsum("ij->ij", tau59)
            + einsum("ij->ij", tau60)
        )

        tau62 = (
            einsum("kj,baki->ijab", tau61, a.t2)
        )

        tau63 = (
            einsum("iabc->iabc", h.v.ovvv)
            - 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau64 = (
            einsum("ci,iabc->ab", a.t1, tau63)
        )

        tau65 = (
            einsum("bc,caji->ijab", tau64, a.t2)
        )

        tau66 = (
            - 2 * einsum("iabj->ijab", h.v.ovvo)
            + einsum("iajb->ijab", h.v.ovov)
        )

        tau67 = (
            einsum("caki,kjbc->ijab", a.t2, tau66)
        )

        tau68 = (
            einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau69 = (
            einsum("ci,jabc->ijab", a.t1, tau68)
        )

        tau70 = (
            einsum("acki,cjkb->ijab", a.t2, h.v.voov.conj())
        )

        tau71 = (
            einsum("ijab->ijab", tau58)
            + einsum("ijab->ijab", tau62)
            + einsum("ijab->ijab", tau65)
            + einsum("ijab->ijab", tau67)
            + einsum("ijab->ijab", tau69)
            + einsum("ijab->ijab", tau70)
        )

        g1 = (
            einsum("jb,jiba->ai", tau3, tau4)
            - einsum("aj,ji->ai", a.t1, tau7)
            + einsum("bakj,ijkb->ai", a.t2, tau9)
            + einsum("jicb,jabc->ai", tau5, h.v.ovvv)
            + einsum("bj,jiab->ai", a.t1, tau10)
            + einsum("bi,ab->ai", a.t1, tau12)
            + einsum("ia->ai", h.f.ov.conj())
        )

        g2 = (
            einsum("ikac,kjbc->abij", tau13, tau14)
            + einsum("ac,cbij->abij", tau16, a.t2)
            + einsum("bk,kjia->abij", a.t1, tau20)
            + einsum("jiba->abij", h.v.oovv.conj())
            + einsum("bc,caji->abij", tau16, a.t2)
            + einsum("abkl,lkji->abij", a.t2, tau19)
            + einsum("dcij,abdc->abij", a.t2, h.v.vvvv)
            + einsum("cbkj,ikac->abij", a.t2, tau23)
            + einsum("caki,jkbc->abij", a.t2, tau25)
            + einsum("ackj,ikbc->abij", a.t2, tau21)
            + einsum("cj,ibac->abij", a.t1, tau26)
            + einsum("ijba->abij", tau49)
            + einsum("jiab->abij", tau49)
            - einsum("ijab->abij", tau71)
            - einsum("jiba->abij", tau71)
        )

        e_ai = cc_denom(h.f, 2, 'dir', 'full')
        e_abij = cc_denom(h.f, 4, 'dir', 'full')

        g1 = g1 - a.t1 / e_ai
        g2 = g2 - a.t2 / e_abij

        return self.RHS_TYPE(g1=g1, g2=g2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.AMPLITUDES_TYPE(
            *(g[ii] * (- cc_denom(h.f, g[ii].ndim, 'dir', 'full'))
              for ii in range(len(g)))
        )

    def calc_residuals(self, h, a, g):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.RESIDUALS_TYPE(
            *[a[ii] / cc_denom(h.f, a[ii].ndim, 'dir', 'full') + g[ii]
              for ii in range(len(a))]
        )

    def calculate_extended_energy(self, h, ea):
        """
        Calculates energy from extended amplitudes -
        simply unwraps T
        """

        return self.calculate_energy(h, ea.t)

    def calculate_lagrangian(self, h, ea):
        """
        Calculates CC Lagrangian
        """
        pass

    def lagrangian_gradient(self, h, ea):
        """
        Calculates gradient of the CC Lagrangian
        """
        pass


class RCCSD_UNIT(RCCSD):
    """
    RCCSD equations based entirely on the unitary group operators
    """
    @property
    def method_name(self):
        return 'RCCSD_UNIT'

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        tau0 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        energy = (
            einsum("abij,jiba->", a.t2, tau0)
            + einsum("ai,ia->", a.t1, tau1)
        )
        return energy

    def update_rhs(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        tau0 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        tau2 = (
            einsum("bj,jiba->ia", a.t1, tau0)
        )

        tau3 = (
            einsum("ia->ia", tau2)
            + einsum("ia->ia", h.f.ov)
        )

        tau4 = (
            - einsum("abji->ijab", a.t2)
            + 2 * einsum("baji->ijab", a.t2)
        )

        tau5 = (
            - einsum("ijka->ijka", h.v.ooov)
            + 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau6 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau7 = (
            einsum("ij->ij", h.f.oo)
            + einsum("aj,ia->ij", a.t1, tau3)
            + einsum("ak,kija->ij", a.t1, tau5)
            + einsum("bakj,kiab->ij", a.t2, tau6)
        )

        tau8 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau9 = (
            einsum("jkia->ijka", h.v.ooov)
            - 2 * einsum("kjia->ijka", h.v.ooov)
            - 2 * einsum("ijka->ijka", tau8)
            + einsum("ikja->ijka", tau8)
        )

        tau10 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", h.v.ovvv)
        )

        tau11 = (
            einsum("ci,iacb->ab", a.t1, tau10)
            + einsum("ab->ab", h.f.vv)
        )

        tau12 = (
            - einsum("iajb->ijab", h.v.ovov)
            + 2 * einsum("iabj->ijab", h.v.ovvo)
        )

        tau13 = (
            2 * einsum("abji->ijab", a.t2)
            - einsum("baji->ijab", a.t2)
        )

        tau14 = (
            einsum("acki,kjbc->ijab", a.t2, tau6)
        )

        tau15 = (
            einsum("acki,kjbc->ijab", a.t2, tau0)
        )

        tau16 = (
            einsum("jica,ijbc->ab", tau4, h.v.oovv)
        )

        tau17 = (
            einsum("bc,caji->ijab", tau16, a.t2)
        )

        tau18 = (
            einsum("caki,kjbc->ijab", a.t2, tau6)
        )

        tau19 = (
            einsum("bckj,ikac->ijab", a.t2, tau18)
        )

        tau20 = (
            einsum("ijab->ijab", tau17)
            + einsum("ijab->ijab", tau19)
        )

        tau21 = (
            - 2 * einsum("abji->ijab", a.t2)
            + einsum("baji->ijab", a.t2)
        )

        tau22 = (
            einsum("kiac,kbcj->ijab", tau21, h.v.ovvo)
        )

        tau23 = (
            - einsum("ijka->ijka", tau8)
            + 2 * einsum("ikja->ijka", tau8)
        )

        tau24 = (
            einsum("balk,ijlb->ijka", a.t2, tau23)
        )

        tau25 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau26 = (
            einsum("bi,jkab->ijka", a.t1, tau25)
        )

        tau27 = (
            einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau28 = (
            einsum("bi,jkab->ijka", a.t1, tau27)
        )

        tau29 = (
            einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
        )

        tau30 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau31 = (
            einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau32 = (
            einsum("ijka->ijka", tau24)
            + einsum("ikja->ijka", tau26)
            + einsum("jaik->ijka", h.v.ovoo)
            - einsum("ikja->ijka", tau28)
            + einsum("ijka->ijka", tau29)
            - einsum("ijka->ijka", tau30)
            + einsum("ikja->ijka", tau31)
        )

        tau33 = (
            einsum("ak,ikjb->ijab", a.t1, tau32)
        )

        tau34 = (
            einsum("ak,kija->ij", a.t1, tau5)
        )

        tau35 = (
            einsum("kjba,kiba->ij", tau4, h.v.oovv)
        )

        tau36 = (
            einsum("ij->ij", tau34)
            + einsum("ij->ij", tau35)
        )

        tau37 = (
            einsum("kj,baki->ijab", tau36, a.t2)
        )

        tau38 = (
            - 2 * einsum("iabc->iabc", h.v.ovvv)
            + einsum("iacb->iabc", h.v.ovvv)
        )

        tau39 = (
            einsum("ci,iacb->ab", a.t1, tau38)
        )

        tau40 = (
            einsum("bc,caji->ijab", tau39, a.t2)
        )

        tau41 = (
            einsum("caki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau42 = (
            einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau43 = (
            einsum("ci,jabc->ijab", a.t1, tau42)
        )

        tau44 = (
            einsum("ijab->ijab", tau22)
            + einsum("ijab->ijab", tau33)
            + einsum("ijab->ijab", tau37)
            + einsum("ijab->ijab", tau40)
            + einsum("ijab->ijab", tau41)
            + einsum("ijab->ijab", tau43)
        )

        tau45 = (
            2 * einsum("cbkj,ikac->ijab", a.t2, tau18)
        )

        tau46 = (
            einsum("ai,jkla->ijkl", a.t1, tau8)
        )

        tau47 = (
            einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau48 = (
            einsum("lkji->ijkl", tau46)
            + einsum("lkji->ijkl", tau47)
            + einsum("jilk->ijkl", h.v.oooo)
        )

        tau49 = (
            einsum("balk,lkji->ijab", a.t2, tau48)
        )

        tau50 = (
            einsum("al,likj->ijka", a.t1, tau48)
        )

        tau51 = (
            einsum("bk,kjia->ijab", a.t1, tau50)
        )

        tau52 = (
            einsum("di,badc->iabc", a.t1, h.v.vvvv)
        )

        tau53 = (
            einsum("ci,jabc->ijab", a.t1, tau52)
        )

        tau54 = (
            einsum("dcji,badc->ijab", a.t2, h.v.vvvv)
        )

        tau55 = (
            einsum("ijab->ijab", tau45)
            + einsum("ijab->ijab", tau49)
            + einsum("baji->ijab", h.v.vvoo)
            + einsum("ijab->ijab", tau51)
            + einsum("ijab->ijab", tau53)
            + einsum("jiba->ijab", tau54)
        )

        tau56 = (
            einsum("abji->ijab", a.t2)
            - 2 * einsum("baji->ijab", a.t2)
        )

        tau57 = (
            einsum("jica,ijbc->ab", tau56, h.v.oovv)
        )

        tau58 = (
            einsum("bc,caji->ijab", tau57, a.t2)
        )

        tau59 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau60 = (
            einsum("caki,kjbc->ijab", a.t2, tau59)
        )

        tau61 = (
            einsum("bckj,ikac->ijab", a.t2, tau60)
        )

        tau62 = (
            einsum("ijab->ijab", tau58)
            + einsum("ijab->ijab", tau61)
        )

        tau63 = (
            - einsum("ib,bakj->ijka", tau3, a.t2)
        )

        tau64 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau65 = (
            einsum("balj,likb->ijka", a.t2, tau64)
        )

        tau66 = (
            einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau67 = (
            einsum("bi,jkab->ijka", a.t1, tau66)
        )

        tau68 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau69 = (
            einsum("al,ijlk->ijka", a.t1, tau68)
        )

        tau70 = (
            einsum("bi,jakb->ijka", a.t1, h.v.ovov)
        )

        tau71 = (
            einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau72 = (
            einsum("ijka->ijka", tau63)
            + einsum("ijka->ijka", tau65)
            + einsum("jkia->ijka", tau67)
            + einsum("jika->ijka", tau69)
            - einsum("jika->ijka", tau70)
            + einsum("jika->ijka", tau71)
        )

        tau73 = (
            - einsum("ak,kijb->ijab", a.t1, tau72)
        )

        tau74 = (
            einsum("ikbc,kjca->ijab", tau25, tau56)
        )

        tau75 = (
            einsum("ci,abjc->ijab", a.t1, h.v.vvov)
        )

        tau76 = (
            einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
        )

        tau77 = (
            einsum("ci,jabc->ijab", a.t1, tau76)
        )

        tau78 = (
            einsum("ai,ja->ij", a.t1, tau3)
        )

        tau79 = (
            einsum("ji->ij", h.f.oo)
            + einsum("ij->ij", tau78)
        )

        tau80 = (
            einsum("ik,bakj->ijab", tau79, a.t2)
        )

        tau81 = (
            einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau82 = (
            einsum("ci,jabc->ijab", a.t1, tau81)
        )

        tau83 = (
            einsum("ac,cbji->ijab", h.f.vv, a.t2)
        )

        tau84 = (
            einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau85 = (
            einsum("ijab->ijab", tau73)
            + einsum("ijab->ijab", tau74)
            - einsum("ijab->ijab", tau75)
            - einsum("ijab->ijab", tau77)
            + einsum("ijab->ijab", tau80)
            + einsum("ijab->ijab", tau82)
            - einsum("ijab->ijab", tau83)
            + einsum("ijab->ijab", tau84)
        )

        g1 = (
            2 * einsum("jb,jiba->ai", tau3, tau4)
            - 2 * einsum("aj,ji->ai", a.t1, tau7)
            + 2 * einsum("bakj,ikjb->ai", a.t2, tau9)
            + 2 * einsum("bi,ab->ai", a.t1, tau11)
            + 2 * einsum("bj,jiab->ai", a.t1, tau12)
            + 2 * einsum("ia->ai", h.f.ov.conj())
            + 2 * einsum("jicb,jabc->ai", tau13, h.v.ovvv)
        )

        g2 = (
            2 * einsum("bckj,ikac->abij", a.t2, tau14)
            + 2 * einsum("bcki,jkac->abij", a.t2, tau15)
            + 2 * einsum("ijba->abij", tau20)
            - 4 * einsum("jiba->abij", tau20)
            - 4 * einsum("ijab->abij", tau44)
            + 2 * einsum("ijba->abij", tau44)
            + 2 * einsum("jiab->abij", tau44)
            - 4 * einsum("jiba->abij", tau44)
            - 2 * einsum("ijba->abij", tau55)
            + 4 * einsum("ijab->abij", tau55)
            + 4 * einsum("ijab->abij", tau62)
            - 2 * einsum("jiab->abij", tau62)
            + 2 * einsum("ijab->abij", tau85)
            - 4 * einsum("ijba->abij", tau85)
            - 4 * einsum("jiab->abij", tau85)
            + 2 * einsum("jiba->abij", tau85)
        )

        e_ai = cc_denom(h.f, 2, 'dir', 'full')
        e_abij = cc_denom(h.f, 4, 'dir', 'full')

        g1 = g1 - 2 * a.t1 / e_ai
        g2 = g2 - 2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])) / e_abij

        return self.RHS_TYPE(g1=g1, g2=g2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.AMPLITUDES_TYPE(
            t1=g.g1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full'),
            t2=(2 * g.g2 + g.g2.transpose([0, 1, 3, 2])
                ) / (- 6) * cc_denom(h.f, 4, 'dir', 'full')
        )

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = 2 * ham.f.ov.transpose().conj() * (- e_ai)
        t2 = 2 * ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)

        return self.AMPLITUDES_TYPE(t1, t2)

    def calc_residuals(self, h, a, g):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.RESIDUALS_TYPE(
            r1=2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full') + g.g1,
            r2=2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                    ) / cc_denom(h.f, 4, 'dir', 'full') + g.g2
        )


def test_mp2_energy():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz', }
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    CCobj = RCCSD(rhf)
    h = CCobj.create_ham()
    a = CCobj.init_amplitudes(h)
    energy = CCobj.calculate_energy(h, a)
    print('E_mp2 - E_cc,init = {:18.12g}'.format(energy - -0.204019967288338))


def test_cc_unitary():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz', }
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)
    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_energy=-1, conv_tol_res=1e-10)


def test_cc_hubbard():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 6, 6, 2, 'n')
    rhf.damp = -1.0
    rhf.scf()

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)
    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_energy=-1, conv_tol_res=1e-10)

if __name__ == '__main__':
    test_mp2_energy()
    test_cc_unitary()
