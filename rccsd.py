import numpy as np
from numpy import einsum
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace


class RCCSD(CC):
    """
    This class implements classic RCCSD method
    with vvoo ordered amplitudes
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
                                                field_names=('t1', 't2'))
        self.types.RHS_TYPE = namedtuple('RCCSD_RHS_FULL',
                                         field_names=('g1', 'g2'))
        self.types.RESIDUALS_TYPE = namedtuple('RCCSD_RESIDUALS_FULL',
                                               field_names=('r1', 'r2'))
        self.types.EXTENDED_AMPLITUDES_TYPE = namedtuple(
            'RCCSD_EXTENDED_AMPLITUDES', field_names=('t1', 'z1', 't2', 'z2'))

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

        return self.types.AMPLITUDES_TYPE(t1, t2)

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

        return self.types.RHS_TYPE(g1=g1, g2=g2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.types.AMPLITUDES_TYPE(
            *(g[ii] * (- cc_denom(h.f, g[ii].ndim, 'dir', 'full'))
              for ii in range(len(g)))
        )

    def calc_residuals(self, h, a, g):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.types.RESIDUALS_TYPE(
            *[a[ii] / cc_denom(h.f, a[ii].ndim, 'dir', 'full') + g[ii]
              for ii in range(len(a))]
        )

    def init_extended_amps(self, h):
        """
        Initializes extended amplitudes
        """

        amps = self.init_amplitudes(h)
        return self.types.EXTENDED_AMPLITUDES_TYPE(
            t1=amps.t1,
            z1=amps.t1.conj(),
            t2=amps.t2,
            z2=amps.t2.conj()
        )

    def calculate_extended_energy(self, h, ea):
        """
        Calculates energy from extended amplitudes -
        simply unwraps T
        """

        return self.calculate_energy(h, self.types.AMPLITUDES_TYPE(
            ea.t1, ea.t2))

    def calculate_lagrangian(self, h, ea):
        """
        Calculates CC Lagrangian
        """
        tau0 = (
            einsum("abik,kjab->ij", ea.t2, h.v.oovv)
        )

        tau1 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
        )

        tau2 = (
            einsum("abki,jkba->ij", ea.t2, tau1)
        )

        tau3 = (
            einsum("ak,ikja->ij", ea.t1, h.v.ooov)
        )

        tau4 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("baij->ijab", ea.z2)
            + einsum("abji->ijab", ea.z2)
            - 2 * einsum("baji->ijab", ea.z2)
        )

        tau5 = (
            einsum("abki,jkba->ij", ea.t2, tau4)
        )

        tau6 = (
            einsum("abij->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau7 = (
            einsum("bi,kjba->ijka", ea.t1, h.v.oovv)
        )

        tau8 = (
            einsum("abij->ijab", ea.t2)
            + einsum("baji->ijab", ea.t2)
        )

        tau9 = (
            einsum("iklb,jlba->ijka", tau7, tau8)
        )

        tau10 = (
            einsum("bi,jabk->ijka", ea.t1, h.v.ovvo)
        )

        tau11 = (
            einsum("ib,abjk->ijka", h.f.ov, ea.t2)
        )

        tau12 = (
            einsum("bi,jakb->ijka", ea.t1, h.v.ovov)
        )

        tau13 = (
            einsum("ib,bajk->ijka", h.f.ov, ea.t2)
        )

        tau14 = (
            einsum("ijka->ijka", tau7)
            - 2 * einsum("ikja->ijka", tau7)
        )

        tau15 = (
            einsum("iklb,jlab->ijka", tau14, tau8)
        )

        tau16 = (
            einsum("ilkb,jlba->ijka", tau7, tau8)
        )

        tau17 = (
            einsum("ijka->ijka", tau15)
            + einsum("ijka->ijka", tau16)
        )

        tau18 = (
            - einsum("jkia->ijka", tau9)
            + 2 * einsum("kjia->ijka", tau9)
            - 4 * einsum("jika->ijka", tau10)
            + 2 * einsum("kija->ijka", tau10)
            + einsum("ijka->ijka", tau11)
            - 2 * einsum("ikja->ijka", tau11)
            + 2 * einsum("jika->ijka", tau12)
            - 4 * einsum("kija->ijka", tau12)
            - 2 * einsum("ijka->ijka", tau13)
            + einsum("ikja->ijka", tau13)
            + 2 * einsum("jkia->ijka", tau17)
            - einsum("kjia->ijka", tau17)
        )

        tau19 = (
            einsum("ikjb,jkba->ia", tau18, tau6)
        )

        tau20 = (
            - einsum("abji->ijab", h.v.vvoo)
            + 2 * einsum("baji->ijab", h.v.vvoo)
        )

        tau21 = (
            einsum("bi,abjk->ijka", ea.t1, ea.z2)
        )

        tau22 = (
            einsum("bi,bajk->ijka", ea.t1, ea.z2)
        )

        tau23 = (
            - 2 * einsum("ijka->ijka", tau21)
            + einsum("ikja->ijka", tau21)
            + einsum("ijka->ijka", tau22)
            - 2 * einsum("ikja->ijka", tau22)
        )

        tau24 = (
            einsum("bajk,ijkb->ia", ea.t2, tau23)
        )

        tau25 = (
            einsum("bj,jiba->ia", ea.t1, h.v.oovv)
        )

        tau26 = (
            einsum("ia,aj->ij", h.f.ov, ea.t1)
        )

        tau27 = (
            2 * einsum("ij->ij", tau26)
            - einsum("ji->ij", tau0)
            + 2 * einsum("ij->ij", h.f.oo)
        )

        tau28 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("baij->ijab", ea.t2)
        )

        tau29 = (
            einsum("ai,aj->ij", ea.t1, ea.z1)
        )

        tau30 = (
            einsum("ikab,abkj->ij", tau28, ea.z2)
            + 4 * einsum("ij->ij", tau29)
        )

        tau31 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
            - 2 * einsum("abji->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau32 = (
            einsum("acki,kbcj->ijab", ea.t2, h.v.ovvo)
            + einsum("ackj,kbic->ijab", ea.t2, h.v.ovov)
        )

        tau33 = (
            einsum("abki,bajk->ij", ea.t2, ea.z2)
        )

        tau34 = (
            einsum("jiab->ijab", h.v.oovv)
            - 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau35 = (
            einsum("bj,jiba->ia", ea.t1, tau34)
        )

        tau36 = (
            einsum("ai,ja->ij", ea.t1, tau35)
        )

        tau37 = (
            einsum("acik,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau38 = (
            einsum("ijab->ijab", tau37)
            - 2 * einsum("ijba->ijab", tau37)
            + einsum("jiba->ijab", tau37)
        )

        tau39 = (
            einsum("caki,cbkj->ijab", ea.t2, ea.z2)
        )

        tau40 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau41 = (
            2 * einsum("jabi->ijab", h.v.ovvo)
            + einsum("acik,kjcb->ijab", ea.t2, tau40)
            - einsum("jaib->ijab", h.v.ovov)
        )

        tau42 = (
            einsum("acik,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau43 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau44 = (
            einsum("caij,jicb->ab", ea.t2, h.v.oovv)
        )

        tau45 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("abji->ijab", ea.t2)
        )

        tau46 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
        )

        tau47 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
            + einsum("caij,jicb->ab", ea.t2, tau46)
        )

        tau48 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("baij->ijab", ea.t2)
        )

        tau49 = (
            einsum("acik,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau50 = (
            einsum("kica,bcjk->ijab", tau49, ea.z2)
        )

        tau51 = (
            einsum("abij,abkl->ijkl", ea.t2, ea.z2)
        )

        tau52 = (
            einsum("klij,lkba->ijab", tau51, h.v.oovv)
        )

        tau53 = (
            einsum("bj,ijba->ia", ea.t1, tau52)
        )

        tau54 = (
            einsum("cdij,badc->ijab", ea.t2, h.v.vvvv)
        )

        tau55 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
            - einsum("baji->ijab", ea.z2)
        )

        tau56 = (
            2 * einsum("ijka->ijka", tau21)
            - einsum("ikja->ijka", tau21)
            - einsum("ijka->ijka", tau22)
            + 2 * einsum("ikja->ijka", tau22)
        )

        tau57 = (
            einsum("abjk,ikjb->ia", ea.t2, tau56)
        )

        tau58 = (
            einsum("bj,jiab->ia", ea.t1, h.v.oovv)
        )

        tau59 = (
            einsum("caik,cbkj->ijab", ea.t2, ea.z2)
        )

        tau60 = (
            einsum("jibc,jcba->ia", tau59, h.v.ovvv)
        )

        tau61 = (
            einsum("caij,bcij->ab", ea.t2, ea.z2)
        )

        tau62 = (
            einsum("bc,icba->ia", tau61, h.v.ovvv)
        )

        tau63 = (
            einsum("bcij,kabc->ijka", ea.t2, h.v.ovvv)
        )

        tau64 = (
            einsum("jkib,abkj->ia", tau63, ea.z2)
        )

        tau65 = (
            einsum("abik,kjba->ij", ea.t2, h.v.oovv)
        )

        tau66 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
        )

        tau67 = (
            einsum("abik,jkba->ij", ea.t2, tau66)
            + einsum("ikab,abkj->ij", tau28, ea.z2)
            + 4 * einsum("ij->ij", tau29)
        )

        tau68 = (
            einsum("caik,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau69 = (
            einsum("ijab->ijab", tau49)
            + einsum("ijab->ijab", tau68)
        )

        tau70 = (
            einsum("caik,kjcb->ijab", ea.t2, tau1)
            + 2 * einsum("acki,cbkj->ijab", ea.t2, ea.z2)
        )

        tau71 = (
            einsum("kica,bckj->ijab", tau49, ea.z2)
        )

        tau72 = (
            2 * einsum("kica,cbjk->ijab", tau69, ea.z2)
            - 4 * einsum("kica,cbkj->ijab", tau49, ea.z2)
            - 2 * einsum("kjcb,kica->ijab", tau70, h.v.oovv)
            + 2 * einsum("ijab->ijab", tau71)
            - einsum("ijba->ijab", tau71)
        )

        tau73 = (
            einsum("acik,bcjk->ijab", ea.t2, ea.z2)
        )

        tau74 = (
            einsum("jibc,jcab->ia", tau73, h.v.ovvv)
        )

        tau75 = (
            einsum("ai,ja->ij", ea.t1, tau58)
        )

        tau76 = (
            einsum("abik,jkba->ij", ea.t2, tau6)
        )

        tau77 = (
            einsum("abij,bakl->ijkl", ea.t2, ea.z2)
        )

        tau78 = (
            einsum("acki,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau79 = (
            2 * einsum("kiac,kjcb->ijab", tau45, h.v.oovv)
            + einsum("ijab->ijab", tau78)
        )

        tau80 = (
            einsum("caik,cbjk->ijab", ea.t2, ea.z2)
        )

        tau81 = (
            einsum("caki,bckj->ijab", ea.t2, ea.z2)
        )

        tau82 = (
            - einsum("ijab->ijab", tau80)
            + 2 * einsum("ijab->ijab", tau81)
        )

        tau83 = (
            einsum("caik,bckj->ijab", ea.t2, ea.z2)
        )

        tau84 = (
            einsum("caki,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau85 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("abji->ijab", ea.z2)
        )

        tau86 = (
            einsum("caik,jkbc->ijab", ea.t2, tau85)
        )

        tau87 = (
            einsum("kjcb,kica->ijab", tau86, h.v.oovv)
        )

        tau88 = (
            einsum("kjcb,kiac->ijab", tau80, h.v.oovv)
        )

        tau89 = (
            - einsum("kica,cbkj->ijab", tau79, ea.z2)
            + einsum("kjcb,kica->ijab", tau82, h.v.oovv)
            + 2 * einsum("kjcb,kiac->ijab", tau83, h.v.oovv)
            + einsum("kica,kjcb->ijab", tau84, tau85)
            + einsum("ijab->ijab", tau87)
            - 2 * einsum("ijba->ijab", tau87)
            + 2 * einsum("ijab->ijab", tau88)
            - einsum("jiab->ijab", tau88)
        )

        tau90 = (
            einsum("abik,abjk->ij", ea.t2, ea.z2)
        )

        tau91 = (
            einsum("abki,kjab->ij", ea.t2, h.v.oovv)
        )

        tau92 = (
            einsum("bj,jiba->ia", ea.t1, tau40)
        )

        tau93 = (
            einsum("ij->ij", tau91)
            + einsum("ai,ja->ij", ea.t1, tau92)
        )

        tau94 = (
            einsum("abil,ljkb->ijka", ea.t2, h.v.ooov)
        )

        tau95 = (
            einsum("jikb,bajk->ia", tau94, ea.z2)
        )

        tau96 = (
            einsum("ij->ij", tau26)
            + einsum("ij->ij", h.f.oo)
        )

        tau97 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
            + 2 * einsum("baji->ijab", ea.z2)
        )

        tau98 = (
            einsum("abki,kjab->ij", ea.t2, tau97)
        )

        tau99 = (
            einsum("bail,jlkb->ijka", ea.t2, h.v.ooov)
        )

        tau100 = (
            einsum("jikb,bakj->ia", tau99, ea.z2)
        )

        tau101 = (
            einsum("caki,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau102 = (
            einsum("acki,jkbc->ijab", ea.t2, tau101)
        )

        tau103 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
        )

        tau104 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", h.v.ovvv)
        )

        tau105 = (
            einsum("ci,iacb->ab", ea.t1, tau104)
        )

        tau106 = (
            einsum("jicb,jabc->ia", tau28, h.v.ovvv)
            + 2 * einsum("bi,ab->ia", ea.t1, tau105)
        )

        tau107 = (
            einsum("jkil,kjla->ia", tau51, h.v.ooov)
        )

        tau108 = (
            einsum("jkli,kjla->ia", tau51, h.v.ooov)
        )

        tau109 = (
            einsum("acik,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau110 = (
            einsum("bckj,ikac->ijab", ea.t2, tau109)
        )

        tau111 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("abji->ijab", ea.z2)
        )

        tau112 = (
            einsum("acij,jibc->ab", ea.t2, h.v.oovv)
        )

        tau113 = (
            2 * einsum("abij->ijab", ea.t2)
            - einsum("abji->ijab", ea.t2)
        )

        tau114 = (
            einsum("ijac,bcij->ab", tau113, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau85)
        )

        tau115 = (
            einsum("abij,lkba->ijkl", ea.t2, h.v.oovv)
        )

        tau116 = (
            einsum("klij,abkl->ijab", tau115, ea.z2)
        )

        tau117 = (
            einsum("bj,ijab->ia", ea.t1, tau116)
        )

        tau118 = (
            einsum("abki,kjba->ij", ea.t2, h.v.oovv)
        )

        tau119 = (
            einsum("abki,jkba->ij", ea.t2, tau97)
        )

        tau120 = (
            einsum("bcij,kacb->ijka", ea.t2, h.v.ovvv)
        )

        tau121 = (
            einsum("jkib,bajk->ia", tau120, ea.z2)
        )

        tau122 = (
            einsum("bail,ljkb->ijka", ea.t2, h.v.ooov)
        )

        tau123 = (
            einsum("jikb,abjk->ia", tau122, ea.z2)
        )

        tau124 = (
            einsum("bcja,bcij->ia", h.v.vvov, ea.z2)
        )

        tau125 = (
            einsum("caki,jkbc->ijab", ea.t2, tau101)
        )

        tau126 = (
            einsum("acik,cbkj->ijab", ea.t2, ea.z2)
        )

        tau127 = (
            einsum("jkba,jikb->ia", tau126, h.v.ooov)
        )

        tau128 = (
            einsum("bj,baji->ia", ea.z1, ea.t2)
        )

        tau129 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
            - einsum("abji->ijab", ea.z2)
            + 2 * einsum("baji->ijab", ea.z2)
        )

        tau130 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau129)
            + einsum("caij,jicb->ab", ea.t2, tau46)
        )

        tau131 = (
            einsum("caij,jibc->ab", ea.t2, h.v.oovv)
        )

        tau132 = (
            - 2 * einsum("abij->ijab", ea.t2)
            + einsum("abji->ijab", ea.t2)
        )

        tau133 = (
            einsum("ijac,bcij->ab", tau132, ea.z2)
        )

        tau134 = (
            einsum("ai,jkla->ijkl", ea.t1, tau21)
        )

        tau135 = (
            - einsum("ijkl->ijkl", tau134)
            + 2 * einsum("ijlk->ijkl", tau134)
            + 2 * einsum("jikl->ijkl", tau134)
            - einsum("jilk->ijkl", tau134)
        )

        tau136 = (
            einsum("bi,jkba->ijka", ea.t1, tau103)
        )

        tau137 = (
            einsum("ak,ikjb->ijab", ea.t1, tau136)
        )

        tau138 = (
            einsum("al,ijkl->ijka", ea.t1, tau134)
        )

        tau139 = (
            2 * einsum("ijka->ijka", tau138)
            - einsum("jika->ijka", tau138)
        )

        tau140 = (
            einsum("jkil,jkla->ia", tau135, h.v.ooov)
            + einsum("jicb,jbac->ia", tau137, h.v.ovvv)
            + einsum("jkib,kjba->ia", tau139, h.v.oovv)
        )

        tau141 = (
            einsum("caik,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau142 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
        )

        tau143 = (
            einsum("caij,cbij->ab", ea.t2, ea.z2)
        )

        tau144 = (
            einsum("bc,icba->ia", tau143, h.v.ovvv)
        )

        tau145 = (
            einsum("acij,bcij->ab", ea.t2, ea.z2)
        )

        tau146 = (
            einsum("bc,icba->ia", tau145, h.v.ovvv)
        )

        tau147 = (
            einsum("caik,bcjk->ijab", ea.t2, ea.z2)
        )

        tau148 = (
            einsum("jibc,jcba->ia", tau147, h.v.ovvv)
        )

        tau149 = (
            einsum("acik,kjcb->ijab", ea.t2, tau34)
        )

        tau150 = (
            - einsum("ijka->ijka", h.v.ooov)
            + 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau151 = (
            einsum("ak,kija->ij", ea.t1, tau150)
            + einsum("ji->ij", tau91)
        )

        tau152 = (
            einsum("ikab,abkj->ij", tau48, ea.z2)
        )

        tau153 = (
            einsum("jibc,jcab->ia", tau80, h.v.ovvv)
        )

        tau154 = (
            einsum("ikab,abkj->ij", tau28, ea.z2)
        )

        tau155 = (
            einsum("bcja,cbij->ia", h.v.vvov, ea.z2)
        )

        tau156 = (
            einsum("jibc,jcab->ia", tau126, h.v.ovvv)
        )

        tau157 = (
            einsum("caik,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau158 = (
            einsum("caik,jkbc->ijab", ea.t2, tau157)
        )

        tau159 = (
            einsum("acik,cbjk->ijab", ea.t2, ea.z2)
        )

        tau160 = (
            einsum("jibc,jcab->ia", tau159, h.v.ovvv)
        )

        tau161 = (
            einsum("abik,bakj->ij", ea.t2, ea.z2)
        )

        tau162 = (
            einsum("abki,bakj->ij", ea.t2, ea.z2)
        )

        tau163 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau164 = (
            einsum("ak,kija->ij", ea.t1, h.v.ooov)
        )

        tau165 = (
            einsum("abkj,kiba->ij", ea.t2, tau163)
            + 2 * einsum("ij->ij", tau164)
        )

        tau166 = (
            einsum("jkil,jkla->ia", tau77, h.v.ooov)
        )

        tau167 = (
            einsum("adij,jbdc->iabc", ea.t2, h.v.ovvv)
        )

        tau168 = (
            einsum("jbca,cbji->ia", tau167, ea.z2)
        )

        tau169 = (
            einsum("acik,bckj->ijab", ea.t2, ea.z2)
        )

        tau170 = (
            einsum("jibc,jcab->ia", tau169, h.v.ovvv)
        )

        tau171 = (
            2 * einsum("iabj->ijab", h.v.ovvo)
            - einsum("iajb->ijab", h.v.ovov)
        )

        tau172 = (
            einsum("daij,jbcd->iabc", ea.t2, h.v.ovvv)
        )

        tau173 = (
            einsum("jbca,cbji->ia", tau172, ea.z2)
        )

        tau174 = (
            einsum("jkba,jikb->ia", tau169, h.v.ooov)
        )

        tau175 = (
            einsum("caki,jkbc->ijab", ea.t2, tau1)
        )

        tau176 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("baij->ijab", ea.t2)
            + 2 * einsum("abji->ijab", ea.t2)
            - einsum("baji->ijab", ea.t2)
        )

        tau177 = (
            2 * einsum("bj,jiab->ia", ea.t1, tau171)
            + einsum("jb,jiab->ia", h.f.ov, tau176)
        )

        tau178 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("abji->ijab", ea.z2)
        )

        tau179 = (
            einsum("acki,jkbc->ijab", ea.t2, tau178)
        )

        tau180 = (
            einsum("bj,abij->ia", ea.z1, ea.t2)
        )

        tau181 = (
            einsum("ci,jabc->ijab", ea.t1, h.v.ovvv)
        )

        tau182 = (
            einsum("bi,jkab->ijka", ea.t1, tau181)
        )

        tau183 = (
            einsum("jkba,jkib->ia", tau103, tau182)
        )

        tau184 = (
            einsum("caik,jkbc->ijab", ea.t2, tau68)
        )

        tau185 = (
            einsum("adij,jbcd->iabc", ea.t2, h.v.ovvv)
        )

        tau186 = (
            einsum("jbca,cbij->ia", tau185, ea.z2)
        )

        tau187 = (
            einsum("jkil,kjla->ia", tau77, h.v.ooov)
        )

        tau188 = (
            einsum("acij,cbij->ab", ea.t2, ea.z2)
        )

        tau189 = (
            einsum("bc,icab->ia", tau188, h.v.ovvv)
        )

        tau190 = (
            einsum("caki,jkbc->ijab", ea.t2, tau84)
        )

        tau191 = (
            einsum("caki,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau192 = (
            einsum("acij,jicb->ab", ea.t2, h.v.oovv)
        )

        tau193 = (
            einsum("ijac,bcij->ab", tau132, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau178)
        )

        tau194 = (
            einsum("klij,ablk->ijab", tau115, ea.z2)
        )

        tau195 = (
            einsum("bj,ijab->ia", ea.t1, tau194)
        )

        tau196 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("abji->ijab", ea.t2)
        )

        tau197 = (
            einsum("ijca,bcij->ab", tau196, ea.z2)
            + einsum("caij,jicb->ab", ea.t2, tau111)
        )

        tau198 = (
            einsum("acik,jkbc->ijab", ea.t2, tau85)
        )

        tau199 = (
            einsum("jkba,ijkb->ia", tau169, h.v.ooov)
        )

        tau200 = (
            einsum("iajb->ijab", h.v.ovov)
            - 2 * einsum("iabj->ijab", h.v.ovvo)
        )

        tau201 = (
            einsum("jikb,abjk->ia", tau94, ea.z2)
        )

        tau202 = (
            einsum("acij,cbji->ab", ea.t2, ea.z2)
        )

        tau203 = (
            einsum("bc,icab->ia", tau202, h.v.ovvv)
        )

        tau204 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("baij->ijab", ea.z2)
        )

        tau205 = (
            einsum("abki,jkba->ij", ea.t2, tau204)
        )

        tau206 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
        )

        tau207 = (
            einsum("jbca,cbij->ia", tau172, ea.z2)
        )

        tau208 = (
            einsum("jibc,jcba->ia", tau169, h.v.ovvv)
        )

        tau209 = (
            2 * einsum("abij->ijab", ea.t2)
            - einsum("baij->ijab", ea.t2)
        )

        tau210 = (
            einsum("bajk,ikjb->ia", ea.t2, tau14)
            + einsum("ijcb,jabc->ia", tau209, h.v.ovvv)
        )

        tau211 = (
            einsum("jibc,jcba->ia", tau126, h.v.ovvv)
        )

        tau212 = (
            einsum("caij,cbji->ab", ea.t2, ea.z2)
        )

        tau213 = (
            einsum("bc,icab->ia", tau212, h.v.ovvv)
        )

        tau214 = (
            einsum("jbca,bcji->ia", tau185, ea.z2)
        )

        tau215 = (
            einsum("jkba,jikb->ia", tau80, h.v.ooov)
        )

        tau216 = (
            einsum("jbca,bcji->ia", tau172, ea.z2)
        )

        tau217 = (
            einsum("ij->ij", tau164)
            + einsum("aj,ia->ij", ea.t1, tau35)
        )

        tau218 = (
            einsum("jkba,ijkb->ia", tau147, h.v.ooov)
        )

        tau219 = (
            einsum("abik,bajk->ij", ea.t2, ea.z2)
        )

        tau220 = (
            einsum("ai,ja->ij", ea.t1, tau25)
        )

        tau221 = (
            einsum("ij->ij", tau91)
            + 2 * einsum("ij->ij", tau220)
        )

        tau222 = (
            einsum("acki,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau223 = (
            einsum("bcki,jkac->ijab", ea.t2, tau78)
            + einsum("acki,jkbc->ijab", ea.t2, tau222)
        )

        tau224 = (
            einsum("ijca,cbij->ab", tau113, ea.z2)
        )

        tau225 = (
            einsum("acij,bcji->ab", ea.t2, ea.z2)
        )

        tau226 = (
            einsum("bc,icab->ia", tau225, h.v.ovvv)
        )

        tau227 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau228 = (
            einsum("bajk,jkib->ia", ea.t2, tau227)
            + einsum("bj,ijab->ia", ea.t1, tau78)
        )

        tau229 = (
            einsum("jkil,jkla->ia", tau51, h.v.ooov)
        )

        tau230 = (
            einsum("ak,kija->ij", ea.t1, tau150)
        )

        tau231 = (
            - 2 * einsum("ijka->ijka", h.v.ooov)
            + einsum("jika->ijka", h.v.ooov)
        )

        tau232 = (
            einsum("abjk,jkib->ia", ea.t2, tau231)
        )

        tau233 = (
            einsum("kjcb,kiac->ijab", tau147, h.v.oovv)
        )

        tau234 = (
            einsum("ikba,abjk->ij", tau28, ea.z2)
        )

        tau235 = (
            einsum("caki,jkbc->ijab", ea.t2, tau204)
        )

        tau236 = (
            einsum("bc,icab->ia", tau61, h.v.ovvv)
        )

        tau237 = (
            einsum("abki,kjba->ij", ea.t2, tau103)
        )

        tau238 = (
            einsum("ab,bi->ia", h.f.vv, ea.t1)
            + einsum("ia->ia", h.f.ov.conj())
        )

        tau239 = (
            einsum("ibjk,bajk->ia", h.v.ovoo, ea.z2)
        )

        tau240 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("baij->ijab", ea.t2)
            + einsum("baji->ijab", ea.t2)
        )

        tau241 = (
            einsum("ikba,abjk->ij", tau240, ea.z2)
        )

        tau242 = (
            einsum("jibc,jcab->ia", tau59, h.v.ovvv)
        )

        tau243 = (
            einsum("caik,kjcb->ijab", ea.t2, tau1)
        )

        tau244 = (
            einsum("bc,icba->ia", tau225, h.v.ovvv)
        )

        tau245 = (
            einsum("jilk,abkl->ijab", h.v.oooo, ea.z2)
        )

        tau246 = (
            einsum("bj,jiab->ia", ea.t1, tau245)
        )

        tau247 = (
            einsum("abki,abjk->ij", ea.t2, ea.z2)
        )

        tau248 = (
            einsum("ai,ja->ij", ea.t1, tau92)
        )

        tau249 = (
            einsum("jikb,abkj->ia", tau122, ea.z2)
        )

        tau250 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau251 = (
            4 * einsum("ij->ij", tau3)
            + einsum("abkj,kiba->ij", ea.t2, tau250)
        )

        tau252 = (
            einsum("ijac,cbij->ab", tau196, ea.z2)
        )

        tau253 = (
            einsum("jkba,ijkb->ia", tau80, h.v.ooov)
        )

        tau254 = (
            einsum("acki,bcjk->ijab", ea.t2, ea.z2)
        )

        tau255 = (
            einsum("ikba,abjk->ij", tau8, ea.z2)
        )

        tau256 = (
            einsum("jbca,cbji->ia", tau185, ea.z2)
        )

        tau257 = (
            einsum("bc,icab->ia", tau145, h.v.ovvv)
        )

        tau258 = (
            einsum("jkba,jikb->ia", tau83, h.v.ooov)
        )

        tau259 = (
            einsum("ijca,bcij->ab", tau196, ea.z2)
        )

        tau260 = (
            einsum("jikb,bajk->ia", tau122, ea.z2)
        )

        tau261 = (
            einsum("caki,kjcb->ijab", ea.t2, tau129)
        )

        tau262 = (
            einsum("ibjk,abjk->ia", h.v.ovoo, ea.z2)
        )

        tau263 = (
            einsum("bajk,ijkb->ia", ea.t2, tau56)
        )

        tau264 = (
            einsum("jbca,bcij->ia", tau172, ea.z2)
        )

        tau265 = (
            einsum("jibc,jcba->ia", tau80, h.v.ovvv)
        )

        tau266 = (
            einsum("dcba,cdij->ijab", h.v.vvvv, ea.z2)
        )

        tau267 = (
            einsum("bj,ijba->ia", ea.t1, tau266)
        )

        tau268 = (
            einsum("jkib,abjk->ia", tau63, ea.z2)
        )

        tau269 = (
            einsum("ibkj,bajk->ia", h.v.ovoo, ea.z2)
        )

        tau270 = (
            einsum("abjk,ijkb->ia", ea.t2, tau14)
        )

        tau271 = (
            einsum("jkba,jikb->ia", tau59, h.v.ooov)
        )

        tau272 = (
            einsum("jbca,bcij->ia", tau185, ea.z2)
        )

        tau273 = (
            einsum("jkba,ijkb->ia", tau73, h.v.ooov)
        )

        tau274 = (
            einsum("caij,bcji->ab", ea.t2, ea.z2)
        )

        tau275 = (
            einsum("bc,icba->ia", tau274, h.v.ovvv)
        )

        tau276 = (
            einsum("jkba,ijkb->ia", tau159, h.v.ooov)
        )

        tau277 = (
            einsum("jkib,bakj->ia", tau120, ea.z2)
        )

        tau278 = (
            einsum("jkib,abjk->ia", tau120, ea.z2)
        )

        tau279 = (
            einsum("abil,jlkb->ijka", ea.t2, h.v.ooov)
        )

        tau280 = (
            einsum("jikb,abjk->ia", tau279, ea.z2)
        )

        tau281 = (
            einsum("caki,kjcb->ijab", ea.t2, tau85)
        )

        tau282 = (
            einsum("jikb,bakj->ia", tau122, ea.z2)
        )

        tau283 = (
            einsum("caki,kjcb->ijab", ea.t2, tau1)
        )

        tau284 = (
            einsum("ji->ij", tau0)
            - 2 * einsum("ij->ij", h.f.oo)
        )

        tau285 = (
            einsum("abjk,ikjb->ia", ea.t2, tau56)
            + 2 * einsum("bj,baij->ia", ea.z1, ea.t2)
        )

        tau286 = (
            einsum("jbca,cbij->ia", tau167, ea.z2)
        )

        tau287 = (
            einsum("ijca,cbij->ab", tau132, ea.z2)
        )

        tau288 = (
            einsum("caik,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau289 = (
            - 2 * einsum("ijab->ijab", tau288)
            + einsum("ijba->ijab", tau288)
            + einsum("jiab->ijab", tau288)
        )

        tau290 = (
            einsum("bc,icba->ia", tau212, h.v.ovvv)
        )

        tau291 = (
            einsum("jikb,abjk->ia", tau99, ea.z2)
        )

        tau292 = (
            einsum("ijac,cbij->ab", tau45, ea.z2)
        )

        tau293 = (
            einsum("bj,abji->ia", ea.z1, ea.t2)
        )

        tau294 = (
            einsum("jkib,abkj->ia", tau120, ea.z2)
        )

        tau295 = (
            einsum("jikb,bajk->ia", tau99, ea.z2)
        )

        tau296 = (
            einsum("jkli,jkla->ia", tau51, h.v.ooov)
        )

        tau297 = (
            einsum("abki,abkj->ij", ea.t2, ea.z2)
        )

        tau298 = (
            - einsum("ji->ij", tau91)
            + einsum("ij->ij", tau164)
        )

        tau299 = (
            einsum("abki,kjab->ij", ea.t2, tau1)
        )

        tau300 = (
            2 * einsum("ij->ij", tau75)
            + einsum("ij->ij", tau0)
        )

        tau301 = (
            einsum("daij,jbdc->iabc", ea.t2, h.v.ovvv)
        )

        tau302 = (
            einsum("jbca,cbji->ia", tau301, ea.z2)
        )

        tau303 = (
            einsum("ibkj,abjk->ia", h.v.ovoo, ea.z2)
        )

        tau304 = (
            einsum("bj,jiab->ia", ea.t1, tau194)
        )

        tau305 = (
            einsum("bj,ijab->ia", ea.t1, tau266)
        )

        tau306 = (
            einsum("jikb,bajk->ia", tau279, ea.z2)
        )

        tau307 = (
            einsum("bcja,cbji->ia", h.v.vvov, ea.z2)
        )

        tau308 = (
            einsum("jbca,cbij->ia", tau301, ea.z2)
        )

        tau309 = (
            einsum("jikb,abkj->ia", tau279, ea.z2)
        )

        tau310 = (
            einsum("caki,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau311 = (
            einsum("jkib,bajk->ia", tau63, ea.z2)
        )

        tau312 = (
            einsum("jkli,kjla->ia", tau77, h.v.ooov)
        )

        tau313 = (
            einsum("jibc,jcba->ia", tau159, h.v.ovvv)
        )

        tau314 = (
            einsum("jikb,bakj->ia", tau94, ea.z2)
        )

        tau315 = (
            einsum("jkba,jikb->ia", tau147, h.v.ooov)
        )

        tau316 = (
            einsum("jbca,bcji->ia", tau301, ea.z2)
        )

        tau317 = (
            einsum("bc,icab->ia", tau274, h.v.ovvv)
        )

        tau318 = (
            einsum("jikb,abkj->ia", tau99, ea.z2)
        )

        tau319 = (
            einsum("ijac,bcij->ab", tau113, ea.z2)
        )

        tau320 = (
            einsum("jikb,bakj->ia", tau279, ea.z2)
        )

        tau321 = (
            einsum("klij,lkba->ijab", tau77, h.v.oovv)
        )

        tau322 = (
            einsum("bj,ijba->ia", ea.t1, tau321)
        )

        tau323 = (
            einsum("jkib,bakj->ia", tau63, ea.z2)
        )

        tau324 = (
            einsum("bj,jiab->ia", ea.t1, tau116)
        )

        tau325 = (
            einsum("ak,kija->ij", ea.t1, tau227)
        )

        tau326 = (
            einsum("bcja,bcji->ia", h.v.vvov, ea.z2)
        )

        tau327 = (
            einsum("jibc,jcab->ia", tau147, h.v.ovvv)
        )

        tau328 = (
            einsum("jkba,ijkb->ia", tau126, h.v.ooov)
        )

        tau329 = (
            einsum("jkba,jikb->ia", tau159, h.v.ooov)
        )

        tau330 = (
            einsum("bj,ijab->ia", ea.t1, tau245)
        )

        tau331 = (
            einsum("jkba,ijkb->ia", tau59, h.v.ooov)
        )

        tau332 = (
            einsum("jbca,bcji->ia", tau167, ea.z2)
        )

        tau333 = (
            einsum("jkba,jikb->ia", tau73, h.v.ooov)
        )

        tau334 = (
            einsum("abik,abkj->ij", ea.t2, ea.z2)
        )

        tau335 = (
            einsum("jibc,jcab->ia", tau83, h.v.ovvv)
        )

        tau336 = (
            einsum("bc,icba->ia", tau188, h.v.ovvv)
        )

        tau337 = (
            einsum("jbca,bcij->ia", tau301, ea.z2)
        )

        tau338 = (
            einsum("jkba,ijkb->ia", tau83, h.v.ooov)
        )

        tau339 = (
            einsum("bc,icba->ia", tau202, h.v.ovvv)
        )

        tau340 = (
            einsum("jbca,bcij->ia", tau167, ea.z2)
        )

        tau341 = (
            einsum("bj,ijab->ia", ea.t1, tau321)
        )

        tau342 = (
            einsum("bc,icab->ia", tau143, h.v.ovvv)
        )

        tau343 = (
            einsum("jikb,abkj->ia", tau94, ea.z2)
        )

        tau344 = (
            einsum("jkli,jkla->ia", tau77, h.v.ooov)
        )

        tau345 = (
            einsum("bj,ijab->ia", ea.t1, tau52)
        )

        tau346 = (
            einsum("jibc,jcba->ia", tau73, h.v.ovvv)
        )

        tau347 = (
            einsum("jibc,jcba->ia", tau83, h.v.ovvv)
        )

        l = (
            einsum("ij,ji->", tau0, tau2) / 4
            + einsum("ij,ij->", tau3, tau5)
            + einsum("ai,ia->", ea.t1, tau19) / 2
            + einsum("jiba,abij->", tau20, ea.z2)
            + einsum("ia,ia->", tau24, tau25)
            - einsum("ij,ij->", tau27, tau30) / 4
            + einsum("ijab,ijba->", tau31, tau32) / 2
            + einsum("ij,ji->", tau33, tau36)
            + einsum("jiab,abij->", tau38, ea.z2) / 2
            + einsum("ijab,jiba->", tau39, tau41)
            - einsum("ijab,jiab->", tau42, tau43)
            + einsum("ab,ba->", tau44, tau47) / 4
            + einsum("ijba,ijab->", tau48, tau50) / 2
            - einsum("ai,ia->", ea.t1, tau53) / 2
            + einsum("ijab,ijba->", tau54, tau55) / 2
            + einsum("ia,ia->", tau57, tau58) / 2
            - einsum("ai,ia->", ea.t1, tau60)
            + einsum("ai,ia->", ea.t1, tau62) / 2
            + einsum("ai,ia->", ea.t1, tau64) / 2
            - einsum("ij,ji->", tau65, tau67) / 2
            + einsum("abij,ijab->", ea.t2, tau72) / 4
            - einsum("ai,ia->", ea.t1, tau74)
            - einsum("ij,ji->", tau75, tau76) / 2
            - einsum("ijkl,jilk->", tau77, h.v.oooo) / 2
            + einsum("abij,jiab->", ea.t2, tau89) / 4
            - einsum("ij,ji->", tau90, tau93)
            + einsum("ai,ia->", ea.t1, tau95)
            - einsum("ij,ij->", tau96, tau98) / 2
            - einsum("ai,ia->", ea.t1, tau100) / 2
            + einsum("ijab,jiab->", tau102, tau103) / 2
            + einsum("ia,ai->", tau106, ea.z1)
            - einsum("abij,jiab->", ea.t2, h.v.oovv)
            + einsum("ai,ia->", ea.t1, tau107)
            - einsum("ai,ia->", ea.t1, tau108) / 2
            + einsum("ijab,ijba->", tau110, tau111) / 2
            + einsum("ab,ba->", tau112, tau114) / 4
            + einsum("ai,ia->", ea.t1, tau117)
            + einsum("ij,ji->", tau118, tau119) / 4
            - einsum("ai,ia->", ea.t1, tau121)
            + einsum("ai,ia->", ea.t1, tau123)
            - einsum("ai,ia->", ea.t1, tau124)
            + einsum("ijab,ijba->", tau125, tau66)
            + einsum("ai,ia->", ea.t1, tau127)
            + 2 * einsum("ia,ia->", tau128, tau92)
            + einsum("ab,ba->", h.f.vv, tau130) / 2
            + einsum("ab,ba->", tau131, tau133) / 2
            + einsum("ai,ia->", ea.t1, tau140)
            - einsum("ijab,ijab->", tau141, tau142) / 2
            - einsum("ai,ia->", ea.t1, tau144)
            + einsum("jlki,iklj->", tau115, tau51) / 2
            - einsum("ai,ia->", ea.t1, tau146)
            - einsum("ai,ia->", ea.t1, tau148)
            + einsum("jiba,ijab->", tau149, tau81) / 2
            + einsum("ij,ij->", tau151, tau152) / 2
            + einsum("jlki,ikjl->", tau115, tau77) / 2
            - einsum("ai,ia->", ea.t1, tau153)
            + einsum("ij,ji->", tau118, tau154) / 4
            + 2 * einsum("ai,ia->", ea.t1, tau155)
            - einsum("ai,ia->", ea.t1, tau156)
            + einsum("ijab,ijab->", tau158, tau66) / 4
            + einsum("ai,ia->", ea.t1, tau160) / 2
            + einsum("ij,ji->", tau161, tau36)
            - einsum("ij,ij->", tau162, tau165) / 4
            + einsum("ai,ia->", ea.t1, tau166)
            - einsum("ai,ia->", ea.t1, tau168)
            + einsum("ai,ia->", ea.t1, tau170) / 2
            + einsum("ijba,ijab->", tau171, tau73)
            - einsum("ai,ia->", ea.t1, tau173)
            - einsum("ai,ia->", ea.t1, tau174) / 2
            + einsum("ijab,jiba->", tau157, tau175) / 4
            + einsum("ia,ai->", tau177, ea.z1)
            + einsum("ijab,jiba->", tau101, tau179) / 2
            + 2 * einsum("ia,ia->", tau180, tau92)
            + einsum("ai,ia->", ea.t1, tau183)
            + einsum("ijab,ijab->", tau1, tau184) / 4
            - einsum("ai,ia->", ea.t1, tau186)
            - einsum("ai,ia->", ea.t1, tau187) / 2
            - einsum("ai,ia->", ea.t1, tau189)
            + einsum("ijba,ijab->", tau103, tau190) / 2
            + einsum("ijab,ijab->", tau178, tau191) / 2
            + einsum("ab,ba->", tau192, tau193) / 2
            - einsum("ai,ia->", ea.t1, tau195) / 2
            + einsum("ab,ba->", tau131, tau197) / 2
            + einsum("ijab,jiba->", tau157, tau198) / 4
            + einsum("ai,ia->", ea.t1, tau199)
            + einsum("ijab,ijba->", tau169, tau200) / 2
            - einsum("ai,ia->", ea.t1, tau201) / 2
            + 2 * einsum("ai,ia->", ea.t1, tau203)
            + einsum("ji,ij->", tau205, tau65) / 2
            + einsum("ab,ba->", tau112, tau206) / 4
            + einsum("ai,ia->", ea.t1, tau207) / 2
            - einsum("ai,ia->", ea.t1, tau208)
            + einsum("ia,ai->", tau210, ea.z1)
            + 2 * einsum("ai,ia->", ea.t1, tau211)
            - einsum("ai,ia->", ea.t1, tau213)
            - einsum("ai,ia->", ea.t1, tau214)
            + einsum("ai,ia->", ea.t1, tau215)
            + einsum("ijlk,jilk->", tau77, h.v.oooo)
            + einsum("ai,ia->", ea.t1, tau216) / 2
            + 2 * einsum("ij,ij->", tau217, tau29)
            + einsum("ai,ia->", ea.t1, tau218)
            + einsum("ij,ji->", tau219, tau221) / 2
            + einsum("ijba,ijba->", tau1, tau223) / 4
            + einsum("ab,ba->", tau112, tau224) / 4
            - einsum("jlik,iklj->", tau115, tau51) / 4
            - einsum("ai,ia->", ea.t1, tau226)
            - einsum("ijlk,jilk->", tau51, h.v.oooo) / 2
            + einsum("ia,ai->", tau228, ea.z1)
            - einsum("jlik,ikjl->", tau115, tau77) / 4
            - einsum("ai,ia->", ea.t1, tau58)
            - einsum("ai,ia->", ea.t1, tau229) / 2
            + einsum("ij,ij->", tau219, tau230) / 2
            + einsum("jlik,ikjl->", tau115, tau51) / 2
            + einsum("ia,ai->", tau232, ea.z1)
            + einsum("ijab,jiab->", tau233, tau45) / 4
            + einsum("ij,ji->", tau118, tau234) / 4
            + einsum("jiba,ijab->", tau235, tau68) / 2
            - einsum("ai,ia->", ea.t1, tau236)
            + einsum("ji,ij->", tau237, tau65) / 2
            + 2 * einsum("ia,ai->", tau238, ea.z1)
            + einsum("ai,ia->", ea.t1, tau239)
            + einsum("ij,ij->", tau164, tau2) / 2
            + einsum("ij,ij->", tau241, tau26) / 2
            + einsum("ai,ia->", ea.t1, tau242) / 2
            + einsum("jiba,ijab->", tau243, tau84) / 4
            + einsum("ai,ia->", ea.t1, tau244) / 2
            - einsum("ai,ia->", ea.t1, tau246)
            + einsum("ij,ji->", tau247, tau248) / 2
            - einsum("ai,ia->", ea.t1, tau249) / 2
            - einsum("ij,ij->", tau251, tau29)
            + einsum("ab,ba->", tau131, tau252) / 2
            - einsum("ai,ia->", ea.t1, tau253) / 2
            + einsum("ijba,ijab->", tau200, tau254) / 2
            + 2 * einsum("ia,ai->", h.f.ov, ea.t1)
            + einsum("ij,ij->", h.f.oo, tau255) / 2
            + einsum("ai,ia->", ea.t1, tau256) / 2
            + 2 * einsum("ai,ia->", ea.t1, tau257)
            + einsum("ai,ia->", ea.t1, tau258)
            + einsum("ab,ba->", tau192, tau259) / 2
            - einsum("ai,ia->", ea.t1, tau260) / 2
            + einsum("ijab,jiba->", tau101, tau261)
            - 2 * einsum("ai,ia->", ea.t1, tau262)
            + einsum("ia,ia->", tau263, tau58) / 2
            - einsum("ai,ia->", ea.t1, tau264)
            + einsum("ai,ia->", ea.t1, tau265) / 2
            - einsum("ai,ia->", ea.t1, tau267)
            - einsum("ai,ia->", ea.t1, tau268)
            - 2 * einsum("ai,ia->", ea.t1, tau269)
            + einsum("ia,ai->", tau270, ea.z1)
            - einsum("ai,ia->", ea.t1, tau271) / 2
            + einsum("ai,ia->", ea.t1, tau272) / 2
            + einsum("ji,ij->", tau205, tau91) / 2
            - 2 * einsum("ai,ia->", ea.t1, tau273)
            - einsum("ai,ia->", ea.t1, tau275)
            + einsum("ai,ia->", ea.t1, tau276)
            + einsum("ai,ia->", ea.t1, tau277) / 2
            + einsum("ai,ia->", ea.t1, tau278) / 2
            + einsum("ai,ia->", ea.t1, tau280)
            + einsum("jiba,ijab->", tau281, tau78) / 4
            + einsum("ai,ia->", ea.t1, tau282)
            + einsum("ijab,jiba->", tau157, tau283) / 4
            + einsum("ij,ij->", tau284, tau90) / 2
            - einsum("ia,ia->", tau25, tau285)
            + einsum("jlik,iklj->", tau115, tau77) / 2
            + 2 * einsum("ai,ia->", ea.t1, tau286)
            + einsum("ab,ba->", tau192, tau287) / 2
            + einsum("ijab,abij->", tau289, ea.z2) / 2
            + einsum("ai,ia->", ea.t1, tau290) / 2
            - einsum("ai,ia->", ea.t1, tau291) / 2
            + einsum("ba,ab->", tau292, tau44) / 4
            + einsum("ia,ia->", tau293, tau35)
            - einsum("ai,ia->", ea.t1, tau294)
            + einsum("ai,ia->", ea.t1, tau295)
            + einsum("ai,ia->", ea.t1, tau296)
            + einsum("ij,ij->", tau297, tau298)
            + einsum("ij,ji->", tau299, tau300) / 4
            + einsum("ai,ia->", ea.t1, tau302) / 2
            + einsum("ai,ia->", ea.t1, tau303)
            + einsum("ai,ia->", ea.t1, tau304)
            + 2 * einsum("ai,ia->", ea.t1, tau305)
            - einsum("ji,ij->", tau0, tau219) / 4
            - 2 * einsum("ai,ia->", ea.t1, tau306)
            - einsum("ai,ia->", ea.t1, tau307)
            - einsum("ai,ia->", ea.t1, tau308)
            - 2 * einsum("ai,ia->", ea.t1, tau309)
            + einsum("ij,ji->", tau220, tau237)
            + einsum("ijab,ijab->", tau310, tau85)
            + einsum("ai,ia->", ea.t1, tau311) / 2
            + einsum("ai,ia->", ea.t1, tau312)
            - einsum("ai,ia->", ea.t1, tau313)
            - einsum("ai,ia->", ea.t1, tau314) / 2
            - einsum("ai,ia->", ea.t1, tau315) / 2
            - einsum("ai,ia->", ea.t1, tau316)
            + 2 * einsum("ai,ia->", ea.t1, tau317)
            + einsum("ai,ia->", ea.t1, tau318)
            + einsum("ba,ab->", tau319, tau44) / 4
            + einsum("ai,ia->", ea.t1, tau320)
            + einsum("ai,ia->", ea.t1, tau322)
            - einsum("ai,ia->", ea.t1, tau323)
            - einsum("ai,ia->", ea.t1, tau324) / 2
            - einsum("jlki,iklj->", tau115, tau77) / 4
            + einsum("ij,ij->", tau325, tau90)
            + 2 * einsum("ai,ia->", ea.t1, tau326)
            + einsum("ai,ia->", ea.t1, tau327) / 2
            - 2 * einsum("ai,ia->", ea.t1, tau328)
            + 2 * einsum("abij,jiba->", ea.t2, h.v.oovv)
            - einsum("ai,ia->", ea.t1, tau329) / 2
            + 2 * einsum("ai,ia->", ea.t1, tau330)
            + einsum("ai,ia->", ea.t1, tau331)
            + 2 * einsum("ai,ia->", ea.t1, tau332)
            + einsum("ai,ia->", ea.t1, tau333)
            + einsum("ij,ji->", tau220, tau334)
            - einsum("ai,ia->", ea.t1, tau335)
            + einsum("ai,ia->", ea.t1, tau336) / 2
            + einsum("ai,ia->", ea.t1, tau337) / 2
            - einsum("ai,ia->", ea.t1, tau338) / 2
            + einsum("ijkl,jilk->", tau51, h.v.oooo)
            - einsum("ai,ia->", ea.t1, tau339)
            - einsum("ai,ia->", ea.t1, tau340)
            - einsum("ai,ia->", ea.t1, tau341) / 2
            + 2 * einsum("ai,ia->", ea.t1, tau342)
            + 2 * einsum("ai,ia->", ea.t1, tau25)
            + einsum("ai,ia->", ea.t1, tau343)
            - einsum("ai,ia->", ea.t1, tau344) / 2
            + einsum("ai,ia->", ea.t1, tau345)
            - einsum("jlki,ikjl->", tau115, tau51) / 4
            + 2 * einsum("ai,ia->", ea.t1, tau346)
            + einsum("ai,ia->", ea.t1, tau347) / 2
        )

        return np.array([l, ])

    def lagrangian_gradient(self, h, ea):
        """
        Calculates gradient of the CC Lagrangian
        """
        no = self._mos.nocc
        nv = self._mos.nvir

        tau0 = (
            einsum("abik,kjab->ij", ea.t2, h.v.oovv)
        )

        tau1 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
        )

        tau2 = (
            einsum("abki,jkba->ij", ea.t2, tau1)
        )

        tau3 = (
            einsum("ak,ikja->ij", ea.t1, h.v.ooov)
        )

        tau4 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("baij->ijab", ea.z2)
            + einsum("abji->ijab", ea.z2)
            - 2 * einsum("baji->ijab", ea.z2)
        )

        tau5 = (
            einsum("abki,jkba->ij", ea.t2, tau4)
        )

        tau6 = (
            einsum("abij->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau7 = (
            einsum("bi,kjba->ijka", ea.t1, h.v.oovv)
        )

        tau8 = (
            einsum("abij->ijab", ea.t2)
            + einsum("baji->ijab", ea.t2)
        )

        tau9 = (
            einsum("iklb,jlba->ijka", tau7, tau8)
        )

        tau10 = (
            einsum("bi,jabk->ijka", ea.t1, h.v.ovvo)
        )

        tau11 = (
            einsum("ib,abjk->ijka", h.f.ov, ea.t2)
        )

        tau12 = (
            einsum("bi,jakb->ijka", ea.t1, h.v.ovov)
        )

        tau13 = (
            einsum("ib,bajk->ijka", h.f.ov, ea.t2)
        )

        tau14 = (
            einsum("ijka->ijka", tau7)
            - 2 * einsum("ikja->ijka", tau7)
        )

        tau15 = (
            einsum("iklb,jlab->ijka", tau14, tau8)
        )

        tau16 = (
            einsum("ilkb,jlba->ijka", tau7, tau8)
        )

        tau17 = (
            einsum("ijka->ijka", tau15)
            + einsum("ijka->ijka", tau16)
        )

        tau18 = (
            - einsum("jkia->ijka", tau9)
            + 2 * einsum("kjia->ijka", tau9)
            - 4 * einsum("jika->ijka", tau10)
            + 2 * einsum("kija->ijka", tau10)
            + einsum("ijka->ijka", tau11)
            - 2 * einsum("ikja->ijka", tau11)
            + 2 * einsum("jika->ijka", tau12)
            - 4 * einsum("kija->ijka", tau12)
            - 2 * einsum("ijka->ijka", tau13)
            + einsum("ikja->ijka", tau13)
            + 2 * einsum("jkia->ijka", tau17)
            - einsum("kjia->ijka", tau17)
        )

        tau19 = (
            einsum("ikjb,jkba->ia", tau18, tau6)
        )

        tau20 = (
            - einsum("abji->ijab", h.v.vvoo)
            + 2 * einsum("baji->ijab", h.v.vvoo)
        )

        tau21 = (
            einsum("bi,abjk->ijka", ea.t1, ea.z2)
        )

        tau22 = (
            einsum("bi,bajk->ijka", ea.t1, ea.z2)
        )

        tau23 = (
            - 2 * einsum("ijka->ijka", tau21)
            + einsum("ikja->ijka", tau21)
            + einsum("ijka->ijka", tau22)
            - 2 * einsum("ikja->ijka", tau22)
        )

        tau24 = (
            einsum("bajk,ijkb->ia", ea.t2, tau23)
        )

        tau25 = (
            einsum("bj,jiba->ia", ea.t1, h.v.oovv)
        )

        tau26 = (
            einsum("ia,aj->ij", h.f.ov, ea.t1)
        )

        tau27 = (
            2 * einsum("ij->ij", tau26)
            - einsum("ji->ij", tau0)
            + 2 * einsum("ij->ij", h.f.oo)
        )

        tau28 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("baij->ijab", ea.t2)
        )

        tau29 = (
            einsum("ai,aj->ij", ea.t1, ea.z1)
        )

        tau30 = (
            einsum("ikab,abkj->ij", tau28, ea.z2)
            + 4 * einsum("ij->ij", tau29)
        )

        tau31 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
            - 2 * einsum("abji->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau32 = (
            einsum("acki,kbcj->ijab", ea.t2, h.v.ovvo)
            + einsum("ackj,kbic->ijab", ea.t2, h.v.ovov)
        )

        tau33 = (
            einsum("abki,bajk->ij", ea.t2, ea.z2)
        )

        tau34 = (
            einsum("jiab->ijab", h.v.oovv)
            - 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau35 = (
            einsum("bj,jiba->ia", ea.t1, tau34)
        )

        tau36 = (
            einsum("ai,ja->ij", ea.t1, tau35)
        )

        tau37 = (
            einsum("acik,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau38 = (
            einsum("ijab->ijab", tau37)
            - 2 * einsum("ijba->ijab", tau37)
            + einsum("jiba->ijab", tau37)
        )

        tau39 = (
            einsum("caki,cbkj->ijab", ea.t2, ea.z2)
        )

        tau40 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau41 = (
            2 * einsum("jabi->ijab", h.v.ovvo)
            + einsum("acik,kjcb->ijab", ea.t2, tau40)
            - einsum("jaib->ijab", h.v.ovov)
        )

        tau42 = (
            einsum("acik,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau43 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
            + einsum("baji->ijab", ea.z2)
        )

        tau44 = (
            einsum("caij,jicb->ab", ea.t2, h.v.oovv)
        )

        tau45 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("abji->ijab", ea.t2)
        )

        tau46 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
        )

        tau47 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
            + einsum("caij,jicb->ab", ea.t2, tau46)
        )

        tau48 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("baij->ijab", ea.t2)
        )

        tau49 = (
            einsum("acik,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau50 = (
            einsum("kica,bcjk->ijab", tau49, ea.z2)
        )

        tau51 = (
            einsum("abij,abkl->ijkl", ea.t2, ea.z2)
        )

        tau52 = (
            einsum("klij,lkba->ijab", tau51, h.v.oovv)
        )

        tau53 = (
            einsum("bj,ijba->ia", ea.t1, tau52)
        )

        tau54 = (
            einsum("cdij,badc->ijab", ea.t2, h.v.vvvv)
        )

        tau55 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
            - einsum("baji->ijab", ea.z2)
        )

        tau56 = (
            2 * einsum("ijka->ijka", tau21)
            - einsum("ikja->ijka", tau21)
            - einsum("ijka->ijka", tau22)
            + 2 * einsum("ikja->ijka", tau22)
        )

        tau57 = (
            einsum("abjk,ikjb->ia", ea.t2, tau56)
        )

        tau58 = (
            einsum("bj,jiab->ia", ea.t1, h.v.oovv)
        )

        tau59 = (
            einsum("caik,cbkj->ijab", ea.t2, ea.z2)
        )

        tau60 = (
            einsum("jibc,jcba->ia", tau59, h.v.ovvv)
        )

        tau61 = (
            einsum("caij,bcij->ab", ea.t2, ea.z2)
        )

        tau62 = (
            einsum("bc,icba->ia", tau61, h.v.ovvv)
        )

        tau63 = (
            einsum("bcij,kabc->ijka", ea.t2, h.v.ovvv)
        )

        tau64 = (
            einsum("jkib,abkj->ia", tau63, ea.z2)
        )

        tau65 = (
            einsum("abik,kjba->ij", ea.t2, h.v.oovv)
        )

        tau66 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
        )

        tau67 = (
            einsum("abik,jkba->ij", ea.t2, tau66)
            + einsum("ikab,abkj->ij", tau28, ea.z2)
            + 4 * einsum("ij->ij", tau29)
        )

        tau68 = (
            einsum("caik,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau69 = (
            einsum("ijab->ijab", tau49)
            + einsum("ijab->ijab", tau68)
        )

        tau70 = (
            einsum("caik,kjcb->ijab", ea.t2, tau1)
            + 2 * einsum("acki,cbkj->ijab", ea.t2, ea.z2)
        )

        tau71 = (
            einsum("kica,bckj->ijab", tau49, ea.z2)
        )

        tau72 = (
            2 * einsum("kica,cbjk->ijab", tau69, ea.z2)
            - 4 * einsum("kica,cbkj->ijab", tau49, ea.z2)
            - 2 * einsum("kjcb,kica->ijab", tau70, h.v.oovv)
            + 2 * einsum("ijab->ijab", tau71)
            - einsum("ijba->ijab", tau71)
        )

        tau73 = (
            einsum("acik,bcjk->ijab", ea.t2, ea.z2)
        )

        tau74 = (
            einsum("jibc,jcab->ia", tau73, h.v.ovvv)
        )

        tau75 = (
            einsum("ai,ja->ij", ea.t1, tau58)
        )

        tau76 = (
            einsum("abik,jkba->ij", ea.t2, tau6)
        )

        tau77 = (
            einsum("abij,bakl->ijkl", ea.t2, ea.z2)
        )

        tau78 = (
            einsum("acki,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau79 = (
            2 * einsum("kiac,kjcb->ijab", tau45, h.v.oovv)
            + einsum("ijab->ijab", tau78)
        )

        tau80 = (
            einsum("caik,cbjk->ijab", ea.t2, ea.z2)
        )

        tau81 = (
            einsum("caki,bckj->ijab", ea.t2, ea.z2)
        )

        tau82 = (
            - einsum("ijab->ijab", tau80)
            + 2 * einsum("ijab->ijab", tau81)
        )

        tau83 = (
            einsum("caik,bckj->ijab", ea.t2, ea.z2)
        )

        tau84 = (
            einsum("caki,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau85 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("abji->ijab", ea.z2)
        )

        tau86 = (
            einsum("caik,jkbc->ijab", ea.t2, tau85)
        )

        tau87 = (
            einsum("kjcb,kica->ijab", tau86, h.v.oovv)
        )

        tau88 = (
            einsum("kjcb,kiac->ijab", tau80, h.v.oovv)
        )

        tau89 = (
            - einsum("kica,cbkj->ijab", tau79, ea.z2)
            + einsum("kjcb,kica->ijab", tau82, h.v.oovv)
            + 2 * einsum("kjcb,kiac->ijab", tau83, h.v.oovv)
            + einsum("kica,kjcb->ijab", tau84, tau85)
            + einsum("ijab->ijab", tau87)
            - 2 * einsum("ijba->ijab", tau87)
            + 2 * einsum("ijab->ijab", tau88)
            - einsum("jiab->ijab", tau88)
        )

        tau90 = (
            einsum("abik,abjk->ij", ea.t2, ea.z2)
        )

        tau91 = (
            einsum("abki,kjab->ij", ea.t2, h.v.oovv)
        )

        tau92 = (
            einsum("bj,jiba->ia", ea.t1, tau40)
        )

        tau93 = (
            einsum("ij->ij", tau91)
            + einsum("ai,ja->ij", ea.t1, tau92)
        )

        tau94 = (
            einsum("abil,ljkb->ijka", ea.t2, h.v.ooov)
        )

        tau95 = (
            einsum("jikb,bajk->ia", tau94, ea.z2)
        )

        tau96 = (
            einsum("ij->ij", tau26)
            + einsum("ij->ij", h.f.oo)
        )

        tau97 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
            + 2 * einsum("baji->ijab", ea.z2)
        )

        tau98 = (
            einsum("abki,kjab->ij", ea.t2, tau97)
        )

        tau99 = (
            einsum("bail,jlkb->ijka", ea.t2, h.v.ooov)
        )

        tau100 = (
            einsum("jikb,bakj->ia", tau99, ea.z2)
        )

        tau101 = (
            einsum("caki,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau102 = (
            einsum("acki,jkbc->ijab", ea.t2, tau101)
        )

        tau103 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("baij->ijab", ea.z2)
        )

        tau104 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", h.v.ovvv)
        )

        tau105 = (
            einsum("ci,iacb->ab", ea.t1, tau104)
        )

        tau106 = (
            einsum("jicb,jabc->ia", tau28, h.v.ovvv)
            + 2 * einsum("bi,ab->ia", ea.t1, tau105)
        )

        tau107 = (
            einsum("jkil,kjla->ia", tau51, h.v.ooov)
        )

        tau108 = (
            einsum("jkli,kjla->ia", tau51, h.v.ooov)
        )

        tau109 = (
            einsum("acik,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau110 = (
            einsum("bckj,ikac->ijab", ea.t2, tau109)
        )

        tau111 = (
            einsum("abij->ijab", ea.z2)
            - 2 * einsum("abji->ijab", ea.z2)
        )

        tau112 = (
            einsum("acij,jibc->ab", ea.t2, h.v.oovv)
        )

        tau113 = (
            2 * einsum("abij->ijab", ea.t2)
            - einsum("abji->ijab", ea.t2)
        )

        tau114 = (
            einsum("ijac,bcij->ab", tau113, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau85)
        )

        tau115 = (
            einsum("abij,lkba->ijkl", ea.t2, h.v.oovv)
        )

        tau116 = (
            einsum("klij,abkl->ijab", tau115, ea.z2)
        )

        tau117 = (
            einsum("bj,ijab->ia", ea.t1, tau116)
        )

        tau118 = (
            einsum("abki,kjba->ij", ea.t2, h.v.oovv)
        )

        tau119 = (
            einsum("abki,jkba->ij", ea.t2, tau97)
        )

        tau120 = (
            einsum("bcij,kacb->ijka", ea.t2, h.v.ovvv)
        )

        tau121 = (
            einsum("jkib,bajk->ia", tau120, ea.z2)
        )

        tau122 = (
            einsum("bail,ljkb->ijka", ea.t2, h.v.ooov)
        )

        tau123 = (
            einsum("jikb,abjk->ia", tau122, ea.z2)
        )

        tau124 = (
            einsum("bcja,bcij->ia", h.v.vvov, ea.z2)
        )

        tau125 = (
            einsum("caki,jkbc->ijab", ea.t2, tau101)
        )

        tau126 = (
            einsum("acik,cbkj->ijab", ea.t2, ea.z2)
        )

        tau127 = (
            einsum("jkba,jikb->ia", tau126, h.v.ooov)
        )

        tau128 = (
            einsum("bj,baji->ia", ea.z1, ea.t2)
        )

        tau129 = (
            2 * einsum("abij->ijab", ea.z2)
            - einsum("baij->ijab", ea.z2)
            - einsum("abji->ijab", ea.z2)
            + 2 * einsum("baji->ijab", ea.z2)
        )

        tau130 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau129)
            + einsum("caij,jicb->ab", ea.t2, tau46)
        )

        tau131 = (
            einsum("caij,jibc->ab", ea.t2, h.v.oovv)
        )

        tau132 = (
            - 2 * einsum("abij->ijab", ea.t2)
            + einsum("abji->ijab", ea.t2)
        )

        tau133 = (
            einsum("ijac,bcij->ab", tau132, ea.z2)
        )

        tau134 = (
            einsum("ai,jkla->ijkl", ea.t1, tau21)
        )

        tau135 = (
            - einsum("ijkl->ijkl", tau134)
            + 2 * einsum("ijlk->ijkl", tau134)
            + 2 * einsum("jikl->ijkl", tau134)
            - einsum("jilk->ijkl", tau134)
        )

        tau136 = (
            einsum("bi,jkba->ijka", ea.t1, tau103)
        )

        tau137 = (
            einsum("ak,ikjb->ijab", ea.t1, tau136)
        )

        tau138 = (
            einsum("al,ijkl->ijka", ea.t1, tau134)
        )

        tau139 = (
            2 * einsum("ijka->ijka", tau138)
            - einsum("jika->ijka", tau138)
        )

        tau140 = (
            einsum("jkil,jkla->ia", tau135, h.v.ooov)
            + einsum("jicb,jbac->ia", tau137, h.v.ovvv)
            + einsum("jkib,kjba->ia", tau139, h.v.oovv)
        )

        tau141 = (
            einsum("caik,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau142 = (
            - einsum("abij->ijab", ea.z2)
            + 2 * einsum("baij->ijab", ea.z2)
            + 2 * einsum("abji->ijab", ea.z2)
        )

        tau143 = (
            einsum("caij,cbij->ab", ea.t2, ea.z2)
        )

        tau144 = (
            einsum("bc,icba->ia", tau143, h.v.ovvv)
        )

        tau145 = (
            einsum("acij,bcij->ab", ea.t2, ea.z2)
        )

        tau146 = (
            einsum("bc,icba->ia", tau145, h.v.ovvv)
        )

        tau147 = (
            einsum("caik,bcjk->ijab", ea.t2, ea.z2)
        )

        tau148 = (
            einsum("jibc,jcba->ia", tau147, h.v.ovvv)
        )

        tau149 = (
            einsum("acik,kjcb->ijab", ea.t2, tau34)
        )

        tau150 = (
            - einsum("ijka->ijka", h.v.ooov)
            + 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau151 = (
            einsum("ak,kija->ij", ea.t1, tau150)
            + einsum("ji->ij", tau91)
        )

        tau152 = (
            einsum("ikab,abkj->ij", tau48, ea.z2)
        )

        tau153 = (
            einsum("jibc,jcab->ia", tau80, h.v.ovvv)
        )

        tau154 = (
            einsum("ikab,abkj->ij", tau28, ea.z2)
        )

        tau155 = (
            einsum("bcja,cbij->ia", h.v.vvov, ea.z2)
        )

        tau156 = (
            einsum("jibc,jcab->ia", tau126, h.v.ovvv)
        )

        tau157 = (
            einsum("caik,kjbc->ijab", ea.t2, h.v.oovv)
        )

        tau158 = (
            einsum("caik,jkbc->ijab", ea.t2, tau157)
        )

        tau159 = (
            einsum("acik,cbjk->ijab", ea.t2, ea.z2)
        )

        tau160 = (
            einsum("jibc,jcab->ia", tau159, h.v.ovvv)
        )

        tau161 = (
            einsum("abik,bakj->ij", ea.t2, ea.z2)
        )

        tau162 = (
            einsum("abki,bakj->ij", ea.t2, ea.z2)
        )

        tau163 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau164 = (
            einsum("ak,kija->ij", ea.t1, h.v.ooov)
        )

        tau165 = (
            einsum("abkj,kiba->ij", ea.t2, tau163)
            + 2 * einsum("ij->ij", tau164)
        )

        tau166 = (
            einsum("jkil,jkla->ia", tau77, h.v.ooov)
        )

        tau167 = (
            einsum("adij,jbdc->iabc", ea.t2, h.v.ovvv)
        )

        tau168 = (
            einsum("jbca,cbji->ia", tau167, ea.z2)
        )

        tau169 = (
            einsum("acik,bckj->ijab", ea.t2, ea.z2)
        )

        tau170 = (
            einsum("jibc,jcab->ia", tau169, h.v.ovvv)
        )

        tau171 = (
            2 * einsum("iabj->ijab", h.v.ovvo)
            - einsum("iajb->ijab", h.v.ovov)
        )

        tau172 = (
            einsum("daij,jbcd->iabc", ea.t2, h.v.ovvv)
        )

        tau173 = (
            einsum("jbca,cbji->ia", tau172, ea.z2)
        )

        tau174 = (
            einsum("jkba,jikb->ia", tau169, h.v.ooov)
        )

        tau175 = (
            einsum("caki,jkbc->ijab", ea.t2, tau1)
        )

        tau176 = (
            - einsum("abij->ijab", ea.t2)
            + 2 * einsum("baij->ijab", ea.t2)
            + 2 * einsum("abji->ijab", ea.t2)
            - einsum("baji->ijab", ea.t2)
        )

        tau177 = (
            2 * einsum("bj,jiab->ia", ea.t1, tau171)
            + einsum("jb,jiab->ia", h.f.ov, tau176)
        )

        tau178 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("abji->ijab", ea.z2)
        )

        tau179 = (
            einsum("acki,jkbc->ijab", ea.t2, tau178)
        )

        tau180 = (
            einsum("bj,abij->ia", ea.z1, ea.t2)
        )

        tau181 = (
            einsum("ci,jabc->ijab", ea.t1, h.v.ovvv)
        )

        tau182 = (
            einsum("bi,jkab->ijka", ea.t1, tau181)
        )

        tau183 = (
            einsum("jkba,jkib->ia", tau103, tau182)
        )

        tau184 = (
            einsum("caik,jkbc->ijab", ea.t2, tau68)
        )

        tau185 = (
            einsum("adij,jbcd->iabc", ea.t2, h.v.ovvv)
        )

        tau186 = (
            einsum("jbca,cbij->ia", tau185, ea.z2)
        )

        tau187 = (
            einsum("jkil,kjla->ia", tau77, h.v.ooov)
        )

        tau188 = (
            einsum("acij,cbij->ab", ea.t2, ea.z2)
        )

        tau189 = (
            einsum("bc,icab->ia", tau188, h.v.ovvv)
        )

        tau190 = (
            einsum("caki,jkbc->ijab", ea.t2, tau84)
        )

        tau191 = (
            einsum("caki,kbjc->ijab", ea.t2, h.v.ovov)
        )

        tau192 = (
            einsum("acij,jicb->ab", ea.t2, h.v.oovv)
        )

        tau193 = (
            einsum("ijac,bcij->ab", tau132, ea.z2)
            + einsum("acij,jicb->ab", ea.t2, tau178)
        )

        tau194 = (
            einsum("klij,ablk->ijab", tau115, ea.z2)
        )

        tau195 = (
            einsum("bj,ijab->ia", ea.t1, tau194)
        )

        tau196 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("abji->ijab", ea.t2)
        )

        tau197 = (
            einsum("ijca,bcij->ab", tau196, ea.z2)
            + einsum("caij,jicb->ab", ea.t2, tau111)
        )

        tau198 = (
            einsum("acik,jkbc->ijab", ea.t2, tau85)
        )

        tau199 = (
            einsum("jkba,ijkb->ia", tau169, h.v.ooov)
        )

        tau200 = (
            einsum("iajb->ijab", h.v.ovov)
            - 2 * einsum("iabj->ijab", h.v.ovvo)
        )

        tau201 = (
            einsum("jikb,abjk->ia", tau94, ea.z2)
        )

        tau202 = (
            einsum("acij,cbji->ab", ea.t2, ea.z2)
        )

        tau203 = (
            einsum("bc,icab->ia", tau202, h.v.ovvv)
        )

        tau204 = (
            - 2 * einsum("abij->ijab", ea.z2)
            + einsum("baij->ijab", ea.z2)
        )

        tau205 = (
            einsum("abki,jkba->ij", ea.t2, tau204)
        )

        tau206 = (
            einsum("ijca,bcij->ab", tau45, ea.z2)
        )

        tau207 = (
            einsum("jbca,cbij->ia", tau172, ea.z2)
        )

        tau208 = (
            einsum("jibc,jcba->ia", tau169, h.v.ovvv)
        )

        tau209 = (
            2 * einsum("abij->ijab", ea.t2)
            - einsum("baij->ijab", ea.t2)
        )

        tau210 = (
            einsum("bajk,ikjb->ia", ea.t2, tau14)
            + einsum("ijcb,jabc->ia", tau209, h.v.ovvv)
        )

        tau211 = (
            einsum("jibc,jcba->ia", tau126, h.v.ovvv)
        )

        tau212 = (
            einsum("caij,cbji->ab", ea.t2, ea.z2)
        )

        tau213 = (
            einsum("bc,icab->ia", tau212, h.v.ovvv)
        )

        tau214 = (
            einsum("jbca,bcji->ia", tau185, ea.z2)
        )

        tau215 = (
            einsum("jkba,jikb->ia", tau80, h.v.ooov)
        )

        tau216 = (
            einsum("jbca,bcji->ia", tau172, ea.z2)
        )

        tau217 = (
            einsum("ij->ij", tau164)
            + einsum("aj,ia->ij", ea.t1, tau35)
        )

        tau218 = (
            einsum("jkba,ijkb->ia", tau147, h.v.ooov)
        )

        tau219 = (
            einsum("abik,bajk->ij", ea.t2, ea.z2)
        )

        tau220 = (
            einsum("ai,ja->ij", ea.t1, tau25)
        )

        tau221 = (
            einsum("ij->ij", tau91)
            + 2 * einsum("ij->ij", tau220)
        )

        tau222 = (
            einsum("acki,kjcb->ijab", ea.t2, h.v.oovv)
        )

        tau223 = (
            einsum("bcki,jkac->ijab", ea.t2, tau78)
            + einsum("acki,jkbc->ijab", ea.t2, tau222)
        )

        tau224 = (
            einsum("ijca,cbij->ab", tau113, ea.z2)
        )

        tau225 = (
            einsum("acij,bcji->ab", ea.t2, ea.z2)
        )

        tau226 = (
            einsum("bc,icab->ia", tau225, h.v.ovvv)
        )

        tau227 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau228 = (
            einsum("bajk,jkib->ia", ea.t2, tau227)
            + einsum("bj,ijab->ia", ea.t1, tau78)
        )

        tau229 = (
            einsum("jkil,jkla->ia", tau51, h.v.ooov)
        )

        tau230 = (
            einsum("ak,kija->ij", ea.t1, tau150)
        )

        tau231 = (
            - 2 * einsum("ijka->ijka", h.v.ooov)
            + einsum("jika->ijka", h.v.ooov)
        )

        tau232 = (
            einsum("abjk,jkib->ia", ea.t2, tau231)
        )

        tau233 = (
            einsum("kjcb,kiac->ijab", tau147, h.v.oovv)
        )

        tau234 = (
            einsum("ikba,abjk->ij", tau28, ea.z2)
        )

        tau235 = (
            einsum("caki,jkbc->ijab", ea.t2, tau204)
        )

        tau236 = (
            einsum("bc,icab->ia", tau61, h.v.ovvv)
        )

        tau237 = (
            einsum("abki,kjba->ij", ea.t2, tau103)
        )

        tau238 = (
            einsum("ab,bi->ia", h.f.vv, ea.t1)
            + einsum("ia->ia", h.f.ov.conj())
        )

        tau239 = (
            einsum("ibjk,bajk->ia", h.v.ovoo, ea.z2)
        )

        tau240 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("baij->ijab", ea.t2)
            + einsum("baji->ijab", ea.t2)
        )

        tau241 = (
            einsum("ikba,abjk->ij", tau240, ea.z2)
        )

        tau242 = (
            einsum("jibc,jcab->ia", tau59, h.v.ovvv)
        )

        tau243 = (
            einsum("caik,kjcb->ijab", ea.t2, tau1)
        )

        tau244 = (
            einsum("bc,icba->ia", tau225, h.v.ovvv)
        )

        tau245 = (
            einsum("jilk,abkl->ijab", h.v.oooo, ea.z2)
        )

        tau246 = (
            einsum("bj,jiab->ia", ea.t1, tau245)
        )

        tau247 = (
            einsum("abki,abjk->ij", ea.t2, ea.z2)
        )

        tau248 = (
            einsum("ai,ja->ij", ea.t1, tau92)
        )

        tau249 = (
            einsum("jikb,abkj->ia", tau122, ea.z2)
        )

        tau250 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau251 = (
            4 * einsum("ij->ij", tau3)
            + einsum("abkj,kiba->ij", ea.t2, tau250)
        )

        tau252 = (
            einsum("ijac,cbij->ab", tau196, ea.z2)
        )

        tau253 = (
            einsum("jkba,ijkb->ia", tau80, h.v.ooov)
        )

        tau254 = (
            einsum("acki,bcjk->ijab", ea.t2, ea.z2)
        )

        tau255 = (
            einsum("ikba,abjk->ij", tau8, ea.z2)
        )

        tau256 = (
            einsum("jbca,cbji->ia", tau185, ea.z2)
        )

        tau257 = (
            einsum("bc,icab->ia", tau145, h.v.ovvv)
        )

        tau258 = (
            einsum("jkba,jikb->ia", tau83, h.v.ooov)
        )

        tau259 = (
            einsum("ijca,bcij->ab", tau196, ea.z2)
        )

        tau260 = (
            einsum("jikb,bajk->ia", tau122, ea.z2)
        )

        tau261 = (
            einsum("caki,kjcb->ijab", ea.t2, tau129)
        )

        tau262 = (
            einsum("ibjk,abjk->ia", h.v.ovoo, ea.z2)
        )

        tau263 = (
            einsum("bajk,ijkb->ia", ea.t2, tau56)
        )

        tau264 = (
            einsum("jbca,bcij->ia", tau172, ea.z2)
        )

        tau265 = (
            einsum("jibc,jcba->ia", tau80, h.v.ovvv)
        )

        tau266 = (
            einsum("dcba,cdij->ijab", h.v.vvvv, ea.z2)
        )

        tau267 = (
            einsum("bj,ijba->ia", ea.t1, tau266)
        )

        tau268 = (
            einsum("jkib,abjk->ia", tau63, ea.z2)
        )

        tau269 = (
            einsum("ibkj,bajk->ia", h.v.ovoo, ea.z2)
        )

        tau270 = (
            einsum("abjk,ijkb->ia", ea.t2, tau14)
        )

        tau271 = (
            einsum("jkba,jikb->ia", tau59, h.v.ooov)
        )

        tau272 = (
            einsum("jbca,bcij->ia", tau185, ea.z2)
        )

        tau273 = (
            einsum("jkba,ijkb->ia", tau73, h.v.ooov)
        )

        tau274 = (
            einsum("caij,bcji->ab", ea.t2, ea.z2)
        )

        tau275 = (
            einsum("bc,icba->ia", tau274, h.v.ovvv)
        )

        tau276 = (
            einsum("jkba,ijkb->ia", tau159, h.v.ooov)
        )

        tau277 = (
            einsum("jkib,bakj->ia", tau120, ea.z2)
        )

        tau278 = (
            einsum("jkib,abjk->ia", tau120, ea.z2)
        )

        tau279 = (
            einsum("abil,jlkb->ijka", ea.t2, h.v.ooov)
        )

        tau280 = (
            einsum("jikb,abjk->ia", tau279, ea.z2)
        )

        tau281 = (
            einsum("caki,kjcb->ijab", ea.t2, tau85)
        )

        tau282 = (
            einsum("jikb,bakj->ia", tau122, ea.z2)
        )

        tau283 = (
            einsum("caki,kjcb->ijab", ea.t2, tau1)
        )

        tau284 = (
            einsum("ji->ij", tau0)
            - 2 * einsum("ij->ij", h.f.oo)
        )

        tau285 = (
            einsum("abjk,ikjb->ia", ea.t2, tau56)
            + 2 * einsum("bj,baij->ia", ea.z1, ea.t2)
        )

        tau286 = (
            einsum("jbca,cbij->ia", tau167, ea.z2)
        )

        tau287 = (
            einsum("ijca,cbij->ab", tau132, ea.z2)
        )

        tau288 = (
            einsum("caik,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau289 = (
            - 2 * einsum("ijab->ijab", tau288)
            + einsum("ijba->ijab", tau288)
            + einsum("jiab->ijab", tau288)
        )

        tau290 = (
            einsum("bc,icba->ia", tau212, h.v.ovvv)
        )

        tau291 = (
            einsum("jikb,abjk->ia", tau99, ea.z2)
        )

        tau292 = (
            einsum("ijac,cbij->ab", tau45, ea.z2)
        )

        tau293 = (
            einsum("bj,abji->ia", ea.z1, ea.t2)
        )

        tau294 = (
            einsum("jkib,abkj->ia", tau120, ea.z2)
        )

        tau295 = (
            einsum("jikb,bajk->ia", tau99, ea.z2)
        )

        tau296 = (
            einsum("jkli,jkla->ia", tau51, h.v.ooov)
        )

        tau297 = (
            einsum("abki,abkj->ij", ea.t2, ea.z2)
        )

        tau298 = (
            - einsum("ji->ij", tau91)
            + einsum("ij->ij", tau164)
        )

        tau299 = (
            einsum("abki,kjab->ij", ea.t2, tau1)
        )

        tau300 = (
            2 * einsum("ij->ij", tau75)
            + einsum("ij->ij", tau0)
        )

        tau301 = (
            einsum("daij,jbdc->iabc", ea.t2, h.v.ovvv)
        )

        tau302 = (
            einsum("jbca,cbji->ia", tau301, ea.z2)
        )

        tau303 = (
            einsum("ibkj,abjk->ia", h.v.ovoo, ea.z2)
        )

        tau304 = (
            einsum("bj,jiab->ia", ea.t1, tau194)
        )

        tau305 = (
            einsum("bj,ijab->ia", ea.t1, tau266)
        )

        tau306 = (
            einsum("jikb,bajk->ia", tau279, ea.z2)
        )

        tau307 = (
            einsum("bcja,cbji->ia", h.v.vvov, ea.z2)
        )

        tau308 = (
            einsum("jbca,cbij->ia", tau301, ea.z2)
        )

        tau309 = (
            einsum("jikb,abkj->ia", tau279, ea.z2)
        )

        tau310 = (
            einsum("caki,kbcj->ijab", ea.t2, h.v.ovvo)
        )

        tau311 = (
            einsum("jkib,bajk->ia", tau63, ea.z2)
        )

        tau312 = (
            einsum("jkli,kjla->ia", tau77, h.v.ooov)
        )

        tau313 = (
            einsum("jibc,jcba->ia", tau159, h.v.ovvv)
        )

        tau314 = (
            einsum("jikb,bakj->ia", tau94, ea.z2)
        )

        tau315 = (
            einsum("jkba,jikb->ia", tau147, h.v.ooov)
        )

        tau316 = (
            einsum("jbca,bcji->ia", tau301, ea.z2)
        )

        tau317 = (
            einsum("bc,icab->ia", tau274, h.v.ovvv)
        )

        tau318 = (
            einsum("jikb,abkj->ia", tau99, ea.z2)
        )

        tau319 = (
            einsum("ijac,bcij->ab", tau113, ea.z2)
        )

        tau320 = (
            einsum("jikb,bakj->ia", tau279, ea.z2)
        )

        tau321 = (
            einsum("klij,lkba->ijab", tau77, h.v.oovv)
        )

        tau322 = (
            einsum("bj,ijba->ia", ea.t1, tau321)
        )

        tau323 = (
            einsum("jkib,bakj->ia", tau63, ea.z2)
        )

        tau324 = (
            einsum("bj,jiab->ia", ea.t1, tau116)
        )

        tau325 = (
            einsum("ak,kija->ij", ea.t1, tau227)
        )

        tau326 = (
            einsum("bcja,bcji->ia", h.v.vvov, ea.z2)
        )

        tau327 = (
            einsum("jibc,jcab->ia", tau147, h.v.ovvv)
        )

        tau328 = (
            einsum("jkba,ijkb->ia", tau126, h.v.ooov)
        )

        tau329 = (
            einsum("jkba,jikb->ia", tau159, h.v.ooov)
        )

        tau330 = (
            einsum("bj,ijab->ia", ea.t1, tau245)
        )

        tau331 = (
            einsum("jkba,ijkb->ia", tau59, h.v.ooov)
        )

        tau332 = (
            einsum("jbca,bcji->ia", tau167, ea.z2)
        )

        tau333 = (
            einsum("jkba,jikb->ia", tau73, h.v.ooov)
        )

        tau334 = (
            einsum("abik,abkj->ij", ea.t2, ea.z2)
        )

        tau335 = (
            einsum("jibc,jcab->ia", tau83, h.v.ovvv)
        )

        tau336 = (
            einsum("bc,icba->ia", tau188, h.v.ovvv)
        )

        tau337 = (
            einsum("jbca,bcij->ia", tau301, ea.z2)
        )

        tau338 = (
            einsum("jkba,ijkb->ia", tau83, h.v.ooov)
        )

        tau339 = (
            einsum("bc,icba->ia", tau202, h.v.ovvv)
        )

        tau340 = (
            einsum("jbca,bcij->ia", tau167, ea.z2)
        )

        tau341 = (
            einsum("bj,ijab->ia", ea.t1, tau321)
        )

        tau342 = (
            einsum("bc,icab->ia", tau143, h.v.ovvv)
        )

        tau343 = (
            einsum("jikb,abkj->ia", tau94, ea.z2)
        )

        tau344 = (
            einsum("jkli,jkla->ia", tau77, h.v.ooov)
        )

        tau345 = (
            einsum("bj,ijab->ia", ea.t1, tau52)
        )

        tau346 = (
            einsum("jibc,jcba->ia", tau73, h.v.ovvv)
        )

        tau347 = (
            einsum("jibc,jcba->ia", tau83, h.v.ovvv)
        )

        tau348 = (
            2 * einsum("ia->ia", h.f.ov)
            + einsum("bj,jiba->ia", ea.t1, tau40)
        )

        tau349 = (
            einsum("aj,bcij->iabc", ea.t1, ea.z2)
        )

        tau350 = (
            einsum("aj,bcji->iabc", ea.t1, ea.z2)
        )

        tau351 = (
            - einsum("iabc->iabc", tau349)
            + 2 * einsum("iacb->iabc", tau349)
            + 2 * einsum("iabc->iabc", tau350)
            - einsum("iacb->iabc", tau350)
        )

        tau352 = (
            einsum("al,likj->ijka", ea.t1, h.v.oooo)
        )

        tau353 = (
            - 2 * einsum("ijka->ijka", tau352)
            + 2 * einsum("iajk->ijka", h.v.ovoo)
            + einsum("ijka->ijka", tau13)
        )

        tau354 = (
            einsum("ljkb,ilab->ijka", tau227, tau8)
        )

        tau355 = (
            einsum("ai,jkla->ijkl", ea.t1, h.v.ooov)
        )

        tau356 = (
            einsum("al,ijlk->ijka", ea.t1, tau355)
        )

        tau357 = (
            einsum("ilba,jlkb->ijka", tau8, h.v.ooov)
        )

        tau358 = (
            einsum("ijka->ijka", tau354)
            + 2 * einsum("ijka->ijka", tau356)
            + einsum("ijka->ijka", tau357)
            - 2 * einsum("ijka->ijka", tau12)
        )

        tau359 = (
            - 2 * einsum("ijka->ijka", tau7)
            + einsum("ikja->ijka", tau7)
        )

        tau360 = (
            einsum("ilkb,jlab->ijka", tau359, tau8)
        )

        tau361 = (
            einsum("ai,jkla->ijkl", ea.t1, tau7)
        )

        tau362 = (
            einsum("ijkl->ijkl", tau115)
            + 2 * einsum("ijkl->ijkl", tau361)
        )

        tau363 = (
            einsum("al,ijkl->ijka", ea.t1, tau362)
        )

        tau364 = (
            einsum("ilkb,jlba->ijka", tau7, tau8)
        )

        tau365 = (
            einsum("kb,baij->ijka", tau35, ea.t2)
        )

        tau366 = (
            einsum("ijka->ijka", tau360)
            - einsum("ijka->ijka", tau63)
            + einsum("ijka->ijka", tau363)
            - 2 * einsum("ijka->ijka", tau182)
            + einsum("ijka->ijka", tau364)
            + einsum("ijka->ijka", tau365)
        )

        tau367 = (
            einsum("kb,abij->ijka", tau35, ea.t2)
        )

        tau368 = (
            einsum("al,ijlk->ijka", ea.t1, tau115)
        )

        tau369 = (
            einsum("iklb,jlba->ijka", tau7, tau8)
        )

        tau370 = (
            einsum("ijka->ijka", tau367)
            + einsum("ijka->ijka", tau368)
            + einsum("ijka->ijka", tau369)
            - einsum("ijka->ijka", tau120)
        )

        tau371 = (
            einsum("ilba,ljkb->ijka", tau8, h.v.ooov)
        )

        tau372 = (
            einsum("al,iljk->ijka", ea.t1, tau355)
        )

        tau373 = (
            einsum("ijka->ijka", tau371)
            + 2 * einsum("ijka->ijka", tau372)
            - 2 * einsum("ijka->ijka", tau10)
        )

        tau374 = (
            - 2 * einsum("ijka->ijka", tau353)
            + einsum("ikja->ijka", tau353)
            - einsum("jika->ijka", tau358)
            + 2 * einsum("kija->ijka", tau358)
            + 2 * einsum("jkia->ijka", tau366)
            - einsum("kjia->ijka", tau366)
            + einsum("ijka->ijka", tau11)
            - 2 * einsum("ikja->ijka", tau11)
            - einsum("jkia->ijka", tau370)
            + 2 * einsum("kjia->ijka", tau370)
            + 2 * einsum("jika->ijka", tau373)
            - einsum("kija->ijka", tau373)
        )

        tau375 = (
            2 * einsum("abij->ijab", ea.t2)
            - einsum("baij->ijab", ea.t2)
            - einsum("abji->ijab", ea.t2)
            + 2 * einsum("baji->ijab", ea.t2)
        )

        tau376 = (
            einsum("ijac,bcij->ab", tau375, ea.z2)
            + 4 * einsum("ai,bi->ab", ea.t1, ea.z1)
            + einsum("caij,jicb->ab", ea.t2, tau46)
            + einsum("ijac,cbij->ab", tau45, ea.z2)
        )

        tau377 = (
            - einsum("iabc->iabc", h.v.ovvv)
            + 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau378 = (
            einsum("abij->ijab", ea.t2)
            - 2 * einsum("baij->ijab", ea.t2)
            - 2 * einsum("abji->ijab", ea.t2)
            + einsum("baji->ijab", ea.t2)
        )

        tau379 = (
            - 2 * einsum("ijka->ijka", tau22)
            + einsum("ikja->ijka", tau22)
            + einsum("ijka->ijka", tau21)
            - 2 * einsum("ikja->ijka", tau21)
        )

        tau380 = (
            - 2 * einsum("abij->ijab", ea.t2)
            + einsum("baij->ijab", ea.t2)
            + einsum("abji->ijab", ea.t2)
            - 2 * einsum("baji->ijab", ea.t2)
        )

        tau381 = (
            einsum("kiac,jkbc->ijab", tau378, tau6)
            + 2 * einsum("ak,ijkb->ijab", ea.t1, tau379)
            + einsum("kiac,jkcb->ijab", tau380, tau6)
        )

        tau382 = (
            einsum("abki,jkba->ij", ea.t2, tau129)
        )

        tau383 = (
            einsum("abik,jkba->ij", ea.t2, tau66)
        )

        tau384 = (
            einsum("ikba,abkj->ij", tau209, ea.z2)
        )

        tau385 = (
            einsum("ij->ij", tau382)
            + einsum("ij->ij", tau383)
            + einsum("ij->ij", tau384)
            + 4 * einsum("ij->ij", tau29)
        )

        tau386 = (
            einsum("bj,jiba->ia", ea.t1, tau40)
        )

        tau387 = (
            einsum("ia->ia", h.f.ov)
            + einsum("ia->ia", tau386)
        )

        tau388 = (
            einsum("bi,jkab->ijka", ea.t1, tau6)
        )

        tau389 = (
            2 * einsum("ijkl->ijkl", tau134)
            + einsum("ijkl->ijkl", tau51)
        )

        tau390 = (
            einsum("ijlk->ijkl", tau77)
            + einsum("jikl->ijkl", tau77)
            + einsum("ijkl->ijkl", tau389)
            + einsum("jilk->ijkl", tau389)
        )

        tau391 = (
            einsum("acki,bckj->ijab", ea.t2, ea.z2)
        )

        tau392 = (
            einsum("bail,jklb->ijka", ea.t2, tau388)
            + einsum("al,jilk->ijka", ea.t1, tau390)
            + einsum("bi,jkab->ijka", ea.t1, tau81)
            + einsum("iklb,jlba->ijka", tau22, tau8)
            + einsum("abjl,iklb->ijka", ea.t2, tau388)
            + einsum("bj,ikab->ijka", ea.t1, tau391)
        )

        tau393 = (
            einsum("ijka->ijka", tau353)
            - 2 * einsum("ikja->ijka", tau353)
            - einsum("jika->ijka", tau373)
            + 2 * einsum("kija->ijka", tau373)
            + 2 * einsum("jika->ijka", tau358)
            - einsum("kija->ijka", tau358)
            - einsum("jkia->ijka", tau366)
            + 2 * einsum("kjia->ijka", tau366)
            - 2 * einsum("ijka->ijka", tau11)
            + einsum("ikja->ijka", tau11)
            + 2 * einsum("jkia->ijka", tau370)
            - einsum("kjia->ijka", tau370)
        )

        tau394 = (
            2 * einsum("ijka->ijka", tau22)
            - einsum("ikja->ijka", tau22)
            - einsum("ijka->ijka", tau21)
            + 2 * einsum("ikja->ijka", tau21)
        )

        tau395 = (
            einsum("aj,ij->ia", ea.t1, tau385)
            + 2 * einsum("bj,jiab->ia", ea.z1, tau378)
            + einsum("ikjb,jkba->ia", tau394, tau8)
            - 4 * einsum("ai->ia", ea.t1)
        )

        tau396 = (
            einsum("bk,ijab->ijka", ea.z1, tau8)
            + einsum("jlkb,ilab->ijka", tau388, tau8)
        )

        tau397 = (
            2 * einsum("kiac,jkbc->ijab", tau176, tau6)
            + 2 * einsum("ak,ijkb->ijab", ea.t1, tau23)
            + einsum("kiac,jkcb->ijab", tau378, tau6)
        )

        tau398 = (
            2 * einsum("aj,ia->ij", ea.t1, tau387)
        )

        tau399 = (
            einsum("kiba,jkba->ij", tau250, tau8)
        )

        tau400 = (
            2 * einsum("ak,kija->ij", ea.t1, tau150)
        )

        tau401 = (
            einsum("ij->ij", tau398)
            + einsum("ij->ij", tau399)
            + einsum("ij->ij", tau400)
            + 2 * einsum("ij->ij", h.f.oo)
        )

        tau402 = (
            2 * einsum("ijkl->ijkl", tau77)
            - einsum("ijlk->ijkl", tau77)
            - einsum("jikl->ijkl", tau77)
            + 2 * einsum("jilk->ijkl", tau77)
            - einsum("ijkl->ijkl", tau389)
            + 2 * einsum("ijlk->ijkl", tau389)
            + 2 * einsum("jikl->ijkl", tau389)
            - einsum("jilk->ijkl", tau389)
        )

        tau403 = (
            einsum("acki,cbjk->ijab", ea.t2, ea.z2)
        )

        tau404 = (
            einsum("caki,cbjk->ijab", ea.t2, ea.z2)
        )

        tau405 = (
            einsum("bi,jkab->ijka", ea.t1, tau403)
            + einsum("bj,ikab->ijka", ea.t1, tau404)
            + einsum("jlkb,ilba->ijka", tau21, tau8)
        )

        tau406 = (
            - einsum("aj,ij->ia", ea.t1, tau385)
            + 2 * einsum("bj,jiab->ia", ea.z1, tau176)
            + einsum("ikjb,jkba->ia", tau379, tau8)
            + 4 * einsum("ai->ia", ea.t1)
        )

        tau407 = (
            einsum("ci,iacb->ab", ea.t1, tau104)
            + einsum("ab->ab", h.f.vv)
        )

        tau408 = (
            2 * nv * einsum("iaab->iab", h.v.ovvv)
            - nv * einsum("iaba->iab", h.v.ovvv)
        )

        tau409 = (
            - no * nv * einsum("jiab->ijab", h.v.oovv)
            + 2 * no * nv * einsum("jiba->ijab", h.v.oovv)
        )

        tau410 = (
            einsum("ai,iab->ab", ea.t1, tau408)
            + einsum("ab->ab", h.f.vv)
        )

        tau411 = (
            no * nv * einsum("jiab->ijab", h.v.oovv)
            - 2 * no * nv * einsum("jiba->ijab", h.v.oovv)
        )

        tau412 = (
            no * einsum("ijia->ija", h.v.ooov)
            - 2 * no * einsum("ijai->ija", h.v.oovo)
        )

        tau413 = (
            - einsum("ij->ij", tau26)
            + einsum("aj,jia->ij", ea.t1, tau412)
            - einsum("ij->ij", h.f.oo)
        )

        tau414 = (
            - 2 * no * einsum("ijia->ija", h.v.ooov)
            + no * einsum("ijai->ija", h.v.oovo)
        )

        tau415 = (
            no * nv * einsum("jiab->ijab", h.v.oovv)
            + no * nv * einsum("jiba->ijab", h.v.oovv)
        )

        tau416 = (
            einsum("ai,bajj->ijab", ea.t1, ea.t2)
            + einsum("bj,aaij->ijab", ea.t1, ea.t2)
        )

        tau417 = (
            - nv * einsum("iaab->iab", h.v.ovvv)
            + 2 * nv * einsum("iaba->iab", h.v.ovvv)
        )

        tau418 = (
            einsum("ijac,ijbc->ab", tau132, h.v.oovv)
        )

        tau419 = (
            einsum("caij,jicb->ab", ea.t2, tau163)
        )

        tau420 = (
            2 * einsum("ci,iacb->ab", ea.t1, tau104)
        )

        tau421 = (
            einsum("ab->ab", tau418)
            + einsum("ab->ab", tau419)
            + einsum("ab->ab", tau420)
            + 2 * einsum("ab->ab", h.f.vv)
        )

        tau422 = (
            einsum("ca,bcij->ijab", tau421, ea.z2)
        )

        tau423 = (
            - 2 * einsum("ka,kijb->ijab", tau387, tau21)
        )

        tau424 = (
            - einsum("ik,abjk->ijab", tau401, ea.z2)
        )

        tau425 = (
            einsum("kiac,kjbc->ijab", tau8, h.v.oovv)
        )

        tau426 = (
            einsum("ci,jacb->ijab", ea.t1, h.v.ovvv)
        )

        tau427 = (
            - 2 * einsum("jaib->ijab", h.v.ovov)
            + einsum("ijab->ijab", tau425)
            - 2 * einsum("ijab->ijab", tau426)
        )

        tau428 = (
            einsum("kica,jkcb->ijab", tau427, tau6)
        )

        tau429 = (
            2 * einsum("ijkl->ijkl", tau355)
            + einsum("lijk->ijkl", tau115)
        )

        tau430 = (
            einsum("kijl,abkl->ijab", tau429, ea.z2)
        )

        tau431 = (
            einsum("ijka->ijka", h.v.ooov)
            + einsum("kjia->ijka", tau7)
        )

        tau432 = (
            2 * einsum("kljb,ikla->ijab", tau388, tau431)
        )

        tau433 = (
            einsum("idab,dcjk->ijkabc", h.v.ovvv, ea.z2)
        )

        tau434 = (
            einsum("ck,kijabc->ijab", ea.t1, tau433)
        )

        tau435 = (
            einsum("bk,ijka->ijab", ea.z1, h.v.ooov)
        )

        tau436 = (
            einsum("ijab->ijab", tau422)
            + einsum("ijab->ijab", tau423)
            + einsum("ijab->ijab", tau424)
            + einsum("ijab->ijab", tau428)
            + einsum("ijab->ijab", tau430)
            + einsum("ijab->ijab", tau432)
            - 2 * einsum("ijab->ijab", tau434)
            - 4 * einsum("ijab->ijab", tau435)
            + einsum("ijab->ijab", tau321)
        )

        tau437 = (
            einsum("kiac,kjcb->ijab", tau163, tau81)
        )

        tau438 = (
            einsum("kiac,jkbc->ijab", tau196, tau6)
        )

        tau439 = (
            einsum("acik,jkcb->ijab", ea.t2, tau6)
        )

        tau440 = (
            einsum("caki,kjcb->ijab", ea.t2, tau178)
        )

        tau441 = (
            einsum("ijab->ijab", tau438)
            + einsum("ijab->ijab", tau439)
            + einsum("ijab->ijab", tau440)
        )

        tau442 = (
            einsum("kjcb,kiac->ijab", tau441, h.v.oovv)
        )

        tau443 = (
            - einsum("ik,abkj->ijab", tau401, ea.z2)
        )

        tau444 = (
            einsum("jabi->ijab", h.v.ovvo)
            + einsum("ijab->ijab", tau101)
            + einsum("ijab->ijab", tau181)
        )

        tau445 = (
            - 2 * einsum("kica,bckj->ijab", tau444, ea.z2)
        )

        tau446 = (
            2 * einsum("kica,kjcb->ijab", tau8, h.v.oovv)
        )

        tau447 = (
            2 * einsum("jabi->ijab", h.v.ovvo)
            + einsum("ijab->ijab", tau446)
            + 2 * einsum("ijab->ijab", tau181)
            - einsum("jaib->ijab", h.v.ovov)
        )

        tau448 = (
            2 * einsum("kica,cbkj->ijab", tau447, ea.z2)
        )

        tau449 = (
            2 * einsum("kjlb,kila->ijab", tau23, tau431)
        )

        tau450 = (
            - einsum("kj,kiba->ijab", tau385, h.v.oovv)
        )

        tau451 = (
            einsum("ijab->ijab", tau78)
            - 2 * einsum("ijab->ijab", tau426)
        )

        tau452 = (
            einsum("kica,kjcb->ijab", tau451, tau6)
        )

        tau453 = (
            2 * einsum("ak,jila->ijkl", ea.t1, tau431)
        )

        tau454 = (
            einsum("ijkl->ijkl", tau453)
            + einsum("lkji->ijkl", tau115)
            + 2 * einsum("jilk->ijkl", h.v.oooo)
        )

        tau455 = (
            einsum("jilk,abkl->ijab", tau454, ea.z2)
        )

        tau456 = (
            einsum("ijca,bcij->ab", tau196, ea.z2)
        )

        tau457 = (
            einsum("acij,jicb->ab", ea.t2, tau4)
        )

        tau458 = (
            einsum("caij,jicb->ab", ea.t2, tau111)
        )

        tau459 = (
            einsum("ab->ab", tau456)
            + einsum("ab->ab", tau457)
            + einsum("ab->ab", tau458)
        )

        tau460 = (
            einsum("cb,jica->ijab", tau459, h.v.oovv)
        )

        tau461 = (
            einsum("ca,cbij->ijab", tau421, ea.z2)
        )

        tau462 = (
            einsum("kiac,kjcb->ijab", tau8, h.v.oovv)
        )

        tau463 = (
            einsum("jkbc,kica->ijab", tau4, tau462)
        )

        tau464 = (
            einsum("kica,kjcb->ijab", tau8, h.v.oovv)
        )

        tau465 = (
            einsum("jabi->ijab", h.v.ovvo)
            + einsum("ijab->ijab", tau464)
        )

        tau466 = (
            - 2 * einsum("kica,cbjk->ijab", tau465, ea.z2)
        )

        tau467 = (
            einsum("caki,bcjk->ijab", ea.t2, ea.z2)
        )

        tau468 = (
            2 * einsum("kiac,kjcb->ijab", tau250, tau467)
        )

        tau469 = (
            2 * einsum("jabi->ijab", h.v.ovvo)
            + 2 * einsum("ijab->ijab", tau101)
            - einsum("jaib->ijab", h.v.ovov)
        )

        tau470 = (
            2 * einsum("kica,bcjk->ijab", tau469, ea.z2)
        )

        tau471 = (
            2 * einsum("kjlb,ikla->ijab", tau388, tau431)
        )

        tau472 = (
            2 * einsum("jkbc,kica->ijab", tau1, tau181)
        )

        tau473 = (
            einsum("klij,lkba->ijab", tau389, h.v.oovv)
        )

        tau474 = (
            - 2 * einsum("ka,kijb->ijab", tau387, tau22)
        )

        tau475 = (
            einsum("cj,icab->ijab", ea.z1, h.v.ovvv)
        )

        tau476 = (
            einsum("bk,kija->ijab", ea.z1, tau7)
        )

        tau477 = (
            einsum("idab,cdjk->ijkabc", h.v.ovvv, ea.z2)
        )

        tau478 = (
            einsum("ck,kijabc->ijab", ea.t1, tau477)
        )

        tau479 = (
            einsum("ijab->ijab", tau437)
            + einsum("ijab->ijab", tau442)
            + einsum("ijab->ijab", tau443)
            + einsum("ijab->ijab", tau445)
            + einsum("ijab->ijab", tau448)
            + einsum("ijab->ijab", tau449)
            + einsum("ijab->ijab", tau450)
            + einsum("ijab->ijab", tau452)
            + einsum("ijab->ijab", tau455)
            + einsum("ijab->ijab", tau460)
            + einsum("ijab->ijab", tau461)
            + einsum("ijab->ijab", tau463)
            + einsum("ijab->ijab", tau466)
            + einsum("ijab->ijab", tau468)
            + 2 * einsum("ijab->ijab", tau266)
            + einsum("ijab->ijab", tau470)
            + einsum("ijab->ijab", tau471)
            + einsum("ijab->ijab", tau472)
            + einsum("ijab->ijab", tau473)
            + einsum("ijab->ijab", tau474)
            + 4 * einsum("ia,bj->ijab", tau387, ea.z1)
            + 4 * einsum("ijab->ijab", tau475)
            - 4 * einsum("ijab->ijab", tau476)
            - 2 * einsum("ijab->ijab", tau478)
        )

        tau480 = (
            no * nv * einsum("iaia->ia", h.v.ovov)
            - 2 * no * nv * einsum("iaai->ia", h.v.ovvo)
        )

        tau481 = (
            2 * einsum("ai,bj->ijab", ea.t1, ea.t1)
            + einsum("abij->ijab", ea.t2)
        )

        tau482 = (
            - 4 * no * nv * einsum("ai,aj->ija", ea.t1, ea.t1)
            - no * nv * einsum("aaij->ija", ea.t2)
            - no * nv * einsum("aaji->ija", ea.t2)
        )

        tau483 = (
            - no * nv * einsum("aaij->ija", ea.t2)
            - no * nv * einsum("aaji->ija", ea.t2)
        )

        tau484 = (
            no ** 2 * nv * einsum("aaij->ija", ea.t2)
            + no ** 2 * nv * einsum("aaji->ija", ea.t2)
        )

        tau485 = (
            - nv ** 2 * einsum("baab->ab", h.v.vvvv)
            + 2 * nv ** 2 * einsum("baba->ab", h.v.vvvv)
        )

        tau486 = (
            - no * nv * einsum("iaia->ia", h.v.ovov)
            + 2 * no * nv * einsum("iaai->ia", h.v.ovvo)
        )

        tau487 = (
            no * nv ** 2 * einsum("abii->iab", ea.t2)
            + no * nv ** 2 * einsum("baii->iab", ea.t2)
        )

        tau488 = (
            2 * einsum("iaab->iab", h.v.ovvv)
            - einsum("iaba->iab", h.v.ovvv)
        )

        tau489 = (
            - no * nv ** 2 * einsum("baji,iaab->ijab", ea.t2, h.v.ovvv)
            + no ** 2 * nv * einsum("abij,jibj->ijab", ea.t2, h.v.oovo)
        )

        tau490 = (
            no * nv * einsum("iajb->ijab", h.v.ovov)
            - 2 * no * nv * einsum("iabj->ijab", h.v.ovvo)
        )

        tau491 = (
            no * nv ** 2 * einsum("iaab->iab", h.v.ovvv)
            - 2 * no * nv ** 2 * einsum("iaba->iab", h.v.ovvv)
        )

        tau492 = (
            no * nv ** 2 * einsum("iaab->iab", h.v.ovvv)
            + no * nv ** 2 * einsum("iaba->iab", h.v.ovvv)
        )

        tau493 = (
            einsum("aaij->ija", ea.t2)
            + einsum("aaji->ija", ea.t2)
        )

        tau494 = (
            - 2 * einsum("bj,aaii->ijab", ea.t1, ea.t2)
            + einsum("bi,ija->ijab", ea.t1, tau493)
        )

        tau495 = (
            - no ** 2 * nv ** 2 * einsum("ai->ia", ea.t1 ** 2)
            + no ** 2 * nv ** 2 * einsum("aaii->ia", ea.t2)
        )

        tau496 = (
            8 * no ** 2 * nv ** 2 *
            einsum("ai,bj->ijab", ea.t1 ** 2, ea.t1 ** 2)
            + 8 * einsum("jb,aaii->ijab", tau495, ea.t2)
            - no ** 2 * nv ** 2 * einsum("abjj,baii->ijab", ea.t2, ea.t2)
        )

        tau497 = (
            no ** 2 * nv ** 2 * einsum("ai->ia", ea.t1 ** 2)
            - no ** 2 * nv ** 2 * einsum("aaii->ia", ea.t2)
        )

        tau498 = (
            - 4 * no ** 2 * nv ** 2 *
            einsum("ai,bj->ijab", ea.t1 ** 2, ea.t1 ** 2)
            - no ** 2 * nv ** 2 * einsum("abjj,baii->ijab", ea.t2, ea.t2)
            + 4 * einsum("jb,aaii->ijab", tau497, ea.t2)
        )

        tau499 = (
            2 * no ** 2 * nv * einsum("ijia->ija", h.v.ooov)
            - no ** 2 * nv * einsum("ijai->ija", h.v.oovo)
        )

        tau500 = (
            2 * no * nv * einsum("ai,aj->ija", ea.t1, ea.t1)
            - no * nv * einsum("aaij->ija", ea.t2)
            - no * nv * einsum("aaji->ija", ea.t2)
        )

        tau501 = (
            - no * nv * einsum("ia,aj->ija", h.f.ov, ea.t1)
            - no * nv * einsum("iaaj->ija", h.v.ovvo)
            - no * nv * einsum("iaja->ija", h.v.ovov)
        )

        tau502 = (
            einsum("abii->iab", ea.t2)
            + einsum("baii->iab", ea.t2)
        )

        tau503 = (
            2 * no * nv * einsum("iabj->ijab", h.v.ovvo)
            - no * nv * einsum("iajb->ijab", h.v.ovov)
        )

        tau504 = (
            - 2 * no * nv * einsum("abij->ijab", ea.t2)
            + no * nv * einsum("baij->ijab", ea.t2)
            - 2 * no * nv * einsum("baji->ijab", ea.t2)
        )

        tau505 = (
            2 * no * nv * einsum("ia,ai->ia", h.f.ov, ea.t1)
            + no * nv * einsum("iaia->ia", h.v.ovov)
            - 2 * no * nv * einsum("iaai->ia", h.v.ovvo)
        )

        tau506 = (
            einsum("ijia->ija", h.v.ooov)
            + einsum("ijai->ija", h.v.oovo)
        )

        tau507 = (
            2 * no ** 2 * nv * einsum("bi,aajj->ijab", ea.t1, ea.t2)
            - no ** 2 * nv * einsum("ai,abjj->ijab", ea.t1, ea.t2)
        )

        tau508 = (
            - no ** 2 * nv * einsum("ijia->ija", h.v.ooov)
            - no ** 2 * nv * einsum("ijai->ija", h.v.oovo)
        )

        tau509 = (
            2 * no ** 2 * einsum("jiji->ij", h.v.oooo)
            - no ** 2 * einsum("jiij->ij", h.v.oooo)
        )

        tau510 = (
            no * nv * einsum("iaaj->ija", h.v.ovvo)
            - 2 * no * nv * einsum("iaja->ija", h.v.ovov)
        )

        tau511 = (
            - 2 * no * einsum("ai,ibij->ijab", ea.t1, h.v.ovoo)
            + no ** 2 * nv ** 2 *
            einsum("ai,bi,abjj,jiba->ijab", ea.t1, ea.t1, ea.t2, h.v.oovv)
            - no * einsum("ii,abij->ijab", h.f.oo, ea.t2)
            + no ** 2 * einsum("abij,jiji->ijab", ea.t2, h.v.oooo)
        )

        tau512 = (
            nv ** 2 * einsum("abij,baba->ijab", ea.t2, h.v.vvvv)
            + nv * einsum("aa,abij->ijab", h.f.vv, ea.t2)
            + 2 * nv * einsum("ai,abaj->ijab", ea.t1, h.v.vvvo)
            + no ** 2 * nv ** 2 *
            einsum("ai,aj,bbij,jiba->ijab", ea.t1, ea.t1, ea.t2, h.v.oovv)
        )

        tau513 = (
            no ** 2 * nv ** 2 * einsum("jiab->ijab", h.v.oovv)
            - 2 * no ** 2 * nv ** 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau514 = (
            3 * no ** 2 * nv ** 2 *
            einsum("abij,abji,jiba->ijab", ea.t2, ea.t2, h.v.oovv)
            + 2 * no ** 2 * nv *
            einsum("ai,abji,jiaj->ijab", ea.t1, ea.t2, h.v.oovo)
            + einsum("jiba,abij->ijab", tau513, ea.t2 ** 2)
        )

        tau515 = (
            no ** 2 * nv ** 2 * einsum("jiab->ijab", h.v.oovv)
            + no ** 2 * nv ** 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau516 = (
            - 4 * einsum("baji->ijab", h.v.vvoo)
            + no ** 2 * nv ** 2 *
            einsum("aaij,bbji,jiba->ijab", ea.t2, ea.t2, h.v.oovv)
        )

        tau517 = (
            - no ** 2 * nv ** 2 *
            einsum("abii,abjj,jiba->ijab", ea.t2, ea.t2, h.v.oovv)
            + 2 * nv * einsum("ba,aaij->ijab", h.f.vv, ea.t2)
        )

        tau518 = (
            - nv ** 2 * einsum("abij,baab->ijab", ea.t2, h.v.vvvv)
            - nv * einsum("aa,baij->ijab", h.f.vv, ea.t2)
            - no ** 2 * nv ** 2 *
            einsum("ai,aj,bbij,jiab->ijab", ea.t1, ea.t1, ea.t2, h.v.oovv)
            - 2 * nv * einsum("ai,abja->ijab", ea.t1, h.v.vvov)
        )

        tau519 = (
            - no ** 2 * einsum("abij,jiij->ijab", ea.t2, h.v.oooo)
            - no ** 2 * nv ** 2 *
            einsum("ai,bi,abjj,jiab->ijab", ea.t1, ea.t1, ea.t2, h.v.oovv)
            + no * einsum("ii,abji->ijab", h.f.oo, ea.t2)
            + 2 * no * einsum("ai,ibji->ijab", ea.t1, h.v.ovoo)
        )

        tau520 = (
            no * nv ** 2 * einsum("abji,iaab->ijab", ea.t2, h.v.ovvv)
            + no * nv ** 2 * einsum("abij,iaba->ijab", ea.t2, h.v.ovvv)
        )

        tau521 = (
            no ** 2 * nv * einsum("abij,ijia->ijab", ea.t2, h.v.ooov)
            + no ** 2 * nv * einsum("baij,ijai->ijab", ea.t2, h.v.oovo)
        )

        dldt1 = (
            einsum("icbd,bdca->ai", tau351, h.v.vvvv)
            + einsum("ijkb,abjk->ai", tau374, ea.z2) / 2
            + einsum("bc,icba->ai", tau376, tau377) / 2
            + einsum("jicb,jbac->ai", tau381, h.v.ovvv) / 2
            - einsum("ji,ja->ai", tau385, tau387) / 2
            + einsum("kjib,jkba->ai", tau392, tau40) / 2
            + einsum("ijkb,bajk->ai", tau393, ea.z2) / 2
            + einsum("jk,jika->ai", tau385, tau231) / 2
            + einsum("jb,jiab->ai", tau395, h.v.oovv) / 2
            + einsum("kjab,kjib->ai", tau163, tau396)
            + einsum("jicb,jbca->ai", tau397, h.v.ovvv) / 2
            - einsum("ij,aj->ai", tau401, ea.z1)
            + einsum("kjli,jkla->ai", tau402, h.v.ooov) / 2
            + einsum("jicb,bcja->ai", tau55, h.v.vvov)
            + einsum("kjab,kjib->ai", tau250, tau405) / 2
            + einsum("jb,jiba->ai", tau406, h.v.oovv)
            + einsum("jkib,jbka->ai", tau379, h.v.ovov)
            + einsum("jikb,jbak->ai", tau379, h.v.ovvo)
            + 2 * einsum("ba,bi->ai", tau407, ea.z1)
            + 2 * einsum("ia->ai", h.f.ov)
            + 2 * einsum("bj,ijba->ai", ea.z1, tau171)
        )

        dldz1 = (
            einsum("jab,baij->ai", tau408, ea.t2)
            + 2 * einsum("bj,aaii,jiba->ai", ea.t1, ea.t2, tau409)
            + 2 * einsum("bi,ab->ai", ea.t1, tau410)
            + 2 * einsum("ai,aj,bi,jiba->ai", ea.t1, ea.t1, ea.t1, tau411)
            + 2 * einsum("aj,ji->ai", ea.t1, tau413)
            + einsum("ijb,abji->ai", tau412, ea.t2)
            + 2 * einsum("bj,jiab->ai", ea.t1, tau171)
            + einsum("ijb,baji->ai", tau414, ea.t2)
            + einsum("jb,ijab->ai", h.f.ov, tau375)
            - einsum("jiab,jiab->ai", tau415, tau416)
            + einsum("jab,baji->ai", tau417, ea.t2)
            + 2 * einsum("ia->ai", h.f.ov.conj())
        )

        dldt2 = (
            - einsum("jiab->abij", h.v.oovv)
            + 2 * einsum("jiba->abij", h.v.oovv)
            - einsum("ijab->abij", tau436) / 4
            + einsum("ijba->abij", tau436) / 2
            + einsum("jiab->abij", tau436) / 2
            - einsum("jiba->abij", tau436) / 4
            + einsum("ijab->abij", tau479) / 2
            - einsum("ijba->abij", tau479) / 4
            - einsum("jiab->abij", tau479) / 4
            + einsum("jiba->abij", tau479) / 2
        )

        dldz2 = (
            einsum("ia,ijba->abij", tau480, tau481) / 2
            + einsum("jia,ibia->abij", tau482, h.v.ovov) / 2
            + einsum("ia,bi,ija->abij", h.f.ov, ea.t1, tau483) / 2
            + einsum("bi,ija,jija->abij", ea.t1, tau484, h.v.ooov) / 2
            + einsum("ai,bj,ba->abij", ea.t1, ea.t1, tau485)
            + einsum("ia,ijab->abij", tau486, tau8)
            + einsum("aj,iba,iba->abij", ea.t1, tau487, tau488) / 2
            + 3 * einsum("ai,ijba->abij", ea.t1, tau489) / 2
            + einsum("ijba,ai->abij", tau490, ea.t1 ** 2)
            + einsum("bj,iba,ai->abij", ea.t1, tau491, ea.t1 ** 2)
            + einsum("iab,ijba->abij", tau492, tau494) / 2
            + einsum("jiba,ijab->abij", tau496, h.v.oovv) / 4
            + einsum("jiba,ijba->abij", tau498, h.v.oovv) / 4
            + einsum("ai,aj,bj,ija->abij", ea.t1, ea.t1, ea.t1, tau499)
            + einsum("jia,ibai->abij", tau500, h.v.ovvo) / 2
            + einsum("ija,iba->abij", tau501, tau502) / 2
            + einsum("ai,bi,bj,iab->abij", ea.t1, ea.t1, ea.t1, tau491)
            + einsum("aaii,ijba->abij", ea.t2, tau503)
            + einsum("ia,ai,ijab->abij", h.f.ov, ea.t1, tau504)
            + einsum("ia,abji->abij", tau505, ea.t2) / 2
            + einsum("ija,ijab->abij", tau506, tau507) / 2
            + einsum("ai,ija,bajj->abij", ea.t1, tau508, ea.t2) / 2
            + einsum("ai,bj,ji->abij", ea.t1, ea.t1, tau509)
            - einsum("bi,ija,jiaj->abij", ea.t1, tau484, h.v.oovo)
            + einsum("bj,jia,ai->abij", ea.t1, tau499, ea.t1 ** 2)
            + einsum("ai,bi,ija->abij", ea.t1, ea.t1, tau510)
            + no ** 2 * nv ** 2 *
            einsum("abij,baij,jiba->abij", ea.t2, ea.t2, h.v.oovv) / 4
            + no ** 2 * nv ** 2 *
            einsum("abij,baij,jiab->abij", ea.t2, ea.t2, h.v.oovv) / 4
            - no ** 2 * nv ** 2 *
            einsum("abji,baji,ijba->abij", ea.t2, ea.t2, h.v.oovv)
            + no ** 2 * nv ** 2 *
            einsum("abji,baji,ijab->abij", ea.t2, ea.t2, h.v.oovv) / 2
            - no ** 2 * nv ** 2 *
            einsum("aaji,bbij,ijab->abij", ea.t2, ea.t2, h.v.oovv) / 2
            + no ** 2 * nv ** 2 *
            einsum("aaij,bbji,ijba->abij", ea.t2, ea.t2, h.v.oovv) / 4
            - 3 * no ** 2 * nv ** 2 *
            einsum("aaij,bbij,jiba->abij", ea.t2, ea.t2, h.v.oovv) / 4
            + no ** 2 * nv ** 2 *
            einsum("aaji,bbji,ijba->abij", ea.t2, ea.t2, h.v.oovv) / 2
            - no ** 2 * nv ** 2 *
            einsum("aaji,bbji,ijab->abij", ea.t2, ea.t2, h.v.oovv) / 4
            + einsum("ijab->abij", tau511)
            - einsum("ijba->abij", tau511) / 2
            - no * nv ** 2 * einsum("ai,baij,ibba->abij",
                                    ea.t1, ea.t2, h.v.ovvv) / 2
            + no * nv ** 2 * einsum("bi,abij,iaab->abij",
                                    ea.t1, ea.t2, h.v.ovvv) / 2
            + einsum("ijab->abij", tau512)
            - einsum("jiab->abij", tau512) / 2
            + no * nv ** 2 * einsum("ai,baji,ibab->abij",
                                    ea.t1, ea.t2, h.v.ovvv)
            + no * nv ** 2 * einsum("bi,abji,iaba->abij",
                                    ea.t1, ea.t2, h.v.ovvv) / 2
            + no ** 2 * nv ** 2 *
            einsum("aj,bi,baij,jiab->abij", ea.t1, ea.t1, ea.t2, h.v.oovv) / 2
            + no ** 2 * nv ** 2 *
            einsum("aj,bi,abji,ijba->abij", ea.t1, ea.t1, ea.t2, h.v.oovv) / 2
            - 2 * no ** 2 * nv ** 2 *
            einsum("ai,bj,baji,ijab->abij", ea.t1, ea.t1, ea.t2, h.v.oovv)
            + einsum("ijab->abij", tau514) / 4
            - einsum("jiab->abij", tau514) / 4
            - einsum("aj,bi,abij,jiba->abij", ea.t1, ea.t1, ea.t2, tau515)
            + einsum("ai,bj,baij,jiab->abij", ea.t1, ea.t1, ea.t2, tau515) / 2
            + einsum("ai,bj,abji,ijba->abij", ea.t1, ea.t1, ea.t2, tau515) / 2
            - einsum("aj,bi,baji,ijab->abij", ea.t1, ea.t1, ea.t2, tau515)
            + einsum("ijba->abij", tau516) / 4
            - einsum("ijab->abij", tau516) / 2
            + einsum("ijab->abij", tau517) / 4
            + einsum("jiab->abij", tau517) / 4
            + einsum("ijab->abij", tau518) / 2
            - einsum("jiab->abij", tau518)
            + no ** 2 * nv ** 2 *
            einsum("abji,baij,ijab->abij", ea.t2, ea.t2, h.v.oovv) / 4
            + no ** 2 * nv ** 2 *
            einsum("abij,baji,ijba->abij", ea.t2, ea.t2, h.v.oovv) / 2
            + einsum("ijab->abij", tau519) / 2
            - einsum("ijba->abij", tau519)
            + no ** 2 * nv ** 2 *
            einsum("abji,baij,ijba->abij", ea.t2, ea.t2, h.v.oovv) / 4
            - no ** 2 * nv ** 2 *
            einsum("abij,baji,ijab->abij", ea.t2, ea.t2, h.v.oovv)
            - einsum("ai,ijba->abij", ea.t1, tau520) / 2
            - einsum("bi,ijab->abij", ea.t1, tau520)
            + einsum("ai,jiab->abij", ea.t1, tau521) / 2
            + einsum("aj,ijab->abij", ea.t1, tau521)
            - no * einsum("ij,abii->abij", h.f.oo, ea.t2) / 2
            - no * einsum("ij,baii->abij", h.f.oo, ea.t2) / 2
            - no ** 2 * nv * einsum("ai,baji,jija->abij",
                                    ea.t1, ea.t2, h.v.ooov)
            - no ** 2 * nv * einsum("aj,baij,ijia->abij",
                                    ea.t1, ea.t2, h.v.ooov) / 2
            + no ** 2 * nv ** 2 *
            einsum("aj,bi,baij,jiba->abij", ea.t1, ea.t1, ea.t2, h.v.oovv) / 2
            + no ** 2 * nv ** 2 *
            einsum("aj,bi,abji,ijab->abij", ea.t1, ea.t1, ea.t2, h.v.oovv) / 2
            + no ** 2 * nv ** 2 *
            einsum("ai,bj,baji,ijba->abij", ea.t1, ea.t1, ea.t2, h.v.oovv)
        )

        return self.types.EXTENDED_AMPLITUDES_TYPE(
            t1=dldt1, z1=dldz1, t2=dldt2, z2=dldz2)


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

        return self.types.RHS_TYPE(g1=g1, g2=g2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.types.AMPLITUDES_TYPE(
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

        return self.types.AMPLITUDES_TYPE(t1, t2)

    def calc_residuals(self, h, a, g):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.types.RESIDUALS_TYPE(
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
    rhf = hubbard_from_scf(scf.RHF, 22, 22, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import residual_diis_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)
    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_res=1e-6, lam=5,
        max_cycle=100)


def test_cc_lagrangian():   # pragma: nocover
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

    from tcc.cc_solvers import lagrange_min_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf)
    converged, energy, _ = lagrange_min_solver(
        cc, conv_tol_lagr=1e-6,
        max_cycle=100)


if __name__ == '__main__':
    #    test_cc_lagrangian()
    test_mp2_energy()
    test_cc_unitary()
    test_cc_hubbard()
