import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace
from tcc.cc_solvers import CC


class RCCSD_MUL(CC):
    """
    This implements RCCSD algorithm with Mulliken ordered
    amplitudes and integrals
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
        return 'RCCSD_MUL'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_MUL
        return HAM_SPINLESS_FULL_CORE_MUL(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'mul', 'full')
        e_aibj = cc_denom(ham.f, 4, 'mul', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.ovov.transpose().conj() * (- e_aibj)

        return self.types.AMPLITUDES_TYPE(t1, t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        tau0 = (
            - einsum("jaib->ijab", h.v.ovov)
            + 2 * einsum("jbia->ijab", h.v.ovov)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        energy = (
            einsum("ai,ia->", a.t1, tau1)
            + einsum("aibj,jiba->", a.t2, tau0)
        )

        return energy

    def calc_residuals(self, h, a):
        """
        Calculates ersiduals of the CC equations
        """

        tau0 = (
            - einsum("jaib->ijab", h.v.ovov)
            + 2 * einsum("jbia->ijab", h.v.ovov)
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
            2 * einsum("aibj->ijab", a.t2)
            - einsum("biaj->ijab", a.t2)
            - einsum("ajbi->ijab", a.t2)
            + 2 * einsum("bjai->ijab", a.t2)
        )

        tau5 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("icba->iabc", h.v.ovvv)
        )

        tau6 = (
            - 2 * einsum("aibj->ijab", a.t2)
            + einsum("biaj->ijab", a.t2)
            + einsum("ajbi->ijab", a.t2)
            - 2 * einsum("bjai->ijab", a.t2)
        )

        tau7 = (
            2 * einsum("ci,icab->ab", a.t1, tau5)
            + einsum("jica,ibjc->ab", tau6, h.v.ovov)
            - 2 * einsum("ai,ib->ab", a.t1, tau3)
            + 2 * einsum("ab->ab", h.f.vv)
        )

        tau8 = (
            2 * einsum("aibj->ijab", a.t2)
            - einsum("biaj->ijab", a.t2)
        )

        tau9 = (
            2 * einsum("ijka->ijka", h.v.ooov)
            - einsum("kjia->ijka", h.v.ooov)
        )

        tau10 = (
            einsum("jkab,kbia->ij", tau8, h.v.ovov)
            + 2 * einsum("ak,ijka->ij", a.t1, tau9)
            + einsum("akbj,kiab->ij", a.t2, tau0)
            + 2 * einsum("ij->ij", h.f.oo)
        )

        tau11 = (
            - einsum("aibj->ijab", a.t2)
            + 2 * einsum("biaj->ijab", a.t2)
            + 2 * einsum("ajbi->ijab", a.t2)
            - einsum("bjai->ijab", a.t2)
        )

        tau12 = (
            2 * einsum("aijb->ijab", h.v.voov)
            - einsum("jiab->ijab", h.v.oovv)
        )

        tau13 = (
            2 * einsum("jaib->ijab", h.v.ovov)
            - einsum("jbia->ijab", h.v.ovov)
        )

        tau14 = (
            einsum("ckai,kjbc->ijab", a.t2, tau13)
        )

        tau15 = (
            einsum("aick,kjbc->ijab", a.t2, tau13)
        )

        tau16 = (
            einsum("akci,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau17 = (
            einsum("ciak,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau18 = (
            einsum("al,lkji->ijka", a.t1, h.v.oooo)
        )

        tau19 = (
            einsum("aj,jcib->iabc", a.t1, h.v.ovov)
        )

        tau20 = (
            einsum("dcba->abcd", h.v.vvvv)
            + einsum("di,ibca->abcd", a.t1, tau19)
        )

        tau21 = (
            einsum("di,dbca->iabc", a.t1, tau20)
        )

        tau22 = (
            einsum("akci,kcjb->ijab", a.t2, h.v.ovov)
        )

        tau23 = (
            einsum("ciak,kcjb->ijab", a.t2, h.v.ovov)
        )

        tau24 = (
            einsum("ak,ijkb->ijab", a.t1, h.v.ooov)
        )

        tau25 = (
            einsum("ijab->ijab", tau24)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau26 = (
            2 * einsum("aj,jicb->iabc", a.t1, tau25)
        )

        tau27 = (
            einsum("diaj,jdbc->iabc", a.t2, h.v.ovvv)
        )

        tau28 = (
            einsum("iabc->iabc", tau26)
            + 2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", tau27)
        )

        tau29 = (
            2 * einsum("ci,jacb->ijab", a.t1, tau28)
        )

        tau30 = (
            - einsum("ai,ib->ab", a.t1, tau3)
        )

        tau31 = (
            einsum("ab->ab", tau30)
            + einsum("ab->ab", h.f.vv)
        )

        tau32 = (
            2 * einsum("ac,bicj->ijab", tau31, a.t2)
        )

        tau33 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("kjia->ijka", h.v.ooov)
        )

        tau34 = (
            einsum("ak,ijkb->ijab", a.t1, tau33)
        )

        tau35 = (
            2 * einsum("ckbi,kjac->ijab", a.t2, tau34)
        )

        tau36 = (
            einsum("ci,jabc->ijab", a.t1, tau5)
        )

        tau37 = (
            einsum("aibj->ijab", a.t2)
            + einsum("bjai->ijab", a.t2)
        )

        tau38 = (
            2 * einsum("ikcb,jkac->ijab", tau36, tau37)
        )

        tau39 = (
            einsum("ai,ja->ij", a.t1, tau3)
        )

        tau40 = (
            einsum("ij->ij", tau39)
            + einsum("ji->ij", h.f.oo)
        )

        tau41 = (
            - 2 * einsum("ik,ajbk->ijab", tau40, a.t2)
        )

        tau42 = (
            einsum("bi,kbja->ijka", a.t1, h.v.ovov)
        )

        tau43 = (
            einsum("ak,ikjb->ijab", a.t1, tau42)
        )

        tau44 = (
            2 * einsum("kjbc,ikac->ijab", tau37, tau43)
        )

        tau45 = (
            - einsum("iabc->iabc", h.v.ovvv)
            + 2 * einsum("icba->iabc", h.v.ovvv)
        )

        tau46 = (
            2 * einsum("ci,iabc->ab", a.t1, tau45)
        )

        tau47 = (
            - 2 * einsum("jaib->ijab", h.v.ovov)
            + einsum("jbia->ijab", h.v.ovov)
        )

        tau48 = (
            einsum("cibj,jica->ab", a.t2, tau47)
        )

        tau49 = (
            einsum("ab->ab", tau46)
            + einsum("ab->ab", tau48)
        )

        tau50 = (
            einsum("cb,ciaj->ijab", tau49, a.t2)
        )

        tau51 = (
            einsum("bicj,kcab->ijka", a.t2, h.v.ovvv)
        )

        tau52 = (
            einsum("aibl,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau53 = (
            - einsum("ijka->ijka", tau51)
            + einsum("ijka->ijka", tau52)
        )

        tau54 = (
            2 * einsum("ak,ijkb->ijab", a.t1, tau53)
        )

        tau55 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau56 = (
            einsum("aibj,lbka->ijkl", a.t2, h.v.ovov)
        )

        tau57 = (
            2 * einsum("ijkl->ijkl", tau55)
            + einsum("iklj->ijkl", tau56)
        )

        tau58 = (
            einsum("akbl,ikjl->ijab", a.t2, tau57)
        )

        tau59 = (
            2 * einsum("ak,kjia->ij", a.t1, tau33)
        )

        tau60 = (
            einsum("aibj->ijab", a.t2)
            - 2 * einsum("biaj->ijab", a.t2)
        )

        tau61 = (
            einsum("kjab,kbia->ij", tau60, h.v.ovov)
        )

        tau62 = (
            einsum("ij->ij", tau59)
            + einsum("ij->ij", tau61)
        )

        tau63 = (
            einsum("kj,akbi->ijab", tau62, a.t2)
        )

        tau64 = (
            einsum("ak,kijb->ijab", a.t1, h.v.ooov)
        )

        tau65 = (
            einsum("aibj->ijab", a.t2)
            - 2 * einsum("ajbi->ijab", a.t2)
            + einsum("bjai->ijab", a.t2)
        )

        tau66 = (
            2 * einsum("jkac,kibc->ijab", tau64, tau65)
        )

        tau67 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau68 = (
            einsum("ijab->ijab", tau23)
            - 2 * einsum("ijba->ijab", tau67)
        )

        tau69 = (
            einsum("akcj,ikbc->ijab", a.t2, tau68)
        )

        tau70 = (
            - 2 * einsum("kiac,kjbc->ijab", tau37, h.v.oovv)
        )

        tau71 = (
            einsum("ijab->ijab", tau29)
            + einsum("ijab->ijab", tau32)
            + einsum("ijab->ijab", tau35)
            + einsum("ijab->ijab", tau38)
            + einsum("ijab->ijab", tau41)
            + einsum("ijab->ijab", tau44)
            + einsum("ijab->ijab", tau50)
            + einsum("ijab->ijab", tau54)
            + einsum("ijab->ijab", tau58)
            + einsum("ijab->ijab", tau63)
            + einsum("ijab->ijab", tau66)
            + einsum("ijab->ijab", tau69)
            + einsum("ijab->ijab", tau70)
        )

        tau72 = (
            - 2 * einsum("ai,ib->ab", a.t1, tau3)
        )

        tau73 = (
            einsum("jaib->ijab", h.v.ovov)
            - 2 * einsum("jbia->ijab", h.v.ovov)
        )

        tau74 = (
            einsum("aicj,jicb->ab", a.t2, tau73)
        )

        tau75 = (
            einsum("ab->ab", tau72)
            + einsum("ab->ab", tau74)
            + 2 * einsum("ab->ab", h.f.vv)
        )

        tau76 = (
            einsum("ac,cibj->ijab", tau75, a.t2)
        )

        tau77 = (
            - 2 * einsum("aibj->ijab", a.t2)
            + einsum("biaj->ijab", a.t2)
        )

        tau78 = (
            einsum("jkab,kbia->ij", tau77, h.v.ovov)
        )

        tau79 = (
            - 2 * einsum("ijka->ijka", h.v.ooov)
            + einsum("kjia->ijka", h.v.ooov)
        )

        tau80 = (
            2 * einsum("ak,ijka->ij", a.t1, tau79)
        )

        tau81 = (
            einsum("akbj,kiab->ij", a.t2, tau73)
        )

        tau82 = (
            einsum("ij->ij", tau78)
            + einsum("ij->ij", tau80)
            + einsum("ij->ij", tau81)
        )

        tau83 = (
            einsum("kj,aibk->ijab", tau82, a.t2)
        )

        tau84 = (
            2 * einsum("ci,iabc->ab", a.t1, tau45)
        )

        tau85 = (
            - 2 * einsum("aibj->ijab", a.t2)
            + einsum("ajbi->ijab", a.t2)
        )

        tau86 = (
            einsum("ijbc,iajc->ab", tau85, h.v.ovov)
        )

        tau87 = (
            einsum("cibj,jiac->ab", a.t2, tau73)
        )

        tau88 = (
            einsum("ab->ab", tau84)
            + einsum("ab->ab", tau86)
            + einsum("ab->ab", tau87)
        )

        tau89 = (
            einsum("cb,aicj->ijab", tau88, a.t2)
        )

        tau90 = (
            einsum("jkia->ijka", tau42)
            + einsum("ijka->ijka", h.v.ooov)
        )

        tau91 = (
            2 * einsum("al,ijka->ijkl", a.t1, tau90)
        )

        tau92 = (
            2 * einsum("lkji->ijkl", h.v.oooo)
            + einsum("ijkl->ijkl", tau91)
            + einsum("ljki->ijkl", tau56)
        )

        tau93 = (
            einsum("akbl,ljki->ijab", a.t2, tau92)
        )

        tau94 = (
            einsum("aick,kcjb->ijab", a.t2, h.v.ovov)
        )

        tau95 = (
            einsum("ijka->ijka", tau42)
            - 2 * einsum("ikja->ijka", tau42)
        )

        tau96 = (
            einsum("ak,ikjb->ijab", a.t1, tau95)
        )

        tau97 = (
            2 * einsum("ijab->ijab", tau94)
            + einsum("ijab->ijab", tau96)
        )

        tau98 = (
            2 * einsum("ckbj,ikac->ijab", a.t2, tau97)
        )

        tau99 = (
            einsum("aibk,kjab->ij", a.t2, tau13)
        )

        tau100 = (
            2 * einsum("ai,ja->ij", a.t1, tau3)
        )

        tau101 = (
            einsum("ij->ij", tau99)
            + einsum("ij->ij", tau100)
            + 2 * einsum("ji->ij", h.f.oo)
        )

        tau102 = (
            - einsum("ik,akbj->ijab", tau101, a.t2)
        )

        tau103 = (
            einsum("aick,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau104 = (
            - einsum("iabc->iabc", h.v.ovvv)
            + einsum("ibac->iabc", tau19)
        )

        tau105 = (
            einsum("di,icba->abcd", a.t1, tau104)
        )

        tau106 = (
            einsum("dcba->abcd", h.v.vvvv)
            + einsum("abcd->abcd", tau105)
        )

        tau107 = (
            2 * einsum("cidj,dbca->ijab", a.t2, tau106)
        )

        tau108 = (
            einsum("ak,ikjb->ijab", a.t1, tau95)
        )

        tau109 = (
            2 * einsum("bjck,ikac->ijab", a.t2, tau108)
        )

        tau110 = (
            2 * einsum("ckai,kjcb->ijab", a.t2, tau12)
        )

        tau111 = (
            einsum("aijb->ijab", h.v.voov)
            + einsum("jiab->ijab", tau67)
        )

        tau112 = (
            2 * einsum("bi,kjba->ijka", a.t1, tau111)
        )

        tau113 = (
            einsum("albi,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau114 = (
            einsum("ijka->ijka", tau112)
            + 2 * einsum("ikja->ijka", h.v.ooov)
            - einsum("ijka->ijka", tau113)
        )

        tau115 = (
            - 2 * einsum("ak,ijkb->ijab", a.t1, tau114)
        )

        tau116 = (
            2 * einsum("jiab->ijab", tau24)
            + einsum("ijab->ijab", tau16)
        )

        tau117 = (
            einsum("cibk,jkac->ijab", a.t2, tau116)
        )

        tau118 = (
            - 2 * einsum("kiac,bjkc->ijab", tau65, h.v.voov)
        )

        tau119 = (
            einsum("ak,ijkb->ijab", a.t1, tau42)
        )

        tau120 = (
            - einsum("ijab->ijab", tau94)
            + einsum("ijab->ijab", tau119)
        )

        tau121 = (
            2 * einsum("ikac,kjbc->ijab", tau120, tau37)
        )

        tau122 = (
            einsum("ckai,kjbc->ijab", a.t2, tau47)
        )

        tau123 = (
            einsum("jkbc,ikca->ijab", tau122, tau37)
        )

        tau124 = (
            einsum("aick,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau125 = (
            - 2 * einsum("aibj->ijab", a.t2)
            + einsum("biaj->ijab", a.t2)
            + einsum("ajbi->ijab", a.t2)
        )

        tau126 = (
            einsum("ikac,kjcb->ijab", tau124, tau125)
        )

        tau127 = (
            einsum("ci,jcab->ijab", a.t1, h.v.ovvv)
        )

        tau128 = (
            - 2 * einsum("ikbc,kjac->ijab", tau127, tau37)
        )

        tau129 = (
            einsum("ijab->ijab", tau76)
            + einsum("ijab->ijab", tau83)
            + einsum("ijab->ijab", tau89)
            + einsum("ijab->ijab", tau93)
            + einsum("ijab->ijab", tau98)
            + einsum("ijab->ijab", tau102)
            - 2 * einsum("ijab->ijab", tau103)
            + einsum("ijab->ijab", tau107)
            + einsum("ijab->ijab", tau109)
            + einsum("ijab->ijab", tau110)
            + einsum("ijab->ijab", tau115)
            + einsum("ijab->ijab", tau117)
            + einsum("ijab->ijab", tau118)
            + einsum("ijab->ijab", tau121)
            + einsum("ijab->ijab", tau123)
            + einsum("ijab->ijab", tau126)
            + einsum("ijab->ijab", tau128)
        )

        r1 = (
            einsum("jb,jiba->ai", tau3, tau4) / 2
            + einsum("bi,ab->ai", a.t1, tau7) / 2
            - einsum("aj,ji->ai", a.t1, tau10) / 2
            + einsum("kjba,jikb->ai", tau6, h.v.ooov) / 2
            + einsum("jicb,jbac->ai", tau11, h.v.ovvv) / 2
            + einsum("ia->ai", h.f.ov.conj())
            + einsum("bj,jiba->ai", a.t1, tau12)
        )

        r2 = (
            einsum("ckai,jkbc->aibj", a.t2, tau14) / 2
            + einsum("bjck,ikac->aibj", a.t2, tau15) / 2
            + einsum("akcj,ikbc->aibj", a.t2, tau16) / 4
            + einsum("jbia->aibj", h.v.ovov)
            + einsum("cjak,ikbc->aibj", a.t2, tau17) / 4
            + einsum("bk,jkia->aibj", a.t1, tau18)
            + einsum("ci,jabc->aibj", a.t1, tau21)
            + einsum("bkcj,ikac->aibj", a.t2, tau22) / 4
            + einsum("cjbk,ikac->aibj", a.t2, tau23) / 4
            + einsum("ijba->aibj", tau71) / 4
            + einsum("jiab->aibj", tau71) / 4
            + einsum("ijab->aibj", tau129) / 4
            + einsum("jiba->aibj", tau129) / 4
        )

        return self.types.RESIDUALS_TYPE(r1=r1, r2=r2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.types.AMPLITUDES_TYPE(
            *(g[ii] * (- cc_denom(h.f, g[ii].ndim, 'mul', 'full'))
              for ii in range(len(g)))
        )

    def update_rhs(self, h, a, r):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.types.RESIDUALS_TYPE(
            *[r[ii] - a[ii] / cc_denom(h.f, a[ii].ndim, 'mul', 'full')
              for ii in range(len(a))]
        )


class RCCSD_MUL_RI(RCCSD_MUL):
    """
    This implements RCCSD algorithm with Mulliken ordered
    amplitudes and density fitted integrals
    """

    @property
    def method_name(self):
        return 'RCCSD_MUL_RI'

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
        e_ai = cc_denom(ham.f, 2, 'mul', 'full')
        e_aibj = cc_denom(ham.f, 4, 'mul', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)

        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()
        t2 = v_vovo * (- e_aibj)

        return self.types.AMPLITUDES_TYPE(t1, t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        tau0 = (
            einsum("qia,qjb->ijab", h.l.pov, h.l.pov)
        )

        tau1 = (
            einsum("bj,ijba->ia", a.t1, tau0)
        )

        tau2 = (
            einsum("qjb,biaj->qia", h.l.pov, a.t2)
        )

        tau3 = (
            einsum("qjb,aibj->qia", h.l.pov, a.t2)
        )

        tau4 = (
            einsum("ai,qia->q", a.t1, h.l.pov)
        )

        tau5 = (
            einsum("ai,qia->q", a.t1, h.l.pov)
        )

        energy = (
            - einsum("ai,ia->", a.t1, tau1)
            + 2 * einsum("ia,ai->", h.f.ov, a.t1)
            - einsum("qia,qia->", tau2, h.l.pov)
            + 2 * einsum("qia,qia->", tau3, h.l.pov)
            + 2 * einsum("q,q->", tau4, tau5)
        )

        return energy

    def calc_residuals(self, h, a):
        """
        Calculates residuals of CC equations
        Automatically generated
        """

        tau0 = (
            einsum("qia,qjb->ijab", h.l.pov, h.l.pov)
        )

        tau1 = (
            einsum("bj,ijba->ia", a.t1, tau0)
        )

        tau2 = (
            einsum("qjb,biaj->qia", h.l.pov, a.t2)
        )

        tau3 = (
            einsum("qjb,aibj->qia", h.l.pov, a.t2)
        )

        tau4 = (
            einsum("ai,qia->q", a.t1, h.l.pov)
        )

        tau5 = (
            einsum("ai,qia->q", a.t1, h.l.pov)
        )

        tau6 = (
            - einsum("aibj->ijab", a.t2)
            + 2 * einsum("biaj->ijab", a.t2)
            + 2 * einsum("ajbi->ijab", a.t2)
            - einsum("bjai->ijab", a.t2)
        )

        tau7 = (
            einsum("qjb,ijba->qia", h.l.pov, tau6)
        )

        tau8 = (
            einsum("q,qij->ij", tau5, h.l.poo)
        )

        tau9 = (
            einsum("qij,qka->ijka", h.l.poo, h.l.pov)
        )

        tau10 = (
            einsum("ak,kija->ij", a.t1, tau9)
        )

        tau11 = (
            einsum("qja,qia->ij", tau7, h.l.pov)
            + 4 * einsum("ij->ij", tau8)
            + 2 * einsum("ij->ij", h.f.oo)
            - 2 * einsum("ji->ij", tau10)
        )

        tau12 = (
            einsum("bj,jiab->ia", a.t1, tau0)
        )

        tau13 = (
            einsum("q,qia->ia", tau4, h.l.pov)
        )

        tau14 = (
            - einsum("ia->ia", tau12)
            + 2 * einsum("ia->ia", tau13)
            + einsum("ia->ia", h.f.ov)
        )

        tau15 = (
            2 * einsum("aibj->ijab", a.t2)
            - einsum("biaj->ijab", a.t2)
            - einsum("ajbi->ijab", a.t2)
            + 2 * einsum("bjai->ijab", a.t2)
        )

        tau16 = (
            einsum("aibj->ijab", a.t2)
            - 2 * einsum("biaj->ijab", a.t2)
            - 2 * einsum("ajbi->ijab", a.t2)
            + einsum("bjai->ijab", a.t2)
        )

        tau17 = (
            einsum("qjb,ijba->qia", h.l.pov, tau16)
        )

        tau18 = (
            einsum("aj,qji->qia", a.t1, h.l.poo)
        )

        tau19 = (
            einsum("qjb,ijba->qia", h.l.pov, tau16)
            + 2 * einsum("qia->qia", tau18)
        )

        tau20 = (
            2 * einsum("aibj->ijab", a.t2)
            - einsum("biaj->ijab", a.t2)
        )

        tau21 = (
            einsum("qjb,jiba->qia", h.l.pov, tau20)
        )

        tau22 = (
            einsum("bi,qab->qia", a.t1, h.l.pvv)
        )

        tau23 = (
            einsum("qia->qia", tau21)
            + 2 * einsum("qia->qia", tau22)
        )

        tau24 = (
            einsum("q,qab->ab", tau5, h.l.pvv)
        )

        tau25 = (
            einsum("aibj->ijab", a.t2)
            - 2 * einsum("biaj->ijab", a.t2)
        )

        tau26 = (
            einsum("qjb,ijba->qia", h.l.pov, tau25)
        )

        tau27 = (
            einsum("q,qia->ia", tau5, h.l.pov)
        )

        tau28 = (
            2 * einsum("ia->ia", tau27)
            + einsum("ia->ia", h.f.ov)
            - einsum("ia->ia", tau1)
        )

        tau29 = (
            - einsum("qia,qib->ab", tau23, h.l.pov)
            + 4 * einsum("ab->ab", tau24)
            + 2 * einsum("ab->ab", h.f.vv)
            + einsum("qia,qib->ab", tau26, h.l.pov)
            - 2 * einsum("ai,ib->ab", a.t1, tau28)
        )

        tau30 = (
            2 * einsum("aibj->ijab", a.t2)
            - einsum("ajbi->ijab", a.t2)
            + 2 * einsum("bjai->ijab", a.t2)
        )

        tau31 = (
            einsum("ai,qja->qij", a.t1, h.l.pov)
        )

        tau32 = (
            einsum("qji->qij", tau31)
            + einsum("qij->qij", h.l.poo)
        )

        tau33 = (
            - einsum("qjb,ijab->qia", h.l.pov, tau30)
            + 2 * einsum("aj,qji->qia", a.t1, tau32)
        )

        tau34 = (
            einsum("aibj->ijab", a.t2)
            + einsum("bjai->ijab", a.t2)
        )

        tau35 = (
            einsum("aj,ijbc->iabc", a.t1, tau0)
        )

        tau36 = (
            einsum("qab,qcd->abcd", h.l.pvv, h.l.pvv)
            + einsum("ai,icbd->abcd", a.t1, tau35)
        )

        tau37 = (
            einsum("akci,jkcb->ijab", a.t2, tau0)
        )

        tau38 = (
            einsum("qia,qjk->ijka", h.l.pov, h.l.poo)
        )

        tau39 = (
            einsum("albi,jlkb->ijka", a.t2, tau38)
        )

        tau40 = (
            - einsum("aibj->ijab", a.t2)
            + 2 * einsum("biaj->ijab", a.t2)
            + 2 * einsum("ajbi->ijab", a.t2)
        )

        tau41 = (
            einsum("aj,qji->qia", a.t1, h.l.poo)
        )

        tau42 = (
            einsum("qjb,jiab->qia", h.l.pov, tau40)
            - 2 * einsum("qia->qia", tau41)
        )

        tau43 = (
            einsum("qjb,jiba->qia", h.l.pov, tau34)
        )

        tau44 = (
            einsum("aick,kjbc->ijab", a.t2, tau0)
        )

        tau45 = (
            einsum("ak,ijkb->ijab", a.t1, tau9)
        )

        tau46 = (
            - einsum("ijab->ijab", tau44)
            + einsum("jiab->ijab", tau45)
        )

        tau47 = (
            einsum("qjb,biaj->qia", h.l.pov, a.t2)
        )

        tau48 = (
            - einsum("qjb,ijba->qia", h.l.pov, tau40)
            + 2 * einsum("qia->qia", tau18)
        )

        tau49 = (
            - einsum("qjb,ijab->qia", h.l.pov, tau30)
            + 2 * einsum("qia->qia", tau41)
        )

        tau50 = (
            - 2 * einsum("q,qij->ij", tau4, h.l.poo)
            + einsum("ak,ikja->ij", a.t1, tau38)
        )

        tau51 = (
            einsum("ckai,kjbc->ijab", a.t2, tau0)
        )

        tau52 = (
            - 2 * einsum("aibj->ijab", a.t2)
            + einsum("biaj->ijab", a.t2)
            + einsum("ajbi->ijab", a.t2)
            - 2 * einsum("bjai->ijab", a.t2)
        )

        tau53 = (
            einsum("qjb,ajbi->qia", h.l.pov, a.t2)
        )

        tau54 = (
            einsum("qjb,jiba->qia", h.l.pov, tau52)
            + 2 * einsum("qia->qia", tau41)
        )

        tau55 = (
            - 2 * einsum("ij->ij", tau8)
            + einsum("ji->ij", tau10)
        )

        tau56 = (
            einsum("jkcb,kica->ijab", tau0, tau34)
        )

        tau57 = (
            einsum("ai,qja->qij", a.t1, h.l.pov)
        )

        tau58 = (
            einsum("qij->qij", h.l.poo)
            + einsum("qji->qij", tau57)
        )

        tau59 = (
            einsum("qkl,qij->ijkl", tau58, h.l.poo)
            + einsum("aj,ikla->ijkl", a.t1, tau38)
        )

        tau60 = (
            einsum("bial,ljkb->ijka", a.t2, tau9)
        )

        tau61 = (
            einsum("qai->qia", h.l.pvo)
            + einsum("qia->qia", tau22)
        )

        tau62 = (
            einsum("qai->qia", h.l.pvo)
            + einsum("bi,qab->qia", a.t1, h.l.pvv)
        )

        tau63 = (
            2 * einsum("jiab->ijab", tau45)
            + einsum("ciak,jkcb->ijab", a.t2, tau0)
        )

        tau64 = (
            einsum("ak,kijb->ijab", a.t1, tau38)
        )

        tau65 = (
            2 * einsum("jiab->ijab", tau64)
            + einsum("ciak,kjbc->ijab", a.t2, tau0)
        )

        tau66 = (
            einsum("aj,qij->qia", a.t1, tau57)
        )

        tau67 = (
            einsum("aj,qji->qia", a.t1, tau32)
        )

        tau68 = (
            - einsum("aibj->ijab", a.t2)
            + 2 * einsum("biaj->ijab", a.t2)
        )

        tau69 = (
            einsum("qjb,ijba->qia", h.l.pov, tau68)
        )

        tau70 = (
            einsum("qia,qib->ab", tau69, h.l.pov)
        )

        tau71 = (
            einsum("qia,qib->ab", tau23, h.l.pov)
        )

        tau72 = (
            - 4 * einsum("ab->ab", tau24)
            + einsum("ab->ab", tau70)
            + einsum("ab->ab", tau71)
        )

        tau73 = (
            einsum("bc,ciaj->ijab", tau72, a.t2)
        )

        tau74 = (
            einsum("qjb,ijba->qia", h.l.pov, tau16)
        )

        tau75 = (
            einsum("qia->qia", tau74)
            + 2 * einsum("qia->qia", tau66)
        )

        tau76 = (
            2 * einsum("qib,qja->ijab", tau22, tau75)
        )

        tau77 = (
            einsum("qjb,jiba->qia", h.l.pov, tau20)
        )

        tau78 = (
            einsum("qia,qja->ij", tau77, h.l.pov)
        )

        tau79 = (
            einsum("jk,akbi->ijab", tau78, a.t2)
        )

        tau80 = (
            einsum("ai,ib->ab", a.t1, tau28)
        )

        tau81 = (
            einsum("ab->ab", tau80)
            - einsum("ab->ab", h.f.vv)
        )

        tau82 = (
            2 * einsum("ac,bicj->ijab", tau81, a.t2)
        )

        tau83 = (
            einsum("ai,ja->ij", a.t1, tau14)
        )

        tau84 = (
            einsum("ij->ij", tau83)
            + einsum("ji->ij", h.f.oo)
        )

        tau85 = (
            2 * einsum("ik,ajbk->ijab", tau84, a.t2)
        )

        tau86 = (
            einsum("aibj,ijcd->abcd", a.t2, tau0)
        )

        tau87 = (
            einsum("qab,qic->iabc", h.l.pvv, h.l.pov)
        )

        tau88 = (
            einsum("ai,ibcd->abcd", a.t1, tau87)
        )

        tau89 = (
            - einsum("abdc->abcd", tau86)
            + 2 * einsum("abcd->abcd", tau88)
        )

        tau90 = (
            einsum("cidj,abcd->ijab", a.t2, tau89)
        )

        tau91 = (
            einsum("qab,qij->ijab", h.l.pvv, h.l.poo)
        )

        tau92 = (
            2 * einsum("kiac,kjbc->ijab", tau34, tau91)
        )

        tau93 = (
            einsum("bi,jkab->ijka", a.t1, tau0)
        )

        tau94 = (
            einsum("ak,ikjb->ijab", a.t1, tau93)
        )

        tau95 = (
            - 2 * einsum("jkcb,ikac->ijab", tau34, tau94)
        )

        tau96 = (
            einsum("ci,jabc->ijab", a.t1, tau87)
        )

        tau97 = (
            2 * einsum("jkac,ikbc->ijab", tau34, tau96)
        )

        tau98 = (
            4 * einsum("qja,qib->ijab", tau18, tau61)
        )

        tau99 = (
            einsum("ijab->ijab", tau73)
            + einsum("ijab->ijab", tau76)
            + einsum("ijab->ijab", tau79)
            + einsum("ijab->ijab", tau82)
            + einsum("ijab->ijab", tau85)
            + einsum("ijab->ijab", tau90)
            + einsum("ijab->ijab", tau92)
            + einsum("ijab->ijab", tau95)
            + einsum("ijab->ijab", tau97)
            + einsum("ijab->ijab", tau98)
        )

        tau100 = (
            einsum("cidj,abdc->ijab", a.t2, tau89)
        )

        tau101 = (
            einsum("qia,qja->ij", tau69, h.l.pov)
        )

        tau102 = (
            2 * einsum("ai,ja->ij", a.t1, tau14)
        )

        tau103 = (
            einsum("ij->ij", tau101)
            + einsum("ij->ij", tau102)
            + 2 * einsum("ji->ij", h.f.oo)
        )

        tau104 = (
            einsum("ik,akbj->ijab", tau103, a.t2)
        )

        tau105 = (
            einsum("bc,aicj->ijab", tau72, a.t2)
        )

        tau106 = (
            2 * einsum("ac,cibj->ijab", tau81, a.t2)
        )

        tau107 = (
            2 * einsum("kica,kjbc->ijab", tau34, tau91)
        )

        tau108 = (
            - 2 * einsum("jkbc,ikac->ijab", tau34, tau94)
        )

        tau109 = (
            einsum("qia,qja->ij", tau7, h.l.pov)
        )

        tau110 = (
            einsum("jk,aibk->ijab", tau109, a.t2)
        )

        tau111 = (
            einsum("di,abcd->iabc", a.t1, tau86)
        )

        tau112 = (
            einsum("ci,jabc->ijab", a.t1, tau111)
        )

        tau113 = (
            2 * einsum("qia,qbj->ijab", tau75, h.l.pvo)
        )

        tau114 = (
            einsum("aj,qij->qia", a.t1, tau31)
        )

        tau115 = (
            2 * einsum("qia,qjb->ijab", tau114, tau7)
        )

        tau116 = (
            2 * einsum("jkca,ikbc->ijab", tau34, tau96)
        )

        tau117 = (
            einsum("ijab->ijab", tau100)
            + einsum("ijab->ijab", tau104)
            + einsum("ijab->ijab", tau105)
            + einsum("ijab->ijab", tau106)
            + einsum("ijab->ijab", tau107)
            + einsum("ijab->ijab", tau108)
            + einsum("ijab->ijab", tau110)
            - 2 * einsum("ijab->ijab", tau112)
            + einsum("ijab->ijab", tau113)
            + einsum("ijab->ijab", tau115)
            + einsum("ijab->ijab", tau116)
        )

        r1 = (
            - einsum("aj,ji->ai", a.t1, tau11) / 2
            + einsum("jb,jiba->ai", tau14, tau15) / 2
            + einsum("qja,qji->ai", tau17, h.l.poo) / 2
            - einsum("qib,qab->ai", tau19, h.l.pvv) / 2
            + einsum("bi,ab->ai", a.t1, tau29) / 2
            + einsum("ia->ai", h.f.ov.conj())
            + 2 * einsum("q,qai->ai", tau5, h.l.pvo)
        )

        r2 = (
            einsum("qjb,qia->aibj", tau18, tau33) / 2
            + einsum("jidc,acbd->aibj", tau34, tau36) / 2
            + einsum("jkca,ikbc->aibj", tau34, tau37) / 4
            + einsum("ak,ikjb->aibj", a.t1, tau39) / 2
            + einsum("qia,qjb->aibj", tau42, tau43) / 2
            + einsum("jkbc,ikac->aibj", tau34, tau46) / 2
            + einsum("qia,qjb->aibj", tau47, tau48) / 4
            + einsum("qjb,qia->aibj", tau2, tau49) / 4
            + einsum("kj,ikab->aibj", tau50, tau34) / 2
            + einsum("ikac,jkbc->aibj", tau51, tau52) / 4
            + einsum("qjb,qia->aibj", tau53, tau54) / 4
            + einsum("ki,kjab->aibj", tau55, tau34) / 2
            + einsum("kiac,jkbc->aibj", tau34, tau56) / 4
            + einsum("lkba,kilj->aibj", tau34, tau59) / 2
            + einsum("bk,jika->aibj", a.t1, tau60) / 2
            + einsum("qia,qjb->aibj", tau61, tau62)
            + einsum("akcj,ikbc->aibj", a.t2, tau63) / 4
            + einsum("jkcb,ikac->aibj", tau34, tau44) / 4
            + einsum("kica,kjbc->aibj", tau34, tau64) / 2
            + einsum("cibk,jkac->aibj", a.t2, tau65) / 4
            + einsum("qjb,qia->aibj", tau66, tau67)
            - einsum("ijba->aibj", tau99) / 4
            - einsum("jiab->aibj", tau99) / 4
            - einsum("ijab->aibj", tau117) / 4
            - einsum("jiba->aibj", tau117) / 4
        )

        return self.types.RESIDUALS_TYPE(r1=r1, r2=r2)


class RCCSD_MUL_RI_HUB(RCCSD_MUL_RI):
    """
    This class implements CCSD RI for Hubbard hamiltonian,
    as it may be not easy to feed a decomposed hamiltonian to
    pyscf directly
    """

    @property
    def method_name(self):
        return 'RCCSD_MUL_RI_HUB'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


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
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    cc = RCCSD_MUL_RI(rhf)
    h = cc.create_ham()
    a = cc.init_amplitudes(h)
    energy = cc.calculate_energy(h, a)
    print('E_mp2 - E_cc,init = {:18.12g}'.format(energy - -0.204019967288338))


def test_cc():  # pragma: nocover
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
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import classic_solver
    from tcc.rccsd_mul import RCCSD_MUL_RI
    cc = RCCSD_MUL_RI(rhf)
    converged, energy, _ = classic_solver(cc)


def test_cc_hubbard_ri():   # pragma: nocover
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 22, 22, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import residual_diis_solver
    from tcc.rccsd_mul import RCCSD_MUL_RI_HUB
    cc = RCCSD_MUL_RI_HUB(rhf)
    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_res=1e-6, lam=5,
        max_cycle=100)

if __name__ == '__main__':
    test_mp2_energy()
    test_cc()
    test_cc_hubbard_ri()
