import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from collections import namedtuple
from types import SimpleNamespace


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

    def calc_residuals(self, h, a):
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
            - einsum("iabc->iabc", h.v.ovvv)
            + 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau3 = (
            einsum("ci,iabc->ab", a.t1, tau2)
            + einsum("ab->ab", h.f.vv)
        )

        tau4 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau5 = (
            einsum("jkia->ijka", h.v.ooov)
            - 2 * einsum("kjia->ijka", h.v.ooov)
            - 2 * einsum("ijka->ijka", tau4)
            + einsum("ikja->ijka", tau4)
        )

        tau6 = (
            2 * einsum("iabj->ijab", h.v.ovvo)
            - einsum("iajb->ijab", h.v.ovov)
        )

        tau7 = (
            einsum("bj,jiba->ia", a.t1, tau0)
        )

        tau8 = (
            einsum("ia->ia", tau7)
            + einsum("ia->ia", h.f.ov)
        )

        tau9 = (
            - einsum("abji->ijab", a.t2)
            + 2 * einsum("baji->ijab", a.t2)
        )

        tau10 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau11 = (
            - einsum("ijka->ijka", h.v.ooov)
            + 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau12 = (
            einsum("bakj,kiab->ij", a.t2, tau10)
            + einsum("ak,kija->ij", a.t1, tau11)
            + einsum("aj,ia->ij", a.t1, tau8)
            + einsum("ij->ij", h.f.oo)
        )

        tau13 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau14 = (
            einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau15 = (
            einsum("ai,jkla->ijkl", a.t1, tau4)
        )

        tau16 = (
            einsum("lkji->ijkl", tau14)
            + einsum("jilk->ijkl", h.v.oooo)
            + einsum("lkji->ijkl", tau15)
        )

        tau17 = (
            einsum("abji->ijab", a.t2)
            - 2 * einsum("baji->ijab", a.t2)
        )

        tau18 = (
            einsum("jica,ijbc->ab", tau17, h.v.oovv)
        )

        tau19 = (
            einsum("bcakij->ijkabc", a.t3)
            + einsum("cabkij->ijkabc", a.t3)
        )

        tau20 = (
            einsum("al,lkji->ijka", a.t1, tau16)
        )

        tau21 = (
            2 * einsum("kica,kjcb->ijab", tau9, h.v.oovv)
        )

        tau22 = (
            einsum("kica,kjbc->ijab", tau17, h.v.oovv)
        )

        tau23 = (
            einsum("ijab->ijab", tau21)
            + einsum("ijab->ijab", tau22)
        )

        tau24 = (
            einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau25 = (
            - 2 * einsum("abji->ijab", a.t2)
            + einsum("baji->ijab", a.t2)
        )

        tau26 = (
            einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau27 = (
            einsum("di,badc->iabc", a.t1, h.v.vvvv)
        )

        tau28 = (
            einsum("ai,ja->ij", a.t1, tau8)
        )

        tau29 = (
            einsum("ij->ij", tau28)
            + einsum("ji->ij", h.f.oo)
        )

        tau30 = (
            - einsum("ik,bakj->ijab", tau29, a.t2)
        )

        tau31 = (
            einsum("ijka->ijka", tau4)
            - 2 * einsum("ikja->ijka", tau4)
        )

        tau32 = (
            einsum("iklc,cabljk->ijab", tau31, a.t3)
        )

        tau33 = (
            einsum("ac,cbji->ijab", h.f.vv, a.t2)
        )

        tau34 = (
            einsum("ci,abjc->ijab", a.t1, h.v.vvov)
        )

        tau35 = (
            - einsum("ib,bakj->ijka", tau8, a.t2)
        )

        tau36 = (
            einsum("bi,jakb->ijka", a.t1, h.v.ovov)
        )

        tau37 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau38 = (
            einsum("balj,likb->ijka", a.t2, tau37)
        )

        tau39 = (
            einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau40 = (
            einsum("libc,cabljk->ijka", tau13, a.t3)
        )

        tau41 = (
            einsum("bi,jkab->ijka", a.t1, tau26)
        )

        tau42 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau43 = (
            einsum("al,ijlk->ijka", a.t1, tau42)
        )

        tau44 = (
            einsum("ijka->ijka", tau35)
            - einsum("jika->ijka", tau36)
            + einsum("ijka->ijka", tau38)
            + einsum("jika->ijka", tau39)
            + einsum("ijka->ijka", tau40)
            + einsum("jkia->ijka", tau41)
            + einsum("jika->ijka", tau43)
        )

        tau45 = (
            einsum("ak,kijb->ijab", a.t1, tau44)
        )

        tau46 = (
            einsum("kljc,bcalik->ijab", h.v.ooov, a.t3)
        )

        tau47 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau48 = (
            einsum("ikbc,kjca->ijab", tau47, tau9)
        )

        tau49 = (
            einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau50 = (
            einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau51 = (
            einsum("ci,jabc->ijab", a.t1, tau50)
        )

        tau52 = (
            einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
        )

        tau53 = (
            einsum("ci,jabc->ijab", a.t1, tau52)
        )

        tau54 = (
            einsum("ijab->ijab", tau30)
            + einsum("ijab->ijab", tau32)
            + einsum("ijab->ijab", tau33)
            + einsum("ijab->ijab", tau34)
            + einsum("ijab->ijab", tau45)
            + einsum("ijab->ijab", tau46)
            + einsum("ijab->ijab", tau48)
            - einsum("ijab->ijab", tau49)
            - einsum("ijab->ijab", tau51)
            + einsum("ijab->ijab", tau53)
        )

        tau55 = (
            einsum("lkcb,abclij->ijka", h.v.oovv, a.t3)
        )

        tau56 = (
            einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau57 = (
            einsum("bi,jkab->ijka", a.t1, tau47)
        )

        tau58 = (
            2 * einsum("ijka->ijka", tau4)
            - einsum("ikja->ijka", tau4)
        )

        tau59 = (
            einsum("balk,iljb->ijka", a.t2, tau58)
        )

        tau60 = (
            einsum("bi,jkab->ijka", a.t1, tau24)
        )

        tau61 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau62 = (
            einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
        )

        tau63 = (
            - einsum("ikja->ijka", tau55)
            + einsum("ikja->ijka", tau56)
            + einsum("ikja->ijka", tau57)
            + einsum("ijka->ijka", tau59)
            - einsum("ikja->ijka", tau60)
            - einsum("ijka->ijka", tau61)
            + einsum("ijka->ijka", tau62)
            + einsum("jaik->ijka", h.v.ovoo)
        )

        tau64 = (
            einsum("ak,ikjb->ijab", a.t1, tau63)
        )

        tau65 = (
            2 * einsum("ijka->ijka", h.v.ooov)
            - einsum("jika->ijka", h.v.ooov)
        )

        tau66 = (
            einsum("kljc,backil->ijab", tau65, a.t3)
        )

        tau67 = (
            einsum("bj,jiba->ia", a.t1, tau0)
        )

        tau68 = (
            einsum("kc,backij->ijab", tau67, a.t3)
        )

        tau69 = (
            einsum("kica,kbcj->ijab", tau17, h.v.ovvo)
        )

        tau70 = (
            einsum("caki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau71 = (
            einsum("iabc->iabc", h.v.ovvv)
            - 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau72 = (
            einsum("kbcd,dackij->ijab", tau71, a.t3)
        )

        tau73 = (
            einsum("ci,iabc->ab", a.t1, tau71)
        )

        tau74 = (
            einsum("bc,caji->ijab", tau73, a.t2)
        )

        tau75 = (
            einsum("kjba,kiba->ij", tau9, h.v.oovv)
        )

        tau76 = (
            einsum("ak,ikja->ij", a.t1, tau65)
        )

        tau77 = (
            einsum("ij->ij", tau75)
            + einsum("ij->ij", tau76)
        )

        tau78 = (
            einsum("kj,baki->ijab", tau77, a.t2)
        )

        tau79 = (
            einsum("kbcd,acdkij->ijab", h.v.ovvv, a.t3)
        )

        tau80 = (
            einsum("kjdc,bdakij->iabc", h.v.oovv, a.t3)
        )

        tau81 = (
            einsum("ci,jabc->ijab", a.t1, tau80)
        )

        tau82 = (
            einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau83 = (
            einsum("ci,jabc->ijab", a.t1, tau82)
        )

        tau84 = (
            einsum("ijab->ijab", tau64)
            + einsum("ijab->ijab", tau66)
            + einsum("ijab->ijab", tau68)
            + einsum("ijab->ijab", tau69)
            + einsum("ijab->ijab", tau70)
            + einsum("ijab->ijab", tau72)
            + einsum("ijab->ijab", tau74)
            + einsum("ijab->ijab", tau78)
            + einsum("ijab->ijab", tau79)
            - einsum("ijab->ijab", tau81)
            + einsum("ijab->ijab", tau83)
        )

        tau85 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau82)
        )

        tau86 = (
            einsum("kljd,cablik->ijabcd", h.v.ooov, a.t3)
        )

        tau87 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau86)
        )

        tau88 = (
            - einsum("ijka->ijka", tau41)
            + einsum("ikja->ijka", tau36)
        )

        tau89 = (
            einsum("balj,iklc->ijkabc", a.t2, tau88)
        )

        tau90 = (
            einsum("kced,beakij->ijabcd", h.v.ovvv, a.t3)
        )

        tau91 = (
            einsum("di,kjabcd->ijkabc", a.t1, tau90)
        )

        tau92 = (
            - einsum("ijkabc->ijkabc", tau87)
            + einsum("ijkabc->ijkabc", tau89)
            + einsum("ijkabc->ijkabc", tau91)
        )

        tau93 = (
            einsum("bakj,kiab->ij", a.t2, tau10)
        )

        tau94 = (
            einsum("aj,ia->ij", a.t1, tau8)
        )

        tau95 = (
            einsum("ij->ij", tau93)
            + einsum("ij->ij", tau94)
            + einsum("ij->ij", h.f.oo)
        )

        tau96 = (
            einsum("li,cbalkj->ijkabc", tau95, a.t3)
        )

        tau97 = (
            einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
        )

        tau98 = (
            einsum("ai,bj->ijab", a.t1, a.t1)
            + einsum("baji->ijab", a.t2)
        )

        tau99 = (
            - einsum("mlba,ijklmc->ijkabc", tau98, tau97)
        )

        tau100 = (
            - 2 * einsum("iabc->iabc", h.v.ovvv)
            + einsum("iacb->iabc", h.v.ovvv)
        )

        tau101 = (
            einsum("ci,jabc->ijab", a.t1, tau100)
        )

        tau102 = (
            einsum("ilcd,dbalkj->ijkabc", tau101, a.t3)
        )

        tau103 = (
            einsum("ijkabc->ijkabc", tau96)
            + einsum("ijkabc->ijkabc", tau99)
            + einsum("ijkabc->ijkabc", tau102)
        )

        tau104 = (
            einsum("kica,kjcb->ijab", tau17, h.v.oovv)
        )

        tau105 = (
            einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau106 = (
            einsum("ijab->ijab", tau104)
            + einsum("ijab->ijab", tau105)
        )

        tau107 = (
            einsum("ilad,cbdljk->ijkabc", tau106, a.t3)
        )

        tau108 = (
            einsum("bamj,imkl->ijklab", a.t2, tau42)
        )

        tau109 = (
            einsum("lkdc,bdalij->ijkabc", h.v.oovv, a.t3)
        )

        tau110 = (
            einsum("ci,kjlabc->ijklab", a.t1, tau109)
        )

        tau111 = (
            einsum("daji,kbcd->ijkabc", a.t2, h.v.ovvv)
        )

        tau112 = (
            einsum("ci,jklabc->ijklab", a.t1, tau111)
        )

        tau113 = (
            einsum("ijklab->ijklab", tau108)
            + einsum("ijlkab->ijklab", tau110)
            - einsum("ijlkab->ijklab", tau112)
        )

        tau114 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau113)
        )

        tau115 = (
            einsum("lcdk,badlij->ijkabc", h.v.ovvo, a.t3)
        )

        tau116 = (
            einsum("ijka->ijka", tau55)
            - einsum("kaij->ijka", h.v.ovoo)
        )

        tau117 = (
            einsum("bali,jklc->ijkabc", a.t2, tau116)
        )

        tau118 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau80)
        )

        tau119 = (
            einsum("ijkabc->ijkabc", tau107)
            + einsum("ijkabc->ijkabc", tau114)
            - einsum("ijkabc->ijkabc", tau115)
            + einsum("ijkabc->ijkabc", tau117)
            + einsum("ijkabc->ijkabc", tau118)
        )

        tau120 = (
            einsum("adji,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau121 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau120)
        )

        tau122 = (
            einsum("jiab->ijab", h.v.oovv)
            - 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau123 = (
            einsum("lkcb,cablij->ijka", tau122, a.t3)
        )

        tau124 = (
            - 2 * einsum("ijka->ijka", h.v.ooov)
            + einsum("jika->ijka", h.v.ooov)
        )

        tau125 = (
            einsum("bali,kljb->ijka", a.t2, tau124)
        )

        tau126 = (
            einsum("ijka->ijka", tau123)
            + einsum("ijka->ijka", tau125)
        )

        tau127 = (
            einsum("bali,jklc->ijkabc", a.t2, tau126)
        )

        tau128 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", h.v.ovvv)
        )

        tau129 = (
            einsum("daji,jbdc->iabc", a.t2, tau128)
        )

        tau130 = (
            einsum("kjcd,dabkij->iabc", tau13, a.t3)
        )

        tau131 = (
            einsum("iabc->iabc", tau129)
            + einsum("abic->iabc", h.v.vvov)
            + einsum("iabc->iabc", tau130)
        )

        tau132 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau131)
        )

        tau133 = (
            einsum("caji,lkcb->ijklab", a.t2, h.v.oovv)
        )

        tau134 = (
            einsum("bi,jklmab->ijklma", a.t1, tau133)
        )

        tau135 = (
            einsum("am,ijklmb->ijklab", a.t1, tau134)
        )

        tau136 = (
            einsum("al,ijklbc->ijkabc", a.t1, tau135)
        )

        tau137 = (
            - einsum("kb,baij->ijka", tau8, a.t2)
        )

        tau138 = (
            einsum("jilk->ijkl", tau14)
            + einsum("ijkl->ijkl", tau15)
        )

        tau139 = (
            einsum("al,jilk->ijka", a.t1, tau138)
        )

        tau140 = (
            einsum("ijka->ijka", tau137)
            + einsum("ijka->ijka", tau139)
        )

        tau141 = (
            einsum("cblk,jila->ijkabc", a.t2, tau140)
        )

        tau142 = (
            einsum("lckd,badlij->ijkabc", h.v.ovov, a.t3)
        )

        tau143 = (
            einsum("baji,jidc->abcd", a.t2, h.v.oovv)
        )

        tau144 = (
            einsum("ecji,baed->ijabcd", a.t2, tau143)
        )

        tau145 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau144)
        )

        tau146 = (
            einsum("ijkabc->ijkabc", tau127)
            + einsum("ijkabc->ijkabc", tau132)
            + einsum("ijkabc->ijkabc", tau136)
            + einsum("ijkabc->ijkabc", tau141)
            - einsum("ijkabc->ijkabc", tau142)
            + einsum("ijkabc->ijkabc", tau145)
        )

        tau147 = (
            - einsum("jaib->ijab", h.v.ovov)
            + einsum("ijab->ijab", tau26)
        )

        tau148 = (
            einsum("caji,lkbc->ijklab", a.t2, tau147)
        )

        tau149 = (
            einsum("kmlc,bacmij->ijklab", h.v.ooov, a.t3)
        )

        tau150 = (
            einsum("ijklab->ijklab", tau148)
            + einsum("ijklab->ijklab", tau149)
        )

        tau151 = (
            - einsum("al,ijlkbc->ijkabc", a.t1, tau150)
        )

        tau152 = (
            einsum("ijka->ijka", tau56)
            + einsum("ijka->ijka", tau57)
        )

        tau153 = (
            einsum("balk,ijlc->ijkabc", a.t2, tau152)
        )

        tau154 = (
            einsum("ijkabc->ijkabc", tau151)
            + einsum("ijkabc->ijkabc", tau153)
        )

        tau155 = (
            einsum("imlc,cbamkj->ijklab", tau65, a.t3)
        )

        tau156 = (
            einsum("ic,cbalkj->ijklab", tau8, a.t3)
        )

        tau157 = (
            einsum("ijklab->ijklab", tau155)
            + einsum("ijklab->ijklab", tau156)
        )

        tau158 = (
            - einsum("al,lijkbc->ijkabc", a.t1, tau157)
        )

        tau159 = (
            einsum("jica,ijbc->ab", tau17, h.v.oovv)
        )

        tau160 = (
            einsum("ab->ab", tau159)
            + einsum("ab->ab", h.f.vv)
        )

        tau161 = (
            einsum("ad,dbckij->ijkabc", tau160, a.t3)
        )

        tau162 = (
            einsum("jiml,cabmkl->ijkabc", tau138, a.t3)
        )

        tau163 = (
            einsum("ijkabc->ijkabc", tau158)
            + einsum("ijkabc->ijkabc", tau161)
            + einsum("ijkabc->ijkabc", tau162)
        )

        tau164 = (
            einsum("eaji,cbed->ijabcd", a.t2, h.v.vvvv)
        )

        tau165 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau164)
        )

        tau166 = (
            einsum("balj,iklb->ijka", a.t2, tau31)
        )

        tau167 = (
            einsum("ijka->ijka", tau166)
            - einsum("ikja->ijka", tau62)
            + einsum("ijka->ijka", tau60)
        )

        tau168 = (
            einsum("balj,iklc->ijkabc", a.t2, tau167)
        )

        tau169 = (
            einsum("kica,kjcb->ijab", tau17, h.v.oovv)
        )

        tau170 = (
            - einsum("jabi->ijab", h.v.ovvo)
            + einsum("ijab->ijab", tau169)
            + einsum("ijab->ijab", tau105)
        )

        tau171 = (
            einsum("caji,lkbc->ijklab", a.t2, tau170)
        )

        tau172 = (
            einsum("mklc,bacmij->ijklab", h.v.ooov, a.t3)
        )

        tau173 = (
            einsum("bami,mjlk->ijklab", a.t2, h.v.oooo)
        )

        tau174 = (
            einsum("lbcd,dackij->ijklab", h.v.ovvv, a.t3)
        )

        tau175 = (
            einsum("ijklab->ijklab", tau171)
            + einsum("ijklab->ijklab", tau172)
            + einsum("ikjlab->ijklab", tau173)
            - einsum("ijlkab->ijklab", tau174)
        )

        tau176 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau175)
        )

        tau177 = (
            einsum("kcde,beakij->ijabcd", h.v.ovvv, a.t3)
        )

        tau178 = (
            einsum("di,kjabcd->ijkabc", a.t1, tau177)
        )

        tau179 = (
            einsum("ijkabc->ijkabc", tau165)
            + einsum("ijkabc->ijkabc", tau168)
            + einsum("ijkabc->ijkabc", tau176)
            - einsum("ijkabc->ijkabc", tau178)
        )

        tau180 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau181 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau180)
        )

        tau182 = (
            einsum("ak,ikja->ij", a.t1, tau124)
        )

        tau183 = (
            einsum("lk,cablij->ijkabc", tau182, a.t3)
        )

        tau184 = (
            einsum("ci,iabc->ab", a.t1, tau2)
        )

        tau185 = (
            einsum("cd,dabkij->ijkabc", tau184, a.t3)
        )

        tau186 = (
            einsum("lkcd,dablij->ijkabc", tau6, a.t3)
        )

        tau187 = (
            einsum("ijkabc->ijkabc", tau183)
            + einsum("ijkabc->ijkabc", tau185)
            + einsum("ijkabc->ijkabc", tau186)
        )

        tau188 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau39)
        )

        tau189 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau120)
        )

        tau190 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau39)
        )

        tau191 = (
            - 2 * einsum("ijka->ijka", tau4)
            + einsum("ikja->ijka", tau4)
        )

        tau192 = (
            einsum("imlc,cbamkj->ijklab", tau191, a.t3)
        )

        tau193 = (
            einsum("al,ikjlcb->ijkabc", a.t1, tau192)
        )

        tau194 = (
            einsum("cbed,eadkij->ijkabc", h.v.vvvv, a.t3)
        )

        tau195 = (
            einsum("mlkj,cabmil->ijkabc", h.v.oooo, a.t3)
        )

        tau196 = (
            einsum("ilad,dcblkj->ijkabc", tau23, a.t3)
        )

        tau197 = (
            einsum("ijkabc->ijkabc", tau193)
            + einsum("ikjacb->ijkabc", tau194)
            + einsum("ikjacb->ijkabc", tau195)
            + einsum("ijkabc->ijkabc", tau196)
        )

        tau198 = (
            einsum("bamj,ikml->ijklab", a.t2, tau42)
        )

        tau199 = (
            einsum("lkcd,bdalij->ijkabc", h.v.oovv, a.t3)
        )

        tau200 = (
            einsum("ci,kjlabc->ijklab", a.t1, tau199)
        )

        tau201 = (
            einsum("daji,kbdc->ijkabc", a.t2, h.v.ovvv)
        )

        tau202 = (
            einsum("ci,jklabc->ijklab", a.t1, tau201)
        )

        tau203 = (
            einsum("baji,klmb->ijklma", a.t2, h.v.ooov)
        )

        tau204 = (
            einsum("am,ijkmlb->ijklab", a.t1, tau203)
        )

        tau205 = (
            einsum("ijklab->ijklab", tau198)
            + einsum("ijlkab->ijklab", tau200)
            - einsum("ijlkab->ijklab", tau202)
            + einsum("ijklab->ijklab", tau204)
        )

        tau206 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau205)
        )

        tau207 = (
            einsum("dcji,kabd->ijkabc", a.t2, tau52)
        )

        tau208 = (
            einsum("lkcd,badlij->ijkabc", h.v.oovv, a.t3)
        )

        tau209 = (
            einsum("adli,jklbcd->ijkabc", a.t2, tau208)
        )

        tau210 = (
            einsum("ijkabc->ijkabc", tau206)
            + einsum("ijkabc->ijkabc", tau207)
            + einsum("ijkabc->ijkabc", tau209)
        )

        tau211 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau82)
        )

        tau212 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau180)
        )

        r1 = (
            einsum("bi,ab->ai", a.t1, tau3)
            + einsum("bakj,ikjb->ai", a.t2, tau5)
            + einsum("bj,jiab->ai", a.t1, tau6)
            + einsum("jb,jiba->ai", tau8, tau9)
            + einsum("ia->ai", h.f.ov.conj())
            + einsum("jibc,jabc->ai", tau9, h.v.ovvv)
            - einsum("aj,ji->ai", a.t1, tau12)
            + einsum("kjbc,bacjik->ai", tau10, a.t3)
            + einsum("kjbc,cbakij->ai", tau13, a.t3)
        )

        r2 = (
            einsum("abkl,lkji->abij", a.t2, tau16)
            + einsum("bc,caji->abij", tau18, a.t2)
            - einsum("kc,kijacb->abij", h.f.ov, tau19)
            + einsum("ac,cbij->abij", tau18, a.t2)
            + einsum("ak,ijkb->abij", a.t1, tau20)
            + einsum("caki,jkbc->abij", a.t2, tau23)
            + einsum("dcij,abdc->abij", a.t2, h.v.vvvv)
            + einsum("ikac,kjbc->abij", tau24, tau25)
            + 2 * einsum("kc,cbakji->abij", tau8, a.t3)
            + einsum("ackj,ikbc->abij", a.t2, tau26)
            + einsum("cj,ibac->abij", a.t1, tau27)
            + einsum("cbkj,ikac->abij", a.t2, tau26)
            + einsum("baji->abij", h.v.vvoo)
            + einsum("ijba->abij", tau54)
            + einsum("jiab->abij", tau54)
            - einsum("ijab->abij", tau84)
            - einsum("jiba->abij", tau84)
        )

        r3 = (
            - einsum("ijkabc->abcijk", tau85)
            - einsum("ikjacb->abcijk", tau85)
            + einsum("jikbca->abcijk", tau85)
            - einsum("jkibca->abcijk", tau85)
            + einsum("kijacb->abcijk", tau85)
            + einsum("kjiabc->abcijk", tau85)
            + einsum("ijkbac->abcijk", tau92)
            - einsum("ijkbca->abcijk", tau92)
            + einsum("ikjabc->abcijk", tau92)
            - einsum("ikjcba->abcijk", tau92)
            - einsum("jikacb->abcijk", tau92)
            + einsum("jikcab->abcijk", tau92)
            + einsum("jkiacb->abcijk", tau92)
            - einsum("jkicab->abcijk", tau92)
            - einsum("kijabc->abcijk", tau92)
            + einsum("kijcba->abcijk", tau92)
            - einsum("kjibac->abcijk", tau92)
            + einsum("kjibca->abcijk", tau92)
            + einsum("ijkbac->abcijk", tau103)
            - einsum("ijkbca->abcijk", tau103)
            + einsum("jikcab->abcijk", tau103)
            - einsum("jikacb->abcijk", tau103)
            - einsum("kijabc->abcijk", tau103)
            + einsum("kijcba->abcijk", tau103)
            + einsum("ijkabc->abcijk", tau119)
            - einsum("ijkcba->abcijk", tau119)
            + einsum("ikjacb->abcijk", tau119)
            - einsum("ikjcab->abcijk", tau119)
            + einsum("jikbac->abcijk", tau119)
            - einsum("jikbca->abcijk", tau119)
            - einsum("jkibac->abcijk", tau119)
            + einsum("jkibca->abcijk", tau119)
            - einsum("kijacb->abcijk", tau119)
            + einsum("kijcab->abcijk", tau119)
            - einsum("kjiabc->abcijk", tau119)
            + einsum("kjicba->abcijk", tau119)
            + einsum("ijkcab->abcijk", tau121)
            + einsum("ikjcba->abcijk", tau121)
            + einsum("jikbac->abcijk", tau121)
            - einsum("jkibac->abcijk", tau121)
            - einsum("kijcba->abcijk", tau121)
            - einsum("kjicab->abcijk", tau121)
            + einsum("ijkacb->abcijk", tau146)
            - einsum("ijkcab->abcijk", tau146)
            + einsum("ikjabc->abcijk", tau146)
            - einsum("ikjcba->abcijk", tau146)
            - einsum("jikbac->abcijk", tau146)
            + einsum("jikbca->abcijk", tau146)
            + einsum("jkibac->abcijk", tau146)
            - einsum("jkibca->abcijk", tau146)
            - einsum("kijabc->abcijk", tau146)
            + einsum("kijcba->abcijk", tau146)
            - einsum("kjiacb->abcijk", tau146)
            + einsum("kjicab->abcijk", tau146)
            + einsum("ijkacb->abcijk", tau154)
            - einsum("ijkcab->abcijk", tau154)
            - einsum("ikjbac->abcijk", tau154)
            + einsum("ikjbca->abcijk", tau154)
            + einsum("jikabc->abcijk", tau154)
            - einsum("jikcba->abcijk", tau154)
            - einsum("jkiabc->abcijk", tau154)
            + einsum("jkicba->abcijk", tau154)
            + einsum("kijbac->abcijk", tau154)
            - einsum("kijbca->abcijk", tau154)
            - einsum("kjiacb->abcijk", tau154)
            + einsum("kjicab->abcijk", tau154)
            - einsum("jikabc->abcijk", tau163)
            + einsum("jikcba->abcijk", tau163)
            - einsum("kijbac->abcijk", tau163)
            + einsum("kijbca->abcijk", tau163)
            + einsum("kjiacb->abcijk", tau163)
            - einsum("kjicab->abcijk", tau163)
            + einsum("ijkbac->abcijk", tau179)
            - einsum("ijkbca->abcijk", tau179)
            - einsum("ikjacb->abcijk", tau179)
            + einsum("ikjcab->abcijk", tau179)
            + einsum("jikabc->abcijk", tau179)
            - einsum("jikcba->abcijk", tau179)
            - einsum("jkiabc->abcijk", tau179)
            + einsum("jkicba->abcijk", tau179)
            + einsum("kijacb->abcijk", tau179)
            - einsum("kijcab->abcijk", tau179)
            - einsum("kjibac->abcijk", tau179)
            + einsum("kjibca->abcijk", tau179)
            + einsum("ijkabc->abcijk", tau181)
            + einsum("ikjacb->abcijk", tau181)
            - einsum("ikjcab->abcijk", tau181)
            + einsum("jikbac->abcijk", tau181)
            - einsum("jkibac->abcijk", tau181)
            - einsum("kijacb->abcijk", tau181)
            + einsum("kijcab->abcijk", tau181)
            - einsum("kjiabc->abcijk", tau181)
            + einsum("jikbac->abcijk", tau187)
            - einsum("jikbca->abcijk", tau187)
            - einsum("kijacb->abcijk", tau187)
            + einsum("kijcab->abcijk", tau187)
            - einsum("kjiabc->abcijk", tau187)
            + einsum("kjicba->abcijk", tau187)
            - einsum("ikjcba->abcijk", tau188)
            + einsum("jikbca->abcijk", tau188)
            - einsum("jkibca->abcijk", tau188)
            + einsum("kijcba->abcijk", tau188)
            - einsum("ijkacb->abcijk", tau189)
            - einsum("ikjabc->abcijk", tau189)
            - einsum("jikbca->abcijk", tau189)
            + einsum("jkibca->abcijk", tau189)
            + einsum("kijabc->abcijk", tau189)
            + einsum("kjiacb->abcijk", tau189)
            + einsum("ijkacb->abcijk", tau190)
            - einsum("ijkcab->abcijk", tau190)
            + einsum("ikjabc->abcijk", tau190)
            - einsum("jikbac->abcijk", tau190)
            + einsum("jkibac->abcijk", tau190)
            - einsum("kijabc->abcijk", tau190)
            - einsum("kjiacb->abcijk", tau190)
            + einsum("kjicab->abcijk", tau190)
            + einsum("ijkabc->abcijk", tau197)
            - einsum("ijkcba->abcijk", tau197)
            - einsum("jikbca->abcijk", tau197)
            + einsum("jikbac->abcijk", tau197)
            - einsum("kijacb->abcijk", tau197)
            + einsum("kijcab->abcijk", tau197)
            - einsum("ijkabc->abcijk", tau210)
            + einsum("ijkcba->abcijk", tau210)
            - einsum("ikjbac->abcijk", tau210)
            + einsum("ikjbca->abcijk", tau210)
            - einsum("jikacb->abcijk", tau210)
            + einsum("jikcab->abcijk", tau210)
            + einsum("jkiacb->abcijk", tau210)
            - einsum("jkicab->abcijk", tau210)
            + einsum("kijbac->abcijk", tau210)
            - einsum("kijbca->abcijk", tau210)
            + einsum("kjiabc->abcijk", tau210)
            - einsum("kjicba->abcijk", tau210)
            + einsum("ijkcba->abcijk", tau211)
            + einsum("ikjcab->abcijk", tau211)
            - einsum("jikbac->abcijk", tau211)
            + einsum("jkibac->abcijk", tau211)
            - einsum("kijcab->abcijk", tau211)
            - einsum("kjicba->abcijk", tau211)
            - einsum("ijkcba->abcijk", tau212)
            - einsum("jikbca->abcijk", tau212)
            + einsum("jkibca->abcijk", tau212)
            + einsum("kjicba->abcijk", tau212)
        )

        return self.types.RESIDUALS_TYPE(r1=r1, r2=r2, r3=r3)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return self.RHS_TYPE(
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

        return self.types.AMPLITUDES_TYPE(
            *(g[ii] * (- cc_denom(h.f, g[ii].ndim, 'dir', 'full'))
              for ii in range(len(g)))
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

    from tcc.cc_solvers import root_solver
    from tcc.rccsdt import RCCSDT
    cc = RCCSDT(rhf)
    converged, energy, _ = root_solver(cc)

if __name__ == '__main__':
    test_cc()
