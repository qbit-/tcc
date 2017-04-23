import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from tcc.rccsd import RCCSD
from collections import namedtuple


class RCCSDT(RCCSD):
    """
    This class implements classic RCCSDT method with
    vvvooo ordered amplitudes and Dirac ordered integrals
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays
    AMPLITUDES_TYPE = namedtuple('RCCSD_AMPLITUDES_FULL',
                                 field_names=('t1', 't2', 't3'))
    RHS_TYPE = namedtuple('RCCSD_RHS_FULL',
                          field_names=('g1', 'g2', 'g3'))
    RESIDUALS_TYPE = namedtuple('RCCSD_RESIDUALS_FULL',
                                field_names=('r1', 'r2', 'r3'))

    @property
    def method_name(self):
        return 'RCCSDT'

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

        return self.AMPLITUDES_TYPE(t1, t2, t3)

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
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau6 = (
            - einsum("iabc->iabc", h.v.ovvv)
            + 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau7 = (
            einsum("ci,iabc->ab", a.t1, tau6)
            + einsum("ab->ab", h.f.vv)
        )

        tau8 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau9 = (
            - 2 * einsum("ijka->ijka", tau8)
            + einsum("ikja->ijka", tau8)
            + einsum("jkia->ijka", h.v.ooov)
            - 2 * einsum("kjia->ijka", h.v.ooov)
        )

        tau10 = (
            2 * einsum("ijka->ijka", h.v.ooov)
            - einsum("jika->ijka", h.v.ooov)
        )

        tau11 = (
            einsum("aj,ia->ij", a.t1, tau3)
            + einsum("ak,ikja->ij", a.t1, tau10)
            + einsum("bakj,kiba->ij", a.t2, tau0)
            + einsum("ij->ij", h.f.oo)
        )

        tau12 = (
            - einsum("iajb->ijab", h.v.ovov)
            + 2 * einsum("bjia->ijab", h.v.voov.conj())
        )

        tau13 = (
            - 2 * einsum("jiab->ijab", h.v.oovv)
            + einsum("jiba->ijab", h.v.oovv)
        )

        tau14 = (
            einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau15 = (
            einsum("ai,jkla->ijkl", a.t1, tau8)
        )

        tau16 = (
            einsum("lkji->ijkl", tau14)
            + einsum("lkji->ijkl", tau15)
            + einsum("jilk->ijkl", h.v.oooo)
        )

        tau17 = (
            einsum("bcakij->ijkabc", a.t3)
            + einsum("cabkij->ijkabc", a.t3)
        )

        tau18 = (
            einsum("acki,kjbc->ijab", a.t2, tau13)
        )

        tau19 = (
            2 * einsum("caki,kjbc->ijab", a.t2, tau5)
        )

        tau20 = (
            einsum("ijab->ijab", tau18)
            + einsum("ijab->ijab", tau19)
        )

        tau21 = (
            einsum("caji,jibc->ab", a.t2, tau13)
        )

        tau22 = (
            einsum("al,lkji->ijka", a.t1, tau16)
        )

        tau23 = (
            einsum("di,badc->iabc", a.t1, h.v.vvvv)
        )

        tau24 = (
            einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau25 = (
            einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau26 = (
            - 2 * einsum("abji->ijab", a.t2)
            + einsum("baji->ijab", a.t2)
        )

        tau27 = (
            einsum("bj,jiba->ia", a.t1, tau0)
        )

        tau28 = (
            einsum("kc,backij->ijab", tau27, a.t3)
        )

        tau29 = (
            einsum("iabc->iabc", h.v.ovvv)
            - 2 * einsum("iacb->iabc", h.v.ovvv)
        )

        tau30 = (
            einsum("ci,iabc->ab", a.t1, tau29)
        )

        tau31 = (
            einsum("bc,caji->ijab", tau30, a.t2)
        )

        tau32 = (
            2 * einsum("ijka->ijka", tau8)
            - einsum("ikja->ijka", tau8)
        )

        tau33 = (
            einsum("balk,iljb->ijka", a.t2, tau32)
        )

        tau34 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau35 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau36 = (
            einsum("bi,jkab->ijka", a.t1, tau35)
        )

        tau37 = (
            einsum("bi,jkab->ijka", a.t1, tau25)
        )

        tau38 = (
            einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau39 = (
            einsum("bi,bkja->ijka", a.t1, h.v.voov.conj())
        )

        tau40 = (
            einsum("lkcb,abclij->ijka", h.v.oovv, a.t3)
        )

        tau41 = (
            einsum("ijka->ijka", tau33)
            - einsum("ijka->ijka", tau34)
            + einsum("ikja->ijka", tau36)
            - einsum("ikja->ijka", tau37)
            + einsum("ikja->ijka", tau38)
            + einsum("ijka->ijka", tau39)
            + einsum("jaik->ijka", h.v.ovoo)
            - einsum("ikja->ijka", tau40)
        )

        tau42 = (
            einsum("ak,ikjb->ijab", a.t1, tau41)
        )

        tau43 = (
            einsum("kbcd,acdkij->ijab", h.v.ovvv, a.t3)
        )

        tau44 = (
            einsum("acki,cjkb->ijab", a.t2, h.v.voov.conj())
        )

        tau45 = (
            einsum("ak,ikja->ij", a.t1, tau10)
        )

        tau46 = (
            einsum("bakj,kiba->ij", a.t2, tau0)
        )

        tau47 = (
            einsum("ij->ij", tau45)
            + einsum("ij->ij", tau46)
        )

        tau48 = (
            einsum("kj,baki->ijab", tau47, a.t2)
        )

        tau49 = (
            einsum("kljc,cablik->ijab", tau10, a.t3)
        )

        tau50 = (
            einsum("kjdc,bdakij->iabc", h.v.oovv, a.t3)
        )

        tau51 = (
            einsum("ci,jabc->ijab", a.t1, tau50)
        )

        tau52 = (
            einsum("iajb->ijab", h.v.ovov)
            - 2 * einsum("bjia->ijab", h.v.voov.conj())
        )

        tau53 = (
            einsum("caki,kjbc->ijab", a.t2, tau52)
        )

        tau54 = (
            einsum("kbcd,dackij->ijab", tau29, a.t3)
        )

        tau55 = (
            einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau56 = (
            einsum("ci,jabc->ijab", a.t1, tau55)
        )

        tau57 = (
            einsum("ijab->ijab", tau28)
            + einsum("ijab->ijab", tau31)
            + einsum("ijab->ijab", tau42)
            + einsum("ijab->ijab", tau43)
            + einsum("ijab->ijab", tau44)
            + einsum("ijab->ijab", tau48)
            + einsum("ijab->ijab", tau49)
            - einsum("ijab->ijab", tau51)
            + einsum("ijab->ijab", tau53)
            + einsum("ijab->ijab", tau54)
            + einsum("ijab->ijab", tau56)
        )

        tau58 = (
            einsum("kljc,bcalik->ijab", h.v.ooov, a.t3)
        )

        tau59 = (
            einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau60 = (
            einsum("ijka->ijka", h.v.ooov)
            - 2 * einsum("jika->ijka", h.v.ooov)
        )

        tau61 = (
            einsum("balj,likb->ijka", a.t2, tau60)
        )

        tau62 = (
            einsum("bi,jkab->ijka", a.t1, tau24)
        )

        tau63 = (
            einsum("libc,cabljk->ijka", tau13, a.t3)
        )

        tau64 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau65 = (
            einsum("al,ijlk->ijka", a.t1, tau64)
        )

        tau66 = (
            - einsum("ib,bakj->ijka", tau3, a.t2)
        )

        tau67 = (
            einsum("bi,jakb->ijka", a.t1, h.v.ovov)
        )

        tau68 = (
            einsum("jika->ijka", tau59)
            + einsum("ijka->ijka", tau61)
            + einsum("jkia->ijka", tau62)
            + einsum("ijka->ijka", tau63)
            + einsum("jika->ijka", tau65)
            + einsum("ijka->ijka", tau66)
            - einsum("jika->ijka", tau67)
        )

        tau69 = (
            einsum("ak,kijb->ijab", a.t1, tau68)
        )

        tau70 = (
            einsum("ijka->ijka", tau8)
            - 2 * einsum("ikja->ijka", tau8)
        )

        tau71 = (
            einsum("iklc,cabljk->ijab", tau70, a.t3)
        )

        tau72 = (
            einsum("ac,cbji->ijab", h.f.vv, a.t2)
        )

        tau73 = (
            einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
        )

        tau74 = (
            einsum("ci,jabc->ijab", a.t1, tau73)
        )

        tau75 = (
            einsum("ai,ja->ij", a.t1, tau3)
        )

        tau76 = (
            einsum("ij->ij", tau75)
            + einsum("ji->ij", h.f.oo)
        )

        tau77 = (
            - einsum("ik,bakj->ijab", tau76, a.t2)
        )

        tau78 = (
            einsum("adji,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau79 = (
            einsum("ci,jabc->ijab", a.t1, tau78)
        )

        tau80 = (
            einsum("ci,abjc->ijab", a.t1, h.v.vvov)
        )

        tau81 = (
            2 * einsum("iabc->iabc", h.v.ovvv)
            - einsum("iacb->iabc", h.v.ovvv)
        )

        tau82 = (
            einsum("ci,jabc->ijab", a.t1, tau81)
        )

        tau83 = (
            einsum("cakj,ikbc->ijab", a.t2, tau82)
        )

        tau84 = (
            einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau85 = (
            einsum("ijab->ijab", tau58)
            + einsum("ijab->ijab", tau69)
            + einsum("ijab->ijab", tau71)
            + einsum("ijab->ijab", tau72)
            + einsum("ijab->ijab", tau74)
            + einsum("ijab->ijab", tau77)
            - einsum("ijab->ijab", tau79)
            + einsum("ijab->ijab", tau80)
            + einsum("ijab->ijab", tau83)
            - einsum("ijab->ijab", tau84)
        )

        tau86 = (
            einsum("lkcd,badlij->ijkabc", h.v.oovv, a.t3)
        )

        tau87 = (
            einsum("adli,jklbcd->ijkabc", a.t2, tau86)
        )

        tau88 = (
            einsum("daji,kbdc->ijkabc", a.t2, h.v.ovvv)
        )

        tau89 = (
            einsum("ci,jklabc->ijklab", a.t1, tau88)
        )

        tau90 = (
            einsum("lkcd,bdalij->ijkabc", h.v.oovv, a.t3)
        )

        tau91 = (
            einsum("ci,kjlabc->ijklab", a.t1, tau90)
        )

        tau92 = (
            einsum("baji,klmb->ijklma", a.t2, h.v.ooov)
        )

        tau93 = (
            einsum("am,ijkmlb->ijklab", a.t1, tau92)
        )

        tau94 = (
            einsum("bamj,ikml->ijklab", a.t2, tau64)
        )

        tau95 = (
            - einsum("ijlkab->ijklab", tau89)
            + einsum("ijlkab->ijklab", tau91)
            + einsum("ijklab->ijklab", tau93)
            + einsum("ijklab->ijklab", tau94)
        )

        tau96 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau95)
        )

        tau97 = (
            einsum("dcji,kabd->ijkabc", a.t2, tau73)
        )

        tau98 = (
            einsum("ijkabc->ijkabc", tau87)
            + einsum("ijkabc->ijkabc", tau96)
            + einsum("ijkabc->ijkabc", tau97)
        )

        tau99 = (
            einsum("jilk->ijkl", tau14)
            + einsum("ijkl->ijkl", tau15)
        )

        tau100 = (
            einsum("al,jilk->ijka", a.t1, tau99)
        )

        tau101 = (
            - einsum("kb,baij->ijka", tau3, a.t2)
        )

        tau102 = (
            einsum("ijka->ijka", tau100)
            + einsum("ijka->ijka", tau101)
        )

        tau103 = (
            einsum("cblk,jila->ijkabc", a.t2, tau102)
        )

        tau104 = (
            einsum("daji,jbdc->iabc", a.t2, tau81)
        )

        tau105 = (
            einsum("jiab->ijab", h.v.oovv)
            - 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau106 = (
            einsum("kjdc,dabkij->iabc", tau105, a.t3)
        )

        tau107 = (
            einsum("iabc->iabc", tau104)
            + einsum("iabc->iabc", tau106)
            + einsum("abic->iabc", h.v.vvov)
        )

        tau108 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau107)
        )

        tau109 = (
            einsum("lkcb,cablij->ijka", tau105, a.t3)
        )

        tau110 = (
            - 2 * einsum("ijka->ijka", h.v.ooov)
            + einsum("jika->ijka", h.v.ooov)
        )

        tau111 = (
            einsum("bali,kljb->ijka", a.t2, tau110)
        )

        tau112 = (
            einsum("ijka->ijka", tau109)
            + einsum("ijka->ijka", tau111)
        )

        tau113 = (
            einsum("bali,jklc->ijkabc", a.t2, tau112)
        )

        tau114 = (
            einsum("caji,lkcb->ijklab", a.t2, h.v.oovv)
        )

        tau115 = (
            einsum("bi,jklmab->ijklma", a.t1, tau114)
        )

        tau116 = (
            einsum("am,ijklmb->ijklab", a.t1, tau115)
        )

        tau117 = (
            einsum("al,ijklbc->ijkabc", a.t1, tau116)
        )

        tau118 = (
            einsum("lckd,badlij->ijkabc", h.v.ovov, a.t3)
        )

        tau119 = (
            einsum("baji,jidc->abcd", a.t2, h.v.oovv)
        )

        tau120 = (
            einsum("ecji,baed->ijabcd", a.t2, tau119)
        )

        tau121 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau120)
        )

        tau122 = (
            einsum("ijkabc->ijkabc", tau103)
            + einsum("ijkabc->ijkabc", tau108)
            + einsum("ijkabc->ijkabc", tau113)
            + einsum("ijkabc->ijkabc", tau117)
            - einsum("ijkabc->ijkabc", tau118)
            + einsum("ijkabc->ijkabc", tau121)
        )

        tau123 = (
            einsum("caki,kjbc->ijab", a.t2, tau13)
        )

        tau124 = (
            einsum("ijab->ijab", tau25)
            + einsum("ijab->ijab", tau123)
        )

        tau125 = (
            einsum("ilad,cbdljk->ijkabc", tau124, a.t3)
        )

        tau126 = (
            einsum("lkdc,bdalij->ijkabc", h.v.oovv, a.t3)
        )

        tau127 = (
            einsum("ci,kjlabc->ijklab", a.t1, tau126)
        )

        tau128 = (
            einsum("daji,kbcd->ijkabc", a.t2, h.v.ovvv)
        )

        tau129 = (
            einsum("ci,jklabc->ijklab", a.t1, tau128)
        )

        tau130 = (
            einsum("bamj,imkl->ijklab", a.t2, tau64)
        )

        tau131 = (
            einsum("ijlkab->ijklab", tau127)
            - einsum("ijlkab->ijklab", tau129)
            + einsum("ijklab->ijklab", tau130)
        )

        tau132 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau131)
        )

        tau133 = (
            einsum("dklc,badlij->ijkabc", h.v.voov.conj(), a.t3)
        )

        tau134 = (
            - einsum("kaij->ijka", h.v.ovoo)
            + einsum("ijka->ijka", tau40)
        )

        tau135 = (
            einsum("bali,jklc->ijkabc", a.t2, tau134)
        )

        tau136 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau50)
        )

        tau137 = (
            einsum("ijkabc->ijkabc", tau125)
            + einsum("ijkabc->ijkabc", tau132)
            - einsum("ijkabc->ijkabc", tau133)
            + einsum("ijkabc->ijkabc", tau135)
            + einsum("ijkabc->ijkabc", tau136)
        )

        tau138 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau78)
        )

        tau139 = (
            einsum("bami,mjlk->ijklab", a.t2, h.v.oooo)
        )

        tau140 = (
            einsum("caki,kjbc->ijab", a.t2, tau13)
        )

        tau141 = (
            einsum("ijab->ijab", tau140)
            + einsum("ijab->ijab", tau25)
            - einsum("bija->ijab", h.v.voov.conj())
        )

        tau142 = (
            einsum("caji,lkbc->ijklab", a.t2, tau141)
        )

        tau143 = (
            einsum("mklc,bacmij->ijklab", h.v.ooov, a.t3)
        )

        tau144 = (
            einsum("lbcd,dackij->ijklab", h.v.ovvv, a.t3)
        )

        tau145 = (
            einsum("ikjlab->ijklab", tau139)
            + einsum("ijklab->ijklab", tau142)
            + einsum("ijklab->ijklab", tau143)
            - einsum("ijlkab->ijklab", tau144)
        )

        tau146 = (
            einsum("al,ijlkbc->ijkabc", a.t1, tau145)
        )

        tau147 = (
            einsum("balj,iklb->ijka", a.t2, tau70)
        )

        tau148 = (
            einsum("ijka->ijka", tau147)
            - einsum("ikja->ijka", tau39)
            + einsum("ijka->ijka", tau37)
        )

        tau149 = (
            einsum("balj,iklc->ijkabc", a.t2, tau148)
        )

        tau150 = (
            einsum("kcde,beakij->ijabcd", h.v.ovvv, a.t3)
        )

        tau151 = (
            einsum("di,kjabcd->ijkabc", a.t1, tau150)
        )

        tau152 = (
            einsum("eaji,cbed->ijabcd", a.t2, h.v.vvvv)
        )

        tau153 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau152)
        )

        tau154 = (
            einsum("ijkabc->ijkabc", tau146)
            + einsum("ijkabc->ijkabc", tau149)
            - einsum("ijkabc->ijkabc", tau151)
            + einsum("ijkabc->ijkabc", tau153)
        )

        tau155 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau59)
        )

        tau156 = (
            - 2 * einsum("iabc->iabc", h.v.ovvv)
            + einsum("iacb->iabc", h.v.ovvv)
        )

        tau157 = (
            einsum("ci,jabc->ijab", a.t1, tau156)
        )

        tau158 = (
            einsum("ilcd,dbalkj->ijkabc", tau157, a.t3)
        )

        tau159 = (
            einsum("aj,ia->ij", a.t1, tau3)
        )

        tau160 = (
            einsum("bakj,kiba->ij", a.t2, tau0)
        )

        tau161 = (
            einsum("ij->ij", tau159)
            + einsum("ij->ij", tau160)
            + einsum("ij->ij", h.f.oo)
        )

        tau162 = (
            einsum("li,cbalkj->ijkabc", tau161, a.t3)
        )

        tau163 = (
            einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
        )

        tau164 = (
            einsum("ai,bj->ijab", a.t1, a.t1)
            + einsum("baji->ijab", a.t2)
        )

        tau165 = (
            - einsum("mlba,ijklmc->ijkabc", tau164, tau163)
        )

        tau166 = (
            einsum("ijkabc->ijkabc", tau158)
            + einsum("ijkabc->ijkabc", tau162)
            + einsum("ijkabc->ijkabc", tau165)
        )

        tau167 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau55)
        )

        tau168 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau59)
        )

        tau169 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau55)
        )

        tau170 = (
            - einsum("jaib->ijab", h.v.ovov)
            + einsum("ijab->ijab", tau24)
        )

        tau171 = (
            einsum("caji,lkbc->ijklab", a.t2, tau170)
        )

        tau172 = (
            einsum("kmlc,bacmij->ijklab", h.v.ooov, a.t3)
        )

        tau173 = (
            einsum("ijklab->ijklab", tau171)
            + einsum("ijklab->ijklab", tau172)
        )

        tau174 = (
            - einsum("al,ijlkbc->ijkabc", a.t1, tau173)
        )

        tau175 = (
            einsum("ijka->ijka", tau36)
            + einsum("ijka->ijka", tau38)
        )

        tau176 = (
            einsum("balk,ijlc->ijkabc", a.t2, tau175)
        )

        tau177 = (
            einsum("ijkabc->ijkabc", tau174)
            + einsum("ijkabc->ijkabc", tau176)
        )

        tau178 = (
            einsum("ic,cbalkj->ijklab", tau3, a.t3)
        )

        tau179 = (
            einsum("imlc,cbamkj->ijklab", tau10, a.t3)
        )

        tau180 = (
            einsum("ijklab->ijklab", tau178)
            + einsum("ijklab->ijklab", tau179)
        )

        tau181 = (
            - einsum("al,lijkbc->ijkabc", a.t1, tau180)
        )

        tau182 = (
            einsum("jiml,cabmkl->ijkabc", tau99, a.t3)
        )

        tau183 = (
            einsum("caji,jibc->ab", a.t2, tau13)
        )

        tau184 = (
            einsum("ab->ab", tau183)
            + einsum("ab->ab", h.f.vv)
        )

        tau185 = (
            einsum("ad,dbckij->ijkabc", tau184, a.t3)
        )

        tau186 = (
            einsum("ijkabc->ijkabc", tau181)
            + einsum("ijkabc->ijkabc", tau182)
            + einsum("ijkabc->ijkabc", tau185)
        )

        tau187 = (
            einsum("daji,kbcd->ijkabc", a.t2, tau78)
        )

        tau188 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau189 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau188)
        )

        tau190 = (
            einsum("lkcd,dablij->ijkabc", tau12, a.t3)
        )

        tau191 = (
            einsum("ak,ikja->ij", a.t1, tau110)
        )

        tau192 = (
            einsum("lk,cablij->ijkabc", tau191, a.t3)
        )

        tau193 = (
            einsum("ci,iabc->ab", a.t1, tau6)
        )

        tau194 = (
            einsum("cd,dabkij->ijkabc", tau193, a.t3)
        )

        tau195 = (
            einsum("ijkabc->ijkabc", tau190)
            + einsum("ijkabc->ijkabc", tau192)
            + einsum("ijkabc->ijkabc", tau194)
        )

        tau196 = (
            einsum("bali,jlkc->ijkabc", a.t2, tau188)
        )

        tau197 = (
            einsum("ilad,dcblkj->ijkabc", tau20, a.t3)
        )

        tau198 = (
            - 2 * einsum("ijka->ijka", tau8)
            + einsum("ikja->ijka", tau8)
        )

        tau199 = (
            einsum("imlc,cbamkj->ijklab", tau198, a.t3)
        )

        tau200 = (
            einsum("al,ikjlcb->ijkabc", a.t1, tau199)
        )

        tau201 = (
            einsum("mlkj,cabmil->ijkabc", h.v.oooo, a.t3)
        )

        tau202 = (
            einsum("cbed,eadkij->ijkabc", h.v.vvvv, a.t3)
        )

        tau203 = (
            einsum("ijkabc->ijkabc", tau197)
            + einsum("ijkabc->ijkabc", tau200)
            + einsum("ikjacb->ijkabc", tau201)
            + einsum("ikjacb->ijkabc", tau202)
        )

        tau204 = (
            - einsum("ijka->ijka", tau62)
            + einsum("ikja->ijka", tau67)
        )

        tau205 = (
            einsum("balj,iklc->ijkabc", a.t2, tau204)
        )

        tau206 = (
            einsum("kced,beakij->ijabcd", h.v.ovvv, a.t3)
        )

        tau207 = (
            einsum("di,kjabcd->ijkabc", a.t1, tau206)
        )

        tau208 = (
            einsum("kljd,cablik->ijabcd", h.v.ooov, a.t3)
        )

        tau209 = (
            einsum("di,jkabcd->ijkabc", a.t1, tau208)
        )

        tau210 = (
            einsum("ijkabc->ijkabc", tau205)
            + einsum("ijkabc->ijkabc", tau207)
            - einsum("ijkabc->ijkabc", tau209)
        )

        g1 = (
            einsum("jb,jiba->ai", tau3, tau4)
            + einsum("ia->ai", h.f.ov.conj())
            + einsum("kjbc,bacjik->ai", tau5, a.t3)
            + einsum("bi,ab->ai", a.t1, tau7)
            + einsum("bakj,ikjb->ai", a.t2, tau9)
            - einsum("aj,ji->ai", a.t1, tau11)
            + einsum("bj,jiab->ai", a.t1, tau12)
            + einsum("kjbc,cbakij->ai", tau13, a.t3)
            + einsum("jibc,jabc->ai", tau4, h.v.ovvv)
        )

        g2 = (
            einsum("abkl,lkji->abij", a.t2, tau16)
            - einsum("kc,kijacb->abij", h.f.ov, tau17)
            + einsum("caki,jkbc->abij", a.t2, tau20)
            + einsum("ac,cbij->abij", tau21, a.t2)
            + einsum("baji->abij", h.v.vvoo)
            + 2 * einsum("kc,cbakji->abij", tau3, a.t3)
            + einsum("bc,caji->abij", tau21, a.t2)
            + einsum("ak,ijkb->abij", a.t1, tau22)
            + einsum("dcij,abdc->abij", a.t2, h.v.vvvv)
            + einsum("cj,ibac->abij", a.t1, tau23)
            + einsum("cbkj,ikac->abij", a.t2, tau24)
            + einsum("ikac,kjbc->abij", tau25, tau26)
            + einsum("ackj,ikbc->abij", a.t2, tau24)
            - einsum("ijab->abij", tau57)
            - einsum("jiba->abij", tau57)
            + einsum("ijba->abij", tau85)
            + einsum("jiab->abij", tau85)
        )

        g3 = (
            - einsum("ijkabc->abcijk", tau98)
            + einsum("ijkcba->abcijk", tau98)
            - einsum("ikjbac->abcijk", tau98)
            + einsum("ikjbca->abcijk", tau98)
            - einsum("jikacb->abcijk", tau98)
            + einsum("jikcab->abcijk", tau98)
            + einsum("jkiacb->abcijk", tau98)
            - einsum("jkicab->abcijk", tau98)
            + einsum("kijbac->abcijk", tau98)
            - einsum("kijbca->abcijk", tau98)
            + einsum("kjiabc->abcijk", tau98)
            - einsum("kjicba->abcijk", tau98)
            + einsum("ijkacb->abcijk", tau122)
            - einsum("ijkcab->abcijk", tau122)
            + einsum("ikjabc->abcijk", tau122)
            - einsum("ikjcba->abcijk", tau122)
            - einsum("jikbac->abcijk", tau122)
            + einsum("jikbca->abcijk", tau122)
            + einsum("jkibac->abcijk", tau122)
            - einsum("jkibca->abcijk", tau122)
            - einsum("kijabc->abcijk", tau122)
            + einsum("kijcba->abcijk", tau122)
            - einsum("kjiacb->abcijk", tau122)
            + einsum("kjicab->abcijk", tau122)
            + einsum("ijkabc->abcijk", tau137)
            - einsum("ijkcba->abcijk", tau137)
            + einsum("ikjacb->abcijk", tau137)
            - einsum("ikjcab->abcijk", tau137)
            + einsum("jikbac->abcijk", tau137)
            - einsum("jikbca->abcijk", tau137)
            - einsum("jkibac->abcijk", tau137)
            + einsum("jkibca->abcijk", tau137)
            - einsum("kijacb->abcijk", tau137)
            + einsum("kijcab->abcijk", tau137)
            - einsum("kjiabc->abcijk", tau137)
            + einsum("kjicba->abcijk", tau137)
            + einsum("ijkcab->abcijk", tau138)
            + einsum("ikjcba->abcijk", tau138)
            + einsum("jikbac->abcijk", tau138)
            - einsum("jkibac->abcijk", tau138)
            - einsum("kijcba->abcijk", tau138)
            - einsum("kjicab->abcijk", tau138)
            + einsum("ijkbac->abcijk", tau154)
            - einsum("ijkbca->abcijk", tau154)
            - einsum("ikjacb->abcijk", tau154)
            + einsum("ikjcab->abcijk", tau154)
            + einsum("jikabc->abcijk", tau154)
            - einsum("jikcba->abcijk", tau154)
            - einsum("jkiabc->abcijk", tau154)
            + einsum("jkicba->abcijk", tau154)
            + einsum("kijacb->abcijk", tau154)
            - einsum("kijcab->abcijk", tau154)
            - einsum("kjibac->abcijk", tau154)
            + einsum("kjibca->abcijk", tau154)
            + einsum("ijkacb->abcijk", tau155)
            - einsum("ijkcab->abcijk", tau155)
            + einsum("ikjabc->abcijk", tau155)
            - einsum("jikbac->abcijk", tau155)
            + einsum("jkibac->abcijk", tau155)
            - einsum("kijabc->abcijk", tau155)
            - einsum("kjiacb->abcijk", tau155)
            + einsum("kjicab->abcijk", tau155)
            + einsum("ijkbac->abcijk", tau166)
            - einsum("ijkbca->abcijk", tau166)
            + einsum("jikcab->abcijk", tau166)
            - einsum("jikacb->abcijk", tau166)
            - einsum("kijabc->abcijk", tau166)
            + einsum("kijcba->abcijk", tau166)
            - einsum("ijkabc->abcijk", tau167)
            - einsum("ikjacb->abcijk", tau167)
            + einsum("jikbca->abcijk", tau167)
            - einsum("jkibca->abcijk", tau167)
            + einsum("kijacb->abcijk", tau167)
            + einsum("kjiabc->abcijk", tau167)
            - einsum("ikjcba->abcijk", tau168)
            + einsum("jikbca->abcijk", tau168)
            - einsum("jkibca->abcijk", tau168)
            + einsum("kijcba->abcijk", tau168)
            + einsum("ijkcba->abcijk", tau169)
            + einsum("ikjcab->abcijk", tau169)
            - einsum("jikbac->abcijk", tau169)
            + einsum("jkibac->abcijk", tau169)
            - einsum("kijcab->abcijk", tau169)
            - einsum("kjicba->abcijk", tau169)
            + einsum("ijkacb->abcijk", tau177)
            - einsum("ijkcab->abcijk", tau177)
            - einsum("ikjbac->abcijk", tau177)
            + einsum("ikjbca->abcijk", tau177)
            + einsum("jikabc->abcijk", tau177)
            - einsum("jikcba->abcijk", tau177)
            - einsum("jkiabc->abcijk", tau177)
            + einsum("jkicba->abcijk", tau177)
            + einsum("kijbac->abcijk", tau177)
            - einsum("kijbca->abcijk", tau177)
            - einsum("kjiacb->abcijk", tau177)
            + einsum("kjicab->abcijk", tau177)
            - einsum("jikabc->abcijk", tau186)
            + einsum("jikcba->abcijk", tau186)
            - einsum("kijbac->abcijk", tau186)
            + einsum("kijbca->abcijk", tau186)
            + einsum("kjiacb->abcijk", tau186)
            - einsum("kjicab->abcijk", tau186)
            - einsum("ijkacb->abcijk", tau187)
            - einsum("ikjabc->abcijk", tau187)
            - einsum("jikbca->abcijk", tau187)
            + einsum("jkibca->abcijk", tau187)
            + einsum("kijabc->abcijk", tau187)
            + einsum("kjiacb->abcijk", tau187)
            + einsum("ijkabc->abcijk", tau189)
            + einsum("ikjacb->abcijk", tau189)
            - einsum("ikjcab->abcijk", tau189)
            + einsum("jikbac->abcijk", tau189)
            - einsum("jkibac->abcijk", tau189)
            - einsum("kijacb->abcijk", tau189)
            + einsum("kijcab->abcijk", tau189)
            - einsum("kjiabc->abcijk", tau189)
            + einsum("jikbac->abcijk", tau195)
            - einsum("jikbca->abcijk", tau195)
            - einsum("kijacb->abcijk", tau195)
            + einsum("kijcab->abcijk", tau195)
            - einsum("kjiabc->abcijk", tau195)
            + einsum("kjicba->abcijk", tau195)
            - einsum("ijkcba->abcijk", tau196)
            - einsum("jikbca->abcijk", tau196)
            + einsum("jkibca->abcijk", tau196)
            + einsum("kjicba->abcijk", tau196)
            + einsum("ijkabc->abcijk", tau203)
            - einsum("ijkcba->abcijk", tau203)
            - einsum("jikbca->abcijk", tau203)
            + einsum("jikbac->abcijk", tau203)
            - einsum("kijacb->abcijk", tau203)
            + einsum("kijcab->abcijk", tau203)
            + einsum("ijkbac->abcijk", tau210)
            - einsum("ijkbca->abcijk", tau210)
            + einsum("ikjabc->abcijk", tau210)
            - einsum("ikjcba->abcijk", tau210)
            - einsum("jikacb->abcijk", tau210)
            + einsum("jikcab->abcijk", tau210)
            + einsum("jkiacb->abcijk", tau210)
            - einsum("jkicab->abcijk", tau210)
            - einsum("kijabc->abcijk", tau210)
            + einsum("kijcba->abcijk", tau210)
            - einsum("kjibac->abcijk", tau210)
            + einsum("kjibca->abcijk", tau210)
        )

        e_ai = cc_denom(h.f, 2, 'dir', 'full')
        e_abij = cc_denom(h.f, 4, 'dir', 'full')
        e_abcijk = cc_denom(h.f, 6, 'dir', 'full')

        g1 = g1 - a.t1 / e_ai
        g2 = g2 - a.t2 / e_abij
        g3 = g3 - a.t3 / e_abcijk

        return self.RHS_TYPE(g1=g1, g2=g2, g3=g3)


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

    from tcc.cc_solvers import residual_diis_solver
    from tcc.rccsdt import RCCSDT
    cc = RCCSDT(rhf)
    converged, energy, _ = residual_diis_solver(cc)

if __name__ == '__main__':
    test_cc()
