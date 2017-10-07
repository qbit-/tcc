from numpy import einsum
from tcc.tensors import Tensors


def _rccsdt_calculate_energy(h, a):
    tau0 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau1 = (
        2 * einsum("ia->ia", h.f.ov)
        + einsum("bj,jiba->ia", a.t1, tau0)
    )

    energy = (
        einsum("ai,ia->", a.t1, tau1)
        + einsum("abij,ijab->", a.t2, tau0)
    )
    return energy


def _rccsdt_calc_residuals(h, a):
    tau0 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau1 = (
        - einsum("backij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau2 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + 2 * einsum("iacb->iabc", h.v.ovvv)
    )

    tau3 = (
        einsum("bi,jkab->ijka", a.t1, tau0)
    )

    tau4 = (
        2 * einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        + einsum("ikja->ijka", tau3)
    )

    tau5 = (
        einsum("bj,jiba->ia", a.t1, tau0)
    )

    tau6 = (
        einsum("ia->ia", h.f.ov)
        + einsum("ia->ia", tau5)
    )

    tau7 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau8 = (
        2 * einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau9 = (
        2 * einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau10 = (
        einsum("ci,iacb->ab", a.t1, tau9)
    )

    tau11 = (
        einsum("ab->ab", h.f.vv)
        + einsum("ab->ab", tau10)
    )

    tau12 = (
        einsum("abki,kjab->ij", a.t2, tau0)
    )

    tau13 = (
        - einsum("ijka->ijka", h.v.ooov)
        + 2 * einsum("jika->ijka", h.v.ooov)
    )

    tau14 = (
        einsum("ak,kija->ij", a.t1, tau13)
    )

    tau15 = (
        einsum("ai,ja->ij", a.t1, tau6)
    )

    tau16 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau12)
        + einsum("ij->ij", tau14)
        + einsum("ji->ij", tau15)
    )

    tau17 = (
        einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
    )

    tau18 = (
        einsum("kbcd,acdkij->ijab", h.v.ovvv, a.t3)
    )

    tau19 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau20 = (
        2 * einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
        + einsum("bk,jiba->ijka", a.t1, tau19)
    )

    tau21 = (
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau22 = (
        einsum("ijka->ijka", h.v.ooov)
        + einsum("kjia->ijka", tau21)
    )

    tau23 = (
        2 * einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau24 = (
        einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
    )

    tau25 = (
        einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau26 = (
        einsum("kjcb,kica->ijab", tau7, h.v.oovv)
    )

    tau27 = (
        einsum("jabi->ijab", h.v.ovvo)
        + einsum("ijab->ijab", tau24)
        - einsum("ijab->ijab", tau25)
        + einsum("jiba->ijab", tau26)
    )

    tau28 = (
        einsum("acbkij->ijkabc", a.t3)
        - 2 * einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau29 = (
        einsum("ci,jabc->ijab", a.t1, tau9)
    )

    tau30 = (
        2 * einsum("jabi->ijab", h.v.ovvo)
        - einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau29)
    )

    tau31 = (
        einsum("ci,jacb->ijab", a.t1, h.v.ovvv)
    )

    tau32 = (
        einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau33 = (
        einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau31)
        - einsum("ijab->ijab", tau32)
    )

    tau34 = (
        einsum("jabi->ijab", h.v.ovvo)
        + einsum("ijab->ijab", tau24)
    )

    tau35 = (
        einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau31)
    )

    tau36 = (
        einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
    )

    tau37 = (
        einsum("ai,jkla->ijkl", a.t1, tau22)
    )

    tau38 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau36)
        + einsum("kjil->ijkl", tau37)
    )

    tau39 = (
        einsum("caij,ijbc->ab", a.t2, tau19)
    )

    tau40 = (
        einsum("ab->ab", h.f.vv)
        - einsum("ab->ab", tau39)
        + einsum("ab->ab", tau10)
    )

    tau41 = (
        einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
    )

    tau42 = (
        einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
    )

    tau43 = (
        einsum("lkcb,abclij->ijka", h.v.oovv, a.t3)
    )

    tau44 = (
        einsum("bi,jkab->ijka", a.t1, tau19)
    )

    tau45 = (
        - einsum("jkia->ijka", h.v.ooov)
        + 2 * einsum("kjia->ijka", h.v.ooov)
        + einsum("ikja->ijka", tau44)
    )

    tau46 = (
        einsum("bali,jlkb->ijka", a.t2, tau45)
    )

    tau47 = (
        einsum("kjia->ijka", h.v.ooov)
        + einsum("ijka->ijka", tau21)
    )

    tau48 = (
        einsum("abli,jlkb->ijka", a.t2, tau47)
    )

    tau49 = (
        einsum("abli,jklb->ijka", a.t2, tau47)
    )

    tau50 = (
        einsum("kb,baij->ijka", tau6, a.t2)
    )

    tau51 = (
        einsum("iajb->ijab", h.v.ovov)
        + einsum("jiab->ijab", tau31)
    )

    tau52 = (
        einsum("bi,jkab->ijka", a.t1, tau51)
    )

    tau53 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lijk->ijkl", tau17)
        + einsum("lkji->ijkl", tau36)
        + einsum("kjil->ijkl", tau37)
    )

    tau54 = (
        - einsum("iakj->ijka", h.v.ovoo)
        - einsum("kija->ijka", tau41)
        - einsum("kjia->ijka", tau42)
        + einsum("kjia->ijka", tau43)
        - einsum("licb,bacljk->ijka", tau19, a.t3)
        - einsum("jkia->ijka", tau46)
        + einsum("jkia->ijka", tau48)
        + einsum("kjia->ijka", tau49)
        - einsum("kjia->ijka", tau50)
        - einsum("jika->ijka", tau52)
        + einsum("al,lijk->ijka", a.t1, tau53)
    )

    tau55 = (
        einsum("iabj->ijab", h.v.ovvo)
        + einsum("jiab->ijab", tau24)
    )

    tau56 = (
        einsum("iajk->ijka", h.v.ovoo)
        + einsum("bk,iajb->ijka", a.t1, h.v.ovov)
        + einsum("jkia->ijka", tau42)
        - einsum("jkia->ijka", tau43)
        + einsum("licb,bcaljk->ijka", tau19, a.t3)
        + einsum("kjia->ijka", tau46)
        - einsum("kjia->ijka", tau48)
        - einsum("jkia->ijka", tau49)
        + einsum("ib,abkj->ijka", tau6, a.t2)
        + einsum("bj,ikab->ijka", a.t1, tau55)
    )

    tau57 = (
        einsum("di,badc->iabc", a.t1, h.v.vvvv)
    )

    tau58 = (
        einsum("abic->iabc", h.v.vvov)
        + einsum("ibac->iabc", tau57)
    )

    tau59 = (
        einsum("kj,abki->ijab", tau16, a.t2)
    )

    tau60 = (
        2 * einsum("jabi->ijab", h.v.ovvo)
        - einsum("jaib->ijab", h.v.ovov)
        + einsum("kjcb,kica->ijab", tau0, tau7)
        + einsum("ijab->ijab", tau29)
    )

    tau61 = (
        einsum("acbkij->ijkabc", a.t3)
        + einsum("backij->ijkabc", a.t3)
        - einsum("bcakij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau62 = (
        2 * einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau63 = (
        einsum("kjbc,kica->ijab", tau62, h.v.oovv)
    )

    tau64 = (
        einsum("iabj->ijab", h.v.ovvo)
        + einsum("jiab->ijab", tau24)
        - einsum("jiab->ijab", tau25)
        + einsum("ijba->ijab", tau63)
    )

    tau65 = (
        einsum("abckij->ijkabc", a.t3)
        - 2 * einsum("acbkij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau66 = (
        einsum("liad,ljkbcd->ijkabc", tau64, tau65)
    )

    tau67 = (
        einsum("lkcd,lijdab->ijkabc", tau64, tau28)
    )

    tau68 = (
        - einsum("bacd->abcd", h.v.vvvv)
        + einsum("badc->abcd", h.v.vvvv)
    )

    tau69 = (
        einsum("backij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau70 = (
        einsum("iajb->ijab", h.v.ovov)
        + einsum("jiab->ijab", tau31)
        - einsum("jiab->ijab", tau32)
    )

    tau71 = (
        einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau72 = (
        einsum("liad,ljkdbc->ijkabc", tau70, tau71)
    )

    tau73 = (
        einsum("ilad,ljkbcd->ijkabc", tau33, tau71)
    )

    tau74 = (
        einsum("jilk->ijkl", h.v.oooo)
        - einsum("kijl->ijkl", tau17)
        + einsum("lkji->ijkl", tau36)
        + einsum("lijk->ijkl", tau37)
    )

    tau75 = (
        einsum("acbkij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau76 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau36)
        + einsum("lijk->ijkl", tau37)
    )

    tau77 = (
        einsum("baji->ijab", a.t2)
        + einsum("ai,bj->ijab", a.t1, a.t1)
    )

    tau78 = (
        einsum("ijba->ijab", tau77)
        - einsum("ijab->ijab", tau77)
    )

    tau79 = (
        einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
    )

    tau80 = (
        - einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau81 = (
        einsum("ijbc,klmabc->ijklma", h.v.oovv, tau80)
    )

    tau82 = (
        - einsum("lmab,mlkijc->ijkabc", tau77, tau81)
    )

    tau83 = (
        einsum("ilmj,lkmabc->ijkabc", tau17, tau80)
    )

    tau84 = (
        einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau85 = (
        einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
    )

    tau86 = (
        einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau87 = (
        einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau88 = (
        einsum("kjdc,bdakij->iabc", h.v.oovv, a.t3)
    )

    tau89 = (
        einsum("bakj,ijkc->iabc", a.t2, tau21)
    )

    tau90 = (
        einsum("jkcd,dabjik->iabc", tau19, a.t3)
    )

    tau91 = (
        einsum("jidc,jadb->iabc", tau7, h.v.ovvv)
    )

    tau92 = (
        einsum("abic->iabc", h.v.vvov)
        + einsum("ibac->iabc", tau57)
        + einsum("iabc->iabc", tau85)
        - einsum("iabc->iabc", tau86)
        - einsum("ibac->iabc", tau87)
        + einsum("ibac->iabc", tau88)
        + einsum("ibac->iabc", tau89)
        - einsum("iabc->iabc", tau90)
        + einsum("ibca->iabc", tau91)
    )

    tau93 = (
        - einsum("acbkij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau94 = (
        einsum("jkda,jikbdc->iabc", h.v.oovv, tau93)
    )

    tau95 = (
        - einsum("abji->ijab", a.t2)
        + einsum("baji->ijab", a.t2)
    )

    tau96 = (
        einsum("jicd,jadb->iabc", tau62, h.v.ovvv)
    )

    tau97 = (
        einsum("ijka->ijka", h.v.ooov)
        - einsum("kija->ijka", tau21)
    )

    tau98 = (
        einsum("abic->iabc", h.v.vvov)
        - einsum("baic->iabc", h.v.vvov)
        + einsum("jkcd,jikdab->iabc", tau19, tau69)
        - einsum("icba->iabc", tau94)
        - einsum("jida,jbcd->iabc", tau95, h.v.ovvv)
        + einsum("ibca->iabc", tau91)
        + einsum("jibd,jacd->iabc", tau84, h.v.ovvv)
        - einsum("iacb->iabc", tau96)
        - einsum("jkab,kjic->iabc", tau95, tau97)
        - einsum("di,badc->iabc", a.t1, tau68)
    )

    tau99 = (
        einsum("jidc,jabd->iabc", tau84, h.v.ovvv)
    )

    tau100 = (
        einsum("bacd->abcd", h.v.vvvv)
        - einsum("badc->abcd", h.v.vvvv)
    )

    tau101 = (
        - einsum("abic->iabc", h.v.vvov)
        + einsum("baic->iabc", h.v.vvov)
        + einsum("jkcd,jikdab->iabc", tau19, tau1)
        - einsum("icab->iabc", tau94)
        - einsum("ibca->iabc", tau99)
        - einsum("ibca->iabc", tau91)
        + einsum("iacb->iabc", tau99)
        + einsum("iacb->iabc", tau91)
        - einsum("jkab,kjic->iabc", tau84, tau97)
        - einsum("di,badc->iabc", a.t1, tau100)
    )

    tau102 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau92)
    )

    tau103 = (
        einsum("baic->iabc", h.v.vvov)
        + einsum("iabc->iabc", tau57)
        + einsum("ibac->iabc", tau85)
        - einsum("iabc->iabc", tau87)
        - einsum("ibac->iabc", tau86)
        + einsum("iabc->iabc", tau88)
        + einsum("iabc->iabc", tau89)
        - einsum("ibac->iabc", tau90)
        + einsum("iacb->iabc", tau96)
    )

    tau104 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau103)
    )

    tau105 = (
        einsum("iacd,jkldbc->ijklab", h.v.ovvv, tau75)
    )

    tau106 = (
        einsum("iacd,jklcdb->ijklab", h.v.ovvv, tau93)
    )

    tau107 = (
        einsum("imjc,mklabc->ijklab", tau22, tau65)
    )

    tau108 = (
        einsum("mijc,mklabc->ijklab", tau22, tau75)
    )

    tau109 = (
        einsum("mijc,mklabc->ijklab", tau22, tau69)
    )

    tau110 = (
        einsum("ijac,klcb->ijklab", tau27, tau95)
    )

    tau111 = (
        - einsum("jiab->ijab", h.v.oovv)
        + einsum("jiba->ijab", h.v.oovv)
    )

    tau112 = (
        einsum("acki,kjcb->ijab", a.t2, tau111)
    )

    tau113 = (
        einsum("caki,kjcb->ijab", a.t2, tau0)
    )

    tau114 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + einsum("iacb->iabc", h.v.ovvv)
    )

    tau115 = (
        einsum("ci,jabc->ijab", a.t1, tau114)
    )

    tau116 = (
        einsum("jabi->ijab", h.v.ovvo)
        - einsum("jaib->ijab", h.v.ovov)
        - einsum("ijab->ijab", tau112)
        + einsum("ijab->ijab", tau113)
        - einsum("ijab->ijab", tau115)
    )

    tau117 = (
        einsum("caij,klbc->ijklab", a.t2, tau116)
    )

    tau118 = (
        einsum("ic,jklabc->ijklab", tau6, tau75)
    )

    tau119 = (
        einsum("acij,klbc->ijklab", a.t2, tau33)
    )

    tau120 = (
        einsum("abij,klba->ijkl", a.t2, tau111)
    )

    tau121 = (
        einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau122 = (
        einsum("bi,jkba->ijka", a.t1, tau121)
    )

    tau123 = (
        - einsum("ijka->ijka", h.v.ooov)
        + einsum("jika->ijka", h.v.ooov)
        - einsum("kjia->ijka", tau122)
    )

    tau124 = (
        einsum("ai,jkla->ijkl", a.t1, tau123)
    )

    tau125 = (
        - einsum("ijka->ijka", h.v.ooov)
        + einsum("jika->ijka", h.v.ooov)
    )

    tau126 = (
        einsum("ai,jkla->ijkl", a.t1, tau125)
    )

    tau127 = (
        - einsum("jikl->ijkl", h.v.oooo)
        + einsum("jilk->ijkl", h.v.oooo)
        - einsum("lkji->ijkl", tau120)
        - einsum("kjil->ijkl", tau124)
        - einsum("lijk->ijkl", tau126)
    )

    tau128 = (
        einsum("abmi,mjkl->ijklab", a.t2, tau127)
    )

    tau129 = (
        einsum("ijkb,lmab->ijklma", tau47, tau84)
    )

    tau130 = (
        einsum("abij,klmb->ijklma", a.t2, tau47)
    )

    tau131 = (
        einsum("baij,klmb->ijklma", a.t2, tau47)
    )

    tau132 = (
        - einsum("jklmia->ijklma", tau129)
        + einsum("mjilka->ijklma", tau130)
        - einsum("jimlka->ijklma", tau131)
    )

    tau133 = (
        einsum("am,ijkmlb->ijklab", a.t1, tau132)
    )

    tau134 = (
        einsum("bi,jkab->ijka", a.t1, tau111)
    )

    tau135 = (
        - einsum("jkia->ijka", h.v.ooov)
        + einsum("kjia->ijka", h.v.ooov)
        - einsum("ikja->ijka", tau134)
    )

    tau136 = (
        einsum("bi,jkab->ijka", a.t1, tau121)
    )

    tau137 = (
        einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        - einsum("ikja->ijka", tau136)
    )

    tau138 = (
        - einsum("bami,jlkb->ijklma", a.t2, tau135)
        - einsum("baji,mlkb->ijklma", a.t2, tau137)
    )

    tau139 = (
        einsum("abmi,mjkl->ijklab", a.t2, tau53)
    )

    tau140 = (
        - einsum("iljkba->ijklab", tau105)
        - einsum("iljkab->ijklab", tau106)
        + einsum("iljkba->ijklab", tau107)
        - einsum("ikjlba->ijklab", tau107)
        + einsum("iljkba->ijklab", tau108)
        - einsum("ikjlba->ijklab", tau108)
        - einsum("ijklab->ijklab", tau109)
        - einsum("jilkab->ijklab", tau110)
        + einsum("kjliab->ijklab", tau117)
        - einsum("ljkiab->ijklab", tau117)
        - einsum("iljkba->ijklab", tau118)
        + einsum("ljkiba->ijklab", tau119)
        - einsum("kjliba->ijklab", tau119)
        - einsum("jilkba->ijklab", tau128)
        + einsum("ljikab->ijklab", tau133)
        + einsum("bm,jlmika->ijklab", a.t1, tau138)
        + einsum("kijlab->ijklab", tau139)
        - einsum("lijkab->ijklab", tau139)
    )

    tau141 = (
        - einsum("iljkba->ijklab", tau105)
        - einsum("iljkab->ijklab", tau106)
        + einsum("iljkba->ijklab", tau107)
        - einsum("ikjlba->ijklab", tau107)
        + einsum("iljkba->ijklab", tau108)
        - einsum("ikjlba->ijklab", tau108)
        - einsum("ijklab->ijklab", tau109)
        - einsum("jilkab->ijklab", tau110)
        + einsum("kjliab->ijklab", tau117)
        - einsum("ljkiab->ijklab", tau117)
        - einsum("iljkba->ijklab", tau118)
        + einsum("ljkiba->ijklab", tau119)
        - einsum("kjliba->ijklab", tau119)
        - einsum("jilkba->ijklab", tau128)
        + einsum("ljikab->ijklab", tau133)
        + einsum("kijlab->ijklab", tau139)
        - einsum("lijkab->ijklab", tau139)
    )

    tau142 = (
        - einsum("abckij->ijkabc", a.t3)
        + 2 * einsum("acbkij->ijkabc", a.t3)
        + einsum("backij->ijkabc", a.t3)
        - 2 * einsum("bcakij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau143 = (
        einsum("mijc,mklabc->ijklab", tau22, tau71)
    )

    tau144 = (
        einsum("caij,klbc->ijklab", a.t2, tau27)
    )

    tau145 = (
        - einsum("imlc,mjkabc->ijklab", tau22, tau142)
        - einsum("ibcd,ljkcda->ijklab", h.v.ovvv, tau1)
        - einsum("iacd,ljkcbd->ijklab", h.v.ovvv, tau1)
        - einsum("milc,mjkcab->ijklab", tau22, tau1)
        - einsum("ikjlba->ijklab", tau143)
        - einsum("ijklab->ijklab", tau143)
        - einsum("libc,kjca->ijklab", tau33, tau84)
        - einsum("liac,kjcb->ijklab", tau33, tau95)
        - einsum("ic,ljkcba->ijklab", tau6, tau1)
        + einsum("ljkiab->ijklab", tau144)
        + einsum("lkjiba->ijklab", tau144)
        - einsum("lkjiab->ijklab", tau144)
        - einsum("ljkiba->ijklab", tau144)
        - einsum("mikl,mjba->ijklab", tau53, tau95)
        - einsum("mijl,mkba->ijklab", tau53, tau84)
    )

    tau146 = (
        einsum("lkbc,baclij->ijka", tau0, a.t3)
    )

    tau147 = (
        einsum("bali,jklb->ijka", a.t2, tau4)
    )

    tau148 = (
        einsum("kaji->ijka", h.v.ovoo)
        + einsum("jkia->ijka", tau41)
        + einsum("jika->ijka", tau42)
        - einsum("jika->ijka", tau43)
        + einsum("ijka->ijka", tau146)
        + einsum("ijka->ijka", tau147)
        - einsum("ijka->ijka", tau48)
        - einsum("jika->ijka", tau49)
        + einsum("jika->ijka", tau50)
        + einsum("ikja->ijka", tau52)
    )

    tau149 = (
        einsum("jklc,liab->ijkabc", tau148, tau95)
    )

    tau150 = (
        einsum("libc,ljkbac->ijka", tau0, tau1)
    )

    tau151 = (
        einsum("libc,ljkacb->ijka", h.v.oovv, tau69)
    )

    tau152 = (
        einsum("jkbc,iabc->ijka", tau84, h.v.ovvv)
    )

    tau153 = (
        einsum("abli,jlkb->ijka", a.t2, tau137)
    )

    tau154 = (
        einsum("abli,jklb->ijka", a.t2, tau135)
    )

    tau155 = (
        einsum("ib,jkba->ijka", tau6, tau84)
    )

    tau156 = (
        einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau157 = (
        einsum("ci,jacb->ijab", a.t1, tau156)
    )

    tau158 = (
        einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
        - einsum("jiab->ijab", tau157)
    )

    tau159 = (
        einsum("bi,jkab->ijka", a.t1, tau158)
    )

    tau160 = (
        einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau161 = (
        einsum("bi,jkab->ijka", a.t1, tau160)
    )

    tau162 = (
        - einsum("jaik->ijka", h.v.ovoo)
        + einsum("jaki->ijka", h.v.ovoo)
        + einsum("jika->ijka", tau150)
        - einsum("jika->ijka", tau151)
        - einsum("jkia->ijka", tau152)
        - einsum("kija->ijka", tau46)
        - einsum("kija->ijka", tau153)
        + einsum("ikja->ijka", tau147)
        + einsum("ikja->ijka", tau154)
        - einsum("jkia->ijka", tau155)
        - einsum("ijka->ijka", tau159)
        + einsum("kjia->ijka", tau161)
    )

    tau163 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau162)
    )

    tau164 = (
        einsum("jaki->ijka", h.v.ovoo)
        + einsum("kjia->ijka", tau41)
        + einsum("kija->ijka", tau42)
        - einsum("kija->ijka", tau43)
        + einsum("ikja->ijka", tau146)
        + einsum("ikja->ijka", tau46)
        - einsum("ikja->ijka", tau48)
        - einsum("kija->ijka", tau49)
        + einsum("kija->ijka", tau50)
        + einsum("ijka->ijka", tau52)
    )

    tau165 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau164)
    )

    r1 = (
        einsum("ia->ai", h.f.ov.conj())
        + einsum("jkbc,jikbac->ai", tau0, tau1)
        + einsum("bcji,jacb->ai", a.t2, tau2)
        - einsum("bajk,ikjb->ai", a.t2, tau4)
        + einsum("jb,jiba->ai", tau6, tau7)
        + einsum("bj,jiab->ai", a.t1, tau8)
        + einsum("bi,ab->ai", a.t1, tau11)
        - einsum("aj,ji->ai", a.t1, tau16)
    )

    r2 = (
        einsum("dcij,abdc->abij", a.t2, h.v.vvvv)
        + einsum("ci,bajc->abij", a.t1, h.v.vvov)
        + einsum("baji->abij", h.v.vvoo)
        + einsum("balk,jkli->abij", a.t2, tau17)
        - einsum("ijab->abij", tau18)
        - einsum("jiba->abij", tau18)
        + einsum("kacd,cdbkij->abij", tau9, a.t3)
        + einsum("kbdc,cadkij->abij", tau2, a.t3)
        - einsum("lkjc,cabkil->abij", tau20, a.t3)
        - einsum("klic,ljkcba->abij", tau22, tau23)
        + einsum("klic,cbakjl->abij", tau22, a.t3)
        + einsum("lkjc,acbkil->abij", tau22, a.t3)
        + einsum("ikac,kjcb->abij", tau27, tau7)
        - einsum("kc,kijcab->abij", tau6, tau28)
        + einsum("caki,jkbc->abij", a.t2, tau30)
        - einsum("cbkj,ikac->abij", a.t2, tau33)
        - einsum("ackj,ikbc->abij", a.t2, tau33)
        - einsum("acki,jkbc->abij", a.t2, tau34)
        - einsum("bcki,jkac->abij", a.t2, tau35)
        + einsum("abkl,klij->abij", a.t2, tau38)
        + einsum("bc,caji->abij", tau40, a.t2)
        + einsum("ac,bcji->abij", tau40, a.t2)
        + einsum("ak,kjib->abij", a.t1, tau54)
        - einsum("bk,kjia->abij", a.t1, tau56)
        + einsum("cj,iabc->abij", a.t1, tau58)
        - einsum("jiab->abij", tau59)
        - einsum("ijba->abij", tau59)
    )

    r3 = (
        einsum("klcd,lijdab->abcijk", tau60, tau1)
        - einsum("klcd,lijbda->abcijk", tau27, tau61)
        - einsum("ijkacb->abcijk", tau66)
        - einsum("ikjacb->abcijk", tau67)
        + einsum("jikacb->abcijk", tau66)
        + einsum("jkiacb->abcijk", tau67)
        - einsum("bade,cdekij->abcijk", tau68, a.t3)
        - einsum("cade,kijdeb->abcijk", h.v.vvvv, tau69)
        - einsum("cbde,kijdea->abcijk", h.v.vvvv, tau1)
        - einsum("ijkabc->abcijk", tau72)
        - einsum("klad,lijcdb->abcijk", tau33, tau1)
        - einsum("jikbac->abcijk", tau72)
        - einsum("klbd,lijcda->abcijk", tau33, tau69)
        - einsum("ijkcab->abcijk", tau73)
        - einsum("jikcba->abcijk", tau73)
        + einsum("jikabc->abcijk", tau72)
        + einsum("ijkbac->abcijk", tau72)
        - einsum("lmji,lkmbca->abcijk", tau74, tau75)
        - einsum("lmki,ljmcab->abcijk", tau76, tau1)
        - einsum("lmkj,limcab->abcijk", tau76, tau69)
        - einsum("lmba,kjilmc->abcijk", tau78, tau79)
        - einsum("jkicab->abcijk", tau82)
        - einsum("ikjcba->abcijk", tau82)
        - einsum("kijabc->abcijk", tau83)
        - einsum("kjibac->abcijk", tau83)
        - einsum("jida,kcbd->abcijk", tau84, tau92)
        - einsum("jibd,kcad->abcijk", tau84, tau92)
        - einsum("cdki,jabd->abcijk", a.t2, tau98)
        - einsum("cdkj,iabd->abcijk", a.t2, tau101)
        + einsum("kijabc->abcijk", tau102)
        + einsum("kjibca->abcijk", tau104)
        - einsum("kjiacb->abcijk", tau104)
        - einsum("kijbac->abcijk", tau102)
        - einsum("al,lkijcb->abcijk", a.t1, tau140)
        - einsum("bl,lkjica->abcijk", a.t1, tau141)
        - einsum("ad,kijcbd->abcijk", tau40, tau1)
        - einsum("bd,kijcad->abcijk", tau40, tau69)
        - einsum("cd,kijdab->abcijk", tau40, tau69)
        - einsum("cl,ljikba->abcijk", a.t1, tau145)
        - einsum("jkiabc->abcijk", tau149)
        - einsum("ikjbac->abcijk", tau149)
        + einsum("kijacb->abcijk", tau163)
        + einsum("kjibca->abcijk", tau163)
        + einsum("jikcab->abcijk", tau165)
        + einsum("ijkcba->abcijk", tau165)
        - einsum("ijkcab->abcijk", tau165)
        - einsum("jikcba->abcijk", tau165)
        - einsum("li,ljkbac->abcijk", tau16, tau80)
        - einsum("lj,likbac->abcijk", tau16, tau71)
        - einsum("lk,lijcba->abcijk", tau16, tau69)
    )
    return Tensors(t1=r1, t2=r2, t3=r3)
