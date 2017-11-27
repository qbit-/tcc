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


def _rccsdt_unit_calculate_energy(h, a):
    tau0 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau1 = (
        2 * einsum("ia->ia", h.f.ov)
        + einsum("bj,jiba->ia", a.t1, tau0)
    )

    tau2 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    energy = (
        einsum("ai,ia->", a.t1, tau1)
        + einsum("baij,ijab->", a.t2, tau2)
    )
    return energy


def _rccsdt_unit_calc_residuals(h, a):
    tau0 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
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
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau4 = (
        2 * einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        - einsum("ijka->ijka", tau3)
        + 2 * einsum("ikja->ijka", tau3)
    )

    tau5 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau6 = (
        einsum("bj,jiba->ia", a.t1, tau5)
    )

    tau7 = (
        einsum("ia->ia", h.f.ov)
        + einsum("ia->ia", tau6)
    )

    tau8 = (
        2 * einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau9 = (
        2 * einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau10 = (
        einsum("ab->ab", h.f.vv)
        + einsum("ci,iabc->ab", a.t1, tau2)
    )

    tau11 = (
        2 * einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
    )

    tau12 = (
        einsum("ij->ij", h.f.oo)
        + einsum("abkj,kiab->ij", a.t2, tau5)
        + einsum("ak,ikja->ij", a.t1, tau11)
        + einsum("aj,ia->ij", a.t1, tau7)
    )

    r1 = (
        2 * einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("jkbc,jikcab->ai", tau0, tau1)
        + 2 * einsum("bcji,jacb->ai", a.t2, tau2)
        - 2 * einsum("bajk,ikjb->ai", a.t2, tau4)
        + 2 * einsum("jb,jiab->ai", tau7, tau8)
        + 2 * einsum("bj,jiab->ai", a.t1, tau9)
        + 2 * einsum("bi,ab->ai", a.t1, tau10)
        - 2 * einsum("aj,ji->ai", a.t1, tau12)
    )
    tau0 = (
        einsum("balk,lkji->ijab", a.t2, h.v.oooo)
    )

    tau1 = (
        einsum("dcji,badc->ijab", a.t2, h.v.vvvv)
    )

    tau2 = (
        einsum("di,badc->iabc", a.t1, h.v.vvvv)
    )

    tau3 = (
        einsum("ci,jabc->ijab", a.t1, tau2)
    )

    tau4 = (
        einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
    )

    tau5 = (
        einsum("acki,jkbc->ijab", a.t2, tau4)
    )

    tau6 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau7 = (
        einsum("bj,jiba->ia", a.t1, tau6)
    )

    tau8 = (
        einsum("ia->ia", h.f.ov)
        + einsum("ia->ia", tau7)
    )

    tau9 = (
        einsum("kc,cabkij->ijab", tau8, a.t3)
    )

    tau10 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau11 = (
        einsum("cbkj,kiac->ijab", a.t2, tau10)
    )

    tau12 = (
        einsum("cbkj,kica->ijab", a.t2, tau11)
    )

    tau13 = (
        einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
    )

    tau14 = (
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau15 = (
        einsum("ai,jkla->ijkl", a.t1, tau14)
    )

    tau16 = (
        einsum("jilk->ijkl", tau13)
        + einsum("ijkl->ijkl", tau15)
    )

    tau17 = (
        einsum("abkl,ijkl->ijab", a.t2, tau16)
    )

    tau18 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau13)
        + einsum("lkji->ijkl", tau15)
    )

    tau19 = (
        einsum("al,lijk->ijka", a.t1, tau18)
    )

    tau20 = (
        einsum("bk,kjia->ijab", a.t1, tau19)
    )

    tau21 = (
        einsum("baji->ijab", h.v.vvoo)
        + einsum("jiba->ijab", tau0)
        + einsum("jiba->ijab", tau1)
        + einsum("ijab->ijab", tau3)
        + einsum("ijab->ijab", tau5)
        + 2 * einsum("ijab->ijab", tau9)
        + 2 * einsum("ijab->ijab", tau12)
        + einsum("jiba->ijab", tau17)
        + einsum("ijba->ijab", tau20)
    )

    tau22 = (
        einsum("acki,kbcj->ijab", a.t2, h.v.ovvo)
    )

    tau23 = (
        einsum("kbcd,acdkij->ijab", h.v.ovvv, a.t3)
    )

    tau24 = (
        einsum("ci,jacb->ijab", a.t1, h.v.ovvv)
    )

    tau25 = (
        einsum("ackj,ikbc->ijab", a.t2, tau24)
    )

    tau26 = (
        einsum("ilkc,bcaljk->ijab", tau14, a.t3)
    )

    tau27 = (
        2 * einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau28 = (
        einsum("kacd,cbdkij->ijab", tau27, a.t3)
    )

    tau29 = (
        2 * einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
    )

    tau30 = (
        einsum("klic,cabljk->ijab", tau29, a.t3)
    )

    tau31 = (
        einsum("bckj,kiac->ijab", a.t2, tau10)
    )

    tau32 = (
        einsum("jaib->ijab", h.v.ovov)
        - 2 * einsum("jabi->ijab", h.v.ovvo)
        + einsum("jiba->ijab", tau31)
    )

    tau33 = (
        einsum("cbkj,ikac->ijab", a.t2, tau32)
    )

    tau34 = (
        einsum("kc,abckij->ijab", tau8, a.t3)
    )

    tau35 = (
        einsum("cbij,ijca->ab", a.t2, tau6)
    )

    tau36 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + 2 * einsum("iacb->iabc", h.v.ovvv)
    )

    tau37 = (
        einsum("ci,iabc->ab", a.t1, tau36)
    )

    tau38 = (
        einsum("ba->ab", tau35)
        - einsum("ab->ab", tau37)
    )

    tau39 = (
        einsum("ac,cbij->ijab", tau38, a.t2)
    )

    tau40 = (
        einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
    )

    tau41 = (
        einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
    )

    tau42 = (
        einsum("lkcb,abclij->ijka", h.v.oovv, a.t3)
    )

    tau43 = (
        einsum("ablj,ilkb->ijka", a.t2, tau14)
    )

    tau44 = (
        - einsum("ijka->ijka", tau14)
        + 2 * einsum("ikja->ijka", tau14)
    )

    tau45 = (
        einsum("balk,ijlb->ijka", a.t2, tau44)
    )

    tau46 = (
        einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
    )

    tau47 = (
        einsum("iabj->ijab", h.v.ovvo)
        + einsum("jiab->ijab", tau46)
    )

    tau48 = (
        einsum("bk,ijab->ijka", a.t1, tau47)
    )

    tau49 = (
        einsum("jaik->ijka", h.v.ovoo)
        - einsum("ijka->ijka", tau40)
        + einsum("ikja->ijka", tau41)
        - einsum("ikja->ijka", tau42)
        - einsum("ikja->ijka", tau43)
        + einsum("ijka->ijka", tau45)
        + einsum("jkia->ijka", tau48)
    )

    tau50 = (
        einsum("bk,ikja->ijab", a.t1, tau49)
    )

    tau51 = (
        einsum("abkj,kiab->ij", a.t2, tau6)
    )

    tau52 = (
        einsum("ak,ikja->ij", a.t1, tau29)
    )

    tau53 = (
        einsum("ij->ij", tau51)
        + einsum("ij->ij", tau52)
    )

    tau54 = (
        einsum("ki,abkj->ijab", tau53, a.t2)
    )

    tau55 = (
        einsum("ijab->ijab", tau22)
        + einsum("ijab->ijab", tau23)
        + einsum("ijab->ijab", tau25)
        - einsum("ijab->ijab", tau26)
        - einsum("ijba->ijab", tau28)
        + einsum("jiab->ijab", tau30)
        + einsum("jiba->ijab", tau33)
        + einsum("ijba->ijab", tau34)
        + einsum("jiba->ijab", tau39)
        + einsum("ijba->ijab", tau50)
        + einsum("jiba->ijab", tau54)
    )

    tau56 = (
        einsum("ac,cbji->ijab", h.f.vv, a.t2)
    )

    tau57 = (
        einsum("ci,abjc->ijab", a.t1, h.v.vvov)
    )

    tau58 = (
        einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
    )

    tau59 = (
        einsum("kljc,bcalik->ijab", h.v.ooov, a.t3)
    )

    tau60 = (
        einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
    )

    tau61 = (
        einsum("balk,iklj->ijab", a.t2, tau60)
    )

    tau62 = (
        einsum("cakj,ikbc->ijab", a.t2, tau24)
    )

    tau63 = (
        einsum("iklc,cabljk->ijab", tau44, a.t3)
    )

    tau64 = (
        2 * einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau65 = (
        einsum("jkbc,kiac->ijab", tau46, tau64)
    )

    tau66 = (
        einsum("bi,jakb->ijka", a.t1, h.v.ovov)
    )

    tau67 = (
        einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
    )

    tau68 = (
        einsum("al,ijlk->ijka", a.t1, tau60)
    )

    tau69 = (
        einsum("ablj,iklb->ijka", a.t2, tau14)
    )

    tau70 = (
        einsum("libc,bacljk->ijka", tau6, a.t3)
    )

    tau71 = (
        einsum("balk,iljb->ijka", a.t2, tau29)
    )

    tau72 = (
        einsum("ib,bajk->ijka", tau8, a.t2)
    )

    tau73 = (
        - einsum("jika->ijka", tau66)
        + einsum("jika->ijka", tau67)
        + einsum("jika->ijka", tau68)
        + einsum("jkia->ijka", tau69)
        - einsum("ijka->ijka", tau70)
        - einsum("ikja->ijka", tau71)
        - einsum("ikja->ijka", tau72)
    )

    tau74 = (
        einsum("bk,kija->ijab", a.t1, tau73)
    )

    tau75 = (
        einsum("aj,ia->ij", a.t1, tau8)
    )

    tau76 = (
        einsum("ji->ij", h.f.oo)
        + einsum("ji->ij", tau75)
    )

    tau77 = (
        einsum("ik,abkj->ijab", tau76, a.t2)
    )

    tau78 = (
        einsum("ijab->ijab", tau56)
        + einsum("ijab->ijab", tau57)
        - einsum("ijab->ijab", tau58)
        + einsum("ijab->ijab", tau59)
        + einsum("ijab->ijab", tau61)
        - einsum("ijab->ijab", tau62)
        - einsum("ijab->ijab", tau63)
        + einsum("jiab->ijab", tau65)
        + einsum("ijba->ijab", tau74)
        - einsum("ijba->ijab", tau77)
    )

    tau79 = (
        einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau80 = (
        einsum("acki,jkbc->ijab", a.t2, tau79)
    )

    r2 = (
        - 2 * einsum("ijba->abij", tau21)
        + 4 * einsum("ijab->abij", tau21)
        - 4 * einsum("ijab->abij", tau55)
        + 2 * einsum("ijba->abij", tau55)
        + 2 * einsum("jiab->abij", tau55)
        - 4 * einsum("jiba->abij", tau55)
        - 2 * einsum("ijab->abij", tau78)
        + 4 * einsum("ijba->abij", tau78)
        + 4 * einsum("jiab->abij", tau78)
        - 2 * einsum("jiba->abij", tau78)
        + 4 * einsum("jiab->abij", tau80)
        - 2 * einsum("jiba->abij", tau80)
    )
    tau0 = (
        2 * einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau1 = (
        einsum("liad,dbcljk->ijkabc", tau0, a.t3)
    )

    tau2 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + 2 * einsum("iacb->iabc", h.v.ovvv)
    )

    tau3 = (
        einsum("ci,iabc->ab", a.t1, tau2)
    )

    tau4 = (
        einsum("ad,dbcijk->ijkabc", tau3, a.t3)
    )

    tau5 = (
        2 * einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
    )

    tau6 = (
        einsum("ak,ikja->ij", a.t1, tau5)
    )

    tau7 = (
        einsum("li,abcljk->ijkabc", tau6, a.t3)
    )

    tau8 = (
        einsum("kijcab->ijkabc", tau1)
        + einsum("kijcab->ijkabc", tau4)
        - einsum("kijcab->ijkabc", tau7)
    )

    tau9 = (
        einsum("lckd,badlij->ijkabc", h.v.ovov, a.t3)
    )

    tau10 = (
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau11 = (
        einsum("bakj,ijkc->iabc", a.t2, tau10)
    )

    tau12 = (
        einsum("dckj,ibad->ijkabc", a.t2, tau11)
    )

    tau13 = (
        einsum("bakj,imlb->ijklma", a.t2, tau10)
    )

    tau14 = (
        einsum("am,ijklmb->ijklab", a.t1, tau13)
    )

    tau15 = (
        einsum("al,ijklbc->ijkabc", a.t1, tau14)
    )

    tau16 = (
        einsum("adji,jbdc->iabc", a.t2, h.v.ovvv)
    )

    tau17 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau18 = (
        einsum("jkad,dbcjik->iabc", tau17, a.t3)
    )

    tau19 = (
        2 * einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau20 = (
        einsum("daji,jbdc->iabc", a.t2, tau19)
    )

    tau21 = (
        - einsum("abic->iabc", h.v.vvov)
        + einsum("iabc->iabc", tau16)
        + einsum("icab->iabc", tau18)
        - einsum("iabc->iabc", tau20)
    )

    tau22 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau21)
    )

    tau23 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau24 = (
        einsum("bj,jiba->ia", a.t1, tau23)
    )

    tau25 = (
        einsum("ia->ia", h.f.ov)
        + einsum("ia->ia", tau24)
    )

    tau26 = (
        einsum("kb,abij->ijka", tau25, a.t2)
    )

    tau27 = (
        einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
    )

    tau28 = (
        einsum("ai,jkla->ijkl", a.t1, tau10)
    )

    tau29 = (
        einsum("jilk->ijkl", tau27)
        + einsum("ijkl->ijkl", tau28)
    )

    tau30 = (
        einsum("al,ijlk->ijka", a.t1, tau29)
    )

    tau31 = (
        - einsum("jika->ijka", tau26)
        + einsum("jika->ijka", tau30)
    )

    tau32 = (
        einsum("abli,jklc->ijkabc", a.t2, tau31)
    )

    tau33 = (
        einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
    )

    tau34 = (
        einsum("libc,cabljk->ijka", tau17, a.t3)
    )

    tau35 = (
        einsum("bali,jlkb->ijka", a.t2, tau5)
    )

    tau36 = (
        einsum("ikja->ijka", tau33)
        - einsum("kija->ijka", tau34)
        - einsum("ikja->ijka", tau35)
    )

    tau37 = (
        einsum("abli,jklc->ijkabc", a.t2, tau36)
    )

    tau38 = (
        - einsum("ijkabc->ijkabc", tau9)
        + einsum("ijkabc->ijkabc", tau12)
        + einsum("ijkabc->ijkabc", tau15)
        - einsum("jikabc->ijkabc", tau22)
        + einsum("kjicba->ijkabc", tau32)
        + einsum("ijkbac->ijkabc", tau37)
    )

    tau39 = (
        einsum("lcdk,badlij->ijkabc", h.v.ovvo, a.t3)
    )

    tau40 = (
        einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
    )

    tau41 = (
        einsum("caki,kjcb->ijab", a.t2, tau23)
    )

    tau42 = (
        einsum("ijab->ijab", tau40)
        - einsum("ijab->ijab", tau41)
    )

    tau43 = (
        einsum("ilad,bcdljk->ijkabc", tau42, a.t3)
    )

    tau44 = (
        einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau45 = (
        einsum("kjdc,bdakij->iabc", h.v.oovv, a.t3)
    )

    tau46 = (
        - einsum("iabc->iabc", tau44)
        + einsum("iabc->iabc", tau45)
    )

    tau47 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau46)
    )

    tau48 = (
        einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
    )

    tau49 = (
        einsum("lkcb,abclij->ijka", h.v.oovv, a.t3)
    )

    tau50 = (
        - einsum("kaij->ijka", h.v.ovoo)
        + einsum("ikja->ijka", tau48)
        + einsum("ijka->ijka", tau49)
    )

    tau51 = (
        einsum("abli,jklc->ijkabc", a.t2, tau50)
    )

    tau52 = (
        einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
    )

    tau53 = (
        einsum("bamj,imkl->ijklab", a.t2, tau52)
    )

    tau54 = (
        einsum("ci,jacb->ijab", a.t1, h.v.ovvv)
    )

    tau55 = (
        einsum("cakj,ilbc->ijklab", a.t2, tau54)
    )

    tau56 = (
        einsum("imlc,bacmjk->ijklab", tau10, a.t3)
    )

    tau57 = (
        einsum("ijklab->ijklab", tau53)
        - einsum("ijlkab->ijklab", tau55)
        + einsum("ijlkab->ijklab", tau56)
    )

    tau58 = (
        einsum("cl,ijlkab->ijkabc", a.t1, tau57)
    )

    tau59 = (
        - einsum("ijkabc->ijkabc", tau39)
        + einsum("ijkacb->ijkabc", tau43)
        + einsum("jikabc->ijkabc", tau47)
        + einsum("ijkbac->ijkabc", tau51)
        + einsum("ijkbca->ijkabc", tau58)
    )

    tau60 = (
        einsum("ijlm,abcmkl->ijkabc", tau29, a.t3)
    )

    tau61 = (
        einsum("caij,ijbc->ab", a.t2, tau17)
    )

    tau62 = (
        einsum("ab->ab", h.f.vv)
        - einsum("ab->ab", tau61)
    )

    tau63 = (
        einsum("ad,dbcijk->ijkabc", tau62, a.t3)
    )

    tau64 = (
        einsum("imjc,cabmkl->ijklab", tau5, a.t3)
    )

    tau65 = (
        einsum("ic,cabjkl->ijklab", tau25, a.t3)
    )

    tau66 = (
        einsum("iljkab->ijklab", tau64)
        + einsum("iljkab->ijklab", tau65)
    )

    tau67 = (
        einsum("cl,lijkab->ijkabc", a.t1, tau66)
    )

    tau68 = (
        einsum("jikbac->ijkabc", tau60)
        + einsum("kijabc->ijkabc", tau63)
        - einsum("ijkbca->ijkabc", tau67)
    )

    tau69 = (
        einsum("cj,iabc->ijab", a.t1, tau19)
    )

    tau70 = (
        einsum("liad,dbcljk->ijkabc", tau69, a.t3)
    )

    tau71 = (
        einsum("abki,kjab->ij", a.t2, tau23)
    )

    tau72 = (
        einsum("aj,ia->ij", a.t1, tau25)
    )

    tau73 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau71)
        + einsum("ij->ij", tau72)
    )

    tau74 = (
        einsum("li,abcljk->ijkabc", tau73, a.t3)
    )

    tau75 = (
        einsum("baji->ijab", a.t2)
        + einsum("ai,bj->ijab", a.t1, a.t1)
    )

    tau76 = (
        einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
    )

    tau77 = (
        einsum("lmbc,ijklma->ijkabc", tau75, tau76)
    )

    tau78 = (
        - einsum("ijkcab->ijkabc", tau70)
        + einsum("ijkcab->ijkabc", tau74)
        - einsum("ikjcba->ijkabc", tau77)
    )

    tau79 = (
        einsum("di,badc->iabc", a.t1, h.v.vvvv)
    )

    tau80 = (
        einsum("dakj,icbd->ijkabc", a.t2, tau79)
    )

    tau81 = (
        einsum("ilcd,badljk->ijkabc", tau54, a.t3)
    )

    tau82 = (
        einsum("bami,mjlk->ijklab", a.t2, h.v.oooo)
    )

    tau83 = (
        einsum("mklc,bacmij->ijklab", h.v.ooov, a.t3)
    )

    tau84 = (
        einsum("lbcd,dackij->ijklab", h.v.ovvv, a.t3)
    )

    tau85 = (
        einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau86 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau87 = (
        einsum("kjcb,kica->ijab", tau86, h.v.oovv)
    )

    tau88 = (
        einsum("jabi->ijab", h.v.ovvo)
        - einsum("ijab->ijab", tau85)
        + einsum("jiba->ijab", tau87)
    )

    tau89 = (
        einsum("caij,klbc->ijklab", a.t2, tau88)
    )

    tau90 = (
        einsum("ikjlab->ijklab", tau82)
        + einsum("ijklab->ijklab", tau83)
        - einsum("ijlkab->ijklab", tau84)
        - einsum("jilkab->ijklab", tau89)
    )

    tau91 = (
        einsum("cl,ijlkab->ijkabc", a.t1, tau90)
    )

    tau92 = (
        einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
    )

    tau93 = (
        einsum("ablj,ilkb->ijka", a.t2, tau10)
    )

    tau94 = (
        2 * einsum("ijka->ijka", tau10)
        - einsum("ikja->ijka", tau10)
    )

    tau95 = (
        einsum("bali,jlkb->ijka", a.t2, tau94)
    )

    tau96 = (
        einsum("ikja->ijka", tau92)
        - einsum("ijka->ijka", tau93)
        + einsum("jika->ijka", tau95)
    )

    tau97 = (
        einsum("abli,jklc->ijkabc", a.t2, tau96)
    )

    tau98 = (
        einsum("ijkabc->ijkabc", tau80)
        - einsum("ijkabc->ijkabc", tau81)
        + einsum("ijkbca->ijkabc", tau91)
        - einsum("jikbac->ijkabc", tau97)
    )

    tau99 = (
        einsum("kmlc,bacmij->ijklab", h.v.ooov, a.t3)
    )

    tau100 = (
        einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau101 = (
        - einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau100)
    )

    tau102 = (
        einsum("caij,klbc->ijklab", a.t2, tau101)
    )

    tau103 = (
        einsum("ijklab->ijklab", tau99)
        + einsum("jilkab->ijklab", tau102)
    )

    tau104 = (
        einsum("cl,ijlkab->ijkabc", a.t1, tau103)
    )

    tau105 = (
        einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
    )

    tau106 = (
        einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
    )

    tau107 = (
        einsum("bi,jkab->ijka", a.t1, tau106)
    )

    tau108 = (
        einsum("ijka->ijka", tau105)
        + einsum("ijka->ijka", tau107)
    )

    tau109 = (
        einsum("abli,jklc->ijkabc", a.t2, tau108)
    )

    tau110 = (
        - einsum("ijkbca->ijkabc", tau104)
        + einsum("kijbac->ijkabc", tau109)
    )

    tau111 = (
        einsum("ilmk,cabmjl->ijkabc", tau52, a.t3)
    )

    tau112 = (
        einsum("ilcd,badljk->ijkabc", tau106, a.t3)
    )

    tau113 = (
        einsum("bi,jakb->ijka", a.t1, h.v.ovov)
    )

    tau114 = (
        einsum("ablj,iklb->ijka", a.t2, tau10)
    )

    tau115 = (
        - einsum("ikja->ijka", tau113)
        + einsum("ijka->ijka", tau114)
    )

    tau116 = (
        einsum("abli,jklc->ijkabc", a.t2, tau115)
    )

    tau117 = (
        - einsum("ijkabc->ijkabc", tau111)
        + einsum("ijkabc->ijkabc", tau112)
        - einsum("jikbac->ijkabc", tau116)
    )

    tau118 = (
        einsum("mlkj,cabmil->ijkabc", h.v.oooo, a.t3)
    )

    tau119 = (
        einsum("cbed,eadkij->ijkabc", h.v.vvvv, a.t3)
    )

    tau120 = (
        einsum("kica,kjcb->ijab", tau23, tau86)
    )

    tau121 = (
        - einsum("lida,dbcljk->ijkabc", tau120, a.t3)
    )

    tau122 = (
        - einsum("ijka->ijka", tau10)
        + 2 * einsum("ikja->ijka", tau10)
    )

    tau123 = (
        einsum("ijmc,cabmkl->ijklab", tau122, a.t3)
    )

    tau124 = (
        einsum("cl,iljkab->ijkabc", a.t1, tau123)
    )

    tau125 = (
        einsum("ikjacb->ijkabc", tau118)
        + einsum("ikjacb->ijkabc", tau119)
        - einsum("ijkabc->ijkabc", tau121)
        - einsum("ikjcba->ijkabc", tau124)
    )

    tau126 = (
        einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
    )

    tau127 = (
        einsum("dcji,kabd->ijkabc", a.t2, tau126)
    )

    tau128 = (
        einsum("ilad,cbdljk->ijkabc", tau100, a.t3)
    )

    tau129 = (
        einsum("baji,klmb->ijklma", a.t2, h.v.ooov)
    )

    tau130 = (
        einsum("am,ijkmlb->ijklab", a.t1, tau129)
    )

    tau131 = (
        einsum("bamj,ikml->ijklab", a.t2, tau52)
    )

    tau132 = (
        einsum("cakj,ilbc->ijklab", a.t2, tau106)
    )

    tau133 = (
        einsum("ilmc,bacmjk->ijklab", tau10, a.t3)
    )

    tau134 = (
        einsum("ijklab->ijklab", tau130)
        + einsum("ijklab->ijklab", tau131)
        - einsum("ijlkab->ijklab", tau132)
        + einsum("ijlkab->ijklab", tau133)
    )

    tau135 = (
        einsum("cl,ijlkab->ijkabc", a.t1, tau134)
    )

    tau136 = (
        einsum("ijkabc->ijkabc", tau127)
        + einsum("ijkabc->ijkabc", tau128)
        + einsum("ijkbca->ijkabc", tau135)
    )

    r3 = (
        - 4 * einsum("jikabc->abcijk", tau8)
        + 2 * einsum("jikacb->abcijk", tau8)
        + 8 * einsum("jikbac->abcijk", tau8)
        - 4 * einsum("jikbca->abcijk", tau8)
        - 4 * einsum("jikcab->abcijk", tau8)
        + 2 * einsum("jikcba->abcijk", tau8)
        + 2 * einsum("kijabc->abcijk", tau8)
        - 4 * einsum("kijacb->abcijk", tau8)
        - 4 * einsum("kijbac->abcijk", tau8)
        + 2 * einsum("kijbca->abcijk", tau8)
        + 8 * einsum("kijcab->abcijk", tau8)
        - 4 * einsum("kijcba->abcijk", tau8)
        - 2 * einsum("kjiabc->abcijk", tau8)
        + 2 * einsum("kjiacb->abcijk", tau8)
        + 2 * einsum("kjibac->abcijk", tau8)
        - 4 * einsum("kjibca->abcijk", tau8)
        - 4 * einsum("kjicab->abcijk", tau8)
        + 8 * einsum("kjicba->abcijk", tau8)
        - 4 * einsum("ijkabc->abcijk", tau38)
        + 8 * einsum("ijkacb->abcijk", tau38)
        + 2 * einsum("ijkbac->abcijk", tau38)
        - 4 * einsum("ijkbca->abcijk", tau38)
        - 4 * einsum("ijkcab->abcijk", tau38)
        + 2 * einsum("ijkcba->abcijk", tau38)
        + 8 * einsum("ikjabc->abcijk", tau38)
        - 4 * einsum("ikjacb->abcijk", tau38)
        - 4 * einsum("ikjbac->abcijk", tau38)
        + 2 * einsum("ikjbca->abcijk", tau38)
        + 2 * einsum("ikjcab->abcijk", tau38)
        - 4 * einsum("ikjcba->abcijk", tau38)
        + 2 * einsum("jikabc->abcijk", tau38)
        - 4 * einsum("jikacb->abcijk", tau38)
        - 2 * einsum("jikbac->abcijk", tau38)
        + 8 * einsum("jikbca->abcijk", tau38)
        + 2 * einsum("jikcab->abcijk", tau38)
        - 4 * einsum("jikcba->abcijk", tau38)
        - 4 * einsum("jkiabc->abcijk", tau38)
        + 2 * einsum("jkiacb->abcijk", tau38)
        + 8 * einsum("jkibac->abcijk", tau38)
        - 4 * einsum("jkibca->abcijk", tau38)
        - 4 * einsum("jkicab->abcijk", tau38)
        + 2 * einsum("jkicba->abcijk", tau38)
        - 2 * einsum("kijabc->abcijk", tau38)
        + 2 * einsum("kijacb->abcijk", tau38)
        + 2 * einsum("kijbac->abcijk", tau38)
        - 4 * einsum("kijbca->abcijk", tau38)
        - 4 * einsum("kijcab->abcijk", tau38)
        + 8 * einsum("kijcba->abcijk", tau38)
        + 2 * einsum("kjiabc->abcijk", tau38)
        - 4 * einsum("kjiacb->abcijk", tau38)
        - 4 * einsum("kjibac->abcijk", tau38)
        + 2 * einsum("kjibca->abcijk", tau38)
        + 8 * einsum("kjicab->abcijk", tau38)
        - 4 * einsum("kjicba->abcijk", tau38)
        + 8 * einsum("ijkabc->abcijk", tau59)
        - 4 * einsum("ijkacb->abcijk", tau59)
        - 4 * einsum("ijkbac->abcijk", tau59)
        + 2 * einsum("ijkbca->abcijk", tau59)
        + 2 * einsum("ijkcab->abcijk", tau59)
        - 4 * einsum("ijkcba->abcijk", tau59)
        - 4 * einsum("ikjabc->abcijk", tau59)
        + 8 * einsum("ikjacb->abcijk", tau59)
        + 2 * einsum("ikjbac->abcijk", tau59)
        - 4 * einsum("ikjbca->abcijk", tau59)
        - 4 * einsum("ikjcab->abcijk", tau59)
        + 2 * einsum("ikjcba->abcijk", tau59)
        - 4 * einsum("jikabc->abcijk", tau59)
        + 2 * einsum("jikacb->abcijk", tau59)
        + 8 * einsum("jikbac->abcijk", tau59)
        - 4 * einsum("jikbca->abcijk", tau59)
        - 4 * einsum("jikcab->abcijk", tau59)
        + 2 * einsum("jikcba->abcijk", tau59)
        + 2 * einsum("jkiabc->abcijk", tau59)
        - 4 * einsum("jkiacb->abcijk", tau59)
        - 2 * einsum("jkibac->abcijk", tau59)
        + 8 * einsum("jkibca->abcijk", tau59)
        + 2 * einsum("jkicab->abcijk", tau59)
        - 4 * einsum("jkicba->abcijk", tau59)
        + 2 * einsum("kijabc->abcijk", tau59)
        - 4 * einsum("kijacb->abcijk", tau59)
        - 4 * einsum("kijbac->abcijk", tau59)
        + 2 * einsum("kijbca->abcijk", tau59)
        + 8 * einsum("kijcab->abcijk", tau59)
        - 4 * einsum("kijcba->abcijk", tau59)
        - 2 * einsum("kjiabc->abcijk", tau59)
        + 2 * einsum("kjiacb->abcijk", tau59)
        + 2 * einsum("kjibac->abcijk", tau59)
        - 4 * einsum("kjibca->abcijk", tau59)
        - 4 * einsum("kjicab->abcijk", tau59)
        + 8 * einsum("kjicba->abcijk", tau59)
        - 3 * einsum("jikabc->abcijk", tau68)
        + 2 * einsum("jikacb->abcijk", tau68)
        + 2 * einsum("jikbac->abcijk", tau68)
        - 4 * einsum("jikbca->abcijk", tau68)
        - 4 * einsum("jikcab->abcijk", tau68)
        + 8 * einsum("jikcba->abcijk", tau68)
        + 2 * einsum("kijabc->abcijk", tau68)
        - 4 * einsum("kijacb->abcijk", tau68)
        - 3 * einsum("kijbac->abcijk", tau68)
        + 8 * einsum("kijbca->abcijk", tau68)
        + 2 * einsum("kijcab->abcijk", tau68)
        - 4 * einsum("kijcba->abcijk", tau68)
        - 4 * einsum("kjiabc->abcijk", tau68)
        + 8 * einsum("kjiacb->abcijk", tau68)
        + 2 * einsum("kjibac->abcijk", tau68)
        - 4 * einsum("kjibca->abcijk", tau68)
        - 4 * einsum("kjicab->abcijk", tau68)
        + 2 * einsum("kjicba->abcijk", tau68)
        + 2 * einsum("ijkbac->abcijk", tau78)
        - 2 * einsum("ijkcab->abcijk", tau78)
        - 2 * einsum("ijkabc->abcijk", tau78)
        + 4 * einsum("ijkcba->abcijk", tau78)
        + 4 * einsum("ijkacb->abcijk", tau78)
        - 8 * einsum("ijkbca->abcijk", tau78)
        - 2 * einsum("jikbac->abcijk", tau78)
        + 4 * einsum("jikcab->abcijk", tau78)
        + 4 * einsum("jikabc->abcijk", tau78)
        - 2 * einsum("jikcba->abcijk", tau78)
        - 8 * einsum("jikacb->abcijk", tau78)
        + 4 * einsum("jikbca->abcijk", tau78)
        + 4 * einsum("kijbac->abcijk", tau78)
        - 2 * einsum("kijcab->abcijk", tau78)
        - 8 * einsum("kijabc->abcijk", tau78)
        + 4 * einsum("kijcba->abcijk", tau78)
        + 4 * einsum("kijacb->abcijk", tau78)
        - 2 * einsum("kijbca->abcijk", tau78)
        - 4 * einsum("ijkabc->abcijk", tau98)
        + 2 * einsum("ijkacb->abcijk", tau98)
        + 8 * einsum("ijkbac->abcijk", tau98)
        - 4 * einsum("ijkbca->abcijk", tau98)
        - 4 * einsum("ijkcab->abcijk", tau98)
        + 2 * einsum("ijkcba->abcijk", tau98)
        + 2 * einsum("ikjabc->abcijk", tau98)
        - 4 * einsum("ikjacb->abcijk", tau98)
        - 4 * einsum("ikjbac->abcijk", tau98)
        + 2 * einsum("ikjbca->abcijk", tau98)
        + 8 * einsum("ikjcab->abcijk", tau98)
        - 4 * einsum("ikjcba->abcijk", tau98)
        + 8 * einsum("jikabc->abcijk", tau98)
        - 4 * einsum("jikacb->abcijk", tau98)
        - 4 * einsum("jikbac->abcijk", tau98)
        + 2 * einsum("jikbca->abcijk", tau98)
        + 2 * einsum("jikcab->abcijk", tau98)
        - 4 * einsum("jikcba->abcijk", tau98)
        - 2 * einsum("jkiabc->abcijk", tau98)
        + 2 * einsum("jkiacb->abcijk", tau98)
        + 2 * einsum("jkibac->abcijk", tau98)
        - 4 * einsum("jkibca->abcijk", tau98)
        - 4 * einsum("jkicab->abcijk", tau98)
        + 8 * einsum("jkicba->abcijk", tau98)
        - 4 * einsum("kijabc->abcijk", tau98)
        + 8 * einsum("kijacb->abcijk", tau98)
        + 2 * einsum("kijbac->abcijk", tau98)
        - 4 * einsum("kijbca->abcijk", tau98)
        - 4 * einsum("kijcab->abcijk", tau98)
        + 2 * einsum("kijcba->abcijk", tau98)
        + 2 * einsum("kjiabc->abcijk", tau98)
        - 4 * einsum("kjiacb->abcijk", tau98)
        - 2 * einsum("kjibac->abcijk", tau98)
        + 8 * einsum("kjibca->abcijk", tau98)
        + 2 * einsum("kjicab->abcijk", tau98)
        - 4 * einsum("kjicba->abcijk", tau98)
        - 2 * einsum("ijkabc->abcijk", tau110)
        + 4 * einsum("ijkacb->abcijk", tau110)
        + 4 * einsum("ijkbac->abcijk", tau110)
        - 2 * einsum("ijkbca->abcijk", tau110)
        - 8 * einsum("ijkcab->abcijk", tau110)
        + 4 * einsum("ijkcba->abcijk", tau110)
        + 4 * einsum("ikjabc->abcijk", tau110)
        - 2 * einsum("ikjacb->abcijk", tau110)
        - 8 * einsum("ikjbac->abcijk", tau110)
        + 4 * einsum("ikjbca->abcijk", tau110)
        + 4 * einsum("ikjcab->abcijk", tau110)
        - 2 * einsum("ikjcba->abcijk", tau110)
        + 2 * einsum("jikabc->abcijk", tau110)
        - 2 * einsum("jikacb->abcijk", tau110)
        - 2 * einsum("jikbac->abcijk", tau110)
        + 4 * einsum("jikbca->abcijk", tau110)
        + 4 * einsum("jikcab->abcijk", tau110)
        - 8 * einsum("jikcba->abcijk", tau110)
        - 8 * einsum("jkiabc->abcijk", tau110)
        + 4 * einsum("jkiacb->abcijk", tau110)
        + 4 * einsum("jkibac->abcijk", tau110)
        - 2 * einsum("jkibca->abcijk", tau110)
        - 2 * einsum("jkicab->abcijk", tau110)
        + 4 * einsum("jkicba->abcijk", tau110)
        - 2 * einsum("kijabc->abcijk", tau110)
        + 4 * einsum("kijacb->abcijk", tau110)
        + 2 * einsum("kijbac->abcijk", tau110)
        - 8 * einsum("kijbca->abcijk", tau110)
        - 2 * einsum("kijcab->abcijk", tau110)
        + 4 * einsum("kijcba->abcijk", tau110)
        + 4 * einsum("kjiabc->abcijk", tau110)
        - 8 * einsum("kjiacb->abcijk", tau110)
        - 2 * einsum("kjibac->abcijk", tau110)
        + 4 * einsum("kjibca->abcijk", tau110)
        + 4 * einsum("kjicab->abcijk", tau110)
        - 2 * einsum("kjicba->abcijk", tau110)
        - 2 * einsum("ijkabc->abcijk", tau117)
        + 4 * einsum("ijkacb->abcijk", tau117)
        + 2 * einsum("ijkbac->abcijk", tau117)
        - 8 * einsum("ijkbca->abcijk", tau117)
        - 2 * einsum("ijkcab->abcijk", tau117)
        + 4 * einsum("ijkcba->abcijk", tau117)
        + 2 * einsum("ikjabc->abcijk", tau117)
        - 2 * einsum("ikjacb->abcijk", tau117)
        - 2 * einsum("ikjbac->abcijk", tau117)
        + 4 * einsum("ikjbca->abcijk", tau117)
        + 4 * einsum("ikjcab->abcijk", tau117)
        - 8 * einsum("ikjcba->abcijk", tau117)
        + 4 * einsum("jikabc->abcijk", tau117)
        - 8 * einsum("jikacb->abcijk", tau117)
        - 2 * einsum("jikbac->abcijk", tau117)
        + 4 * einsum("jikbca->abcijk", tau117)
        + 4 * einsum("jikcab->abcijk", tau117)
        - 2 * einsum("jikcba->abcijk", tau117)
        - 2 * einsum("jkiabc->abcijk", tau117)
        + 4 * einsum("jkiacb->abcijk", tau117)
        + 4 * einsum("jkibac->abcijk", tau117)
        - 2 * einsum("jkibca->abcijk", tau117)
        - 8 * einsum("jkicab->abcijk", tau117)
        + 4 * einsum("jkicba->abcijk", tau117)
        - 8 * einsum("kijabc->abcijk", tau117)
        + 4 * einsum("kijacb->abcijk", tau117)
        + 4 * einsum("kijbac->abcijk", tau117)
        - 2 * einsum("kijbca->abcijk", tau117)
        - 2 * einsum("kijcab->abcijk", tau117)
        + 4 * einsum("kijcba->abcijk", tau117)
        + 4 * einsum("kjiabc->abcijk", tau117)
        - 2 * einsum("kjiacb->abcijk", tau117)
        - 8 * einsum("kjibac->abcijk", tau117)
        + 4 * einsum("kjibca->abcijk", tau117)
        + 4 * einsum("kjicab->abcijk", tau117)
        - 2 * einsum("kjicba->abcijk", tau117)
        - 4 * einsum("ijkacb->abcijk", tau125)
        + 8 * einsum("ijkabc->abcijk", tau125)
        + 2 * einsum("ijkbca->abcijk", tau125)
        - 4 * einsum("ijkbac->abcijk", tau125)
        - 4 * einsum("ijkcba->abcijk", tau125)
        + 2 * einsum("ijkcab->abcijk", tau125)
        + 2 * einsum("jikacb->abcijk", tau125)
        - 4 * einsum("jikabc->abcijk", tau125)
        - 3 * einsum("jikbca->abcijk", tau125)
        + 8 * einsum("jikbac->abcijk", tau125)
        + 2 * einsum("jikcba->abcijk", tau125)
        - 4 * einsum("jikcab->abcijk", tau125)
        - 3 * einsum("kijacb->abcijk", tau125)
        + 2 * einsum("kijabc->abcijk", tau125)
        + 2 * einsum("kijbca->abcijk", tau125)
        - 4 * einsum("kijbac->abcijk", tau125)
        - 4 * einsum("kijcba->abcijk", tau125)
        + 8 * einsum("kijcab->abcijk", tau125)
        - 2 * einsum("ijkabc->abcijk", tau136)
        + 2 * einsum("ijkacb->abcijk", tau136)
        + 2 * einsum("ijkbac->abcijk", tau136)
        - 4 * einsum("ijkbca->abcijk", tau136)
        - 4 * einsum("ijkcab->abcijk", tau136)
        + 8 * einsum("ijkcba->abcijk", tau136)
        + 2 * einsum("ikjabc->abcijk", tau136)
        - 4 * einsum("ikjacb->abcijk", tau136)
        - 2 * einsum("ikjbac->abcijk", tau136)
        + 8 * einsum("ikjbca->abcijk", tau136)
        + 2 * einsum("ikjcab->abcijk", tau136)
        - 4 * einsum("ikjcba->abcijk", tau136)
        + 2 * einsum("jikabc->abcijk", tau136)
        - 4 * einsum("jikacb->abcijk", tau136)
        - 4 * einsum("jikbac->abcijk", tau136)
        + 2 * einsum("jikbca->abcijk", tau136)
        + 8 * einsum("jikcab->abcijk", tau136)
        - 4 * einsum("jikcba->abcijk", tau136)
        - 4 * einsum("jkiabc->abcijk", tau136)
        + 8 * einsum("jkiacb->abcijk", tau136)
        + 2 * einsum("jkibac->abcijk", tau136)
        - 4 * einsum("jkibca->abcijk", tau136)
        - 4 * einsum("jkicab->abcijk", tau136)
        + 2 * einsum("jkicba->abcijk", tau136)
        - 4 * einsum("kijabc->abcijk", tau136)
        + 2 * einsum("kijacb->abcijk", tau136)
        + 8 * einsum("kijbac->abcijk", tau136)
        - 4 * einsum("kijbca->abcijk", tau136)
        - 4 * einsum("kijcab->abcijk", tau136)
        + 2 * einsum("kijcba->abcijk", tau136)
        + 8 * einsum("kjiabc->abcijk", tau136)
        - 4 * einsum("kjiacb->abcijk", tau136)
        - 4 * einsum("kjibac->abcijk", tau136)
        + 2 * einsum("kjibca->abcijk", tau136)
        + 2 * einsum("kjicab->abcijk", tau136)
        - 4 * einsum("kjicba->abcijk", tau136)
    )
    return Tensors(t1=r1, t2=r2, t3=r3)


def _rccsdt_unit_anti_calculate_energy(h, a):
    tau0 = (
        einsum("bj,jiab->ia", a.t1, h.v.oovv)
    )

    tau1 = (
        einsum("bj,jiba->ia", a.t1, h.v.oovv)
    )

    energy = (
        2 * einsum("ia,ai->", h.f.ov, a.t1)
        - einsum("baji,jiab->", a.t2, h.v.oovv)
        + 2 * einsum("baji,jiba->", a.t2, h.v.oovv)
        - einsum("ai,ia->", a.t1, tau0)
        + 2 * einsum("ai,ia->", a.t1, tau1)
    )
    return energy


def _rccsdt_unit_anti_calc_residuals(h, a):
    tau0 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + 2 * einsum("iacb->iabc", h.v.ovvv)
    )

    tau1 = (
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau2 = (
        2 * einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        - einsum("ijka->ijka", tau1)
        + 2 * einsum("ikja->ijka", tau1)
    )

    tau3 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau4 = (
        einsum("bj,jiba->ia", a.t1, tau3)
    )

    tau5 = (
        einsum("ia->ia", h.f.ov)
        + einsum("ia->ia", tau4)
    )

    tau6 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau7 = (
        2 * einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau8 = (
        einsum("ci,iabc->ab", a.t1, tau0)
    )

    tau9 = (
        einsum("ab->ab", h.f.vv)
        + einsum("ab->ab", tau8)
    )

    tau10 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau11 = (
        einsum("abki,kjba->ij", a.t2, tau10)
    )

    tau12 = (
        2 * einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
    )

    tau13 = (
        einsum("ak,ikja->ij", a.t1, tau12)
    )

    tau14 = (
        einsum("ai,ja->ij", a.t1, tau5)
    )

    tau15 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau11)
        + einsum("ij->ij", tau13)
        + einsum("ji->ij", tau14)
    )

    tau16 = (
        einsum("balk,lkji->ijab", a.t2, h.v.oooo)
    )

    tau17 = (
        einsum("dcji,badc->ijab", a.t2, h.v.vvvv)
    )

    tau18 = (
        einsum("di,badc->iabc", a.t1, h.v.vvvv)
    )

    tau19 = (
        einsum("ci,jabc->ijab", a.t1, tau18)
    )

    tau20 = (
        einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
    )

    tau21 = (
        einsum("acki,jkbc->ijab", a.t2, tau20)
    )

    tau22 = (
        einsum("caki,kjbc->ijab", a.t2, tau10)
    )

    tau23 = (
        einsum("caki,jkbc->ijab", a.t2, tau22)
    )

    tau24 = (
        einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
    )

    tau25 = (
        einsum("ai,jkla->ijkl", a.t1, tau1)
    )

    tau26 = (
        einsum("jilk->ijkl", tau24)
        + einsum("ijkl->ijkl", tau25)
    )

    tau27 = (
        einsum("abkl,ijkl->ijab", a.t2, tau26)
    )

    tau28 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau24)
        + einsum("lkji->ijkl", tau25)
    )

    tau29 = (
        einsum("al,lijk->ijka", a.t1, tau28)
    )

    tau30 = (
        einsum("ak,kjib->ijab", a.t1, tau29)
    )

    tau31 = (
        einsum("baji->ijab", h.v.vvoo)
        + einsum("jiba->ijab", tau16)
        + einsum("jiba->ijab", tau17)
        + einsum("ijab->ijab", tau19)
        + einsum("ijab->ijab", tau21)
        + 2 * einsum("jiba->ijab", tau23)
        + einsum("jiba->ijab", tau27)
        + einsum("ijab->ijab", tau30)
    )

    tau32 = (
        einsum("acki,kbcj->ijab", a.t2, h.v.ovvo)
    )

    tau33 = (
        einsum("ci,jacb->ijab", a.t1, h.v.ovvv)
    )

    tau34 = (
        einsum("ackj,ikbc->ijab", a.t2, tau33)
    )

    tau35 = (
        einsum("acki,kjbc->ijab", a.t2, tau10)
    )

    tau36 = (
        einsum("jaib->ijab", h.v.ovov)
        - 2 * einsum("jabi->ijab", h.v.ovvo)
        + einsum("ijab->ijab", tau35)
    )

    tau37 = (
        einsum("caki,jkbc->ijab", a.t2, tau36)
    )

    tau38 = (
        einsum("caij,ijcb->ab", a.t2, tau3)
    )

    tau39 = (
        einsum("ab->ab", tau38)
        - einsum("ab->ab", tau8)
    )

    tau40 = (
        einsum("bc,caij->ijab", tau39, a.t2)
    )

    tau41 = (
        einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
    )

    tau42 = (
        einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
    )

    tau43 = (
        einsum("ablj,ilkb->ijka", a.t2, tau1)
    )

    tau44 = (
        2 * einsum("ijka->ijka", tau1)
        - einsum("ikja->ijka", tau1)
    )

    tau45 = (
        einsum("bali,jlkb->ijka", a.t2, tau44)
    )

    tau46 = (
        einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
    )

    tau47 = (
        einsum("iabj->ijab", h.v.ovvo)
        + einsum("jiab->ijab", tau46)
    )

    tau48 = (
        einsum("bi,jkab->ijka", a.t1, tau47)
    )

    tau49 = (
        einsum("jaik->ijka", h.v.ovoo)
        - einsum("ijka->ijka", tau41)
        + einsum("ikja->ijka", tau42)
        - einsum("ikja->ijka", tau43)
        + einsum("kija->ijka", tau45)
        + einsum("ijka->ijka", tau48)
    )

    tau50 = (
        einsum("ak,ikjb->ijab", a.t1, tau49)
    )

    tau51 = (
        einsum("ji->ij", tau11)
        + einsum("ij->ij", tau13)
    )

    tau52 = (
        einsum("kj,abki->ijab", tau51, a.t2)
    )

    tau53 = (
        einsum("ijab->ijab", tau32)
        + einsum("ijab->ijab", tau34)
        + einsum("ijab->ijab", tau37)
        + einsum("jiab->ijab", tau40)
        + einsum("ijab->ijab", tau50)
        + einsum("ijba->ijab", tau52)
    )

    tau54 = (
        einsum("ac,cbji->ijab", h.f.vv, a.t2)
    )

    tau55 = (
        einsum("ci,abjc->ijab", a.t1, h.v.vvov)
    )

    tau56 = (
        einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
    )

    tau57 = (
        einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
    )

    tau58 = (
        einsum("balk,iklj->ijab", a.t2, tau57)
    )

    tau59 = (
        einsum("ackj,ikbc->ijab", a.t2, tau46)
    )

    tau60 = (
        2 * einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau61 = (
        einsum("ci,jabc->ijab", a.t1, tau60)
    )

    tau62 = (
        einsum("caki,jkbc->ijab", a.t2, tau61)
    )

    tau63 = (
        einsum("bi,jakb->ijka", a.t1, h.v.ovov)
    )

    tau64 = (
        einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
    )

    tau65 = (
        einsum("al,ijlk->ijka", a.t1, tau57)
    )

    tau66 = (
        einsum("ablj,iklb->ijka", a.t2, tau1)
    )

    tau67 = (
        einsum("bali,jlkb->ijka", a.t2, tau12)
    )

    tau68 = (
        einsum("kb,baij->ijka", tau5, a.t2)
    )

    tau69 = (
        - einsum("jika->ijka", tau63)
        + einsum("jika->ijka", tau64)
        + einsum("jika->ijka", tau65)
        + einsum("jkia->ijka", tau66)
        - einsum("jika->ijka", tau67)
        - einsum("kjia->ijka", tau68)
    )

    tau70 = (
        einsum("ak,kijb->ijab", a.t1, tau69)
    )

    tau71 = (
        einsum("ji->ij", h.f.oo)
        + einsum("ij->ij", tau14)
    )

    tau72 = (
        einsum("jk,abki->ijab", tau71, a.t2)
    )

    tau73 = (
        - einsum("ijab->ijab", tau54)
        - einsum("ijab->ijab", tau55)
        + einsum("ijab->ijab", tau56)
        - einsum("ijab->ijab", tau58)
        + einsum("ijab->ijab", tau59)
        - einsum("jiab->ijab", tau62)
        - einsum("ijab->ijab", tau70)
        + einsum("jiba->ijab", tau72)
    )

    tau74 = (
        einsum("kbcd,dackij->ijab", h.v.ovvv, a.t3)
    )

    tau75 = (
        einsum("lkcb,cablij->ijka", h.v.oovv, a.t3)
    )

    tau76 = (
        einsum("ak,ijkb->ijab", a.t1, tau75)
    )

    tau77 = (
        einsum("ijab->ijab", tau74)
        - einsum("ijab->ijab", tau76)
    )

    tau78 = (
        einsum("kljc,cablik->ijab", h.v.ooov, a.t3)
    )

    tau79 = (
        einsum("ilkc,cabljk->ijab", tau1, a.t3)
    )

    tau80 = (
        - einsum("ijba->ijab", tau78)
        + einsum("ijba->ijab", tau79)
    )

    tau81 = (
        einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau82 = (
        einsum("acki,jkbc->ijab", a.t2, tau81)
    )

    tau83 = (
        einsum("lcdk,dablij->ijkabc", h.v.ovvo, a.t3)
    )

    tau84 = (
        einsum("lckd,dablij->ijkabc", h.v.ovov, a.t3)
    )

    tau85 = (
        einsum("cbed,eadkij->ijkabc", h.v.vvvv, a.t3)
    )

    tau86 = (
        einsum("ab->ab", h.f.vv)
        - einsum("ab->ab", tau38)
    )

    tau87 = (
        einsum("cd,dabijk->ijkabc", tau86, a.t3)
    )

    tau88 = (
        einsum("lc,cabijk->ijklab", tau5, a.t3)
    )

    tau89 = (
        einsum("al,kijlbc->ijkabc", a.t1, tau88)
    )

    tau90 = (
        - einsum("ijkacb->ijkabc", tau85)
        + einsum("kijbca->ijkabc", tau87)
        + einsum("ijkacb->ijkabc", tau89)
    )

    tau91 = (
        einsum("mlkj,cabmil->ijkabc", h.v.oooo, a.t3)
    )

    tau92 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau11)
        + einsum("ji->ij", tau14)
    )

    tau93 = (
        einsum("lk,abclij->ijkabc", tau92, a.t3)
    )

    tau94 = (
        einsum("ikjcab->ijkabc", tau91)
        + einsum("jkicab->ijkabc", tau93)
    )

    tau95 = (
        einsum("bakj,ijkc->iabc", a.t2, tau1)
    )

    tau96 = (
        einsum("dckj,ibad->ijkabc", a.t2, tau95)
    )

    tau97 = (
        einsum("bakj,imlb->ijklma", a.t2, tau1)
    )

    tau98 = (
        einsum("am,ijklmb->ijklab", a.t1, tau97)
    )

    tau99 = (
        einsum("al,ijklbc->ijkabc", a.t1, tau98)
    )

    tau100 = (
        einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau101 = (
        2 * einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau102 = (
        einsum("jicd,jadb->iabc", tau101, h.v.ovvv)
    )

    tau103 = (
        einsum("abic->iabc", h.v.vvov)
        - einsum("iabc->iabc", tau100)
        + einsum("ibca->iabc", tau102)
    )

    tau104 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau103)
    )

    tau105 = (
        einsum("kb,abij->ijka", tau5, a.t2)
    )

    tau106 = (
        einsum("al,ijlk->ijka", a.t1, tau26)
    )

    tau107 = (
        - einsum("jika->ijka", tau105)
        + einsum("jika->ijka", tau106)
    )

    tau108 = (
        einsum("abli,jklc->ijkabc", a.t2, tau107)
    )

    tau109 = (
        einsum("ijka->ijka", tau64)
        - einsum("ijka->ijka", tau67)
    )

    tau110 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau109)
    )

    tau111 = (
        einsum("ijkabc->ijkabc", tau96)
        + einsum("ijkabc->ijkabc", tau99)
        + einsum("jikabc->ijkabc", tau104)
        + einsum("kjicba->ijkabc", tau108)
        + einsum("ijkbac->ijkabc", tau110)
    )

    tau112 = (
        einsum("dakj,icbd->ijkabc", a.t2, tau18)
    )

    tau113 = (
        einsum("bami,mjlk->ijklab", a.t2, h.v.oooo)
    )

    tau114 = (
        einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau115 = (
        einsum("kjcb,kica->ijab", tau6, h.v.oovv)
    )

    tau116 = (
        einsum("jabi->ijab", h.v.ovvo)
        - einsum("ijab->ijab", tau114)
        + einsum("jiba->ijab", tau115)
    )

    tau117 = (
        einsum("caij,klbc->ijklab", a.t2, tau116)
    )

    tau118 = (
        einsum("ikjlab->ijklab", tau113)
        - einsum("jilkab->ijklab", tau117)
    )

    tau119 = (
        einsum("al,ijlkbc->ijkabc", a.t1, tau118)
    )

    tau120 = (
        einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
    )

    tau121 = (
        einsum("ikja->ijka", tau120)
        - einsum("ijka->ijka", tau43)
        + einsum("jika->ijka", tau45)
    )

    tau122 = (
        einsum("abli,jklc->ijkabc", a.t2, tau121)
    )

    tau123 = (
        einsum("ijkabc->ijkabc", tau112)
        + einsum("ijkabc->ijkabc", tau119)
        - einsum("jikbac->ijkabc", tau122)
    )

    tau124 = (
        einsum("bi,jkab->ijka", a.t1, tau46)
    )

    tau125 = (
        einsum("ijka->ijka", tau42)
        + einsum("ijka->ijka", tau124)
    )

    tau126 = (
        einsum("abli,jklc->ijkabc", a.t2, tau125)
    )

    tau127 = (
        - einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau81)
    )

    tau128 = (
        einsum("caij,klbc->ijklab", a.t2, tau127)
    )

    tau129 = (
        - einsum("al,jiklbc->ijkabc", a.t1, tau128)
    )

    tau130 = (
        einsum("kijbac->ijkabc", tau126)
        + einsum("ijkabc->ijkabc", tau129)
    )

    tau131 = (
        - einsum("ikja->ijka", tau63)
        + einsum("ijka->ijka", tau66)
    )

    tau132 = (
        einsum("abli,jklc->ijkabc", a.t2, tau131)
    )

    tau133 = (
        einsum("imlc,cabmjk->ijklab", tau1, a.t3)
    )

    tau134 = (
        einsum("al,ikjlcb->ijkabc", a.t1, tau133)
    )

    tau135 = (
        einsum("ijab->ijab", tau20)
        - einsum("ijab->ijab", tau22)
    )

    tau136 = (
        einsum("klcd,dablij->ijkabc", tau135, a.t3)
    )

    tau137 = (
        einsum("ikjacb->ijkabc", tau134)
        + einsum("jkibca->ijkabc", tau136)
    )

    tau138 = (
        einsum("bali,jklc->ijkabc", a.t2, tau75)
    )

    tau139 = (
        einsum("kjdc,dabkij->iabc", h.v.oovv, a.t3)
    )

    tau140 = (
        - einsum("daji,kcbd->ijkabc", a.t2, tau139)
    )

    tau141 = (
        einsum("baji->ijab", a.t2)
        + einsum("ai,bj->ijab", a.t1, a.t1)
    )

    tau142 = (
        einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
    )

    tau143 = (
        einsum("lmbc,ijklma->ijkabc", tau141, tau142)
    )

    tau144 = (
        einsum("ilad,dbcljk->ijkabc", tau81, a.t3)
    )

    tau145 = (
        einsum("ilmc,cabmjk->ijklab", tau1, a.t3)
    )

    tau146 = (
        einsum("al,ikjlcb->ijkabc", a.t1, tau145)
    )

    tau147 = (
        einsum("ikjacb->ijkabc", tau144)
        + einsum("ikjacb->ijkabc", tau146)
    )

    tau148 = (
        einsum("jkml,abclim->ijkabc", tau26, a.t3)
    )

    tau149 = (
        einsum("lk,abclij->ijkabc", tau13, a.t3)
    )

    tau150 = (
        - einsum("kjicab->ijkabc", tau148)
        - einsum("ijkcab->ijkabc", tau149)
    )

    tau151 = (
        einsum("kmlc,cabmij->ijklab", h.v.ooov, a.t3)
    )

    tau152 = (
        - einsum("al,ijlkcb->ijkabc", a.t1, tau151)
    )

    tau153 = (
        einsum("mklc,cabmij->ijklab", h.v.ooov, a.t3)
    )

    tau154 = (
        - einsum("al,ijlkcb->ijkabc", a.t1, tau153)
    )

    tau155 = (
        einsum("ilmk,cabmjl->ijkabc", tau57, a.t3)
    )

    tau156 = (
        einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau157 = (
        einsum("daji,kbcd->ijkabc", a.t2, tau156)
    )

    tau158 = (
        - einsum("jaik->ijka", h.v.ovoo)
        + einsum("ijka->ijka", tau41)
    )

    tau159 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau158)
    )

    tau160 = (
        einsum("bamj,imkl->ijklab", a.t2, tau57)
    )

    tau161 = (
        einsum("cakj,ilbc->ijklab", a.t2, tau33)
    )

    tau162 = (
        einsum("ijklab->ijklab", tau160)
        - einsum("ijlkab->ijklab", tau161)
    )

    tau163 = (
        einsum("al,ijlkbc->ijkabc", a.t1, tau162)
    )

    tau164 = (
        einsum("ijkabc->ijkabc", tau157)
        - einsum("ijkbac->ijkabc", tau159)
        - einsum("ijkabc->ijkabc", tau163)
    )

    tau165 = (
        einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
    )

    tau166 = (
        einsum("dcji,kabd->ijkabc", a.t2, tau165)
    )

    tau167 = (
        einsum("baji,klmb->ijklma", a.t2, h.v.ooov)
    )

    tau168 = (
        einsum("am,ijkmlb->ijklab", a.t1, tau167)
    )

    tau169 = (
        einsum("bamj,ikml->ijklab", a.t2, tau57)
    )

    tau170 = (
        einsum("cakj,ilbc->ijklab", a.t2, tau46)
    )

    tau171 = (
        einsum("ijklab->ijklab", tau168)
        + einsum("ijklab->ijklab", tau169)
        - einsum("ijlkab->ijklab", tau170)
    )

    tau172 = (
        einsum("al,ijlkbc->ijkabc", a.t1, tau171)
    )

    tau173 = (
        einsum("ijkabc->ijkabc", tau166)
        + einsum("ijkabc->ijkabc", tau172)
    )

    tau174 = (
        einsum("ilcd,dabljk->ijkabc", tau46, a.t3)
    )

    tau175 = (
        einsum("cd,dabijk->ijkabc", tau8, a.t3)
    )

    tau176 = (
        einsum("lbcd,dackij->ijklab", h.v.ovvv, a.t3)
    )

    tau177 = (
        einsum("al,ijklbc->ijkabc", a.t1, tau176)
    )

    tau178 = (
        einsum("ilcd,dabljk->ijkabc", tau33, a.t3)
    )

    r1 = (
        12 * einsum("kjcb,cabkij->ai", h.v.oovv, a.t3)
        + 2 * einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("bcji,jacb->ai", a.t2, tau0)
        - 2 * einsum("bajk,ikjb->ai", a.t2, tau2)
        + 2 * einsum("jb,jiba->ai", tau5, tau6)
        + 2 * einsum("bj,jiab->ai", a.t1, tau7)
        + 2 * einsum("bi,ab->ai", a.t1, tau9)
        - 2 * einsum("aj,ji->ai", a.t1, tau15)
    )

    r2 = (
        - 2 * einsum("ijba->abij", tau31)
        + 4 * einsum("ijab->abij", tau31)
        - 4 * einsum("ijab->abij", tau53)
        + 2 * einsum("ijba->abij", tau53)
        + 2 * einsum("jiab->abij", tau53)
        - 4 * einsum("jiba->abij", tau53)
        + 2 * einsum("ijab->abij", tau73)
        - 4 * einsum("ijba->abij", tau73)
        - 4 * einsum("jiab->abij", tau73)
        + 2 * einsum("jiba->abij", tau73)
        + 24 * einsum("jiab->abij", tau77)
        - 24 * einsum("jiba->abij", tau77)
        + 24 * einsum("ijba->abij", tau80)
        - 24 * einsum("jiba->abij", tau80)
        + 4 * einsum("jiab->abij", tau82)
        - 2 * einsum("jiba->abij", tau82)
        + 24 * einsum("kc,cabkij->abij", tau5, a.t3)
    )

    r3 = (
        - 48 * einsum("jikabc->abcijk", tau83)
        + 24 * einsum("jikacb->abcijk", tau83)
        + 24 * einsum("jikcba->abcijk", tau83)
        + 24 * einsum("kijabc->abcijk", tau83)
        - 48 * einsum("kijacb->abcijk", tau83)
        - 24 * einsum("kijcba->abcijk", tau83)
        - 16 * einsum("kjiabc->abcijk", tau83)
        + 24 * einsum("kjiacb->abcijk", tau83)
        + 48 * einsum("kjicba->abcijk", tau83)
        + 22 * einsum("jikabc->abcijk", tau84)
        - 24 * einsum("jikacb->abcijk", tau84)
        - 24 * einsum("jikcba->abcijk", tau84)
        - 22 * einsum("kijabc->abcijk", tau84)
        + 24 * einsum("kijacb->abcijk", tau84)
        + 24 * einsum("kijcba->abcijk", tau84)
        + 22 * einsum("kjiabc->abcijk", tau84)
        - 24 * einsum("kjiacb->abcijk", tau84)
        - 24 * einsum("kjicba->abcijk", tau84)
        + 23 * einsum("kijabc->abcijk", tau90)
        + 23 * einsum("kijbca->abcijk", tau90)
        - 24 * einsum("kijcba->abcijk", tau90)
        - 24 * einsum("ijkcab->abcijk", tau94)
        + 23 * einsum("jikcab->abcijk", tau94)
        - 23 * einsum("kijcab->abcijk", tau94)
        - 4 * einsum("ijkabc->abcijk", tau111)
        + 8 * einsum("ijkacb->abcijk", tau111)
        + 2 * einsum("ijkbac->abcijk", tau111)
        - 4 * einsum("ijkbca->abcijk", tau111)
        - 4 * einsum("ijkcab->abcijk", tau111)
        + 2 * einsum("ijkcba->abcijk", tau111)
        + 8 * einsum("ikjabc->abcijk", tau111)
        - 4 * einsum("ikjacb->abcijk", tau111)
        - 4 * einsum("ikjbac->abcijk", tau111)
        + 2 * einsum("ikjbca->abcijk", tau111)
        + 2 * einsum("ikjcab->abcijk", tau111)
        - 4 * einsum("ikjcba->abcijk", tau111)
        + 2 * einsum("jikabc->abcijk", tau111)
        - 4 * einsum("jikacb->abcijk", tau111)
        - 2 * einsum("jikbac->abcijk", tau111)
        + 8 * einsum("jikbca->abcijk", tau111)
        + 2 * einsum("jikcab->abcijk", tau111)
        - 4 * einsum("jikcba->abcijk", tau111)
        - 4 * einsum("jkiabc->abcijk", tau111)
        + 2 * einsum("jkiacb->abcijk", tau111)
        + 8 * einsum("jkibac->abcijk", tau111)
        - 4 * einsum("jkibca->abcijk", tau111)
        - 4 * einsum("jkicab->abcijk", tau111)
        + 2 * einsum("jkicba->abcijk", tau111)
        - 2 * einsum("kijabc->abcijk", tau111)
        + 2 * einsum("kijacb->abcijk", tau111)
        + 2 * einsum("kijbac->abcijk", tau111)
        - 4 * einsum("kijbca->abcijk", tau111)
        - 4 * einsum("kijcab->abcijk", tau111)
        + 8 * einsum("kijcba->abcijk", tau111)
        + 2 * einsum("kjiabc->abcijk", tau111)
        - 4 * einsum("kjiacb->abcijk", tau111)
        - 4 * einsum("kjibac->abcijk", tau111)
        + 2 * einsum("kjibca->abcijk", tau111)
        + 8 * einsum("kjicab->abcijk", tau111)
        - 4 * einsum("kjicba->abcijk", tau111)
        - 4 * einsum("ijkabc->abcijk", tau123)
        + 2 * einsum("ijkacb->abcijk", tau123)
        + 8 * einsum("ijkbac->abcijk", tau123)
        - 4 * einsum("ijkbca->abcijk", tau123)
        - 4 * einsum("ijkcab->abcijk", tau123)
        + 2 * einsum("ijkcba->abcijk", tau123)
        + 2 * einsum("ikjabc->abcijk", tau123)
        - 4 * einsum("ikjacb->abcijk", tau123)
        - 4 * einsum("ikjbac->abcijk", tau123)
        + 2 * einsum("ikjbca->abcijk", tau123)
        + 8 * einsum("ikjcab->abcijk", tau123)
        - 4 * einsum("ikjcba->abcijk", tau123)
        + 8 * einsum("jikabc->abcijk", tau123)
        - 4 * einsum("jikacb->abcijk", tau123)
        - 4 * einsum("jikbac->abcijk", tau123)
        + 2 * einsum("jikbca->abcijk", tau123)
        + 2 * einsum("jikcab->abcijk", tau123)
        - 4 * einsum("jikcba->abcijk", tau123)
        - 2 * einsum("jkiabc->abcijk", tau123)
        + 2 * einsum("jkiacb->abcijk", tau123)
        + 2 * einsum("jkibac->abcijk", tau123)
        - 4 * einsum("jkibca->abcijk", tau123)
        - 4 * einsum("jkicab->abcijk", tau123)
        + 8 * einsum("jkicba->abcijk", tau123)
        - 4 * einsum("kijabc->abcijk", tau123)
        + 8 * einsum("kijacb->abcijk", tau123)
        + 2 * einsum("kijbac->abcijk", tau123)
        - 4 * einsum("kijbca->abcijk", tau123)
        - 4 * einsum("kijcab->abcijk", tau123)
        + 2 * einsum("kijcba->abcijk", tau123)
        + 2 * einsum("kjiabc->abcijk", tau123)
        - 4 * einsum("kjiacb->abcijk", tau123)
        - 2 * einsum("kjibac->abcijk", tau123)
        + 8 * einsum("kjibca->abcijk", tau123)
        + 2 * einsum("kjicab->abcijk", tau123)
        - 4 * einsum("kjicba->abcijk", tau123)
        - 2 * einsum("ijkabc->abcijk", tau130)
        + 4 * einsum("ijkacb->abcijk", tau130)
        + 4 * einsum("ijkbac->abcijk", tau130)
        - 2 * einsum("ijkbca->abcijk", tau130)
        - 8 * einsum("ijkcab->abcijk", tau130)
        + 4 * einsum("ijkcba->abcijk", tau130)
        + 4 * einsum("ikjabc->abcijk", tau130)
        - 2 * einsum("ikjacb->abcijk", tau130)
        - 8 * einsum("ikjbac->abcijk", tau130)
        + 4 * einsum("ikjbca->abcijk", tau130)
        + 4 * einsum("ikjcab->abcijk", tau130)
        - 2 * einsum("ikjcba->abcijk", tau130)
        + 2 * einsum("jikabc->abcijk", tau130)
        - 2 * einsum("jikacb->abcijk", tau130)
        - 2 * einsum("jikbac->abcijk", tau130)
        + 4 * einsum("jikbca->abcijk", tau130)
        + 4 * einsum("jikcab->abcijk", tau130)
        - 8 * einsum("jikcba->abcijk", tau130)
        - 8 * einsum("jkiabc->abcijk", tau130)
        + 4 * einsum("jkiacb->abcijk", tau130)
        + 4 * einsum("jkibac->abcijk", tau130)
        - 2 * einsum("jkibca->abcijk", tau130)
        - 2 * einsum("jkicab->abcijk", tau130)
        + 4 * einsum("jkicba->abcijk", tau130)
        - 2 * einsum("kijabc->abcijk", tau130)
        + 4 * einsum("kijacb->abcijk", tau130)
        + 2 * einsum("kijbac->abcijk", tau130)
        - 8 * einsum("kijbca->abcijk", tau130)
        - 2 * einsum("kijcab->abcijk", tau130)
        + 4 * einsum("kijcba->abcijk", tau130)
        + 4 * einsum("kjiabc->abcijk", tau130)
        - 8 * einsum("kjiacb->abcijk", tau130)
        - 2 * einsum("kjibac->abcijk", tau130)
        + 4 * einsum("kjibca->abcijk", tau130)
        + 4 * einsum("kjicab->abcijk", tau130)
        - 2 * einsum("kjicba->abcijk", tau130)
        + 2 * einsum("jikbac->abcijk", tau132)
        - 4 * einsum("jikcab->abcijk", tau132)
        - 2 * einsum("jikabc->abcijk", tau132)
        + 8 * einsum("jikcba->abcijk", tau132)
        + 2 * einsum("jikacb->abcijk", tau132)
        - 4 * einsum("jikbca->abcijk", tau132)
        - 2 * einsum("kijbac->abcijk", tau132)
        + 2 * einsum("kijcab->abcijk", tau132)
        + 2 * einsum("kijabc->abcijk", tau132)
        - 4 * einsum("kijcba->abcijk", tau132)
        - 4 * einsum("kijacb->abcijk", tau132)
        + 8 * einsum("kijbca->abcijk", tau132)
        - 4 * einsum("ijkbac->abcijk", tau132)
        + 8 * einsum("ijkcab->abcijk", tau132)
        + 2 * einsum("ijkabc->abcijk", tau132)
        - 4 * einsum("ijkcba->abcijk", tau132)
        - 4 * einsum("ijkacb->abcijk", tau132)
        + 2 * einsum("ijkbca->abcijk", tau132)
        + 2 * einsum("kjibac->abcijk", tau132)
        - 4 * einsum("kjicab->abcijk", tau132)
        - 4 * einsum("kjiabc->abcijk", tau132)
        + 2 * einsum("kjicba->abcijk", tau132)
        + 8 * einsum("kjiacb->abcijk", tau132)
        - 4 * einsum("kjibca->abcijk", tau132)
        + 8 * einsum("ikjbac->abcijk", tau132)
        - 4 * einsum("ikjcab->abcijk", tau132)
        - 4 * einsum("ikjabc->abcijk", tau132)
        + 2 * einsum("ikjcba->abcijk", tau132)
        + 2 * einsum("ikjacb->abcijk", tau132)
        - 4 * einsum("ikjbca->abcijk", tau132)
        - 4 * einsum("jkibac->abcijk", tau132)
        + 2 * einsum("jkicab->abcijk", tau132)
        + 8 * einsum("jkiabc->abcijk", tau132)
        - 4 * einsum("jkicba->abcijk", tau132)
        - 4 * einsum("jkiacb->abcijk", tau132)
        + 2 * einsum("jkibca->abcijk", tau132)
        - 48 * einsum("ijkabc->abcijk", tau137)
        - 24 * einsum("ijkbca->abcijk", tau137)
        + 24 * einsum("ijkcba->abcijk", tau137)
        + 24 * einsum("jikabc->abcijk", tau137)
        + 44 * einsum("jikbca->abcijk", tau137)
        - 24 * einsum("jikcba->abcijk", tau137)
        - 20 * einsum("kijabc->abcijk", tau137)
        - 24 * einsum("kijbca->abcijk", tau137)
        + 48 * einsum("kijcba->abcijk", tau137)
        - 48 * einsum("ikjabc->abcijk", tau138)
        + 48 * einsum("ikjacb->abcijk", tau138)
        + 24 * einsum("ikjbac->abcijk", tau138)
        - 24 * einsum("ikjbca->abcijk", tau138)
        - 24 * einsum("ikjcab->abcijk", tau138)
        + 24 * einsum("ikjcba->abcijk", tau138)
        + 24 * einsum("jkiabc->abcijk", tau138)
        - 24 * einsum("jkiacb->abcijk", tau138)
        - 40 * einsum("jkibac->abcijk", tau138)
        + 48 * einsum("jkibca->abcijk", tau138)
        + 24 * einsum("jkicab->abcijk", tau138)
        - 24 * einsum("jkicba->abcijk", tau138)
        - 16 * einsum("kjiabc->abcijk", tau138)
        + 24 * einsum("kjiacb->abcijk", tau138)
        + 24 * einsum("kjibac->abcijk", tau138)
        - 24 * einsum("kjibca->abcijk", tau138)
        - 48 * einsum("kjicab->abcijk", tau138)
        + 48 * einsum("kjicba->abcijk", tau138)
        - 48 * einsum("ijkacb->abcijk", tau140)
        - 24 * einsum("ijkbac->abcijk", tau140)
        + 24 * einsum("ijkcab->abcijk", tau140)
        + 48 * einsum("ikjacb->abcijk", tau140)
        + 24 * einsum("ikjbac->abcijk", tau140)
        - 24 * einsum("ikjcab->abcijk", tau140)
        + 24 * einsum("jikacb->abcijk", tau140)
        + 48 * einsum("jikbac->abcijk", tau140)
        - 24 * einsum("jikcab->abcijk", tau140)
        - 24 * einsum("jkiacb->abcijk", tau140)
        - 40 * einsum("jkibac->abcijk", tau140)
        + 24 * einsum("jkicab->abcijk", tau140)
        - 24 * einsum("kijacb->abcijk", tau140)
        - 24 * einsum("kijbac->abcijk", tau140)
        + 48 * einsum("kijcab->abcijk", tau140)
        + 16 * einsum("kjiacb->abcijk", tau140)
        + 24 * einsum("kjibac->abcijk", tau140)
        - 48 * einsum("kjicab->abcijk", tau140)
        - 22 * einsum("kijcba->abcijk", tau143)
        + 24 * einsum("kijbca->abcijk", tau143)
        - 24 * einsum("kijacb->abcijk", tau143)
        + 24 * einsum("ijkabc->abcijk", tau147)
        + 24 * einsum("ijkbca->abcijk", tau147)
        - 24 * einsum("ijkcba->abcijk", tau147)
        - 22 * einsum("jikabc->abcijk", tau147)
        - 23 * einsum("jikbca->abcijk", tau147)
        + 24 * einsum("jikcba->abcijk", tau147)
        + 23 * einsum("kijabc->abcijk", tau147)
        + 22 * einsum("kijbca->abcijk", tau147)
        - 24 * einsum("kijcba->abcijk", tau147)
        - 24 * einsum("jikcab->abcijk", tau150)
        + 24 * einsum("kijcab->abcijk", tau150)
        - 22 * einsum("kjicab->abcijk", tau150)
        - 20 * einsum("jikacb->abcijk", tau152)
        - 24 * einsum("jikbac->abcijk", tau152)
        + 48 * einsum("jikcab->abcijk", tau152)
        + 24 * einsum("kijacb->abcijk", tau152)
        + 44 * einsum("kijbac->abcijk", tau152)
        - 24 * einsum("kijcab->abcijk", tau152)
        - 48 * einsum("kjiacb->abcijk", tau152)
        - 24 * einsum("kjibac->abcijk", tau152)
        + 24 * einsum("kjicab->abcijk", tau152)
        + 23 * einsum("jikacb->abcijk", tau154)
        + 24 * einsum("jikbac->abcijk", tau154)
        - 24 * einsum("jikcab->abcijk", tau154)
        - 24 * einsum("kijacb->abcijk", tau154)
        - 23 * einsum("kijbac->abcijk", tau154)
        + 24 * einsum("kijcab->abcijk", tau154)
        + 22 * einsum("kjiacb->abcijk", tau154)
        + 22 * einsum("kjibac->abcijk", tau154)
        - 24 * einsum("kjicab->abcijk", tau154)
        + 24 * einsum("ijkabc->abcijk", tau155)
        - 24 * einsum("ikjabc->abcijk", tau155)
        - 24 * einsum("jikabc->abcijk", tau155)
        + 22 * einsum("jkiabc->abcijk", tau155)
        + 24 * einsum("kijabc->abcijk", tau155)
        - 22 * einsum("kjiabc->abcijk", tau155)
        - 8 * einsum("ijkabc->abcijk", tau164)
        + 4 * einsum("ijkacb->abcijk", tau164)
        + 4 * einsum("ijkbac->abcijk", tau164)
        - 2 * einsum("ijkbca->abcijk", tau164)
        - 2 * einsum("ijkcab->abcijk", tau164)
        + 4 * einsum("ijkcba->abcijk", tau164)
        + 4 * einsum("ikjabc->abcijk", tau164)
        - 8 * einsum("ikjacb->abcijk", tau164)
        - 2 * einsum("ikjbac->abcijk", tau164)
        + 4 * einsum("ikjbca->abcijk", tau164)
        + 4 * einsum("ikjcab->abcijk", tau164)
        - 2 * einsum("ikjcba->abcijk", tau164)
        + 4 * einsum("jikabc->abcijk", tau164)
        - 2 * einsum("jikacb->abcijk", tau164)
        - 8 * einsum("jikbac->abcijk", tau164)
        + 4 * einsum("jikbca->abcijk", tau164)
        + 4 * einsum("jikcab->abcijk", tau164)
        - 2 * einsum("jikcba->abcijk", tau164)
        - 2 * einsum("jkiabc->abcijk", tau164)
        + 4 * einsum("jkiacb->abcijk", tau164)
        + 2 * einsum("jkibac->abcijk", tau164)
        - 8 * einsum("jkibca->abcijk", tau164)
        - 2 * einsum("jkicab->abcijk", tau164)
        + 4 * einsum("jkicba->abcijk", tau164)
        - 2 * einsum("kijabc->abcijk", tau164)
        + 4 * einsum("kijacb->abcijk", tau164)
        + 4 * einsum("kijbac->abcijk", tau164)
        - 2 * einsum("kijbca->abcijk", tau164)
        - 8 * einsum("kijcab->abcijk", tau164)
        + 4 * einsum("kijcba->abcijk", tau164)
        + 2 * einsum("kjiabc->abcijk", tau164)
        - 2 * einsum("kjiacb->abcijk", tau164)
        - 2 * einsum("kjibac->abcijk", tau164)
        + 4 * einsum("kjibca->abcijk", tau164)
        + 4 * einsum("kjicab->abcijk", tau164)
        - 8 * einsum("kjicba->abcijk", tau164)
        - 2 * einsum("ijkabc->abcijk", tau173)
        + 2 * einsum("ijkacb->abcijk", tau173)
        + 2 * einsum("ijkbac->abcijk", tau173)
        - 4 * einsum("ijkbca->abcijk", tau173)
        - 4 * einsum("ijkcab->abcijk", tau173)
        + 8 * einsum("ijkcba->abcijk", tau173)
        + 2 * einsum("ikjabc->abcijk", tau173)
        - 4 * einsum("ikjacb->abcijk", tau173)
        - 2 * einsum("ikjbac->abcijk", tau173)
        + 8 * einsum("ikjbca->abcijk", tau173)
        + 2 * einsum("ikjcab->abcijk", tau173)
        - 4 * einsum("ikjcba->abcijk", tau173)
        + 2 * einsum("jikabc->abcijk", tau173)
        - 4 * einsum("jikacb->abcijk", tau173)
        - 4 * einsum("jikbac->abcijk", tau173)
        + 2 * einsum("jikbca->abcijk", tau173)
        + 8 * einsum("jikcab->abcijk", tau173)
        - 4 * einsum("jikcba->abcijk", tau173)
        - 4 * einsum("jkiabc->abcijk", tau173)
        + 8 * einsum("jkiacb->abcijk", tau173)
        + 2 * einsum("jkibac->abcijk", tau173)
        - 4 * einsum("jkibca->abcijk", tau173)
        - 4 * einsum("jkicab->abcijk", tau173)
        + 2 * einsum("jkicba->abcijk", tau173)
        - 4 * einsum("kijabc->abcijk", tau173)
        + 2 * einsum("kijacb->abcijk", tau173)
        + 8 * einsum("kijbac->abcijk", tau173)
        - 4 * einsum("kijbca->abcijk", tau173)
        - 4 * einsum("kijcab->abcijk", tau173)
        + 2 * einsum("kijcba->abcijk", tau173)
        + 8 * einsum("kjiabc->abcijk", tau173)
        - 4 * einsum("kjiacb->abcijk", tau173)
        - 4 * einsum("kjibac->abcijk", tau173)
        + 2 * einsum("kjibca->abcijk", tau173)
        + 2 * einsum("kjicab->abcijk", tau173)
        - 4 * einsum("kjicba->abcijk", tau173)
        - 16 * einsum("ikjabc->abcijk", tau174)
        + 24 * einsum("ikjacb->abcijk", tau174)
        + 48 * einsum("ikjcba->abcijk", tau174)
        + 24 * einsum("jkiabc->abcijk", tau174)
        - 48 * einsum("jkiacb->abcijk", tau174)
        - 24 * einsum("jkicba->abcijk", tau174)
        - 48 * einsum("kjiabc->abcijk", tau174)
        + 24 * einsum("kjiacb->abcijk", tau174)
        + 24 * einsum("kjicba->abcijk", tau174)
        - 22 * einsum("jkibac->abcijk", tau175)
        + 24 * einsum("jkicab->abcijk", tau175)
        + 24 * einsum("jkibca->abcijk", tau175)
        + 22 * einsum("kijabc->abcijk", tau177)
        - 24 * einsum("kijacb->abcijk", tau177)
        - 22 * einsum("kijbac->abcijk", tau177)
        + 24 * einsum("kijbca->abcijk", tau177)
        + 24 * einsum("kijcab->abcijk", tau177)
        - 24 * einsum("kijcba->abcijk", tau177)
        + 22 * einsum("ikjabc->abcijk", tau178)
        - 24 * einsum("ikjacb->abcijk", tau178)
        - 24 * einsum("ikjcba->abcijk", tau178)
        - 22 * einsum("jkiabc->abcijk", tau178)
        + 24 * einsum("jkiacb->abcijk", tau178)
        + 24 * einsum("jkicba->abcijk", tau178)
        + 22 * einsum("kjiabc->abcijk", tau178)
        - 24 * einsum("kjiacb->abcijk", tau178)
        - 24 * einsum("kjicba->abcijk", tau178)
    )

    return Tensors(t1=r1, t2=r2, t3=r3)
