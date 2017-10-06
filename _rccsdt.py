from numpy import einsum
from tcc.tensors import Tensors


def _rccsdt_calculate_energy(h, a):
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


def _rccsdt_calc_residuals(h, a):
    tau0 = (
        - einsum("backij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau1 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + 2 * einsum("iacb->iabc", h.v.ovvv)
    )

    tau2 = (
        - einsum("jiab->ijab", h.v.oovv)
        + 2 * einsum("jiba->ijab", h.v.oovv)
    )

    tau3 = (
        einsum("bi,jkab->ijka", a.t1, tau2)
    )

    tau4 = (
        2 * einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        + einsum("ikja->ijka", tau3)
    )

    tau5 = (
        2 * einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau6 = (
        einsum("bj,jiab->ia", a.t1, tau5)
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
        2 * einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau11 = (
        einsum("ci,iacb->ab", a.t1, tau10)
    )

    tau12 = (
        einsum("ab->ab", h.f.vv)
        + einsum("ab->ab", tau11)
    )

    tau13 = (
        einsum("abki,kjba->ij", a.t2, tau5)
    )

    tau14 = (
        - einsum("ijka->ijka", h.v.ooov)
        + 2 * einsum("jika->ijka", h.v.ooov)
    )

    tau15 = (
        einsum("ak,kija->ij", a.t1, tau14)
    )

    tau16 = (
        einsum("ai,ja->ij", a.t1, tau7)
    )

    tau17 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau13)
        + einsum("ij->ij", tau15)
        + einsum("ji->ij", tau16)
    )

    tau18 = (
        einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
    )

    tau19 = (
        - einsum("iabc->iabc", h.v.ovvv)
        + einsum("iacb->iabc", h.v.ovvv)
    )

    tau20 = (
        einsum("jiab->ijab", h.v.oovv)
        - einsum("jiba->ijab", h.v.oovv)
    )

    tau21 = (
        einsum("bi,jkab->ijka", a.t1, tau20)
    )

    tau22 = (
        einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
        - einsum("kjia->ijka", tau21)
    )

    tau23 = (
        einsum("bi,kjba->ijka", a.t1, h.v.oovv)
    )

    tau24 = (
        einsum("ijka->ijka", h.v.ooov)
        + einsum("kjia->ijka", tau23)
    )

    tau25 = (
        einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau26 = (
        einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
    )

    tau27 = (
        einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau28 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau29 = (
        einsum("kjcb,kica->ijab", tau28, h.v.oovv)
    )

    tau30 = (
        einsum("jabi->ijab", h.v.ovvo)
        + einsum("ijab->ijab", tau26)
        - einsum("ijab->ijab", tau27)
        + einsum("jiba->ijab", tau29)
    )

    tau31 = (
        2 * einsum("jabi->ijab", h.v.ovvo)
        - einsum("jaib->ijab", h.v.ovov)
        + einsum("ci,jabc->ijab", a.t1, tau10)
    )

    tau32 = (
        einsum("ci,jacb->ijab", a.t1, h.v.ovvv)
    )

    tau33 = (
        einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
    )

    tau34 = (
        einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau32)
        - einsum("ijab->ijab", tau33)
    )

    tau35 = (
        einsum("jabi->ijab", h.v.ovvo)
        + einsum("ijab->ijab", tau26)
    )

    tau36 = (
        einsum("jaib->ijab", h.v.ovov)
        + einsum("ijab->ijab", tau32)
    )

    tau37 = (
        einsum("baji,lkba->ijkl", a.t2, h.v.oovv)
    )

    tau38 = (
        einsum("ai,jkla->ijkl", a.t1, tau24)
    )

    tau39 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau37)
        + einsum("kjil->ijkl", tau38)
    )

    tau40 = (
        einsum("caij,ijcb->ab", a.t2, tau2)
    )

    tau41 = (
        einsum("ab->ab", h.f.vv)
        - einsum("ab->ab", tau40)
        + einsum("ab->ab", tau11)
    )

    tau42 = (
        einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
    )

    tau43 = (
        einsum("cbji,kabc->ijka", a.t2, h.v.ovvv)
    )

    tau44 = (
        einsum("bali,jklb->ijka", a.t2, tau4)
    )

    tau45 = (
        einsum("kjia->ijka", h.v.ooov)
        + einsum("ijka->ijka", tau23)
    )

    tau46 = (
        einsum("abli,jlkb->ijka", a.t2, tau45)
    )

    tau47 = (
        einsum("abli,jklb->ijka", a.t2, tau45)
    )

    tau48 = (
        einsum("kb,baij->ijka", tau7, a.t2)
    )

    tau49 = (
        einsum("iajb->ijab", h.v.ovov)
        + einsum("jiab->ijab", tau32)
    )

    tau50 = (
        einsum("bi,jkab->ijka", a.t1, tau49)
    )

    tau51 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lijk->ijkl", tau18)
        + einsum("lkji->ijkl", tau37)
        + einsum("kjil->ijkl", tau38)
    )

    tau52 = (
        - 3 * einsum("iakj->ijka", h.v.ovoo)
        - 3 * einsum("kija->ijka", tau42)
        - 3 * einsum("kjia->ijka", tau43)
        - einsum("licb,bacljk->ijka", tau20, a.t3)
        - 3 * einsum("jkia->ijka", tau44)
        + 3 * einsum("jkia->ijka", tau46)
        + 3 * einsum("kjia->ijka", tau47)
        - 3 * einsum("kjia->ijka", tau48)
        - 3 * einsum("jika->ijka", tau50)
        + 3 * einsum("al,lijk->ijka", a.t1, tau51)
    )

    tau53 = (
        einsum("acbkij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau54 = (
        einsum("bi,jkab->ijka", a.t1, tau5)
    )

    tau55 = (
        - einsum("jkia->ijka", h.v.ooov)
        + 2 * einsum("kjia->ijka", h.v.ooov)
        + einsum("ikja->ijka", tau54)
    )

    tau56 = (
        einsum("bali,jlkb->ijka", a.t2, tau55)
    )

    tau57 = (
        einsum("iabj->ijab", h.v.ovvo)
        + einsum("jiab->ijab", tau26)
    )

    tau58 = (
        3 * einsum("iajk->ijka", h.v.ovoo)
        + 3 * einsum("bk,iajb->ijka", a.t1, h.v.ovov)
        + 3 * einsum("jkia->ijka", tau43)
        - einsum("libc,ljkbca->ijka", h.v.oovv, tau53)
        + 3 * einsum("kjia->ijka", tau56)
        - 3 * einsum("kjia->ijka", tau46)
        - 3 * einsum("jkia->ijka", tau47)
        + 3 * einsum("ib,abkj->ijka", tau7, a.t2)
        + 3 * einsum("bj,ikab->ijka", a.t1, tau57)
    )

    tau59 = (
        einsum("di,badc->iabc", a.t1, h.v.vvvv)
    )

    tau60 = (
        einsum("abic->iabc", h.v.vvov)
        + einsum("ibac->iabc", tau59)
    )

    tau61 = (
        einsum("kj,abki->ijab", tau17, a.t2)
    )

    tau62 = (
        - einsum("jiab->ijab", h.v.oovv)
        + einsum("jiba->ijab", h.v.oovv)
    )

    tau63 = (
        einsum("acki,kjcb->ijab", a.t2, tau62)
    )

    tau64 = (
        einsum("caki,kjcb->ijab", a.t2, tau2)
    )

    tau65 = (
        einsum("iabc->iabc", h.v.ovvv)
        - einsum("iacb->iabc", h.v.ovvv)
    )

    tau66 = (
        einsum("ci,jacb->ijab", a.t1, tau65)
    )

    tau67 = (
        einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
        - einsum("jiab->ijab", tau63)
        + einsum("jiab->ijab", tau64)
        - einsum("jiab->ijab", tau66)
    )

    tau68 = (
        - einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau69 = (
        einsum("lkcd,lijadb->ijkabc", tau67, tau25)
    )

    tau70 = (
        einsum("ci,jabc->ijab", a.t1, tau19)
    )

    tau71 = (
        einsum("jabi->ijab", h.v.ovvo)
        - einsum("jaib->ijab", h.v.ovov)
        - einsum("ijab->ijab", tau63)
        + einsum("ijab->ijab", tau64)
        - einsum("ijab->ijab", tau70)
    )

    tau72 = (
        einsum("bacd->abcd", h.v.vvvv)
        - einsum("badc->abcd", h.v.vvvv)
    )

    tau73 = (
        einsum("backij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau74 = (
        einsum("ilad,ljkbcd->ijkabc", tau34, tau68)
    )

    tau75 = (
        einsum("jilk->ijkl", h.v.oooo)
        - einsum("kijl->ijkl", tau18)
        + einsum("lkji->ijkl", tau37)
        + einsum("lijk->ijkl", tau38)
    )

    tau76 = (
        - einsum("acbkij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau77 = (
        einsum("jilk->ijkl", h.v.oooo)
        + einsum("lkji->ijkl", tau37)
        + einsum("lijk->ijkl", tau38)
    )

    tau78 = (
        einsum("lmjk,limabc->ijkabc", tau77, tau73)
    )

    tau79 = (
        einsum("baji->ijab", a.t2)
        + einsum("ai,bj->ijab", a.t1, a.t1)
    )

    tau80 = (
        einsum("ijba->ijab", tau79)
        - einsum("ijab->ijab", tau79)
    )

    tau81 = (
        einsum("mlcb,cabkij->ijklma", h.v.oovv, a.t3)
    )

    tau82 = (
        einsum("ijbc,klmbac->ijklma", h.v.oovv, tau68)
    )

    tau83 = (
        - einsum("lmab,mlkijc->ijkabc", tau79, tau82)
    )

    tau84 = (
        einsum("ilmj,lkmabc->ijkabc", tau18, tau68)
    )

    tau85 = (
        einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau86 = (
        einsum("bakj,jkic->iabc", a.t2, h.v.ooov)
    )

    tau87 = (
        einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau88 = (
        einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
    )

    tau89 = (
        einsum("bakj,ijkc->iabc", a.t2, tau23)
    )

    tau90 = (
        einsum("jkdc,dabjik->iabc", tau62, a.t3)
    )

    tau91 = (
        einsum("jidc,jadb->iabc", tau28, h.v.ovvv)
    )

    tau92 = (
        3 * einsum("abic->iabc", h.v.vvov)
        + 3 * einsum("ibac->iabc", tau59)
        + 3 * einsum("iabc->iabc", tau86)
        - 3 * einsum("iabc->iabc", tau87)
        - 3 * einsum("ibac->iabc", tau88)
        + 3 * einsum("ibac->iabc", tau89)
        - einsum("iabc->iabc", tau90)
        + 3 * einsum("ibca->iabc", tau91)
    )

    tau93 = (
        - einsum("abji->ijab", a.t2)
        + einsum("baji->ijab", a.t2)
    )

    tau94 = (
        einsum("jidc,jabd->iabc", tau85, h.v.ovvv)
    )

    tau95 = (
        einsum("jicd,jadb->iabc", tau8, h.v.ovvv)
    )

    tau96 = (
        einsum("jicd,jabd->iabc", tau93, h.v.ovvv)
    )

    tau97 = (
        einsum("ijka->ijka", h.v.ooov)
        - einsum("kija->ijka", tau23)
    )

    tau98 = (
        3 * einsum("abic->iabc", h.v.vvov)
        - 3 * einsum("baic->iabc", h.v.vvov)
        - einsum("jkdc,jikdba->iabc", h.v.oovv, tau73)
        + 3 * einsum("ibca->iabc", tau91)
        + 3 * einsum("ibca->iabc", tau94)
        - 3 * einsum("iacb->iabc", tau95)
        - 3 * einsum("iacb->iabc", tau96)
        - 3 * einsum("jkab,kjic->iabc", tau93, tau97)
        - 3 * einsum("di,bacd->iabc", a.t1, tau72)
    )

    tau99 = (
        - einsum("bacd->abcd", h.v.vvvv)
        + einsum("badc->abcd", h.v.vvvv)
    )

    tau100 = (
        - 3 * einsum("abic->iabc", h.v.vvov)
        + 3 * einsum("baic->iabc", h.v.vvov)
        - einsum("jkdc,jikdba->iabc", h.v.oovv, tau0)
        - 3 * einsum("ibca->iabc", tau94)
        - 3 * einsum("ibca->iabc", tau91)
        + 3 * einsum("iacb->iabc", tau96)
        + 3 * einsum("iacb->iabc", tau95)
        - 3 * einsum("jkab,kjic->iabc", tau85, tau97)
        - 3 * einsum("di,bacd->iabc", a.t1, tau99)
    )

    tau101 = (
        einsum("iacd,jkldbc->ijklab", h.v.ovvv, tau53)
    )

    tau102 = (
        einsum("iacd,jklbdc->ijklab", h.v.ovvv, tau53)
    )

    tau103 = (
        einsum("imjc,mklcab->ijklab", tau22, tau76)
    )

    tau104 = (
        einsum("bi,jkab->ijka", a.t1, tau62)
    )

    tau105 = (
        - einsum("ijka->ijka", h.v.ooov)
        + einsum("jika->ijka", h.v.ooov)
        - einsum("kjia->ijka", tau104)
    )

    tau106 = (
        einsum("mklc,mijcab->ijklab", tau105, tau76)
    )

    tau107 = (
        einsum("mijc,mklabc->ijklab", tau24, tau73)
    )

    tau108 = (
        einsum("ijac,klbc->ijklab", tau30, tau85)
    )

    tau109 = (
        einsum("caij,klbc->ijklab", a.t2, tau71)
    )

    tau110 = (
        einsum("ic,jklcab->ijklab", tau7, tau76)
    )

    tau111 = (
        einsum("acij,klbc->ijklab", a.t2, tau34)
    )

    tau112 = (
        einsum("abij,klab->ijkl", a.t2, tau20)
    )

    tau113 = (
        einsum("ai,jkla->ijkl", a.t1, tau22)
    )

    tau114 = (
        einsum("ijka->ijka", h.v.ooov)
        - einsum("jika->ijka", h.v.ooov)
    )

    tau115 = (
        einsum("ai,jkla->ijkl", a.t1, tau114)
    )

    tau116 = (
        - einsum("jikl->ijkl", h.v.oooo)
        + einsum("jilk->ijkl", h.v.oooo)
        - einsum("lkji->ijkl", tau112)
        - einsum("kijl->ijkl", tau113)
        - einsum("ljik->ijkl", tau115)
    )

    tau117 = (
        einsum("abmi,mjkl->ijklab", a.t2, tau116)
    )

    tau118 = (
        einsum("ijkb,lmba->ijklma", tau45, tau85)
    )

    tau119 = (
        einsum("baij,klmb->ijklma", a.t2, tau45)
    )

    tau120 = (
        einsum("abij,klmb->ijklma", a.t2, tau45)
    )

    tau121 = (
        - einsum("jklmia->ijklma", tau118)
        + einsum("jimlka->ijklma", tau119)
        - einsum("mjilka->ijklma", tau120)
    )

    tau122 = (
        einsum("am,ijkmlb->ijklab", a.t1, tau121)
    )

    tau123 = (
        einsum("jkia->ijka", h.v.ooov)
        - einsum("kjia->ijka", h.v.ooov)
        - einsum("ikja->ijka", tau21)
    )

    tau124 = (
        - einsum("jkia->ijka", h.v.ooov)
        + einsum("kjia->ijka", h.v.ooov)
        - einsum("ikja->ijka", tau104)
    )

    tau125 = (
        - einsum("bami,jlkb->ijklma", a.t2, tau123)
        - einsum("baji,mlkb->ijklma", a.t2, tau124)
    )

    tau126 = (
        einsum("abmi,mjkl->ijklab", a.t2, tau51)
    )

    tau127 = (
        - einsum("iljkba->ijklab", tau101)
        - einsum("iljkab->ijklab", tau102)
        + einsum("ikjlab->ijklab", tau103)
        - einsum("jkilab->ijklab", tau106)
        - einsum("ijklab->ijklab", tau107)
        - 3 * einsum("jilkab->ijklab", tau108)
        + 3 * einsum("kjliab->ijklab", tau109)
        - 3 * einsum("ljkiab->ijklab", tau109)
        - einsum("iljkab->ijklab", tau110)
        + 3 * einsum("ljkiba->ijklab", tau111)
        - 3 * einsum("kjliba->ijklab", tau111)
        - 3 * einsum("jilkba->ijklab", tau117)
        + 3 * einsum("kjilab->ijklab", tau122)
        + 3 * einsum("bm,jlimka->ijklab", a.t1, tau125)
        + 3 * einsum("kijlab->ijklab", tau126)
        - 3 * einsum("lijkab->ijklab", tau126)
    )

    tau128 = (
        einsum("jkda,jikbdc->iabc", h.v.oovv, tau68)
    )

    tau129 = (
        3 * einsum("abic->iabc", h.v.vvov)
        + 3 * einsum("ibac->iabc", tau59)
        + 3 * einsum("iabc->iabc", tau86)
        - 3 * einsum("iabc->iabc", tau87)
        - 3 * einsum("ibac->iabc", tau88)
        + 3 * einsum("ibac->iabc", tau89)
        - einsum("icab->iabc", tau128)
        + 3 * einsum("ibca->iabc", tau91)
    )

    tau130 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau129)
    )

    tau131 = (
        einsum("jkda,jikdbc->iabc", h.v.oovv, tau25)
    )

    tau132 = (
        3 * einsum("baic->iabc", h.v.vvov)
        + 3 * einsum("iabc->iabc", tau59)
        + 3 * einsum("ibac->iabc", tau86)
        - 3 * einsum("iabc->iabc", tau88)
        - 3 * einsum("ibac->iabc", tau87)
        + 3 * einsum("iabc->iabc", tau89)
        - einsum("icba->iabc", tau131)
        + 3 * einsum("iacb->iabc", tau91)
    )

    tau133 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau132)
    )

    tau134 = (
        - einsum("iljkba->ijklab", tau101)
        - einsum("iljkab->ijklab", tau102)
        + einsum("ikjlab->ijklab", tau103)
        - einsum("jkilab->ijklab", tau106)
        - einsum("ijklab->ijklab", tau107)
        - 3 * einsum("jilkab->ijklab", tau108)
        + 3 * einsum("kjliab->ijklab", tau109)
        - 3 * einsum("ljkiab->ijklab", tau109)
        - einsum("iljkab->ijklab", tau110)
        + 3 * einsum("ljkiba->ijklab", tau111)
        - 3 * einsum("kjliba->ijklab", tau111)
        - 3 * einsum("jilkba->ijklab", tau117)
        + 3 * einsum("kjilab->ijklab", tau122)
        + 3 * einsum("kijlab->ijklab", tau126)
        - 3 * einsum("lijkab->ijklab", tau126)
    )

    tau135 = (
        einsum("caij,klbc->ijklab", a.t2, tau30)
    )

    tau136 = (
        - einsum("ibcd,ljkcda->ijklab", h.v.ovvv, tau73)
        - einsum("iacd,ljkcbd->ijklab", h.v.ovvv, tau73)
        - einsum("milc,mjkcba->ijklab", tau22, tau0)
        - einsum("mijc,mklbac->ijklab", tau24, tau25)
        - einsum("mikc,mjlbac->ijklab", tau24, tau68)
        - 3 * einsum("libc,kjca->ijklab", tau34, tau93)
        - 3 * einsum("liac,kjcb->ijklab", tau34, tau85)
        - einsum("ic,ljkcba->ijklab", tau7, tau73)
        + 3 * einsum("lkjiab->ijklab", tau135)
        + 3 * einsum("ljkiba->ijklab", tau135)
        - 3 * einsum("ljkiab->ijklab", tau135)
        - 3 * einsum("lkjiba->ijklab", tau135)
        - 3 * einsum("mijl,mkba->ijklab", tau51, tau93)
        - 3 * einsum("mikl,mjba->ijklab", tau51, tau85)
    )

    tau137 = (
        einsum("lkcb,baclij->ijka", tau62, a.t3)
    )

    tau138 = (
        3 * einsum("kaji->ijka", h.v.ovoo)
        + 3 * einsum("jkia->ijka", tau42)
        + 3 * einsum("jika->ijka", tau43)
        - einsum("ijka->ijka", tau137)
        + 3 * einsum("ijka->ijka", tau56)
        - 3 * einsum("ijka->ijka", tau46)
        - 3 * einsum("jika->ijka", tau47)
        + 3 * einsum("jika->ijka", tau48)
        + 3 * einsum("ikja->ijka", tau50)
    )

    tau139 = (
        einsum("libc,ljkbac->ijka", h.v.oovv, tau73)
    )

    tau140 = (
        einsum("jkcb,iabc->ijka", tau93, h.v.ovvv)
    )

    tau141 = (
        einsum("abli,jklb->ijka", a.t2, tau124)
    )

    tau142 = (
        einsum("abli,jlkb->ijka", a.t2, tau123)
    )

    tau143 = (
        einsum("ib,jkab->ijka", tau7, tau93)
    )

    tau144 = (
        einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
        - einsum("jiab->ijab", tau66)
    )

    tau145 = (
        einsum("bi,jkab->ijka", a.t1, tau144)
    )

    tau146 = (
        einsum("iabj->ijab", h.v.ovvo)
        - einsum("iajb->ijab", h.v.ovov)
    )

    tau147 = (
        einsum("bi,jkab->ijka", a.t1, tau146)
    )

    tau148 = (
        - 3 * einsum("jaik->ijka", h.v.ovoo)
        + 3 * einsum("jaki->ijka", h.v.ovoo)
        - einsum("jika->ijka", tau139)
        - 3 * einsum("jkia->ijka", tau140)
        + 3 * einsum("ikja->ijka", tau44)
        + 3 * einsum("ikja->ijka", tau141)
        - 3 * einsum("kija->ijka", tau56)
        - 3 * einsum("kija->ijka", tau142)
        - 3 * einsum("jkia->ijka", tau143)
        - 3 * einsum("ijka->ijka", tau145)
        + 3 * einsum("kjia->ijka", tau147)
    )

    tau149 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau148)
    )

    tau150 = (
        einsum("libc,ljkabc->ijka", h.v.oovv, tau25)
    )

    tau151 = (
        3 * einsum("jaki->ijka", h.v.ovoo)
        + 3 * einsum("kjia->ijka", tau42)
        + 3 * einsum("kija->ijka", tau43)
        - einsum("jika->ijka", tau150)
        + 3 * einsum("ikja->ijka", tau44)
        - 3 * einsum("ikja->ijka", tau46)
        - 3 * einsum("kija->ijka", tau47)
        + 3 * einsum("kija->ijka", tau48)
        + 3 * einsum("ijka->ijka", tau50)
    )

    tau152 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau151)
    )

    r1 = (
        einsum("ia->ai", h.f.ov.conj())
        + einsum("jkbc,jikbac->ai", h.v.oovv, tau0) / 3
        + einsum("bcji,jacb->ai", a.t2, tau1)
        - einsum("bajk,ikjb->ai", a.t2, tau4)
        + einsum("jb,jiab->ai", tau7, tau8)
        + einsum("bj,jiab->ai", a.t1, tau9)
        + einsum("bi,ab->ai", a.t1, tau12)
        - einsum("aj,ji->ai", a.t1, tau17)
    )

    r2 = (
        einsum("dcij,abdc->abij", a.t2, h.v.vvvv)
        - einsum("kbcd,acdkij->abij", h.v.ovvv, a.t3) / 3
        + einsum("kbdc,dackij->abij", h.v.ovvv, a.t3) / 3
        + einsum("ci,bajc->abij", a.t1, h.v.vvov)
        + einsum("baji->abij", h.v.vvoo)
        + einsum("balk,jkli->abij", a.t2, tau18)
        + einsum("kadc,cdbkij->abij", tau19, a.t3) / 3
        + einsum("klic,cbakjl->abij", tau22, a.t3) / 3
        + einsum("lkjc,kilacb->abij", tau24, tau25) / 3
        + einsum("kjcb,ikac->abij", tau28, tau30)
        + einsum("kc,kijcab->abij", tau7, tau25) / 3
        + einsum("caki,jkbc->abij", a.t2, tau31)
        - einsum("cbkj,ikac->abij", a.t2, tau34)
        - einsum("ackj,ikbc->abij", a.t2, tau34)
        - einsum("acki,jkbc->abij", a.t2, tau35)
        - einsum("bcki,jkac->abij", a.t2, tau36)
        + einsum("abkl,klij->abij", a.t2, tau39)
        + einsum("bc,caji->abij", tau41, a.t2)
        + einsum("ac,bcji->abij", tau41, a.t2)
        + einsum("ak,kjib->abij", a.t1, tau52) / 3
        - einsum("bk,kjia->abij", a.t1, tau58) / 3
        + einsum("cj,iabc->abij", a.t1, tau60)
        - einsum("jiab->abij", tau61)
        - einsum("ijba->abij", tau61)
    )

    r3 = (
        - einsum("liad,ljkdbc->abcijk", tau67, tau68) / 3
        - einsum("ikjacb->abcijk", tau69) / 3
        - einsum("klcd,lijdba->abcijk", tau71, tau0) / 3
        + einsum("ikjbca->abcijk", tau69) / 3
        + einsum("jkiacb->abcijk", tau69) / 3
        + einsum("bade,cdekij->abcijk", tau72, a.t3) / 3
        + einsum("cade,kijdeb->abcijk", h.v.vvvv, tau0) / 3
        + einsum("cbde,kijdea->abcijk", h.v.vvvv, tau73) / 3
        + einsum("klad,lijcbd->abcijk", tau34, tau0) / 3
        + einsum("klbd,lijcad->abcijk", tau34, tau73) / 3
        + einsum("ijkcab->abcijk", tau74) / 3
        + einsum("jikcba->abcijk", tau74) / 3
        + einsum("lmji,lkmbca->abcijk", tau75, tau76) / 3
        + einsum("jkicab->abcijk", tau78) / 3
        + einsum("ikjcba->abcijk", tau78) / 3
        + einsum("lmab,kjilmc->abcijk", tau80, tau81) / 3
        + einsum("jkicab->abcijk", tau83) / 3
        + einsum("ikjcba->abcijk", tau83) / 3
        + einsum("kijbac->abcijk", tau84) / 3
        + einsum("kjiabc->abcijk", tau84) / 3
        - einsum("jida,kcbd->abcijk", tau85, tau92) / 3
        - einsum("kcad,jidb->abcijk", tau92, tau93) / 3
        - einsum("cdki,jabd->abcijk", a.t2, tau98) / 3
        - einsum("cdkj,iabd->abcijk", a.t2, tau100) / 3
        - einsum("al,lkijcb->abcijk", a.t1, tau127) / 3
        + einsum("kijabc->abcijk", tau130) / 3
        + einsum("kjibca->abcijk", tau133) / 3
        - einsum("kjiacb->abcijk", tau133) / 3
        - einsum("kijbac->abcijk", tau130) / 3
        + einsum("ad,kijcdb->abcijk", tau41, tau0) / 3
        + einsum("bd,kijcda->abcijk", tau41, tau73) / 3
        + einsum("cd,kijdba->abcijk", tau41, tau73) / 3
        - einsum("bl,lkjica->abcijk", a.t1, tau134) / 3
        - einsum("cl,ljikab->abcijk", a.t1, tau136) / 3
        - einsum("kilc,ljba->abcijk", tau138, tau85) / 3
        - einsum("kjlc,liba->abcijk", tau138, tau93) / 3
        - einsum("kjiacb->abcijk", tau149) / 3
        - einsum("kijbca->abcijk", tau149) / 3
        + einsum("jikcab->abcijk", tau152) / 3
        + einsum("ijkcba->abcijk", tau152) / 3
        - einsum("ijkcab->abcijk", tau152) / 3
        - einsum("jikcba->abcijk", tau152) / 3
        + einsum("li,ljkabc->abcijk", tau17, tau68) / 3
        + einsum("lj,likabc->abcijk", tau17, tau25) / 3
        + einsum("lk,lijcab->abcijk", tau17, tau73) / 3
    )
    return Tensors(t1=r1, t2=r2, t3=r3)
