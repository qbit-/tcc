from numpy import einsum
from tcc.tensors import Tensors


def _rccsdt_ri_calculate_energy(h, a):
    tau0 = (
        einsum("ai,sia->s", a.t1, h.l.pov)
    )

    tau1 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau2 = (
        2 * einsum("s,ai->ias", tau0, a.t1)
        + einsum("sjb,jiba->ias", h.l.pov, tau1)
    )

    tau3 = (
        einsum("ai,sja->ijs", a.t1, h.l.pov)
    )

    tau4 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("jis,sja->ia", tau3, h.l.pov)
    )

    energy = (
        einsum("ias,sia->", tau2, h.l.pov)
        + einsum("ai,ia->", a.t1, tau4)
    )

    return energy


def _rccsdt_ri_calc_residuals(h, a):
    tau0 = (
        einsum("sij,ska->ijka", h.l.poo, h.l.pov)
    )

    tau1 = (
        einsum("ai,sia->s", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("sia,sjb->ijab", h.l.pov, h.l.pov)
    )

    tau3 = (
        2 * einsum("acbkij->ijkabc", a.t3)
        + einsum("backij->ijkabc", a.t3)
        - 2 * einsum("bcakij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau4 = (
        einsum("ai,sja->ijs", a.t1, h.l.pov)
    )

    tau5 = (
        einsum("iks,sja->ijka", tau4, h.l.pov)
    )

    tau6 = (
        einsum("ai,sja->ijs", a.t1, h.l.pov)
    )

    tau7 = (
        einsum("sij->ijs", h.l.poo)
        + einsum("jis->ijs", tau6)
    )

    tau8 = (
        einsum("jks,sia->ijka", tau7, h.l.pov)
    )

    tau9 = (
        - einsum("ijka->ijka", tau5)
        + 2 * einsum("kjia->ijka", tau8)
    )

    tau10 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau11 = (
        einsum("sjb,jiba->ias", h.l.pov, tau10)
    )

    tau12 = (
        einsum("sij->ijs", h.l.poo)
        + einsum("jis->ijs", tau4)
    )

    tau13 = (
        - einsum("ias->ias", tau11)
        + einsum("aj,jis->ias", a.t1, tau12)
    )

    tau14 = (
        einsum("jis,sja->ia", tau4, h.l.pov)
    )

    tau15 = (
        einsum("ai,sia->s", a.t1, h.l.pov)
    )

    tau16 = (
        einsum("s,sia->ia", tau15, h.l.pov)
    )

    tau17 = (
        einsum("ia->ia", h.f.ov)
        - einsum("ia->ia", tau14)
        + 2 * einsum("ia->ia", tau16)
    )

    tau18 = (
        einsum("s,sab->ab", tau1, h.l.pvv)
    )

    tau19 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau18)
    )

    tau20 = (
        einsum("s,sij->ij", tau1, h.l.poo)
    )

    tau21 = (
        einsum("kjs,ski->ij", tau4, h.l.poo)
    )

    tau22 = (
        einsum("jas,sia->ij", tau11, h.l.pov)
    )

    tau23 = (
        einsum("s,sia->ia", tau1, h.l.pov)
    )

    tau24 = (
        einsum("jis,sja->ia", tau6, h.l.pov)
    )

    tau25 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau23)
        - einsum("ia->ia", tau24)
    )

    tau26 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau20)
        - einsum("ji->ij", tau21)
        + einsum("ij->ij", tau22)
        + einsum("aj,ia->ij", a.t1, tau25)
    )

    tau27 = (
        einsum("ijs,ska->ijka", tau6, h.l.pov)
    )

    tau28 = (
        einsum("sia,sjk->ijka", h.l.pov, h.l.poo)
    )

    tau29 = (
        einsum("sab,scd->abcd", h.l.pvv, h.l.pvv)
    )

    tau30 = (
        einsum("sab,sic->iabc", h.l.pvv, h.l.pov)
    )

    tau31 = (
        einsum("bcakij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
        - 2 * einsum("cbakij->ijkabc", a.t3)
    )

    tau32 = (
        einsum("abckij->ijkabc", a.t3)
        - 2 * einsum("acbkij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau33 = (
        2 * einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau34 = (
        einsum("jks,sia->ijka", tau12, h.l.pov)
    )

    tau35 = (
        einsum("abckij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau36 = (
        einsum("ijs,sab->ijab", tau12, h.l.pvv)
    )

    tau37 = (
        einsum("kica,jkcb->ijab", tau10, tau2)
        + einsum("jiab->ijab", tau36)
    )

    tau38 = (
        einsum("acki,kjbc->ijab", a.t2, tau2)
    )

    tau39 = (
        - einsum("ijab->ijab", tau38)
        + einsum("jiab->ijab", tau36)
    )

    tau40 = (
        einsum("baji,klab->ijkl", a.t2, tau2)
    )

    tau41 = (
        einsum("kls,ijs->ijkl", tau12, tau7)
    )

    tau42 = (
        einsum("jlik->ijkl", tau40)
        + einsum("ijkl->ijkl", tau41)
    )

    tau43 = (
        einsum("bi,sab->ias", a.t1, h.l.pvv)
    )

    tau44 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau43)
    )

    tau45 = (
        einsum("bi,sab->ias", a.t1, h.l.pvv)
    )

    tau46 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau45)
        + einsum("ias->ias", tau11)
    )

    tau47 = (
        einsum("sjb,jiba->ias", h.l.pov, tau10)
    )

    tau48 = (
        einsum("aj,jis->ias", a.t1, tau7)
    )

    tau49 = (
        - einsum("ias->ias", tau47)
        + einsum("ias->ias", tau48)
    )

    tau50 = (
        einsum("aj,ijs->ias", a.t1, tau6)
    )

    tau51 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau43)
        - einsum("ias->ias", tau50)
    )

    tau52 = (
        einsum("sjb,baji->ias", h.l.pov, a.t2)
    )

    tau53 = (
        einsum("ias,sib->ab", tau52, h.l.pov)
    )

    tau54 = (
        einsum("sjb,abji->ias", h.l.pov, a.t2)
    )

    tau55 = (
        einsum("ias->ias", tau43)
        - einsum("ias->ias", tau54)
    )

    tau56 = (
        einsum("ibs,sia->ab", tau55, h.l.pov)
    )

    tau57 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau18)
        - 2 * einsum("ab->ab", tau53)
        - einsum("ba->ab", tau56)
    )

    tau58 = (
        einsum("cbji,kabc->ijka", a.t2, tau30)
    )

    tau59 = (
        einsum("ilbc,ljkacb->ijka", tau2, tau31)
    )

    tau60 = (
        einsum("bali,kljb->ijka", a.t2, tau34)
    )

    tau61 = (
        einsum("kija->ijka", tau0)
        + einsum("ijka->ijka", tau5)
    )

    tau62 = (
        einsum("abli,jklb->ijka", a.t2, tau61)
    )

    tau63 = (
        einsum("ijs,kas->ijka", tau12, tau44)
    )

    tau64 = (
        einsum("kb,baij->ijka", tau25, a.t2)
    )

    tau65 = (
        einsum("kas,sij->ijka", tau47, h.l.poo)
    )

    tau66 = (
        - einsum("jkia->ijka", tau58)
        + einsum("ijka->ijka", tau59)
        + einsum("jkia->ijka", tau60)
        + einsum("kjia->ijka", tau62)
        - einsum("ikja->ijka", tau63)
        - einsum("kjia->ijka", tau64)
        - einsum("ikja->ijka", tau65)
        + einsum("al,ljik->ijka", a.t1, tau42)
    )

    tau67 = (
        einsum("bali,jklb->ijka", a.t2, tau61)
    )

    tau68 = (
        einsum("abli,kljb->ijka", a.t2, tau34)
    )

    tau69 = (
        einsum("jkia->ijka", tau58)
        - einsum("ijka->ijka", tau59)
        - einsum("jkia->ijka", tau67)
        - einsum("kjia->ijka", tau68)
        + einsum("ikja->ijka", tau63)
        + einsum("kjia->ijka", tau64)
    )

    tau70 = (
        einsum("s,sij->ij", tau15, h.l.poo)
    )

    tau71 = (
        einsum("aj,sji->ias", a.t1, h.l.poo)
        - einsum("ias->ias", tau11)
    )

    tau72 = (
        einsum("ai,ja->ij", a.t1, tau17)
    )

    tau73 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau70)
        - einsum("jas,sia->ij", tau71, h.l.pov)
        + einsum("ji->ij", tau72)
    )

    tau74 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau20)
        - einsum("ji->ij", tau21)
        + einsum("ij->ij", tau22)
        + einsum("ji->ij", tau72)
    )

    tau75 = (
        einsum("backij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau76 = (
        - einsum("backij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau77 = (
        einsum("kica,kjbc->ijab", tau10, tau2)
        + einsum("jiab->ijab", tau36)
    )

    tau78 = (
        - einsum("jiab->ijab", tau38)
        + einsum("ijab->ijab", tau36)
    )

    tau79 = (
        - einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau80 = (
        einsum("liad,ljkbdc->ijkabc", tau78, tau79)
    )

    tau81 = (
        einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau82 = (
        einsum("caki,kjbc->ijab", a.t2, tau2)
    )

    tau83 = (
        - einsum("acbkij->ijkabc", a.t3)
        - einsum("backij->ijkabc", a.t3)
        + einsum("bcakij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau84 = (
        einsum("ilad,ljkbcd->ijkabc", tau82, tau32)
    )

    tau85 = (
        einsum("ilad,ljkbdc->ijkabc", tau82, tau31)
    )

    tau86 = (
        einsum("jls,iks->ijkl", tau4, tau6)
    )

    tau87 = (
        einsum("kls,sij->ijkl", tau7, h.l.poo)
    )

    tau88 = (
        einsum("jlik->ijkl", tau40)
        + einsum("jlik->ijkl", tau86)
        + einsum("klij->ijkl", tau87)
    )

    tau89 = (
        einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau90 = (
        einsum("ijab,klba->ijkl", tau2, tau89)
    )

    tau91 = (
        - einsum("ljik->ijkl", tau86)
        - einsum("iklj->ijkl", tau90)
        + einsum("ijkl->ijkl", tau41)
    )

    tau92 = (
        einsum("ijs,skl->ijkl", tau6, h.l.poo)
    )

    tau93 = (
        einsum("kls,sij->ijkl", tau12, h.l.poo)
    )

    tau94 = (
        einsum("jikl->ijkl", tau92)
        + einsum("ijkl->ijkl", tau93)
    )

    tau95 = (
        einsum("baji->ijab", a.t2)
        + einsum("ai,bj->ijab", a.t1, a.t1)
    )

    tau96 = (
        - einsum("acbkij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
    )

    tau97 = (
        - einsum("lmbc,kijbac->ijklma", tau2, tau96)
    )

    tau98 = (
        einsum("ijbc,klmcba->ijklma", tau2, tau75)
    )

    tau99 = (
        - einsum("lmab,lmkijc->ijkabc", tau95, tau98)
    )

    tau100 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau43)
        - einsum("ias->ias", tau50)
        + einsum("ias->ias", tau47)
    )

    tau101 = (
        - einsum("abckij->ijkabc", a.t3)
        + einsum("acbkij->ijkabc", a.t3)
        + 2 * einsum("backij->ijkabc", a.t3)
        - einsum("bcakij->ijkabc", a.t3)
        - 2 * einsum("cabkij->ijkabc", a.t3)
        + einsum("cbakij->ijkabc", a.t3)
    )

    tau102 = (
        - einsum("skc,kijcab->ijabs", h.l.pov, tau101)
    )

    tau103 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau43)
        + einsum("ias->ias", tau47)
        - einsum("ias->ias", tau48)
    )

    tau104 = (
        einsum("skc,kijacb->ijabs", h.l.pov, tau31)
    )

    tau105 = (
        einsum("ias,jkbcs->ijkabc", tau103, tau104)
    )

    tau106 = (
        einsum("ias,jkbcs->ijkabc", tau100, tau104)
    )

    tau107 = (
        - einsum("abji->ijab", a.t2)
        + einsum("baji->ijab", a.t2)
    )

    tau108 = (
        einsum("daji,jbdc->iabc", a.t2, tau30)
    )

    tau109 = (
        einsum("adji,jbdc->iabc", a.t2, tau30)
    )

    tau110 = (
        einsum("jkcd,bdakij->iabc", tau2, a.t3)
    )

    tau111 = (
        2 * einsum("ijab->ijab", tau2)
        - einsum("ijba->ijab", tau2)
    )

    tau112 = (
        einsum("kjcd,dabjik->iabc", tau111, a.t3)
    )

    tau113 = (
        einsum("abjk,kjic->iabc", a.t2, tau34)
    )

    tau114 = (
        einsum("ics,sab->iabc", tau46, h.l.pvv)
    )

    tau115 = (
        - einsum("iabc->iabc", tau108)
        - einsum("ibac->iabc", tau109)
        + einsum("ibac->iabc", tau110)
        - einsum("iabc->iabc", tau112)
        + einsum("iabc->iabc", tau113)
        + einsum("ibca->iabc", tau114)
    )

    tau116 = (
        einsum("bakj,jkic->iabc", a.t2, tau28)
    )

    tau117 = (
        einsum("ijab->ijab", tau2)
        - einsum("ijba->ijab", tau2)
    )

    tau118 = (
        einsum("jicd,jadb->iabc", tau107, tau30)
    )

    tau119 = (
        einsum("jadb,jidc->iabc", tau30, tau89)
    )

    tau120 = (
        - einsum("kija->ijka", tau27)
        + einsum("ijka->ijka", tau34)
    )

    tau121 = (
        einsum("ics,sab->iabc", tau11, h.l.pvv)
    )

    tau122 = (
        einsum("ics,sab->iabc", tau44, h.l.pvv)
    )

    tau123 = (
        - einsum("ibac->iabc", tau116)
        + einsum("kjcd,jikdba->iabc", tau111, tau75)
        + einsum("kjcd,bdajik->iabc", tau117, a.t3)
        - einsum("ibca->iabc", tau118)
        + einsum("iacb->iabc", tau119)
        + einsum("bajk,kjic->iabc", a.t2, tau120)
        + einsum("iacb->iabc", tau114)
        - einsum("ibca->iabc", tau121)
        - einsum("ibca->iabc", tau122)
    )

    tau124 = (
        - einsum("ijab->ijab", tau2)
        + einsum("ijba->ijab", tau2)
    )

    tau125 = (
        - einsum("ijka->ijka", tau5)
        + einsum("kjia->ijka", tau8)
    )

    tau126 = (
        einsum("ackj,jikb->iabc", a.t2, tau0)
        + einsum("kjbd,jikdac->iabc", tau111, tau76)
        - einsum("kjbd,cdajik->iabc", tau124, a.t3)
        - einsum("icba->iabc", tau118)
        + einsum("iabc->iabc", tau119)
        - einsum("cajk,ikjb->iabc", a.t2, tau125)
        + einsum("iabc->iabc", tau114)
        - einsum("icba->iabc", tau121)
        - einsum("icba->iabc", tau122)
    )

    tau127 = (
        einsum("jkdc,bdakij->iabc", tau2, a.t3)
    )

    tau128 = (
        einsum("bakj,ijkc->iabc", a.t2, tau27)
    )

    tau129 = (
        einsum("ibac->iabc", tau116)
        - einsum("iabc->iabc", tau108)
        - einsum("ibac->iabc", tau109)
        + einsum("iabc->iabc", tau127)
        + einsum("iabc->iabc", tau128)
        - einsum("iabc->iabc", tau112)
        + einsum("ibca->iabc", tau121)
        + einsum("ibca->iabc", tau122)
    )

    tau130 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau129)
    )

    tau131 = (
        einsum("abjk,jkic->iabc", a.t2, tau8)
    )

    tau132 = (
        - einsum("iabc->iabc", tau108)
        - einsum("ibac->iabc", tau109)
        + einsum("iabc->iabc", tau127)
        - einsum("iabc->iabc", tau112)
        + einsum("ibac->iabc", tau131)
        + einsum("ibca->iabc", tau121)
        + einsum("ibca->iabc", tau122)
    )

    tau133 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau132)
    )

    tau134 = (
        einsum("ijka->ijka", tau0)
        + einsum("jkia->ijka", tau5)
    )

    tau135 = (
        - einsum("skc,kijcab->ijabs", h.l.pov, tau101)
    )

    tau136 = (
        einsum("acki,jkcb->ijab", a.t2, tau2)
    )

    tau137 = (
        - einsum("ijab->ijab", tau136)
        + einsum("jiab->ijab", tau36)
    )

    tau138 = (
        einsum("caki,jkcb->ijab", a.t2, tau2)
    )

    tau139 = (
        einsum("jbs,sia->ijab", tau11, h.l.pov)
    )

    tau140 = (
        einsum("jbs,sia->ijab", tau44, h.l.pov)
    )

    tau141 = (
        - einsum("ijab->ijab", tau138)
        + einsum("jiba->ijab", tau139)
        + einsum("jiba->ijab", tau140)
    )

    tau142 = (
        einsum("caij,klbc->ijklab", a.t2, tau141)
    )

    tau143 = (
        einsum("jikl->ijkl", tau92)
        + einsum("ljki->ijkl", tau40)
        + einsum("ljki->ijkl", tau86)
        + einsum("ijkl->ijkl", tau93)
    )

    tau144 = (
        einsum("ljki->ijkl", tau40)
        + einsum("ijs,lks->ijkl", tau12, tau6)
        + einsum("klij->ijkl", tau87)
    )

    tau145 = (
        einsum("lmab,jkib->ijklma", tau107, tau34)
    )

    tau146 = (
        einsum("baij,mlkb->ijklma", a.t2, tau8)
    )

    tau147 = (
        einsum("kjia->ijka", tau28)
        + einsum("ijka->ijka", tau27)
    )

    tau148 = (
        einsum("abij,klmb->ijklma", a.t2, tau147)
    )

    tau149 = (
        - einsum("jkmlia->ijklma", tau145)
        + einsum("jilkma->ijklma", tau146)
        - einsum("ljikma->ijklma", tau148)
    )

    tau150 = (
        einsum("am,ijmklb->ijklab", a.t1, tau149)
    )

    tau151 = (
        - einsum("ibcd,ljkdca->ijklab", tau30, tau76)
        - einsum("iacd,ljkdcb->ijklab", tau30, tau75)
        - einsum("mkic,mjlabc->ijklab", tau134, tau79)
        - einsum("imjc,mklbac->ijklab", tau34, tau79)
        - einsum("imlc,mjkcab->ijklab", tau34, tau76)
        - einsum("sil,jkbas->ijklab", h.l.poo, tau135)
        - einsum("kjac,libc->ijklab", tau107, tau137)
        - einsum("kjcb,liac->ijklab", tau107, tau137)
        - einsum("ic,ljkcba->ijklab", tau25, tau76)
        + einsum("ljkiab->ijklab", tau142)
        + einsum("lkjiba->ijklab", tau142)
        - einsum("lkjiab->ijklab", tau142)
        - einsum("ljkiba->ijklab", tau142)
        - einsum("mjba,mkil->ijklab", tau107, tau143)
        - einsum("mjil,mkba->ijklab", tau144, tau89)
        + einsum("jlkiab->ijklab", tau150)
        - einsum("jlkiba->ijklab", tau150)
    )

    tau152 = (
        einsum("iacd,jklcbd->ijklab", tau30, tau96)
    )

    tau153 = (
        einsum("acbkij->ijkabc", a.t3)
        - einsum("cabkij->ijkabc", a.t3)
    )

    tau154 = (
        einsum("iacd,jkldcb->ijklab", tau30, tau153)
    )

    tau155 = (
        einsum("mklc,mijcab->ijklab", tau134, tau96)
    )

    tau156 = (
        einsum("imjc,mklcab->ijklab", tau34, tau153)
    )

    tau157 = (
        einsum("imjc,mklacb->ijklab", tau34, tau75)
    )

    tau158 = (
        einsum("klbc,ijca->ijklab", tau141, tau89)
    )

    tau159 = (
        einsum("ikca,kjbc->ijab", tau2, tau89)
    )

    tau160 = (
        - einsum("jiba->ijab", tau159)
        - einsum("jiab->ijab", tau36)
        + einsum("jiba->ijab", tau139)
        + einsum("jiba->ijab", tau140)
    )

    tau161 = (
        einsum("caij,klbc->ijklab", a.t2, tau160)
    )

    tau162 = (
        einsum("ic,jklabc->ijklab", tau25, tau96)
    )

    tau163 = (
        einsum("skc,kijacb->ijabs", h.l.pov, tau31)
    )

    tau164 = (
        einsum("sij,klabs->ijklab", h.l.poo, tau163)
    )

    tau165 = (
        einsum("acij,klbc->ijklab", a.t2, tau137)
    )

    tau166 = (
        - einsum("jkil->ijkl", tau92)
        - einsum("ljik->ijkl", tau86)
        - einsum("iklj->ijkl", tau90)
        + einsum("ijkl->ijkl", tau41)
        - einsum("kjil->ijkl", tau93)
    )

    tau167 = (
        einsum("abmi,jkml->ijklab", a.t2, tau166)
    )

    tau168 = (
        einsum("kija->ijka", tau0)
        + einsum("ijka->ijka", tau5)
        - einsum("kjia->ijka", tau8)
    )

    tau169 = (
        - einsum("kjia->ijka", tau28)
        - einsum("ijka->ijka", tau27)
        + einsum("jkia->ijka", tau34)
    )

    tau170 = (
        - einsum("bami,jlkb->ijklma", a.t2, tau168)
        + einsum("baji,mlkb->ijklma", a.t2, tau169)
    )

    tau171 = (
        einsum("lkij->ijkl", tau92)
        + einsum("jlik->ijkl", tau40)
        + einsum("jlik->ijkl", tau86)
        + einsum("klij->ijkl", tau87)
    )

    tau172 = (
        einsum("abmi,jkml->ijklab", a.t2, tau171)
    )

    tau173 = (
        einsum("abmi,jkml->ijklab", a.t2, tau42)
    )

    tau174 = (
        - einsum("lkijba->ijklab", tau152)
        - einsum("lkijab->ijklab", tau154)
        - einsum("ijklab->ijklab", tau155)
        - einsum("ljikab->ijklab", tau156)
        - einsum("lijkab->ijklab", tau157)
        - einsum("kjilba->ijklab", tau158)
        - einsum("jiklab->ijklab", tau161)
        + einsum("kijlab->ijklab", tau161)
        - einsum("lkijba->ijklab", tau162)
        + einsum("ljikab->ijklab", tau164)
        + einsum("jiklba->ijklab", tau165)
        - einsum("kijlba->ijklab", tau165)
        - einsum("ilkjba->ijklab", tau167)
        - einsum("bm,ikmlja->ijklab", a.t1, tau170)
        + einsum("kljiab->ijklab", tau172)
        - einsum("jlkiab->ijklab", tau173)
    )

    tau175 = (
        - einsum("lkijba->ijklab", tau152)
        - einsum("lkijab->ijklab", tau154)
        - einsum("ijklab->ijklab", tau155)
        - einsum("ljikab->ijklab", tau156)
        - einsum("lijkab->ijklab", tau157)
        - einsum("kjilba->ijklab", tau158)
        - einsum("jiklab->ijklab", tau161)
        + einsum("kijlab->ijklab", tau161)
        - einsum("lkijba->ijklab", tau162)
        + einsum("ljikab->ijklab", tau164)
        + einsum("jiklba->ijklab", tau165)
        - einsum("kijlba->ijklab", tau165)
        - einsum("ilkjba->ijklab", tau167)
        + einsum("kljiab->ijklab", tau172)
        - einsum("jlkiab->ijklab", tau173)
    )

    tau176 = (
        einsum("ilbc,ljkbac->ijka", tau2, tau32)
    )

    tau177 = (
        einsum("kb,baij->ijka", tau17, a.t2)
    )

    tau178 = (
        einsum("kas,ijs->ijka", tau11, tau6)
    )

    tau179 = (
        einsum("ijka->ijka", tau58)
        - einsum("kija->ijka", tau176)
        - einsum("ijka->ijka", tau60)
        - einsum("jika->ijka", tau68)
        + einsum("kjia->ijka", tau63)
        + einsum("jika->ijka", tau177)
        + einsum("jkia->ijka", tau178)
        + einsum("kjia->ijka", tau65)
    )

    tau180 = (
        einsum("kas,ijs->ijka", tau11, tau7)
    )

    tau181 = (
        einsum("ijka->ijka", tau58)
        - einsum("kija->ijka", tau176)
        - einsum("ijka->ijka", tau67)
        - einsum("jika->ijka", tau68)
        + einsum("kjia->ijka", tau63)
        + einsum("kjia->ijka", tau180)
        + einsum("jika->ijka", tau177)
    )

    tau182 = (
        einsum("abckij->ijkabc", a.t3)
        - 2 * einsum("acbkij->ijkabc", a.t3)
        - einsum("backij->ijkabc", a.t3)
        + 2 * einsum("bcakij->ijkabc", a.t3)
        + einsum("cabkij->ijkabc", a.t3)
        - einsum("cbakij->ijkabc", a.t3)
    )

    tau183 = (
        einsum("ilbc,ljkabc->ijka", tau2, tau182)
    )

    tau184 = (
        einsum("iabc,jkbc->ijka", tau30, tau89)
    )

    tau185 = (
        einsum("sai->ias", h.l.pvo)
        + einsum("ias->ias", tau43)
        + einsum("ias->ias", tau47)
    )

    tau186 = (
        einsum("ijs,kas->ijka", tau12, tau185)
    )

    tau187 = (
        einsum("ib,jkab->ijka", tau17, tau89)
    )

    tau188 = (
        - einsum("jika->ijka", tau183)
        - einsum("jkia->ijka", tau184)
        - einsum("liab,kljb->ijka", tau107, tau147)
        - einsum("lkba,jlib->ijka", tau107, tau8)
        - einsum("jkia->ijka", tau186)
        + einsum("jika->ijka", tau63)
        - einsum("jkia->ijka", tau187)
        + einsum("ijs,kas->ijka", tau4, tau47)
        + einsum("kas,sji->ijka", tau11, h.l.poo)
    )

    tau189 = (
        - einsum("jika->ijka", tau183)
        - einsum("jkia->ijka", tau184)
        - einsum("liab,jlkb->ijka", tau107, tau34)
        - einsum("lkba,ijlb->ijka", tau107, tau61)
        + einsum("jika->ijka", tau63)
        + einsum("jika->ijka", tau180)
        - einsum("jkia->ijka", tau63)
        - einsum("jkia->ijka", tau187)
        - einsum("kjia->ijka", tau178)
        - einsum("jkia->ijka", tau65)
    )

    tau190 = (
        einsum("ikja->ijka", tau58)
        - einsum("jika->ijka", tau59)
        - einsum("ikja->ijka", tau60)
        - einsum("kija->ijka", tau68)
        + einsum("jkia->ijka", tau63)
        + einsum("kija->ijka", tau177)
        + einsum("kjia->ijka", tau178)
        + einsum("jkia->ijka", tau65)
    )

    tau191 = (
        einsum("bali,jlkb->ijka", a.t2, tau147)
    )

    tau192 = (
        einsum("ikja->ijka", tau58)
        - einsum("jika->ijka", tau59)
        - einsum("ikja->ijka", tau191)
        - einsum("ablk,jlib->ijka", a.t2, tau8)
        + einsum("jkia->ijka", tau186)
        + einsum("kija->ijka", tau177)
    )

    tau193 = (
        einsum("ikja->ijka", tau58)
        - einsum("jika->ijka", tau59)
        - einsum("ikja->ijka", tau60)
        - einsum("kija->ijka", tau62)
        + einsum("jkia->ijka", tau63)
        + einsum("kija->ijka", tau177)
        + einsum("kjia->ijka", tau178)
        + einsum("jkia->ijka", tau65)
    )

    tau194 = (
        einsum("ikja->ijka", tau58)
        - einsum("jika->ijka", tau59)
        - einsum("ikja->ijka", tau191)
        - einsum("ablk,iljb->ijka", a.t2, tau147)
        + einsum("jkia->ijka", tau186)
        + einsum("kija->ijka", tau177)
    )

    tau195 = (
        einsum("kis,skj->ij", tau6, h.l.poo)
    )

    tau196 = (
        einsum("jas,sia->ij", tau47, h.l.pov)
    )

    tau197 = (
        einsum("ij->ij", h.f.oo)
        - einsum("ij->ij", tau195)
        + 2 * einsum("ij->ij", tau70)
        + einsum("ij->ij", tau196)
        + einsum("ji->ij", tau72)
    )

    tau198 = (
        einsum("aj,sji->ias", a.t1, h.l.poo)
        - einsum("ias->ias", tau47)
    )

    tau199 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau20)
        - einsum("jas,sia->ij", tau198, h.l.pov)
        + einsum("ji->ij", tau72)
    )

    r1 = (
        einsum("abkj,jikb->ai", a.t2, tau0)
        + einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("s,sai->ai", tau1, h.l.pvo)
        + einsum("jkbc,kijbac->ai", tau2, tau3)
        - einsum("bajk,ikjb->ai", a.t2, tau9)
        - einsum("ibs,sab->ai", tau13, h.l.pvv)
        + einsum("jb,jiba->ai", tau17, tau10)
        + einsum("bi,ab->ai", a.t1, tau19)
        - einsum("aj,ji->ai", a.t1, tau26)
    )

    r2 = (
        einsum("iklc,cbaljk->abij", tau5, a.t3)
        - 2 * einsum("jklc,cablik->abij", tau27, a.t3)
        + einsum("kilc,abcljk->abij", tau0, a.t3)
        - 2 * einsum("kljc,baclik->abij", tau28, a.t3)
        + einsum("dcji,acbd->abij", a.t2, tau29)
        - einsum("kacd,kijcdb->abij", tau30, tau31)
        - einsum("kbcd,kijcad->abij", tau30, tau32)
        - einsum("lkic,ljkcba->abij", tau8, tau33)
        + einsum("lkjc,kilbca->abij", tau34, tau35)
        - einsum("kc,kijacb->abij", tau17, tau31)
        - einsum("caki,jkbc->abij", a.t2, tau37)
        - einsum("cbkj,ikac->abij", a.t2, tau39)
        - einsum("bcki,jkac->abij", a.t2, tau39)
        - einsum("ackj,kibc->abij", a.t2, tau36)
        + einsum("bakl,likj->abij", a.t2, tau42)
        + einsum("ias,jbs->abij", tau44, tau46)
        - einsum("jbs,ias->abij", tau11, tau49)
        + einsum("ias,jbs->abij", tau11, tau51)
        + einsum("ac,bcji->abij", tau57, a.t2)
        + einsum("bc,caji->abij", tau57, a.t2)
        + einsum("bk,kija->abij", a.t1, tau66)
        - einsum("ak,kjib->abij", a.t1, tau69)
        - einsum("kj,baki->abij", tau73, a.t2)
        - einsum("ki,abkj->abij", tau74, a.t2)
    )

    r3 = (
        - einsum("adbe,kijcde->abcijk", tau29, tau75)
        - einsum("adce,kijebd->abcijk", tau29, tau76)
        - einsum("bdce,kijead->abcijk", tau29, tau75)
        - einsum("klcd,lijdab->abcijk", tau77, tau76)
        - einsum("ijkabc->abcijk", tau80)
        - einsum("ilcd,ljkbad->abcijk", tau39, tau79)
        - einsum("jikbac->abcijk", tau80)
        - einsum("jlcd,likbad->abcijk", tau39, tau81)
        - einsum("klad,lijcbd->abcijk", tau39, tau75)
        - einsum("klbd,lijcad->abcijk", tau39, tau76)
        + einsum("libd,ljkdac->abcijk", tau78, tau81)
        + einsum("jikabc->abcijk", tau80)
        - einsum("klcd,lijbda->abcijk", tau82, tau83)
        + einsum("ijkacb->abcijk", tau84)
        + einsum("jikbac->abcijk", tau85)
        - einsum("ijkbca->abcijk", tau84)
        - einsum("jikabc->abcijk", tau85)
        - einsum("limk,ljmabc->abcijk", tau42, tau79)
        - einsum("ljmk,limabc->abcijk", tau88, tau81)
        + einsum("limj,acblkm->abcijk", tau91, a.t3)
        - einsum("limj,bcalkm->abcijk", tau94, a.t3)
        - einsum("lmab,kjilmc->abcijk", tau95, tau97)
        - einsum("jikacb->abcijk", tau99)
        - einsum("ijkbca->abcijk", tau99)
        + einsum("klmj,limcba->abcijk", tau92, tau75)
        - einsum("kcs,jiabs->abcijk", tau100, tau102)
        - einsum("ikjacb->abcijk", tau105)
        + einsum("ikjbca->abcijk", tau105)
        - einsum("jkibca->abcijk", tau106)
        + einsum("jkiacb->abcijk", tau106)
        - einsum("jidb,kcad->abcijk", tau107, tau115)
        - einsum("jiad,kcbd->abcijk", tau107, tau115)
        + einsum("cdki,jabd->abcijk", a.t2, tau123)
        - einsum("cdkj,iadb->abcijk", a.t2, tau126)
        + einsum("kijabc->abcijk", tau130)
        - einsum("kijbac->abcijk", tau130)
        + einsum("kjibac->abcijk", tau133)
        - einsum("kjiabc->abcijk", tau133)
        - einsum("cl,ljikba->abcijk", a.t1, tau151)
        - einsum("ad,kijcdb->abcijk", tau57, tau75)
        - einsum("bd,kijcad->abcijk", tau57, tau75)
        - einsum("cd,kijdba->abcijk", tau57, tau76)
        - einsum("al,kjilcb->abcijk", a.t1, tau174)
        + einsum("bl,kjilca->abcijk", a.t1, tau175)
        + einsum("liab,kjlc->abcijk", tau107, tau179)
        + einsum("ljba,kilc->abcijk", tau107, tau181)
        + einsum("bclk,ilja->abcijk", a.t2, tau188)
        - einsum("aclk,iljb->abcijk", a.t2, tau189)
        - einsum("cali,jlkb->abcijk", a.t2, tau190)
        - einsum("cblj,ilka->abcijk", a.t2, tau192)
        + einsum("calj,ilkb->abcijk", a.t2, tau193)
        + einsum("cbli,jlka->abcijk", a.t2, tau194)
        + einsum("lj,likabc->abcijk", tau197, tau81)
        + einsum("lk,lijcab->abcijk", tau197, tau75)
        + einsum("li,ljkabc->abcijk", tau199, tau79)
    )

    return Tensors(t1=r1, t2=r2, t3=r3)
