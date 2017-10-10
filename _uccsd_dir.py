from numpy import einsum
from .tensors import Tensors


def _uccsd_calculate_energy(h, a):
    tau0 = (
        - einsum("ijab->ijab", h.v.bbbb.oovv)
        + einsum("ijba->ijab", h.v.bbbb.oovv)
        + einsum("jiab->ijab", h.v.bbbb.oovv)
        - einsum("jiba->ijab", h.v.bbbb.oovv)
    )

    tau1 = (
        - einsum("ijab->ijab", h.v.aaaa.oovv)
        + einsum("ijba->ijab", h.v.aaaa.oovv)
        + einsum("jiab->ijab", h.v.aaaa.oovv)
        - einsum("jiba->ijab", h.v.aaaa.oovv)
    )

    tau2 = (
        - einsum("ijab->ijab", h.v.aaaa.oovv)
        + einsum("ijba->ijab", h.v.aaaa.oovv)
    )

    tau3 = (
        2 * einsum("ia->ia", h.f_a.ov)
        + 2 * einsum("bj,ijab->ia", a.t1.b, h.v.abab.oovv)
        - einsum("bj,jiba->ia", a.t1.a, tau2)
    )

    tau4 = (
        - einsum("ijab->ijab", h.v.bbbb.oovv)
        + einsum("jiab->ijab", h.v.bbbb.oovv)
    )

    tau5 = (
        2 * einsum("ia->ia", h.f_b.ov)
        - einsum("bj,ijab->ia", a.t1.b, tau4)
    )

    energy = (
        einsum("baji,ijab->", a.t2.ab, h.v.abab.oovv)
        - einsum("abij,ijab->", a.t2.bb, tau0) / 4
        - einsum("abij,ijab->", a.t2.aa, tau1) / 4
        + einsum("ai,ia->", a.t1.a, tau3) / 2
        + einsum("ai,ia->", a.t1.b, tau5) / 2
    )
    return energy


def _uccsd_calc_residuals(h, a):

    tau0 = (
        - einsum("iabc->iabc", h.v.aaaa.ovvv)
        + einsum("iacb->iabc", h.v.aaaa.ovvv)
    )

    tau1 = (
        einsum("abij->ijab", a.t2.aa)
        - einsum("abji->ijab", a.t2.aa)
    )

    tau2 = (
        - einsum("abij->ijab", a.t2.aa)
        + einsum("baij->ijab", a.t2.aa)
    )

    tau3 = (
        einsum("ijab->ijab", h.v.aaaa.oovv)
        - einsum("ijba->ijab", h.v.aaaa.oovv)
        - einsum("jiab->ijab", h.v.aaaa.oovv)
        + einsum("jiba->ijab", h.v.aaaa.oovv)
    )

    tau4 = (
        2 * einsum("jkia->ijka", h.v.aaaa.ooov)
        - 2 * einsum("kjia->ijka", h.v.aaaa.ooov)
        + einsum("bi,kjab->ijka", a.t1.a, tau3)
    )

    tau5 = (
        einsum("jkia->ijka", h.v.abab.ooov)
        + einsum("bi,jkba->ijka", a.t1.a, h.v.abab.oovv)
    )

    tau6 = (
        einsum("bj,ijab->ia", a.t1.b, h.v.abab.oovv)
    )

    tau7 = (
        einsum("bj,ijab->ia", a.t1.a, tau3)
    )

    tau8 = (
        2 * einsum("ia->ia", h.f_a.ov)
        + 2 * einsum("ia->ia", tau6)
        + einsum("ia->ia", tau7)
    )

    tau9 = (
        einsum("abij->ijab", a.t2.aa)
        - einsum("baij->ijab", a.t2.aa)
        - einsum("abji->ijab", a.t2.aa)
        + einsum("baji->ijab", a.t2.aa)
    )

    tau10 = (
        einsum("bj,jiba->ia", a.t1.a, h.v.abab.oovv)
    )

    tau11 = (
        einsum("ijab->ijab", h.v.bbbb.oovv)
        - einsum("ijba->ijab", h.v.bbbb.oovv)
        - einsum("jiab->ijab", h.v.bbbb.oovv)
        + einsum("jiba->ijab", h.v.bbbb.oovv)
    )

    tau12 = (
        einsum("bj,ijab->ia", a.t1.b, tau11)
    )

    tau13 = (
        2 * einsum("ia->ia", h.f_b.ov)
        + 2 * einsum("ia->ia", tau10)
        + einsum("ia->ia", tau12)
    )

    tau14 = (
        einsum("iabj->ijab", h.v.aaaa.ovvo)
        - einsum("iajb->ijab", h.v.aaaa.ovov)
    )

    tau15 = (
        einsum("iabc->iabc", h.v.aaaa.ovvv)
        - einsum("iacb->iabc", h.v.aaaa.ovvv)
    )

    tau16 = (
        einsum("ab->ab", h.f_a.vv)
        + einsum("ci,aibc->ab", a.t1.b, h.v.abab.vovv)
        - einsum("ci,iabc->ab", a.t1.a, tau15)
    )

    tau17 = (
        - einsum("ijab->ijab", h.v.aaaa.oovv)
        + einsum("ijba->ijab", h.v.aaaa.oovv)
        + einsum("jiab->ijab", h.v.aaaa.oovv)
        - einsum("jiba->ijab", h.v.aaaa.oovv)
    )

    tau18 = (
        einsum("ijka->ijka", h.v.aaaa.ooov)
        - einsum("jika->ijka", h.v.aaaa.ooov)
    )

    tau19 = (
        4 * einsum("ij->ij", h.f_a.oo)
        + 4 * einsum("ak,ikja->ij", a.t1.b, h.v.abab.ooov)
        + 4 * einsum("bakj,ikab->ij", a.t2.ab, h.v.abab.oovv)
        - einsum("kjab,ikba->ij", tau1, tau17)
        - 4 * einsum("ak,kija->ij", a.t1.a, tau18)
        + 2 * einsum("aj,ia->ij", a.t1.a, tau8)
    )

    tau20 = (
        - einsum("iabc->iabc", h.v.bbbb.ovvv)
        + einsum("iacb->iabc", h.v.bbbb.ovvv)
    )

    tau21 = (
        einsum("abij->ijab", a.t2.bb)
        - einsum("abji->ijab", a.t2.bb)
    )

    tau22 = (
        - einsum("abij->ijab", a.t2.bb)
        + einsum("baij->ijab", a.t2.bb)
    )

    tau23 = (
        2 * einsum("jkia->ijka", h.v.bbbb.ooov)
        - 2 * einsum("kjia->ijka", h.v.bbbb.ooov)
        + einsum("bi,kjab->ijka", a.t1.b, tau11)
    )

    tau24 = (
        einsum("jkai->ijka", h.v.abab.oovo)
        + einsum("bi,jkab->ijka", a.t1.b, h.v.abab.oovv)
    )

    tau25 = (
        einsum("abij->ijab", a.t2.bb)
        - einsum("baij->ijab", a.t2.bb)
        - einsum("abji->ijab", a.t2.bb)
        + einsum("baji->ijab", a.t2.bb)
    )

    tau26 = (
        einsum("iabj->ijab", h.v.bbbb.ovvo)
        - einsum("iajb->ijab", h.v.bbbb.ovov)
    )

    tau27 = (
        einsum("iabc->iabc", h.v.bbbb.ovvv)
        - einsum("iacb->iabc", h.v.bbbb.ovvv)
    )

    tau28 = (
        einsum("ab->ab", h.f_b.vv)
        + einsum("ci,iacb->ab", a.t1.a, h.v.abab.ovvv)
        - einsum("ci,iabc->ab", a.t1.b, tau27)
    )

    tau29 = (
        - einsum("ijab->ijab", h.v.bbbb.oovv)
        + einsum("ijba->ijab", h.v.bbbb.oovv)
        + einsum("jiab->ijab", h.v.bbbb.oovv)
        - einsum("jiba->ijab", h.v.bbbb.oovv)
    )

    tau30 = (
        einsum("ijka->ijka", h.v.bbbb.ooov)
        - einsum("jika->ijka", h.v.bbbb.ooov)
    )

    tau31 = (
        4 * einsum("ij->ij", h.f_b.oo)
        + 4 * einsum("ak,kiaj->ij", a.t1.a, h.v.abab.oovo)
        + 4 * einsum("abkj,kiab->ij", a.t2.ab, h.v.abab.oovv)
        - einsum("kjab,ikba->ij", tau21, tau29)
        - 4 * einsum("ak,kija->ij", a.t1.b, tau30)
        + 2 * einsum("aj,ia->ij", a.t1.b, tau13)
    )

    r1_a = (
        einsum("bj,ajib->ai", a.t1.b, h.v.abab.voov)
        + einsum("cbji,ajbc->ai", a.t2.ab, h.v.abab.vovv)
        + einsum("ai->ai", h.f_a.vo)
        + einsum("jabc,jicb->ai", tau0, tau1) / 2
        + einsum("jkba,ikjb->ai", tau2, tau4) / 4
        - einsum("bajk,ikjb->ai", a.t2.ab, tau5)
        + einsum("jb,ijab->ai", tau8, tau9) / 4
        + einsum("jb,baji->ai", tau13, a.t2.ab) / 2
        + einsum("bj,jiab->ai", a.t1.a, tau14)
        + einsum("bi,ab->ai", a.t1.a, tau16)
        - einsum("aj,ji->ai", a.t1.a, tau19) / 4
    )

    r1_b = (
        einsum("bj,jabi->ai", a.t1.a, h.v.abab.ovvo)
        + einsum("bcji,jabc->ai", a.t2.ab, h.v.abab.ovvv)
        + einsum("ai->ai", h.f_b.vo)
        + einsum("jabc,jicb->ai", tau20, tau21) / 2
        + einsum("jkba,ikjb->ai", tau22, tau23) / 4
        - einsum("abjk,ikjb->ai", a.t2.ab, tau24)
        + einsum("jb,ijab->ai", tau13, tau25) / 4
        + einsum("jb,baji->ai", tau8, a.t2.ab) / 2
        + einsum("bj,jiab->ai", a.t1.b, tau26)
        + einsum("bi,ab->ai", a.t1.b, tau28)
        - einsum("aj,ji->ai", a.t1.b, tau31) / 4
    )
    tau0 = (
        einsum("caki,jkbc->ijab", a.t2.ab, h.v.abab.oovv)
    )

    tau1 = (
        - einsum("abij->ijab", a.t2.aa)
        + einsum("baij->ijab", a.t2.aa)
        + einsum("abji->ijab", a.t2.aa)
        - einsum("baji->ijab", a.t2.aa)
    )

    tau2 = (
        einsum("ijab->ijab", h.v.aaaa.oovv)
        - einsum("ijba->ijab", h.v.aaaa.oovv)
        - einsum("jiab->ijab", h.v.aaaa.oovv)
        + einsum("jiba->ijab", h.v.aaaa.oovv)
    )

    tau3 = (
        einsum("ikca,kjcb->ijab", tau1, tau2)
    )

    tau4 = (
        einsum("iabc->iabc", h.v.aaaa.ovvv)
        - einsum("iacb->iabc", h.v.aaaa.ovvv)
    )

    tau5 = (
        einsum("ci,jacb->ijab", a.t1.a, tau4)
    )

    tau6 = (
        4 * einsum("jabi->ijab", h.v.aaaa.ovvo)
        - 4 * einsum("jaib->ijab", h.v.aaaa.ovov)
        + 4 * einsum("ijab->ijab", tau0)
        + einsum("ijab->ijab", tau3)
        - 4 * einsum("ijab->ijab", tau5)
    )

    tau7 = (
        einsum("abij->ijab", a.t2.aa)
        - einsum("baij->ijab", a.t2.aa)
        - einsum("abji->ijab", a.t2.aa)
        + einsum("baji->ijab", a.t2.aa)
    )

    tau8 = (
        einsum("ikac,kjcb->ijab", tau6, tau7)
    )

    tau9 = (
        - einsum("abcd->abcd", h.v.aaaa.vvvv)
        + einsum("abdc->abcd", h.v.aaaa.vvvv)
        + einsum("bacd->abcd", h.v.aaaa.vvvv)
        - einsum("badc->abcd", h.v.aaaa.vvvv)
    )

    tau10 = (
        - einsum("abij->ijab", a.t2.aa)
        + einsum("abji->ijab", a.t2.aa)
    )

    tau11 = (
        einsum("jabi->ijab", h.v.aaaa.ovvo)
        - einsum("jaib->ijab", h.v.aaaa.ovov)
        + einsum("ijab->ijab", tau0)
        - einsum("ijab->ijab", tau5)
    )

    tau12 = (
        einsum("ci,ajcb->ijab", a.t1.a, h.v.abab.vovv)
    )

    tau13 = (
        einsum("ijab->ijab", h.v.bbbb.oovv)
        - einsum("ijba->ijab", h.v.bbbb.oovv)
        - einsum("jiab->ijab", h.v.bbbb.oovv)
        + einsum("jiba->ijab", h.v.bbbb.oovv)
    )

    tau14 = (
        einsum("caki,kjcb->ijab", a.t2.ab, tau13)
    )

    tau15 = (
        2 * einsum("ajib->ijab", h.v.abab.voov)
        + 2 * einsum("ijab->ijab", tau12)
        + einsum("ijab->ijab", tau14)
    )

    tau16 = (
        einsum("caki,jkbc->ijab", a.t2.ab, tau15)
    )

    tau17 = (
        - einsum("abij->ijab", a.t2.aa)
        + einsum("baij->ijab", a.t2.aa)
    )

    tau18 = (
        einsum("abij->ijab", a.t2.aa)
        - einsum("abji->ijab", a.t2.aa)
    )

    tau19 = (
        einsum("bi,jkab->ijka", a.t1.a, tau2)
    )

    tau20 = (
        - 2 * einsum("ijka->ijka", h.v.aaaa.ooov)
        + 2 * einsum("jika->ijka", h.v.aaaa.ooov)
        + einsum("kija->ijka", tau19)
    )

    tau21 = (
        einsum("ijka->ijka", h.v.aaaa.ooov)
        - einsum("jika->ijka", h.v.aaaa.ooov)
    )

    tau22 = (
        2 * einsum("ijkl->ijkl", h.v.aaaa.oooo)
        - 2 * einsum("ijlk->ijkl", h.v.aaaa.oooo)
        - 2 * einsum("jikl->ijkl", h.v.aaaa.oooo)
        + 2 * einsum("jilk->ijkl", h.v.aaaa.oooo)
        - einsum("lkab,ijab->ijkl", tau18, tau2)
        - 2 * einsum("ak,jila->ijkl", a.t1.a, tau20)
        - 4 * einsum("al,jika->ijkl", a.t1.a, tau21)
    )

    tau23 = (
        einsum("ajib->ijab", h.v.abab.voov)
        + einsum("ijab->ijab", tau12)
    )

    tau24 = (
        einsum("caki,jkbc->ijab", a.t2.ab, tau23)
    )

    tau25 = (
        einsum("ci,aibc->ab", a.t1.b, h.v.abab.vovv)
    )

    tau26 = (
        einsum("caji,ijbc->ab", a.t2.ab, h.v.abab.oovv)
    )

    tau27 = (
        einsum("ijcb,ijca->ab", tau17, tau2)
    )

    tau28 = (
        einsum("ci,iabc->ab", a.t1.a, tau4)
    )

    tau29 = (
        4 * einsum("ab->ab", h.f_a.vv)
        + 4 * einsum("ab->ab", tau25)
        - 4 * einsum("ab->ab", tau26)
        + einsum("ba->ab", tau27)
        - 4 * einsum("ab->ab", tau28)
    )

    tau30 = (
        einsum("bc,ijca->ijab", tau29, tau7)
    )

    tau31 = (
        - einsum("iabc->iabc", h.v.aaaa.ovvv)
        + einsum("iacb->iabc", h.v.aaaa.ovvv)
    )

    tau32 = (
        2 * einsum("jkia->ijka", h.v.aaaa.ooov)
        - 2 * einsum("kjia->ijka", h.v.aaaa.ooov)
        + einsum("ikja->ijka", tau19)
    )

    tau33 = (
        einsum("liab,jlkb->ijka", tau1, tau32)
    )

    tau34 = (
        einsum("bi,jkba->ijka", a.t1.a, h.v.abab.oovv)
    )

    tau35 = (
        einsum("jkia->ijka", h.v.abab.ooov)
        + einsum("ijka->ijka", tau34)
    )

    tau36 = (
        einsum("bali,jklb->ijka", a.t2.ab, tau35)
    )

    tau37 = (
        einsum("bj,ijab->ia", a.t1.b, h.v.abab.oovv)
    )

    tau38 = (
        einsum("bj,ijab->ia", a.t1.a, tau2)
    )

    tau39 = (
        2 * einsum("ia->ia", h.f_a.ov)
        + 2 * einsum("ia->ia", tau37)
        + einsum("ia->ia", tau38)
    )

    tau40 = (
        einsum("ci,jabc->ijab", a.t1.a, tau31)
    )

    tau41 = (
        einsum("iabj->ijab", h.v.aaaa.ovvo)
        - einsum("iajb->ijab", h.v.aaaa.ovov)
        - einsum("jiab->ijab", tau40)
    )

    tau42 = (
        einsum("bi,jkab->ijka", a.t1.a, tau41)
    )

    tau43 = (
        - einsum("ijab->ijab", h.v.aaaa.oovv)
        + einsum("ijba->ijab", h.v.aaaa.oovv)
        + einsum("jiab->ijab", h.v.aaaa.oovv)
        - einsum("jiba->ijab", h.v.aaaa.oovv)
    )

    tau44 = (
        2 * einsum("ijka->ijka", h.v.aaaa.ooov)
        - 2 * einsum("jika->ijka", h.v.aaaa.ooov)
        + einsum("bk,ijab->ijka", a.t1.a, tau43)
    )

    tau45 = (
        - einsum("ijka->ijka", h.v.aaaa.ooov)
        + einsum("jika->ijka", h.v.aaaa.ooov)
    )

    tau46 = (
        - 2 * einsum("ijkl->ijkl", h.v.aaaa.oooo)
        + 2 * einsum("ijlk->ijkl", h.v.aaaa.oooo)
        + 2 * einsum("jikl->ijkl", h.v.aaaa.oooo)
        - 2 * einsum("jilk->ijkl", h.v.aaaa.oooo)
        - einsum("lkab,ijab->ijkl", tau18, tau43)
        - 2 * einsum("ak,jila->ijkl", a.t1.a, tau44)
        - 4 * einsum("al,jika->ijkl", a.t1.a, tau45)
    )

    tau47 = (
        einsum("iabj->ijab", h.v.aaaa.ovvo)
        - einsum("iajb->ijab", h.v.aaaa.ovov)
    )

    tau48 = (
        einsum("bi,jkab->ijka", a.t1.a, tau47)
    )

    tau49 = (
        - 4 * einsum("jaik->ijka", h.v.aaaa.ovoo)
        + 4 * einsum("jaki->ijka", h.v.aaaa.ovoo)
        + 2 * einsum("kibc,jacb->ijka", tau18, tau31)
        - einsum("ikja->ijka", tau33)
        + einsum("kija->ijka", tau33)
        + 4 * einsum("ikja->ijka", tau36)
        - 4 * einsum("kija->ijka", tau36)
        + einsum("jb,ikba->ijka", tau39, tau1)
        - 4 * einsum("ijka->ijka", tau42)
        + einsum("al,ljik->ijka", a.t1.a, tau46)
        + 4 * einsum("kjia->ijka", tau48)
    )

    tau50 = (
        4 * einsum("iajk->ijka", h.v.aaaa.ovoo)
        - 4 * einsum("iakj->ijka", h.v.aaaa.ovoo)
        + 2 * einsum("jkbc,iabc->ijka", tau10, tau31)
        - einsum("jlba,kilb->ijka", tau1, tau32)
        + einsum("lkab,jilb->ijka", tau1, tau32)
        - 4 * einsum("jkia->ijka", tau36)
        + 4 * einsum("kjia->ijka", tau36)
        + einsum("ib,kjab->ijka", tau39, tau7)
        + 4 * einsum("jika->ijka", tau42)
        - 4 * einsum("kija->ijka", tau48)
    )

    tau51 = (
        einsum("ak,ikja->ij", a.t1.b, h.v.abab.ooov)
    )

    tau52 = (
        einsum("baki,jkab->ij", a.t2.ab, h.v.abab.oovv)
    )

    tau53 = (
        einsum("kiab,jkba->ij", tau18, tau43)
    )

    tau54 = (
        einsum("ak,kija->ij", a.t1.a, tau21)
    )

    tau55 = (
        einsum("ai,ja->ij", a.t1.a, tau39)
    )

    tau56 = (
        4 * einsum("ij->ij", h.f_a.oo)
        + 4 * einsum("ij->ij", tau51)
        + 4 * einsum("ji->ij", tau52)
        - einsum("ji->ij", tau53)
        - 4 * einsum("ij->ij", tau54)
        + 2 * einsum("ji->ij", tau55)
    )

    tau57 = (
        einsum("kj,kiab->ijab", tau56, tau1)
    )

    tau58 = (
        2 * einsum("abic->iabc", h.v.aaaa.vvov)
        - 2 * einsum("baic->iabc", h.v.aaaa.vvov)
        + einsum("di,badc->iabc", a.t1.a, tau9)
    )

    tau59 = (
        - einsum("abic->iabc", h.v.aaaa.vvov)
        + einsum("baic->iabc", h.v.aaaa.vvov)
    )

    tau60 = (
        einsum("caki,kjcb->ijab", a.t2.ab, h.v.abab.oovv)
    )

    tau61 = (
        - einsum("abij->ijab", a.t2.bb)
        + einsum("baij->ijab", a.t2.bb)
        + einsum("abji->ijab", a.t2.bb)
        - einsum("baji->ijab", a.t2.bb)
    )

    tau62 = (
        einsum("kjcb,ikca->ijab", tau13, tau61)
    )

    tau63 = (
        einsum("iabc->iabc", h.v.bbbb.ovvv)
        - einsum("iacb->iabc", h.v.bbbb.ovvv)
    )

    tau64 = (
        einsum("cj,iacb->ijab", a.t1.b, tau63)
    )

    tau65 = (
        4 * einsum("jabi->ijab", h.v.bbbb.ovvo)
        - 4 * einsum("jaib->ijab", h.v.bbbb.ovov)
        + 4 * einsum("ijab->ijab", tau60)
        + einsum("ijab->ijab", tau62)
        - 4 * einsum("jiab->ijab", tau64)
    )

    tau66 = (
        einsum("abij->ijab", a.t2.bb)
        - einsum("baij->ijab", a.t2.bb)
        - einsum("abji->ijab", a.t2.bb)
        + einsum("baji->ijab", a.t2.bb)
    )

    tau67 = (
        einsum("ikac,kjcb->ijab", tau65, tau66)
    )

    tau68 = (
        - einsum("abcd->abcd", h.v.bbbb.vvvv)
        + einsum("abdc->abcd", h.v.bbbb.vvvv)
        + einsum("bacd->abcd", h.v.bbbb.vvvv)
        - einsum("badc->abcd", h.v.bbbb.vvvv)
    )

    tau69 = (
        - einsum("abij->ijab", a.t2.bb)
        + einsum("abji->ijab", a.t2.bb)
    )

    tau70 = (
        einsum("jabi->ijab", h.v.bbbb.ovvo)
        - einsum("jaib->ijab", h.v.bbbb.ovov)
        + einsum("ijab->ijab", tau60)
        - einsum("jiab->ijab", tau64)
    )

    tau71 = (
        einsum("ci,jabc->ijab", a.t1.b, h.v.abab.ovvv)
    )

    tau72 = (
        einsum("caki,kjcb->ijab", a.t2.ab, tau2)
    )

    tau73 = (
        2 * einsum("jabi->ijab", h.v.abab.ovvo)
        + 2 * einsum("ijab->ijab", tau71)
        + einsum("ijab->ijab", tau72)
    )

    tau74 = (
        einsum("cbkj,ikac->ijab", a.t2.ab, tau73)
    )

    tau75 = (
        - einsum("abij->ijab", a.t2.bb)
        + einsum("baij->ijab", a.t2.bb)
    )

    tau76 = (
        einsum("abij->ijab", a.t2.bb)
        - einsum("abji->ijab", a.t2.bb)
    )

    tau77 = (
        einsum("bi,jkab->ijka", a.t1.b, tau13)
    )

    tau78 = (
        - 2 * einsum("ijka->ijka", h.v.bbbb.ooov)
        + 2 * einsum("jika->ijka", h.v.bbbb.ooov)
        + einsum("kija->ijka", tau77)
    )

    tau79 = (
        einsum("ijka->ijka", h.v.bbbb.ooov)
        - einsum("jika->ijka", h.v.bbbb.ooov)
    )

    tau80 = (
        2 * einsum("ijkl->ijkl", h.v.bbbb.oooo)
        - 2 * einsum("ijlk->ijkl", h.v.bbbb.oooo)
        - 2 * einsum("jikl->ijkl", h.v.bbbb.oooo)
        + 2 * einsum("jilk->ijkl", h.v.bbbb.oooo)
        - einsum("ijab,lkab->ijkl", tau13, tau76)
        - 2 * einsum("ak,jila->ijkl", a.t1.b, tau78)
        - 4 * einsum("al,jika->ijkl", a.t1.b, tau79)
    )

    tau81 = (
        einsum("jabi->ijab", h.v.abab.ovvo)
        + einsum("ijab->ijab", tau71)
    )

    tau82 = (
        einsum("cbkj,ikac->ijab", a.t2.ab, tau81)
    )

    tau83 = (
        einsum("ci,iacb->ab", a.t1.a, h.v.abab.ovvv)
    )

    tau84 = (
        einsum("acji,ijcb->ab", a.t2.ab, h.v.abab.oovv)
    )

    tau85 = (
        einsum("ijcb,ijca->ab", tau13, tau75)
    )

    tau86 = (
        einsum("ci,iabc->ab", a.t1.b, tau63)
    )

    tau87 = (
        4 * einsum("ab->ab", h.f_b.vv)
        + 4 * einsum("ab->ab", tau83)
        - 4 * einsum("ab->ab", tau84)
        + einsum("ab->ab", tau85)
        - 4 * einsum("ab->ab", tau86)
    )

    tau88 = (
        einsum("bc,ijca->ijab", tau87, tau66)
    )

    tau89 = (
        - einsum("iabc->iabc", h.v.bbbb.ovvv)
        + einsum("iacb->iabc", h.v.bbbb.ovvv)
    )

    tau90 = (
        - einsum("ijab->ijab", h.v.bbbb.oovv)
        + einsum("ijba->ijab", h.v.bbbb.oovv)
        + einsum("jiab->ijab", h.v.bbbb.oovv)
        - einsum("jiba->ijab", h.v.bbbb.oovv)
    )

    tau91 = (
        einsum("bk,ijab->ijka", a.t1.b, tau90)
    )

    tau92 = (
        - 2 * einsum("jkia->ijka", h.v.bbbb.ooov)
        + 2 * einsum("kjia->ijka", h.v.bbbb.ooov)
        + einsum("kjia->ijka", tau91)
    )

    tau93 = (
        einsum("liab,jklb->ijka", tau61, tau92)
    )

    tau94 = (
        einsum("bi,jkab->ijka", a.t1.b, h.v.abab.oovv)
    )

    tau95 = (
        einsum("jkai->ijka", h.v.abab.oovo)
        + einsum("ijka->ijka", tau94)
    )

    tau96 = (
        einsum("balk,iljb->ijka", a.t2.ab, tau95)
    )

    tau97 = (
        einsum("bj,jiba->ia", a.t1.a, h.v.abab.oovv)
    )

    tau98 = (
        einsum("bj,ijab->ia", a.t1.b, tau13)
    )

    tau99 = (
        2 * einsum("ia->ia", h.f_b.ov)
        + 2 * einsum("ia->ia", tau97)
        + einsum("ia->ia", tau98)
    )

    tau100 = (
        einsum("cj,iabc->ijab", a.t1.b, tau89)
    )

    tau101 = (
        einsum("iabj->ijab", h.v.bbbb.ovvo)
        - einsum("iajb->ijab", h.v.bbbb.ovov)
        - einsum("ijab->ijab", tau100)
    )

    tau102 = (
        einsum("bk,ijab->ijka", a.t1.b, tau101)
    )

    tau103 = (
        2 * einsum("ijka->ijka", h.v.bbbb.ooov)
        - 2 * einsum("jika->ijka", h.v.bbbb.ooov)
        + einsum("ijka->ijka", tau91)
    )

    tau104 = (
        - einsum("ijka->ijka", h.v.bbbb.ooov)
        + einsum("jika->ijka", h.v.bbbb.ooov)
    )

    tau105 = (
        - 2 * einsum("ijkl->ijkl", h.v.bbbb.oooo)
        + 2 * einsum("ijlk->ijkl", h.v.bbbb.oooo)
        + 2 * einsum("jikl->ijkl", h.v.bbbb.oooo)
        - 2 * einsum("jilk->ijkl", h.v.bbbb.oooo)
        - einsum("lkab,ijab->ijkl", tau76, tau90)
        - 2 * einsum("ak,jila->ijkl", a.t1.b, tau103)
        - 4 * einsum("al,jika->ijkl", a.t1.b, tau104)
    )

    tau106 = (
        einsum("iabj->ijab", h.v.bbbb.ovvo)
        - einsum("iajb->ijab", h.v.bbbb.ovov)
    )

    tau107 = (
        einsum("bk,ijab->ijka", a.t1.b, tau106)
    )

    tau108 = (
        - 4 * einsum("jaik->ijka", h.v.bbbb.ovoo)
        + 4 * einsum("jaki->ijka", h.v.bbbb.ovoo)
        + 2 * einsum("kibc,jacb->ijka", tau76, tau89)
        - einsum("ikja->ijka", tau93)
        + einsum("kija->ijka", tau93)
        + 4 * einsum("kjia->ijka", tau96)
        - 4 * einsum("ijka->ijka", tau96)
        + einsum("jb,ikba->ijka", tau99, tau61)
        - 4 * einsum("jkia->ijka", tau102)
        + einsum("al,ljik->ijka", a.t1.b, tau105)
        + 4 * einsum("jika->ijka", tau107)
    )

    tau109 = (
        2 * einsum("jkia->ijka", h.v.bbbb.ooov)
        - 2 * einsum("kjia->ijka", h.v.bbbb.ooov)
        + einsum("ikja->ijka", tau77)
    )

    tau110 = (
        4 * einsum("iajk->ijka", h.v.bbbb.ovoo)
        - 4 * einsum("iakj->ijka", h.v.bbbb.ovoo)
        + 2 * einsum("jkbc,iabc->ijka", tau69, tau89)
        + einsum("jilb,lkab->ijka", tau109, tau61)
        - einsum("kilb,jlba->ijka", tau109, tau61)
        + 4 * einsum("jika->ijka", tau96)
        - 4 * einsum("kija->ijka", tau96)
        + einsum("ib,kjab->ijka", tau99, tau66)
        + 4 * einsum("ikja->ijka", tau102)
        - 4 * einsum("ijka->ijka", tau107)
    )

    tau111 = (
        einsum("ak,kiaj->ij", a.t1.a, h.v.abab.oovo)
    )

    tau112 = (
        einsum("abki,kjab->ij", a.t2.ab, h.v.abab.oovv)
    )

    tau113 = (
        einsum("kiab,jkba->ij", tau76, tau90)
    )

    tau114 = (
        einsum("ak,kija->ij", a.t1.b, tau79)
    )

    tau115 = (
        einsum("aj,ia->ij", a.t1.b, tau99)
    )

    tau116 = (
        4 * einsum("ij->ij", h.f_b.oo)
        + 4 * einsum("ij->ij", tau111)
        + 4 * einsum("ji->ij", tau112)
        - einsum("ji->ij", tau113)
        - 4 * einsum("ij->ij", tau114)
        + 2 * einsum("ij->ij", tau115)
    )

    tau117 = (
        einsum("kj,kiab->ijab", tau116, tau61)
    )

    tau118 = (
        2 * einsum("abic->iabc", h.v.bbbb.vvov)
        - 2 * einsum("baic->iabc", h.v.bbbb.vvov)
        + einsum("di,badc->iabc", a.t1.b, tau68)
    )

    tau119 = (
        - einsum("abic->iabc", h.v.bbbb.vvov)
        + einsum("baic->iabc", h.v.bbbb.vvov)
    )

    tau120 = (
        2 * einsum("ajib->ijab", h.v.abab.voov)
        + 2 * einsum("ijab->ijab", tau12)
        + einsum("ikca,kjcb->ijab", tau1, h.v.abab.oovv)
        + einsum("ijab->ijab", tau14)
    )

    tau121 = (
        einsum("jabi->ijab", h.v.bbbb.ovvo)
        - einsum("jaib->ijab", h.v.bbbb.ovov)
        - einsum("jiab->ijab", tau64)
    )

    tau122 = (
        einsum("jaib->ijab", h.v.abab.ovov)
        + einsum("ci,jacb->ijab", a.t1.a, h.v.abab.ovvv)
        - einsum("acki,jkcb->ijab", a.t2.ab, h.v.abab.oovv)
    )

    tau123 = (
        einsum("ci,ajbc->ijab", a.t1.b, h.v.abab.vovv)
    )

    tau124 = (
        einsum("ajbi->ijab", h.v.abab.vovo)
        + einsum("ijab->ijab", tau123)
    )

    tau125 = (
        einsum("ai,jkla->ijkl", a.t1.b, h.v.abab.ooov)
    )

    tau126 = (
        einsum("baji,klab->ijkl", a.t2.ab, h.v.abab.oovv)
    )

    tau127 = (
        einsum("ijak->ijka", h.v.abab.oovo)
        + einsum("kija->ijka", tau94)
    )

    tau128 = (
        einsum("al,ijka->ijkl", a.t1.a, tau127)
    )

    tau129 = (
        einsum("ijkl->ijkl", h.v.abab.oooo)
        + einsum("lijk->ijkl", tau125)
        + einsum("klij->ijkl", tau126)
        + einsum("ijlk->ijkl", tau128)
    )

    tau130 = (
        einsum("iabj->ijab", h.v.abab.ovvo)
        + einsum("jiab->ijab", tau71)
    )

    tau131 = (
        2 * einsum("jaik->ijka", h.v.abab.ovoo)
        + 2 * einsum("bk,jaib->ijka", a.t1.b, h.v.abab.ovov)
        + 2 * einsum("cbki,jabc->ijka", a.t2.ab, h.v.abab.ovvv)
        + einsum("ijlb,lkab->ijka", tau35, tau61)
        + einsum("balk,ijlb->ijka", a.t2.ab, tau32)
        - 2 * einsum("abli,kjlb->ijka", a.t2.ab, tau95)
        + einsum("jb,abki->ijka", tau39, a.t2.ab)
        + 2 * einsum("bi,jkab->ijka", a.t1.a, tau130)
        - 2 * einsum("al,jlik->ijka", a.t1.b, tau129)
    )

    tau132 = (
        einsum("aibj->ijab", h.v.abab.vovo)
        + einsum("jiab->ijab", tau123)
    )

    tau133 = (
        2 * einsum("aijk->ijka", h.v.abab.vooo)
        + 2 * einsum("bk,aijb->ijka", a.t1.b, h.v.abab.voov)
        + 2 * einsum("cbkj,aibc->ijka", a.t2.ab, h.v.abab.vovv)
        + einsum("jlba,klib->ijka", tau1, tau95)
        + einsum("balj,kilb->ijka", a.t2.ab, tau109)
        - 2 * einsum("ablk,jlib->ijka", a.t2.ab, tau35)
        + einsum("ib,bakj->ijka", tau99, a.t2.ab)
        + 2 * einsum("bj,ikab->ijka", a.t1.a, tau132)
    )

    tau134 = (
        einsum("abci->iabc", h.v.abab.vvvo)
        + einsum("di,abcd->iabc", a.t1.b, h.v.abab.vvvv)
    )

    r2_aa = (
        einsum("abij->abij", h.v.aaaa.vvoo) / 2
        - einsum("baij->abij", h.v.aaaa.vvoo) / 2
        - einsum("abji->abij", h.v.aaaa.vvoo) / 2
        + einsum("baji->abij", h.v.aaaa.vvoo) / 2
        + einsum("ijab->abij", tau8) / 8
        - einsum("ijba->abij", tau8) / 8
        + einsum("ijdc,bacd->abij", tau10, tau9) / 4
        + einsum("ikca,jkbc->abij", tau1, tau11) / 2
        - einsum("kibc,jkac->abij", tau1, tau11) / 2
        + einsum("jiba->abij", tau16) / 2
        - einsum("jiab->abij", tau16) / 2
        + einsum("klab,lkij->abij", tau17, tau22) / 8
        + einsum("ijab->abij", tau24)
        - einsum("ijba->abij", tau24)
        + einsum("jiab->abij", tau30) / 8
        + einsum("ijba->abij", tau30) / 8
        - einsum("ak,jkib->abij", a.t1.a, tau49) / 4
        - einsum("bk,kjia->abij", a.t1.a, tau50) / 4
        + einsum("jiab->abij", tau57) / 8
        + einsum("ijba->abij", tau57) / 8
        - einsum("ci,jabc->abij", a.t1.a, tau58) / 2
        - einsum("cj,iabc->abij", a.t1.a, tau59)
    )

    r2_bb = (
        einsum("abij->abij", h.v.bbbb.vvoo) / 2
        - einsum("baij->abij", h.v.bbbb.vvoo) / 2
        - einsum("abji->abij", h.v.bbbb.vvoo) / 2
        + einsum("baji->abij", h.v.bbbb.vvoo) / 2
        + einsum("ijab->abij", tau67) / 8
        - einsum("ijba->abij", tau67) / 8
        + einsum("bacd,ijdc->abij", tau68, tau69) / 4
        + einsum("ikca,jkbc->abij", tau61, tau70) / 2
        - einsum("kibc,jkac->abij", tau61, tau70) / 2
        + einsum("ijab->abij", tau74) / 2
        - einsum("ijba->abij", tau74) / 2
        + einsum("klab,lkij->abij", tau75, tau80) / 8
        + einsum("jiba->abij", tau82)
        - einsum("jiab->abij", tau82)
        + einsum("jiab->abij", tau88) / 8
        + einsum("ijba->abij", tau88) / 8
        - einsum("ak,jkib->abij", a.t1.b, tau108) / 4
        - einsum("bk,kjia->abij", a.t1.b, tau110) / 4
        + einsum("jiab->abij", tau117) / 8
        + einsum("ijba->abij", tau117) / 8
        - einsum("ci,jabc->abij", a.t1.b, tau118) / 2
        - einsum("cj,iabc->abij", a.t1.b, tau119)
    )

    r2_ab = (
        einsum("dcji,abcd->abij", a.t2.ab, h.v.abab.vvvv)
        + einsum("cj,abic->abij", a.t1.b, h.v.abab.vvov)
        + einsum("abij->abij", h.v.abab.vvoo)
        + einsum("ikac,kjcb->abij", tau120, tau66) / 4
        - einsum("ikca,jkbc->abij", tau7, tau73) / 4
        + einsum("cbkj,ikac->abij", a.t2.ab, tau11)
        + einsum("caki,jkbc->abij", a.t2.ab, tau121)
        - einsum("ackj,ikbc->abij", a.t2.ab, tau122)
        - einsum("bcki,jkac->abij", a.t2.ab, tau124)
        + einsum("balk,klij->abij", a.t2.ab, tau129)
        + einsum("bc,caji->abij", tau87, a.t2.ab) / 4
        + einsum("ac,bcji->abij", tau29, a.t2.ab) / 4
        - einsum("ak,ikjb->abij", a.t1.a, tau131) / 2
        - einsum("bk,kija->abij", a.t1.b, tau133) / 2
        - einsum("ki,abkj->abij", tau56, a.t2.ab) / 4
        - einsum("kj,baki->abij", tau116, a.t2.ab) / 4
        + einsum("ci,jabc->abij", a.t1.a, tau134)
    )

    return Tensors(
        t1=Tensors(a=r1_a, b=r1_b),
        t2=Tensors(aa=r2_aa, bb=r2_bb, ab=r2_ab)
    )
