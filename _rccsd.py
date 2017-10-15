from numpy import einsum
from .tensors import Tensors

def _rccsd_calculate_energy(h, a):
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


def _rccsd_calc_residuals(h, a):
    """
    Calculates CC residuals for CC equations
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

    r1 = (
        einsum("jb,jiba->ai", tau3, tau4)
        - einsum("aj,ji->ai", a.t1, tau7)
        + einsum("bakj,ijkb->ai", a.t2, tau9)
        + einsum("jicb,jabc->ai", tau5, h.v.ovvv)
        + einsum("bj,jiab->ai", a.t1, tau10)
        + einsum("bi,ab->ai", a.t1, tau12)
        + einsum("ia->ai", h.f.ov.conj())
    )

    r2 = (
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

    return Tensors(t1=r1, t2=r2)


def _rccsd_unit_calculate_energy(h, a):
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

def _rccsd_unit_calc_residuals(h, a):
    """
    Updates residuals of the CC equations
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

    r1 = (
        2 * einsum("jb,jiba->ai", tau3, tau4)
        - 2 * einsum("aj,ji->ai", a.t1, tau7)
        + 2 * einsum("bakj,ikjb->ai", a.t2, tau9)
        + 2 * einsum("bi,ab->ai", a.t1, tau11)
        + 2 * einsum("bj,jiab->ai", a.t1, tau12)
        + 2 * einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("jicb,jabc->ai", tau13, h.v.ovvv)
    )

    r2 = (
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

    return Tensors(t1=r1, t2=r2)

def _rccsd_ri_calculate_energy(h, a):
    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        - einsum("abij->ijab", a.t2)
        + 2 * einsum("baij->ijab", a.t2)
    )

    tau2 = (
        2 * einsum("w,ai->iaw", tau0, a.t1)
        + einsum("wjb,ijba->iaw", h.l.pov, tau1)
    )

    tau3 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau4 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau3)
    )

    energy = (
        einsum("wia,iaw->", h.l.pov, tau2)
        + einsum("ai,ia->", a.t1, tau4)
    )

    return energy

def _rccsd_ri_calc_residuals(h, a):
    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau3 = (
        einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau1)
        + 2 * einsum("w,wia->ia", tau2, h.l.pov)
    )

    tau4 = (
        - einsum("abij->ijab", a.t2)
        + 2 * einsum("baij->ijab", a.t2)
        + 2 * einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau5 = (
        einsum("wjb,jiab->iaw", h.l.pov, tau4)
    )

    tau6 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau7 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau6)
    )

    tau8 = (
        - einsum("iaw->iaw", tau5)
        + 2 * einsum("aj,jiw->iaw", a.t1, tau7)
    )

    tau9 = (
        einsum("aj,ijw->iaw", a.t1, tau6)
    )

    tau10 = (
        2 * einsum("iaw->iaw", tau9)
        - einsum("iaw->iaw", tau5)
    )

    tau11 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau0, h.l.pov)
    )

    tau12 = (
        2 * einsum("ij->ij", h.f.oo)
        + 4 * einsum("w,wij->ij", tau0, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau10)
        + 2 * einsum("aj,ia->ij", a.t1, tau11)
    )

    tau13 = (
        - einsum("abij->ijab", a.t2)
        + 2 * einsum("baij->ijab", a.t2)
    )

    tau14 = (
        einsum("wjb,jiab->iaw", h.l.pov, tau13)
    )

    tau15 = (
        einsum("wjb,ijba->iaw", h.l.pov, tau13)
    )

    tau16 = (
        2 * einsum("ab->ab", h.f.vv)
        + 4 * einsum("w,wab->ab", tau0, h.l.pvv)
        - einsum("wib,iaw->ab", h.l.pov, tau14)
        - einsum("wib,iaw->ab", h.l.pov, tau15)
    )

    rt1 = (
        einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("w,wai->ai", tau0, h.l.pvo)
        + einsum("jb,jiab->ai", tau3, tau4) / 2
        - einsum("wab,ibw->ai", h.l.pvv, tau8) / 2
        + einsum("wji,jaw->ai", h.l.poo, tau10) / 2
        - einsum("aj,ji->ai", a.t1, tau12) / 2
        + einsum("bi,ab->ai", a.t1, tau16) / 2
    )
    tau0 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau1 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau2 = (
        einsum("wjb,abij->iaw", h.l.pov, a.t2)
    )

    tau3 = (
        einsum("wjb,abij->iaw", h.l.pov, a.t2)
    )

    tau4 = (
        einsum("wjb,baji->iaw", h.l.pov, a.t2)
    )

    tau5 = (
        einsum("wjb,baji->iaw", h.l.pov, a.t2)
    )

    tau6 = (
        einsum("wjb,abji->iaw", h.l.pov, a.t2)
    )

    tau7 = (
        einsum("wjb,baij->iaw", h.l.pov, a.t2)
    )

    tau8 = (
        einsum("wjb,abji->iaw", h.l.pov, a.t2)
    )

    tau9 = (
        einsum("wjb,baij->iaw", h.l.pov, a.t2)
    )

    tau10 = (
        einsum("wia,wjb->ijab", h.l.pov, h.l.pov)
    )

    tau11 = (
        einsum("acik,kjbc->ijab", a.t2, tau10)
    )

    tau12 = (
        einsum("caki,jkcb->ijab", a.t2, tau10)
    )

    tau13 = (
        einsum("caki,kjbc->ijab", a.t2, tau10)
    )

    tau14 = (
        einsum("acki,jkcb->ijab", a.t2, tau10)
    )

    tau15 = (
        einsum("acki,kjbc->ijab", a.t2, tau10)
    )

    tau16 = (
        einsum("caik,kjbc->ijab", a.t2, tau10)
    )

    tau17 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau18 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau19 = (
        einsum("acw,bdw->abcd", tau17, tau18)
    )

    tau20 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau21 = (
        einsum("aj,ijw->iaw", a.t1, tau20)
    )

    tau22 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau23 = (
        einsum("aj,ijw->iaw", a.t1, tau22)
    )

    tau24 = (
        einsum("wij,abw->ijab", h.l.poo, tau17)
    )

    tau25 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau26 = (
        einsum("wij,abw->ijab", h.l.poo, tau18)
    )

    tau27 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau28 = (
        einsum("wkl,ijw->ijkl", h.l.poo, tau22)
    )

    tau29 = (
        einsum("wjk,ilw->ijkl", h.l.poo, tau20)
    )

    tau30 = (
        einsum("wja,iaw->ij", h.l.pov, tau25)
    )

    tau31 = (
        einsum("wia,jaw->ij", h.l.pov, tau27)
    )

    tau32 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau33 = (
        einsum("w,wij->ij", tau32, h.l.poo)
    )

    tau34 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau35 = (
        einsum("w,wij->ij", tau34, h.l.poo)
    )

    tau36 = (
        einsum("wab,wcd->abcd", h.l.pvv, h.l.pvv)
    )

    tau37 = (
        einsum("wij,wkl->ijkl", h.l.poo, h.l.poo)
    )

    tau38 = (
        einsum("kj,abik->ijab", h.f.oo, a.t2)
    )

    tau39 = (
        einsum("bc,acij->ijab", h.f.vv, a.t2)
    )

    tau40 = (
        einsum("ai,ib->ab", a.t1, h.f.ov)
    )

    tau41 = (
        einsum("ac,cbij->ijab", tau40, a.t2)
    )

    tau42 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau43 = (
        einsum("ik,abkj->ijab", tau42, a.t2)
    )

    tau44 = (
        einsum("wab,wij->ijab", h.l.pvv, h.l.poo)
    )

    tau45 = (
        einsum("acik,kjbc->ijab", a.t2, tau44)
    )

    tau46 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau6)
    )

    tau47 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau7)
    )

    tau48 = (
        einsum("caki,kjbc->ijab", a.t2, tau44)
    )

    tau49 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau3)
    )

    tau50 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau4)
    )

    tau51 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau21)
    )

    tau52 = (
        einsum("wbc,adw->abcd", h.l.pvv, tau18)
    )

    tau53 = (
        einsum("cdij,abdc->ijab", a.t2, tau52)
    )

    tau54 = (
        einsum("wab,ijw->ijab", h.l.pvv, tau20)
    )

    tau55 = (
        einsum("ackj,ikbc->ijab", a.t2, tau54)
    )

    tau56 = (
        einsum("cajk,ikbc->ijab", a.t2, tau54)
    )

    tau57 = (
        einsum("wac,cbw->ab", h.l.pvv, tau18)
    )

    tau58 = (
        einsum("bc,acij->ijab", tau57, a.t2)
    )

    tau59 = (
        einsum("w,wab->ab", tau34, h.l.pvv)
    )

    tau60 = (
        einsum("bc,acij->ijab", tau59, a.t2)
    )

    tau61 = (
        einsum("wja,iaw->ij", h.l.pov, tau3)
    )

    tau62 = (
        einsum("jk,abik->ijab", tau61, a.t2)
    )

    tau63 = (
        einsum("wja,iaw->ij", h.l.pov, tau7)
    )

    tau64 = (
        einsum("jk,abik->ijab", tau63, a.t2)
    )

    tau65 = (
        einsum("wja,iaw->ij", h.l.pov, tau6)
    )

    tau66 = (
        einsum("jk,abik->ijab", tau65, a.t2)
    )

    tau67 = (
        einsum("wja,iaw->ij", h.l.pov, tau4)
    )

    tau68 = (
        einsum("jk,abik->ijab", tau67, a.t2)
    )

    tau69 = (
        einsum("ik,abkj->ijab", tau61, a.t2)
    )

    tau70 = (
        einsum("ik,abkj->ijab", tau63, a.t2)
    )

    tau71 = (
        einsum("abij,klab->ijkl", a.t2, tau10)
    )

    tau72 = (
        einsum("abkl,ijkl->ijab", a.t2, tau71)
    )

    tau73 = (
        einsum("wib,iaw->ab", h.l.pov, tau3)
    )

    tau74 = (
        einsum("bc,acij->ijab", tau73, a.t2)
    )

    tau75 = (
        einsum("wib,iaw->ab", h.l.pov, tau8)
    )

    tau76 = (
        einsum("bc,acij->ijab", tau75, a.t2)
    )

    tau77 = (
        einsum("wib,iaw->ab", h.l.pov, tau5)
    )

    tau78 = (
        einsum("bc,acij->ijab", tau77, a.t2)
    )

    tau79 = (
        einsum("wib,iaw->ab", h.l.pov, tau7)
    )

    tau80 = (
        einsum("bc,acij->ijab", tau79, a.t2)
    )

    tau81 = (
        einsum("ac,cbij->ijab", tau73, a.t2)
    )

    tau82 = (
        einsum("ac,cbij->ijab", tau75, a.t2)
    )

    tau83 = (
        einsum("iaw,jbw->ijab", tau23, tau3)
    )

    tau84 = (
        einsum("iaw,jbw->ijab", tau23, tau4)
    )

    tau85 = (
        einsum("w,wia->ia", tau34, h.l.pov)
    )

    tau86 = (
        einsum("ai,ib->ab", a.t1, tau85)
    )

    tau87 = (
        einsum("ac,cbij->ijab", tau86, a.t2)
    )

    tau88 = (
        einsum("w,wia->ia", tau32, h.l.pov)
    )

    tau89 = (
        einsum("ai,ja->ij", a.t1, tau88)
    )

    tau90 = (
        einsum("ik,abkj->ijab", tau89, a.t2)
    )

    tau91 = (
        einsum("abw,ijw->ijab", tau17, tau20)
    )

    tau92 = (
        einsum("bcjk,ikac->ijab", a.t2, tau91)
    )

    tau93 = (
        einsum("iaw,jbw->ijab", tau23, tau6)
    )

    tau94 = (
        einsum("iaw,jbw->ijab", tau23, tau7)
    )

    tau95 = (
        einsum("cbkj,ikac->ijab", a.t2, tau91)
    )

    tau96 = (
        einsum("wib,baw->ia", h.l.pov, tau18)
    )

    tau97 = (
        einsum("ai,ib->ab", a.t1, tau96)
    )

    tau98 = (
        einsum("ac,cbij->ijab", tau97, a.t2)
    )

    tau99 = (
        einsum("jlw,ikw->ijkl", tau20, tau22)
    )

    tau100 = (
        einsum("abkl,ijkl->ijab", a.t2, tau99)
    )

    tau101 = (
        einsum("wib,baw->ia", h.l.pov, tau17)
    )

    tau102 = (
        einsum("ai,ja->ij", a.t1, tau101)
    )

    tau103 = (
        einsum("ik,abkj->ijab", tau102, a.t2)
    )

    tau104 = (
        2 * einsum("ijab->ijab", tau38)
        - 2 * einsum("ijab->ijab", tau39)
        + 2 * einsum("ijab->ijab", tau41)
        + 2 * einsum("ijab->ijab", tau43)
        + 2 * einsum("ijab->ijab", tau45)
        + 2 * einsum("ijab->ijab", tau46)
        + 2 * einsum("ijab->ijab", tau47)
        + 2 * einsum("ijab->ijab", tau48)
        - 4 * einsum("ijab->ijab", tau49)
        - 4 * einsum("ijab->ijab", tau50)
        + 4 * einsum("ijab->ijab", tau51)
        + 2 * einsum("ijab->ijab", tau53)
        + 2 * einsum("ijab->ijab", tau55)
        + 2 * einsum("ijab->ijab", tau56)
        + 2 * einsum("ijab->ijab", tau58)
        - 4 * einsum("ijab->ijab", tau60)
        + 2 * einsum("ijab->ijab", tau62)
        - einsum("ijab->ijab", tau64)
        - einsum("ijab->ijab", tau66)
        + 2 * einsum("ijab->ijab", tau68)
        + 2 * einsum("ijab->ijab", tau69)
        - einsum("ijab->ijab", tau70)
        - einsum("ijab->ijab", tau72)
        + 2 * einsum("ijab->ijab", tau74)
        - einsum("ijab->ijab", tau76)
        + 2 * einsum("ijab->ijab", tau78)
        - einsum("ijab->ijab", tau80)
        + 2 * einsum("ijab->ijab", tau81)
        - einsum("ijab->ijab", tau82)
        + 4 * einsum("ijab->ijab", tau83)
        + 4 * einsum("ijab->ijab", tau84)
        + 4 * einsum("ijab->ijab", tau87)
        + 4 * einsum("ijab->ijab", tau90)
        - 2 * einsum("ijab->ijab", tau92)
        - 2 * einsum("ijab->ijab", tau93)
        - 2 * einsum("ijab->ijab", tau94)
        - 2 * einsum("ijab->ijab", tau95)
        - 2 * einsum("ijab->ijab", tau98)
        - 2 * einsum("ijab->ijab", tau100)
        - 2 * einsum("ijab->ijab", tau103)
    )

    tau105 = (
        einsum("kj,abki->ijab", h.f.oo, a.t2)
    )

    tau106 = (
        einsum("bc,caij->ijab", h.f.vv, a.t2)
    )

    tau107 = (
        einsum("ac,bcij->ijab", tau40, a.t2)
    )

    tau108 = (
        einsum("ik,abjk->ijab", tau42, a.t2)
    )

    tau109 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau27)
    )

    tau110 = (
        einsum("acki,kjbc->ijab", a.t2, tau44)
    )

    tau111 = (
        einsum("caik,kjbc->ijab", a.t2, tau44)
    )

    tau112 = (
        einsum("ibw,jaw->ijab", tau0, tau27)
    )

    tau113 = (
        einsum("cdij,abcd->ijab", a.t2, tau52)
    )

    tau114 = (
        einsum("acjk,ikbc->ijab", a.t2, tau54)
    )

    tau115 = (
        einsum("ibw,jaw->ijab", tau0, tau6)
    )

    tau116 = (
        einsum("ibw,jaw->ijab", tau0, tau7)
    )

    tau117 = (
        einsum("cakj,ikbc->ijab", a.t2, tau54)
    )

    tau118 = (
        einsum("bc,caij->ijab", tau57, a.t2)
    )

    tau119 = (
        einsum("ibw,jaw->ijab", tau0, tau3)
    )

    tau120 = (
        einsum("ibw,jaw->ijab", tau0, tau4)
    )

    tau121 = (
        einsum("bc,caij->ijab", tau59, a.t2)
    )

    tau122 = (
        einsum("ibw,jaw->ijab", tau0, tau21)
    )

    tau123 = (
        einsum("jk,abki->ijab", tau65, a.t2)
    )

    tau124 = (
        einsum("jk,abki->ijab", tau67, a.t2)
    )

    tau125 = (
        einsum("abij,klba->ijkl", a.t2, tau10)
    )

    tau126 = (
        einsum("abkl,ijkl->ijab", a.t2, tau125)
    )

    tau127 = (
        einsum("bc,caij->ijab", tau79, a.t2)
    )

    tau128 = (
        einsum("bc,caij->ijab", tau77, a.t2)
    )

    tau129 = (
        einsum("ac,bcij->ijab", tau86, a.t2)
    )

    tau130 = (
        einsum("ik,abjk->ijab", tau89, a.t2)
    )

    tau131 = (
        einsum("bckj,ikac->ijab", a.t2, tau91)
    )

    tau132 = (
        einsum("cbjk,ikac->ijab", a.t2, tau91)
    )

    tau133 = (
        einsum("ac,bcij->ijab", tau97, a.t2)
    )

    tau134 = (
        einsum("ik,abjk->ijab", tau102, a.t2)
    )

    tau135 = (
        2 * einsum("ijab->ijab", tau105)
        - 2 * einsum("ijab->ijab", tau106)
        + 2 * einsum("ijab->ijab", tau107)
        + 2 * einsum("ijab->ijab", tau108)
        + 4 * einsum("ijab->ijab", tau109)
        + 2 * einsum("ijab->ijab", tau110)
        + 2 * einsum("ijab->ijab", tau111)
        + 4 * einsum("ijab->ijab", tau112)
        + 2 * einsum("ijab->ijab", tau113)
        + 2 * einsum("ijab->ijab", tau114)
        + 2 * einsum("ijab->ijab", tau115)
        + 2 * einsum("ijab->ijab", tau116)
        + 2 * einsum("ijab->ijab", tau117)
        + 2 * einsum("ijab->ijab", tau118)
        - 4 * einsum("ijab->ijab", tau119)
        - 4 * einsum("ijab->ijab", tau120)
        - 4 * einsum("ijab->ijab", tau121)
        + 4 * einsum("ijab->ijab", tau122)
        - einsum("ijab->ijab", tau123)
        + 2 * einsum("ijab->ijab", tau124)
        - einsum("ijab->ijab", tau126)
        - einsum("ijab->ijab", tau127)
        + 2 * einsum("ijab->ijab", tau128)
        + 4 * einsum("ijab->ijab", tau129)
        + 4 * einsum("ijab->ijab", tau130)
        - 2 * einsum("ijab->ijab", tau131)
        - 2 * einsum("ijab->ijab", tau132)
        - 2 * einsum("ijab->ijab", tau133)
        - 2 * einsum("ijab->ijab", tau134)
    )

    rt2 = (
        einsum("wai,wbj->abij", h.l.pvo, h.l.pvo)
        + einsum("wbj,iaw->abij", h.l.pvo, tau0)
        + einsum("wai,jbw->abij", h.l.pvo, tau1)
        + einsum("iaw,jbw->abij", tau2, tau3)
        + einsum("iaw,jbw->abij", tau2, tau4)
        + einsum("jbw,iaw->abij", tau3, tau5)
        + einsum("jbw,iaw->abij", tau4, tau5)
        - einsum("iaw,jbw->abij", tau2, tau6) / 2
        - einsum("iaw,jbw->abij", tau2, tau7) / 2
        - einsum("jbw,iaw->abij", tau3, tau8) / 2
        - einsum("jbw,iaw->abij", tau4, tau8) / 2
        - einsum("jbw,iaw->abij", tau3, tau9) / 2
        - einsum("iaw,jbw->abij", tau5, tau6) / 2
        - einsum("jbw,iaw->abij", tau4, tau9) / 2
        - einsum("iaw,jbw->abij", tau5, tau7) / 2
        - einsum("bcjk,ikac->abij", a.t2, tau11) / 2
        - einsum("acik,jkbc->abij", a.t2, tau12) / 2
        - einsum("bcjk,ikac->abij", a.t2, tau13) / 2
        - einsum("cbkj,ikac->abij", a.t2, tau13) / 2
        + einsum("jbw,iaw->abij", tau6, tau8) / 4
        + einsum("jbw,iaw->abij", tau7, tau8) / 4
        + einsum("jbw,iaw->abij", tau6, tau9) / 4
        + einsum("jbw,iaw->abij", tau7, tau9) / 4
        + einsum("acik,jkbc->abij", a.t2, tau14) / 4
        + einsum("cbjk,ikac->abij", a.t2, tau11) / 4
        + einsum("bcjk,ikac->abij", a.t2, tau15) / 4
        + einsum("cbkj,ikac->abij", a.t2, tau15) / 4
        + einsum("bcki,jkac->abij", a.t2, tau15) / 4
        + einsum("cbik,jkac->abij", a.t2, tau15) / 4
        + einsum("bcjk,ikac->abij", a.t2, tau16) / 4
        + einsum("cajk,ikbc->abij", a.t2, tau14) / 4
        + einsum("bckj,ikac->abij", a.t2, tau13) / 4
        + einsum("caik,jkbc->abij", a.t2, tau12) / 4
        + einsum("cbik,jkac->abij", a.t2, tau16) / 4
        + einsum("cbjk,ikac->abij", a.t2, tau13) / 4
        + einsum("cdij,abcd->abij", a.t2, tau19) / 2
        + einsum("cdji,abdc->abij", a.t2, tau19) / 2
        + einsum("jbw,iaw->abij", tau21, tau23)
        + einsum("bcki,kjac->abij", a.t2, tau24) / 2
        + einsum("cbik,kjac->abij", a.t2, tau24) / 2
        + einsum("iaw,jbw->abij", tau25, tau6) / 2
        + einsum("iaw,jbw->abij", tau25, tau7) / 2
        + einsum("bcjk,kiac->abij", a.t2, tau26) / 2
        + einsum("cbkj,kiac->abij", a.t2, tau26) / 2
        + einsum("acik,kjbc->abij", a.t2, tau24) / 2
        + einsum("caki,kjbc->abij", a.t2, tau24) / 2
        + einsum("jbw,iaw->abij", tau27, tau8) / 2
        + einsum("jbw,iaw->abij", tau27, tau9) / 2
        + einsum("ackj,kibc->abij", a.t2, tau26) / 2
        + einsum("cajk,kibc->abij", a.t2, tau26) / 2
        + einsum("abkl,iklj->abij", a.t2, tau28) / 2
        + einsum("bakl,ilkj->abij", a.t2, tau28) / 2
        + einsum("abkl,jkil->abij", a.t2, tau29) / 2
        + einsum("bakl,jlik->abij", a.t2, tau29) / 2
        + einsum("ik,abkj->abij", tau30, a.t2) / 2
        + einsum("ik,bajk->abij", tau30, a.t2) / 2
        + einsum("kj,abik->abij", tau31, a.t2) / 2
        + einsum("kj,baki->abij", tau31, a.t2) / 2
        - einsum("iaw,jbw->abij", tau25, tau3)
        - einsum("iaw,jbw->abij", tau25, tau4)
        - einsum("iaw,jbw->abij", tau2, tau27)
        - einsum("jbw,iaw->abij", tau27, tau5)
        - einsum("kj,abik->abij", tau33, a.t2)
        - einsum("kj,baki->abij", tau33, a.t2)
        - einsum("ki,abkj->abij", tau35, a.t2)
        - einsum("ki,bajk->abij", tau35, a.t2)
        + einsum("iaw,jbw->abij", tau23, tau27)
        + einsum("jbw,iaw->abij", tau21, tau25)
        + einsum("cdij,acbd->abij", a.t2, tau36) / 2
        + einsum("cdji,adbc->abij", a.t2, tau36) / 2
        + einsum("iaw,jbw->abij", tau0, tau1)
        + einsum("abkl,kilj->abij", a.t2, tau37) / 2
        + einsum("bakl,likj->abij", a.t2, tau37) / 2
        + einsum("iaw,jbw->abij", tau25, tau27)
        - einsum("ijab->abij", tau104) / 4
        - einsum("jiba->abij", tau104) / 4
        - einsum("ijba->abij", tau135) / 4
        - einsum("jiab->abij", tau135) / 4
    )
    return Tensors(t1=rt1, t2=rt2)
