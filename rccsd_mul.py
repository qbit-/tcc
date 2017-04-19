import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from collections import namedtuple
from tcc.rccsd import RCCSD

class RCCSD_MUL(RCCSD):
    """
    This implements RCCSD algorithm with Mulliken ordered
    amplitudes and integrals
    """

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

        return self.AMPLITUDES_TYPE(t1, t2)

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

    def update_rhs(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
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
    
        g1 = (
            einsum("jb,jiba->ai", tau3, tau4) / 2
            + einsum("bi,ab->ai", a.t1, tau7) / 2
            - einsum("aj,ji->ai", a.t1, tau10) / 2
            + einsum("kjba,jikb->ai", tau6, h.v.ooov) / 2
            + einsum("jicb,jbac->ai", tau11, h.v.ovvv) / 2
            + einsum("ia->ai", h.f.ov.conj())
            + einsum("bj,jiba->ai", a.t1, tau12)
        )
    
        g2 = (
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
        
        e_ai = cc_denom(h.f, 2, 'mul', 'full')
        e_aibj = cc_denom(h.f, 4, 'mul', 'full')

        g1 = g1 - a.t1 / e_ai
        g2 = g2 - a.t2 / e_aibj
        
        return self.RHS_TYPE(g1=g1, g2=g2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.AMPLITUDES_TYPE(
            *(g[ii] * (- cc_denom(h.f, g[ii].ndim, 'mul', 'full'))
              for ii in range(len(g)))
        )

    def calc_residuals(self, h, a, g):
        """
        Calculates CC residuals from RHS and amplitudes
        """
        return self.RESIDUALS_TYPE(
            *[a[ii] / cc_denom(h.f, a[ii].ndim, 'mul', 'full') + g[ii]
              for ii in range(len(a))]
        )

def test_mp2_energy(): # pragma: nocover
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
    rhf.scf()  # -76.0267656731
    cc = RCCSD_MUL(rhf)
    h = cc.create_ham()
    a = cc.init_amplitudes(h)
    energy = cc.calculate_energy(h, a)
    print('E_mp2 - E_cc,init = {:18.12g}'.format(energy - -0.204019967288338))

def test_cc(): # pragma: nocover
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
    
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd_mul import RCCSD_MUL
    cc = RCCSD_MUL(rhf)
    converged, energy, _ = classic_solver(cc)
    
if __name__ == '__main__':
    test_mp2_energy()
    test_cc()
