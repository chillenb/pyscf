#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Integrals of numerical (UPF) norm-conserving pseudopotentials with Bloch GTOs.

For an atom with a UPF potential, cell._pseudo holds a GTH entry with only the
long-range local term  -Z erf(r/(sqrt(2) rloc))/r,  which every DF/FFT backend
already handles. This module adds the rest:

    dV(r) = vloc(r) + Z erf(r/(sqrt(2) rloc))/r                 (short range)
    V_nl  = sum_{ij,m} |beta_i Y_lm> D_ij <beta_j Y_lm|          (nonlocal)

Both are localized on the atom, so for Bloch AOs phi^k (which are periodic up
to a phase) the unit-cell matrix elements are integrals over all space around
each atom:

    <phi^k_p|dV_a|phi^k_q>   = int dr phi^k_p(r)* dV(|r-R_a|) phi^k_q(r)
    <phi^k_p|V_nl,a|phi^k_q> = sum P^k_{ip}* D_ij P^k_{jq},
                               P^k_{ip} = int dr beta_i Y_lm(r-R_a) phi^k_p(r)

These are evaluated on an atom-centered radial x Lebedev grid.
'''

import numpy
import scipy.special
from scipy.interpolate import CubicSpline
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.gto.basis import parse_upf
from pyscf.data.elements import _symbol, _std_symbol_without_ghost, is_ghost_atom

NRAD = getattr(__config__, 'pbc_gto_pseudo_upf_nrad', 120)
NANG = getattr(__config__, 'pbc_gto_pseudo_upf_nang', 302)
# Radius (bohr) up to which the short-range local remainder is integrated.
# None: the whole UPF mesh, as Quantum ESPRESSO does (beyond the mesh the
# remainder is the analytic erfc tail, which is negligible). The tabulated
# vloc approaches -Z/r only to ~1e-5 Ha, so a shorter radius drops a tiny tail.
RCUT_LOC = getattr(__config__, 'pbc_gto_pseudo_upf_rcut_loc', None)
MAX_MEMORY = getattr(__config__, 'pbc_gto_pseudo_upf_max_memory', 2000)


def upf_pseudo(cell):
    '''{symbol: parsed UPF data} for the atoms that use a UPF potential.
    Re-derived from cell.pseudo so that it survives copy/dumps/loads.'''
    pseudo = cell.pseudo
    if not pseudo:
        return {}
    from pyscf.gto.mole import _parse_default_basis
    uniq_atoms = {a[0] for a in cell._atom}
    atoms_wo_ghost = [a for a in uniq_atoms if not is_ghost_atom(a)]
    if isinstance(pseudo, str):
        if not parse_upf.is_upf(pseudo):
            return {}
    elif isinstance(pseudo, dict):
        if not any(parse_upf.is_upf(v) for v in pseudo.values()):
            return {}
    else:
        return {}
    tab = _parse_default_basis(pseudo, atoms_wo_ghost)
    out = {}
    for atom, name in tab.items():
        if parse_upf.is_upf(name):
            symb = _symbol(atom)
            stdsymb = _std_symbol_without_ghost(symb)
            out[symb] = parse_upf.parse(parse_upf.resolve(name, stdsymb))
    return out


def has_upf(cell):
    return bool(upf_pseudo(cell))


class _RadialFunctions:
    '''Splines of the short-range local remainder and the projectors.'''
    def __init__(self, data):
        r = data['r']
        z = data['z']
        rloc = data['rloc']
        self.z = z
        self.rloc = rloc
        self.rmesh = r[-1]
        mask = r > 0
        self._vloc = CubicSpline(r[mask], data['vloc'][mask])
        self._rmin = r[mask][0]
        a = 1. / (numpy.sqrt(2) * rloc)
        rtail = 1.
        while z * scipy.special.erfc(a * rtail) / rtail > 1e-14:
            rtail += .1
        if RCUT_LOC is None:
            self.rcut_loc = max(self.rmesh, rtail)
        else:
            self.rcut_loc = max(RCUT_LOC, rtail)

        self.ls = [b[0] for b in data['beta']]
        self.rcut_nl = max([b[2] for b in data['beta']], default=0.)
        self._beta = [CubicSpline(r, b[1]) for b in data['beta']]
        self._beta_rcut = [b[2] for b in data['beta']]
        self.dij = data['dij']

    def vloc(self, r):
        r = numpy.asarray(r, dtype=float)
        v = numpy.empty_like(r)
        inside = r <= self.rmesh
        v[inside] = self._vloc(numpy.maximum(r[inside], self._rmin))
        v[~inside] = -self.z / r[~inside]
        return v

    def dv(self, r):
        r = numpy.asarray(r, dtype=float)
        a = 1. / (numpy.sqrt(2) * self.rloc)
        # Z erf(a r)/r with its r -> 0 limit
        with numpy.errstate(invalid='ignore', divide='ignore'):
            lr = numpy.where(r > 1e-8, self.z * scipy.special.erf(a * r) / r,
                             self.z * 2 * a / numpy.sqrt(numpy.pi))
        return self.vloc(r) + lr

    def beta(self, i, r):
        '''beta_i(r) (the radial part, without r)'''
        r = numpy.asarray(r, dtype=float)
        b = self._beta[i](r) / r
        b[r > self._beta_rcut[i]] = 0
        return b


def _atom_grid(rcut_nl, rcut_loc, nrad, nang):
    '''Radial x angular grid on [0, rcut_loc]. Gauss-Legendre in x on
    r = rcut_nl x^2 (dense near the nucleus) up to the projector cutoff, where
    the projectors may have a kink, then Gauss-Legendre on [rcut_nl, rcut_loc].
    Returns r, radial weights (including r^2), unit vectors, angular weights.
    '''
    from pyscf.dft import LebedevGrid
    rs, ws = [], []
    if rcut_nl > 0:
        n1 = nrad if rcut_loc <= rcut_nl else max(nrad * 2 // 3, 8)
        x, wx = numpy.polynomial.legendre.leggauss(n1)
        x, wx = .5 * (x + 1), .5 * wx
        r = rcut_nl * x**2
        rs.append(r)
        ws.append(wx * 2 * rcut_nl * x * r**2)
    else:
        n1 = 0
    if rcut_loc > rcut_nl:
        n2 = max(nrad - n1, 8)
        x, wx = numpy.polynomial.legendre.leggauss(n2)
        if rcut_nl > 0:
            r = rcut_nl + .5 * (x + 1) * (rcut_loc - rcut_nl)
            w = .5 * wx * (rcut_loc - rcut_nl) * r**2
        else:  # no projectors: x^2 map over the whole range
            x, wx = .5 * (x + 1), .5 * wx
            r = rcut_loc * x**2
            w = wx * 2 * rcut_loc * x * r**2
        rs.append(r)
        ws.append(w)
    ang = LebedevGrid.MakeAngularGrid(nang)
    # Lebedev weights sum to 1
    return (numpy.concatenate(rs), numpy.concatenate(ws),
            ang[:,:3], ang[:,3] * 4 * numpy.pi)


def get_pp_upf(cell, kpts=None, nrad=None, nang=None):
    '''Matrix elements of the numerical parts (short-range local remainder +
    nonlocal projectors) of all UPF pseudopotentials in the cell.

    Same output convention as pp_int.get_pp_nl: (nao,nao) for kpts None or a
    single k-point, (nkpts,nao,nao) otherwise; real at the Gamma point.
    Also works for a molecule (gto.Mole with a UPF .pseudo), without kpts.
    '''
    from pyscf.symm.sph import real_sph_vec
    if nrad is None: nrad = NRAD
    if nang is None: nang = NANG
    log = logger.new_logger(cell)
    t0 = (logger.process_clock(), logger.perf_counter())

    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    vpp = numpy.zeros((nkpts, nao, nao), dtype=numpy.complex128)

    upf = upf_pseudo(cell)
    if not upf:
        return _format(vpp, kpts)
    # a molecule (gto.Mole): plain AOs, no k-points
    is_pbc = getattr(cell, 'dimension', 0) > 0 and hasattr(cell, 'pbc_eval_gto')
    if not is_pbc:
        assert kpts is None or numpy.allclose(kpts, 0)
    radfuncs = {symb: _RadialFunctions(data) for symb, data in upf.items()}

    atom_coords = cell.atom_coords()
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in radfuncs or cell.atom_charge(ia) == 0:
            continue
        rf = radfuncs[symb]
        r, wr, uvec, wang = _atom_grid(rf.rcut_nl, rf.rcut_loc, nrad, nang)
        nr, na = len(r), len(wang)
        log.debug1('UPF atom %d %s: rloc %.4f  rcut_loc %.3f  rcut_nl %.3f  '
                   'nrad %d  nang %d', ia, symb, rf.rloc, rf.rcut_loc,
                   rf.rcut_nl, nr, na)
        dv = rf.dv(r)
        dv[r > rf.rcut_loc] = 0
        ylm = real_sph_vec(uvec, max(rf.ls, default=0))
        betas = [rf.beta(i, r) for i in range(len(rf.ls))]
        # projector functions on the grid: (nproj_lm, nr, na)
        proj = []
        proj_l = []
        proj_idx = []
        for i, l in enumerate(rf.ls):
            for m in range(2*l+1):
                proj.append(betas[i][:,None] * ylm[l][m][None,:])
                proj_l.append(l)
                proj_idx.append(i)
        nproj = len(proj)
        if nproj > 0:
            proj = numpy.asarray(proj)
            proj_idx = numpy.asarray(proj_idx)
            # D expanded over m: D[(i,m),(j,m')] = D_ij delta_mm'
            mvals = numpy.concatenate([numpy.arange(2*l+1) for l in rf.ls])
            dfull = (rf.dij[proj_idx[:,None], proj_idx[None,:]] *
                     (mvals[:,None] == mvals[None,:]) *
                     (numpy.asarray(proj_l)[:,None] == numpy.asarray(proj_l)[None,:]))
        pk = numpy.zeros((nkpts, nproj, nao), dtype=numpy.complex128)

        # loop over radial shells in blocks
        mem_per_shell = nkpts * na * nao * 16 * 2 / 1e6
        blk = max(1, int(MAX_MEMORY / max(mem_per_shell, 1e-6)))
        for r0 in range(0, nr, blk):
            r1 = min(nr, r0 + blk)
            coords = (r[r0:r1,None,None] * uvec[None,:,:]).reshape(-1,3)
            coords = coords + atom_coords[ia]
            w = (wr[r0:r1,None] * wang[None,:]).ravel()
            if is_pbc:
                ao = cell.pbc_eval_gto('GTOval', coords, kpts=kpts_lst)
                ao = numpy.asarray(ao).reshape(nkpts, len(coords), nao)
            else:
                ao = cell.eval_gto('GTOval', coords)[None]
            wdv = w * numpy.repeat(dv[r0:r1], na)
            if nproj > 0:
                pblk = proj[:,r0:r1].reshape(nproj, -1) * w
            for k in range(nkpts):
                aok = ao[k]
                if rf.rcut_loc > 0:
                    vpp[k] += lib.dot(aok.conj().T * wdv, aok)
                if nproj > 0:
                    pk[k] += lib.dot(pblk, aok)
            ao = None
        if nproj > 0:
            for k in range(nkpts):
                vpp[k] += lib.dot(pk[k].conj().T, lib.dot(dfull, pk[k]))
        t0 = log.timer_debug1(f'UPF integrals for atom {ia}', *t0)
    return _format(vpp, kpts)


def _format(vpp, kpts):
    from pyscf.pbc.lib.kpts_helper import gamma_point
    if kpts is None or gamma_point(kpts):
        vpp = vpp.real
    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp
