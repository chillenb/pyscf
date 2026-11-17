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
Parser for norm-conserving pseudopotentials in the UPF (v2) format, e.g. the
SG15 ONCV set (http://www.quantum-simulation.org/potentials/sg15_oncv/).

A UPF potential is used in PySCF as a "GTH-like" pseudopotential whose GTH
part only holds the long-range local term  -Z erf(r/(sqrt(2) rloc))/r.
Everything else (the short-range remainder of the local potential and the
nonlocal projectors) is tabulated on a radial grid and integrated numerically,
see pyscf.pbc.gto.pseudo.upf.

The .pseudo input of a Cell accepts
    * a path to a .upf file, e.g. {'C': '/path/C_ONCV_PBE-1.2.upf'}
    * 'sg15' (or 'sg15-1.0', 'sg15-1.1', 'sg15-1.2'), which loads
      <SG15_PATH>/<symb>_ONCV_PBE-<version>.upf. SG15_PATH is taken from
      pbc_gto_pseudo_sg15_path in the pyscf config file, or from the
      environment variable PYSCF_SG15_PATH. Without a version, the newest
      scalar-relativistic version available for the element is used.
'''

import os
import re
import glob
import functools
import xml.etree.ElementTree as ET
import numpy
import scipy.optimize
import scipy.special
from pyscf import __config__

SG15_PATH = getattr(__config__, 'pbc_gto_pseudo_sg15_path',
                    os.environ.get('PYSCF_SG15_PATH'))
_SG15_PATTERN = re.compile(r'SG15(-\d+\.\d+)?$', re.IGNORECASE)

# Bounds for the width of the Gaussian charge whose potential carries the
# long-range part of the local potential.
RLOC_MIN = getattr(__config__, 'pbc_gto_pseudo_upf_rloc_min', 0.25)
RLOC_MAX = getattr(__config__, 'pbc_gto_pseudo_upf_rloc_max', 1.0)

ANGMOM = 'spdfghik'


def is_upf(name):
    '''Whether the pseudo name refers to a UPF potential.'''
    if not isinstance(name, str):
        return False
    if _SG15_PATTERN.match(name.strip()):
        return True
    return name.lower().endswith('.upf') and os.path.isfile(name)


def resolve(name, symb):
    '''Path to the UPF file for element symb.'''
    name = name.strip()
    m = _SG15_PATTERN.match(name)
    if not m:
        if not os.path.isfile(name):
            raise FileNotFoundError(name)
        return os.path.abspath(name)

    sg15_path = SG15_PATH or os.environ.get('PYSCF_SG15_PATH')
    if not sg15_path:
        raise RuntimeError(
            'The SG15 directory is not set. Set pbc_gto_pseudo_sg15_path in '
            'the pyscf config file or the environment variable PYSCF_SG15_PATH '
            'to the directory of sg15_oncv_upf_2020-02-06.tar.gz from '
            'http://www.quantum-simulation.org/potentials/sg15_oncv/')
    symb = ''.join(c for c in symb if c.isalpha())
    if m.group(1):
        fname = os.path.join(sg15_path, f'{symb}_ONCV_PBE{m.group(1)}.upf')
        if not os.path.isfile(fname):
            raise FileNotFoundError(fname)
        return fname
    # newest scalar-relativistic version
    files = glob.glob(os.path.join(sg15_path, f'{symb}_ONCV_PBE-*.upf'))
    versions = []
    for f in files:
        v = re.match(re.escape(symb) + r'_ONCV_PBE-(\d+)\.(\d+)\.upf$',
                     os.path.basename(f))
        if v:
            versions.append(((int(v.group(1)), int(v.group(2))), f))
    if not versions:
        raise FileNotFoundError(f'No SG15 potential for {symb} in {sg15_path}')
    return max(versions)[1]


def _array(node):
    return numpy.array(node.text.split(), dtype=float)


def _valence_config(root, z_valence):
    '''Valence electrons per angular momentum, from the ONCVPSP input in
    PP_INFO if available. Only used for initial guesses and bookkeeping.'''
    info = root.find('PP_INFO')
    inp = None if info is None else info.find('PP_INPUTFILE')
    if inp is not None and inp.text:
        lines = [l.split() for l in inp.text.splitlines()
                 if l.strip() and not l.lstrip().startswith('#')]
        try:
            nc, nv = int(lines[0][2]), int(lines[0][3])
            nelec = []
            for n, l, f in (map(float, x[:3]) for x in lines[1+nc:1+nc+nv]):
                l = int(l)
                nelec.extend([0] * (l + 1 - len(nelec)))
                nelec[l] += f
            if abs(sum(nelec) - z_valence) < 1e-6:
                return [int(x) if float(x).is_integer() else x for x in nelec]
        except (IndexError, ValueError):
            pass
    pswfc = root.find('PP_PSWFC')
    if pswfc is not None:
        nelec = []
        for chi in pswfc:
            l = int(chi.attrib['l'])
            nelec.extend([0] * (l + 1 - len(nelec)))
            nelec[l] += float(chi.attrib.get('occupation', 0))
        if nelec and abs(sum(nelec) - z_valence) < 1e-6:
            return [int(x) if float(x).is_integer() else x for x in nelec]
    return [int(z_valence) if float(z_valence).is_integer() else z_valence]


def _fit_rloc(r, rab, vloc, z):
    '''Width of the Gaussian charge that makes the remainder
    vloc(r) + Z erf(r/(sqrt(2) rloc))/r as small as possible.'''
    mask = r > 0
    r, rab, vloc = r[mask], rab[mask], vloc[mask]
    def cost(rloc):
        dv = vloc + z * scipy.special.erf(r / (numpy.sqrt(2) * rloc)) / r
        return numpy.sum(dv**2 * r**2 * rab)
    res = scipy.optimize.minimize_scalar(cost, bounds=(RLOC_MIN, RLOC_MAX),
                                         method='bounded')
    return float(res.x)


@functools.lru_cache(maxsize=None)
def parse(fname):
    '''Read a norm-conserving UPF v2 file.

    Returns a dict (in atomic units, Hartree):
        z      : valence charge
        nelec  : valence electrons per l
        r, rab : radial mesh and integration weights
        vloc   : local potential on the mesh
        rloc   : width of the long-range Gaussian charge
        beta   : list of (l, r*beta(r) on the mesh, cutoff radius)
        dij    : coupling matrix of the projectors
    '''
    root = ET.parse(fname).getroot()
    if not root.tag.startswith('UPF'):
        raise NotImplementedError(f'{fname}: only UPF v2 files are supported')
    header = root.find('PP_HEADER')
    attr = {k: v.strip() for k, v in header.attrib.items()}
    pp_type = attr.get('pseudo_type', 'NC').upper()
    if pp_type not in ('NC', 'SL'):
        raise NotImplementedError(f'{fname}: pseudo_type {pp_type}; only '
                                  'norm-conserving potentials are supported')
    for key in ('is_ultrasoft', 'is_paw'):
        if attr.get(key, 'F').upper().startswith('T'):
            raise NotImplementedError(f'{fname}: {key}')
    if attr.get('core_correction', 'F').upper().startswith('T'):
        raise NotImplementedError(f'{fname}: nonlinear core correction is '
                                  'not supported')
    if attr.get('has_so', 'F').upper().startswith('T'):
        raise NotImplementedError(f'{fname}: fully relativistic potentials '
                                  'are not supported')
    z = float(attr['z_valence'])

    mesh = root.find('PP_MESH')
    r = _array(mesh.find('PP_R'))
    rab = _array(mesh.find('PP_RAB'))
    # Rydberg -> Hartree
    vloc = .5 * _array(root.find('PP_LOCAL'))
    n = min(len(r), len(vloc))
    r, rab, vloc = r[:n], rab[:n], vloc[:n]

    beta = []
    dij = numpy.zeros((0, 0))
    nl = root.find('PP_NONLOCAL')
    if nl is not None:
        betas = {}
        for child in nl:
            if child.tag.startswith('PP_BETA'):
                idx = int(child.attrib.get('index', child.tag.split('.')[-1]))
                l = int(child.attrib['angular_momentum'])
                rb = _array(child)
                icut = int(child.attrib.get('cutoff_radius_index', len(rb)))
                icut = min(icut, len(rb), n)
                rb = numpy.append(rb[:icut], numpy.zeros(n - icut))
                rcut = r[min(icut, n-1)]
                betas[idx] = (l, rb, rcut)
            elif child.tag == 'PP_DIJ':
                dij = _array(child)
        beta = [betas[i] for i in sorted(betas)]
        nproj = len(beta)
        dij = .5 * dij.reshape(nproj, nproj)  # Rydberg -> Hartree

    return {
        'file': fname,
        'z': z,
        'nelec': _valence_config(root, z),
        'r': r,
        'rab': rab,
        'vloc': vloc,
        'rloc': _fit_rloc(r, rab, vloc, z),
        'beta': beta,
        'dij': dij,
    }


def load(name, symb):
    '''GTH-format entry for cell._pseudo. It only holds the long-range local
    part; the rest is evaluated by pyscf.pbc.gto.pseudo.upf.'''
    data = parse(resolve(name, symb))
    nelec = list(data['nelec'])
    return [nelec, data['rloc'], 0, [], 0]
