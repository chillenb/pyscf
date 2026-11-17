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
UPF pseudopotentials. A GTH potential tabulated in the UPF format must give
the same integrals as the analytic GTH code.
'''

import os
import tempfile
import unittest
import numpy as np
from scipy.special import erf, gamma
from pyscf.gto.basis import load_pseudo, parse_upf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import df
from pyscf.pbc.gto.pseudo import upf


def gth_to_upf(pp, fname, dr=0.002, rmax=8.0):
    '''Write a GTH potential (pyscf format) as a minimal UPF v2 file.'''
    nelec, rloc, nexp, cexp = pp[0], pp[1], pp[2], pp[3]
    z = sum(nelec)
    r = np.arange(0, rmax + dr/2, dr)
    rb = r / rloc
    with np.errstate(divide='ignore', invalid='ignore'):
        vloc = np.where(r > 0, -z / r * erf(rb / np.sqrt(2)),
                        -z * np.sqrt(2/np.pi) / rloc)
    vloc += np.exp(-rb**2 / 2) * sum(c * rb**(2*i) for i, c in enumerate(cexp))
    betas, blocks = [], []
    for l, (rl, nl, hl) in enumerate(p[:3] for p in pp[5:]):
        for i in range(1, nl+1):
            p = (np.sqrt(2) * r**(l + 2*(i-1)) * np.exp(-r**2 / (2*rl**2)) /
                 (rl**(l + (4*i-1)/2) * np.sqrt(gamma(l + (4*i-1)/2))))
            betas.append((l, r * p))
        if nl > 0:
            blocks.append(np.asarray(hl))
    nproj = len(betas)
    dij = np.zeros((nproj, nproj))
    p0 = 0
    for hl in blocks:
        dij[p0:p0+len(hl), p0:p0+len(hl)] = 2 * hl  # Ha -> Ry
        p0 += len(hl)
    fmt = lambda a: ' '.join('%.15e' % x for x in a)
    out = ['<UPF version="2.0.1">',
           f'<PP_HEADER pseudo_type="NC" z_valence="{z:.1f}" core_correction="F"/>',
           '<PP_MESH>', f'<PP_R>{fmt(r)}</PP_R>',
           f'<PP_RAB>{fmt(np.full_like(r, dr))}</PP_RAB>', '</PP_MESH>',
           f'<PP_LOCAL>{fmt(2 * vloc)}</PP_LOCAL>', '<PP_NONLOCAL>']
    for i, (l, rp) in enumerate(betas):
        icut = int(np.where(abs(rp) > 1e-14)[0][-1]) + 2
        out.append(f'<PP_BETA.{i+1} index="{i+1}" angular_momentum="{l}" '
                   f'cutoff_radius_index="{icut}">{fmt(rp)}</PP_BETA.{i+1}>')
    out += [f'<PP_DIJ>{fmt(dij.ravel())}</PP_DIJ>', '</PP_NONLOCAL>', '</UPF>']
    with open(fname, 'w') as f:
        f.write('\n'.join(out))


def setUpModule():
    global tmpdir, cells, files
    tmpdir = tempfile.TemporaryDirectory()
    files = {}
    for el in ('C', 'O', 'Cu'):
        files[el] = os.path.join(tmpdir.name, f'{el}.upf')
        gth_to_upf(load_pseudo('gth-pade', el), files[el])
    a = 1.78339987
    diamond = dict(atom='C 0 0 0; C .89169994 .89169994 .89169994',
                   a=[[0, a, a], [a, 0, a], [a, a, 0]], basis='gth-dzvp')
    cu = dict(atom='Cu 0 0 0', a=[[0, 1.805, 1.805], [1.805, 0, 1.805], [1.805, 1.805, 0]],
              basis='gth-szv-molopt-sr', spin=1)
    cells = {}
    for name, kw, el in (('diamond', diamond, 'C'), ('cu', cu, 'Cu')):
        ref = pbcgto.Cell(pseudo='gth-pade', verbose=0, **kw).build()
        new = pbcgto.Cell(pseudo={el: files[el]}, verbose=0, **kw).build()
        cells[name] = (ref, new)

def tearDownModule():
    global tmpdir, cells, files
    tmpdir.cleanup()
    del tmpdir, cells, files


class KnownValues(unittest.TestCase):
    def test_build(self):
        ref, new = cells['diamond']
        self.assertEqual(ref.nelectron, new.nelectron)
        self.assertTrue(upf.has_upf(new))
        self.assertFalse(upf.has_upf(ref))
        # the GTH entry only holds the long-range local part
        pp = new._pseudo['C']
        self.assertEqual(pp[2], 0)
        self.assertEqual(pp[4], 0)
        new1 = pbcgto.loads(new.dumps())
        self.assertTrue(upf.has_upf(new1))

    def test_gdf_kpts(self):
        for name in ('diamond', 'cu'):
            ref, new = cells[name]
            kpts = ref.make_kpts([2, 1, 1])
            v0 = df.GDF(ref, kpts).get_pp(kpts)
            v1 = df.GDF(new, kpts).get_pp(kpts)
            self.assertAlmostEqual(abs(v0 - v1).max(), 0, 7)

    def test_fftdf_gamma(self):
        for name in ('diamond', 'cu'):
            ref, new = cells[name]
            v0 = df.FFTDF(ref).get_pp(np.zeros(3))
            v1 = df.FFTDF(new).get_pp(np.zeros(3))
            self.assertEqual(v1.dtype, np.double)
            self.assertAlmostEqual(abs(v0 - v1).max(), 0, 7)

    def test_mixed(self):
        # one atom with GTH, one with UPF
        ref, new = cells['diamond']
        mixed = ref.copy()
        mixed.atom = 'C 0 0 0; C1 .89169994 .89169994 .89169994'
        mixed.basis = {'C': 'gth-dzvp', 'C1': 'gth-dzvp'}
        mixed.pseudo = {'C': 'gth-pade', 'C1': upf.upf_pseudo(new)['C']['file']}
        mixed.build()
        v0 = df.GDF(ref).get_pp()
        v1 = df.GDF(mixed).get_pp()
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 7)

    def test_molecule(self):
        from pyscf import gto, scf
        kw = dict(atom='C 0 0 0; O 0 0 1.128', basis='gth-dzvp', verbose=0)
        ref = gto.M(pseudo='gth-pade', **kw)
        new = gto.M(pseudo={'C': files['C'], 'O': files['O']}, **kw)
        self.assertEqual(ref.nelectron, new.nelectron)
        h0 = scf.hf.get_hcore(ref)
        h1 = scf.hf.get_hcore(new)
        self.assertAlmostEqual(abs(h0 - h1).max(), 0, 7)

    @unittest.skipIf(not (parse_upf.SG15_PATH or os.environ.get('PYSCF_SG15_PATH')),
                     'SG15 directory not set')
    def test_sg15(self):
        data = parse_upf.parse(parse_upf.resolve('sg15', 'C'))
        self.assertEqual(data['z'], 4)
        self.assertEqual(data['nelec'], [2, 2])
        self.assertEqual([b[0] for b in data['beta']], [0, 0, 1, 1])


if __name__ == '__main__':
    print('Full Tests for UPF pseudopotentials')
    unittest.main()
