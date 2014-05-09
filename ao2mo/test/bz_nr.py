#!/usr/bin/env python

import scf
import gto

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_bz'

mol.atom.extend([
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C", ( 0.73685281,  0.61123287, -0.00800148)],
    ["C", ( 1.43439081,  1.81898387, -0.00800148)],
    ["C", ( 0.73673681,  3.02749287, -0.00920048)],
    ["C", (-0.65808819,  3.02741487, -0.00967948)],
    ["C", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],
    ["H", ( 2.53407081,  1.81906387, -0.00736748)],
    ["H", ( 1.28693681,  3.97963587, -0.00925948)],
    ["H", (-1.20821019,  3.97969587, -0.01063248)],
    ["H", (-2.45529319,  1.81939187, -0.00886348)],])


mol.basis = {"H": 'ccpvdz',
             "C": 'ccpvdz',}

mol.build()

##############
# SCF result
import time
rhf = scf.RHF(mol)
print 'E_RHF =', rhf.scf()
print time.clock()

import os
import tempfile
import numpy
import h5py
import ao2mo
f, eritmp = tempfile.mkstemp()
os.close(f)

nocc = mol.nelectron / 2
co = rhf.mo_coeff[:,:nocc]
cv = rhf.mo_coeff[:,nocc:]
ao2mo.direct.general(mol, (co,cv,co,cv), eritmp, max_memory=100, dataname='mp2_bz')
f = h5py.File(eritmp, 'r')
g = f['mp2_bz']
g = numpy.array(g) # copy to memory, otherwise will be very slow to loop over it
f.close()
os.remove(eritmp)

print time.clock()
eia = rhf.mo_energy[:nocc].reshape(nocc,1) - rhf.mo_energy[nocc:]
nvir = eia.shape[1]
emp2 = 0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            ia = i * nvir + a
            ja = j * nvir + a
            for b in range(nvir):
                ib = i * nvir + b
                jb = j * nvir + b
                emp2 += g[ia,jb] * (g[ia,jb]*2-g[ib,ja]) \
                        / (eia[i,a]+eia[j,b])
print 'E_MP2 =', emp2 # -0.80003653259
print time.clock()

