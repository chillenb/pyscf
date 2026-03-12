import os
import contextlib
import random
import numpy
import h5py
from pyscf import lib

from pyscf.lib import logger


from pyscf.pbc.df import aft
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import mpi_df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf.pbc.df.gdf_builder import libpbc, _CCNucBuilder
# from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder, _RSNucBuilder, LINEAR_DEP_THR
from pyscf.pbc.df.mpi_rsdf_builder import _MPIRSGDFBuilder, _MPIRSNucBuilder, LINEAR_DEP_THR
from pyscf.pbc.df.aft import estimate_eta, _check_kpts
from pyscf.pbc.df.df import GDF, make_modrho_basis, _load3c, CDERIArray

from pyscf import __config__


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _gen_unique_name(pre='', suf=''):
    random.seed()
    for seq in range(10000):
        name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        filename = os.path.join(pre + name + suf)
        try:
            f = open(filename, 'x')
        except FileExistsError:
            continue    # try again
        f.close()
        os.unlink(filename)
        return filename
    raise FileExistsError("No usable temporary file name found")

class MPIGDF(GDF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if rank == 0:
            _cderi_to_save = _gen_unique_name(pre='mpigdf', suf='.h5')
        else:
            _cderi_to_save = None
        self._cderi_to_save = comm.bcast(_cderi_to_save, root=0)

    def dump_flags(self, verbose=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            log = logger.new_logger(self, verbose)
            if log.verbose < logger.INFO:
                return self
            log.info('\n')
            log.info('******** %s ********', self.__class__)
            if self.auxcell is None:
                log.info('auxbasis = %s', self.auxbasis)
            else:
                log.info('auxbasis = %s', self.auxcell.basis)
            if self.eta is not None:
                log.info('eta = %s', self.eta)
            if self.mesh is not None:
                log.info('mesh = %s (%d PWs)', self.mesh, numpy.prod(self.mesh))
            log.info('exp_to_discard = %s', self.exp_to_discard)
            if isinstance(self._cderi, str):
                log.info('_cderi = %s  where DF integrals are loaded (readonly).',
                        self._cderi)
            elif isinstance(self._cderi_to_save, str):
                log.info('_cderi_to_save = %s', self._cderi_to_save)
            else:
                log.info('_cderi_to_save = %s', self._cderi_to_save.name)

            kpts = self.kpts
            log.info('len(kpts) = %d', len(kpts))
            log.debug1('    kpts = %s', kpts)
            if self.kpts_band is not None:
                log.info('len(kpts_band) = %d', len(self.kpts_band))
                log.debug1('    kpts_band = %s', self.kpts_band)

            log.info('MPI_COMM_WORLD size: %d', comm.Get_size())
        comm.Barrier()
        return self

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if j_only is not None:
            self._j_only = j_only
        if self.kpts_band is not None:
            self.kpts_band = numpy.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(numpy.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = make_modrho_basis(self.cell, self.auxbasis,
                                         self.exp_to_discard)

        if with_j3c and self._cderi_to_save is not None:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if rank == 0:
                if isinstance(self._cderi, str):
                    if self._cderi == cderi and os.path.isfile(cderi):
                        logger.warn(self, 'File %s (specified by ._cderi) is '
                                    'overwritten by GDF initialization.', cderi)
                        os.remove(cderi)
                    else:
                        logger.warn(self, 'Value of ._cderi is ignored. '
                                    'DF integrals will be saved in file %s .', cderi)
            self._cderi = cderi
            comm.Barrier()
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
            comm.Barrier()
        return self

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        comm = MPI.COMM_WORLD
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(numpy.vstack([self.kpts, self.kpts_band]))[0]

        if self._prefer_ccdf or cell.omega > 0:
            raise NotImplementedError
        dfbuilder = _MPIRSGDFBuilder(cell, auxcell, kpts_union)
        dfbuilder.mesh = self.mesh
        dfbuilder.linear_dep_threshold = self.linear_dep_threshold
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

    def cderi_array(self, label=None):
        '''
        Returns CDERIArray object which provides numpy APIs to access cderi tensor.
        '''
        if label is None:
            label = self._dataname
        if self._cderi is None:
            self.build(j_only=self._j_only)
        return CDERIArray(self._cderi, label)

    def has_kpts(self, kpts):
        if kpts is None:
            return True
        else:
            kpts = numpy.asarray(kpts).reshape(-1,3)
            cached_kpts = self.kpts
            if self.kpts_band is None:
                return all((len(member(kpt, cached_kpts))>0) for kpt in kpts)
            else:
                return all((len(member(kpt, cached_kpts))>0 or
                            len(member(kpt, self.kpts_band))>0) for kpt in kpts)

    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None, aux_slice=None):
        '''Short range part'''
        if self._cderi is None:
            self.build(j_only=self._j_only)
        cell = self.cell
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        is_real = is_zero(kpti_kptj)
        nao = cell.nao_nr()
        if blksize is None:
            if is_real:
                blksize = max_memory*1e6/8/(nao**2*2)
            else:
                blksize = max_memory*1e6/16/(nao**2*2)
            blksize /= 2  # For prefetch
            blksize = max(16, min(int(blksize), self.blockdim))
            logger.debug3(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

        def load(aux_slice):
            b0, b1 = aux_slice
            naux = b1 - b0
            if is_real:
                LpqR = numpy.asarray(j3c[b0:b1].real)
                if compact and LpqR.shape[1] == nao**2:
                    LpqR = lib.pack_tril(LpqR.reshape(naux,nao,nao))
                elif unpack and LpqR.shape[1] != nao**2:
                    LpqR = lib.unpack_tril(LpqR).reshape(naux,nao**2)
                LpqI = numpy.zeros_like(LpqR)
            else:
                Lpq = numpy.asarray(j3c[b0:b1])
                LpqR = numpy.asarray(Lpq.real, order='C')
                LpqI = numpy.asarray(Lpq.imag, order='C')
                Lpq = None
                if compact and LpqR.shape[1] == nao**2:
                    LpqR = lib.pack_tril(LpqR.reshape(naux,nao,nao))
                    LpqI = lib.pack_tril(LpqI.reshape(naux,nao,nao))
                elif unpack and LpqR.shape[1] != nao**2:
                    LpqR = lib.unpack_tril(LpqR).reshape(naux,nao**2)
                    LpqI = lib.unpack_tril(LpqI, lib.ANTIHERMI).reshape(naux,nao**2)
            return LpqR, LpqI

        with _load3c(self._cderi, self._dataname, kpti_kptj) as j3c:
            if aux_slice is None:
                slices = lib.prange(0, j3c.shape[0], blksize)
            else:
                slices = lib.prange(*aux_slice, blksize)
            for LpqR, LpqI in lib.map_with_prefetch(load, slices):
                yield LpqR, LpqI, 1
                LpqR = LpqI = None

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            # Truncated Coulomb operator is not positive definite. Load the
            # CDERI tensor of negative part.
            with _load3c(self._cderi, self._dataname+'-', kpti_kptj,
                         ignore_key_error=True) as j3c:
                if aux_slice is None:
                    slices = lib.prange(0, j3c.shape[0], blksize)
                else:
                    slices = lib.prange(*aux_slice, blksize)
                for LpqR, LpqI in lib.map_with_prefetch(load, slices):
                    yield LpqR, LpqI, -1
                    LpqR = LpqI = None

    def get_pp(self, kpts=None):
        '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.

        The output of this function depends on the input `kpts`. Generally, the
        output is a (Nk, Nao, Nao) array, where Nk is the number of k-points in the
        provided `kpts`. If `kpts` is a (3,) array, corresponding to a single
        k-point, the output will be a (Nao, Nao) matrix. If the optional input
        `kpts` is not specified, this function will read the GDF.kpts for the
        k-mesh and return a (Nk, Nao, Nao) array.

        Note: This API has changed since PySCF-2.10. In PySCF 2.9 (or older), if
        `kpts` is not specified, this funciton may return a (Nao, Nao) matrix for
        the gamma point and a (Nk, Nao, Nao) array for other k-points.
        '''
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            dfbuilder = _CCNucBuilder(cell, kpts).build()
        else:
            dfbuilder = _MPIRSNucBuilder(cell, kpts).build()
        vpp = dfbuilder.get_pp()
        if is_single_kpt:
            vpp = vpp[0]
        return vpp

    def get_nuc(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.

        The output of this function depends on the input `kpts`. Generally, the
        output is a (Nk, Nao, Nao) array, where Nk is the number of k-points in the
        provided `kpts`. If `kpts` is a (3,) array, corresponding to a single
        k-point, the output will be a (Nao, Nao) matrix. If the optional input
        `kpts` is not specified, this function will read the GDF.kpts for the
        k-mesh and return a (Nk, Nao, Nao) array.

        Note: This API has changed since PySCF-2.10. In PySCF 2.9 (or older), if
        `kpts` is not specified, this funciton may return a (Nao, Nao) matrix for
        the gamma point and a (Nk, Nao, Nao) array for other k-points.
        '''
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            dfbuilder = _CCNucBuilder(cell, kpts).build()
        else:
            dfbuilder = _RSNucBuilder(cell, kpts).build()
        nuc = dfbuilder.get_nuc()
        if is_single_kpt:
            nuc = nuc[0]
        return nuc

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None and omega != 0:  # J/K for RSH functionals
            cell = self.cell
            # * AFT is computationally more efficient than GDF if the Coulomb
            #   attenuation tends to the long-range role (i.e. small omega).
            # * Note: changing to AFT integrator may cause small difference to
            #   the GDF integrator.
            # * The sparse mesh is not appropriate for low dimensional systems
            #   with infinity vacuum since the ERI may require large mesh to
            #   sample density in vacuum.
            if (omega > 0 and
                cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum'):
                mydf = aft.AFTDF(cell, self.kpts)
                ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
            else:
                mydf = self
            with mydf.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            return df_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mpi_df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = mpi_df_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = df_ao2mo.get_eri
    ao2mo = get_mo_eri = df_ao2mo.general
    ao2mo_7d = df_ao2mo.ao2mo_7d

    def update_mp(self):
        mf = self.copy()
        mf.with_df = self
        return mf

    def update_cc(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def prange(self, start, stop, step):
        '''This is a hook for MPI parallelization. DO NOT use it out of the
        scope of AFTDF/GDF/MDF.
        '''
        return lib.prange(start, stop, step)

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        '''Creates a temporary density fitting object for RSH-DF integrals.
        In this context, only LR or SR integrals for mol and auxmol are computed.
        '''
        cell = self.cell
        if cell.dimension != 0:
            assert omega < 0

        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            rsh_df._dataname = f'{self._dataname}-sr/{key}'
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        auxcell = getattr(self, 'auxcell', None)

        cell_omega = cell.omega
        cell.omega = omega
        auxcell_omega = None
        if auxcell is not None:
            auxcell_omega = auxcell.omega
            auxcell.omega = omega

        assert rsh_df.cell.omega == omega
        if getattr(rsh_df, 'auxcell', None) is not None:
            assert rsh_df.auxcell.omega == omega

        try:
            yield rsh_df
        finally:
            cell.omega = cell_omega
            if auxcell_omega is not None:
                auxcell.omega = auxcell_omega

################################################################################
# With this function to mimic the molecular DF.loop function, the pbc gamma
# point DF object can be used in the molecular code
    def loop(self, blksize=None):
        cell = self.cell
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise RuntimeError('ERIs of PBC-2D systems are not positive '
                               'definite. Current API only supports positive '
                               'definite ERIs.')

        if blksize is None:
            blksize = self.blockdim
        for LpqR, LpqI, sign in self.sr_loop(compact=True, blksize=blksize):
            # LpqI should be 0 for gamma point DF
            # assert (numpy.linalg.norm(LpqI) < 1e-12)
            yield LpqR

    def get_naoaux(self):
        '''The dimension of auxiliary basis at gamma point'''
# determine naoaux with self._cderi, because DF object may be used as CD
# object when self._cderi is provided.
        if self._cderi is None:
            self.build(j_only=self._j_only)

        cell = self.cell
        if isinstance(self._cderi, numpy.ndarray):
            # self._cderi is likely offered by user. Ensure
            # cderi.shape = (nkpts,naux,nao_pair)
            nao = cell.nao
            if self._cderi.shape[-1] == nao:
                assert self._cderi.ndim == 4
                naux = self._cderi.shape[1]
            elif self._cderi.shape[-1] in (nao**2, nao*(nao+1)//2):
                assert self._cderi.ndim == 3
                naux = self._cderi.shape[1]
            else:
                raise RuntimeError('cderi shape')
            return naux

        # self._cderi['j3c/k_id/seg_id']
        with h5py.File(self._cderi, 'r') as feri:
            key = next(iter(feri[self._dataname].keys()))
            dat = feri[f'{self._dataname}/{key}']
            if isinstance(dat, h5py.Group):
                naux = dat['0'].shape[0]
            else:
                naux = dat.shape[0]

            if (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum' and
                f'{self._dataname}-' in feri):
                key = next(iter(feri[f'{self._dataname}-'].keys()))
                dat = feri[f'{self._dataname}-/{key}']
                if isinstance(dat, h5py.Group):
                    naux += dat['0'].shape[0]
                else:
                    naux += dat.shape[0]
        return naux

