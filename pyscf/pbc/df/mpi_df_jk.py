
import numpy
from pyscf import lib
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import __config__

from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_jks, _format_kpts_band
from pyscf.pbc.df.df_jk import _format_mo, _mo_from_dm, _format_dms

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    log.info("MPI DF JK get_j_kpts")
    t0 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        raise NotImplementedError("kpts_band unsupported")

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        # If mydf._cderi is the file that generated from another calculation,
        # guess naux based on the contents of the integral file.
        naux = mydf.get_naoaux()
    else:
        naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band) and not numpy.iscomplexobj(dms)

    t1 = (logger.process_clock(), logger.perf_counter())
    dmsR = dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    rhoR = numpy.zeros((nset,naux))
    rhoI = numpy.zeros((nset,naux))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    nkpts = len(kpts)
    sub_ranges = numpy.array_split(numpy.arange(nkpts), comm_size)
    my_range = sub_ranges[rank]
    print(f"rank: {rank}, my_range: {my_range}")


    for k, kpt in zip(my_range, kpts[my_range]):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).reshape(-1,nao,nao)
            #:rhoR[:,p0:p1] += numpy.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).real
            #:rhoI[:,p0:p1] += numpy.einsum('Lpq,xqp->xL', Lpq, dms[:,k]).imag
            rhoR[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoI[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            if LpqI is not None:
                rhoR[:,p0:p1] -= sign * numpy.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
                rhoI[:,p0:p1] += sign * numpy.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            LpqR = LpqI = None

    allreduce_inplace_contiguous(comm, rhoR)
    allreduce_inplace_contiguous(comm, rhoI)


    t1 = log.timer_debug1('get_j pass 1', *t1)

    weight = 1./nkpts
    rhoR *= weight
    rhoI *= weight
    if hermi == 0:
        aos2symm = False
        vjR = numpy.zeros((nset,nband,nao**2))
        vjI = numpy.zeros((nset,nband,nao**2))
    else:
        aos2symm = True
        vjR = numpy.zeros((nset,nband,nao_pair))
        vjI = numpy.zeros((nset,nband,nao_pair))

    nkpts_band = len(kpts_band)
    sub_ranges = numpy.array_split(numpy.arange(nkpts_band), comm_size)
    my_range = sub_ranges[rank]

    for k, kpt in zip(my_range, kpts_band[my_range]):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, sign in mydf.sr_loop(kptii, max_memory, aos2symm):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j)#.reshape(-1,nao,nao)
            #:vjR[:,k] += numpy.dot(rho[:,p0:p1], Lpq).real
            #:vjI[:,k] += numpy.dot(rho[:,p0:p1], Lpq).imag
            vjR[:,k] += numpy.dot(rhoR[:,p0:p1], LpqR)
            if not j_real:
                vjI[:,k] += numpy.dot(rhoI[:,p0:p1], LpqR)
                if LpqI is not None:
                    vjR[:,k] -= numpy.dot(rhoI[:,p0:p1], LpqI)
                    vjI[:,k] += numpy.dot(rhoR[:,p0:p1], LpqI)
            LpqR = LpqI = None

    allreduce_inplace_contiguous(comm, vjR)
    if not j_real:
        allreduce_inplace_contiguous(comm, vjI)

    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    if aos2symm:
        vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))
    vj_kpts = vj_kpts.reshape(nset,nband,nao,nao)

    log.timer('get_j', *t0)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)


def get_j_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=numpy.zeros((1,3)), kpts_band=None):
    raise NotImplementedError

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)

    log.info("mpi_df_jk get_k_kpts")

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t0 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(j_only=False, kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_k_kpts', *t0)
    elif mydf._j_only:
        log.warn('DF integrals for HF exchange were not initialized. '
                 'df.j_only cannot be used with hybrid functional. DF integrals will be rebuilt.')
        mydf.build(j_only=False, kpts_band=kpts_band)

    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    if mo_coeff is not None:
        mo_occ = dm_kpts.mo_occ

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]



    skmoR = skmo2R = None
    if not mydf.force_dm_kbuild:
        if mo_coeff is not None:
            if isinstance(mo_coeff[0], (list, tuple)) or (isinstance(mo_coeff[0], numpy.ndarray)
                                                          and mo_coeff[0].ndim == 3):
                mo_coeff = [mo for mo1 in mo_coeff for mo in mo1]
            if len(mo_coeff) != nset*nkpts: # wrong shape
                log.warn('mo_coeff from dm tag has wrong shape. '
                         'Calculating mo from dm instead.')
                mo_coeff = None
            elif isinstance(mo_occ[0], (list, tuple)) or (isinstance(mo_occ[0], numpy.ndarray)
                                                          and mo_occ[0].ndim == 2):
                mo_occ = [mo for mo1 in mo_occ for mo in mo1]
        if mo_coeff is not None:
            skmoR, skmoI = _format_mo(mo_coeff, mo_occ, shape=(nset,nkpts), order='F',
                                      precision=cell.precision)
        elif hermi == 1:
            skmoR, skmoI = _mo_from_dm(dms.reshape(-1,nao,nao), method='eigh',
                                       shape=(nset,nkpts), order='F',
                                       precision=cell.precision)
            if skmoR is None:
                log.debug1('get_k_kpts: Eigh fails for input dm due to non-PSD. '
                           'Try SVD instead.')
        if skmoR is None:
            skmoR, skmoI, skmo2R, skmo2I = _mo_from_dm(dms.reshape(-1,nao,nao),
                                                   method='svd', shape=(nset,nkpts),
                                                   order='F', precision=cell.precision)
            if skmoR[0,0].shape[1] > nao//2:
                log.debug1('get_k_kpts: rank(dm) = %d exceeds half of nao = %d. '
                           'Fall back to DM-based build.', skmoR[0,0].shape[1], nao)
                skmoR = skmo2R = None

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vkR = numpy.zeros((nset,nband,nao,nao))
    vkI = numpy.zeros((nset,nband,nao,nao))

    tspans = numpy.zeros((7,2))
    tspannames = ['buf1', 'ct11', 'ct12', 'buf2', 'ct21', 'ct22', 'load']
    if rank == 0:
        log.info(f"skmoR: {skmoR is not None}")
        log.info(f"skmo2R: {skmo2R is not None}")

    # K(p,q; k2 from k1)
    #     = V(r k1, q k2, p k2, s k1) * D(s,r; k1)
    #     = V(L, r k1, q k2) * V(L, s k1, p k2).conj() * D(s,r; k1)         eqn (1)
    # --> in case of Hermitian & PSD DM
    #     = ( V(L, s k1, p k2) * C(s,i; k1).conj() ).conj()
    #       * V(L, r k1, q k2) * C(r,i; k1).conj()                          eqn (2)
    #     = W(L, i k1, p k2).conj() * W(L, i k1, q k2)                      eqn (3)
    # --> in case of non-Hermitian or non-PSD DM
    #     = ( V(L, s k1, p k2) * A(s,i; k1).conj() ).conj()
    #       * V(L, r k1, q k2) * B(r,i; k1).conj()                          eqn (4)
    #     = X(L, i k1, p k2).conj() * Y(L, i k1, q k2)                      eqn (5)

    # if swap_2e:
    # K(p,q; k1 from k2)
    #     = V(p k1, s k2, r k2, q k1) * D(s,r; k2)
    #     = V(L, p k1, s k2) * V(L, q k1, r k2).conj() * D(s,r; k2)         eqn (1')
    # --> in case of Hermitian & PSD DM
    #     = V(L, p k1, s k2) * C(s,i; k2)
    #       * ( V(L, q k1, r k2) * C(r,i; k2) ).conj()                      eqn (2')
    #     = W(L, p k1, i k2) * W(L, q k1, i k2).conj()                      eqn (3')
    # --> in case of non-Hermitian or non-PSD DM
    #     = V(L, p k1, s k2) * A(s,i; k2)
    #       * ( V(L, q k1, r k2) * B(r,i; k2) ).conj()                      eqn (4')
    #     = X(L, p k1, i k2) * Y(L, q k1, i k2).conj()                      eqn (5')

    # Mode 1: DM-based K-build uses eqn (1) and eqn (1')
    # Mode 2: Symm MO-based K-build uses eqns (2,3) and eqns (2',3')
    # Mode 3: Asymm MO-based K-build uses eqns (4,5) and eqns (4',5')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )

    if skmoR is None: # input dm is not Hermitian/PSD --> build K from dm
        log.debug2('get_k_kpts: build K from dm')
        dmsR = numpy.asarray(dms.real, order='C')
        dmsI = numpy.asarray(dms.imag, order='C')
        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        def make_kpt(ki, kj, swap_2e, inverse_idx=None):
            kpti = kpts[ki]
            kptj = kpts_band[kj]

            tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

            for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
                nrow = LpqR.shape[0]

                tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[6] += tick - tock

                pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                tmpR = numpy.ndarray((nao,nrow*nao), buffer=LpqR)
                tmpI = numpy.ndarray((nao,nrow*nao), buffer=LpqI)
                pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick

                for i in range(nset):
                    zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                           pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[1] += tick - tock
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                           sign, vkR[i,kj], vkI[i,kj], 1)
                    tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[2] += tock - tick

                if swap_2e:
                    tmpR = tmpR.reshape(nao*nrow,nao)
                    tmpI = tmpI.reshape(nao*nrow,nao)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                    ki_tmp = ki
                    kj_tmp = kj
                    if inverse_idx:
                        ki_tmp = inverse_idx[0]
                        kj_tmp = inverse_idx[1]
                    for i in range(nset):
                        zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                               dmsR[i,kj_tmp], dmsI[i,kj_tmp], 1, tmpR, tmpI)
                        tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[4] += tock - tick
                        zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                               pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                               sign, vkR[i,ki_tmp], vkI[i,ki_tmp], 1)
                        tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[5] += tick - tock

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

                LpqR = LpqI = pLqR = pLqI = tmpR = tmpI = None
    elif skmo2R is None:
        log.debug2('get_k_kpts: build K from symm mo coeff')
        nmo = skmoR[0,0].shape[1]
        log.debug2('get_k_kpts: rank(dm) = %d / %d', nmo, nao)
        skmoI_mask = numpy.asarray([[abs(skmoI[i,k]).max() > cell.precision
                                     for k in range(nkpts)] for i in range(nset)])
        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        def make_kpt(ki, kj, swap_2e, inverse_idx=None):
            kpti = kpts[ki]
            kptj = kpts_band[kj]

            tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

            for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
                nrow = LpqR.shape[0]

                tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[6] += tick - tock

                pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                tmpR = numpy.ndarray((nmo,nrow*nao), buffer=LpqR)
                tmpI = numpy.ndarray((nmo,nrow*nao), buffer=LpqI)
                pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick

                for i in range(nset):
                    moR = skmoR[i,ki]
                    if skmoI_mask[i,ki]:
                        moI = skmoI[i,ki]
                        zdotCN(moR.T, moI.T, pLqR.reshape(nao,-1), pLqI.reshape(nao,-1),
                               1, tmpR, tmpI)
                    else:
                        lib.ddot(moR.T, pLqR.reshape(nao,-1), 1, tmpR)
                        lib.ddot(moR.T, pLqI.reshape(nao,-1), 1, tmpI)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[1] += tick - tock
                    zdotCN(tmpR.reshape(-1,nao).T, tmpI.reshape(-1,nao).T,
                           tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                           sign, vkR[i,kj], vkI[i,kj], 1)
                    tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[2] += tock - tick

                if swap_2e:
                    tmpR = tmpR.reshape(nrow*nao,nmo)
                    tmpI = tmpI.reshape(nrow*nao,nmo)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                    ki_tmp = ki
                    kj_tmp = kj
                    if inverse_idx:
                        ki_tmp = inverse_idx[0]
                        kj_tmp = inverse_idx[1]
                    for i in range(nset):
                        moR = skmoR[i,kj_tmp]
                        if skmoI_mask[i,kj_tmp]:
                            moI = skmoI[i,kj_tmp]
                            zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao), moR, moI,
                                   1, tmpR, tmpI)
                        else:
                            lib.ddot(pLqR.reshape(-1,nao), moR, 1, tmpR)
                            lib.ddot(pLqI.reshape(-1,nao), moR, 1, tmpI)
                        tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[4] += tock - tick
                        zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                               tmpR.reshape(nao,-1).T, tmpI.reshape(nao,-1).T,
                               sign, vkR[i,ki_tmp], vkI[i,ki_tmp], 1)
                        tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[5] += tick - tock

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

                LpqR = LpqI = pLqR = pLqI = tmpR = tmpI = None
    else:
        log.debug2('get_k_kpts: build K from asymm mo coeff')
        skmo1R = skmoR
        skmo1I = skmoI
        nmo = skmoR[0,0].shape[1]
        log.debug2('get_k_kpts: rank(dm) = %d / %d', nmo, nao)
        skmoI_mask = numpy.asarray([[max(abs(skmo1I[i,k]).max(),
                                         abs(skmo2I[i,k]).max()) > cell.precision
                                     for k in range(nkpts)] for i in range(nset)])
        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        def make_kpt(ki, kj, swap_2e, inverse_idx=None):
            kpti = kpts[ki]
            kptj = kpts_band[kj]

            tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

            for LpqR, LpqI, sign in mydf.sr_loop((kpti,kptj), max_memory, False):
                nrow = LpqR.shape[0]

                tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[6] += tick - tock

                pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                tmp1R = numpy.ndarray((nmo,nrow*nao), buffer=LpqR)
                tmp1I = numpy.ndarray((nmo,nrow*nao), buffer=LpqI)
                tmp2R = numpy.ndarray((nmo,nrow*nao),
                                      buffer=LpqR.reshape(-1)[tmp1R.size:])
                tmp2I = numpy.ndarray((nmo,nrow*nao),
                                      buffer=LpqI.reshape(-1)[tmp1I.size:])
                pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += tock - tick

                for i in range(nset):
                    mo1R = skmo1R[i,ki]
                    mo2R = skmo2R[i,ki]
                    if skmoI_mask[i,ki]:
                        mo1I = skmo1I[i,ki]
                        mo2I = skmo2I[i,ki]
                        zdotCN(mo1R.T, mo1I.T, pLqR.reshape(nao,-1), pLqI.reshape(nao,-1),
                               1, tmp1R, tmp1I)
                        zdotCN(mo2R.T, mo2I.T, pLqR.reshape(nao,-1), pLqI.reshape(nao,-1),
                               1, tmp2R, tmp2I)
                    else:
                        lib.ddot(mo1R.T, pLqR.reshape(nao,-1), 1, tmp1R)
                        lib.ddot(mo1R.T, pLqI.reshape(nao,-1), 1, tmp1I)
                        lib.ddot(mo2R.T, pLqR.reshape(nao,-1), 1, tmp2R)
                        lib.ddot(mo2R.T, pLqI.reshape(nao,-1), 1, tmp2I)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[1] += tick - tock
                    zdotCN(tmp1R.reshape(-1,nao).T, tmp1I.reshape(-1,nao).T,
                           tmp2R.reshape(-1,nao), tmp2I.reshape(-1,nao),
                           sign, vkR[i,kj], vkI[i,kj], 1)
                    tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[2] += tock - tick

                if swap_2e:
                    tmp1R = tmp1R.reshape(nrow*nao,nmo)
                    tmp1I = tmp1I.reshape(nrow*nao,nmo)
                    tmp2R = tmp2R.reshape(nrow*nao,nmo)
                    tmp2I = tmp2I.reshape(nrow*nao,nmo)
                    tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += tick - tock
                    ki_tmp = ki
                    kj_tmp = kj
                    if inverse_idx:
                        ki_tmp = inverse_idx[0]
                        kj_tmp = inverse_idx[1]
                    for i in range(nset):
                        mo1R = skmo1R[i,kj_tmp]
                        mo2R = skmo2R[i,kj_tmp]
                        if skmoI_mask[i,kj_tmp]:
                            mo1I = skmo1I[i,kj_tmp]
                            mo2I = skmo2I[i,kj_tmp]
                            zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao), mo1R, mo1I,
                                   1, tmp1R, tmp1I)
                            zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao), mo2R, mo2I,
                                   1, tmp2R, tmp2I)
                        else:
                            lib.ddot(pLqR.reshape(-1,nao), mo1R, 1, tmp1R)
                            lib.ddot(pLqI.reshape(-1,nao), mo1R, 1, tmp1I)
                            lib.ddot(pLqR.reshape(-1,nao), mo2R, 1, tmp2R)
                            lib.ddot(pLqI.reshape(-1,nao), mo2R, 1, tmp2I)
                        tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[4] += tock - tick
                        zdotNC(tmp1R.reshape(nao,-1), tmp1I.reshape(nao,-1),
                               tmp2R.reshape(nao,-1).T, tmp2I.reshape(nao,-1).T,
                               sign, vkR[i,ki_tmp], vkI[i,ki_tmp], 1)
                        tick = numpy.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[5] += tick - tock

                tock = numpy.asarray((logger.process_clock(), logger.perf_counter()))

                LpqR = LpqI = pLqR = pLqI = tmp1R = tmp1I = tmp2R = tmp2I = None

    t1 = (logger.process_clock(), logger.perf_counter())



    make_kpt_args_list = []

    if kpts_band is kpts:  # normal k-points HF/DFT
        for ki in range(nkpts):
            for kj in range(ki):
                make_kpt_args_list.append((ki, kj, True))
            make_kpt_args_list.append((ki, ki, False))
    else:
        raise NotImplementedError

    ntasks = len(make_kpt_args_list)
    sub_ranges = list(lib.prange_split(ntasks, size))
    my_range = sub_ranges[rank]

    for itask in range(*my_range):
        task = make_kpt_args_list[itask]
        make_kpt(*task)

    t1 = log.timer_debug1('get_k_kpts: make_kpt ki>=kj (%d,*)'%ki, *t1)


    for tspan, tspanname in zip(tspans,tspannames):
        log.debug1('    CPU time for %s %10.2f sec, wall time %10.2f sec',
                   tspanname, *tspan)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        allreduce_inplace_contiguous(comm, vkR)
        vk_kpts = vkR
    else:
        allreduce_inplace_contiguous(comm, vkR)
        allreduce_inplace_contiguous(comm, vkI)
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    if exxdiv == 'ewald' and cell.dimension != 0:
        # Integrals are computed analytically in GDF and RSJK.
        # Finite size correction for exx is not needed.
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    log.timer('get_k_kpts', *t0)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def get_k_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=numpy.zeros((1,3)), kpts_band=None,
                      exxdiv=None):
    raise NotImplementedError


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf

    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    mf = pscf.RHF(cell)
    dm = mf.get_init_guess()
    auxbasis = 'weigend'
    #from pyscf import df
    #auxbasis = df.addons.aug_etb_for_dfbasis(cell, beta=1.5, start_at=0)
    #from pyscf.pbc.df import mdf
    #mf.with_df = mdf.MDF(cell)
    #mf.auxbasis = auxbasis
    mf = density_fit(mf, auxbasis)
    mf.with_df.mesh = (n,) * 3
    vj = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv, with_k=False)[0]
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    vj, vk = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.348163681114187')
    print(numpy.einsum('ij,ji->', mf.get_hcore(cell), dm), 'ref=-75.5758086593503')

    kpts = cell.make_kpts([2]*3)[:4]
    from pyscf.pbc.df import DF
    with_df = DF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.mesh = [n] * 3
    dms = numpy.array([dm]*len(kpts))
    vj, vk = with_df.get_jk(dms, exxdiv=mf.exxdiv, kpts=kpts)
    print(numpy.einsum('ij,ji->', vj[0], dms[0]) - 46.69784067248350)
    print(numpy.einsum('ij,ji->', vj[1], dms[1]) - 46.69814992718212)
    print(numpy.einsum('ij,ji->', vj[2], dms[2]) - 46.69526120279135)
    print(numpy.einsum('ij,ji->', vj[3], dms[3]) - 46.69570739526301)
    print(numpy.einsum('ij,ji->', vk[0], dms[0]) - 37.26974254415191)
    print(numpy.einsum('ij,ji->', vk[1], dms[1]) - 37.27001407288309)
    print(numpy.einsum('ij,ji->', vk[2], dms[2]) - 37.27000643285160)
    print(numpy.einsum('ij,ji->', vk[3], dms[3]) - 37.27010299675364)
