from pyscf import lib
from pyscf.lib.numpy_helper import _np_helper
libgw = lib.load_library('libgw')
import numpy as np
import ctypes
from ctypes import c_int
from scipy.linalg import blas

# Use ndpointer and argtypes
# to make the ctypes function calls more readable
libgw.dmul_Lia_response.restype = None
libgw.dmul_Lia_response.argtypes = [
    # int naux, int nocc, int nvir
    c_int, c_int, c_int,
    # double *Lpq
    np.ctypeslib.ndpointer(dtype=np.double, ndim=3),
    # size_t lstride0, size_t lstride1
    ctypes.c_size_t, ctypes.c_size_t,
    # double *mo_energy_occ
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
    # double *mo_energy_vir
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
    # double *out
    np.ctypeslib.ndpointer(dtype=np.double, ndim=3),
    # size_t ostride0, size_t ostride1
    ctypes.c_size_t, ctypes.c_size_t,
    # double alpha, double omega
    ctypes.c_double, ctypes.c_double
]

_np_helper.NPomp_dcopy_012.restype = None
_np_helper.NPomp_dcopy_012.argtypes = [
    # size_t ishape0, size_t ishape1, size_t ishape2
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    # double *in
    np.ctypeslib.ndpointer(dtype=np.double, ndim=3),
    # size_t istride0, size_t istride1
    ctypes.c_size_t, ctypes.c_size_t,
    # double *out
    np.ctypeslib.ndpointer(dtype=np.double, ndim=3),
    # size_t ostride0, size_t ostride1
    ctypes.c_size_t, ctypes.c_size_t
]

_np_helper.NPomp_zcopy_012.restype = None
_np_helper.NPomp_zcopy_012.argtypes = [
    # int conja
    ctypes.c_int,
    # size_t ishape0, size_t ishape1, size_t ishape2
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    # double *in
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=3),
    # size_t istride0, size_t istride1
    ctypes.c_size_t, ctypes.c_size_t,
    # double *out
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=3),
    # size_t ostride0, size_t ostride1
    ctypes.c_size_t, ctypes.c_size_t
]

def copy_Lia_nocc_slice(Lia,
                         out=None,
                         nocc_range: tuple[int, int]|None = None,
                         conj=False):
    """Copy a chunk of the Lia array along the nocc axis.
    This is done to make the slice BLAS-able.
    It's much faster than np.ascontiguousarray.

    Parameters
    ----------
    Lia : np.ndarray
        OV block of cderi
    out : np.ndarray, optional
        output buffer, by default None
    nocc_range : tuple[int, int] | None, optional
        (start, end) indices

    Returns
    -------
    np.ndarray
        array containing Lia[:, nocc_range[0]:nocc_range[1], :]
    """
    naux, nocc, nvir = Lia.shape

    # make sure nocc_range is valid
    if nocc_range is None:
        nocc_range = (0, nocc)
    else:
        assert nocc_range[0] >= 0 and nocc_range[1] <= nocc
        assert nocc_range[0] < nocc_range[1]
    n_nocc = nocc_range[1] - nocc_range[0]

    # create output array
    out_arr = np.ndarray(shape=(naux, n_nocc, nvir), dtype=Lia.dtype, buffer=out)
    assert out_arr.flags.c_contiguous

    # slicing and strides
    Lia_slice = Lia[:, nocc_range[0]:nocc_range[1], :]
    liastrides = [s // Lia_slice.itemsize for s in Lia_slice.strides]
    outstrides = [s // out_arr.itemsize for s in out_arr.strides]

    if Lia.dtype == np.complex128:
        if conj:
            _np_helper.NPomp_zcopy_012(
                1, naux, n_nocc, nvir,
                Lia_slice, liastrides[0], liastrides[1],
                out_arr, outstrides[0], outstrides[1]
            )
        else:
            _np_helper.NPomp_zcopy_012(
                0, naux, n_nocc, nvir,
                Lia_slice, liastrides[0], liastrides[1],
                out_arr, outstrides[0], outstrides[1]
            )
    else:
        _np_helper.NPomp_dcopy_012(
            naux, n_nocc, nvir,
            Lia_slice, liastrides[0], liastrides[1],
            out_arr, outstrides[0], outstrides[1]
        )

    return out_arr

def dmul_Lia_response(Lia, mo_energy, omega, alpha=4.0, out=None,
                     nocc_range: tuple[int, int]|None = None):
    """
    Calculates Pia = alpha * Lia * (eia) / (eia**2 + omega**2)
    for nocc_range[0] <= i < nocc_range[1]

    Parameters
    ----------
    Lia : np.ndarray
        OV block of cderi
    mo_energy : np.ndarray
        MO energies
    omega : float
        imaginary part of the frequency
    alpha : float, optional
        scale factor, by default 4.0
    out : np.ndarray, optional
        buffer for output, by default None
    nocc_slice : tuple[int] | None, optional
        start and end indices of desired occupied orbitals, by default None

    Returns
    -------
    np.ndarray
        Pia
    """
    naux, nocc, nvir = Lia.shape

    # make sure nocc_range is valid
    if nocc_range is None:
        nocc_range = (0, nocc)
    else:
        assert nocc_range[0] >= 0 and nocc_range[1] <= nocc
        assert nocc_range[0] < nocc_range[1]
    n_nocc = nocc_range[1] - nocc_range[0]

    # create output array
    Pia = np.ndarray(shape=(naux, n_nocc, nvir), dtype=Lia.dtype, buffer=out)

    # slicing and strides
    Lia_slice = Lia[:, nocc_range[0]:nocc_range[1], :]
    liastrides = [s // Lia_slice.itemsize for s in Lia_slice.strides]
    piastrides = [s // Pia.itemsize for s in Pia.strides]

    # do the work.
    libgw.dmul_Lia_response(
        naux, n_nocc, nvir,
        Lia_slice, liastrides[0], liastrides[1],
        mo_energy[nocc_range[0]:nocc_range[1]],
        mo_energy[nocc:],
        Pia, piastrides[0], piastrides[1],
        alpha, omega
    )
    return Pia

def rho_response_restricted(omega, mo_energy, Lia, max_memory=60):
    """Compute density-density response function in auxiliary basis at freq iw.
    See equation 58 in 10.1088/1367-2630/14/5/053020,
    and equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters:
    -----------

    omega : float
        imaginary frequency
    mo_energy : np.ndarray
        orbital energies
    Lia : np.ndarray
        occ-vir block of three-center density-fitting matrix.
    max_memory : float
        max memory in MB

    
    Returns:
    --------
    Pi : np.ndarray
        density-density response function in auxiliary basis at freq iw.
    """
    naux, nocc, nvir = Lia.shape

    # This is the original implementation
    # eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    # eia = eia/(omega**2+eia*eia)
    # Pia = einsum('Pia,ia->Pia',Lpq,eia)
    # # Response from both spin-up and spin-down density
    # Pi = 4. * einsum('Pia,Qia->PQ',Pia,Lpq)

    # return Pi

    nocc_slice_size = (max_memory*1024**2) // (2 * naux * nvir)
    nocc_slice_size = max(1, nocc_slice_size)

    Pi = lib.zeros((naux, naux))

    # compute the tensor contraction in batches
    # of occupied orbitals to save memory.
    lia_buf = np.empty((naux * nocc_slice_size * nvir))
    pia_buf = np.empty((naux * nocc_slice_size * nvir))

    for nocc_slice in lib.prange(0, nocc, nocc_slice_size):
        Lia_slice = copy_Lia_nocc_slice(Lia, nocc_range=nocc_slice,
                                         out=lia_buf)
        Pia_slice = dmul_Lia_response(Lia, mo_energy, omega,
                                     alpha=4.0, nocc_range=nocc_slice,
                                     out=pia_buf)

        blas.dgemm(alpha=1.0,
                   a=Pia_slice.reshape(naux, -1).T,
                   b=Lia_slice.reshape(naux, -1).T,
                   c=Pi.T,
                   beta=1.0,
                   trans_a=1,
                   trans_b=0,
                   overwrite_c=True)

    return Pi

def rho_response_unrestricted(omega, mo_energy, Lia_a, Lia_b, max_memory=60):
    """Compute density-density response function in auxiliary basis at freq iw.
    See equation 58 in 10.1088/1367-2630/14/5/053020,
    and equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters:
    -----------

    omega : float
        imaginary frequency
    mo_energy : np.ndarray
        orbital energies
    Lia_a : np.ndarray
        occ-vir block of three-center density-fitting matrix (alpha orbitals).
    Lia_b : np.ndarray
        occ-vir block of three-center density-fitting matrix (beta orbitals).
    max_memory : float
        max memory in MB

    
    Returns:
    --------
    Pi : np.ndarray
        density-density response function in auxiliary basis at freq iw.
    """
    naux, nocca, nvira = Lia_a.shape
    naux, noccb, nvirb = Lia_b.shape

    # This is the original implementation
    # naux, nocca, nvira = Lia_a.shape
    # naux, noccb, nvirb = Lia_b.shape
    # eia_a = mo_energy[0,:nocca,None] - mo_energy[0,None,nocca:]
    # eia_b = mo_energy[1,:noccb,None] - mo_energy[1,None,noccb:]
    # eia_a = eia_a/(omega**2+eia_a*eia_a)
    # eia_b = eia_b/(omega**2+eia_b*eia_b)
    # Pia_a = einsum('Pia,ia->Pia',Lia_a,eia_a)
    # Pia_b = einsum('Pia,ia->Pia',Lia_b,eia_b)
    # # Response from both spin-up and spin-down density
    # Pi = 2.* (einsum('Pia,Qia->PQ',Pia_a,Lia_a) + einsum('Pia,Qia->PQ',Pia_b,Lia_b))

    # return Pi

    nvir_max = max(nvira, nvirb)

    nocc_slice_size = (max_memory*1024**2) // (2 * naux * nvir_max)
    nocc_slice_size = max(1, nocc_slice_size)

    Pi = lib.zeros((naux, naux))

    # compute the tensor contraction in batches
    # of occupied orbitals to save memory.
    lia_buf = np.empty((naux * nocc_slice_size * nvir_max))
    pia_buf = np.empty((naux * nocc_slice_size * nvir_max))

    for Lia, nocc, mo_energy_s in [(Lia_a, nocca, mo_energy[0]),
                                   (Lia_b, noccb, mo_energy[1])]:
        for nocc_slice in lib.prange(0, nocc, nocc_slice_size):
            Lia_slice = copy_Lia_nocc_slice(Lia, nocc_range=nocc_slice,
                                            out=lia_buf)
            Pia_slice = dmul_Lia_response(Lia, mo_energy_s, omega,
                                        alpha=2.0, nocc_range=nocc_slice,
                                        out=pia_buf)
            blas.dgemm(alpha=1.0,
                    a=Pia_slice.reshape(naux, -1).T,
                    b=Lia_slice.reshape(naux, -1).T,
                    c=Pi.T,
                    beta=1.0,
                    trans_a=1,
                    trans_b=0,
                    overwrite_c=True)

    return Pi
