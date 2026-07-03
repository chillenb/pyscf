#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

from types import SimpleNamespace

import numpy
import scipy.linalg
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf.smearing import _fermi_smearing_occ


GCSCF_SIGMA = getattr(__config__, 'scf_gcscf_sigma', None)
GCSCF_MU0 = getattr(__config__, 'scf_gcscf_mu0', None)
GCSCF_CONV_TOL_GRAD = getattr(__config__, 'scf_gcscf_conv_tol_grad', None)
GCSCF_STEP = getattr(__config__, 'scf_gcscf_step', 1.0)
GCSCF_MIN_STEP = getattr(__config__, 'scf_gcscf_min_step', 1e-4)
GCSCF_LINE_MAX_CYCLE = getattr(__config__, 'scf_gcscf_line_max_cycle', 6)


def gcscf(mf, sigma=None, mu0=None):
    '''Grand-canonical SCF via auxiliary-Hamiltonian minimization.

    Args:
        mf : an RHF or RKS object
            Molecular restricted mean-field object to decorate.
        sigma : float
            Electronic temperature in Hartree.
        mu0 : float
            Fixed chemical potential in Hartree.

    Examples:

    >>> mf = gcscf(scf.RHF(mol), sigma=.1, mu0=-.2)
    >>> mf.kernel()
    '''
    if isinstance(mf, _GCSCF):
        if sigma is not None:
            mf.sigma = sigma
        if mu0 is not None:
            mf.mu0 = mu0
        return mf

    if mf.istype('_CIAH_SOSCF'):
        raise NotImplementedError('GC-SCF with second order SCF is not supported')
    if mf.istype('KSCF') or hasattr(mf, 'cell'):
        raise NotImplementedError('Use the PBC GC-SCF implementation for k-point objects')
    if mf.istype('ROHF') or mf.istype('UHF') or mf.istype('GHF'):
        raise NotImplementedError('GC-SCF currently supports molecular RHF/RKS objects')
    if not mf.istype('RHF'):
        raise NotImplementedError('GC-SCF currently supports molecular RHF/RKS objects')

    return lib.set_class(_GCSCF(mf, sigma, mu0), (_GCSCF, mf.__class__))


def gcscf_(mf, *args, **kwargs):
    mf1 = gcscf(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


grand_canonical = gcscf
grand_canonical_ = gcscf_


def remove_gcscf(mf):
    '''Remove the GC-SCF decorator.'''
    return mf.undo_gcscf()


class _GCSCF:
    '''Grand-canonical SCF via auxiliary-Hamiltonian minimization.'''

    __name_mixin__ = 'GC-SCF'

    _keys = {
        'sigma', 'mu0', 'auxh_step', 'auxh_min_step',
        'auxh_line_max_cycle', 'entropy', 'e_free', 'e_zero', 'e_grand',
        'mu', 'nelectron', 'haux', 'n_haux_eval', 'auxh_residual_norm',
    }

    def __init__(self, mf, sigma, mu0):
        self.__dict__.update(mf.__dict__)
        self.sigma = GCSCF_SIGMA if sigma is None else sigma
        self.mu0 = GCSCF_MU0 if mu0 is None else mu0
        if GCSCF_CONV_TOL_GRAD is not None:
            self.conv_tol_grad = GCSCF_CONV_TOL_GRAD
        self.auxh_step = GCSCF_STEP
        self.auxh_min_step = GCSCF_MIN_STEP
        self.auxh_line_max_cycle = GCSCF_LINE_MAX_CYCLE
        self.entropy = None
        self.e_free = None
        self.e_zero = None
        self.e_grand = None
        self.mu = None
        self.nelectron = None
        self.haux = None
        self.n_haux_eval = 0
        self.auxh_residual_norm = None

    def undo_gcscf(self):
        '''Remove the GC-SCF mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _GCSCF))
        del obj.sigma
        del obj.mu0
        del obj.auxh_step
        del obj.auxh_min_step
        del obj.auxh_line_max_cycle
        del obj.entropy
        del obj.e_free
        del obj.e_zero
        del obj.e_grand
        del obj.mu
        del obj.nelectron
        del obj.haux
        del obj.n_haux_eval
        del obj.auxh_residual_norm
        return obj

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        super().dump_flags(verbose)
        log.info('******** GC-SCF flags ********')
        log.info('sigma = %s', self.sigma)
        log.info('mu0 = %s', self.mu0)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('auxh_step = %g', self.auxh_step)
        log.info('auxh_min_step = %g', self.auxh_min_step)
        log.info('auxh_line_max_cycle = %d', self.auxh_line_max_cycle)
        return self

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.sigma is None or self.mu0 is None:
            return super().get_occ(mo_energy, mo_coeff)
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GC-SCF')
        mo_occ = 2.0 * _fermi_smearing_occ(self.mu0, mo_energy, self.sigma)
        self.entropy = _fermi_entropy(mo_occ)
        logger.info(self, '    sigma = %g  mu0 = %.12g  entropy = %.12g',
                    self.sigma, self.mu0, self.entropy)
        return mo_occ

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = super().energy_tot(dm, h1e, vhf)
        if self.sigma is not None and self.mu0 is not None and self.mo_occ is not None:
            self.entropy = _fermi_entropy(self.mo_occ)
            self.nelectron = float(numpy.sum(self.mo_occ))
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            self.e_grand = self.e_free - self.mu0 * self.nelectron
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  '
                        'Grand potential = %.15g',
                        e_tot, self.e_free, self.e_grand)
        return e_tot

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if self.sigma is None:
            raise ValueError('sigma must be specified for GC-SCF')
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GC-SCF')
        if self.mu0 is None:
            raise ValueError('mu0 must be specified for GC-SCF')

        conv_tol = kwargs.pop('conv_tol', self.conv_tol)
        conv_tol_grad = kwargs.pop('conv_tol_grad', self.conv_tol_grad)
        if conv_tol_grad is None:
            conv_tol_grad = conv_tol ** .5
            logger.info(self, 'Set gradient conv threshold to %g',
                        conv_tol_grad)
        max_cycle = kwargs.pop('max_cycle', self.max_cycle)
        callback = kwargs.pop('callback', self.callback)
        if kwargs:
            logger.warn(self, 'GC-SCF kernel ignored unsupported kwargs %s',
                        sorted(kwargs))

        self.build(self.mol)
        self.dump_flags()
        self.pre_kernel(locals())

        hcore_ao = self.get_hcore(self.mol)
        S = self.get_ovlp(self.mol)
        x = self.check_linear_dependency(S, verbose=self.verbose)
        hcore = _hermitian_part(x.conj().T @ hcore_ao @ x)
        n_haux_eval = 0

        def haux_eval_counting(mo_energy, mo_coeff_orth):
            nonlocal n_haux_eval
            n_haux_eval += 1
            return _haux_eval(self, mo_energy, mo_coeff_orth, x,
                              hcore_ao, hcore)

        if isinstance(dm0, str):
            dm0 = self.from_chk(dm0)
        elif dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            dm0 = self.make_rdm1(self.mo_coeff, self.mo_occ)
        elif dm0 is None:
            dm0 = self.get_init_guess(self.mol, key=self.init_guess)

        vhf_ao = self.get_veff(self.mol, dm0)
        haux = _hermitian_part(hcore + x.conj().T @ vhf_ao @ x)
        mo_energy, mo_coeff_orth = scipy.linalg.eigh(haux)
        state = haux_eval_counting(mo_energy, mo_coeff_orth)
        residual_norm = float(numpy.linalg.norm(state.haux_gradient))
        alpha_t = float(self.auxh_step)
        previous_gradient = None
        previous_direction = None
        previous_gknorm = None
        niter_done = 0
        converged = residual_norm < conv_tol_grad

        log = logger.new_logger(self)
        log.info('GC-SCF auxiliary-Hamiltonian minimization, '
                 'sigma = %.12g Ha, mu0 = %.12g Ha',
                 self.sigma, self.mu0)

        for cycle in range(1, max_cycle + 1):
            if converged:
                break

            gradient = state.haux_gradient
            steepest_direction = _hermitian_part(
                state.hsub - numpy.diag(numpy.asarray(state.mo_energy, dtype=float))
            )
            gknorm = -float(numpy.vdot(gradient, steepest_direction).real)
            beta = 0.0
            cg_reset = False
            if (previous_gradient is not None and previous_direction is not None
                    and previous_gknorm is not None and previous_gknorm > 0.0):
                dot_gprev_kg = -float(numpy.vdot(previous_gradient,
                                                 steepest_direction).real)
                beta = (gknorm - dot_gprev_kg) / previous_gknorm
                if beta < 0.0 or not numpy.isfinite(beta):
                    beta = 0.0
                    cg_reset = True

            if previous_direction is None:
                direction = steepest_direction
            else:
                direction = _hermitian_part(steepest_direction
                                            + beta * previous_direction)
            gdotd = float(numpy.vdot(gradient, direction).real)
            if gdotd >= 0.0 and beta != 0.0:
                beta = 0.0
                cg_reset = True
                direction = steepest_direction
                gdotd = float(numpy.vdot(gradient, direction).real)
            if gdotd >= 0.0:
                beta = 0.0
                cg_reset = True
                direction = _hermitian_part(-gradient)

            line_eval_start = n_haux_eval
            line_result = _line_minimize_auxiliary_hamiltonian(
                haux_eval_counting, haux, direction, state,
                alpha_t=alpha_t, alpha_t_min=self.auxh_min_step,
                max_cycle=self.auxh_line_max_cycle)
            if not line_result.success and beta != 0.0:
                beta = 0.0
                cg_reset = True
                direction = steepest_direction
                line_result = _line_minimize_auxiliary_hamiltonian(
                    haux_eval_counting, haux, direction, state,
                    alpha_t=alpha_t, alpha_t_min=self.auxh_min_step,
                    max_cycle=self.auxh_line_max_cycle)
            line_haux_eval = n_haux_eval - line_eval_start
            if not line_result.success:
                if residual_norm < conv_tol_grad:
                    converged = True
                    log.info('GC-SCF line minimization stopped at cycle %d '
                             'with converged residual: %s',
                             cycle, line_result.message)
                    break
                log.info('GC-SCF line minimization failed at cycle %d after '
                         '%d iterations: %s',
                         cycle, line_result.niter, line_result.message)
                break

            old_objective = state.objective
            haux = line_result.haux
            state = line_result.state
            residual_norm = line_result.residual_norm
            alpha_t = line_result.next_alpha_t
            if alpha_t < self.auxh_min_step:
                alpha_t = float(self.auxh_step)
            previous_gradient = _hermitian_part(
                line_result.rotation.T.conj() @ gradient @ line_result.rotation
            )
            previous_direction = _hermitian_part(
                line_result.rotation.T.conj() @ direction @ line_result.rotation
            )
            previous_gknorm = gknorm
            d_objective = state.objective - old_objective
            niter_done = cycle

            log.info('cycle=%3d E=%.15g A=%.15g Omega=%.15g N=%.9g '
                     'S=%.9g |R|=%.3e alpha=%.3g alphaT=%.3g beta=%.3g '
                     'reset=%d line_iter=%d evals=%d line_haux_eval=%d',
                     cycle, state.e_tot, state.e_free, state.e_grand,
                     state.nelectron, state.entropy, residual_norm,
                     line_result.alpha, alpha_t, beta, int(cg_reset),
                     line_result.niter, n_haux_eval, line_haux_eval)

            if callable(callback):
                callback(locals())

            if abs(d_objective) < conv_tol and residual_norm < conv_tol_grad:
                converged = True
                break

        self.converged = converged
        self.e_tot = float(state.e_tot)
        self.e_free = float(state.e_free)
        self.e_zero = float(state.e_tot - self.sigma * state.entropy * .5)
        self.e_grand = float(state.e_grand)
        self.mu = float(self.mu0)
        self.entropy = float(state.entropy)
        self.nelectron = float(state.nelectron)
        self.mo_energy = numpy.array(state.mo_energy, copy=True)
        self.mo_coeff = numpy.array(state.mo_coeff, copy=True)
        self.mo_occ = numpy.array(state.mo_occ, copy=True)
        self.haux = numpy.array(_hermitian_part(S @ x @ haux @ x.conj().T @ S),
                                copy=True)
        self.n_haux_eval = n_haux_eval
        self.auxh_residual_norm = float(residual_norm)
        self.cycles = niter_done

        if self.chkfile:
            self.dump_chk({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                           'mo_coeff': self.mo_coeff, 'mo_occ': self.mo_occ})
        self.post_kernel(locals())
        logger.timer(self, 'GC-SCF', *cput0)
        self._finalize()
        return self.e_tot

    scf = kernel

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged GC-SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'GC-SCF not converged.')
            logger.note(self, 'GC-SCF energy = %.15g', self.e_tot)
        logger.note(self, 'GC-SCF free energy = %.15g', self.e_free)
        logger.note(self, 'GC-SCF grand potential = %.15g', self.e_grand)
        logger.note(self, 'GC-SCF electron number = %.15g', self.nelectron)
        logger.note(self, 'GC-SCF entropy = %.15g', self.entropy)
        logger.note(self, 'GC-SCF AuxH evaluations = %d', self.n_haux_eval)
        logger.note(self, 'GC-SCF AuxH residual norm = %.3e',
                    self.auxh_residual_norm)
        return self

    def to_gpu(self):
        obj = gcscf(self.undo_gcscf().to_gpu(), self.sigma, self.mu0)
        obj.conv_tol_grad = self.conv_tol_grad
        obj.auxh_step = self.auxh_step
        obj.auxh_min_step = self.auxh_min_step
        obj.auxh_line_max_cycle = self.auxh_line_max_cycle
        return obj


def _haux_eval(mf, mo_energy, mo_coeff_orth, x, hcore_ao, hcore):
    mo_energy = numpy.asarray(mo_energy, dtype=float)
    mo_coeff_orth = numpy.asarray(mo_coeff_orth)
    mo_coeff = x @ mo_coeff_orth
    mo_occ = 2.0 * _fermi_smearing_occ(mf.mu0, mo_energy, mf.sigma)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    fock = hcore + x.conj().T @ vhf @ x
    entropy = _fermi_entropy(mo_occ)
    e_tot = float(mf.energy_elec(dm=dm, h1e=hcore_ao, vhf=vhf)[0]
                  + mf.energy_nuc())
    e_free = e_tot - mf.sigma * entropy
    nelectron = float(numpy.sum(mo_occ))
    e_grand = e_free - mf.mu0 * nelectron
    hsub = mo_coeff_orth.conj().T @ fock @ mo_coeff_orth
    grad_filling = _hermitian_part(numpy.asarray(hsub)) - numpy.diag(mo_energy)
    haux_gradient = _smearing_matrix_gradient(
        mo_energy, mo_occ, mf.sigma, grad_filling)

    return SimpleNamespace(
        mo_energy=mo_energy, mo_coeff=mo_coeff, mo_coeff_orth=mo_coeff_orth,
        mo_occ=mo_occ, dm=dm, vhf=vhf, fock=fock, hsub=hsub,
        entropy=entropy, nelectron=nelectron, e_tot=e_tot,
        e_free=e_free, e_grand=e_grand, objective=e_grand,
        haux_gradient=haux_gradient)


def _line_minimize_auxiliary_hamiltonian(haux_eval, haux, direction, state, alpha_t,
                                         alpha_t_min, max_cycle):
    if alpha_t <= 0:
        raise ValueError('auxh_step must be positive')
    if alpha_t_min <= 0:
        raise ValueError('auxh_min_step must be positive')
    if max_cycle <= 0:
        raise ValueError('auxh_line_max_cycle must be positive')

    alpha_reduce_factor = 0.1
    alpha_increase_factor = 3.0
    objective_slop = 1e-12
    objective0 = float(state.objective)
    gdotd = float(numpy.vdot(state.haux_gradient, direction).real)
    identity = numpy.eye(state.mo_energy.size)
    haux_eigenbasis = numpy.diag(state.mo_energy)
    mo_coeff0 = state.mo_coeff_orth
    line_candidates = [
        (0.0, state, float(numpy.linalg.norm(state.haux_gradient)), haux, identity)
    ]

    def line_result(candidate, niter, next_alpha_t, success, message):
        alpha, candidate_state, residual, candidate_haux, rotation = candidate
        return SimpleNamespace(
            haux=candidate_haux, state=candidate_state, alpha=alpha,
            residual_norm=residual, niter=niter, next_alpha_t=next_alpha_t,
            rotation=rotation, success=success, message=message)

    if gdotd >= 0.0:
        return line_result(line_candidates[0], 0, alpha_t, False,
                           'search direction is not downhill')

    def make_trialstate(alpha):
        alpha = float(max(alpha, 0.0))
        trial_haux_eigenbasis = _hermitian_part(haux_eigenbasis
                                                + alpha * direction)
        mo_energy, rotation = scipy.linalg.eigh(trial_haux_eigenbasis)
        mo_coeff_orth = mo_coeff0 @ rotation
        trial_state = haux_eval(mo_energy, mo_coeff_orth)
        weighted_coeff = mo_coeff_orth * numpy.asarray(mo_energy, dtype=float)
        trial_haux = _hermitian_part(weighted_coeff @ mo_coeff_orth.T.conj())
        result = (
            alpha, trial_state,
            float(numpy.linalg.norm(trial_state.haux_gradient)),
            trial_haux, rotation
        )
        line_candidates.append(result)
        return result

    niter = 0
    alpha_trial = float(alpha_t)
    alpha = alpha_trial
    for unused in range(max_cycle):
        if alpha_trial < alpha_t_min:
            return line_result(line_candidates[0], niter, alpha_trial, False,
                               'test step fell below auxh_min_step')

        niter += 1
        test_candidate = make_trialstate(alpha_trial)
        test_state = test_candidate[1]
        test_residual = test_candidate[2]
        test_objective = float(test_state.objective)
        if not numpy.isfinite(test_objective):
            alpha_trial *= alpha_reduce_factor
            continue

        denominator = alpha_trial * gdotd + objective0 - test_objective
        denominator_tol = numpy.finfo(float).eps * max(
            1.0, abs(alpha_trial * gdotd), abs(objective0),
            abs(test_objective))
        if abs(denominator) <= denominator_tol:
            if test_objective <= objective0 or (
                    test_objective <= objective0 + objective_slop
                    and test_residual < line_candidates[0][2]):
                return line_result(test_candidate, niter,
                                   alpha_trial * alpha_increase_factor, True,
                                   'accepted test step with flat quadratic curvature')
            alpha_trial *= alpha_reduce_factor
            continue

        alpha = 0.5 * alpha_trial * alpha_trial * gdotd / denominator
        if not numpy.isfinite(alpha):
            alpha_trial *= alpha_reduce_factor
            continue
        if alpha < 0.0:
            return line_result(test_candidate, niter,
                               alpha_trial * alpha_increase_factor, True,
                               'accepted downhill test step with wrong quadratic curvature')
        if alpha / alpha_trial > alpha_increase_factor:
            alpha_trial *= alpha_increase_factor
            continue
        if alpha > 0.0 and alpha_trial / alpha < alpha_reduce_factor:
            alpha_trial *= alpha_reduce_factor
            continue
        break
    else:
        return _best_line_min_candidate(line_candidates, niter,
                                        'test step adjustment failed')

    for unused in range(max_cycle):
        niter += 1
        trial_state = make_trialstate(alpha)[1]
        trial_objective = float(trial_state.objective)
        if not numpy.isfinite(trial_objective):
            alpha *= alpha_reduce_factor
            continue
        line_candidate = _best_line_min_candidate(
            line_candidates, niter, 'accepted best line candidate',
            objective_slop=objective_slop)
        if line_candidate.success:
            return line_candidate
        if trial_objective > objective0:
            alpha *= alpha_reduce_factor
            continue

    return _best_line_min_candidate(line_candidates, niter,
                                    'predicted step failed to reduce objective',
                                    objective_slop=objective_slop)


def _best_line_min_candidate(line_candidates, niter, failure_message,
                             objective_slop=0.0):
    finite_candidates = [candidate for candidate in line_candidates
                         if numpy.isfinite(candidate[1].objective)]
    objective_threshold = min(candidate[1].objective
                              for candidate in finite_candidates) + objective_slop
    best_alpha, best_state, best_residual, best_haux, best_rotation = min(
        (candidate for candidate in finite_candidates
         if candidate[1].objective <= objective_threshold),
        key=lambda item: (item[2], item[1].objective))
    return SimpleNamespace(
        haux=best_haux, state=best_state, alpha=best_alpha,
        residual_norm=best_residual, niter=niter, next_alpha_t=best_alpha,
        rotation=best_rotation, success=best_alpha != 0.0,
        message=('%s; accepted best lower-energy trial' % failure_message
                 if best_alpha != 0.0 else failure_message))


def _smearing_matrix_gradient(eta, mo_occ, sigma, grad_filling):
    eta = numpy.asarray(eta, dtype=float)
    mo_occ = numpy.asarray(mo_occ, dtype=float)
    occ_prime = _fermi_occupation_derivative(mo_occ, sigma)
    energy_diff = eta[:,None] - eta[None,:]
    occ_diff = mo_occ[:,None] - mo_occ[None,:]

    with numpy.errstate(divide='ignore', invalid='ignore'):
        factors = numpy.divide(
            occ_diff, energy_diff,
            out=numpy.zeros_like(grad_filling, dtype=float),
            where=numpy.abs(energy_diff) > 1e-12)

    near_degenerate = numpy.isclose(energy_diff, 0.0, atol=1e-12, rtol=1e-12)
    derivative_average = .5 * (occ_prime[:,None] + occ_prime[None,:])
    factors[near_degenerate] = derivative_average[near_degenerate]

    return _hermitian_part(_hermitian_part(grad_filling) * factors)


def _fermi_occupation_derivative(mo_occ, sigma):
    f = numpy.asarray(mo_occ, dtype=float) * .5
    return -2.0 * f * (1.0 - f) / sigma


def _fermi_entropy(mo_occ):
    f = numpy.asarray(mo_occ, dtype=float) * .5
    f = f[(f > 0.0) & (f < 1.0)]
    if f.size == 0:
        return 0.0
    entropy_per_spin = -(f * numpy.log(f) + (1.0 - f) * numpy.log(1.0 - f)).sum()
    return float(2.0 * entropy_per_spin)


def _hermitian_part(matrix):
    return 0.5 * lib.hermi_sum(matrix)
