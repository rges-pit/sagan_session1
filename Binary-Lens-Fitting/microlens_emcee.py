"""
Microlensing MCMC Analysis Module

This module provides a class-based interface for performing MCMC analysis
on microlensing models using the emcee package and MulensModel.
"""

import json
import os
import warnings
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocess import Pool
import MulensModel as mm


# ============================================================================
# Log parameter handling
# ============================================================================

LOG_PARAM_MAP = {'log_t_E': 't_E', 'log_s': 's', 'log_q': 'q', 'log_rho': 'rho'}


# ============================================================================
# Per-worker state for multiprocessing (pool-initializer pattern)
# ============================================================================

_worker_state = {}


def _init_worker(data_file, data_kwargs, model_params, mag_methods,
                 log_prob_func, params_to_fit, bounds):
    """
    Pool initializer: creates a process-local MulensModel Event so that the
    un-picklable VBMicrolensing C++ object is never sent between processes.

    Parameters
    ----------
    data_file : str
        Path to the data file for ``mm.MulensData``.
    data_kwargs : dict
        Extra keyword arguments for ``mm.MulensData`` (e.g. phot_fmt, chi2_fmt).
    model_params : dict
        Parameter dictionary for ``mm.Model``.
    mag_methods : list
        Arguments for ``model.set_magnification_methods()``.
    log_prob_func : callable
        Log-probability function with signature
        ``(theta, params_to_fit, event, bounds) -> float``.
    params_to_fit : list[str]
        Parameter names being fitted.
    bounds : dict | None
        Bounds dictionary passed through to ``log_prob_func``.
    """
    data = mm.MulensData(file_name=data_file, **data_kwargs)
    model = mm.Model(model_params, coords=None, ephemerides_file=None)
    model.set_magnification_methods(mag_methods)
    _worker_state['event'] = mm.Event(datasets=[data], model=model)
    _worker_state['log_prob'] = log_prob_func
    _worker_state['params_to_fit'] = params_to_fit
    _worker_state['bounds'] = bounds


def _log_prob_worker(theta):
    """Log-probability evaluated inside a pool worker using process-local state."""
    return _worker_state['log_prob'](
        theta,
        _worker_state['params_to_fit'],
        _worker_state['event'],
        _worker_state['bounds'],
    )


def chi2_mm(x0, params_to_fit, event):
    """
    Calculates chi-squared by updating the MulensModel Event with the given parameters.
    
    Parameters
    ----------
    x0 : array-like
        Parameter values (in the order of params_to_fit)
    params_to_fit : list[str]
        Parameter names. Use 'log_' prefix for t_E, s, q, rho to fit in log space.
    event : mm.Event
        MulensModel Event object
        
    Returns
    -------
    float
        Chi-squared value
    """
    for i, param_name in enumerate(params_to_fit):
        if param_name in LOG_PARAM_MAP:
            actual_param = LOG_PARAM_MAP[param_name]
            setattr(event.model.parameters, actual_param, 10**x0[i])
        else:
            setattr(event.model.parameters, param_name, x0[i])
    
    event.fit_fluxes()
    return event.get_chi2()


# ============================================================================
# Main MCMC Analysis Class
# ============================================================================

class micro_mc:
    """
    A class for performing MCMC analysis on microlensing models using MulensModel.
    
    This class encapsulates all functionality needed to fit microlensing
    light curves using MCMC methods, including convergence diagnostics
    and visualization tools.
    
    Parameters
    ----------
    event : mm.Event
        MulensModel Event object with model and datasets already set.
    params_to_fit : list[str]
        Parameter names to fit. Use 'log_' prefix (log_t_E, log_s, log_q, log_rho)
        to fit in log space. Parameters NOT in this list are held fixed.
    log_probability : callable
        User-provided log probability function with signature:
        log_probability(x, params_to_fit, event, bounds) -> float
    bounds : dict | None, optional
        Optional map of parameter name -> (min, max). If None, use defaults.
    
    Attributes
    ----------
    results : dict
        Dictionary containing MCMC analysis results (set after running perform_mcmc_analysis)
    """
    
    def __init__(self, event, params_to_fit, log_probability, bounds=None):
        """
        Initialize the micro_mc analysis class.
        
        Parameters
        ----------
        event : mm.Event
            MulensModel Event object with model and datasets already set.
        params_to_fit : list[str]
            Parameter names to fit. Use 'log_' prefix for t_E, s, q, rho.
        log_probability : callable
            User-provided log probability function with signature:
            log_probability(x, params_to_fit, event, bounds) -> float
        bounds : dict | None, optional
            Optional map of parameter name -> (min, max). If None, use defaults.
        """
        self.event = event
        self.params_to_fit = params_to_fit
        self.log_probability = log_probability
        self.bounds = bounds
        
        # Results will be stored here after running analysis
        self.results = None
    
    def _get_initial_values(self):
        """Extract initial parameter values from the event model."""
        x0 = []
        for param_name in self.params_to_fit:
            if param_name in LOG_PARAM_MAP:
                actual_param = LOG_PARAM_MAP[param_name]
                val = getattr(self.event.model.parameters, actual_param)
                x0.append(np.log10(val))
            else:
                x0.append(getattr(self.event.model.parameters, param_name))
        return np.array(x0)
    
    def _convert_to_linear(self, x0):
        """Convert log parameters back to linear space."""
        result = {}
        for i, param_name in enumerate(self.params_to_fit):
            if param_name in LOG_PARAM_MAP:
                actual_param = LOG_PARAM_MAP[param_name]
                result[actual_param] = 10**x0[i]
            else:
                result[param_name] = x0[i]
        return result
    
    def run_mcmc(self, steps=500, walkers=100, step_scale=1e-4, 
                 param_scales=None, verbose=False, pool=None):
        """
        Run MCMC sampling on the parameters specified in params_to_fit.
        
        Parameters
        ----------
        steps : int, optional
            Number of MCMC steps (default: 500)
        walkers : int, optional
            Number of walkers (default: 100)
        step_scale : float, optional
            Global step size scaling factor (default: 1e-4)
        param_scales : dict[str, float] | np.ndarray | None, optional
            Dict of per-parameter step sizes or array matching params_to_fit order.
        verbose : bool, optional
            Print step size information (default: False)
        pool : multiprocessing.Pool | None, optional
            Multiprocessing pool for parallel evaluation. If None, runs in serial.
        
        Returns
        -------
        sampler : emcee.EnsembleSampler
            The MCMC sampler
        pos : np.ndarray
            Final walker positions
        prob : np.ndarray
            Final log probabilities
        state : object
            Random state
        """
        t = self.event.datasets[0].time
        ndim = len(self.params_to_fit)
        
        if ndim == 0:
            raise ValueError("No parameters to sample. Provide at least one parameter in params_to_fit.")

        # Default bounds (in fitting space)
        default_bounds = {
            't_0': (t.min(), t.max()),
            't_E': (0.1, t.max() - t.min()),
            'u_0': (-10.0, 10.0),
            'rho': (1e-5, 1.0),
            's': (1e-2, 20.0),
            'q': (1e-7, 1.0),
            'alpha': (0.0, 360.0),
            'log_t_E': (-1, 3),
            'log_s': (-2, 2),
            'log_q': (-7, 0),
            'log_rho': (-5, 0),
        }
        bounds_map = default_bounds.copy()
        if self.bounds is not None:
            bounds_map.update(self.bounds)

        # Get initial values from event model
        x0 = self._get_initial_values()

        # Step sizes
        if isinstance(param_scales, dict):
            scales = np.array([param_scales.get(name, step_scale) for name in self.params_to_fit])
            if verbose:
                print("Using dict step sizes:")
                for name, sc in zip(self.params_to_fit, scales):
                    print(f"  {name}: {sc:.2e}")
        elif isinstance(param_scales, np.ndarray):
            if len(param_scales) != ndim:
                raise ValueError(f"param_scales array must match params_to_fit length ({ndim}).")
            scales = param_scales
            if verbose:
                print("Using array step sizes:")
                for name, sc in zip(self.params_to_fit, scales):
                    print(f"  {name}: {sc:.2e}")
        elif param_scales is None:
            scales = np.array([step_scale] * ndim)
            if verbose:
                print(f"Using global step size: {step_scale:.2e}")
        else:
            raise ValueError("param_scales must be a dict, numpy array, or None")

        # Create initial walker positions
        nwalkers = walkers
        p0 = x0 + scales * np.random.randn(nwalkers, ndim)

        # Clip to bounds
        for j, name in enumerate(self.params_to_fit):
            lower, upper = bounds_map.get(name, (-np.inf, np.inf))
            p0[:, j] = np.clip(p0[:, j], lower, upper)

        # When a pool is provided, use the module-level _log_prob_worker which
        # accesses a process-local Event (set up by _init_worker) so that the
        # un-picklable VBMicrolensing object never crosses process boundaries.
        if pool is not None:
            log_prob_fn = _log_prob_worker
        else:
            def log_prob_fn(theta):
                return self.log_probability(theta, self.params_to_fit, self.event, self.bounds)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, pool=pool)

        # Run MCMC
        pos, prob, state = sampler.run_mcmc(p0, steps, progress=True)

        return sampler, pos, prob, state

    @staticmethod
    def gelman_rubin_statistic(chains):
        """Calculate Gelman-Rubin R-hat statistic for convergence assessment"""
        n_walkers, n_steps, n_params = chains.shape
        
        if n_steps < 4:
            return np.full(n_params, np.nan)
        
        # Split each chain in half
        half_steps = n_steps // 2
        chains_split = chains[:, -2*half_steps:, :].reshape(2*n_walkers, half_steps, n_params)
        
        r_hat = np.zeros(n_params)
        
        for i in range(n_params):
            chain_means = np.mean(chains_split[:, :, i], axis=1)
            chain_vars = np.var(chains_split[:, :, i], axis=1, ddof=1)
            
            overall_mean = np.mean(chain_means)
            between_var = half_steps * np.var(chain_means, ddof=1)
            within_var = np.mean(chain_vars)
            
            var_estimate = ((half_steps - 1) * within_var + between_var) / half_steps
            r_hat[i] = np.sqrt(var_estimate / within_var) if within_var > 0 else 1.0
        
        return r_hat

    @staticmethod
    def effective_sample_size(chains):
        """Estimate effective sample size"""
        n_walkers, n_steps, n_params = chains.shape
        eff_sizes = np.zeros(n_params)
        
        for i in range(n_params):
            # Flatten chains for this parameter
            flat_chain = chains[:, :, i].flatten()
            
            # Simple autocorrelation-based estimate
            autocorr = np.correlate(flat_chain - np.mean(flat_chain), 
                                   flat_chain - np.mean(flat_chain), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find where autocorr drops below 1/e
            try:
                cutoff = np.where(autocorr < 1/np.e)[0][0]
                eff_sizes[i] = len(flat_chain) / (2 * cutoff + 1)
            except IndexError:
                eff_sizes[i] = len(flat_chain) / 10  # Conservative estimate
        
        return eff_sizes

    def plot_corner_mcmc(self, samples=None, param_names=None, title="MCMC Posterior Distributions", show_titles=True, **kwargs):
        """
        Plot corner plot of MCMC samples
        
        Parameters
        ----------
        samples : np.ndarray, optional
            MCMC samples. If None, uses self.results['samples']
        param_names : list, optional
            Parameter names. If None, uses self.results['params_to_fit']
        title : str, optional
            Plot title
        show_titles : bool, optional
            Whether to show parameter titles with quantiles (default: True)
        **kwargs : dict
            Additional arguments passed to corner.corner
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The corner plot figure
        """
        if samples is None:
            if self.results is None:
                raise ValueError("No results available. Run perform_mcmc_analysis first.")
            samples = self.results['samples']
            param_names = self.params_to_fit
        
        fig = corner.corner(samples, labels=param_names, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=show_titles, title_kwargs={"fontsize": 12},
                           label_kwargs={"fontsize": 14}, **kwargs)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_mcmc_fit(self, mle_delta_chi2=None):
        """
        Plot the best fit model from MCMC using MulensModel Event.
        
        Parameters
        ----------
        mle_delta_chi2 : float, optional
            Delta chi-squared value. If None, uses self.results['mle_delta_chi2']
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The fit plot figure
        """
        if mle_delta_chi2 is None:
            if self.results is None:
                raise ValueError("No results available. Run perform_mcmc_analysis first.")
            mle_delta_chi2 = self.results['mle_delta_chi2']
        
        # Get data from event
        data = self.event.datasets[0]
        t = data.time
        f = data.flux
        sig = data.err_flux
        
        # Get model flux from event (MLE params should already be set)
        self.event.fit_fluxes()
        fs, fb = self.event.get_ref_fluxes()
        model_mag = self.event.model.get_magnification(t)
        model_flux = fs * model_mag + fb
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t, f, yerr=sig, fmt='o', color='black', label='Data', markersize=2, alpha=0.7)
        ax.plot(t, model_flux, color='red', linewidth=2, 
                label=f'MCMC Best Fit - $\\Delta \\chi^2 = {mle_delta_chi2:.2f}$')
        ax.set_xlabel('Time (HJD)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title('Microlensing Model - MCMC Fit', fontsize=14)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def plot_mcmc_traces(sampler, param_names, burn_in):
        """
        Plot MCMC trace plots for convergence diagnosis.

        In addition to one row per fitted parameter, an extra bottom row shows
        chi-squared (= -2 * log-probability) as a function of step for every
        walker.  Requires ``sampler.lnprobability`` (shape
        ``(n_walkers, n_steps)``) to be present; if it is missing the chi^2
        row is omitted with a warning.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler or object with .chain (and optionally
            .lnprobability)
            The MCMC sampler (or a lightweight wrapper around saved arrays).
        param_names : list
            Parameter names
        burn_in : int
            Burn-in period

        Returns
        -------
        fig : matplotlib.figure.Figure
            The trace plot figure
        """
        has_lnprob = hasattr(sampler, 'lnprobability') and sampler.lnprobability is not None
        if not has_lnprob:
            warnings.warn(
                "sampler has no 'lnprobability' attribute; skipping chi^2 trace row. "
                "Re-run MCMC with the updated save_results to generate a "
                "'<prefix>_lnprob.npy' file and pass it via "
                "SimpleNamespace(chain=..., lnprobability=...).",
                stacklevel=2,
            )

        n_rows = len(param_names) + (1 if has_lnprob else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2*n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        n_walkers = sampler.chain.shape[0]
        random_walkers = np.random.choice(
            n_walkers, size=min(5, n_walkers), replace=False
        )
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i in range(len(param_names)):
            for j in range(n_walkers):
                axes[i].plot(sampler.chain[j, :, i], color='k', alpha=0.4, linewidth=0.5)

            for idx, walker in enumerate(random_walkers):
                color = colors[idx % len(colors)]
                axes[i].plot(sampler.chain[walker, :, i], color=color, alpha=0.7, linewidth=1.2,
                            label=f'Walker {walker}' if i == 0 else '')

            axes[i].axvline(burn_in, color='red', linestyle='--', linewidth=2, label='Burn-in' if i == 0 else '')
            axes[i].set_ylabel(f'{param_names[i]}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if has_lnprob:
            ax_chi2 = axes[len(param_names)]
            lnp = np.asarray(sampler.lnprobability)
            # log_probability = log_prior + (-0.5 * chi^2), with log_prior = 0
            # inside the bounds and -inf outside.  Mask out-of-bounds proposals.
            chi2 = np.where(np.isfinite(lnp), -2.0 * lnp, np.nan)

            for j in range(n_walkers):
                ax_chi2.plot(chi2[j, :], color='k', alpha=0.4, linewidth=0.5)

            for idx, walker in enumerate(random_walkers):
                color = colors[idx % len(colors)]
                ax_chi2.plot(chi2[walker, :], color=color, alpha=0.7, linewidth=1.2)

            ax_chi2.axvline(burn_in, color='red', linestyle='--', linewidth=2)
            ax_chi2.set_ylabel(r'$\chi^2$', fontsize=12)
            ax_chi2.set_yscale('log')
            ax_chi2.grid(True, alpha=0.3, which='both')

        axes[-1].set_xlabel('Step Number', fontsize=12)
        plt.suptitle('MCMC Trace Plots - Individual Walker Chains', fontsize=16)
        plt.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def plot_convergence_diagnostics(sampler, param_names, burn_in):
        """
        Plot running statistics to check convergence
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The MCMC sampler
        param_names : list
            Parameter names
        burn_in : int
            Burn-in period
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The convergence diagnostics figure
        """
        fig, axes = plt.subplots(len(param_names), 2, figsize=(16, 2*len(param_names)))
        if len(param_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(len(param_names)):
            # Calculate running mean and std
            chain_flat = sampler.chain[:, :, i]
            running_mean = np.zeros((chain_flat.shape[0], chain_flat.shape[1]))
            running_std = np.zeros((chain_flat.shape[0], chain_flat.shape[1]))
            
            for walker in range(chain_flat.shape[0]):
                for step in range(1, chain_flat.shape[1]):
                    running_mean[walker, step] = np.mean(chain_flat[walker, :step+1])
                    running_std[walker, step] = np.std(chain_flat[walker, :step+1])
            
            # Plot running mean
            step_numbers = np.arange(chain_flat.shape[1])
            for walker in range(min(10, chain_flat.shape[0])):
                axes[i, 0].plot(step_numbers, running_mean[walker, :], alpha=0.6, linewidth=1)
            
            axes[i, 0].axvline(burn_in, color='red', linestyle='--', linewidth=2, label='Burn-in')
            axes[i, 0].set_ylabel(f'Running Mean - {param_names[i]}', fontsize=12)
            axes[i, 0].set_title(f'{param_names[i]} - Running Mean Convergence', fontsize=12)
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].legend()
            
            # Plot running standard deviation
            for walker in range(min(10, chain_flat.shape[0])):
                axes[i, 1].plot(step_numbers, running_std[walker, :], alpha=0.6, linewidth=1)
            
            axes[i, 1].axvline(burn_in, color='red', linestyle='--', linewidth=2, label='Burn-in')
            axes[i, 1].set_ylabel(f'Running Std - {param_names[i]}', fontsize=12)
            axes[i, 1].set_title(f'{param_names[i]} - Running Std Convergence', fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 1].legend()
        
        axes[-1, 0].set_xlabel('Step Number', fontsize=12)
        axes[-1, 1].set_xlabel('Step Number', fontsize=12)
        plt.suptitle('MCMC Convergence Diagnostics - Running Statistics', fontsize=16)
        plt.tight_layout()
        plt.show()
        return fig

    def save_results(self, filename_prefix='mcmc_results'):
        """
        Save MCMC chains, posterior samples, and metadata to disk.

        Four files are written:

        * ``<filename_prefix>_chains.npy``      – full walker chains,
          shape ``(n_walkers, n_steps, n_params)``
        * ``<filename_prefix>_posteriors.npy``  – flat posterior samples
          after burn-in, shape ``(n_samples, n_params)``
        * ``<filename_prefix>_lnprob.npy``      – per-walker / per-step
          log-probability, shape ``(n_walkers, n_steps)``.  Needed for the
          chi^2 row in :meth:`plot_mcmc_traces`.
        * ``<filename_prefix>_metadata.json``   – parameter names, MLE
          values, medians, uncertainties, and convergence statistics

        Parameters
        ----------
        filename_prefix : str, optional
            Path prefix (including directory) for the output files.
            Defaults to ``'mcmc_results'`` (current directory).

        Returns
        -------
        saved_files : list[str]
            Absolute paths of the three files that were written.
        """
        if self.results is None:
            raise ValueError("No results available. Run perform_mcmc_analysis first.")

        sampler = self.results['sampler']
        samples = self.results['samples']

        chains_path = filename_prefix + '_chains.npy'
        posteriors_path = filename_prefix + '_posteriors.npy'
        lnprob_path = filename_prefix + '_lnprob.npy'
        metadata_path = filename_prefix + '_metadata.json'

        # Full chains: shape (n_walkers, n_steps, n_params)
        np.save(chains_path, sampler.chain)

        # Flat posterior samples after burn-in: shape (n_samples, n_params)
        np.save(posteriors_path, samples)

        # Per-walker / per-step log-probability: shape (n_walkers, n_steps)
        np.save(lnprob_path, sampler.lnprobability)

        # Metadata (JSON-serialisable scalars / lists only)
        autocorr_times = self.results['autocorr_times']
        metadata = {
            'params_to_fit': self.results['params_to_fit'],
            'burn_in': int(self.results['burn_in']),
            'n_walkers': int(sampler.chain.shape[0]),
            'n_steps': int(sampler.chain.shape[1]),
            'n_params': int(sampler.chain.shape[2]),
            'n_posterior_samples': int(samples.shape[0]),
            'has_lnprob': True,
            'mle_fitting_space': {k: float(v) for k, v in self.results['mle_fitting_space'].items()},
            'mle_linear': {k: float(v) for k, v in self.results['mle_linear'].items()},
            'all_model_params': {k: float(v) for k, v in self.results['all_model_params'].items()},
            'medians': self.results['medians'].tolist(),
            'upper_errors': self.results['upper_errors'].tolist(),
            'lower_errors': self.results['lower_errors'].tolist(),
            'mle_chi2': float(self.results['mle_chi2']),
            'mle_delta_chi2': float(self.results['mle_delta_chi2']),
            'mean_acceptance': float(self.results['mean_acceptance']),
            'autocorr_times': autocorr_times.tolist() if autocorr_times is not None else None,
            'r_hat_values': self.results['r_hat_values'].tolist(),
            'eff_sizes': self.results['eff_sizes'].tolist(),
            'convergence_issues': self.results['convergence_issues'],
            'recommendations': self.results['recommendations'],
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        saved_files = [
            os.path.abspath(chains_path),
            os.path.abspath(posteriors_path),
            os.path.abspath(lnprob_path),
            os.path.abspath(metadata_path),
        ]

        print(f"MCMC results saved:")
        for path in saved_files:
            print(f"  {path}")

        return saved_files

    def perform_mcmc_analysis(self, steps=3000, walkers=50, 
                             step_scale=1e-4, param_scales=None,
                             verbose=False, plot_corner=False, show_titles=True, plot_fit=False, 
                             plot_traces=False, plot_convergence=False, n_threads=None,
                             pool_init_data=None, save_chains=None, **kwargs):
        """
        Perform complete MCMC analysis of microlensing model.
        
        Parameters
        ----------
        steps : int, optional
            Number of MCMC steps (default: 3000)
        walkers : int, optional
            Number of walkers (default: 50)
        step_scale : float, optional
            Global step size scaling factor (default: 1e-4)
        param_scales : dict[str, float] | np.ndarray | None, optional
            Dict step sizes by param name (preferred) or array in params_to_fit order.
        verbose : bool, optional
            If True, print detailed diagnostics (default: False)
        plot_corner : bool, optional
            If True, show corner plot (default: False)
        plot_fit : bool, optional
            If True, show best fit model plot (default: False)
        plot_traces : bool, optional
            If True, show trace plots (default: False)
        plot_convergence : bool, optional
            If True, show convergence diagnostics (default: False)
        n_threads : int | None, optional
            Number of parallel processes to use for MCMC.
            If None, 0, or 1, runs in serial mode.
        pool_init_data : dict | None, optional
            Required when ``n_threads > 1``.  Supplies the information each
            worker process needs to construct its own ``mm.Event`` (avoiding
            pickling of the un-serialisable VBMicrolensing C++ object).
            Keys:

            * ``'data_file'`` (str) -- path to the data file
            * ``'data_kwargs'`` (dict) -- extra kwargs for ``mm.MulensData``
              (e.g. ``{'phot_fmt': 'mag', 'chi2_fmt': 'flux'}``)
            * ``'model_params'`` (dict) -- parameter dict for ``mm.Model``
            * ``'mag_methods'`` (list) -- args for
              ``model.set_magnification_methods()``
        save_chains : str | None, optional
            If provided, save chains, posteriors, and metadata using this string as
            the filename prefix (e.g. ``'results/binary_mcmc'``).
            Calls :meth:`save_results` automatically after the run.
            If None (default), nothing is saved.

        Returns
        -------
        results : dict
            Dictionary containing sampler, samples, MLE parameters, statistics, etc.
        """
        ndim = len(self.params_to_fit)
        n_data = len(self.event.datasets[0].flux)
        
        if verbose:
            print("Running MCMC with microlensing model...")
            x0 = self._get_initial_values()
            print("Initial parameters:", dict(zip(self.params_to_fit, x0)))
        
        # Create multiprocessing pool only when n_threads > 1.
        # Each worker creates its own mm.Event via _init_worker so that the
        # un-picklable VBMicrolensing C++ object never crosses processes.
        pool = None
        if n_threads is not None and n_threads > 1:
            if pool_init_data is None:
                raise ValueError(
                    "pool_init_data is required when n_threads > 1 to avoid "
                    "VBMicrolensing pickling errors.  Pass a dict with keys: "
                    "'data_file', 'data_kwargs', 'model_params', 'mag_methods'."
                )
            pool = Pool(
                processes=n_threads,
                initializer=_init_worker,
                initargs=(
                    pool_init_data['data_file'],
                    pool_init_data.get('data_kwargs', {}),
                    pool_init_data['model_params'],
                    pool_init_data['mag_methods'],
                    self.log_probability,
                    self.params_to_fit,
                    self.bounds,
                ),
            )
            if verbose:
                print(f"Using {n_threads} parallel processes for MCMC")
        elif verbose and n_threads is not None:
            print(f"Using serial mode (n_threads={n_threads})")
        
        try:
            # Run MCMC
            sampler, pos, prob, state = self.run_mcmc(
                steps=steps, walkers=walkers,
                step_scale=step_scale, param_scales=param_scales, 
                verbose=verbose, pool=pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
         
        # Calculate acceptance fraction
        mean_acceptance = np.mean(sampler.acceptance_fraction)
        if verbose:
            print(f"\nMean acceptance fraction: {mean_acceptance:.3f}")
        
        # Calculate autocorrelation time with error handling
        try:
            autocorr_times = sampler.get_autocorr_time()
            if verbose:
                print(f"Autocorrelation times: {autocorr_times}")
                print(f"Max autocorrelation time: {np.max(autocorr_times):.1f}")
                
                chain_length = sampler.chain.shape[1]
                steps_needed = int(50 * np.max(autocorr_times))
                if chain_length < steps_needed:
                    print(f"WARNING: Chain may not be converged. Consider running {steps_needed} steps.")
                else:
                    print("Chain length appears adequate for convergence.")
                    
        except Exception as e:
            if verbose:
                print(f"Cannot reliably calculate autocorrelation time: {e}")
                print("Using alternative convergence diagnostics...")
            autocorr_times = None
        
        # Determine burn-in period
        if autocorr_times is not None:
            burn_in = min(int(2 * np.max(autocorr_times)), int(0.5 * sampler.chain.shape[1]))
        else:
            burn_in = int(0.3 * sampler.chain.shape[1])
        
        if verbose:
            print(f"Using burn-in period: {burn_in} steps ({burn_in/sampler.chain.shape[1]*100:.1f}% of chain)")
        
        samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
        
        # Calculate convergence diagnostics
        post_burnin_chains = sampler.chain[:, burn_in:, :]
        r_hat_values = self.gelman_rubin_statistic(post_burnin_chains)
        eff_sizes = self.effective_sample_size(post_burnin_chains)
        
        if verbose:
            print(f"\nGelman-Rubin R-hat statistics (should be < 1.1 for convergence):")
            for i, param in enumerate(self.params_to_fit):
                if not np.isnan(r_hat_values[i]):
                    status = "✓ Good" if r_hat_values[i] < 1.1 else "⚠ Poor" if r_hat_values[i] < 1.2 else "✗ Bad"
                    print(f"{param}: {r_hat_values[i]:.4f} ({status})")
                else:
                    print(f"{param}: Unable to calculate")
            
            print(f"\nEffective sample sizes:")
            for i, param in enumerate(self.params_to_fit):
                print(f"{param}: {eff_sizes[i]:.0f} (out of {len(samples)} total samples)")
        
        # Find MLE sample
        max_likelihood_idx = np.argmax(sampler.lnprobability[:, burn_in:].flatten())
        mle_sample = samples[max_likelihood_idx]
        
        # Convert MLE to linear space and dict (fitted params only)
        mle_linear = self._convert_to_linear(mle_sample)
        
        # Also include values in fitting space
        mle_fitting_space = dict(zip(self.params_to_fit, mle_sample))
        
        # Get ALL model parameters (fitted + frozen) from the event
        all_model_params = dict(self.event.model.parameters.parameters)
        # Update with fitted MLE values (in linear space)
        all_model_params.update(mle_linear)
        
        # Calculate percentiles
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        medians = percentiles[1]
        upper_errors = percentiles[2] - percentiles[1]
        lower_errors = percentiles[1] - percentiles[0]
        
        # Set MLE parameters in event and calculate chi-squared
        chi2_mm(mle_sample, self.params_to_fit, self.event)
        mle_chi2 = self.event.get_chi2()
        
        # Degrees of freedom: n_data - n_params - 2 (for Fs, Fb)
        dof = n_data - ndim - 2
        mle_delta_chi2 = mle_chi2 - dof
        
        # Print basic results
        print("\nMaximum Likelihood Estimate (MLE) parameters:")
        print("  Fitted parameters (in fitting space):")
        for param, value in mle_fitting_space.items():
            print(f"    {param}: {value:.6f}")
        print("  All model parameters (linear space, fitted + frozen):")
        for param, value in all_model_params.items():
            frozen_marker = "" if param in mle_linear else " [frozen]"
            print(f"    {param}: {value:.6f}{frozen_marker}")
        
        print("\nMedian values with 1σ uncertainties (fitting space):")
        for i, param in enumerate(self.params_to_fit):
            print(f"  {param}: {medians[i]:.6f} +{upper_errors[i]:.6f} -{lower_errors[i]:.6f}")
        
        print(f"\nMLE Chi-squared: {mle_chi2:.2f}")
        print(f"MLE Reduced chi-squared: {mle_chi2/dof:.2f}")
        print(f"MLE Delta chi-squared: {mle_delta_chi2:.2f}")
        
        # Assessment and recommendations
        convergence_issues = []
        recommendations = []
        
        if mean_acceptance < 0.2:
            convergence_issues.append("Low acceptance fraction")
            recommendations.append("- Consider decreasing step size in proposal distribution")
        elif mean_acceptance > 0.7:
            convergence_issues.append("High acceptance fraction")
            recommendations.append("- Consider increasing step size in proposal distribution")
        
        if not np.isnan(r_hat_values).all():
            poor_rhat = np.sum((r_hat_values > 1.1) & ~np.isnan(r_hat_values))
            if poor_rhat > 0:
                convergence_issues.append(f"{poor_rhat} parameters have R-hat > 1.1")
                recommendations.append("- Run longer chains for better mixing")
        
        min_eff_size = np.min(eff_sizes)
        if min_eff_size < 100:
            convergence_issues.append("Low effective sample sizes")
            recommendations.append("- Run longer chains to increase effective sample size")
        
        if autocorr_times is not None:
            max_tau = np.max(autocorr_times)
            chain_length = sampler.chain.shape[1]
            if chain_length < 50 * max_tau:
                convergence_issues.append("Chain too short relative to autocorrelation time")
                recommendations.append(f"- Run at least {int(50 * max_tau)} steps for reliable results")
        
        print("\n" + "="*60)
        print("CONVERGENCE SUMMARY")
        print("="*60)
        
        if convergence_issues:
            print("⚠ POTENTIAL CONVERGENCE ISSUES:")
            for issue in convergence_issues:
                print(f"  • {issue}")
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("✅ MCMC appears to have converged well!")
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Total steps: {sampler.chain.shape[1]}")
        print(f"  • Burn-in: {burn_in} steps")
        print(f"  • Effective samples: {len(samples)}")
        print(f"  • Acceptance rate: {mean_acceptance:.3f}")
        if autocorr_times is not None:
            print(f"  • Max autocorr time: {np.max(autocorr_times):.1f}")
        print(f"  • Min effective size: {min_eff_size:.0f}")
        if not np.isnan(r_hat_values).all():
            print(f"  • Max R-hat: {np.nanmax(r_hat_values):.3f}")
        
        print("\n" + "="*60)
        
        # Build results dictionary and save *before* plotting so that a
        # plotting error can never discard a long MCMC run.
        results = {
            'sampler': sampler,
            'samples': samples,
            'params_to_fit': self.params_to_fit,
            'mle_sample': mle_sample,
            'mle_linear': mle_linear,
            'mle_fitting_space': mle_fitting_space,
            'all_model_params': all_model_params,  # fitted + frozen in linear space
            'medians': medians,
            'upper_errors': upper_errors,
            'lower_errors': lower_errors,
            'mle_chi2': mle_chi2,
            'mle_delta_chi2': mle_delta_chi2,
            'burn_in': burn_in,
            'mean_acceptance': mean_acceptance,
            'autocorr_times': autocorr_times,
            'r_hat_values': r_hat_values,
            'eff_sizes': eff_sizes,
            'convergence_issues': convergence_issues,
            'recommendations': recommendations
        }
        
        self.results = results

        if save_chains is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_chains + '_chains.npy')), exist_ok=True)
            self.save_results(filename_prefix=save_chains)

        # Generate plots if requested
        if plot_corner:
            self.plot_corner_mcmc(samples, self.params_to_fit, show_titles=show_titles, **kwargs)
        
        if plot_fit:
            self.plot_mcmc_fit(mle_delta_chi2)
        
        if plot_traces:
            self.plot_mcmc_traces(sampler, self.params_to_fit, burn_in)
        
        if plot_convergence:
            self.plot_convergence_diagnostics(sampler, self.params_to_fit, burn_in)

        return self.results
