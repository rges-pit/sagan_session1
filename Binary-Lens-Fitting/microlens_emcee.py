"""
Microlensing MCMC Analysis Module

This module provides a class-based interface for performing MCMC analysis
on microlensing models using the emcee package.
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocess import Pool
from scipy.optimize import lsq_linear
import VBMicrolensing as vbm


# ============================================================================
# External Model Functions (can be overridden by user)
# ============================================================================

def get_mag(x0, p, t):
    '''
    Get the magnification curve for a given model and parameters.
    
    Parameters
    ----------
    x0: np.ndarray with initial parameters (t0, tE, u0, rho, s, q, alpha)
    p: list[str] with parameter names
    t: np.ndarray with data epochs
    
    Returns
    -------
    mag: np.ndarray with magnification curve
    '''
    # We will use the VBMicrolensing package to calculate magnifications for different source positions.
    VBM = vbm.VBMicrolensing()
     #create a dictionary of parameters
    params = dict(zip(p, x0))
    paramsc = params.copy()
    tau = (t - paramsc['t0']) / paramsc['tE'] 
    # Single lens model
    if len(p) < 7 :
        ul = np.sqrt(tau**2 + paramsc['u0']**2)
        if 'logrho' in paramsc.keys():
            paramsc['rho'] = 10**(paramsc['logrho'])
        mag = np.array([VBM.ESPLMag2(u, paramsc['rho']) for u in ul])
    # Binary lens model
    elif len(p) >= 7:
        salpha = np.sin(np.radians(paramsc['alpha']))
        calpha = np.cos(np.radians(paramsc['alpha']))
        xs = -paramsc['u0'] * salpha + tau * calpha 
        ys = paramsc['u0'] * calpha + tau * salpha
        if 'logs' in paramsc.keys():
            paramsc['s'] = 10**(paramsc['logs'])
        if 'logq' in paramsc.keys():
            paramsc['q'] = 10**(paramsc['logq'])
        if 'logrho' in paramsc.keys():
            paramsc['rho'] = 10**(paramsc['logrho'])
        mag = np.array([VBM.BinaryMag2(paramsc['s'], paramsc['q'], xs[i], ys[i], paramsc['rho']) for i in range(len(ys))])
    return mag


def calc_Fs(modelmag: np.ndarray, f: np.ndarray, sig2: np.ndarray) -> tuple[float, float]:
    '''
    Solves for the flux parameters for a given model using least squares.
    
    Parameters
    ----------
    model : np.ndarray
        Model magnification curve.
    f : np.ndarray
        Observed flux values.
    sig2 : np.ndarray
        Flux errors.
    
    Returns
    -------
    FS : float
        Source flux.
    FB : float
        Blend flux.
    '''

    """
    Solves for FS, FB by weighted least squares with FB >= 0 using lsq_linear.
    """
    modelmag = np.asarray(modelmag, float).ravel()
    f = np.asarray(f, float).ravel()
    sig2 = np.asarray(sig2, float).ravel()

    w = 1.0 / np.sqrt(sig2)
    A = np.column_stack([modelmag, np.ones_like(modelmag)])
    Aw = A * w.reshape(-1, 1)
    bw = f * w

    res = lsq_linear(Aw, bw)
    FS, FB = res.x
    return float(FS), float(FB)


def chi2(x0: np.ndarray, p: list, t: np.ndarray, f: np.ndarray, sig: np.ndarray) -> float:
    '''
    Calculates the chi squared value for a given model and parameters.
    
    Parameters
    ----------
    x0 : np.ndarray
        Initial parameters.
    p : List[str]
        List of parameter names.
    t : np.ndarray
        Data epochs.
    f : np.ndarray
        Observed flux values.
    sig : np.ndarray
        Flux errors.
    
    Returns
    -------
    chi2 : float
        Chi squared value.
   '''
    mag = get_mag(x0, p, t)
    FS, FB = calc_Fs(mag, f, sig**2)
    model = FS * mag + FB
    
    chi2_value = np.sum((f - model)**2 / sig**2)
    return chi2_value


# ============================================================================
# Main MCMC Analysis Class
# ============================================================================

class micro_mc:
    """
    A class for performing MCMC analysis on microlensing models.
    
    This class encapsulates all functionality needed to fit microlensing
    light curves using MCMC methods, including convergence diagnostics
    and visualization tools.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data epochs, flux, and flux errors.
        Expected columns: 'HJD-2450000', 'I_band_flux', 'I_band_flux_err'
    x0 : np.ndarray
        Initial parameter values
    param_names : list[str]
        List of parameter names
    log_probability : callable
        User-provided log probability function with signature:
        log_probability(x, p, t, f, sig, bounds) -> float
    bounds : dict | None, optional
        Optional map of parameter name -> (min, max). If None, use defaults.
    fixed_params : dict | None, optional
        Parameters to freeze as {name: value}. Only other parameters are sampled.
    get_mag_func : callable | None, optional
        Custom magnification function. If None, uses default get_mag.
    calc_Fs_func : callable | None, optional
        Custom flux calculation function. If None, uses default calc_Fs.
    chi2_func : callable | None, optional
        Custom chi-squared function. If None, uses default chi2.
    
    Attributes
    ----------
    results : dict
        Dictionary containing MCMC analysis results (set after running perform_mcmc_analysis)
    """
    
    def __init__(self, df, x0, param_names, log_probability, 
                 bounds=None, fixed_params=None,
                 get_mag_func=None, calc_Fs_func=None, chi2_func=None):
        """Initialize the micro_mc analysis class.
        
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data epochs, flux, and flux errors.
        Expected columns: 'HJD-2450000', 'I_band_flux', 'I_band_flux_err'
    x0 : np.ndarray
        Initial parameter values
    param_names : list[str]
        List of parameter names
    log_probability : callable
        User-provided log probability function with signature:
        log_probability(x, p, t, f, sig, bounds) -> float
    bounds : dict | None, optional
        Optional map of parameter name -> (min, max). If None, use defaults.
    fixed_params : dict | None, optional
        Parameters to freeze as {name: value}. Only other parameters are sampled.
    get_mag_func : callable | None, optional
        Custom magnification function. If None, uses default get_mag.
    calc_Fs_func : callable | None, optional
        Custom flux calculation function. If None, uses default calc_Fs.
    chi2_func : callable | None, optional
        Custom chi-squared function. If None, uses default chi2.
    
    Attributes
    ----------
    results : dict
        Dictionary containing MCMC analysis results (set after running perform_mcmc_analysis)
    """
        self.df = df
        self.x0 = x0
        self.param_names = param_names
        self.log_probability = log_probability
        self.bounds = bounds
        self.fixed_params = fixed_params
        
        # Store function references (use defaults if not provided)
        self._get_mag_func = get_mag_func if get_mag_func is not None else get_mag
        self._calc_Fs_func = calc_Fs_func if calc_Fs_func is not None else calc_Fs
        self._chi2_func = chi2_func if chi2_func is not None else chi2
        
        # Results will be stored here after running analysis
        self.results = None
    
    def run_mcmc(self, steps=500, walkers=100, step_scale=1e-4, 
                 param_scales=None, verbose=False, pool=None):
        """
        Fit an "event" with "parameters_to_fit" as free parameters.
        
        Parameters
        ----------
        steps : int, optional
            Number of MCMC steps (default: 500)
        walkers : int, optional
            Number of walkers (default: 100)
        step_scale : float, optional
            Global step size scaling factor (default: 1e-4)
        param_scales : dict[str, float] | np.ndarray | None, optional
            Either dict of per-parameter step sizes or array in p-order.
        verbose : bool, optional
            Print step size information (default: False)
        pool : multiprocessing.Pool | None, optional
            Multiprocessing pool for parallel evaluation of log probability across walkers.
            If None, runs in serial mode. (default: None)
        
        Returns
        -------
        sampler, pos, prob, state, free_param_names
            MCMC sampler results
        """
        # Take the initial starting point from the event.
        t = np.array(self.df['HJD-2450000'])
        f = np.array(self.df['I_band_flux'])
        sig = np.array(self.df['I_band_flux_err'])

        # Default bounds for clipping initial positions
        default_bounds = {
            't0': (t.min(), t.max()),
            'tE': (0.1, t.max() - t.min()),
            'u0': (-10.0, 10.0),
            'rho': (1e-5, 1.0),
            's': (1e-2, 20.0),
            'q': (1e-7, 1.0),
            'alpha': (0.0, 360.0),
            'logs': (-2, 2),
            'logq': (-7, 0),
            'logrho': (-5, 0),
        }
        bounds_map = default_bounds.copy()
        if self.bounds is not None:
            bounds_map.update({k: tuple(v) for k, v in self.bounds.items() if k in bounds_map})

        fixed_params = self.fixed_params or {}
        fixed_params = {k: float(v) for k, v in fixed_params.items()}

        # free/fixed split
        free_param_names = [name for name in self.param_names if name not in fixed_params]
        ndim = len(free_param_names)
        if ndim == 0:
            raise ValueError("No free parameters to sample. Provide some free parameters or omit fixed_params.")

        # Build x0 map and free starting positions
        x0_map = dict(zip(self.param_names, self.x0))
        x0_free = np.array([x0_map[name] for name in free_param_names])

        # Step sizes dict -> array for free params
        if isinstance(param_scales, dict):
            scales_free = np.array([param_scales.get(name, step_scale) for name in free_param_names])
            if verbose:
                print("Using dict step sizes for free parameters:")
                for name, sc in zip(free_param_names, scales_free):
                    print(f"  {name}: {sc:.2e}")
        elif isinstance(param_scales, np.ndarray):
            if len(param_scales) != len(self.param_names):
                raise ValueError(f"param_scales array must match param_names length ({len(self.param_names)}).")
            name_to_scale = dict(zip(self.param_names, param_scales))
            scales_free = np.array([name_to_scale[name] for name in free_param_names])
            if verbose:
                print("Using array step sizes mapped to free parameters:")
                for name, sc in zip(free_param_names, scales_free):
                    print(f"  {name}: {sc:.2e}")
        elif param_scales is None:
            scales_free = np.array([step_scale] * ndim)
            if verbose:
                print(f"Using global step size: {step_scale:.2e}")
        else:
            raise ValueError("param_scales must be a dict, numpy array, or None")

        # Create initial starting points for all walkers in free space
        nwalkers = walkers
        p0state_free = x0_free + scales_free * np.random.randn(nwalkers, ndim)

        # Clip initial positions to within bounds for free params
        for j, name in enumerate(free_param_names):
            lower, upper = bounds_map.get(name, (-np.inf, np.inf))
            p0state_free[:, j] = np.clip(p0state_free[:, j], lower, upper)

        # Log prob wrapper that reconstructs full vector
        def log_prob_free(theta_free: np.ndarray) -> float:
            full_map = {**x0_map, **fixed_params}
            for j, name in enumerate(free_param_names):
                full_map[name] = float(theta_free[j])
            x_full = np.array([full_map[name] for name in self.param_names])
            return self.log_probability(x_full, self.param_names, t, f, sig, self.bounds)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_free, pool=pool)

        # Run MCMC
        pos, prob, state = sampler.run_mcmc(p0state_free, steps, progress=True)

        return sampler, pos, prob, state, free_param_names

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
            Parameter names. If None, uses self.results['free_param_names']
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
            param_names = self.results['free_param_names']
        
        fig = corner.corner(samples, labels=param_names, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=show_titles, title_kwargs={"fontsize": 12},
                           label_kwargs={"fontsize": 14}, **kwargs)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_mcmc_fit(self, mle_params=None, param_names=None, mle_delta_chi2=None):
        """
        Plot the best fit model from MCMC
        
        Parameters
        ----------
        mle_params : np.ndarray, optional
            MLE parameter values. If None, uses self.results['mle_params']
        param_names : list, optional
            Parameter names. If None, uses self.param_names
        mle_delta_chi2 : float, optional
            Delta chi-squared value. If None, uses self.results['mle_delta_chi2']
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The fit plot figure
        """
        if mle_params is None:
            if self.results is None:
                raise ValueError("No results available. Run perform_mcmc_analysis first.")
            mle_params = self.results['mle_params']
            param_names = self.param_names
            mle_delta_chi2 = self.results['mle_delta_chi2']
        
        mle_modelmag = self._get_mag_func(mle_params, param_names, np.array(self.df['HJD-2450000']))
        source_flux, blend_flux = self._calc_Fs_func(mle_modelmag, np.array(self.df['I_band_flux']), np.array(self.df['I_band_flux_err'])**2)
        mle_model = source_flux * mle_modelmag + blend_flux
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.errorbar(np.array(self.df['HJD-2450000']), np.array(self.df['I_band_flux']), yerr=np.array(self.df['I_band_flux_err']), 
                     fmt='o', color='black', label='Data', markersize=2, alpha=0.7)
        plt.plot(np.array(self.df['HJD-2450000']), mle_model, color='red', linewidth=2, 
                 label=f'MCMC Best Fit - $\\Delta \\chi^2 = {mle_delta_chi2:.2f}$')
        plt.xlabel('HJD - 2450000', fontsize=12)
        plt.ylabel('Flux', fontsize=12)
        plt.title('Binary Lens Model - MCMC Fit', fontsize=14)
        plt.legend(loc='upper left', framealpha=0.8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def plot_mcmc_traces(sampler, param_names, burn_in):
        """
        Plot MCMC trace plots for convergence diagnosis
        
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
            The trace plot figure
        """
        fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 2*len(param_names)), sharex=True)
        if len(param_names) == 1:
            axes = [axes]
        
        for i in range(len(param_names)):
            # Plot all walker chains
            for j in range(sampler.chain.shape[0]):
                axes[i].plot(sampler.chain[j, :, i], color='k', alpha=0.4, linewidth=0.5)
            
            # Highlight a few random walkers for clarity
            random_walkers = np.random.choice(sampler.chain.shape[0], size=min(5, sampler.chain.shape[0]), replace=False)
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for idx, walker in enumerate(random_walkers):
                color = colors[idx % len(colors)]
                axes[i].plot(sampler.chain[walker, :, i], color=color, alpha=0.7, linewidth=1.2, 
                            label=f'Walker {walker}' if i == 0 else '')
            
            axes[i].axvline(burn_in, color='red', linestyle='--', linewidth=2, label='Burn-in' if i == 0 else '')
            axes[i].set_ylabel(f'{param_names[i]}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
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

    def perform_mcmc_analysis(self, steps=3000, walkers=50, 
                             step_scale=1e-4, param_scales=None,
                             verbose=False, plot_corner=False, show_titles=True, plot_fit=False, 
                             plot_traces=False, plot_convergence=False, n_threads=None, **kwargs):
        """
        Perform complete MCMC analysis of binary lens model
        
        Parameters
        ----------
        steps : int, optional
            Number of MCMC steps (default: 3000)
        walkers : int, optional
            Number of walkers (default: 50)
        step_scale : float, optional
            Global step size scaling factor (default: 1e-4)
        param_scales : dict[str, float] | np.ndarray | None, optional
            Dict step sizes by param name (preferred) or array in p-order.
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
            Number of parallel threads to use for MCMC. If None, runs in serial mode.
            Uses Python's multiprocessing Pool to parallelize log probability evaluations
            across walkers. (default: None)
        
        Returns
        -------
        results : dict
            Dictionary containing sampler, samples, MLE parameters, statistics, etc.
            Also stored in self.results
        """
        
        if verbose:
            print("Running MCMC with binary lens model...")
            print("Initial parameters:", dict(zip(self.param_names, self.x0)))
        
        # Create multiprocessing pool if n_threads is specified
        pool = None
        if n_threads is not None:
            pool = Pool(processes=n_threads)
            if verbose:
                print(f"Using {n_threads} parallel processes for MCMC")
        
        try:
            # Run MCMC
            sampler, pos, prob, state, free_param_names = self.run_mcmc(
                steps=steps, walkers=walkers,
                step_scale=step_scale, param_scales=param_scales, 
                verbose=verbose, pool=pool)
        finally:
            # Clean up pool
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
        
        samples = sampler.chain[:, burn_in:, :].reshape((-1, len(free_param_names)))
        
        # Calculate convergence diagnostics
        post_burnin_chains = sampler.chain[:, burn_in:, :]
        r_hat_values = self.gelman_rubin_statistic(post_burnin_chains)
        eff_sizes = self.effective_sample_size(post_burnin_chains)
        
        if verbose:
            print(f"\nGelman-Rubin R-hat statistics (should be < 1.1 for convergence):")
            for i, param in enumerate(free_param_names):
                if not np.isnan(r_hat_values[i]):
                    status = "✓ Good" if r_hat_values[i] < 1.1 else "⚠ Poor" if r_hat_values[i] < 1.2 else "✗ Bad"
                    print(f"{param}: {r_hat_values[i]:.4f} ({status})")
                else:
                    print(f"{param}: Unable to calculate")
            
            print(f"\nEffective sample sizes:")
            for i, param in enumerate(free_param_names):
                print(f"{param}: {eff_sizes[i]:.0f} (out of {len(samples)} total samples)")
        
        # Find MLE in free space and reconstruct full params
        max_likelihood_idx = np.argmax(sampler.lnprobability[:, burn_in:].flatten())
        mle_free = samples[max_likelihood_idx]
        
        # Reconstruct full parameter vector from free samples
        x0_map = dict(zip(self.param_names, self.x0))
        fixed_params_map = self.fixed_params or {}
        full_map = {**x0_map, **fixed_params_map}
        for j, name in enumerate(free_param_names):
            full_map[name] = float(mle_free[j])
        mle_params = np.array([full_map[name] for name in self.param_names])
        mle_dict = dict(zip(self.param_names, mle_params))
        
        # Calculate percentiles on free params only
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        medians_free = percentiles[1]
        upper_errors_free = percentiles[2] - percentiles[1]
        lower_errors_free = percentiles[1] - percentiles[0]
        
        # Calculate chi-squared for full MLE
        mle_chi2 = self._chi2_func(mle_params, self.param_names, np.array(self.df['HJD-2450000']), 
                                    np.array(self.df['I_band_flux']), np.array(self.df['I_band_flux_err']))
        dof_params = len(free_param_names)
        mle_delta_chi2 = mle_chi2 - (len(self.df) - dof_params)
        
        # Print basic results (always shown)
        print("\nMaximum Likelihood Estimate (MLE) parameters (full):")
        for param, value in mle_dict.items():
            print(f"{param}: {value:.6f}")
        
        print("\nMedian values with 1σ uncertainties (free params only):")
        for i, param in enumerate(free_param_names):
            print(f"{param}: {medians_free[i]:.6f} +{upper_errors_free[i]:.6f} -{lower_errors_free[i]:.6f}")
        
        print(f"\nMLE Chi-squared: {mle_chi2:.2f}")
        print(f"MLE Reduced chi-squared: {mle_chi2/(len(self.df)-dof_params):.2f}")
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
        
        # Generate plots if requested
        if plot_corner:
            self.plot_corner_mcmc(samples, free_param_names, show_titles=show_titles, **kwargs)
        
        if plot_fit:
            self.plot_mcmc_fit(mle_params, self.param_names, mle_delta_chi2)
        
        if plot_traces:
            self.plot_mcmc_traces(sampler, free_param_names, burn_in)
        
        if plot_convergence:
            self.plot_convergence_diagnostics(sampler, free_param_names, burn_in)
        
        # Build results dictionary
        results = {
            'sampler': sampler,
            'samples': samples,  # Free parameter samples only
            'free_param_names': free_param_names,
            'fixed_params': fixed_params_map,
            'mle_params': mle_params,  # Full parameter vector
            'mle_dict': mle_dict,  # Full parameter dict
            'medians': medians_free,  # Free parameter medians
            'upper_errors': upper_errors_free,  # Free parameter upper errors
            'lower_errors': lower_errors_free,  # Free parameter lower errors
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
        
        # Store results as class attribute and return
        self.results = results
        return self.results
