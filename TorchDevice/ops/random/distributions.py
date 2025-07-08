"""
TorchDevice Random Distributions Module
----------------------------------
Tensor creation and distribution functions.
"""

import torch
from ...core.logger import log_info

# Store patch status to avoid infinite recursion
_patched = False


def apply_patches() -> None:
    """Apply tensor creation patches."""
    global _patched

    # Only patch once
    if _patched:
        return

    from ...core.device import DeviceManager  # Local import
    # Set flag before patching to avoid recursion
    _patched = True

    # Patch torch.distributions functions
    if hasattr(torch, "distributions"):
        # Only patch if we have distributions in torch
        log_info("Patching torch.distributions")

        # Patch distribution constructors with device parameters
        for dist_name in [
            'Bernoulli', 'Beta', 'Binomial', 'Categorical', 'Cauchy',
            'Chi2', 'ContinuousBernoulli', 'Dirichlet', 'Exponential',
            'FisherSnedecor', 'Gamma', 'Geometric', 'Gumbel',
            'HalfCauchy', 'HalfNormal', 'Independent', 'Laplace',
            'LogNormal', 'LowRankMultivariateNormal', 'Multinomial',
            'MultivariateNormal', 'NegativeBinomial', 'Normal',
            'OneHotCategorical', 'Pareto', 'Poisson', 'RelaxedBernoulli',
            'RelaxedOneHotCategorical', 'StudentT', 'Uniform', 'VonMises',
            'Weibull'
        ]:

            if hasattr(torch.distributions, dist_name):
                dist_class = getattr(torch.distributions, dist_name)
                orig_init = dist_class.__init__

                # Patch the __init__ method to handle device redirection
                def patched_init(self, *args, **kwargs):
                    # Process device argument if present
                    if 'device' in kwargs:
                        kwargs['device'] = DeviceManager.torch_device_replacement(kwargs['device'])
                    # Call original init
                    orig_init(self, *args, **kwargs)

                dist_class.__init__ = patched_init
                log_info("Patched %s distribution", dist_name)


def apply_patches_all() -> None:
    """Convenience function to patch all distributions."""
    apply_patches()


# Module initialization
log_info("Initializing TorchDevice random distributions module")

__all__: list[str] = [
    'apply_patches',
    'apply_patches_all'
]

log_info("TorchDevice random distributions module initialized")