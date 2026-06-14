"""PSS Resolver: Multivariate curve resolution for photostationary state spectra."""

__version__ = "0.2.0"

from pss_resolver.fit import mcr_factors, get_acceptable_solutions, calc_reconstruction_error
from pss_resolver.simulate import gen_pss_specs
from pss_resolver.utils import (
    pymcr_handler_for_file,
    proc_data,
    export_dcs_to_csv,
    export_to_csv,
)

__all__ = [
    "mcr_factors",
    "get_acceptable_solutions",
    "calc_reconstruction_error",
    "gen_pss_specs",
    "pymcr_handler_for_file",
    "proc_data",
    "export_dcs_to_csv",
    "export_to_csv",
]
