[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martinp23/pss-resolver/HEAD?urlpath=%2Fdoc%2Ftree%2Fdemo+notebook.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/pss-resolver.svg)](https://pypi.org/project/pss-resolver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# pss-resolver

PSS resolver is designed for studying molecular photoswitches. It uses non-negative matrix factorization (NNMF) to extract the pure spectrum of a metastable photoisomer from a series of photostationary state spectra, obtained under irradiation at different wavelengths. Unless one PSS contains 100% metastable isomer, the NNMF solution has some ambiguity. The PSS resolver tool presents the user with a range of possible solutions.

## Quick start

The repository contains a Jupyter notebook demonstrating the use of the library. You can run this notebook online, without installing anything on your own computer, using Binder. [Click here](https://mybinder.org/v2/gh/martinp23/pss-resolver/HEAD?urlpath=%2Fdoc%2Ftree%2Fdemo+notebook.ipynb).

## Installation and usage

For more advanced use, you will need to install python on your system. You can then install the `pss_resolver` library using `pip`:

```bash
pip install pss-resolver
```

And import it into your project. For examples of usage, follow the [Jupyter notebook](demo%20notebook.ipynb).

## Help and support

If you have any questions or need help with the library, please feel free to open an issue on the GitHub repository or contact Martin (m.peeks@unsw.edu.au).