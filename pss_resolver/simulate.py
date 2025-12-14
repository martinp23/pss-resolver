# mamba create -n general-fitting scipy numpy lmfit pymcr scikit-learn matplotlib ipykernel openpyxl pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
try:
    from pymcr.mcr import McrAR
    from pymcr.regressors import NNLS, OLS
    from pymcr.constraints import ConstraintNonneg, ConstraintNorm
    PYMCR_AVAILABLE = True
except ImportError:
    print("pymcr not found, proceeding without it.")
    PYMCR_AVAILABLE = False
try:
    import mcrnmf
    MCRNMF_AVAILABLE = True
except ImportError:
    print("mcrnmf not found, proceeding without it.")
    MCRNMF_AVAILABLE = False

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def spec_gauss(x,ngauss, minw=100, rel_noise=0.001):
    specs = np.zeros((ngauss, len(x)))
    for i in range(ngauss):
        amp = np.random.uniform(0.1,.6)
        cen = np.random.uniform(x[0]+20, x[-1]-20)
        wid = np.random.uniform(minw, minw*12)
        specs[i,:] = gaussian(x, amp, cen, wid)
        specs[i,:] += np.random.normal(scale=rel_noise*amp, size=len(x))
    allspec =  specs.sum(axis=0)
    return allspec/np.max(allspec)

def wavenumber_to_nm(wavenumber: np.ndarray) -> np.ndarray:
    return 1e7 / wavenumber

def nm_to_wavenumber(nm: np.ndarray) -> np.ndarray:
    return 1e7 / nm

def gen_pss_specs(x=None,npss = 5, fA=1):
    if x is None:
      x = np.linspace(200,800,601)
    n_spectra = 10
    # first simulate two unique basis spectra
    s1 = spec_gauss(nm_to_wavenumber(x), n_spectra, minw=500)
    s2 = spec_gauss(nm_to_wavenumber(x), n_spectra, minw=500)

    # now create linear combinations of them

    if fA is not None:
        if isinstance(fA, (int,float)):
            fA = [1, *np.random.uniform(0.001,1,npss-1)]
        elif len(fA) != npss:
            raise ValueError("fA must be a scalar or a list/array of length npss.")
        else:
            fA = np.array(fA)
    else:
        fA = np.random.uniform(0.001,1,npss)
    
    fB = 1 - np.array(fA)
    
    C = np.vstack([fA, fB]).T
    ST = np.vstack([s1, s2])

    return C@ST, C,ST


def simulate_and_plot(start=200,end=800,nspec=5):
  x = np.linspace(start,end,end-start+1)
  s,C,ST = gen_pss_specs(npss=nspec, fA=1)
  plt.plot(x,s.T)
  plt.xlabel('Wavelength [nm]')
  plt.ylabel('Absorbance')
  plt.title(f'Simulated PSS Spectra ({C.shape[0]} spectra)')
  plt.show()
  plt.figure()
  plt.plot(x,ST.T)
  plt.title("Basis spectra")
  plt.show()
  return s,C,ST


