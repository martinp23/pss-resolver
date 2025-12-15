import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .fit import mcr_factors,get_acceptable_solutions,calc_reconstruction_error



def pymcr_handler_for_file(file: str, threshold: float=1.001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_excel(file,index_col=0)
    X = data.values[:,:].T
    print(f"##### Results for file: {file} #####")
    return proc_data(data.index,X,data.columns,threshold=threshold)


def proc_data(wavelengths,X,labels,threshold=1.001):

    c,spec,X_calc = mcr_factors(X, n_components=2, known_id=0, init_guess="nmf",method='mvol')
    res_ST,res_C = get_acceptable_solutions(X, spec, c, n=201, lb=-1, ub=1,threshold=threshold)

    min_C = np.min(np.array(res_C),axis=0)
    max_C = np.max(np.array(res_C),axis=0)
    print(" #### Reconstruction error: ####")
    print(calc_reconstruction_error(X,c,spec))
    
    print(" #### Isomer Ratios: ####")
    for i,x in enumerate(labels):
        # print(f"{x}: {c[i,0]:.2f}:{c[i,1]:.2f}")
        print(f"{x}:  Acceptable solutions: {min_C[i,0]:.2f}-{max_C[i,0]:.2f} : {min_C[i,1]:.2f}-{max_C[i,1]:.2f}")
    plt.subplots(2,1,figsize=(4,5))
    plt.subplot(211)
    plt.plot(wavelengths,X_calc.T,label=labels)
    plt.plot(wavelengths,X.T,'--')
    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance')
    plt.subplot(212)
    plt.plot(wavelengths,spec[0,:].T,label='Known spec')
    for ii,spec in enumerate(res_ST):
        if ii == 0:
            plt.plot(wavelengths,spec[1,:].T,'r',label=['Extracted spec'],alpha=0.3)
        else:
            plt.plot(wavelengths,spec[1,:].T,'r',alpha=0.3)
    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance')
    plt.tight_layout()
    plt.show()

    return c,res_ST,res_C