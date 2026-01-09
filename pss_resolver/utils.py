import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .fit import mcr_factors,get_acceptable_solutions,calc_reconstruction_error
from typing import Optional,Union



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

def export_to_csv(title: str, dtype: str, data: Union[list,np.ndarray], wavelengths: Optional[np.ndarray]=None) -> None:
    """ Exports a data matrix or list to CSV file. Each file is named title_dtype.csv where dtype is 'C', 'S', or 'D'.
    
    Args:
        title (str): Base title for the CSV file.
        dtype (str): Type of data: 'C' for concentration, 'S' for spectra, 'D' for data matrix.
        data (np.ndarray or list): Data matrix or a list of matrices corresponding to different valid solutions.
        wavelengths (np.ndarray, optional): Wavelengths corresponding to the rows of D and S. Required if dtype is 'S' or 'D'.
    """
    if dtype not in ['C','S','D']:
        raise ValueError("dtype must be one of 'C', 'S', or 'D'")
    if dtype in ['S','D'] and wavelengths is None:
        raise ValueError("wavelengths must be provided when dtype is 'S' or 'D'")
    df=None
    if dtype == 'C':
        labels = []
        if isinstance(data,list):
            n_solutions = len(data)
            data = np.vstack(data)
            block_size = data.shape[0] // n_solutions
            
            for sol in range(n_solutions):
                for sample in range(block_size):
                    labels.append(f'Solution {sol+1} sample {sample+1}')

        if len(labels)>0:
            df = pd.DataFrame(data, index=labels,columns=['Component 1', 'Component 2'])
        else:
            df = pd.DataFrame(data, columns=['Component 1', 'Component 2'])

    elif dtype == 'S':
        labels = []
        if isinstance(data,list):
            n_solutions = len(data)
            data = np.vstack(data)
            block_size = data.shape[0] // n_solutions

            for sol in range(n_solutions):
                for species in range(block_size):
                    labels.append(f'Solution {sol+1} species {species+1}')

            #data = data.T
        if len(labels)>0:
            df = pd.DataFrame(data, index=labels, columns=[f'{w} nm' for w in wavelengths])
        else:
            df = pd.DataFrame(data, columns=[f'{w} nm' for w in wavelengths])
        # df.index.name = 'Wavelength (nm)'

    elif dtype == 'D':
        df = pd.DataFrame(data.T, index=wavelengths)
        df.index.name = 'Wavelength (nm)'
        df.columns = [f'Sample {i+1}' for i in range(data.shape[0])]
    
    if isinstance(df,pd.DataFrame):
        df.to_csv(f"{title}_{dtype}.csv")
    else:
        raise ValueError("DataFrame creation failed.")

def export_dcs_to_csv(title: str, wavelengths: np.ndarray, D: np.ndarray, C: Union[list,np.ndarray], S: Union[list,np.ndarray]):
    """ Exports the D, C, and S matrices to CSV files. Each file is named title_D.csv, title_C.csv, and title_S.csv respectively.
    
    Args:
        title (str): Base title for the CSV files.
        wavelengths (np.ndarray): Wavelengths corresponding to the rows of D and S.
        D (np.ndarray): Data matrix.
        C (np.ndarray or list): Concentration matrix or a list of concentration matrices corresponding to different valid solutions.
        S (np.ndarray or list): Spectra matrixor a list of concentration matrices corresponding to different valid solutions.
    """
    export_to_csv(title, 'D', D, wavelengths)
    export_to_csv(title, 'C', C)
    export_to_csv(title, 'S', S, wavelengths)
    