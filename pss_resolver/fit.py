import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

REG=1
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


def mse(C: np.ndarray, ST: np.ndarray, D_actual: np.ndarray, D_calculated: np.ndarray) -> float:
    """ Follows mse signature from pymcr.metrics"""
    return ((D_actual - D_calculated)**2).sum()/D_actual.size

def mse_regularized(C: np.ndarray, ST: np.ndarray, D_actual: np.ndarray, D_calculated: np.ndarray) -> float:
    """ Follows mse signature from pymcr.metrics"""
    # calculate first-derivative regularization on ST
    first_deriv = np.diff(ST, n=1, axis=1)
    reg_term = REG * (first_deriv**2).sum()/first_deriv.size

    print(f"Regularization term: {reg_term:.2e}")
    return ((D_actual - D_calculated)**2).sum()/D_actual.size + reg_term

def mse_scaled(C: np.ndarray, ST: np.ndarray, D_actual: np.ndarray, D_calculated: np.ndarray) -> float:
    """ Scale the error such that areas of low absorption in ST are weighted more (i.e. scrutiny ~0)."""
    weights = np.min(D_actual, axis=0)
    return (((D_actual - D_calculated)**2) * weights).sum()/D_actual.size

def calc_frobenius_error(X: np.ndarray, C: np.ndarray, ST: np.ndarray) -> float:
    X_calc = C @ ST
    return np.linalg.norm(X - X_calc, 'fro')

def calc_reconstruction_error(X: np.ndarray, C: np.ndarray, ST: np.ndarray) -> float:
    return calc_frobenius_error(X,C,ST) / np.linalg.norm(X, 'fro')


def mcr_factors(X: np.ndarray, n_components: int = 2, known_id: int = 0, svd_reduce: bool = False, init_guess: str = "rand", method: str = 'mvol') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ MCR-ALS factorization with one known spectrum, with the constraint that concentrations sum to one.
    
    We use NMF from sklearn to get an initial guess for the spectra, then replace the first spectrum with the known spectrum. 
    We then run MCR-ALS using pyMCR, starting from the adjusted guess spectra and constraining the first to remain fixed throughout.
    MCR-ALS is run with non-negativity constraints on both concentrations and spectra, 
    and a normalization constraint on the concentrations (sum to one per sample).

    arguments:
        X: data matrix (m samples x n wavelengths)
        n_components: total number of components (including known) (default 2)
        known_id: index of column the known spectrum in the data matrix X (default 0)
        svd_reduce: whether to perform SVD reduction before MCR-ALS (default False)
        
    returns:
        C: concentration matrix (m_samples x n_components)
        ST: spectra matrix (n_components x n_wavelengths)
        X_calc: reconstructed data matrix (m x n)"""

    if svd_reduce:
        u,s,t = np.linalg.svd(X)
        X = (u[:,:n_components] @ np.diag(s[:n_components]) @ t[:n_components,:]).copy()


    # set negatives to zero
    if np.min(X)<0:
        #print("The lowest value in X is ",np.min(X),", setting negatives to zero.")
        X[X<0]=0+1e-15

    if method == 'pymcr':
        if not PYMCR_AVAILABLE:
            raise ImportError("pymcr is not available. Please install it to use this method.")
         # set up MCR-ALS model
        mcrar = McrAR(max_iter=1000, st_regr=NNLS(), c_regr=OLS(),tol_increase=0.2,
                    c_constraints=[ConstraintNonneg(), ConstraintNorm()],err_fcn=mse)


        if init_guess == "nmf":
        # make initial guess
            m = NMF(n_components=n_components, init='nndsvda', solver='mu')
            c_nmf = m.fit_transform(X)
            S_guess = m.components_.copy()
            # find index of spectrum closest to known spectrum k
            k = X[known_id,:]
            dists = np.linalg.norm(S_guess - k, axis=1)
            closest_idx = np.argmin(dists)
            S_guess[closest_idx,:] = X[known_id,:]  # set known spectrum
            
            # re-order S_guess so that known spectrum is first
            S_guess[[0, closest_idx],:] = S_guess[[closest_idx, 0], :]

            
            
            S_guess[1,:] /= np.linalg.norm(m.components_)



        elif init_guess == "flat":
            S_guess = np.zeros((n_components,X.shape[1]))

            k = X[known_id,:]
            S_guess[0,:] = k
            S_guess[1,:] += 0.02
        
        elif init_guess == "projection":
            S_guess = np.zeros((n_components,X.shape[1]))
            k = X[known_id,:]
            # initialize: subtract projection onto k
            a0 = np.maximum(0, (X @ k) / (k @ k + 1e-15))
            resid = X - a0[:, None] * k
            u = np.maximum(0, resid.mean(axis=0))
            S_guess[0,:] = k
            S_guess[1,:] = u
        else:  # random
            S_guess = np.random.rand(n_components,X.shape[1])
            k = X[known_id,:]
            S_guess[0,:] = k  # set known spectrum


        mcrar.fit(X,  ST=S_guess ,st_fix=[0],verbose=False)
        X_calc = (mcrar.C_@mcrar.ST_)

        print("Error MCR-AR: {:.2e}".format(np.linalg.norm(X - X_calc,'fro')/np.linalg.norm(X,'fro')))

        return mcrar.C_, mcrar.ST_,X_calc
    elif method in ['mvol','FroALS','FroFPGM']:
        if not MCRNMF_AVAILABLE:
            raise ImportError("mcrnmf is not available. Please install it to use this method.")
        guess = mcrnmf.SNPA(rank=n_components,iter_max=1000)
        guess.fit(X.T)

        known_H = np.full(guess.H.shape, np.nan)
        known_W = np.full(guess.W.shape, np.nan)
        known_H[:,0] = [1,0]
        known_W[:,0] = X[known_id,:] 


        if guess.H[0,1] > guess.H[0,0]:
            # swap to ensure known spectrum is first
            guess.W[:,[0,1]] = guess.W[:,[1,0]]
            guess.H[[0,1],:] = guess.H[[1,0],:]


        if method == 'FroALS':
            model = mcrnmf.FroALS(rank=n_components, constraint_kind=4,iter_max=2000,tol=1e-4,order='ceu')
        elif method == 'FroFPGM':
            model = mcrnmf.FroFPGM(rank=n_components, constraint_kind=4,iter_max=2000,tol=1e-4)
        elif method == 'mvol':
            model = mcrnmf.MinVol(rank=n_components, constraint_kind=4,iter_max=2000,tol=1e-9,order='ceu')

        model.fit(X=X.T, Wi=guess.W, Hi=guess.H, known_H=known_H, known_W=known_W)
        return model.H.T, model.W.T, (model.W @ model.H).T
    else:
        raise ValueError("Unknown method: {}".format(method))


def rotation_simple(ST: np.ndarray, C,x) -> (np.ndarray,np.ndarray):
    T = np.array([[1, 0],
                  [x, 1-x]])
    
    C_rot = C@T
    # ensure non-negativity
    C_rot[C_rot < 0] = 0
    # apply closure by normalizing C_rot rows to sum to 1
    C_rot = C_rot/np.sum(C_rot,axis=1)[:,None]  
    X_orig = C @ ST
    ST_rot = np.linalg.lstsq(C_rot, X_orig,rcond=None)[0]

    # non-neg on ST
    ST_rot[ST_rot < 0] = 0
    return C_rot,ST_rot

def get_acceptable_solutions(s, ST_calc, C_calc,n=201,lb=-1,ub=1,threshold=1.001) -> (list,list):
    orig_err = calc_reconstruction_error(s,C_calc,ST_calc)
    res = []
    for x in np.linspace(lb,ub,n):
        C_rot,ST_rot = rotation_simple(ST_calc, C_calc, x)
        recon_err = calc_reconstruction_error(s,C_rot,ST_rot)
        rel_err = (recon_err/orig_err)
        #if np.linalg.norm(ST_rot[0,:] - ST[0,:]) > 0.1:
        known_err = np.linalg.norm(ST_rot[0,:] - ST_calc[0,:]) / np.linalg.norm(ST_calc[0,:])
        err = rel_err  + known_err # add an error penalty if the first spectrum deviates too much from the known

        res.append([x,err,C_rot,ST_rot])
    
    res_ST = [r[3] for r in res if r[1] <= threshold]
    res_C = [r[2] for r in res if r[1] <= threshold]
    return res_ST,res_C


