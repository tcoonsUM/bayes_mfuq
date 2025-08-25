import numpy as np
from numpy import linalg as la
import corr_utils
import torch
from scipy import stats

def cov2corr(c, return_sds=False):
    if torch.is_tensor(c):
        try:
            d = torch.diag(c)
        except ValueError:
            # scalar covariance
            # nan if incorrect value (nan, inf, 0), 1 otherwise
            return c / c
        stddev = torch.sqrt(d.real)
        c /= stddev[:, None]
        c /= stddev[None, :]
        if return_sds:
            return c, stddev
        else:
            return c
    else:
        try:
            d = np.diag(c)
        except ValueError:
            # scalar covariance
            # nan if incorrect value (nan, inf, 0), 1 otherwise
            return c / c
        stddev = np.sqrt(d.real)
        c /= stddev[:, None]
        c /= stddev[None, :]
        if return_sds:
            return c, stddev
        else:
            return c
    

def _W_entry(pilot_values_ii, pilot_values_jj):
    nqoi = pilot_values_ii.shape[1]
    npilot_samples = pilot_values_ii.shape[0]
    assert pilot_values_jj.shape[0] == npilot_samples
    means_ii = pilot_values_ii.mean(axis=0)
    means_jj = pilot_values_jj.mean(axis=0)
    centered_values_ii = pilot_values_ii - means_ii
    centered_values_jj = pilot_values_jj - means_jj
    centered_values_sq_ii = np.einsum(
        'nk,nl->nkl', centered_values_ii, centered_values_ii).reshape(
            npilot_samples, -1)
    centered_values_sq_jj = np.einsum(
        'nk,nl->nkl', centered_values_jj, centered_values_jj).reshape(
            npilot_samples, -1)
    centered_values_sq_ii_mean = centered_values_sq_ii.mean(axis=0)
    centered_values_sq_jj_mean = centered_values_sq_jj.mean(axis=0)
    centered_values_sq = np.einsum(
        'nk,nl->nkl',
        centered_values_sq_ii-centered_values_sq_ii_mean,
        centered_values_sq_jj-centered_values_sq_jj_mean).reshape(
        npilot_samples, -1)
    mc_cov = centered_values_sq.sum(axis=0).reshape(nqoi**2, nqoi**2)/(
        npilot_samples)
    return mc_cov


def _get_W_from_pilot(pilot_values, nmodels):
    # for one model 1 qoi this is the kurtosis
    nqoi = pilot_values.shape[1] // nmodels
    W = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[:, ii*nqoi:(ii+1)*nqoi]
        W[ii][ii] = _W_entry(pilot_values_ii, pilot_values_ii)
        for jj in range(ii+1, nmodels):
            pilot_values_jj = pilot_values[:, jj*nqoi:(jj+1)*nqoi]
            W[ii][jj] = _W_entry(pilot_values_ii, pilot_values_jj)
            W[jj][ii] = W[ii][jj].T
    return np.block(W)

def _V_entry(cov):
    V = np.kron(cov, cov)
    ones = np.ones((cov.shape[0], 1))
    V += (np.kron(np.kron(ones.T, cov), ones) *
          np.kron(np.kron(ones, cov), ones.T))
    return V


def _get_V_from_covariance(cov, nmodels):
    nqoi = cov.shape[0] // nmodels
    V = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        V[ii][ii] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, ii*nqoi:(ii+1)*nqoi])
        for jj in range(ii+1, nmodels):
            V[ii][jj] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi])
            V[jj][ii] = V[ii][jj].T
    return np.block(V)

def _covariance_of_variance_estimator(W, V, nsamples):
    return W/nsamples+V/(nsamples*(nsamples-1))

def high_fidelity_estimator_covariance(W, V, nhf_samples, nqoi):
    return _covariance_of_variance_estimator(
        W,
        V, nhf_samples)
    # return _covariance_of_variance_estimator(
    #     W[:nqoi**2, :nqoi**2],
    #     V[:nqoi**2, :nqoi**2], nhf_samples)

def vec_to_cov( cov_vec_samples, n_models ):    
    n_samps = cov_vec_samples.shape[0]
    n_beta = cov_vec_samples.shape[1]
    cov_mat_samps= np.zeros((n_models,n_models,n_samps))

    for i in range(n_samps):
        cov_vec = cov_vec_samples[i,:]
        cov_mat = np.ones((n_models,n_models))
        linds = np.tril_indices_from(cov_mat, k=0)
        uinds = np.triu_indices_from(cov_mat, k=0)
        cov_mat[uinds] = cov_vec
        for ind_i in range(n_beta):
            cov_mat[linds[0][ind_i],linds[1][ind_i]] = cov_mat[linds[1][ind_i],linds[0][ind_i]]
        cov_mat_samps[:,:,i] = cov_mat
        
    return cov_mat_samps

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


def pilotSamplesToBeta(pilot_samples, n_models, n_pilot, n_samps=100):
    """
    Full process for obtaining sample covariances, specifying MVN and converting resulting samples to beta samples. We will only return beta samples in the right shape.
    """

    c_hat = pilot_samples.cov()
    W = _get_W_from_pilot(pilot_samples.T, 1)
    V = _get_V_from_covariance(c_hat, 1)
    cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
    for ii in range(n_models):
        for jj in range(n_models):
            cov_cov[ii][jj] = cov_cov[jj][ii]

    # convert to vector of size M(M+1)/2
    u_inds = np.triu_indices_from(c_hat, k=0)
    c_hat_vec = c_hat[u_inds]
    n_beta = c_hat_vec.shape[0]
    cov_cov_vec = np.zeros((n_beta, n_beta))
    lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
    for i in range(len(u_inds[0])):
        cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]

    # sampling
    cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
    cov_samps = vec_to_cov(cov_vec_samps, n_models)
    for i in range(n_samps):
        cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])

    beta_samps = corr_utils.cov_to_beta_samples(cov_samps)
    n_params = beta_samps.shape[0]
    assert n_params==(n_models+1)*n_models/2, "beta size does not match n_models"

    return beta_samps.squeeze(-1)

def generate_cov_samples_wishart(y_pilot, n_samps, return_corr_sds=False, bias=True, seed=0):
    # y_pilot: n_models x n_pilot ( x n_x )
    
    if len(y_pilot.shape)==2:
        c_hat = np.cov(y_pilot)
        S = np.cov(y_pilot, bias=bias)
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
    
        # uncertainty samples from Wishart distribution
        cov_samps = stats.wishart.rvs(n_pilot, S, size=n_samps,random_state=seed).swapaxes(0,2)/(n_pilot)
        cov_samps_copy = np.copy(cov_samps)
        
        if return_corr_sds:
            corr_samps = np.zeros(cov_samps.shape)
            sds_samps = np.zeros((n_models, n_samps))
            for i in range(n_samps):
                corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps_copy[:,:,i], return_sds=True)
            return cov_samps, corr_samps, sds_samps
        else:
            return cov_samps_copy
        
    elif len(y_pilot.shape)==3:
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
        n_x = y_pilot.shape[2]
        cov_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        corr_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        sds_samps_all = np.zeros((n_models, n_samps, n_x))
        for ix in range(n_x):
            y_pil = y_pilot[:,:,ix]
            c_hat = np.cov(y_pil)
            S = np.cov(y_pil,bias=bias)
        
            cov_samps = stats.wishart.rvs(n_pilot, S, size=n_samps,random_state=seed).swapaxes(0,2)/(n_pilot)
            cov_samps_copy = np.copy(cov_samps)
            
            if return_corr_sds:
                corr_samps = np.zeros(cov_samps.shape)
                sds_samps = np.zeros((n_models, n_samps))
                cov_samps_all[:,:,:,ix] = cov_samps
                for i in range(n_samps):
                    corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps_copy[:,:,i], return_sds=True)
                corr_samps_all[:,:,:,ix] = corr_samps
                sds_samps_all[:,:,ix] = sds_samps
            else:
                cov_samps_all[:,:,:,ix] = cov_samps
        if return_corr_sds:
            return cov_samps_all, corr_samps_all, sds_samps_all
        else:
            return cov_samps_all
    else:
        print("y_pilot should be n_models x n_pilot x n_x, or n_models x n_pilot")
        return
    
def generate_cov_samples_wishart_S(S, n_pilot, n_samps, return_corr_sds=False, seed=0):
    
    n_models = S.shape[0]

    # uncertainty samples from Wishart distribution
    cov_samps = stats.wishart.rvs(n_pilot, S, size=n_samps,random_state=seed).swapaxes(0,2)/(n_pilot)
    cov_samps_return = np.copy(cov_samps)
    
    if return_corr_sds:
        corr_samps = np.zeros(cov_samps.shape)
        sds_samps = np.zeros((n_models, n_samps))
        for i in range(n_samps):
            corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
        return cov_samps_return, corr_samps, sds_samps
    else:
        return cov_samps_return
    
def generate_cov_samples_wishart_S_hierarchical(S_samps, n_pilot, return_corr_sds=False, seed=0):
    
    # S_samps should be n_models x n_models x n_samps
    n_samps = S_samps.shape[2]
    n_models = S_samps.shape[0]
    cov_samps = np.zeros((n_models, n_models, n_samps))

    # uncertainty samples from Wishart distribution
    for samp in range(n_samps):
        S = nearestPD(S_samps[:,:,samp])
        cov_samp = stats.wishart.rvs(n_pilot, S, size=1,random_state=seed+samp)/(n_pilot)
        cov_samps[:,:,samp] = cov_samp
    cov_samps_return = np.copy(cov_samps)
        
    if return_corr_sds:
        corr_samps = np.zeros(cov_samps.shape)
        sds_samps = np.zeros((n_models, n_samps))
        for i in range(n_samps):
            corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
        return cov_samps_return, corr_samps, sds_samps
    else:
        return cov_samps_return
        

def generate_cov_samples(y_pilot, n_samps, return_corr_sds=False):
    # y_pilot: n_models x n_pilot x n_x
    
    if len(y_pilot.shape)==2:
        c_hat = np.cov(y_pilot)
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
    
        # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
        W = _get_W_from_pilot(y_pilot.T, 1)
        V = _get_V_from_covariance(c_hat, 1)
        cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
        for ii in range(n_models):
            for jj in range(n_models):
                cov_cov[ii][jj] = cov_cov[jj][ii]
        u_inds = np.triu_indices_from(c_hat, k=0)
        c_hat_vec = c_hat[u_inds]
        n_beta = c_hat_vec.shape[0]
        cov_cov_vec = np.zeros((n_beta, n_beta))
        lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
        for i in range(n_beta):
            cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]
        cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
        cov_samps = vec_to_cov(cov_vec_samps, n_models)
        for i in range(n_samps):
            cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
        
        if return_corr_sds:
            corr_samps = np.zeros(cov_samps.shape)
            sds_samps = np.zeros((n_models, n_samps))
            for i in range(n_samps):
                corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
            return cov_samps, corr_samps, sds_samps
        else:
            return cov_samps
        
    elif len(y_pilot.shape)==3:
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
        n_x = y_pilot.shape[2]
        cov_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        corr_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        sds_samps_all = np.zeros((n_models, n_samps, n_x))
        for ix in range(n_x):
            y_pil = y_pilot[:,:,ix]
            c_hat = np.cov(y_pil)
        
            # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
            W = _get_W_from_pilot(y_pil.T, 1)
            V = _get_V_from_covariance(c_hat, 1)
            cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
            for ii in range(n_models):
                for jj in range(n_models):
                    cov_cov[ii][jj] = cov_cov[jj][ii]
            u_inds = np.triu_indices_from(c_hat, k=0)
            c_hat_vec = c_hat[u_inds]
            n_beta = c_hat_vec.shape[0]
            cov_cov_vec = np.zeros((n_beta, n_beta))
            lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
            for i in range(n_beta):
                cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]        
            # sampling
            cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
            cov_samps = vec_to_cov( cov_vec_samps, n_models )
            for i in range(n_samps):
                cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
            
            if return_corr_sds:
                corr_samps = np.zeros(cov_samps.shape)
                sds_samps = np.zeros((n_models, n_samps))
                cov_samps_all[:,:,:,ix] = cov_samps
                for i in range(n_samps):
                    corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
                corr_samps_all[:,:,:,ix] = corr_samps
                sds_samps_all[:,:,ix] = sds_samps
            else:
                cov_samps_all[:,:,:,ix] = cov_samps
        if return_corr_sds:
            return cov_samps_all, corr_samps_all, sds_samps_all
        else:
            return cov_samps_all
    else:
        print("y_pilot should be n_models x n_pilot x n_x, or n_models x n_pilot")
        return

def generate_cov_samples_projection(y_pilot, n_pilot, n_samps, return_corr_sds=False):
    # y_pilot: n_models x n_pilot x n_x
    
    if len(y_pilot.shape)==2:
        c_hat = np.cov(y_pilot)
        n_models = y_pilot.shape[0]
        n_pilot = n_pilot#y_pilot.shape[1]
    
        # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
        W = _get_W_from_pilot(y_pilot.T, 1)
        V = _get_V_from_covariance(c_hat, 1)
        cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
        for ii in range(n_models):
            for jj in range(n_models):
                cov_cov[ii][jj] = cov_cov[jj][ii]
        u_inds = np.triu_indices_from(c_hat, k=0)
        c_hat_vec = c_hat[u_inds]
        n_beta = c_hat_vec.shape[0]
        cov_cov_vec = np.zeros((n_beta, n_beta))
        lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
        for i in range(n_beta):
            cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]
        cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
        cov_samps = vec_to_cov(cov_vec_samps, n_models)
        for i in range(n_samps):
            cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
        
        if return_corr_sds:
            corr_samps = np.zeros(cov_samps.shape)
            sds_samps = np.zeros((n_models, n_samps))
            for i in range(n_samps):
                corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
            return cov_samps, corr_samps, sds_samps
        else:
            return cov_samps
        
    elif len(y_pilot.shape)==3:
        n_models = y_pilot.shape[0]
        n_pilot = n_pilot#y_pilot.shape[1]
        n_x = y_pilot.shape[2]
        cov_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        corr_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        sds_samps_all = np.zeros((n_models, n_samps, n_x))
        for ix in range(n_x):
            y_pil = y_pilot[:,:,ix]
            c_hat = np.cov(y_pil)
        
            # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
            W = _get_W_from_pilot(y_pil.T, 1)
            V = _get_V_from_covariance(c_hat, 1)
            cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
            for ii in range(n_models):
                for jj in range(n_models):
                    cov_cov[ii][jj] = cov_cov[jj][ii]
            u_inds = np.triu_indices_from(c_hat, k=0)
            c_hat_vec = c_hat[u_inds]
            n_beta = c_hat_vec.shape[0]
            cov_cov_vec = np.zeros((n_beta, n_beta))
            lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
            for i in range(n_beta):
                cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]        
            # sampling
            cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
            cov_samps = vec_to_cov( cov_vec_samps, n_models )
            for i in range(n_samps):
                cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
            
            if return_corr_sds:
                corr_samps = np.zeros(cov_samps.shape)
                sds_samps = np.zeros((n_models, n_samps))
                cov_samps_all[:,:,:,ix] = cov_samps
                for i in range(n_samps):
                    corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
                corr_samps_all[:,:,:,ix] = corr_samps
                sds_samps_all[:,:,ix] = sds_samps
            else:
                cov_samps_all[:,:,:,ix] = cov_samps
        if return_corr_sds:
            return cov_samps_all, corr_samps_all, sds_samps_all
        else:
            return cov_samps_all
    else:
        print("y_pilot should be n_models x n_pilot x n_x, or n_models x n_pilot")
        return

def generate_cov_samples2(y_pilot, n_samps):
    # y_pilot: n_models x n_pilot x n_x
    """Return covs, corrs and sds by default"""
    if len(y_pilot.shape)==2:
        c_hat = np.cov(y_pilot)
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
    
        # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
        W = _get_W_from_pilot(y_pilot.T, 1)
        V = _get_V_from_covariance(c_hat, 1)
        cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
        for ii in range(n_models):
            for jj in range(n_models):
                cov_cov[ii][jj] = cov_cov[jj][ii]
        u_inds = np.triu_indices_from(c_hat, k=0)
        c_hat_vec = c_hat[u_inds]
        n_beta = c_hat_vec.shape[0]
        cov_cov_vec = np.zeros((n_beta, n_beta))
        lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
        for i in range(n_beta):
            cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]
        cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
        cov_samps = vec_to_cov(cov_vec_samps, n_models)
        for i in range(n_samps):
            cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
        

        corr_samps = np.zeros(cov_samps.shape)
        sds_samps = np.zeros((n_models, n_samps))
        for i in range(n_samps):
            corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
        return cov_samps, corr_samps, sds_samps

        
    elif len(y_pilot.shape)==3:
        n_models = y_pilot.shape[0]
        n_pilot = y_pilot.shape[1]
        n_x = y_pilot.shape[2]
        cov_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        corr_samps_all = np.zeros((n_models, n_models, n_samps, n_x))
        sds_samps_all = np.zeros((n_models, n_samps, n_x))
        for ix in range(n_x):
            y_pil = y_pilot[:,:,ix]
            c_hat = np.cov(y_pil)
        
            # uncertainty samples from pilot samples, based on cov ~ MVN(c_hat, cov_cov)
            W = _get_W_from_pilot(y_pil.T, 1)
            V = _get_V_from_covariance(c_hat, 1)
            cov_cov = high_fidelity_estimator_covariance(W, V, n_pilot, 1)
            for ii in range(n_models):
                for jj in range(n_models):
                    cov_cov[ii][jj] = cov_cov[jj][ii]
            u_inds = np.triu_indices_from(c_hat, k=0)
            c_hat_vec = c_hat[u_inds]
            n_beta = c_hat_vec.shape[0]
            cov_cov_vec = np.zeros((n_beta, n_beta))
            lin_index  = np.ravel_multi_index(u_inds,(n_models,n_models))
            for i in range(n_beta):
                cov_cov_vec[i,:] = cov_cov[lin_index[i],lin_index]        
            # sampling
            cov_vec_samps = np.random.multivariate_normal(c_hat_vec, cov_cov_vec, size=n_samps)
            cov_samps = vec_to_cov( cov_vec_samps, n_models )
            for i in range(n_samps):
                cov_samps[:,:,i] = nearestPD(cov_samps[:,:,i])
            
            corr_samps = np.zeros(cov_samps.shape)
            sds_samps = np.zeros((n_models, n_samps))
            cov_samps_all[:,:,:,ix] = cov_samps
            for i in range(n_samps):
                corr_samps[:,:,i], sds_samps[:,i] = cov2corr(cov_samps[:,:,i], return_sds=True)
            corr_samps_all[:,:,:,ix] = corr_samps
            sds_samps_all[:,:,ix] = sds_samps

        return cov_samps_all, corr_samps_all, sds_samps_all
    else:
        print("y_pilot should be n_models x n_pilot x n_x, or n_models x n_pilot")
        return


def pilot_samples_to_gamma(pilot_samples, n_models, n_samps=100):
    # returns matrix log gamma's (n_gamma x n_samps x n_x_queries)
    cov_samps, corr_samps, sds_samps = generate_cov_samples(pilot_samples, n_samps, return_corr_sds=True)
    gamma_samps = corr_utils.corrs_to_gammas(corr_samps)
    return gamma_samps

def pilot_samples_to_gamma_wishart(pilot_samples, n_models, n_samps=100, return_corr_sds=False):
    # returns matrix log gamma's (n_gamma x n_samps x n_x_queries)
    cov_samps, corr_samps, sds_samps = generate_cov_samples_wishart(pilot_samples, n_samps, return_corr_sds=True)
    if len(pilot_samples.shape)==2:
        gamma_samps = corr_utils.corrs_to_gammas_1d(corr_samps)
    else:
        gamma_samps = corr_utils.corrs_to_gammas(corr_samps)
    if return_corr_sds:
        return gamma_samps, corr_samps, sds_samps
    else:
        return gamma_samps

def compute_oracle_cov(x, g_list):
    
    n_models = len(g_list)
    assert n_models==3
    oracle_xis = np.random.randn(10000)
    y_pilot_oracle = torch.stack((g_list[0](x, oracle_xis), 
                                    g_list[1](x, oracle_xis), 
                                    g_list[2](x, oracle_xis)))
    return np.cov(y_pilot_oracle)


def pilotSamplesToChol(pilot_samples, n_models, n_pilot, n_samps=100):
    # returns chol factors's (n_chol x n_samps x n_x_queries)
    cov_samps, corr_samps, sds_samps = generate_cov_samples(pilot_samples, n_samps, return_corr_sds=True)
    # convert to nearest PD
    corr_samps_psd = np.zeros(corr_samps.shape)
    nPD = []
    for ix in range(corr_samps.shape[-1]):
        for i in range(n_samps):
            if corr_utils.isPD(corr_samps[:,:,i,ix]):
                nPD.append(1)
            corr_samps_psd[:,:,i,ix] = nearestPD(corr_samps[:,:,i,ix])

    nPD_total = np.sum(np.array(nPD))
    chol_samps = corr_utils.corrs_to_chol(corr_samps_psd)
    # return chol_samps, nPD_total
    print("Number of PD matrices in corr samples: ", nPD_total)
    return chol_samps

def pilotSamplesToBetaCookbook(pilot_samples, n_models, n_pilot, n_samps=100):
    # returns beta factors's from cookbook (n_beta x n_samps x n_x_queries)
    cov_samps, corr_samps, sds_samps = generate_cov_samples(pilot_samples, n_samps, return_corr_sds=True)
    beta_samps = corr_utils.corrs_to_betas_cookbook(corr_samps)
    return beta_samps
