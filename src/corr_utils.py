import numpy as np
import arviz as az
import parametrization_cookbook.numpy as pc
import torch

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
        
def covs2corrs(cov_matrices):
    """
    Convert a set of covariance matrices to correlation matrices.

    Parameters:
    cov_matrices (np.ndarray): A 3D numpy array of shape (M, M, n_samps), 
                               where each slice along the third axis is an M x M covariance matrix.

    Returns:
    np.ndarray: A 3D numpy array of shape (M, M, n_samps), where each slice along the 
                third axis is the corresponding M x M correlation matrix.
    """
    # Verify the input dimensions
    if cov_matrices.ndim != 3 or cov_matrices.shape[0] != cov_matrices.shape[1]:
        raise ValueError("Input must be an array of shape (M, M, n_samps)")

    M, _, n_samps = cov_matrices.shape
    corr_matrices = np.zeros_like(cov_matrices)

    for i in range(n_samps):
        cov_matrix = cov_matrices[:, :, i]
        stddevs = np.sqrt(np.diag(cov_matrix))

        # Avoid division by zero in case of zero standard deviations
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = cov_matrix / np.outer(stddevs, stddevs)
            corr_matrix[~np.isfinite(corr_matrix)] = 0  # Set NaNs and infs to zero

        np.fill_diagonal(corr_matrix, 1)  # Ensure the diagonal is exactly 1
        corr_matrices[:, :, i] = corr_matrix

    return corr_matrices

# %% Checks for PD

from numpy import linalg as la

# def isPD(B):
#     """Returns true when input is positive-definite, via Cholesky"""
#     try:
#         _ = la.cholesky(B)
#         return True
#     except la.LinAlgError:
#         return False
    
def isPD(B):
    eigs = np.linalg.eigh(B)[0]
    if eigs.min()<0.:
        return False
    else:
        return True

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



# %% Thomas: Trace to prior, posterior covariance
def trace_to_posterior_samples(trace_objs, corr_key, n_d, n_post_samples, n_models):
    """
    Converts an Arviz inference object into numpy arrays of samples.

    This function extracts the posterior samples for correlation matrices and standard deviations
    from the provided Arviz trace objects. It then constructs the correlation matrices and 
    computes the covariance matrices.

    @param trace_objs List of Arviz inference data objects containing the trace results.
    @param corr_key The key used to extract the correlation data from the trace objects.
    @param n_d Number of different datasets or conditions.
    @param n_post_samples Number of posterior samples to extract for each dataset.
    @param n_models Number of models or variables in the correlation matrices.
    
    @return Tuple containing:
        - corr_post_samples (numpy.ndarray): Array of shape (n_models, n_models, n_post_samples, n_d)
          containing the posterior samples for the correlation matrices.
        - sds_posterior_samples (numpy.ndarray): Array of shape (n_models, n_post_samples, n_d)
          containing the posterior samples for the standard deviations.
        - cov_post_samples (numpy.ndarray): Array of shape (n_models, n_models, n_post_samples, n_d)
          containing the posterior samples for the covariance matrices.
    """
    corr_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))
    corr_posteriors = []
    sds_posteriors = []
    
    for i in range(n_d):
        corr_posterior_vec = az.extract(trace_objs[i], var_names=corr_key, num_samples=n_post_samples).data

        sds_posterior = az.extract(trace_objs[i], var_names='sds', num_samples=n_post_samples).data

        corr_posteriors.append(corr_posterior_vec)
        sds_posteriors.append(sds_posterior)
    
    sds_post_samples = np.stack(sds_posteriors, axis=2)

    # Convert to matrix:
    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_posteriors[i][:, j]
            corr_mat = np.eye(n_models)
            linds = np.tril_indices_from(corr_mat, k=-1)
            uinds = np.triu_indices_from(corr_mat, k=1)
            corr_mat[linds] = corr
            corr_mat[uinds] = corr
            corr_post_samples[:, :, j, i] = corr_mat
    
    # Create covariance posterior samples
    cov_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))
    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_post_samples[:, :, j, i]
            sds = sds_post_samples[:, j, i]
            cov_post_samples[:, :, j, i] = np.diag(sds) @ corr @ np.diag(sds)
    
    return corr_post_samples, sds_post_samples, cov_post_samples

def trace_to_prior_samples(trace_objs, corr_key, n_d, n_post_samples, n_models):
    """
    Converts a list of dictionaries composed of Arviz inference objects i.e. {'corr': prior_corrs, 'sds': prior_sds} into prior covariance samples.
    """
    corr_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))
    corr_posteriors = []
    sds_posteriors = []

    for i in range(n_d):
        corr_obj = trace_objs[i]['corr']
        sds_obj = trace_objs[i]['sds']

        corr_posterior_vec = az.extract(corr_obj, var_names=corr_key, num_samples=n_post_samples).data
        corr_posteriors.append(corr_posterior_vec)

        sds_posterior_vec = az.extract(sds_obj, var_names='sds', num_samples=n_post_samples).data

        sds_posteriors.append(sds_posterior_vec)

    sds_post_samples = np.stack(sds_posteriors, axis=2)

    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_posteriors[i][:, j]
            corr_mat = np.eye(n_models)
            linds = np.tril_indices_from(corr_mat, k=-1)
            uinds = np.triu_indices_from(corr_mat, k=1)
            corr_mat[linds] = corr
            corr_mat[uinds] = corr
            corr_post_samples[:, :, j, i] = corr_mat

    # convert to covariance.
    cov_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))

    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_post_samples[:, :, j, i]
            sds = sds_post_samples[:, j, i]
            cov_post_samples[:, :, j, i] = np.diag(sds) @ corr @ np.diag(sds)

    return corr_post_samples, sds_post_samples, cov_post_samples


def trace_to_samples_corr_only(trace_objs, corr_key, n_d, n_post_samples, n_models):
    
    corr_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))
    corr_posteriors = []
    
    for i in range(n_d):
        corr_posterior_vec = az.extract(trace_objs[i], var_names=corr_key, num_samples=n_post_samples).data
        corr_posteriors.append(corr_posterior_vec)
        
    # Convert to matrix:
    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_posteriors[i][:, j]
            corr_mat = np.eye(n_models)
            linds = np.tril_indices_from(corr_mat, k=-1)
            uinds = np.triu_indices_from(corr_mat, k=1)
            corr_mat[linds] = corr
            corr_mat[uinds] = corr
            corr_post_samples[:, :, j, i] = corr_mat
    
    return corr_post_samples

def trace_to_samples_sds_only(trace_objs, n_d, n_post_samples, n_models):
    
    sds_posteriors = []
    
    for i in range(n_d):
        sds_posterior = az.extract(trace_objs[i], var_names='sds', num_samples=n_post_samples).data
        sds_posteriors.append(sds_posterior)
    
    sds_post_samples = np.stack(sds_posteriors, axis=2)
    
    return sds_post_samples

def sds_corrs_to_covs(corr_post_samples, sds_post_samples, n_d, n_post_samples, n_models):
    cov_post_samples = np.zeros((n_models, n_models, n_post_samples, n_d))
    for i in range(n_d):
        for j in range(n_post_samples):
            corr = corr_post_samples[:, :, j, i]
            sds = sds_post_samples[:, j, i]
            cov_post_samples[:, :, j, i] = np.diag(sds) @ corr @ np.diag(sds)
            
    return cov_post_samples

# %% Map covariance matrices to covariance vec samples i.e. concatenate rows of lower triangle and diagonal in first dimension.

def cov_to_vec(cov_samples):
    """
    Convert covariance matrices to covariance vector samples.
    """
    n_models = cov_samples.shape[0]
    n_draws = cov_samples.shape[2]
    n_d = cov_samples.shape[3]

    cov_vec_samples = np.zeros((int(n_models*(n_models+1)/2), n_draws, n_d))
    for i in range(n_d):
        for j in range(n_draws):
            cov = cov_samples[:, :, j, i]
            linds = np.tril_indices_from(cov, k=-1)
            cov_vec = np.concatenate((cov[linds], np.diag(cov)))
            cov_vec_samples[:, j, i] = cov_vec
    
    return cov_vec_samples


# %% Archakov and Hansen (2021) - log transform mappings

from scipy.linalg import expm, norm

def GFT_forward_mapping(C):
    """
    Take the matrix logarithm. If C has the following eigendecomposition:
    Q \Lambda Q', then log(C) = Q log(\Lambda) Q'. 
    Extract lower triangle elements of log(C) and return.
    """
    n = C.shape[0]

    # Get the eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # there are some numerically negative and very small eigenvals
    #eigvals[np.where(np.logical_and(eigvals<1e-15, eigvals>-1e-14))]=1e-12

    # Take the log of the eigenvalues
    log_eigvals = np.log(eigvals)

    # Reconstruct the matrix
    log_C = eigvecs @ np.diag(log_eigvals) @ eigvecs.T


    # Extract the lower triangle
    uinds = np.triu_indices(n, k=1)
    log_C_vec = log_C[uinds]

    return log_C_vec

def GFT_forward_mapping_with_correction(C):
    """
    Take the matrix logarithm. If C has the following eigendecomposition:
    Q \Lambda Q', then log(C) = Q log(\Lambda) Q'. 
    Extract lower triangle elements of log(C) and return.
    """
    n = C.shape[0]

    # Get the eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # there are some numerically negative and very small eigenvals
    eigvals[np.where(np.logical_and(eigvals<1e-15, eigvals>-1e-14))]=1e-14

    # Take the log of the eigenvalues
    log_eigvals = np.log(eigvals)

    # Reconstruct the matrix
    log_C = eigvecs @ np.diag(log_eigvals) @ eigvecs.T


    # Extract the lower triangle
    uinds = np.triu_indices(n, k=1)
    log_C_vec = log_C[uinds]

    return log_C_vec

# inverse mapping code is taken from the web appendix to the paper. forward map is my crude implementation of matrix log definition for symmetric matrix. 
# I found 1e-7 is pretty good and fast enough for our purposes.
def GFT_inverse_mapping(gamma_in, tol_value=1e-7):
    C = []
    iter_number = -1
    try:
        # Check if input is of proper format: gamma is of suitable length # and tolerance value belongs to a proper interval
        n = 0.5*(1+np.sqrt(1+8*len(gamma_in)))
        if not all([gamma_in.ndim == 1, n.is_integer(), 1e-14 <= tol_value <= 1e-4]):
            raise ValueError
        # Place elements from gamma into off-diagonal parts
        # and put zeros on the main diagonal of nxn symmetric matrix A
        n = int(n)
        A = np.zeros(shape=(n,n))
        A[np.triu_indices(n,1)] = gamma_in
        A = A + A.T
        # Read properties of the input matrix
        diag_vec = np.diag(A)
        diag_ind = np.diag_indices_from(A)
        # Iterative algorithm to get the proper diagonal vector
        dist = np.sqrt(n)
        while dist > np.sqrt(n)*tol_value:
            diag_delta = np.log(np.diag(expm(A)))
            diag_vec = diag_vec - diag_delta
            A[diag_ind] = diag_vec
            dist = norm(diag_delta)
            iter_number += 1
        # Get a unique reciprocal correlation matrix
        C = expm(A)
        np.fill_diagonal(C, 1)
    except ValueError:
        print("Error: input is of wrong format")

    return C, iter_number

#%% convert above to usable algorithm for multiple samples

def corrs_to_gammas(corr_samps):
    # take m x m x n_draws x n_x corr_samps matrix
    m = corr_samps.shape[0]
    n_draws = corr_samps.shape[2]
    n_x = corr_samps.shape[3]
    
    n_elems = int(m*(m-1)/2)
    
    if torch.is_tensor(corr_samps):
        corr_samps = corr_samps.detach().numpy()
    
    gamma_samps = np.zeros([n_elems, n_draws, n_x])
    for ix in range(n_x):
        for i in range(n_draws):
            gamma_samps[:,i,ix] = GFT_forward_mapping(corr_samps[:,:,i,ix])
        
    return gamma_samps

def corrs_to_gammas_1d(corr_samps, correction=False):
    # take m x m x n_draws corr_samps matrix
    m = corr_samps.shape[0]
    n_draws = corr_samps.shape[2]
    
    n_elems = int(m*(m-1)/2)
    
    if torch.is_tensor(corr_samps):
        corr_samps = corr_samps.detach().numpy()
    
    gamma_samps = np.zeros([n_elems, n_draws])
    for i in range(n_draws):
        if correction:
            gamma_samps[:,i] = GFT_forward_mapping_with_correction(corr_samps[:,:,i])
        else:
            gamma_samps[:,i] = GFT_forward_mapping((corr_samps[:,:,i]+corr_samps[:,:,i].T)/2)
        
    return gamma_samps

def gammas_to_corrs_1d(gamma_samps, n_models=None):
    # take n_elems x n_draws gamma_samps matrix
    n_elems = gamma_samps.shape[0]
    n_draws = gamma_samps.shape[1]
    
    if n_models is None:
        n_models = int((1+np.sqrt(1+8*n_elems))/2)
    
    corr_samps = np.zeros([n_models, n_models, n_draws])
    
    for i in range(n_draws):
        corr_samps[:,:,i] = GFT_inverse_mapping(gamma_samps[:,i])[0]
        
    return corr_samps
