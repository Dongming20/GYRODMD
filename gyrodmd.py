import scipy.linalg
import numpy as np
import cmath,math
import warnings


class GYRODMD:
    def __init__(self, rank=0, tls_rank=None, DMD_order=1):
        """
        Initialize the GYRODMD object.

        Parameters:
        rank (int, optional): Number of modes to retain in truncation. Default is 0 (no truncation).
        tls_rank (int, optional): Number of Singular values to compute Total Least Square (TLS) DMD. Default is None, meaning standard DMD is used.
        DMD_order (int, optional): Default is 1, greater than 1 means higher order DMD. 
        """
        self.rank = rank
        self.tls_rank = tls_rank
        self.order = DMD_order
        self.singularvalue = None
        self.eigenvalues = None
        self.omega = None
        self.mode = None
        

    def fit(self, data=None, dt=None):
        """
        Fit the GYRODMD model.

        Parameters:
        data (np.ndarray)
        dt (float): Time step

        Returns:
        self: Fitted model.
        """
        if data is None:
            print("There is no input data, data is None.")
            exit()

        [nk, nt] = data.shape

        nh = nk*self.order # the height of Y
        nl = math.floor(nt/self.order) # the length of Y
        
        # DMD
        # construct the data matrices        
        X = np.zeros((nh, nl-1), dtype = 'complex_')
        Y = np.zeros((nh, nl-1), dtype = 'complex_')
        
        
        for l in range(nl-1):
            for s in range(self.order):
                X[s*nk:((s+1))*nk, l] = data[:, l*self.order+s]
                Y[s*nk:((s+1))*nk, l] = data[:, l*self.order+s+1]

        # compute the optimal truncation of singular values
        if self.tls_rank is None:
            X_hat = X
            Y_hat = Y

        else:
            """
            Total Least Squares (TLS) DMD
            
            References:
            Hemati, Maziar S. and and Rowley, Clarence W. and Deem, Eric A. and Cattafesta, Louis N.,
            De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets
            Theor. Comput. Fluid Dyn. (2017) 31:349-368
            https://link.springer.com/article/10.1007/s00162-017-0432-2
            """
            # Stack X and Y to form the augmented data matrix
            Z = np.vstack((X, Y))

            # Perform Singular Value Decomposition (SVD)
            _, Z_Sigma, Z_Vh = np.linalg.svd(Z, full_matrices=False)

            if self.tls_rank==0:
                # compute the optimal truncation of TLS singular values
                self.tls_rank = svht(Z_Sigma, 2*nh, nl-1)
                Z_Vh = Z_Vh[:self.tls_rank, :]

            else:
                # Truncate if rank is specified
                Z_Vh = Z_Vh[:self.tls_rank, :]

            X_hat = X @ Z_Vh.conj().T
            Y_hat = Y @ Z_Vh.conj().T

        # Perform Singular Value Decomposition (SVD) on X_hat
        U, S, Vh = scipy.linalg.svd(X_hat, full_matrices=False)

        if self.rank > 0:
            # Truncate if rank is specified
            U = U[:, :self.rank]
            self.singularvalue = S[:self.rank]
            Vh = Vh[:self.rank, :]

        elif self.rank==0:
            # compute the optimal truncation of singular values
            self.rank = svht(S, nh, nl-1)
            
            U = U[:, :self.rank]
            self.singularvalue = S[:self.rank]
            Vh = Vh[:self.rank, :]

        # Compute A_tilde
        A_tilde = np.dot(np.dot(U.conj().T, Y_hat), np.dot(Vh.conj().T, np.diag(1/self.singularvalue)))

        # Compute eigenvalues and eigenvectors of A_tilde
        self.eigenvalues, W = scipy.linalg.eig(A_tilde)

        # Compute DMD modes
        self.mode = np.dot(np.dot(np.dot(Y_hat, Vh.conj().T), np.diag(1/self.singularvalue)), W)
        # self.mode = np.dot(U, W)
        
        # Compute DMD mode amplitudes
        b = np.dot(np.linalg.pinv(self.mode), X_hat[:, 0])

        if dt is None:
            self.omega = self.eigenvalues
        else:  
            self.omega = np.log(self.eigenvalues)/dt # continuous-time eigenvalues

        return self


def svht(sigma_svd, rows, cols):
    """
    Singular Value Hard Threshold.

    :param sigma_svd: Singual values computed by SVD
    :type sigma_svd: np.ndarray
    :param rows: Number of rows of original data matrix.
    :type rows: int
    :param cols: Number of columns of original data matrix.
    :type cols: int
    :return: Computed rank.
    :rtype: int

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    https://ieeexplore.ieee.org/document/6846297
    """
    beta = np.divide(*sorted((rows, cols)))
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = np.median(sigma_svd) * omega
    rank = np.sum(sigma_svd > tau)

    if rank == 0:
        warnings.warn(
            "SVD optimal rank is 0. The largest singular values are "
            "indistinguishable from noise. Setting rank truncation to 1.",
            RuntimeWarning,
        )
        rank = 1

    return rank