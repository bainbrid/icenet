# Linear (correlation) and non-linear dependency tools
#
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba
import copy
import scipy.stats as stats

# Needed for tests only
import pandas as pd


def freedman_diaconis(x, mode="nbins"):
    """
    Freedman-Diaconis rule for a histogram bin width
    
    D. Freedman & P. Diaconis (1981)
    “On the histogram as a density estimator: L2 theory”.
    
    Args:
        x    : array of 1D data
        mode : return 'width' or 'nbins'
    """
    IQR  = stats.iqr(x, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = len(x)
    bw   = (2 * IQR) / N**(1.0/3.0)

    if mode == "width":
        y = bw
    else:
        y = int((np.max(x) - np.min(x)) / bw + 1)
    return y


def H_score(p):
    """
    Shannon Entropy (log_e ~ nats units)

    Args:
        p : probability vector
    Returns:
        entropy
    """
    # Make sure it is normalized
    p_ = p[p>0]/np.sum(p[p>0])

    return -np.sum(p_*np.log(p_))


def I_score(C, normalized=None, EPS=1E-15):
    """
    Mutual information score (log_e ~ nats units)

    Args:
        C : (X,Y) 2D-histogram array with event counts
        normalized : return normalized version (None, 'additive', 'multiplicative')
    
    Returns:
        mutual information score
    """
    def Pnorm(x):
        return np.maximum(x / np.sum(x[:]), EPS)

    nX, nY   = np.nonzero(C)
    Pi       = np.ravel(np.sum(C,axis=1))
    Pj       = np.ravel(np.sum(C,axis=0))
    
    # Joint 2D
    P_ij     = Pnorm(C[nX, nY])
    log_P_ij = np.log(P_ij)
    
    # Factorized 1D x 1D
    Pi_Pj = Pi.take(nX) * Pj.take(nY)
    Pi_Pj = Pi_Pj / np.maximum(np.sum(Pi) * np.sum(Pj), EPS)

    # Definition
    I = np.sum(P_ij * (log_P_ij - np.log(Pi_Pj) ))
    I = np.clip(I, 0.0, None)

    # Normalization
    if   normalized == None:
        return I
    elif normalized == 'additive':
        return 2*I/(H_score(Pi) + H_score(Pj))
    elif normalized == 'multiplicative':
        return I/np.sqrt(H_score(Pi) * H_score(Pj))
    else:
        raise Exception(f'I_score: Error with unknown normalization parameter "{normalized}"')


def mutual_information(x, y, weights = None, bins_x=None, bins_y=None, normalized=None, alpha=0.5):
    """
    Mutual information entropy (non-linear measure of dependency)
    between x and y variables
    Args:
        x         : array of values
        y         : array of values
        w         : weights (default None)
        bins_x    : x binning array  If None, then automatic.
        bins_y    : y binning array.
        normalized: normalize the mutual information (see I_score() function)
        alpha     : 0.5 (autobinning parameter)
    
    Returns:
        mutual information
    """
    
    if bins_x is None:
        bins_x = np.linspace(np.min(x), np.max(x), int(alpha*freedman_diaconis(x,'bins')))
    if bins_y is None:
        bins_y = np.linspace(np.min(y), np.max(y), int(alpha*freedman_diaconis(y,'bins')))

    XY = np.histogram2d(x=x, y=y, bins=[bins_x,bins_y], weights=weights)[0]
    mi = I_score(C=XY, normalized=normalized)

    return mi


def gaussian_mutual_information(rho):
    """
    Analytical 2D-Gaussian mutual information
    using a correlation coefficient rho.
    
    I(X1,X2) = H(X1) + H(X2) - H(X1,X2)
    Args:
        rho : correlation coefficient between (-1,1)
    Returns:
        mutual information
    """
    return -0.5*np.log(1-rho**2)


def test_gaussian():

    ## Create synthetic gaussian data
    N = int(1e6)

    for rho in np.linspace(-0.99, 0.99, 10):
    
        z1  = np.random.randn(N)
        z2  = np.random.randn(N)
        
        x1  = z1
        x2  = rho*z1 + np.sqrt(1-rho**2)*z2

        print(f'<rho = {rho:.3f}>')
        print(f'Gaussian exact MI = {gaussian_mutual_information(rho)}')
        print(f'Numerical      MI = {mutual_information(x=x1, y=x2)}')
        print('')


def test_data():

    # Read toy dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv")
    #target = df['Classification']

    df.drop(['Classification'], axis=1, inplace=True)

    # Build MI matrix for each pair of features
    D = df.shape[1]
    MI   = np.zeros((D,D))
    MI_A = np.zeros((D,D))
    MI_M = np.zeros((D,D))

    for i,col_i in enumerate(df):
        for j,col_j in enumerate(df):

            MI[i,j]   = mutual_information(x=df[col_i], y=df[col_j], normalized=None)
            MI_A[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='additive')
            MI_M[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='multiplicative')

    # Print out
    print('>> Raw Mutual Information')
    print(pd.DataFrame(MI,   columns = df.columns, index = df.columns))
    print('')
    print('>> Additively Normalized Mutual Information')
    print(pd.DataFrame(MI_A, columns = df.columns, index = df.columns))
    print('')
    print('>> Multiplicatively Normalized Mutual Information')
    print(pd.DataFrame(MI_M, columns = df.columns, index = df.columns))


# Run tests
test_gaussian()
test_data()
