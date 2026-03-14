import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from lc import lc
from scipy.interpolate import interp1d

def Chebyhev_coefficients (df, t0, tE, degree):
    """
    Compute Chebyshev polynomial coefficients for a microlensing event amplification curve.
    
    Uses Chebyshev-Gauss sampling points and the discrete cosine transform approach to
    fit a polynomial approximation to the amplification data within a ±3tE window
    around the event peak.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the light curve data with columns:
            - 't' : time values
            - 'A' : amplification values
    t0 : float
        Time of peak amplification (Einstein time center).
    tE : float
        Einstein radius crossing time; defines the fitting window as [t0 - 3*tE, t0 + 3*tE].
    degree : int
        Degree (number of terms) of the Chebyshev polynomial approximation.
    
    Returns
    -------
    cheby_coefficients : list of float
        List of `degree` Chebyshev coefficients [c0, c1, ..., c_{n-1}] representing
        the polynomial approximation of the amplification curve over the event window.
    
    Notes
    -----
    - The amplification curve is interpolated using cubic splines before sampling.
    - Chebyshev nodes are used to minimize Runge's phenomenon and improve approximation accuracy.
    - The fitting domain is normalized from [xmin, xmax] to [-1, 1] internally.
    """
    n = degree
    event = (df['t'] > t0 - 3*tE) & (df['t'] < t0 + 3*tE)
    xmin = min(df['t'][event])
    xmax = max(df['t'][event])
    bma = 0.5 * (xmax - xmin)
    bpa = 0.5 * (xmax + xmin)
    interpoll = interp1d(df['t'],df['A'], kind='cubic')
    f = [interpoll(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
    fac = 2.0 / n
    cheby_coefficients = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n)]) for j in range(n)]
    
    return cheby_coefficients

def evaluate_chebyshev(df, t0, tE, cheby_coefficients):
    """
    Evaluate a Chebyshev polynomial approximation over a microlensing event window.
    
    Reconstructs the amplification curve from precomputed Chebyshev coefficients
    using Clenshaw's recurrence algorithm, which is numerically stable and efficient.
    Evaluation is performed at all observed time points within ±3tE of the event peak.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the light curve data with column:
            - 't' : time values
    t0 : float
        Time of peak amplification (Einstein time center).
    tE : float
        Einstein radius crossing time; defines the evaluation window as [t0 - 3*tE, t0 + 3*tE].
    cheby_coefficients : list of float
        Chebyshev coefficients as returned by `Chebyhev_coefficients()`.
    
    Returns
    -------
    Cheby_func : numpy.ndarray
        Array of evaluated amplification values at each time point within the event
        window, sorted in ascending time order.
    
    Notes
    -----
    - Time values are mapped from [xmin, xmax] to [-1, 1] before evaluation.
    - Clenshaw's recurrence is used for stable and efficient polynomial evaluation,
      avoiding explicit computation of individual Chebyshev basis polynomials.
    - Output ordering corresponds to np.sort(df['t'][event].values).
    """
    
    Cheby_func = []
    event = (df['t'] > t0 - 3*tE) & (df['t'] < t0 + 3*tE)
    xmin = min(df['t'][event])
    xmax = max(df['t'][event])
    
    for t_i in np.sort(df['t'][event].values):
        y = (2.0 * t_i - xmin - xmax) * (1.0 / (xmax - xmin))
        y2 = 2.0 * y
        (d, dd) = (cheby_coefficients[-1], 0)             # Special case first step for efficiency
        
        for cj in cheby_coefficients[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        Cheby_func.append(y * d - dd + 0.5 * cheby_coefficients[0])

    Cheby_func = np.asarray(Cheby_func)
    return Cheby_func