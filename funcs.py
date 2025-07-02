import numpy as np

# The speed of light
c = 299792.458 #km/s

def background(x, bkg):
    """
    Generates the background level of the spectrum.

    Parameters
    ----------
    lam : array
        The wavelength range over which the background should be generated.
    bkg : float
        The level of the background spectrum.

    Returns
    -------
    array
        The background level for each of the input wavelength values.
    """
    
    return x*0 + bkg


def gaussian(x, A, lam_rf, vel, sig, sig_resolution):
    """
    Produces a Gaussian curve with a background level.
    
    Parameters
    ----------
    x : float
        The wavelength range over which the Gaussian is generated.
    A : float
        The amplitude of the Gaussian.
    lam_rf : float
        The wavelength of the peak in its rest frame in Angstrom.
    vel : float
        The relative velocity of the source in km/s.
    sig : float
        The sigma of the Gaussian in km/s.
    sig_resolution : float
        The resolution of the detector in Angstrom.

    Returns
    -------
    array
        The Gaussian.
    """
    
    lam_obs = lam_rf * (1 + vel/c)
    sig_intr = sig / c * lam_obs # Intrinsic sigma
    sig_obs = np.sqrt(sig_intr**2 + sig_resolution**2)
    return A * np.exp(-0.5*(x - lam_obs)**2 / sig_obs**2)

def flux(A, sig, A_unc=0, sig_unc=0):
    """
    Find the flux of a Gaussian curve.

    Parameters
    ----------
    A : float
        The amplitude of the Gaussian.
    sig : float
        The sigma of the Gaussian.
    A_unc : float, default=0
        The unertainty in the amplitude of the Gaussian.
    sig_unc : float, default=0
        The unertainty in the sigma of the Gaussian.

    Returns
    -------
    f : float
        The flux of the Gaussian.
    f_unc : float
        The unertainty in the amplitude of the Gaussian.
    """
    
    f = A * np.abs(sig) * np.sqrt(2 * np.pi)
    f_unc = f * np.sqrt((A_unc/A)**2 + (sig_unc/sig)**2)
    return f, f_unc