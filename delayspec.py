import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light

def get_peak_size(tau, S, delays, window_size=100.0, nsig=3.0, nsmooth=0):

    ntau, nprod, ntime = S.shape

    ndelay = delays.shape[1]

    Ppos = np.zeros((ndelay, nprod, ntime), dtype=np.float64)
    Pneg = np.zeros((ndelay, nprod, ntime), dtype=np.float64)

    for tt in range(ntime):

        for pp in range(nprod):

            this_S = S[:, pp, tt]

            tau_start, tau_end = peak_finder(tau, this_S,
                                             window_size=window_size, nsig=nsig, nsmooth=nsmooth)

            for ii, dd in enumerate(delays[pp, :]):

                tau0 = np.abs(dd)

                igroup = list(np.flatnonzero((tau0 >= tau_start) & (tau0 <= tau_end)))

                keep_flag = np.zeros(tau.size, dtype=np.bool)
                for ig in igroup:
                    keep_flag |= (tau >= tau_start[ig]) & (tau <= tau_end[ig])

                if np.any(keep_flag):
                    Ppos[ii, pp, tt] = np.max(np.abs(this_S[keep_flag]))


                igroup = list(np.flatnonzero((-tau0 >= tau_start) & (-tau0 <= tau_end)))

                keep_flag = np.zeros(tau.size, dtype=np.bool)
                for ig in igroup:
                    keep_flag |= (tau >= tau_start[ig]) & (tau <= tau_end[ig])

                if np.any(keep_flag):
                    Pneg[ii, pp, tt] = np.max(np.abs(this_S[keep_flag]))

    return Ppos, Pneg

def peak_finder(tau, S, window_size=100.0, nsig=5.0, nsmooth=5):

    from scipy.signal import medfilt

    # Sort spectrum
    isort = np.argsort(tau)

    x = tau[isort]
    y = np.abs(S[isort])

    is_nonzero = y > 0.0
    if not np.any(is_nonzero):
        return [], []
    ind_nonzero = np.flatnonzero(is_nonzero)
    a = np.argmin(ind_nonzero)
    b = np.argmax(ind_nonzero)

    # Look at fluctuations about median
    nwindow = int(window_size / np.median(np.diff(x)))
    nwindow += ~(nwindow % 2)

    ymed = np.zeros_like(y)
    ymed[a:b] = medfilt(y[a:b], nwindow)

    dy = (y - ymed) * is_nonzero

    sig = 1.4826 * np.median(np.abs(dy[is_nonzero]))

    # Define peak as any fluctuation greater than nsig times the local median
    peak_flag = dy > (nsig * sig)

    if not np.any(peak_flag):
        ValueError("No peaks found.")

    # Smooth the mean
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    if nsmooth > 1:
        smooth_peak_flag = running_mean(peak_flag, nsmooth) > 0
        aa = nsmooth/2
        bb = aa + smooth_peak_flag.size
        peak_flag[aa:bb] |= smooth_peak_flag

    # Place peaks into groups
    peak_ind = np.arange(peak_flag.size)[peak_flag]

    tmp = np.flatnonzero(np.diff(peak_ind) > 1)

    tau_start, tau_end = None, None
    if tmp.size > 0:

        group_start = np.concatenate((peak_ind[0, None], peak_ind[tmp+1]))
        group_end = np.concatenate((peak_ind[tmp], peak_ind[-1, None]))

        tau_start = x[group_start]
        tau_end = x[group_end]

    return tau_start, tau_end

def reconstruct_spectrum(tau, S, B, delays, pos=False, neg=False,
                                         window_size=100.0, nsig=5.0, nfreq=1024, nsmooth=5):

    # Parse input
    if not hasattr(delays, '__iter__'):
        delays = [delays]

    if not pos and not neg:
        pos = True
        neg = True

    # Find peaks
    tau_start, tau_end = peak_finder(tau, S, window_size=window_size, nsig=nsig, nsmooth=nsmooth)

    # Create peak info
    peak_info = {}
    keys = ['freq', 'amp', 'width']
    for kk in keys:
        peak_info[kk] = []

    # Loop through the requested delays
    ndelay = len(delays)

    keep_flag = np.zeros(tau.size, dtype=np.bool)
    for tau0 in delays:

        # tau0 = np.abs(tau0)

        igroup = []
        if pos:
            igroup += list(np.flatnonzero((tau0 >= tau_start) & (tau0 <= tau_end)))

        if neg:
            igroup += list(np.flatnonzero((-tau0 >= tau_start) & (-tau0 <= tau_end)))

        for ig in igroup:
            this_flag = (tau >= tau_start[ig]) & (tau <= tau_end[ig])
            keep_flag |= this_flag

            if np.any(this_flag):
                this_index = np.flatnonzero(this_flag)
                imax = np.argmax(np.abs(S[this_index]))
                peak_info['freq'].append(tau[this_index[imax]])
                peak_info['amp'].append(S[this_index[imax]])
                peak_info['width'].append(tau[this_index].max() - tau[this_index].min())

    # Take inverse fft of delay spectrum and clean beam
    Speak = S.copy()
    Speak[~keep_flag] = 0.0

    Vpeak = np.fft.ifft(Speak) * Speak.size
    VB = np.fft.ifft(B) * B.size

    # Restrict focus to our frequency range
    Vpeak = Vpeak[0:nfreq]
    VB = VB[0:nfreq]

    # Normalize clean beam in visibility space
    VB *= invert_no_zero(np.max(np.abs(VB)))

    # # Correct reconstructed visibility for clean beam
    Vpeak *= invert_no_zero(VB)

    return Vpeak, keep_flag, peak_info

def get_delay(baseline, ndelay=3, focus=5.0, is_intra=True):

    # Hardcoded parameters
    bnc = 2.0

    # Parse inputs
    dist = np.atleast_1d(baseline)

    nprod = dist.size
    npath = ndelay if is_intra else 3

    delay = np.zeros((nprod, npath), dtype=np.float64)

    # We have different expected delays for
    # intercylinder and intracylinder baselines
    if is_intra:

        for bb in range(ndelay):
            delay[:, bb] = bnc * np.sqrt((bb * focus)**2 + (dist / bnc)**2) / (speed_of_light * 1e-9)

    else:

        delay[:, 0] = dist / (speed_of_light * 1e-9)
        delay[:, 1] = (1.80*dist - 28.0) / (speed_of_light * 1e-9)
        delay[:, 2] = (0.80*dist + 14.0) / (speed_of_light * 1e-9)

    if nprod == 1:
        delay = delay[0]

    return delay

def myicmap(ind,nant):
    nvis=nant*(nant+1)/2
    nn=nvis-ind
    myrow=nant-(-1+np.sqrt(1+8.0*nn))/2
    myrow=(np.floor(myrow+1e-10))

    nconj=nant-myrow
    nsofar=nvis-nconj*(nconj+1)/2

    mycol=ind-nsofar+myrow

    ant1=myrow.astype('int')
    ant2=mycol.astype('int')
    return ant1,ant2

def invert_no_zero(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x == 0, 0.0, 1.0 / x)

def delay_spectrum(fmeas, vis_arr, weight_arr, h=0.1, res=10,
                   niter=None, verbose=False, dual_peak=True,
                   mean_subtract=True, check_delta=True,
                   window_type='zeropad'):
    """Calculate the delay spectrum using a CLEAN-like algorithm.

    For more information, see:

        D. H. Roberts, J. Lehar, and J. W. Dreher.
        “Time Series Analysis with Clean - Part One - Derivation of a Spectrum”.
        In: AJ 93 (Apr. 1987), p. 968. doi: 10.1086/114383

    Parameters
    ----------
    fmeas : np.ndarray[nfreq,]
        Frequency in MHz.
    vis_arr : np.ndarray[nfreq, nprod, ntime]
        Visibility matrix.  Delay spectrum is calculated for each product and time.
    weight_arr : np.ndarray[nfreq, nprod, ntime]
        Weights for the delay transform.  This should be 0's and 1's indicating
        good and bad frequency bins.  More complicated weighting schemes can also be used.
    h : float
        At each iteration, clean this fraction of the largest peak in the delay spectrum.
    res : float
        Increase the resolution of the delay spectrum by this factor:
    niter : int
        Number of iterations in the CLEAN algorithm.
    verbose : bool
        Print messages regarding the total number of iterations that were used.
    dual_peak : bool
        Assume the delay spectrum is symmetric during cleaning, i.e., S(-tau) = S(tau).
    mean_subtract : bool
        Subtract the mean over frequency prior to calculating delay spectrum.
    check_delta : bool
        Stop iterating if the sum squared residuals starts to increase.
    window_type : str
        One of 'zeropad', 'truncate', 'interpolate', or 'average'.  Uses different
        methods for increasing the resolution of the delay spectrum.

    Returns
    -------
    tau : np.ndarray[ntau,]
        Delay in nano-seconds.
    Sdelay : np.ndarray[ntau, nprod, ntime]
        Delay spectrum.
    tau_b : np.ndarray[ntau * res * 8,]
        Delay in nano-seconds for the clean beam.
    mpar : np.ndarray[nparam, nprod, ntime]
        Parameters of the clean beam.  The clean beam can be generated via
        `model_gaussian_with_phase_gradient(tau_b, *mpar[:, iprod, itime])`.
    """
    nfmeas, nprod, ntime = vis_arr.shape

    # Determine number of iterations
    if niter is None:
        niter = int(fmeas.size * (0.1 / h))

    # Define the nominal set of frequencies
    nfreq = 1024
    freq = np.sort(np.linspace(800.0, 400.0, nfreq, endpoint=False))
    df = np.median(np.diff(freq))

    # Match nominal frequencies to input frequencies
    imatch = np.zeros(nfreq, dtype=np.int) - 1
    for ii, ff in enumerate(freq):
        jj = np.argmin(np.abs(ff - fmeas))
        if np.abs(fmeas[jj] - ff) < (0.1 * df):
            imatch[ii] = jj

    # Create sampling function
    grid_index = imatch >= 0
    meas_index = imatch[grid_index]

    # Initialize outputs
    tau, Sdelay, tau_b, mpar = None, None, None, None

    # Loop over time and product
    for tt in range(ntime):

        for bb in range(nprod):

            vis = np.zeros((nfreq, ), dtype=np.complex128)
            smp = np.zeros((nfreq, ), dtype=np.float64)

            wbb = bb % weight_arr.shape[1]
            wtt = tt % weight_arr.shape[2]

            vis[grid_index] = vis_arr[meas_index, bb, tt]
            smp[grid_index] = weight_arr[meas_index, wbb, wtt]# > 0

            # Make sure we have a non-negligible sampling of the
            # full frequency range
            if np.sum(smp > 0.0) > 0.2*nfreq:

                # Subtract mean
                if mean_subtract:
                    vis -= np.sum(vis * smp) * invert_no_zero(np.sum(smp))

                # Perform clean
                tau, S, fval, acomp, tau_b, popt = func_clean(freq, vis, smp, niter, h=h,
                                                              res=res, output_all=False,
                                                              dual_peak=dual_peak,
                                                              check_delta=check_delta,
                                                              window_type=window_type,
                                                              verbose=verbose)

                # Create array to hold results
                if Sdelay is None:
                    ntau = tau.size
                    Sdelay = np.zeros((ntau, nprod, ntime), dtype=np.complex128)
                    mpar = np.zeros((popt.size, nprod, ntime), dtype=popt.dtype)

                # Save to array
                Sdelay[:, bb, tt] = S
                mpar[:, bb, tt] = popt

    # Check for case where no succeful runs
    if Sdelay is None:
        return tau, Sdelay, tau_b, mpar

    # Shift to increasing tau
    tau = np.fft.fftshift(tau)
    Sdelay = np.fft.fftshift(Sdelay, axes=0)

    # Return delay spectrum
    return tau, Sdelay, tau_b, mpar

def func_clean(freq, vis, smp, niter, h=0.1, res=1, output_all=False,
                        verbose=True, check_delta=True, dual_peak=True,
                        window_type='zeropad'):

    from scipy.optimize import curve_fit

    nfreq = freq.size
    dfreq = np.median(np.diff(freq))

    if (vis.size != nfreq) or (smp.size != nfreq):
        ValueError("dimension mismatch")

    # Deal with various options for ensuring the transform of the
    # window is twice the size of the transform of the data.
    navg = 2

    if window_type == 'zeropad':

        # Normalize the weights
        smpa = smp * invert_no_zero(np.sum(smp))
        visa = vis * smpa

        ntau = visa.size
        nsmp = smpa.size

        # Zero-pad to increase resolution
        if res > 0:
            visa = np.concatenate((visa, np.zeros(res*ntau, dtype=visa.dtype)))
            smpa = np.concatenate((smpa, np.zeros(res*nsmp, dtype=smpa.dtype)))

            ntau = visa.size
            nsmp = smpa.size

        # Calculate the window function and dirty spectra
        W = np.fft.fft(smpa, n=nsmp)

        R = np.fft.fft(visa, n=ntau)

        # Calculate delays
        tau = np.fft.fftfreq(nsmp, d=dfreq*1e6) * 1e9
        tau_d = np.fft.fftfreq(ntau, d=dfreq*1e6) * 1e9

        W = np.concatenate((W[tau >= 0], np.zeros_like(W), W[tau < 0]))

        pad_spectra = False

    elif window_type == 'truncate':

        # Normalize the weights
        smpa = smp * invert_no_zero(np.sum(smp))
        visa = vis * smpa

        ntau = visa.size
        nsmp = smpa.size

        # Zero-pad to increase resolution
        if res > 0:
            visa = np.concatenate((visa, np.zeros(res*ntau, dtype=visa.dtype)))
            smpa = np.concatenate((smpa, np.zeros(res*nsmp, dtype=smpa.dtype)))

            ntau = visa.size
            nsmp = smpa.size

        # Calculate the window function and dirty spectra
        W = np.fft.fft(smpa, n=nsmp)

        R = np.fft.fft(visa, n=ntau)

        # Calculate delays
        tau = np.fft.fftfreq(nsmp, d=dfreq*1e6) * 1e9
        tau_d = np.fft.fftfreq(ntau, d=dfreq*1e6) * 1e9

        # Discard high delays
        index_keep = np.concatenate((np.arange(0, ntau / (2 * navg)),
                                     np.arange(ntau - ntau / (2 * navg), ntau)))

        R = R[index_keep]
        tau_d = tau_d[index_keep]
        ntau = tau_d.size

        pad_spectra = True

    elif window_type == 'interpolate':

        # Interpolate and normalize weights
        func_smp = interp1d(np.arange(nfreq), smp, kind='nearest', fill_value='extrapolate')
        smpa = func_smp(np.arange(nfreq*navg) / float(navg))
        smpa *= invert_no_zero(np.sum(smpa))

        visa = vis * smpa[0:nfreq*navg:2]

        ntau = visa.size
        nsmp = smpa.size

        # Zero-pad to increase resolution
        if res > 0:
            visa = np.concatenate((visa, np.zeros(res*ntau, dtype=visa.dtype)))
            smpa = np.concatenate((smpa, np.zeros(res*nsmp, dtype=smpa.dtype)))

            ntau = visa.size
            nsmp = smpa.size

        # Calculate the window function and dirty spectra
        W = np.fft.fft(smpa, n=nsmp)

        R = np.fft.fft(visa, n=ntau)

        # Calculate delays
        tau = np.fft.fftfreq(nsmp, d=(dfreq/navg)*1e6) * 1e9
        tau_d = np.fft.fftfreq(ntau, d=dfreq*1e6) * 1e9

        pad_spectra = False

    elif window_type == 'average':

        ntau = nfreq / navg
        nsmp = navg * ntau

        # Normalize the weights
        smpa = smp[0:nsmp]
        smpa *= invert_no_zero(np.sum(smpa))

        # Average adjacent bins of spectrum
        visb = vis[0:nsmp].reshape(-1, navg)
        smpb = smpa.reshape(-1, navg)

        visa = np.sum(smpb * visb, axis=-1)

        # Zero-pad to increase resolution
        if res > 0:
            visa = np.concatenate((visa, np.zeros(res*ntau, dtype=visa.dtype)))
            smpa = np.concatenate((smpa, np.zeros(res*nsmp, dtype=smpa.dtype)))

            ntau = visa.size
            nsmp = smpa.size

        # Calculate the window function and dirty spectra
        W = np.fft.fft(smpa, n=nsmp)

        R = np.fft.fft(visa, n=ntau)

        # Calculate delays
        tau = np.fft.fftfreq(nsmp, d=dfreq*1e6) * 1e9
        tau_d = np.fft.fftfreq(ntau, d=navg*dfreq*1e6) * 1e9

        pad_spectra = True

    else:
        InputError("Do not recognize window_type = %s." % window_type)

    # Sort in order of increasing delay
    isort_d = np.argsort(tau_d)

    # Create arrays to hold results
    fcomp = np.zeros(niter, dtype=np.int)
    acomp = np.zeros(niter, dtype=np.complex128)
    ssr = np.zeros(niter+1, dtype=np.float64)

    # Iterate
    count = 0

    # Create array to hold work
    if output_all:
        Rall = np.zeros((niter+1, ntau), dtype=np.complex128)
        Rall[0, :] = R.copy()

    ssr[count] = np.sum(np.abs(R)**2)
    delta_ssr = ssr[count]

    while (count < niter) and ((delta_ssr > 0.0) or not check_delta):

        # Locate the peak of the dirty spectra
        ipeak = np.argmax(np.abs(R))
        ipeak_reflect = ntau - ipeak

        # Estimate amplitude
        if (ipeak == 0) or not dual_peak:
            amp = h * R[ipeak]
        else:
            amp = h * (R[ipeak] - R[ipeak_reflect]*W[2*ipeak]) / (1 - np.abs(W[2*ipeak])**2)

        # Save component
        fcomp[count] = ipeak
        acomp[count] = amp

        # Subtract window function from dirty spectra at location of peak
        xshift = isort_d - isort_d[ipeak]

        R -= amp * W[xshift]

        # Increase counter
        count += 1

        # Calculate sum squared residual
        ssr[count] = np.sum(np.abs(R)**2)

        delta_ssr = ssr[count-1] - ssr[count]

        # Save spectra
        if output_all:
            Rall[count, :] = R.copy()

    # Resize output arrays
    niter = int(count)
    fcomp = fcomp[0:niter]
    acomp = acomp[0:niter]
    ssr = ssr[0:(niter+1)]
    if output_all:
        Rall = Rall[0:(niter+1), :]

    if verbose:
        print "%d iterations." % niter

    # Calculate clean beam
    ndiv = 8
    zsmp = np.concatenate((smpa, np.zeros((ndiv-1)*nsmp, dtype=smpa.dtype)))
    B = np.fft.fft(zsmp, n=ndiv*nsmp)

    tau_b = np.fft.fftfreq(ndiv*nsmp, d=dfreq*1e6) * 1e9

    # Fit clean beam to gaussian
    ifit = np.arange(-res*ndiv/2, res*ndiv/2 + 1)

    x = tau_b[ifit]
    x2 = np.tile(x, 2)

    amp = np.abs(B[ifit])
    phi = np.angle(B[ifit], deg=True)
    y = np.concatenate((B[ifit].real, B[ifit].imag))

    p0 = np.array([np.max(amp),
                   0.5*np.mean(np.diff(x))*ndiv*res,
                   np.mean(phi),
                   np.mean(np.diff(phi)) / np.mean(np.diff(x))])

    popt, pcov = curve_fit(lambda v, a0, a1, a2, a3: func_gaussian_with_phase_gradient(v, a0, 0.0, a1, a2, a3),
                           x2, y, p0=p0)

    popt = np.insert(popt, 1, 0.0)

    # Calculate clean spectrum
    S = np.zeros(ntau, dtype=np.complex128)
    for ii in range(niter):
        S += acomp[ii] * model_gaussian_with_phase_gradient((tau_d - tau_d[fcomp[ii]]), *popt)
        #S += acomp[ii] * (tau_d  == tau_d[fcomp[ii]])

    S += R

    # Zero-pad S so that we return to original
    # frequency spacing when peforming an inverse FFT
    if pad_spectra:
        S = np.concatenate((S[tau_d >= 0], np.zeros_like(S), S[tau_d < 0]))
    else:
        tau = tau_d

    if output_all:
        return tau, S, Rall, fcomp, acomp, W, B, popt

    else:
        return tau, S, fcomp, acomp, tau_b, popt


# Fitting Gaussian + Line to clean beam
# --------------------------------------
def fit_gaussian_with_phase_gradient(ra, response, response_error, flag=None):
    """ Fits the complex point source response to a model that
        consists of a gaussian in amplitude and a line in phase.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    ra : np.ndarray[nra, ]
        Transit right ascension.
    response : np.ndarray[nfreq, ninput, nra]
        Complex array that contains point source response.
    response_error : np.ndarray[nfreq, ninput, nra]
        Real array that contains 1-sigma error on
        point source response.
    flag : np.ndarray[nfreq, ninput, nra]
        Boolean array that indicates which data points to fit.

    Returns
    -------
    param : np.ndarray[nfreq, ninput, nparam]
        Best-fit parameters for each frequency and input:
        [peak_amplitude, centroid, fwhm, phase_intercept, phase_slope].
    param_cov: np.ndarray[nfreq, ninput, nparam, nparam]
        Parameter covariance for each frequency and input.
    """

    # Check if boolean flag was input
    if flag is None:
        flag = np.ones(response.shape, dtype=np.bool)
    elif flag.dtype != np.bool:
        flag = flag.astype(np.bool)

    # Create arrays to hold best-fit parameters and
    # parameter covariance.  Initialize to NaN.
    nfreq = response.shape[0]
    ninput = response.shape[1]
    nparam = 5

    param = np.full([nfreq, ninput, nparam], np.nan, dtype=np.float64)
    param_cov = np.full([nfreq, ninput, nparam, nparam], np.nan, dtype=np.float64)

    # Iterate over frequency/inputs and fit point source transit
    for ff in range(nfreq):
        for ii in range(ninput):

            this_flag = flag[ff, ii]

            # Only perform fit if there is enough data.
            # Otherwise, leave parameter values as NaN.
            if np.sum(this_flag) < 5:
                continue

            # We will fit the complex data.  Break n-element complex array g(ra)
            # into 2n-element real array [Re{g(ra)}, Im{g(ra)}] for fit.
            x = np.tile(ra[this_flag], 2)

            y_complex = response[ff, ii, this_flag]
            y = np.concatenate((y_complex.real, y_complex.imag))

            y_error = np.tile(response_error[ff, ii, this_flag], 2)

            # Initial estimate of parameter values:
            # [peak_amplitude, centroid, fwhm,
            #  phase_intercept, phase_slope]
            p0 = np.array([np.max(np.nan_to_num(np.abs(y_complex))), np.median(x), 2.0,
                           np.median(np.nan_to_num(np.angle(y_complex, deg=True))), 0.0])

            # Perform the fit.  If there is an error,
            # then we leave parameter values as NaN.
            try:
                popt, pcov = curve_fit(func_gaussian_with_phase_gradient, x, y,
                                        p0=p0, sigma=y_error, absolute_sigma=True)
            except Exception as error:
                print("Feed %d, Freq %d: %s" % (ii, ff, error))
                continue

            # Save the results
            param[ff, ii] = popt
            param_cov[ff, ii] = pcov

    # Return the best-fit parameters and parameter covariance
    return param, param_cov


def func_gaussian_with_phase_gradient(x, peak_amplitude, centroid, fwhm,
                                      phase_intercept, phase_slope):
    """ Computes parameteric model for the point source transit.
    Model consists of a gaussian in amplitude and a line in phase.
    To be used within curve fitting routine.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    x : np.ndarray[2*nra, ]
        Right ascension in degrees, replicated twice to accomodate
        the real and imaginary components of the response, i.e.,
        x = [ra, ra].
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid : float
        Model parameter.  Centroid of the gaussian in degrees RA.
    fwhm : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees RA.
    phase_intercept : float
        Model parameter.  Phase at the centroid in units of degrees.
    phase_slope : float
        Model parameter.  Fringe rate in degrees phase per degrees RA.

    Returns
    -------
    model : np.ndarray[2*nra, ]
        Model prediction for the complex point source response,
        packaged as [real{g(ra)}, imag{g(ra)}].
    """

    model = np.empty_like(x)
    nreal = len(x)/2

    model_amp = peak_amplitude*np.exp(-4.0*np.log(2.0)*((x[:nreal] - centroid)/fwhm)**2)
    model_phase = np.deg2rad(phase_intercept + phase_slope*(x[:nreal] - centroid))
    model[:nreal] = model_amp*np.cos(model_phase)
    model[nreal:] = model_amp*np.sin(model_phase)

    return model


def model_gaussian_with_phase_gradient(x, peak_amplitude, centroid, fwhm,
                                       phase_intercept, phase_slope):
    """ Computes parameteric model for the point source transit.
    Model consists of a gaussian in amplitude and a line in phase.

    .. math::
        g(ra) = peak_amplitude * \exp{-4 \ln{2} [(ra - centroid)/fwhm]^2} *
                \exp{j [phase_intercept + phase_slope * (ra - centroid)]}

    Parameters
    ----------
    x : np.ndarray[nra, ]
        Right ascension in degrees.
    peak_amplitude : float
        Model parameter.  Normalization of the gaussian.
    centroid : float
        Model parameter.  Centroid of the gaussian in degrees RA.
    fwhm : float
        Model parameter.  Full width at half maximum of the gaussian
        in degrees RA.
    phase_intercept : float
        Model parameter.  Phase at the centroid in units of degrees.
    phase_slope : float
        Model parameter.  Fringe rate in degrees phase per degrees RA.

    Returns
    -------
    model : np.ndarray[nra, ]
        Model prediction for the complex point source response,
        packaged as complex numbers.
    """

    model_amp = peak_amplitude*np.exp(-4.0*np.log(2.0)*((x - centroid)/fwhm)**2)
    model_phase = np.deg2rad(phase_intercept + phase_slope*(x - centroid))
    model = model_amp*np.exp(1.0j*model_phase)

    return model