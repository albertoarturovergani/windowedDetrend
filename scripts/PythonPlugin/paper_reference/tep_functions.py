"""Unmodified published TEP numerical functions, commit 534e926.
Reference validation only, requires Python 3.12 for original f-string syntax.
Original GPL-3.0 functions; see the source provenance JSON. Not imported by core.
"""

def computeTimeMasks(epochs, chan, trial, json_data, do_plot=False, offset=0.20, plot_path=None, plot_title=None):
    import numpy as np
    t=epochs.times

    n=t.size
    data=epochs.get_data()
    sigAll=data[trial,chan,:] if data.ndim==3 else data[chan,:]

    #print("len(t) =", len(t))
    #print("data.shape =", data.shape)
    #print("len(sigAll) =", len(sigAll))
    
    def range_to_mask_by_t(t_start,t_end):
        i_start=int(np.searchsorted(t,t_start,side='left'))
        i_end=int(np.searchsorted(t,t_end,side='right'))-1
        i_start=max(0,min(i_start,n-1))
        i_end=max(0,min(i_end,n-1))
        if i_end<i_start:i_end=i_start
        m=np.zeros(n,dtype=bool)
        m[i_start:i_end+1]=True
        return m
    t_min_off=float(json_data['detrend_minTimeWindowOffset'])
    t_max_off=float(offset)
    maskPreOffset=range_to_mask_by_t(t.min(),t_min_off)
    maskTempOffset=range_to_mask_by_t(t_min_off,t_max_off)
    idx_temp=np.flatnonzero(maskTempOffset)
    if idx_temp.size==0:
        i0=int(np.searchsorted(t,t_min_off,side='left'))
        i0=max(0,min(i0,n-1))
        maskTempOffset=np.zeros(n,dtype=bool)
        maskTempOffset[i0]=True
        idx_temp=np.array([i0],dtype=int)
    sig=sigAll[idx_temp]
    extreme=json_data.get('detrendExtremeTechinque','max')
    if extreme=='derivative' and sig.size>=3:
        diff=np.diff(sig)
        zc=np.where((diff[:-1]<0)&(diff[1:]>0))[0]
        k=(zc[0]+1) if zc.size>0 else int(np.argmax(np.abs(sig)))
    else:
        k=int(np.argmax(np.abs(sig)))
    if k<0:k=0
    if k>=sig.size:k=sig.size-1
    i_peak=idx_temp[k]
    t_peak=t[i_peak]
    maskOffset=np.zeros(n,dtype=bool)
    maskOffset[np.flatnonzero(range_to_mask_by_t(t_min_off,t_peak))]=True
    maskPostOffset=np.zeros(n,dtype=bool)
    i_post_start=min(i_peak+1,n-1)
    maskPostOffset[i_post_start:]=True
    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11,7))
        plt.plot(t,sigAll,label='signal (all)')
        plt.plot(t[idx_temp],sig,label='signal (temp window)')
        plt.scatter(t_peak,sigAll[i_peak],s=60,label=f'peak @ {t_peak:.4f}s')
        plt.axvline(x=t_min_off,linestyle='--',label=f"minTimeWindowOffset={t_min_off}")
        plt.axvline(x=t_max_off,linestyle='--',label=f"maxTimeWindowOffset={t_max_off}")
        plt.title(plot_title or f"chan={chan}, trial={trial}")
        plt.legend(loc='best')
        if plot_path:
            plt.savefig(plot_path,dpi=150,bbox_inches='tight');plt.close()
        else:
            plt.show()

    assert len(maskPreOffset) == len(t), f"maskPreOffset={len(maskPreOffset)} t={len(t)}"
    assert len(maskOffset) == len(t), f"maskOffset={len(maskOffset)} t={len(t)}"
    assert len(maskPostOffset) == len(t), f"maskPostOffset={len(maskPostOffset)} t={len(t)}"
    
    return maskPreOffset,maskOffset,maskPostOffset


def computeSlopes_v4(epochs, json_data, experiment_dir, sub, saveNote=f'plotTrialTepVariability'):
    """
    Compute slopes of linear regressions for EEG data across time windows and trials.
    
    Parameters:
        epochs: MNE Epochs object
            EEG data segmented into epochs.
        normalized_distances: array
            Array of normalized distances for each channel.
        channel_names: list
            List of channel names corresponding to the distances.
        saveNote: str
            A string used for saving plots or notes.

    Returns:
        df_slopes: DataFrame
            Dataframe containing slopes, intercepts, and distance information.
    """
    # compute distance
    evoked = epochs
    positions = np.array([ch['loc'][:3] for ch in evoked.info['chs']])  # Estrai solo le coordinate (x, y, z)
    seed_channels = json_data['seedChans']
    seed_indices = [evoked.info['ch_names'].index(ch_name) for ch_name in seed_channels]
    seed_positions = positions[seed_indices]
    mean_seed_position = np.mean(seed_positions, axis=0)
    distances_from_seed = np.zeros(len(positions))
    for i in range(len(positions)):
        distances_from_seed[i] = euclidean(mean_seed_position, positions[i])
    min_distance = np.min(distances_from_seed)
    normalized_distances = (distances_from_seed - min_distance) / (np.max(distances_from_seed) - min_distance)
    # Create a mapping of channel names to distances
    channel_distance_mapping = dict(zip(evoked.info['ch_names'], normalized_distances))
    
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    slopes = []

    for chan in tqdm(epochs.ch_names):
        id_chan = np.where(np.array(epochs.ch_names) == chan)[0][0]

        if len(epochs.get_data().shape) > 2:
            for id_trial in range(epochs.get_data().shape[0]):
                timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
                for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                    y = epochs.get_data()[id_trial, id_chan, t_mask]
                    slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                    slopes.append([t_label, id_trial, chan, intercept, slope])

        elif len(epochs.get_data().shape) == 2:
            id_trial = 0
            timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                y = epochs.get_data()[id_chan, t_mask]
                slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                slopes.append([t_label, id_trial, chan, intercept, slope])

    # Convert slopes list to DataFrame
    df_slopes = pd.DataFrame(data=slopes, 
                             columns=['id_twindow', 'id_trial', 'chan', 'intercept', 'slope'])

    # Add the 'distanceFromSeed' column using the channel distance mapping
    df_slopes['distanceFromSeed'] = df_slopes['chan'].map(channel_distance_mapping)

    # Compute Z-scores for slopes within each time window
    df_slopes['Zslope'] = np.nan
    for time_label in timeMaskLabels:
        mask = df_slopes['id_twindow'] == time_label
        vals = df_slopes.loc[mask, 'slope'].values.astype(float)
        std = np.nanstd(vals)
        mean = np.nanmean(vals)
        if np.isnan(std) or std == 0:
            df_slopes.loc[mask, 'Zslope'] = 0.0
        else:
            df_slopes.loc[mask, 'Zslope'] = (vals - mean) / std

    # add plot distance from seed vs slope

    return df_slopes


def computeDetrend_v6(EPOCHS, 
                        json_data, experiment_dir, sub,
                        typeOffsetRise,
                        typeOffsetDecay,
                        fitConstraint=True,
                        correctMode='resample', 
                        oddSamples=5, 
                        offsetChans=['Cz', 'Fz'], 
                        lag_correction=True,
                        detrendMode='single',
                        doDetrendOnlyOffsetChans=True,
                     ):
    # CHECK CONSTRAINT START OFFSET 0 or MIN OFFSET TIMEMASK
    # CHECK OTPIMIZATION ACROSS METHODS POLY
    
    from scipy.optimize import curve_fit
    import numpy as np


    base_seed=int(
        json_data.get(
            "detrend_noise_seed",
            42
        )
    )
    
    json_data[
        "detrend_noise_seed"
    ]=base_seed
    
    json_data[
        "detrend_noise_generator"
    ]="numpy.random.default_rng"
    
    json_data[
        "detrend_noise_seed_strategy"
    ]="SeedSequence(base_seed,channel_index,epoch_index)"

    
    # POLY
    def poly_func(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))
    def fit_polynomial_curvefit(x, y, max_order=5):
        best_order = 1
        best_mse = np.inf
        best_coeffs = None
        for order in range(1, max_order + 1):
            p0 = [0.0] * (order + 1)
            try:
                coeffs, _ = curve_fit(lambda x, *c: poly_func(x, *c), x, y, p0=p0)
                y_fit = poly_func(x, *coeffs)
                mse = np.mean((y - y_fit)**2)
                if mse < best_mse:
                    best_mse = mse
                    best_order = order
                    best_coeffs = coeffs
            except Exception as e:
                continue
        return best_order, best_coeffs

    # POWER LAW
    def power_law_decay(x, a, b, c):
        return a * np.power(x, -b) + c
        
    def power_law_decay_constrained(x, b, c, x0, y0):
        a = (y0 - c) * np.power(x0, b)
        return a * np.power(x, -b) + c

    # === MODELLI BASE con parametri diretti (tau_rise, tau_decay) ===
    def alpha_func(x, A, tau, c):
        return A * x * np.exp(-x / tau) + c
    def exp_func_single_rise(x, a, tau, c):
        return a * (1 - np.exp(-x / tau)) + c
    def exp_func_single_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    def exp_func_double_decay(x, a1, tau1, a2, tau2, c):
        return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c
    def exp_func_biexp(x, A, tau_rise, tau_decay, c):
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + c
    def exp_func_biexp_double_decay(x, A, tau_rise, tau_decay1, tau_decay2, B, c):
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    def exp_func_triexp_decay(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c):
        rise = 1 - np.exp(-x / tau_rise)
        B3 = 1.0 - B1 - B2
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + c
    def exp_func_biexp_double_rise_decay(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c):
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    
    # === MODELLI VINCOLATI ===
    # constrained to first element
    def alpha_func_constrained(x, tau, x0, y0):
        A = (y0 / x0) * np.exp(x0 / tau)
        return A * x * np.exp(-x / tau)
    def exp_func_single_rise_constrained(x, tau, c, x0, y0):
        a = (y0 - c) / (1 - np.exp(-x0 / tau))
        return a * (1 - np.exp(-x / tau)) + c
    def exp_func_single_decay_constrained(x, tau, c, x0, y0):
        a = (y0 - c) / np.exp(-x0 / tau)
        return a * np.exp(-x / tau) + c
    def exp_func_double_decay_constrained(x, tau1, tau2, a2, c, x0, y0):
        a1 = y0 - (a2 * np.exp(-x0 / tau2) + c)
        return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c
  
    def exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0):
        f0 = A * (1 - np.exp(-x0 / tau_rise)) * np.exp(-x0 / tau_decay)
        correction = y0 - f0
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + correction
    
    def exp_func_biexp_double_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, B, x0, y0):
        rise0 = 1 - np.exp(-x0 / tau_rise)
        decay0 = B * np.exp(-x0 / tau_decay1) + (1 - B) * np.exp(-x0 / tau_decay2)
        f0 = A * rise0 * decay0
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + (y0 - f0)
    
    def exp_func_triexp_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, x0, y0):
        B3 = 1.0 - B1 - B2
        rise0 = 1 - np.exp(-x0 / tau_rise)
        decay0 = B1 * np.exp(-x0 / tau_decay1) + B2 * np.exp(-x0 / tau_decay2) + B3 * np.exp(-x0 / tau_decay3)
        f0 = A * rise0 * decay0
        rise = 1 - np.exp(-x / tau_rise)
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + (y0 - f0)
    
    def exp_func_biexp_double_rise_decay_constrained(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, x0, y0):
        rise0 = B_r * (1 - np.exp(-x0 / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x0 / tau_rise2))
        decay0 = B_d * np.exp(-x0 / tau_decay1) + (1 - B_d) * np.exp(-x0 / tau_decay2)
        f0 = A * rise0 * decay0
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + (y0 - f0)

    # constrained to 0
    def alpha_func_constrained_to_zero(x, A, tau):
        return A * x * np.exp(-x / tau)
    def exp_func_biexp_constrained_to_zero(x, A, tau_rise, tau_decay, c=0):
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + c
    def exp_func_biexp_double_decay_constrained_to_zero(x, A, tau_rise, tau_decay1, tau_decay2, B, c=0):
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    def exp_func_triexp_decay_constrained_to_zero(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c=0):
        rise = 1 - np.exp(-x / tau_rise)
        B3 = 1.0 - B1 - B2
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + c
    def exp_func_biexp_double_rise_decay_constrained_to_zero(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c=0):
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    
    # === FIT EXP GENERICO CON OPZIONE VINCOLO ===
    def fit_exp_model(x, y, model='singlerise', constrain_start=None):
        x = np.array(x)
        y = np.array(y)
        if len(x) < 3:
            return np.zeros_like(x), [np.nan] * (5 if model == 'doubledecay' else 3)

        elif model == 'wind_alpha':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau):
                    return alpha_func_constrained(x, tau, x0, y0)
                p0 = [0.01]
                bounds = ([1e-6], [np.inf])
                popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = func(x, *popt)
                return yfit, [*popt, x0, y0]  # se vuoi salvare anche x0, y0
            else:
                # fallback: libera
                p0 = [0, 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(alpha_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = alpha_func(x, *popt)
                return yfit, popt

        if model == 'wind_singlerise':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau, c):
                    return exp_func_single_rise_constrained(x, tau, c, x0, y0)
                p0 = [0.01, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [np.max(y), 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(exp_func_single_rise, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = exp_func_single_rise(x, *popt)
                return yfit, popt

        elif model == 'wind_powerlaw':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, b, c):
                    return power_law_decay_constrained(x, b, c, x0, y0)
                p0 = [1.0, 0.0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [1.0, 0.0, np.min(y)]
                bounds = ([0.001, 0.0, -np.inf], [5.0, 10.0, np.inf])
                popt, _ = curve_fit(power_law_decay, x, y, p0=p0, bounds=bounds, maxfev=5000)
                return power_law_decay(x, *popt), popt

        elif model == 'wind_singledecay':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau, c):
                    return exp_func_single_decay_constrained(x, tau, c, x0, y0)
                p0 = [0.01, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [0.01, 0.01, 0.0]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(exp_func_single_decay, x, y, p0=p0, bounds=bounds, maxfev=5000)
                return exp_func_single_decay(x, *popt), popt

        elif model == 'wind_doubledecay':
            if len(x) < 5:
                return np.zeros_like(x), [np.nan] * 5
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau1, tau2, a2, c):
                    return exp_func_double_decay_constrained(x, tau1, tau2, a2, c, x0, y0)
                p0 = [0.01, 0.1, y0/2, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                return func(x, *popt), [*popt]
            else:
                p0 = [0.01, 0.1, 0.01, 0.01, 0.0]
                popt, _ = curve_fit(exp_func_double_decay, x, y, p0=p0, maxfev=10000)
                return exp_func_double_decay(x, *popt), popt

        
        elif model == 'nowind_biexp':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)
                    p0 = [0, 0.01, 0.05]
                    bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                    bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                    popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexp: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                return np.full_like(x, np.nan), [np.nan] * 4

        elif model == 'nowind_biexpdouble':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay1, tau_decay2):
                        return exp_func_biexp_double_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, x0, y0)
                    p0 = [0, 0.01, 0.05, 0.1, 0.5]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 0], [np.inf, 1.0, 1.0, 1.0, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise, tau_decay1, tau_decay2, B, c
                    p0 = [0, 0.01, 0.05, 0.1, 0.5, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 0, -np.inf], [np.inf, 1.0, 1.0, 1.0, 1, np.inf])
                    popt, _ = curve_fit(exp_func_biexp_double_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp_double_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdouble: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico a biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)
                        p0 = [0, 0.01, 0.05]
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_biexptriple':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_triexp_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, x0, y0)                   
                    p0 = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.3]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 1e-6, 0, 0],
                              [np.inf, 1.0, 1.0, 1.0, 1.0, 1, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c
                    p0 = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.3, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 1e-6, 0, 0, -np.inf],
                              [np.inf, 1.0, 1.0, 1.0, 1.0, 1, 1, np.inf])
                    popt, _ = curve_fit(exp_func_triexp_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_triexp_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdouble: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico al modello biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)    
                        p0 = [0, 0.01, 0.05]  # A, tau_rise, tau_decay
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_biexpdoublerd':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_biexp_double_rise_decay_constrained(x, A, tau_rise, tau_decay, x0, y0)   
                    p0 = [0, 0.01, 0.05, 0.5, 0.05, 0.1, 0.5]
                    bounds = ([-np.inf, 1e-6, 1e-6, 0, 1e-6, 1e-6, 0],
                              [np.inf, 1.0, 1.0, 1, 1.0, 1.0, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c
                    p0 = [0, 0.01, 0.05, 0.5, 0.05, 0.1, 0.5, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 0, 1e-6, 1e-6, 0, -np.inf],
                              [np.inf, 1.0, 1.0, 1, 1.0, 1.0, 1, np.inf])
                    popt, _ = curve_fit(exp_func_biexp_double_rise_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp_double_rise_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdoublerd: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico al modello biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)   
                        p0 = [0, 0.01, 0.05]  # A, tau_rise, tau_decay
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_alpha':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, A, tau_rise, tau_decay):
                    return alpha_func_constrained(x, A, tau_rise, x0, y0)   
                p0 = [0, 0.01]  # A, tau
                bounds = ([-np.inf, 1e-6], [np.inf, np.inf])
                try:
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                    yfit = func(x, *popt)
                    return yfit, list(popt)  + [x0, y0]
                except Exception as e:
                    print(f"[fit_exp_model] Fit failed for alpha (c=0): {e}")
                    return np.full_like(x, np.nan), [np.nan] * 3
            else:
                p0 = [0, 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                try:
                    popt, _ = curve_fit(alpha_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                    yfit = alpha_func(x, *popt)
                    return yfit, popt
                except Exception as e:
                    print(f"[fit_exp_model] Fit failed for alpha: {e}")
                    return np.full_like(x, np.nan), [np.nan] * 3
        else:
            raise ValueError("Unsupported model. Available models: 'singlerise', 'singledecay', 'doubledecay', 'biexp', 'biexpdouble', 'biexptriple', 'biexpdoublerd', 'alpha', 'powerlaw'")
   
    times = EPOCHS.times
    data_detrended = EPOCHS.get_data().copy()
    tabStat = []
    mse_list = []
    fitted_params_post = []
    max_order_offset_list = []
    markerChans = offsetChans
    nplot = 10
    totplot = EPOCHS.get_data().shape[0]*len(markerChans)
    print(totplot)
    PROB = nplot/totplot
    print('p plot', PROB)


    for chan in tqdm(
        EPOCHS.ch_names
    ):
        if (
            doDetrendOnlyOffsetChans
            and chan not in offsetChans
        ):
            continue  # salta questo canale, mantiene i dati originali
    
        id_chan=int(
            np.where(
                np.asarray(
                    EPOCHS.ch_names
                )==chan
            )[0][0]
        )
    
        for epoch_idx in range(
            data_detrended.shape[0]
        ):
            rng=np.random.default_rng(
                np.random.SeedSequence([
                    base_seed,
                    int(id_chan),
                    int(epoch_idx)
                ])
            )
    
            tep=data_detrended[
                epoch_idx,
                id_chan,
                :
            ].reshape(
                -1,
                1
            )

            #print("len(tep) =", len(tep))
            #assert len(tep) == len(times), f"len(tep)={len(tep)} len(times)={len(times)}"
            
            TEP = tep.copy()
            timeMask = computeTimeMasks(EPOCHS, id_chan, epoch_idx, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            """
            print("len(times) =", len(times))
            print("len(mask0) =", len(timeMask[0]))
            print("len(mask1) =", len(timeMask[1]))
            print("len(mask2) =", len(timeMask[2]))
            assert len(timeMask[0]) == len(times)
            assert len(timeMask[1]) == len(times)
            assert len(timeMask[2]) == len(times)
            """
            # --- PRE OFFSET detrending (segmento centrale: timeMask[0]) ---
            poly_coeffs_A = np.polyfit(times[timeMask[0]], tep[timeMask[0]].flatten(), json_data['detrend_polOrder_preOffset'])
            trend_line_A = np.polyval(poly_coeffs_A, times[timeMask[0]])
            tepA = tep[timeMask[0]].squeeze() - trend_line_A
            mseA = np.mean((tepA) ** 2)
            OPTPARS_A = f'wind_poly_{json_data['detrend_polOrder_preOffset']}'

            # --- NOWIND FIT ---
            if typeOffsetRise == typeOffsetDecay and typeOffsetRise.startswith('nowind_'):
                start_idx = np.where(timeMask[1])[0][0]
                timeMask_ext = np.zeros_like(times, dtype=bool)
                timeMask_ext[start_idx:] = True
                x_fit = times[timeMask_ext]
                y_fit = tep[timeMask_ext].flatten()
                if typeOffsetRise == 'nowind_poly':
                    # === Ordine ottimale come sqrt(round) dell'ordine lagrange ===
                    n_fit = len(x_fit)
                    order_lagrange = n_fit - 1
                    order_guess = int(np.round(np.sqrt(order_lagrange)))
                    order_guess = 5 #int(np.round(max(1, order_guess)*0.5))
                    if fitConstraint:
                        x0 = times[timeMask[0]][-1]
                        y0 = tep[timeMask[0]][-1][0]
                        best_order = order_guess
                        best_mse = np.inf
                        trend_line, coeffs = polyfit_constrained_start(x_fit, y_fit, order_guess, x0=x0, y0=y0)
                        mse = np.mean((y_fit - trend_line) ** 2)
                        trend_line_B = trend_line
                        popt_B = best_order
                    else:
                        best_order, best_coeffs = fit_polynomial_curvefit(x_fit, y_fit, max_order=order_guess)
                        trend_line_B = poly_func(x_fit, *best_coeffs)
                        popt_B = best_order
                else:
                    # === Modelli esponenziali, alpha, biexp, ecc. ===
                    trend_line_B, popt_B = fit_exp_model(
                        x_fit,
                        y_fit,
                        model=typeOffsetRise,
                        constrain_start=(times[timeMask[0]][-1], tep[timeMask[0]][-1][0]) if fitConstraint else None
                    )
                # === Common detrending post-model ===
                y_detrended = y_fit - trend_line_B
                tepB = y_detrended[:np.sum(timeMask[1])]
                tepC = y_detrended[np.sum(timeMask[1]):]
                mseB = np.mean(tepB ** 2)
                mseC = np.mean(tepC ** 2)
                max_order_offset_list.append([chan, epoch_idx, f"{typeOffsetRise}_({popt_B})"])
                OPTPARS_B = popt_B
                OPTPARS_C = popt_B
                tabStat.append([chan, epoch_idx, mseA, mseB, mseC, OPTPARS_A, OPTPARS_B, OPTPARS_C])
                mse_list.append(mseA + mseB + mseC)
                # 16102025
                #tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
                # === dopo aver calcolato tepA, tepB, tepC ===
                # Costruisco un vettore full-length (stessa lunghezza di tep / EPOCHS.times)
                tep_agg = tep.flatten().copy()
                tep_agg[timeMask[0]] = tepA
                tep_agg[timeMask[1]] = tepB
                tep_agg[timeMask[2]] = tepC
                
                if lag_correction:
                    tep_agg_shifted, n_shift = shift_signal_by_mask(tep_agg, timeMask[1])
                if correctMode!=False:
                    tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models, rng=rng)
                data_detrended[epoch_idx, id_chan, :] = tep_agg
                is_marker = chan in markerChans
                plot_detrend_example_v3(
                    typeOffsetRise=typeOffsetRise,
                    typeOffsetDecay=typeOffsetDecay,
                    OPTPARS_A=OPTPARS_A,
                    OPTPARS_B=OPTPARS_B,
                    OPTPARS_C=OPTPARS_C,
                    sub=sub,
                    chan=chan,
                    epoch_idx=epoch_idx,
                    times=times,
                    TEP=TEP,
                    trend_line_A=trend_line_A,
                    trend_line_B=trend_line_B[:np.sum(timeMask[1])],
                    trend_line_C=trend_line_B[np.sum(timeMask[1]):],
                    tep_agg=tep_agg,
                    timeMask=timeMask,
                    fitConstraint=fitConstraint,
                    markerChan=is_marker,
                    experiment_dir=experiment_dir,
                    PROB=PROB
                )
                continue  # Salta post-processing standard
            
            # --- WIND FIT ---
            n_offset = len(times[timeMask[1]])
            x_offset = times[timeMask[1]]
            y_offset = tep[timeMask[1]].flatten()
            if isinstance(typeOffsetRise, str):
                typeOffsetRise_lower = typeOffsetRise.lower()
            # -- parsing per wind_poly_xx --
            if typeOffsetRise_lower.startswith("wind_poly_"):
                suffix = typeOffsetRise_lower.split('_')[-1]
                if suffix == "lagrange":
                    order_poly_offset = n_offset - 1
                    poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                    trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"lagrange({order_poly_offset})"])
                    OPTPARS_B = order_poly_offset
                elif suffix == "opt":
                    if json_data.get('detrend_offsetStart', False):
                        target_time = json_data.get('detrend_minTimeWindowOffset', x_offset[0])
                        idx_local = np.argmin(np.abs(x_offset - target_time))
                        x0_start = x_offset[idx_local]
                        y0_start = y_offset[idx_local]
                        best_order = 1
                        best_mse = np.inf
                        best_coeffs = None
                        best_trend = None
                        for order in range(1, 4):
                            try:
                                trend, coeffs = polyfit_constrained_start(x_offset, y_offset, order, x0=x0_start, y0=y0_start)
                                mse = np.mean((y_offset - trend) ** 2)
                                if mse < best_mse:
                                    best_order = order
                                    best_mse = mse
                                    best_coeffs = coeffs
                                    best_trend = trend
                            except Exception:
                                continue
                        trend_line_B = best_trend
                        tepB = y_offset - trend_line_B
                        mseB = best_mse
                        max_order_offset_list.append([chan, epoch_idx, f"poly_opt_constr({best_order})"])
                        OPTPARS_B = best_order
                    else:
                        best_order, best_coeffs = fit_polynomial_curvefit(x_offset, y_offset, max_order=3)
                        trend_line_B = poly_func(x_offset, *best_coeffs)
                        tepB = y_offset - trend_line_B
                        mseB = np.mean((tepB) ** 2)
                        max_order_offset_list.append([chan, epoch_idx, f"poly_opt({best_order})"])
                        OPTPARS_B = best_order
                elif suffix.isdigit():
                    order_poly_offset = int(suffix)
                    if json_data.get('detrend_offsetStart', False):
                        target_time = json_data.get('detrend_minTimeWindowOffset', x_offset[0])
                        idx_local = np.argmin(np.abs(x_offset - target_time))
                        x0_start = x_offset[idx_local]
                        y0_start = y_offset[idx_local]
                        trend_line_B, poly_coeffs_B = polyfit_constrained_start(x_offset, y_offset, order_poly_offset, x0_start, y0_start)
                    else:
                        poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                        trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"poly_{order_poly_offset}"])
                    OPTPARS_B = order_poly_offset
            
                else:
                    raise ValueError(f"❌ Suffix '{suffix}' in '{typeOffsetRise}' non riconosciuto.")
            elif typeOffsetRise_lower == 'wind_spline':
                from scipy.interpolate import UnivariateSpline
                spline = UnivariateSpline(x_offset, y_offset, s=1e-14)
                trend_line_B = spline(x_offset)
                tepB = y_offset - trend_line_B
                mseB = np.mean(tepB ** 2)
                max_order_offset_list.append([chan, epoch_idx, 'wind_spline'])
                OPTPARS_B = typeOffsetRise_lower
            elif typeOffsetRise_lower in ['wind_singlerise', 'wind_powerlaw']:
                try:
                    target_time = json_data['detrend_minTimeWindowOffset']
                    idx_local = np.argmin(np.abs(x_offset - target_time))
                    x0_start = x_offset[idx_local]
                    y0_start = y_offset[idx_local]
                    trend_line_B, popt_B = fit_exp_model(
                        x_offset,
                        y_offset,
                        model=typeOffsetRise_lower,
                        constrain_start=(x0_start, y0_start) if json_data.get('detrend_offsetStart', False) else None
                    )
                    tepB = y_offset - trend_line_B
                    mseB = np.mean(tepB ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"{typeOffsetRise_lower}_exp"])
                    OPTPARS_B = popt_B
                except Exception as e:
                    order_poly_offset = n_offset - 1
                    print(f"[WARNING] Fitting {typeOffsetRise_lower} failed on chan {chan}, trial {epoch_idx} – fallback to lagrange. Error: {e}")
                    poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                    trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"fallback_lagrange({order_poly_offset})"])
                    OPTPARS_B = order_poly_offset
            else:
                raise ValueError(f"typeOffset '{typeOffsetRise}' non riconosciuto.")

            # --- POST OFFSET detrending (segment: timeMask[2]) ---
            # exp power law
            if typeOffsetDecay in ['wind_singledecay', 'wind_doubledecay', 'wind_powerlaw']:
                y0_start = tep[timeMask[1]][-1][0]
                x0_start = times[timeMask[1]][-1]
                try:
                    trend_line_C, popt_C = fit_exp_model(
                        times[timeMask[2]],
                        tep[timeMask[2]].flatten(),
                        model=typeOffsetDecay,
                        constrain_start=(x0_start, y0_start) if fitConstraint else None
                    )
                    model_used = typeOffsetDecay
                except Exception as e:
                    print(f"[⚠️] Errore nel fit '{typeOffsetDecay}' su {chan}, epoca {epoch_idx}: {e} – fallback su 'single'")
                    trend_line_C, popt_C = fit_exp_model(
                        times[timeMask[2]],
                        tep[timeMask[2]].flatten(),
                        model='wind_singledecay',
                        constrain_start=(x0_start, y0_start) if fitConstraint else None
                    )
                    model_used = 'wind_singledecay'
                OPTPARS_C = popt_C
                tepC = tep[timeMask[2]].squeeze() - trend_line_C
                mseC = np.mean((tep[timeMask[2]].squeeze() - trend_line_C) ** 2)
                
            # --- POST OFFSET DECAY detrending con wind_poly_* ---
            if isinstance(typeOffsetDecay, str) and typeOffsetDecay.lower().startswith('wind_poly_'):
                suffix_decay = typeOffsetDecay.lower().split('_')[-1]
                maskPostOffset = timeMask[2]
                x_post = times[maskPostOffset]
                y_post = tep[maskPostOffset].flatten()
            
                if suffix_decay == 'opt':
                    if fitConstraint:
                        target_time_post = json_data.get('detrend_minTimeWindowPostOffset', x_post[0])
                        idx_post = np.argmin(np.abs(x_post - target_time_post))
                        x0_post = x_post[idx_post]
                        y0_post = y_post[idx_post]
                        best_order_post = 1
                        best_mse_post = np.inf
                        best_trend_post = None
                        best_coeffs_post = None
                        for order in range(1, 4):
                            try:
                                trend_post, coeffs_post = polyfit_constrained_start(
                                    x_post, y_post, order=order, x0=x0_post, y0=y0_post
                                )
                                mse = np.mean((y_post - trend_post) ** 2)
                                if mse < best_mse_post:
                                    best_order_post = order
                                    best_mse_post = mse
                                    best_trend_post = trend_post
                                    best_coeffs_post = coeffs_post
                            except Exception:
                                continue
                        trend_line_C = best_trend_post
                        tepC = y_post - trend_line_C
                        mseC = best_mse_post
                        OPTPARS_C = best_order_post
                    else:
                        best_order_post, best_coeffs_post = fit_polynomial_curvefit(x_post, y_post, max_order=3)
                        trend_line_C = poly_func(x_post, *best_coeffs_post)
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                        OPTPARS_C = best_order_post
            
                elif suffix_decay.isdigit():
                    order_poly_decay = int(suffix_decay) if chan in offsetChans else max(0, int(suffix_decay) - 1)

                    if fitConstraint:
                        end_idx = np.where(timeMask[1])[0][-1]
                        x0_post = times[end_idx]
                        y0_post = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                        trend_line_C, poly_coeffs_C = polyfit_constrained_start(
                            x_post, y_post, order=order_poly_decay, x0=x0_post, y0=y0_post
                        )
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                    else:
                        poly_coeffs_C = np.polyfit(x_post, y_post, order_poly_decay)
                        trend_line_C = np.polyval(poly_coeffs_C, x_post)
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                    OPTPARS_C = order_poly_decay
                else:
                    raise ValueError(f"❌ Suffix '{suffix_decay}' in '{typeOffsetDecay}' non riconosciuto.")

            # spline
            if typeOffsetDecay == 'wind_spline':
                from scipy.interpolate import UnivariateSpline
                mask_post = timeMask[2]
                x_post = times[mask_post]
                y_post = tep[mask_post].flatten()
                if fitConstraint:
                    # Trova punto iniziale per vincolo
                    end_idx = np.where(timeMask[1])[0][-1]
                    x0 = times[end_idx]
                    y0 = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                    # Shift y così che passi da y0 in x0
                    y_shift = y_post - (y_post[0] - y0)  # forza partenza da y0
                    spline = UnivariateSpline(x_post, y_shift, s=1e-14)
                else:
                    spline = UnivariateSpline(x_post, y_post, s=1e-14)
                trend_line_C = spline(x_post)
                tepC = y_post - trend_line_C
                mseC = np.mean(tepC ** 2)
                mse_list.append(mseA + mseB + mseC)
                OPTPARS_C = typeOffsetDecay
            # pchip
            elif typeOffsetDecay == 'wind_pchip':
                from scipy.interpolate import PchipInterpolator
                mask_post = timeMask[2]
                x_post = times[mask_post]
                y_post = tep[mask_post].flatten()
                if fitConstraint:
                    end_idx = np.where(timeMask[1])[0][-1]
                    x0 = times[end_idx]
                    y0 = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                    delta_y = y_post[0] - y0
                    y_shift = y_post - delta_y
                    pchip = PchipInterpolator(x_post, y_shift)
                else:
                    pchip = PchipInterpolator(x_post, y_post)
                trend_line_C = pchip(x_post)
                tepC = y_post - trend_line_C
                mseC = np.mean(tepC ** 2)
                mse_list.append(mseA + mseB + mseC)
                OPTPARS_C = typeOffsetDecay

            #####################################################################################
            tabStat.append([chan, epoch_idx, mseA, mseB, mseC, OPTPARS_A, OPTPARS_B, OPTPARS_C])
            mse_list.append(mseA + mseB + mseC)
            # 16102025
            #tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
            # === dopo aver calcolato tepA, tepB, tepC ===
            # Costruisco un vettore full-length (stessa lunghezza di tep / EPOCHS.times)
            tep_agg = tep.flatten().copy()
            tep_agg[timeMask[0]] = tepA
            tep_agg[timeMask[1]] = tepB
            tep_agg[timeMask[2]] = tepC
            #####################################################################################
            if lag_correction:
                tep_agg_shifted, n_shift = shift_signal_by_mask(tep_agg, timeMask[1])
            ######################################################################################
            if correctMode!=False:
                tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models, rng=rng)
            ######################################################################################            
            data_detrended[epoch_idx, id_chan, :] = tep_agg
            ######################################################################################            
            is_marker = chan in markerChans
            plot_detrend_example_v3(
                    typeOffsetRise=typeOffsetRise,
                    typeOffsetDecay=typeOffsetDecay,
                sub=sub,
                chan=chan,
                epoch_idx=epoch_idx,
                times=times,
                TEP=TEP,
                OPTPARS_A=OPTPARS_A,
                OPTPARS_B=OPTPARS_B,
                OPTPARS_C=OPTPARS_C,
                trend_line_A=trend_line_A,
                trend_line_B=trend_line_B,
                trend_line_C=trend_line_C,
                tep_agg=tep_agg,
                timeMask=timeMask,
                fitConstraint=fitConstraint,
                markerChan=is_marker,
                experiment_dir=experiment_dir,
                PROB=PROB
                )

    MSE = np.mean(mse_list)
    EPOCHS_DETRENDED = EPOCHS.copy()
    EPOCHS_DETRENDED._data = np.asarray(data_detrended).copy()   
    print("INPUT data shape:", EPOCHS.get_data().shape)
    print("INPUT times len:", len(EPOCHS.times))
    print("DETRENDED data shape:", EPOCHS_DETRENDED.get_data().shape)
    print("DETRENDED times len:", len(EPOCHS_DETRENDED.times))
    assert EPOCHS_DETRENDED.get_data().shape[2] == len(EPOCHS_DETRENDED.times), \
        f"{EPOCHS_DETRENDED.get_data().shape[2]} vs {len(EPOCHS_DETRENDED.times)}"
    df_tabStat = pd.DataFrame(
        tabStat,
        columns=['chan', 'epoch_idx', 'mseA', 'mseB', 'mseC', 'OPTPARS_A', 'OPTPARS_B', 'OPTPARS_C']
    )
    tabStat_path = os.path.join(experiment_dir, '3.detrend', f'tabStatDetrend_{typeOffsetDecay}.csv')
    df_tabStat.to_csv(tabStat_path, index=False)
    print(f"[INFO] Salvato tabStat in: {tabStat_path}")

    return EPOCHS_DETRENDED, MSE, max_order_offset_list


def polyfit_constrained_start(x, y, order, x0, y0):
    """
    Fit polinomiale di grado `order`, vincolato a passare da (x0, y0).
    Ritorna: trend_line, coeffs (inclusi i coef. vincolati, cioè con il termine costante = y0)
    """
    # Shift dei dati rispetto al vincolo
    x_shifted = x - x0
    y_shifted = y - y0

    # Costruzione della matrice di Vandermonde senza termine costante (che è y0)
    X = np.vander(x_shifted, N=order+1)[:, :-1]  # esclude il termine x^0

    # Fit dei coefficienti rimanenti
    coeffs_reduced = np.linalg.lstsq(X, y_shifted, rcond=None)[0]

    # Ricostruzione del polinomio completo: y0 + a₁(x−x₀) + a₂(x−x₀)² + ...
    trend_line = y0 + np.polyval(np.append(coeffs_reduced, 0), x_shifted)

    # Coefficienti del polinomio globale (per compatibilità)
    # NB: per avere anche il termine costante effettivo, ricostruiamo il polinomio globale
    full_poly = np.poly1d(np.append(coeffs_reduced, 0))  # append 0 per x^0
    poly_coeffs = full_poly.integ()(x - x0) * 0 + y0 + np.polyval(full_poly, x - x0)

    return trend_line, coeffs_reduced


def apply_offset_correction(
    tep_agg,
    tep,
    timeMask,
    correctMode,
    oddSamples,
    EPOCHS,
    supported_models,
    rng=None
):
    import numpy as np
    from scipy.signal import resample

    if rng is None:
        rng=np.random.default_rng(42)

    tep_agg=np.asarray(
        tep_agg,
        dtype=float
    ).copy()

    tep=np.asarray(
        tep,
        dtype=float
    ).reshape(-1)

    times=np.asarray(
        EPOCHS.times,
        dtype=float
    )

    odd_samples_ms=(
        0.0
        if oddSamples is None or oddSamples is False
        else float(oddSamples)
    )

    margin_seconds=(
        odd_samples_ms
        /1000.0
    )

    pre_indices=np.flatnonzero(
        timeMask[0]
    )

    offset_indices=np.flatnonzero(
        timeMask[1]
    )

    if (
        pre_indices.size==0
        or offset_indices.size==0
    ):
        return tep_agg

    pre_end_time=float(
        times[
            pre_indices[-1]
        ]
    )

    offset_start_time=float(
        times[
            offset_indices[0]
        ]
    )

    offset_end_time=float(
        times[
            offset_indices[-1]
        ]
    )

    precorrectionMask=(
        (times>=float(times[pre_indices[0]]))
        &(times<=pre_end_time-margin_seconds)
    )

    if not np.any(
        precorrectionMask
    ):
        precorrectionMask=timeMask[
            0
        ].copy()

    correctionMask=(
        (times>=offset_start_time-margin_seconds)
        &(times<=offset_end_time+margin_seconds)
    )

    num_samples=int(
        np.sum(
            correctionMask
        )
    )

    pre_series=tep[
        precorrectionMask
    ]

    pre_series=pre_series[
        np.isfinite(
            pre_series
        )
    ]

    if (
        num_samples==0
        or pre_series.size==0
    ):
        return tep_agg

    mode=(
        str(correctMode).strip()
        if correctMode is not False
        else ""
    )

    mode_lower=mode.lower()

    supported_lower={
        str(item).lower()
        for item in supported_models
    }

    if mode_lower=="moving_average":
        window_size=max(
            1,
            int(
                round(
                    odd_samples_ms
                )
            )
        )

        if pre_series.size>=window_size:
            value=float(
                np.mean(
                    pre_series[
                        -window_size:
                    ]
                )
            )
        else:
            value=float(
                np.mean(
                    pre_series
                )
            )

        new_samples=np.full(
            num_samples,
            value,
            dtype=float
        )

    elif mode_lower=="median":
        new_samples=np.full(
            num_samples,
            float(
                np.median(
                    pre_series
                )
            ),
            dtype=float
        )

    elif mode_lower=="zeros":
        new_samples=np.zeros(
            num_samples,
            dtype=float
        )

    elif mode_lower=="resample":
        new_samples=resample(
            pre_series,
            num=num_samples
        )

    elif mode_lower in supported_lower:
        new_samples=generate_noise_from_distribution(
            time_series=pre_series,
            model=mode,
            n_samples=num_samples,
            rng=rng
        )

    else:
        raise ValueError(
            f"Metodo di correzione offset "
            f"non riconosciuto: {correctMode}"
        )

    tep_agg[
        correctionMask
    ]=new_samples

    return tep_agg


def generate_noise_from_distribution(
    time_series,
    model="Gaussian",
    n_samples=1000,
    rng=None
):
    import numpy as np

    if rng is None:
        rng=np.random.default_rng(42)

    x=np.asarray(
        time_series,
        dtype=float
    ).ravel()

    x=x[
        np.isfinite(x)
    ]

    if x.size==0:
        return np.zeros(
            n_samples,
            dtype=float
        )

    mean=float(
        np.mean(x)
    )

    std=float(
        np.std(x)*0.5
    )

    median=float(
        np.median(x)
    )

    minimum=float(
        np.min(x)
    )

    maximum=float(
        np.max(x)
    )

    epsilon=1e-8
    std=max(
        std,
        epsilon
    )

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        minimum=mean-std
        maximum=mean+std

    if minimum==maximum:
        minimum-=std
        maximum+=std

    if minimum>maximum:
        minimum,maximum=maximum,minimum

    model_name=str(
        model
    ).strip().lower()

    if model_name=="gaussian":
        return rng.normal(
            loc=mean,
            scale=std,
            size=n_samples
        )

    if model_name=="exponential":
        return rng.exponential(
            scale=max(std,epsilon),
            size=n_samples
        )

    if model_name=="laplace":
        return rng.laplace(
            loc=median,
            scale=std,
            size=n_samples
        )

    if model_name=="poisson":
        return rng.poisson(
            lam=max(mean,0.0),
            size=n_samples
        )

    if model_name=="rayleigh":
        return rng.rayleigh(
            scale=std,
            size=n_samples
        )

    if model_name=="gamma":
        shape=(
            mean**2/std**2
            if std>epsilon
            else 1.0
        )

        shape=max(
            shape,
            epsilon
        )

        scale=(
            std**2
            /max(
                abs(mean),
                epsilon
            )
        )

        return rng.gamma(
            shape=shape,
            scale=scale,
            size=n_samples
        )

    if model_name in (
        "studentt",
        "student_t",
        "t"
    ):
        return (
            rng.standard_t(
                df=2.0,
                size=n_samples
            )
            *std
            +mean
        )

    if model_name=="uniform":
        return rng.uniform(
            low=minimum,
            high=maximum,
            size=n_samples
        )

    raise ValueError(
        f"Modello non supportato: {model}"
    ) 


def shift_signal_by_mask(signal, mask):
    """
    Shift a signal forward in time by the length of a boolean mask.
    Pads zeros at the beginning and trims the end to maintain the same length.

    Parameters:
    - signal: 1D np.array
    - mask: boolean np.array

    Returns:
    - shifted_signal: 1D np.array, same shape as signal
    - n_shift: number of samples shifted
    """
    n_shift = np.sum(mask)
    shifted_signal = np.pad(signal, (n_shift, 0), mode='constant')[:len(signal)]
    return shifted_signal, n_shift

