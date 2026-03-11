def computeDetrendSteps(epochs, json_data, experiment_dir, sub, computeFOOOF=True):
    print(f"\n🔍 [{sub}] Step 1: Verifica della necessità di detrending...")
    json_data = check_detrend_need(epochs, json_data, experiment_dir, sub)

    print(f"🧼 [{sub}] Step 2: Esecuzione del pipeline di detrending sui canali selezionati...")
    detrendedEpochs, json_data = run_detrend_pipeline(epochs, json_data, sub, experiment_dir)

    print(f"✅ [{sub}] Detrending completato.\n")
    print(f"\n🔍 [{sub}] Step 3: Compute FOOOF")

    if computeFOOOF:
        df = extract_psd_features(detrendedEpochs, 'postDetrend', experiment_dir, json_data)

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    return detrendedEpochs, json_data

def check_detrend_need(epochs, json_data, experiment_dir, sub):

    print(f"\n🔍 [{sub}] Step 1: analisi delle latenze offset")
    json_data, df = analyze_offset_times(
        epochs,
        json_data,
        experiment_dir,
        sub,
        do_plot_variability=True
    )

    print(f"📉 [{sub}] Step 2: Calcolo delle pendenze (computeSlopes_v4)")
    df_slopes = computeSlopes_v4(epochs, json_data, experiment_dir, sub)

    print(f"📊 [{sub}] Step 3: Plot delle pendenze normalizzate (Zslope)")
    computeSlopesPlot(df_slopes, json_data, experiment_dir, sub, saveNote='ALL-TRIALS_preDetrend', zvalue=True)

    print(f"📈 [{sub}] Step 4: Calcolo media Zslope per canale e finestra")
    threshold = json_data['detrend_slopeThr']
    mean_df = df_slopes.groupby(['id_twindow', 'chan'], as_index=False)['Zslope'].mean()
    
    outliers_df = mean_df[np.abs(mean_df['Zslope']) >= threshold]
    found_outliers = not outliers_df.empty

    json_data['offsetChans'] = outliers_df['chan'].unique().tolist()
    json_data['do_detrend'] = found_outliers if json_data.get('do_detrend', True) else False

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 17))
    for ax, title, mask in zip(
        [ax1, ax2],
        ['All Channels', f'offsetChans Channels: {json_data["offsetChans"]}'],
        [df['chan'].isin(df['chan']), df['chan'].isin(json_data['offsetChans'])]
    ):
        hist_values, bin_edges = np.histogram(df[mask]['toffsetmax'], bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.3, edgecolor='black')
        ax.axvline(json_data['detrend_modeTimeWindowOffset'], color='purple', linestyle='-.', linewidth=3, label=f'Mode {json_data['detrend_modeTimeWindowOffset']:.4f}')
        ax.axvline(json_data['detrend_minTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Min Detrend')
        ax.axvline(json_data['detrend_maxTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Max Detrend')
        ax.axvspan(json_data['pulse_artifact_rej_timewindow_min'],
                   json_data['pulse_artifact_rej_timewindow_max'],
                   color='k', alpha=0.3, label='Pulse Artifact Window')
        ax.set_title(f'Histogram of toffsetmax — {title}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency')
        #ax.set_xlim(json_data['detrend_minTimeWindowOffset'], json_data['detrend_maxTimeWindowOffset'])
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plot_path = Path(experiment_dir) / '2.detrend' / 'histogram_toffsetmax_subplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n📌 [{sub}] Risultati:")
    print(f"   • Zslope threshold = {threshold}")
    print(f"   • Canali oltre soglia ({len(json_data['offsetChans'])}): {json_data['offsetChans']}")
    print(f"   • do_detrend = {json_data['do_detrend']}\n")

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    # === Salvataggio CSV dei risultati ===
    detrend_dir = Path(experiment_dir) / '2.detrend'
    detrend_dir.mkdir(parents=True, exist_ok=True)

    mean_df.to_csv(detrend_dir / 'mean_Zslope_per_chan_twindow.csv', index=False)
    outliers_df.to_csv(detrend_dir / 'outlier_Zslope.csv', index=False)

    return json_data

def analyze_offset_times(epochs, json_data, experiment_dir, sub, do_plot_variability=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from pathlib import Path

    offsetTimes = []
    for chan in tqdm(epochs.ch_names, desc="Processing channels"):
        id_chan = epochs.ch_names.index(chan)
        for id_trial in range(len(epochs)):
            maskPreOffset, maskOffset, maskPostOffset = computeTimeMasks(
                epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset']
            )
            offsetTimes.append([
                chan, id_trial,
                epochs.times[maskOffset].min(),
                epochs.times[maskOffset].max()
            ])
    
    df = pd.DataFrame(offsetTimes, columns=['chan', 'trial', 'toffsetmin', 'toffsetmax'])
    df.to_csv(Path(experiment_dir) / '2.detrend' / 'offsetTimes_df.csv', index=False)

    mean_offset = df['toffsetmax'].mean()
    std_offset = df['toffsetmax'].std()
    hist_values, bin_edges = np.histogram(df['toffsetmax'], bins=15)
    mode_offset = bin_edges[np.argmax(hist_values)] + np.diff(bin_edges)[0]/2

    json_data['detrend_modeTimeWindowOffset'] = round(mode_offset, 3)
    json_data['detrend_meanTimeWindowOffset'] = round(mean_offset, 4)
    json_data['detrend_stdTimeWindowOffset'] = round(std_offset, 4)

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    if do_plot_variability:
        for name in tqdm(epochs.ch_names):
            plotTrialTepVariability(epochs, json_data, experiment_dir, sub, chanNAME=name, operator=np.mean, save=True, parDir='preDetrend')
    
    # see results in 3.trials
    return json_data, df

def computeTimeMasks(epochs, chan, trial, json_data, do_plot=False, offset=0.20, plot_path=None, plot_title=None):
    import numpy as np
    t=epochs.times
    n=t.size
    data=epochs.get_data()
    sigAll=data[trial,chan,:] if data.ndim==3 else data[chan,:]
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
    return maskPreOffset,maskOffset,maskPostOffset


def run_detrend_pipeline(epochs, json_data, sub, experiment_dir, do_plot_variability=True):

    # === CASE 1: windowed detrend attivo ===
    if json_data['do_detrend']:
        print(f"I am doing {json_data['detrend_type']} detrend")
        print('###')
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)

            epochs_detrended, mse_detrend, max_order_pre_list = computeDetrend_v6(
                epochs,
                json_data, experiment_dir, sub,
                detrendMode=json_data['detrend_type'],
                fitConstraint=json_data['detrend_fitConstraint'],
                typeOffsetRise=json_data['detrend_typeOffsetRise'],
                typeOffsetDecay=json_data['detrend_typeOffsetDecay'],
                correctMode=json_data['detrend_offsetCorrectionType'],
                oddSamples=json_data['detrend_offsetOddSamples'],
                offsetChans=json_data['offsetChans'],
                lag_correction=json_data['detrend_lag_correction'],
                doDetrendOnlyOffsetChans=json_data['do_detrend_onlyOffsetChans']
            )

        json_data[f'detrend_{json_data["detrend_type"]}_pars'] = [
            json_data['detrend_typeOffsetRise'],
            json_data['detrend_typeOffsetDecay']
        ]
        json_data['detrend_MSE'] = mse_detrend

        df_slopes_detrended = computeSlopes_v4(epochs_detrended, json_data, experiment_dir, sub)
        computeSlopesPlot(
            df_slopes_detrended,
            json_data, experiment_dir, sub,
            saveNote=f'ALL-DET_fit{json_data["detrend_fitConstraint"]}',
            subPath='2.detrend',
            sharex=True
        )
        detrendedEpochs = epochs_detrended
        if json_data['sourceData']!='SIMS':
            detrendedEpochs, json_data = notch_filter_offset_chans(detrendedEpochs, json_data)
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'{post_label}', subPath='2.detrend', show=False)
        else:
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'{post_label}', subPath='2.detrend', show=False)

        
    # === CASE 2: detrend disattivato ===
    else:
        print('I am not doing windowed detrend')
        print('###')
        json_data['detrend_polOrder_preOffset'] = np.NaN
        #json_data['detrend_polOrder_Offset'] = np.NaN
        #json_data['detrend_polOrder_postOffset'] = np.NaN
        json_data['detrend_offsetCorrectionType'] = np.NaN
        json_data['detrend_offsetOddSamples'] = np.NaN

        detrendedEpochs = epochs

        if json_data['detrend_overall']:
            def nonlinear_detrend(signal, order=3):
                times = np.arange(len(signal))
                poly_coeffs = np.polyfit(times, signal, order)
                trend = np.polyval(poly_coeffs, times)
                return signal - trend

            order = json_data['detrend_noWindowedOrder']
            detrendedEpochs = detrendedEpochs.apply_function(lambda x: nonlinear_detrend(x, order=order))
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'overallPolyOrder{order}', subPath='2.detrend', show=False)

    # === Salvataggio ===
    pkl_path = Path(experiment_dir) / '6.pkls' / f'{sub}_detrendedEpochs.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(detrendedEpochs, f)

    json_data['cn_detrendedEpochs'] = compute_condition_number_epochs_average(detrendedEpochs)

    # === Opzionale: plottaggio variabilità TEP per canale ===
    if do_plot_variability:
        for name in detrendedEpochs.ch_names:
            plotTrialTepVariability(detrendedEpochs, json_data, experiment_dir, sub, chanNAME=name, operator=np.mean, save=True, parDir='postDetrend')

    return detrendedEpochs, json_data


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

    for chan in tqdm(EPOCHS.ch_names):

        if doDetrendOnlyOffsetChans and chan not in offsetChans:
            continue  # salta questo canale, mantiene i dati originali

        id_chan = np.where(np.array(EPOCHS.ch_names) == chan)[0][0]
        for epoch_idx in range(data_detrended.shape[0]):
            tep = data_detrended[epoch_idx, id_chan, :].reshape(-1, 1)
            TEP = tep.copy()
            timeMask = computeTimeMasks(EPOCHS, id_chan, epoch_idx, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            
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
                    tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models)
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
                tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models)
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
    EPOCHS_DETRENDED = mne.EpochsArray(data_detrended, EPOCHS.info, tmin=EPOCHS.tmin)
    df_tabStat = pd.DataFrame(tabStat, columns=['chan', 'epoch_idx', 'mseA', 'mseB', 'mseC', 'OPTPARS_A', 'OPTPARS_B', 'OPTPARS_C'])
    tabStat_path = os.path.join(experiment_dir, '2.detrend', f'tabStatDetrend_{typeOffsetDecay}.csv')
    df_tabStat.to_csv(tabStat_path, index=False)
    print(f"[INFO] Salvato tabStat in: {tabStat_path}")

    return EPOCHS_DETRENDED, MSE, max_order_offset_list

