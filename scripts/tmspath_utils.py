# Import required general libraries
#
import matplotlib.pyplot as plt # library for plotting
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.pyplot.ion()
import glob  # File pattern matching
import mne # library for EEG data analysis
import numpy as np # library for manging numerical data 
from PyQt5.QtWidgets import QFileDialog # library for creating dialogue windows
import plotly.graph_objs as go # library for graphic objects
import scipy 
import seaborn as sns
import re
import pandas as pd
from mne.channels import make_standard_montage
from mne_icalabel import label_components
import os
from pyprep.prep_pipeline import PrepPipeline
from pyprep.find_noisy_channels import NoisyChannels
from datetime import datetime
from scipy.spatial.distance import euclidean
import os
import sys
import glob
import io
import json
import re
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.spatial.distance import euclidean
from scipy.stats import (
    expon, gamma, laplace, linregress, norm, poisson, rayleigh, t, uniform
)
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('Qt5Agg')  # GUI backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import plotly.graph_objs as go
import tkinter as tk
from tkinter import simpledialog
from PyQt5.QtWidgets import QFileDialog
import itertools
import mne
from mne import create_info, EvokedArray
from mne.channels import make_standard_montage
from mne.io import RawArray
import mne_connectivity
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
import tmspath_utils_adj


def import_modules():
    # ===============================
    # Import standard libraries
    # ===============================
    import os
    import sys
    import glob
    import io
    import json
    import re
    import pickle
    import random
    import warnings
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    from datetime import datetime
    from pathlib import Path
    import time
    
    # ===============================
    # Import scientific computing libraries
    # ===============================
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.interpolate import interp1d
    from scipy.signal import resample
    from scipy.spatial.distance import euclidean
    from scipy.stats import (
        expon, gamma, laplace, linregress, norm, poisson, rayleigh, t, uniform
    )
    
    # ===============================
    # Import time series analysis
    # ===============================
    from statsmodels.tsa.arima.model import ARIMA
    
    # ===============================
    # Import data visualization libraries
    # ===============================
    import matplotlib
    matplotlib.use('Qt5Agg')  # GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    from tabulate import tabulate
    from tqdm import tqdm
    from tqdm.notebook import tqdm as notebook_tqdm
    import plotly.graph_objs as go
    
    # ===============================
    # GUI interaction (Tkinter + PyQt5)
    # ===============================
    import tkinter as tk
    from tkinter import simpledialog
    from PyQt5.QtWidgets import QFileDialog
    import itertools
    
    # ===============================
    # EEG data analysis (MNE + connectivity)
    # ===============================
    import mne
    from mne import create_info, EvokedArray
    from mne.channels import make_standard_montage
    from mne.io import RawArray
    import mne_connectivity
    from mne_icalabel import label_components
    
    # ===============================
    # EEG preprocessing tools
    # ===============================
    from pyprep.find_noisy_channels import NoisyChannels
    from pyprep.prep_pipeline import PrepPipeline
    
    # ===============================
    # FOOOF (spectral fitting)
    # ===============================
    from fooof import FOOOF
    from fooof.sim.gen import gen_power_spectrum
    from fooof.sim.utils import set_random_seed
    from fooof.plts.spectra import plot_spectra
    from fooof.plts.annotate import plot_annotated_model
    
    # ===============================
    # User-defined functions
    # ===============================
    import tmspath_utils as tmsu
    
    # ===============================
    # Init: date, timer, json
    # ===============================
    now = datetime.now()
    date = now.strftime("%Y%m%d%H%M%S")
    start_time = time.time()  # ⏱️ start timer
    
    return date, start_time

def make_json_serializable(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            new_d[k] = v
        elif isinstance(v, (list, tuple)):
            new_d[k] = [make_json_serializable({'v': x})['v'] if isinstance(x, dict) else str(x) for x in v]
        else:
            # Converti tutto il resto in stringa
            new_d[k] = str(v)
    return new_d

def directorySetup(json_data):
    # === 5. Directory Setup ===
    extraNote = f'{json_data['detrend_typeOffsetRise']}_{json_data['detrend_typeOffsetDecay']}'
    sub = json_data['subject']
    #experiment_dir = os.path.join(json_data['mainDir'], f"{json_data['date']}_{json_data['emispheric_stimulation']}_{extraNote}")
    #experiment_dir = os.path.join(json_data['mainDir'], f"{json_data['detrend_fitConstraint']}_{json_data['detrend_offsetCorrectionType']}_{json_data['emispheric_stimulation']}_{extraNote}")
    fit_letter = str(json_data['detrend_fitConstraint'])[0]  # 'T' o 'F'
    offset_letter = str(json_data['detrend_offsetCorrectionType'])[0]  # Prima lettera della stringa
    date = json_data['date']
    experiment_dir = os.path.join(
        json_data['mainDir'],
        f"{date}_{fit_letter}_{offset_letter}_{extraNote}"
    )

    subdirs = [
        '1.basic',
        os.path.join('2.detrend'),
        os.path.join('2.detrend', 'examples'),
        os.path.join('3.trials', 'preDetrend'),
        os.path.join('3.trials', 'postDetrend'),
        '4.postICA',
        os.path.join('5.final'),
        os.path.join('5.final', 'FE'),
        '6.pkls',
        '7.FOOOF'
    ]
    os.makedirs(experiment_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
 
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    return json_data, experiment_dir, sub


def loadEDF(json_data, fileName):
    raw = mne.io.read_raw(fileName+'.EDF', preload=True)
    channel_types = raw.get_channel_types()
    raw.rename_channels({'EEG FPz': 'EEG Fpz'})
        
    if 'EMG' in raw.ch_names:
        print("Il canale 'EMG' è presente e verrà escluso.")
        raw.set_channel_types({'EMG': 'emg'})
        
        raw = raw.drop_channels('EMG')
        #print(raw.ch_names)
    else:
        print("Il canale 'EMG' non è presente.")
    
    rename_dict = {ch: ch.replace('EEG ', '') for ch in raw.ch_names}
    raw.rename_channels(rename_dict)
    print(raw.ch_names)
    
    return raw


def loadASCII(json_data, fileName):
    
    with open(fileName+'.asc', 'r') as f:
        lines = f.readlines()
    metaData=lines[0:11]
    
    header1, Patient_data, Trace_date, Start_seconds, Finish_seconds, SamplingRate_hz, potential_uV, _, channel_TR, channel_names, _ = metaData
    
    header1 = header1.strip('"')
    Patient_data = Patient_data.strip('"')
    Trace_date = Trace_date.strip('"')
    Start_seconds = Start_seconds.strip('"')
    Finish_seconds = Finish_seconds.strip('"')
    SamplingRate_hz = SamplingRate_hz.strip('"')
    potential_uV = potential_uV.strip('"')
    channel_TR = channel_TR.strip('"')
    channel_names = channel_names.strip('"')
    
    print(header1)
    print(Patient_data)
    print(Trace_date)
    print(Start_seconds)
    print(Finish_seconds)
    print(SamplingRate_hz)
    print(potential_uV)
    print(channel_TR)
    
    #print(channel_names)
    channel_names = channel_names.rstrip('\n')
    channel_names = [item.strip('"') for item in channel_names.split('", "')]
    #print(channel_names)
    channel_names = [name.replace('-RF', '') for name in channel_names]
    print(channel_names)
    
    with open(fileName+'_onlyData.asc', 'w') as f:
        f.writelines(lines[11::])
    
    df = pd.read_csv(fileName+'_onlyData.asc', 
                     sep='\t', 
                     decimal=',', 
                     names=channel_names, 
                     index_col=False,
                     #header=None,
                     #skiprows='None', 
                     dtype=np.float64)
    
    # check sampling rate
    for i in ['MK', 'EMG', 'TM']:
        if i in df.columns: print(f"Il canale {i} è presente nel DataFrame.")
        else: print(f"Il canale {i} non è presente nel DataFrame.")

    lenChannel = len(df[df.columns[0]].values)
    time = float(Finish_seconds[11:14])
    print(f'Esperimento ha durata time={time} con sr={lenChannel/time} e samples={lenChannel}')

    

    return df

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


def add_exp_artifact(epochs, json_data, experiment_dir, sub, tau_rise=0.005, tau_decay=0.1, gain=-3e-6*30, chans=None):
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    times = epochs.times  # in seconds
    zero_idx = np.where(times >= 0)[0][0]
    t_post = times[zero_idx:] - times[zero_idx]

    # Artefatto esponenziale charge/discharge
    artifact = gain * (1 - np.exp(-t_post / tau_rise)) * np.exp(-t_post / tau_decay)

    chan_idxs = np.arange(data.shape[1]) if chans is None else [
        epochs.ch_names.index(ch) for ch in chans
    ]

    for trial in data:
        for ch in chan_idxs:
            trial[ch, zero_idx:] += artifact

    epochs_artifacted = epochs.copy()
    epochs_artifacted._data = data  # attenzione: modifica diretta

    # Save filtered raw and ICA model
    pkl_raw_path = Path(experiment_dir) / "6.pkls" / f"{sub}_epochs_artifacted.pkl"
    with open(pkl_raw_path, 'wb') as f:
        pickle.dump(epochs_artifacted, f)
            
    return epochs_artifacted

 
def computeBasicSteps(raw, events, json_data, experiment_dir, sub, 
                      FIGSIZE=(13, 6), 
                      computeFOOOF=True,
                     ):

    if json_data['do_pulseArtifactRej']:
        print(f"🔧 [{sub}] Step 1: Rimozione artefatto TMS + PSD")
        raw = remove_tms_artifact_and_plot_psd(
            raw=raw,
            events=events,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
            do_plot=False,
            ica_continuum=json_data['do_ica_continuum'],
        )

    if json_data['do_filter_and_plot_raw']:
        print(f"🔧 [{sub}] Step 2: Filtro broad-band e notch")
        raw = filter_and_plot_raw(
            raw=raw,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
        )

    if json_data['do_clean_trials_channels']:
        print(f"🔧 [{sub}] Step 3: Pulizia epoche e canali artefattati")
        temp_epochs, json_data = clean_trials_channels(
            raw=raw,
            events=events,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub
        )

    if json_data['do_prepare_epochs']:
        print(f"🔧 [{sub}] Step 4: Creazione oggetto Epochs finale")
        epochs, json_data = prepare_epochs(raw, events, temp_epochs, json_data, experiment_dir, sub)

    else:
        epochs=raw
        temp_epochs =raw
    print(f"✅ [{sub}] Completato. Informazioni finali su Epochs:")
    print(epochs.info)

    if computeFOOOF:
        print(f"🔧 [{sub}] Step 5: FOOOF computation")
        df = extract_psd_features(epochs, 'preDetrend', experiment_dir, json_data)
    
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    if json_data['do_artifact']:
        epochs = add_exp_artifact(epochs,json_data, experiment_dir, sub, 
                                  tau_rise=json_data['do_artifact_rise'], 
                                  tau_decay=json_data['do_artifact_decay'], 
                                  gain=json_data['do_artifact_gain'], 
                                  chans=json_data['do_artifact_chans'])
        basicPlots(epochs, 
                   json_data, experiment_dir, 
                   sub, key='epochs_artifacted', subPath='2.Detrend', show=False)


    detrendedEpochs, json_data = computeDetrendSteps(epochs, 
                                            json_data, experiment_dir, sub, 
                                            computeFOOOF=computeFOOOF)

    return raw, epochs, detrendedEpochs, temp_epochs, json_data

def load_and_prepare_raw_data(fileName, json_data, experiment_dir, sub):
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import mne
    from tmspath_utils import loadASCII, loadEDF  # definizioni esterne

    json_data, experiment_dir, sub = directorySetup(json_data)

    if 'sourceData' not in json_data:
        raise KeyError("⚠️ 'sourceData' non è definito in json_data.")

    source = json_data['sourceData']
    dataType = json_data['dataType']

    json_data['pulse_artifact_rej_smoothingvalue'] = 0.002
    saveNote = ''
    raw, events = None, None

    def save_layout_and_metadata(raw, note):
        # Salva layout
        fig = raw.plot_sensors(show_names=True)
        layout_path = Path(experiment_dir) / '1.basic' / f'{sub}_{note}_scalplayout.png'
        fig.savefig(layout_path)
        plt.close(fig)
        # Salva parametri
        with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as txt_file:
            for key, value in sorted(json_data.items()):
                txt_file.write(f'{key}: {value}\n')

    # === CASO SIMS ===
    if source == 'SIMS':
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002  # not used in sims
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008  # not used in sims
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max'] # not used in sims

        fileName = f"{json_data['mainDir']}\\{json_data['subject']}.fif"
        epochs = mne.read_epochs(fileName, preload=True)
        basicPlots(epochs, json_data, experiment_dir, sub, key='epochsOK', subPath='1.basic')
        with open(Path(experiment_dir) / '6.pkls' / f'{sub}_epochsOK.pkl', 'wb') as f:
            pickle.dump(epochs, f)
        json_data['sfreq'] = epochs.info['sfreq']
        json_data['r_sfreq'] = 512
        raw = epochs
        events = raw.events
        json_data['TEP_ID_events'] = 'no_events'
        save_layout_and_metadata(raw, 'no_events')
        return raw, events, json_data

    # === CASO MAYER ===
    if source == 'MAYER':
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}
            raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']

            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    # === CASO CHALFONT ===
    if 'Chalfont' in source:
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002 * 1.5
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008 * 1.5
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            try:
                raw.set_montage('easycap-M1', verbose=True)
            except ValueError as e:
                if 'channel positions not present' in str(e):
                    raw.set_channel_types({'65': 'misc', '66': 'misc'})
                    raw.set_montage('easycap-M1', on_missing='ignore', verbose=True)
                else:
                    raise
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1015
            events = events[events[:, 2] == TMScode]
            saveNote = 'NGH_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    # === CASO UNIMI ===
    if 'UNIMI' in source:
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            raw.set_montage('easycap-M1', verbose=True)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1128
            events = events[events[:, 2] == TMScode]
            saveNote = 'MI_events'
            if Path(fileName).stem == 'prova_Betta_0002':
                shift_sec = - ((0.0001 * 4) + 0.002)
                shift_samples = int(shift_sec * raw.info['sfreq'])
                events[:, 0] += shift_samples
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
            
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}
            raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']

            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    raise ValueError(f"⚠️ Origine dati '{source}' non riconosciuta o mal configurata.")

def run_ica_continuum_pipeline(raw, events, json_data, experiment_dir, sub):
    from pathlib import Path
    import pickle

    # Set soglia dinamica
    do_run = json_data.get('do_ica_continuum', False)
    json_data['ica_continuum_tr'] = 30 if do_run else 10000
    ica_threshold_uv = json_data['ica_continuum_tr']

    if do_run:
        print(f"⚙️ Running ICA on continuous raw with threshold {ica_threshold_uv} µV...")
        raw_ica, ica_model = run_ica_artist_tms_events(
            raw.copy(),
            events,
            ext_threshold_uv=ica_threshold_uv,
            manualCheck=False,
            demean_between_events=True
        )

        # Save filtered raw and ICA model
        pkl_raw_path = Path(experiment_dir) / "6.pkls" / f"{sub}_raw_ICA_continuum.pkl"
        with open(pkl_raw_path, 'wb') as f:
            pickle.dump(raw_ica, f)

        pkl_ica_path = Path(experiment_dir) / "6.pkls" / f"{sub}_ica_model_continuum.pkl"
        with open(pkl_ica_path, 'wb') as f:
            pickle.dump(ica_model, f)

        # Optional epoching post-ICA for inspection
        epochs_cont = mne.Epochs(
            raw_ica,
            events,
            tmin=json_data['epochs_timewindow_min'],
            tmax=json_data['epochs_timewindow_max'],
            detrend=0,
            preload=True
        )
        epochs_cont = epochs_cont.pick('eeg').set_eeg_reference('average')

        # Save basic plots
        basicPlots(
            epochs_cont,
            key=f'epochs_continumm_tr{ica_threshold_uv}',
            subPath='1.basic'
        )

        print("✅ ICA on continuum complete.")
        return raw_ica, ica_model

    else:
        print("⏭️ ICA on continuum skipped.")
        return raw, None
    
import matplotlib.pyplot as plt
from pathlib import Path

def remove_tms_artifact_and_plot_psd(raw, events, json_data, experiment_dir, sub, figsize=(10, 6), subPath='1.basic', do_plot=False, ica_continuum=False):
    from tmspath_utils_adj import tms_pulse_removal_init

    psd_dir = Path(experiment_dir) / subPath
    psd_dir.mkdir(parents=True, exist_ok=True)

    # === PSD PRE-removal ===
    print("📉 Plotting PSD before pulse removal...")
    fig = raw.plot_psd(
        fmin=json_data['l_freq'],
        fmax=json_data['broad_band_h_freq'],
        xscale='log',
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(psd_dir / '1.psdrawnoPulseRemovalnoBroadBandnoNotch.png')
    plt.close(fig)

    # === Optional: plot raw signal to inspect pulse
    if do_plot:
        raw.plot(n_channels=raw.info['nchan'])

    # === Pulse Artifact Removal ===
    print("⚡ Removing TMS pulse artifact...")
    raw = tms_pulse_removal_init(
        raw=raw,
        sfreq=raw.info['sfreq'],
        events_sample=events[:, 0],
        window=(json_data['pulse_artifact_rej_timewindow_min'],
                json_data['pulse_artifact_rej_timewindow_max']),
        smooth_window=(-json_data['pulse_artifact_rej_smoothingvalue'],
                       json_data['pulse_artifact_rej_smoothingvalue']),
        span=2
    )

    # === PSD POST-removal ===
    print("📉 Plotting PSD after pulse removal...")
    fig = raw.plot_psd(
        fmin=json_data['l_freq'],
        fmax=json_data['broad_band_h_freq'],
        xscale='log',
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(psd_dir / '2.psdrawwithPulseRemovalnoBroadBandnoNotch.png')
    plt.close(fig)

    # === Optional: plot raw after correction
    if do_plot:
        raw.plot(n_channels=raw.info['nchan'], scalings={'eeg': 50e-6})

    if ica_continuum:
        print('do ica continuum')
        raw, ica_model = run_ica_continuum_pipeline(
                                                raw=raw,
                                                events=events,
                                                json_data=json_data,
                                                experiment_dir=experiment_dir,
                                                sub=sub
                                                )

    return raw

import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def filter_and_plot_raw(raw, json_data, experiment_dir, sub, figsize=(10, 6), subPath='1.basic'):
    print('DO-BROADBAND------------------------------------')

    raw = raw.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['broad_band_h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    # === PSD after broadband filtering
    fig = raw.plot_psd(fmin=json_data['l_freq'], fmax=json_data['broad_band_h_freq'], xscale='log', show=False)
    fig.set_size_inches(figsize)
    psd_path = Path(experiment_dir) / subPath / '3.psdrawwithPulseRemovalwithBroadBand_noNotch.png'
    fig.savefig(psd_path)
    plt.close(fig)

    print('DO-NOTCH------------------------------------')
    base_freq = json_data['powerline_freq']
    centers = [base_freq * i for i in range(1, 6)]
    print('DO-NOTCH------------------------------------')
    centers = [json_data['powerline_freq'], json_data['powerline_freq']*2, json_data['powerline_freq']*3, json_data['powerline_freq']*4, json_data['powerline_freq']*5]
    raw = raw.notch_filter(freqs=centers)

    # === PSD after notch filtering
    fig = raw.plot_psd(fmin=json_data['l_freq'], fmax=json_data['broad_band_h_freq'], xscale='log', show=False)
    fig.set_size_inches(figsize)
    psd_path_notch = Path(experiment_dir) / subPath / '4.psdrawwithPulseRemovalwithBroadBandwithNotch.png'
    fig.savefig(psd_path_notch)
    plt.close(fig)

    # === Save raw object
    raw_pkl_path = Path(experiment_dir) / '6.pkls' / f'{sub}_raw.pkl'
    with open(raw_pkl_path, 'wb') as f:
        pickle.dump(raw, f)

    print(f"[INFO] Raw filtered and saved → {raw_pkl_path}")
    return raw


def clean_trials_channels(raw, events, json_data, experiment_dir, sub, seedChans=None):
    import mne, numpy as np, sys, io
    from pathlib import Path

    if seedChans is None:
        seedChans = json_data.get("seedChans", [])

    # === Filtering
    raw_copy = raw.copy()
    raw_copy.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    # === Epoching
    temp_epochs = mne.Epochs(
        raw_copy,
        events,
        tmin=json_data['epochs_timewindow_min'],
        tmax=json_data['epochs_timewindow_max'],
        detrend=None,
        preload=True
    )
    temp_epochs = temp_epochs.pick('eeg')
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # === Automatic detection: bad channels
    data = temp_epochs.get_data()
    chan_var = np.var(data, axis=(0, 2))
    thresh_low = np.percentile(chan_var, 5)
    thresh_high = np.percentile(chan_var, 95)
    bad_channels = [ch for ch, var in zip(temp_epochs.ch_names, chan_var) if var < thresh_low or var > thresh_high]
    bad_channels_non_seed = [ch for ch in bad_channels if ch not in seedChans]

    # === Automatic detection: bad trials
    trial_var = np.var(data, axis=(1, 2))
    t_low = np.percentile(trial_var, 5)
    t_high = np.percentile(trial_var, 95)
    bad_trials = np.where((trial_var < t_low) | (trial_var > t_high))[0]

    if not json_data['do_chan_trials_selection_automatic']:
        print("🖱️ Manual artifact rejection")
        buffer = io.StringIO()
        sys.stdout = buffer
        fig = temp_epochs.plot(butterfly=False, n_epochs=20, n_channels=raw.info['nchan'], block=True, use_opengl=True, scalings={'eog': 50e-6})
        sys.stdout = sys.__stdout__
        log_message = buffer.getvalue()
        save_bad_epochs_and_channels(log_message, json_data)
        print(log_message)
    else:
        print("🤖 Automatic artifact rejection")
        temp_epochs.info['bads'] = bad_channels_non_seed
        json_data['bad_trials'] = bad_trials.tolist()
        mask = ~np.isin(temp_epochs.selection, json_data['bad_trials'])
        trialsToTake = temp_epochs.selection[mask]
        temp_epochs = temp_epochs[trialsToTake]

    # === Aggiornamenti json
    json_data['trials_tot'] = len(temp_epochs.annotations)
    json_data['trials_selected'] = temp_epochs.events.shape[0]
    json_data['bad_channels'] = temp_epochs.info['bads']

    # === Final steps
    temp_epochs = temp_epochs.resample(sfreq=json_data['r_sfreq'])
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    return temp_epochs, json_data

def apply_notch_to_offsetChans(epochs, json_data, centers=[50], apply_to_all=False):
    from mne.filter import notch_filter

    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    data_filtered = data.copy()

    if apply_to_all:
        print(f"🔧 Applico filtro Notch *aggressivo* a TUTTI i canali | Frequenze: {centers} Hz")
        chan_idx = range(len(epochs.ch_names))
        json_data['notch_applied_chans'] = epochs.ch_names  # salva tutti
    else:
        chans = json_data.get('offsetChans', [])
        if not chans:
            print("⚠️ Nessun canale in 'offsetChans'. Nessun notch applicato.")
            return epochs, json_data
        chan_idx = [epochs.ch_names.index(ch) for ch in chans if ch in epochs.ch_names]
        print(f"🔧 Applico filtro Notch *aggressivo* a {len(chan_idx)} canali: {chans} | Frequenze: {centers} Hz")
        json_data['notch_applied_chans'] = chans

    # Applica filtro notch aggressivo solo sui canali selezionati
    data_filtered[:, chan_idx, :] = notch_filter(
        data[:, chan_idx, :],
        Fs=sfreq,
        freqs=centers,
        method='iir',
        iir_params=dict(
            ftype='butter',  # filtro Butterworth
            gpass=0.5,       # ripple massimo nel pass-band (dB)
            gstop=40,        # attenuazione minima nello stop-band (dB)
            order=12          # ordine elevato
        )
    )

    epochs._data = data_filtered
    return epochs, json_data

import numpy as np
import pickle
import warnings
from pathlib import Path

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

def notch_filter_offset_chans(epochs, json_data):
    """
    Applica notch filter IIR ai canali presenti in json_data['offsetChans']
    per rimuovere armoniche della powerline.

    Parametri:
    - epochs: mne.Epochs object
    - json_data: dict contenente 'powerline_freq' e 'offsetChans'

    Output:
    - epochs modificato (in-place)
    """

    from mne.filter import notch_filter

    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    centers = [
        json_data['powerline_freq'],
        json_data['powerline_freq'] * 2,
        json_data['powerline_freq'] * 3,
        json_data['powerline_freq'] * 4,
        json_data['powerline_freq'] * 5
    ]

    chans = json_data.get('offsetChans', [])
    chan_idx = [epochs.ch_names.index(ch) for ch in chans if ch in epochs.ch_names]

    if chan_idx:
        print(f"🔧 Applico notch IIR a canali offset {chans} su frequenze: {centers}")
        for f in centers:
            """
            data[:, chan_idx, :] = notch_filter(
                data[:, chan_idx, :],
                Fs=sfreq,
                freqs=f,
                method='iir',
                iir_params=dict(
                    ftype='butter',
                    gpass=0.5,    # ripple massimo nel pass-band (dB)
                    gstop=60,   # attenuazione minima nello stop-band (dB)
                    order=8     # ordine del filtro
                )
            )
            """
            data[:, chan_idx, :] = notch_filter(
                                data[:, chan_idx, :],
                                Fs=sfreq,
                                freqs=centers,            # tutti i notch insieme
                                method='fir',
                                notch_widths=2,         # più largo (es. ±1.5 Hz)
                                trans_bandwidth=2.0,      # zona di transizione
                            )
        epochs._data = data
    else:
        print("⚠️ Nessun canale offset trovato, nessun notch applicato.")

    return epochs, json_data

from scipy import signal
import numpy as np

def apply_notch_filter(data, fs, notch_freqs, quality_factor=30):
    """
    Applica un filtro notch alle frequenze specificate.

    Parameters:
    - data: array 1D (n_samples) o 2D (n_channels x n_samples)
    - fs: frequenza di campionamento (Hz)
    - notch_freqs: lista delle frequenze notch (Hz) da eliminare
    - quality_factor: Q-factor del filtro notch (maggiore = più stretto, default=30)

    Returns:
    - filtered_data: array filtrato della stessa forma di 'data'
    """
    filtered_data = np.copy(data)

    for f0 in notch_freqs:
        b, a = signal.iirnotch(f0, quality_factor, fs)
        filtered_data = signal.filtfilt(b, a, filtered_data, axis=-1)

    return filtered_data

def prepare_epochs(raw, events, temp_epochs, json_data, experiment_dir, sub):
    from pathlib import Path
    import pickle
    import mne

    epochs = mne.Epochs(
        raw,
        events,
        tmin=json_data['epochs_timewindow_min'], 
        tmax=json_data['epochs_timewindow_max'], 
        detrend=0,
        preload=True
    )
    epochs = epochs[temp_epochs.selection]
    epochs.info['bads'] = temp_epochs.info['bads']
    epochs = epochs.resample(json_data['r_sfreq'])
    epochs = epochs.pick('eeg')
    epochs = epochs.set_eeg_reference('average')
    basicPlots(epochs, json_data, experiment_dir, sub, key='epochs', subPath='1.basic')

    pkl_path = Path(experiment_dir) / '6.pkls' / f'{sub}_epochs.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(epochs, f)

    json_data['cn_epochs'] = compute_condition_number_epochs_average(epochs)

    # Ricarica opzionale (sanity check)
    with open(pkl_path, 'rb') as f:
        epochs = pickle.load(f)

    return epochs, json_data

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

from pathlib import Path
import numpy as np

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

def add_TEP_to_json(json_file, postICA_final):
    """
    Aggiunge le versioni TEP_3d, TEP_2d, TEP_1d al dizionario json_file esistente.

    Parametri:
        - json_file: dict già esistente con altri metadati
        - postICA_final: oggetto mne.Epochs
    """
    data_3d = postICA_final.get_data()  # (epochs, chans, times)

    json_file['TEP_3d'] = np.transpose(data_3d, (1, 2, 0)).tolist()     # (chan, time, trial)
    json_file['TEP_2d'] = data_3d.mean(axis=0).tolist()                 # (chan, time)
    json_file['TEP_1d'] = data_3d.mean(axis=0).mean(axis=0).tolist()   # (time,)

    print("[INFO] Aggiunte TEP_3d, TEP_2d, TEP_1d a json_file")
    return json_file

def ICAprocessing(file,
                  json_data, experiment_dir, sub,
                    autoReject=True,
                    manualCheck=True, 
                    computeFOOOF=False
      
                 ):
    
    autoReject=json_data['do_ica_automaticRej'],
    label_prob_threshold=json_data['do_label_prob_threshold'],
    threshold_percentile=json_data['do_ica_eigThresh'],
    
    # Stampa del riferimento scientifico sui TEP
    print("""
    📚 Riferimento TEP:
    TMS of the primary motor cortex (M1) evokes several peaks, described
    at approximately 15 (N15), 30 (P30), 45 (N45), 60 (P60), 100 (N100),
    and 180 (P180) milliseconds [28,32,35,36]. However, recently it has
    been shown that later peaks (>~80 ms) such as N100 and P180 may
    be contaminated by sensory-evoked responses (see Sections 3.5, 4.2.3, and 4.2.4),
    while very early peaks, such as the N15, can be contaminated by cranial muscle responses (see Section 4.2.2).
    TEPs are detectable up to 400–500 ms around the stimulation area as well as in distant inter-connected brain areas [4,32,37].
    
    Link (materiale di riferimento): https://drive.google.com/drive/folders/1jqUiGEBVzhRdN7iISeIFFDoSeJvb4YqM
    """)
    
    # Log dei parametri di ICA
    print(f"⚙️ ICA eigThresh = {json_data['do_ica_eigThresh']}, label_prob_threshold = {json_data['do_label_prob_threshold']}")


    if isinstance(file, str) and file.endswith('.pkl'):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File non trovato: {file}")
        with open(file, 'rb') as f:
            detrendedEpochs = pickle.load(f)
        print(f"[INFO] Oggetto caricato da: {file}")
    else:
        detrendedEpochs = file
        print(f"[INFO] Oggetto passato direttamente")


    ######################################################################################    
    postICA_raw, ica_model = run_ica_filtering_v3(detrendedEpochs, json_data, experiment_dir, sub,
                                                  autoReject=autoReject,
                                                  manualCheck=json_data['do_ica_manualCheck'], 
                                                  label_prob_threshold=json_data['do_label_prob_threshold'],
                                                  threshold_percentile=json_data['do_ica_eigThresh'])
    if computeFOOOF:
        print('Computing postICA raw FOOOF')
        df = extract_psd_features(postICA_raw, 'postICA_raw', experiment_dir, json_data)
        
    with open(f'{experiment_dir}\\6.pkls\\{sub}_postICA_raw.pkl', 'wb') as f:
        pickle.dump(postICA_raw, f)
        
    json_data['ICA_includedComponents_tot'] = ica_model.n_components_ - len(ica_model.exclude)
    json_data['ICA_components_tot'] = ica_model.n_components_
    json_data['ICA_excludedComponents'] = ica_model.exclude

    #####################################################################################
    # filtering in narrow band 
    # resampling to raw sr
    # bad chans interpolations
    postICA_final = postICAsteps(postICA_raw, json_data, experiment_dir, sub)
    basicPlots(postICA_final, json_data, experiment_dir, sub, key='postICA_final', subPath='4.postICA', show=False)
    condition_number_evoked = compute_condition_number_epochs_average(postICA_final)
    json_data['cn_postICA_final'] = condition_number_evoked
    # Salva parametri
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)
        
    with open(f'{experiment_dir}\\6.pkls\\{sub}_postICA_final.pkl', 'wb') as f:
        pickle.dump(postICA_final, f)

    #####################################################################################
    do_run=False
    if do_run:
        postICA_final.plot();

    do_run=True
    if do_run:
        plot_topomap(postICA_final, 
                     json_data, experiment_dir, sub,
                     subDir='4.postICA', saveNote=f'postICA')
        json_data['feats_smfp'] = plot_gmfp(postICA_final, json_data, experiment_dir, sub, FEAT=json_data['seedChans'])

    do_run=True
    if do_run:
        plt.figure(figsize=(10, 5))
        plt.plot(postICA_final.times*1e3, np.mean(postICA_final.get_data(), axis=0).T*1e6, c='k', linewidth=0.2)
        #plt.axvline(x=0+10, c='r', linestyle='--', label='stim')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (V)')
        plt.xlim(-100, 400)
        plt.grid(False)
        plt.savefig(f'{experiment_dir}\\5.final\\tep_comparison.png')
        plt.close()
    
    #plot_ersp(postICA_final, channel=json_data['seedChans'], subDir='4.postICA', saveNote='postICA')
    #####################################################################################
    if computeFOOOF:
        print('Computing postICA final FOOOF')
        df = extract_psd_features(postICA_final, 'postICA_final', experiment_dir, json_data)

    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)
    
    return postICA_final, json_data


def computeFeatExtraction(postICA_final, json_data, experiment_dir, sub):
    # 1
    times = postICA_final.times
    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    json_data['feats_step'] = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    reduced_data = json_data['feats_step']
    reduced_data, times_filtered = reduced_data[np.where(times > 0)[0]], times[np.where(times > 0)[0]]
    data = postICA_final.get_data()
    tep_integral = scipy.integrate.trapezoid(np.abs(reduced_data), times_filtered)
    tep_energy =  np.mean(reduced_data ** 2)
    # complexity metrics
    #sampen = ant.sample_entropy(reduced_data)
    #perm_entropy = ant.perm_entropy(reduced_data)
    # save feats
    #json_data['feat_step_sampleEntropy'] = sampen
    #json_data['feat_step_permEntropy'] = perm_entropy
    json_data['feat_step_energy'] = tep_energy
    json_data['feat_step_integral'] = tep_integral
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_parsjsontxt', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')

    # 2
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", #annot=False, fmt=".2f",
                yticklabels=postICA_final.ch_names, 
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1, 
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_corrMatrix.png')
    plt.close()
    
    # 3    
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix[seed_indices,:], 
                 #mask=mask, 
                cmap="coolwarm", #annot=False, fmt=".2f",
                yticklabels=postICA_final.ch_names, 
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1, 
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_corrMatrix_seed.png')
    plt.close()

    # 4
    plt.figure(figsize=(FIGSIZE))
    plt.plot(postICA_final.times, postICA_final.average().get_data()[seed_indices, :].T , label=f'seed chans {json_data['seedChans']}', c='r')
    plt.plot(postICA_final.times, json_data['feats_step'], label='average TEP', linewidth=10)
    plt.xlabel("Time (ms)")
    plt.ylabel('Amplitude (µV) Seed TEP')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Average Seed TEP')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_STEP.png')
    plt.close()

    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    signals = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    len(signals.shape)

    # 5
    do_run=True
    if do_run:
        selected_peaks = selectTEPfeat(postICA_final,  json_data, experiment_dir, sub, seed=json_data['seedChans'])
        selected_peaks
    
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)

    return json_data

def computeFeatExtraction_v2(postICA_final, json_data, experiment_dir, sub):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.integrate
    import antropy as ant
    from fooof import FOOOF
    from pathlib import Path

    # 1 - Seed TEP features
    times = postICA_final.times
    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    json_data['feats_step'] = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    reduced_data = json_data['feats_step']
    reduced_data, times_filtered = reduced_data[np.where(times > 0)[0]], times[np.where(times > 0)[0]]
    data = postICA_final.get_data()
    tep_integral = scipy.integrate.trapezoid(np.abs(reduced_data), times_filtered)
    tep_energy = np.mean(reduced_data ** 2)

    # Complexity metrics
    sampen = ant.sample_entropy(reduced_data)
    perm_entropy = ant.perm_entropy(reduced_data)

    # FOOOF metrics
    fs = 1.0 / (times[1] - times[0])
    from scipy.signal import welch
    f, psd = welch(reduced_data, fs=fs, nperseg=128)
    fm = FOOOF()
    fm.fit(f, psd)
    fooof_offset, fooof_exponent = fm.get_params('aperiodic_params')

    # Save features
    json_data['feat_step_energy'] = tep_energy
    json_data['feat_step_integral'] = tep_integral
    json_data['feat_step_sampleEntropy'] = sampen
    json_data['feat_step_permEntropy'] = perm_entropy
    json_data['feat_step_fooofOffset'] = fooof_offset
    json_data['feat_step_fooofExponent'] = fooof_exponent

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')

    # 2 - Correlation matrix full
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm",
                yticklabels=postICA_final.ch_names,
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1,
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}/5.final/FE/{sub}_FE_corrMatrix.png')
    plt.close()

    # 3 - Correlation matrix seed vs all
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix[seed_indices, :],
                cmap="coolwarm",
                yticklabels=np.array(postICA_final.ch_names)[seed_indices],
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1,
                linewidths=0.5, cbar=True)
    plt.title("Seed vs All Correlation Matrix")
    plt.savefig(f'{experiment_dir}/5.final/FE/{sub}_FE_corrMatrix_seed.png')
    plt.close()

    # 4 - Plot TEP signal
    plt.figure(figsize=(10, 5))
    plt.plot(postICA_final.times, postICA_final.average().get_data()[seed_indices, :].T, label=f"seed chans {json_data['seedChans']}", c='r')
    plt.plot(postICA_final.times, json_data['feats_step'], label='average TEP', linewidth=3)
    plt.xlabel("Time (ms)")
    plt.ylabel('Amplitude (µV) Seed TEP')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Average Seed TEP')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/5.final/FE/{sub}_FE_STEP.png')
    plt.close()

    # 5 - Peak selection
    do_run = True
    if do_run:
        selected_peaks = selectTEPfeat(postICA_final, json_data, experiment_dir, sub, seed=json_data['seedChans'])

    from scipy.signal import hilbert
    # 6 - Phase Locking Value (PLV)
    def compute_PLV(data, seed_indices):
        analytic_signal = hilbert(data, axis=-1)
        phase = np.angle(analytic_signal)
    
        seed_phase = np.mean(phase[seed_indices, :], axis=0)
        plv_values = []
    
        for i in range(data.shape[0]):
            phase_diff = seed_phase - phase[i, :]
            plv = np.abs(np.sum(np.exp(1j * phase_diff))) / len(phase_diff)
            plv_values.append(plv)
    
        return np.array(plv_values)
    
    # Compute PLV on average TEP
    avg_data = postICA_final.average().get_data()
    plv_seed_all = compute_PLV(avg_data, seed_indices)
    json_data['feat_step_meanPLV_seed'] = float(np.mean(plv_seed_all))
    json_data['feat_step_maxPLV_seed'] = float(np.max(plv_seed_all))

    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)

    return json_data


import matplotlib.pyplot as plt
import numpy as np
import os

def plot_TEP_1d_with_shading(epochs, output_path):
    """
    Plot della TEP media su canali e trial, con shading ± std, e salvataggio PNG.

    Parametri:
        - epochs: oggetto mne.Epochs (es. postICA_final)
        - output_path: percorso completo al file PNG da salvare
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    times_ms = epochs.times * 1e3
    data_uV = data * 1e6  # conversione in microvolt

    mean_1d = data_uV.mean(axis=(0, 1))  # media su epoche e canali
    std_1d = data_uV.std(axis=(0, 1))    # std su epoche e canali

    plt.figure(figsize=(10, 5))
    plt.plot(times_ms, mean_1d, color='black', linewidth=1.5, label='Mean TEP')
    plt.fill_between(times_ms, mean_1d - std_1d, mean_1d + std_1d, 
                     color='gray', alpha=0.3, label='±1 STD')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.xlim(-100, 400)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot TEP_1d salvato in: {output_path}")


def saveLoadTestFinal(postICA_final, json_data, experiment_dir, sub, start_time):
    
    filePKL=f'{experiment_dir}\\6.pkls\\{sub}_notebookState.pkl'
    
    # Function to filter out non-pickleable objects
    def is_pickleable(obj):
        try:
            pickle.dumps(obj)
            return True
        except:
            return False
    
    # Save the current state of all pickleable variables in the notebook
    state = {name: val for name, val in globals().items() if is_pickleable(val) and not name.startswith('_')}
    with open(filePKL, 'wb') as f:
        pickle.dump(state, f)
    
    with open(filePKL, 'rb') as f:
        data = pickle.load(f)
    
    for i in data.keys():
        print(i)
    
    data = {}
    # 'epochs'
    # 'epochs_detrended'
    # , 'epochs_ica', 
    for file in ['postICA_final']:
        filePKL=f'{experiment_dir}\\6.pkls\\{sub}_{file}.pkl'
        with open(filePKL, 'rb') as f:
            data[file] = pickle.load(f)
            basicPlots(data[file], json_data, experiment_dir, sub, key=file, subPath='5.final', show=json_data['showPlotsEnd'])
            plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(data['postICA_final'].times*1e3, np.mean(data['postICA_final'].get_data(), axis=0).T*1e6, c='k', linewidth=0.2)
    #plt.axvline(x=0+10, c='r', linestyle='--', label='stim')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.xlim(-100, 400)
    plt.grid(False)
    plt.savefig(f'{experiment_dir}\\5.final\\tep_comparison.png')
    plt.close()

    json_data = add_TEP_to_json(json_data, postICA_final)
    # Salva il file di parametri aggiornato con tempo incluso
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)
        
    plot_TEP_1d_with_shading(postICA_final, f'{experiment_dir}/5.final/tep_1D.png')

    # Calcolo tempo totale di esecuzione
    elapsed_time = time.time() - start_time
    json_data['finish_time'] = time.time()
    json_data['total_elapsed_time_sec'] = round(elapsed_time, 2)
    
    # Salva il file di parametri aggiornato con tempo incluso
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)

            
    print(f"✅ Tempo totale di esecuzione: {elapsed_time:.2f} secondi")

    return json_data


def selectTEPfeat(EPOCH,  json_data, experiment_dir, sub, seed=['Cz', 'Fz']):
    
    times = EPOCH.times * 1000

    if seed==None:
        signals = EPOCH.average().get_data()
    else:
        seed_indices = [EPOCH.ch_names.index(chan) for chan in seed if chan in EPOCH.ch_names]
        signals = np.mean(EPOCH.average().get_data()[seed_indices, :], axis=0)

    
    ch_names = EPOCH.ch_names
    selected_peaks = []
    colors = itertools.cycle(plt.cm.get_cmap("tab10").colors)
    
    def find_nearest_channel(y_value):
        distances = [np.abs(y_value - np.mean(signals[i] * 1e6)) for i in range(len(ch_names))]
        min_distance = np.min(distances)
        if min_distance > np.std(signals) * 3:
            return np.nan
        return ch_names[np.argmin(distances)]

    def onclick(event):
        if event.xdata is None or event.ydata is None or event.dblclick or event.button != 1:
            return  # Ignora zoom, pan, doppi clic e tasti diversi dal sinistro
        
        root = tk.Tk()
        root.withdraw()
        peak_name = simpledialog.askstring("Peak Name", "Enter peak name (e.g., N15, P30):")
        if peak_name:
            nearest_channel = find_nearest_channel(event.ydata)
            peak_data = {
                "name": peak_name,
                "latency_ms": round(event.xdata, 2),
                "amplitude_uv": round(event.ydata, 2),
                "channel": nearest_channel if not pd.isna(nearest_channel) else "NaN"
            }
            selected_peaks.append(peak_data)
            ax.scatter(event.xdata, event.ydata, color='red', s=100, marker='o')
            fig.canvas.draw()

    root = tk.Tk()
    root.withdraw()

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(signals.shape)!=1:
        for signal, ch_name, color in zip(signals, ch_names, colors):
            ax.plot(times, signal * 1e6, label=ch_name, color=color, alpha=0.8, linewidth=5)
    else:
        ax.plot(times, signals * 1e6, label=f'average seed of {seed}', color='k', alpha=0.8, linewidth=5)

    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Click to Select Peaks")
    ax.legend()
    fig.canvas.mpl_connect("button_press_event", onclick)

    #save_button = tk.Button(root, text="Save Peaks", command=save_to_json)
    #save_button.pack()
    
    #plt.show()
    plt.close()

    json_data['feat_tep_manual'] = selected_peaks
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)


def set_plot_params(fontsize=16):
    plt.rcParams.update({
        'font.size': fontsize,            # Dimensione generale del font
        'axes.labelsize': fontsize,       # Dimensione dei label degli assi
        'axes.titlesize': fontsize,       # Dimensione del titolo degli assi
        'xtick.labelsize': fontsize,      # Dimensione dei tick dell'asse x
        'ytick.labelsize': fontsize,      # Dimensione dei tick dell'asse y
        'legend.fontsize': fontsize*0.75,      # Dimensione della legenda
        'figure.titlesize': fontsize      # Dimensione del titolo della figura
    })

FIGSIZE=19, 11
set_plot_params(fontsize=22)


import numpy as np

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


import numpy as np

def compute_condition_number_epochs_average(epochs):
    """
    Calcola il numero di condizionamento della matrice dei dati EEG
    dell'oggetto `epochs.average()`, ovvero la media degli Epochs.

    Restituisce:
    - Il numero di condizionamento della matrice EEG media.
    """
    evoked_data = epochs.average().data  # Matrice (n_channels, n_times)

    # Calcolo del numero di condizionamento usando la SVD
    condition_number = np.linalg.cond(evoked_data)

    return condition_number

def run_ica_filtering_v3(EPOCHS, json_data, experiment_dir, sub,
                         n_components=None, manualCheck=True,
                          autoReject=True, label_prob_threshold=0,
                          threshold_percentile=75,
                          subPath='4.postICA',
                          ):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    # ICA decomposition
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    # 1. Auto-tagging delle componenti
    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other']
    auto_excluded = []
    low_eigen_excluded = []

    # 2. Esclusione automatica (opzionale)
    if autoReject:
        for i, label in enumerate(labels):
            probs = np.array(ic_labels['y_pred_proba'][i], ndmin=1)
            max_prob = probs.max()
            if label in artifact_tags and max_prob >= label_prob_threshold:
                print(f"❌ IC {i}: {label} (prob: {max_prob:.2f}) – escluso")
                auto_excluded.append(i)
            else:
                print(f"✅ IC {i}: {label} (prob: {max_prob:.2f}) – mantenuto")

        print(f"[Auto-tagging] Componenti escluse per label ICLabel: {auto_excluded}")

        # 3. Filtro per soglia su autovalori
        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)
        low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()
        print(f"[Autovalori] Componenti escluse (eigenvalue < {threshold:.4f}): {low_eigen_excluded}")
    else:
        print("🚫 Esclusione automatica disattivata.")
        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)

    excluded_components = sorted(set(auto_excluded + low_eigen_excluded)) if autoReject else []
    ica.exclude = excluded_components

    # 4. Applica ICA per rimuovere componenti escluse
    postICA_raw = ica.apply(EPOCHS.copy())

    # 5. Salvataggio grafici riassuntivi
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    if excluded_components:
        fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
        fig1.savefig(os.path.join(experiment_dir, subPath, f'{sub}_excluded_ICAs.png'))
        plt.close(fig1)

    if remaining_components:
        fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
        fig2.savefig(os.path.join(experiment_dir, subPath, f'{sub}_included_ICAs.png'))
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(os.path.join(experiment_dir, subPath, f'{sub}_eigenvalueDist.png'))
    plt.close(fig)

    # === SALVA COMPONENTI ICA SINGOLE CON TAG ===
    components_dir = os.path.join(experiment_dir, subPath, 'components')
    os.makedirs(components_dir, exist_ok=True)

    print(f"[INFO] Salvataggio componenti ICA in: {components_dir}")

    for idx in range(ica.n_components_):
        tag = labels[idx] if labels is not None else 'Unknown'
        tag_clean = tag.replace('/', '_').replace(' ', '')
        fig = ica.plot_components(picks=idx, show=False)
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                fname = os.path.join(components_dir, f"component_{idx}_{tag_clean}_view{i}.png")
                f.savefig(fname, dpi=150)
                plt.close(f)
        else:
            fname = os.path.join(components_dir, f"component_{idx}_{tag_clean}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    # 6. Controllo manuale finale (opzionale)
    if manualCheck:
        try:
            import tmspath_utils_adj
            ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
            postICA_clean = ica.apply(postICA_raw.copy())
        except ImportError:
            print("⚠️ tmspath_utils_adj non disponibile. Salto ispezione manuale.")
            postICA_clean = postICA_raw
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica



def postICAsteps(postICA_raw, json_data, experiment_dir, sub):
    postICA_final = postICA_raw.copy().filter(l_freq=json_data['l_freq'],
                                              h_freq=json_data['h_freq']*0.90, # *2
                                              method = 'fir', #'iir', #, # NEW
                                              #iir_params = dict(order=3, ftype='butter', phase='zero-double',btype='bandpass'), # TEST IF WAS THIS FOR CIRCLING->YES
                                              verbose=True)
    
    newrate=json_data['sfreq'] 
    postICA_final=postICA_final.resample(sfreq = newrate)
    postICA_final=postICA_final.interpolate_bads()
    
    postICA_final = postICA_final.pick('eeg') # Pick only eeg channels
    times = postICA_final.times
    ch_names = postICA_final.ch_names
    
    with open(f'{experiment_dir}\\6.pkls\\{sub}_postICA_final.pkl', 'wb') as f:
        pickle.dump(postICA_final, f)
        json_data['mneEpochArrayFinal']=postICA_final # NEW
        
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
        json.dump(json_data_clean, json_file, indent=4)

    return postICA_final

def plot_ersp(postICA_final, channel=['Cz', 'Fx'], subDir='4.postICA', saveNote='postICA'):
    do_run=1
    if do_run:
        # plot of event-related spectral perturbations 
        # for a single channel specified by the user
        cols = 8 if len(raw.ch_names)==31 else 6
        fig, ax = plt.subplots(4, cols, figsize=(13, 11))
        ax = ax.flatten()
        fig.tight_layout()
        plt.subplots_adjust(wspace=1.25, hspace=1)
        for i, FEAT in enumerate(raw.ch_names):
            print(FEAT)
            ersp, freqs = tmspath_utils_adj.plot_ersp(postICA_final, FEAT, n_cycle=2, show=False, ax=ax[i])
        fig.savefig(f'{experiment_dir}\\{subDir}\\{sub}_ERSP_{saveNote}.png')

def plot_topomap(postICA_final, 
                 json_data, experiment_dir, sub,
                 subDir='4.postICA',saveNote='postICA'):
    do_run=1
    if do_run:
        # Concatenare gli intervalli
        times = [postICA_final.times.min(), 0, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.200, 0.299, postICA_final.times.max()]
        fig=postICA_final.average().plot_topomap(times, show=False,
                             ch_type='eeg', 
                         #mask=mask, mask_params=mask_params, 
                         #image_interp="linear", 
                         contours=10)
        
        fig.savefig(f'{experiment_dir}\\{subDir}\\{sub}_scalpmaptime_{saveNote}.png')
        plt.close()

        times = np.linspace(postICA_final.times.min(), postICA_final.times.max())
        evoked = postICA_final.average()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Assicuriamoci che ci sia un solo asse
        fig.suptitle("Scalp Topomap over Time")
        def update(frame):
            ax.clear()  # Puliamo l'asse per il nuovo tempo
            evoked.plot_topomap(times=[frame], ch_type='eeg', contours=10, axes=ax, show=False, colorbar=False)
            ax.set_title(f"Time: {frame:.3f} s")
        ani = FuncAnimation(fig, update, frames=times, repeat=False)
        output_path = os.path.join(experiment_dir, f"{subDir}/{sub}_scalpmaptime_{saveNote}.gif")
        ani.save(output_path, writer='ffmpeg', fps=2)  # Cambia fps per regolare la velocità
        print(f"Video salvato in: {output_path}")
        plt.close()

def plot_gmfp(postICA_final, json_data, experiment_dir, sub, FEAT=['Cz', 'Fx']):
    do_run=1
    if do_run:
        # plot of either 
        # GMFP time course (if channels='all') or 
        # LMFP (if channels = ['Fz', 'Cz',...]
        #FEAT = json_data['seedChans'] #'all'
        g_l_mfp = tmspath_utils_adj.plot_gmfp(postICA_final, channels=FEAT, show=False)
        json_data['feats_smfp'] = g_l_mfp
        # add saving
    return g_l_mfp


def find_outlier_channels_by_twindow_v2(df_slopes, threshold=3):
    """
    Identifica i canali che hanno almeno un trial con |Zslope| > threshold
    in ogni finestra temporale (id_twindow).
    
    Parametri:
    - df_slopes: DataFrame con colonne 'id_twindow', 'chan', 'Zslope'.
    - threshold: soglia in unità Z-score (default=3).
    
    Ritorna:
    - outlier_channels: dict con time window come chiave e lista di canali outlier come valore.
    - found_outlier: True se almeno un outlier è stato trovato.
    """
    outlier_channels = {}
    found_outlier = False

    for time_window in df_slopes['id_twindow'].unique():
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]
        outliers = []

        for chan in df_time_window['chan'].unique():
            zvals = df_time_window[df_time_window['chan'] == chan]['Zslope']
            if any(abs(zvals) > threshold):
                outliers.append(chan)

        if outliers:
            outlier_channels[time_window] = outliers
            found_outlier = True

    return outlier_channels, found_outlier


def find_outlier_channels_by_twindow(df_slopes, threshold=3):
    """
    Identifica i canali che hanno una media di ZSlope superiore a 3 sigma in ogni finestra temporale.
    
    Parametri:
    - df_slopes: DataFrame contenente le colonne 'id_twindow', 'chan' e 'Zslope'.
    - threshold: Soglia in deviazioni standard per considerare un canale anomalo (default=3).
    
    Ritorna:
    - Dizionario con finestre temporali come chiavi e liste di canali outlier come valori.
    - Booleano True se almeno una finestra temporale contiene outlier, altrimenti False.
    """
    # Dizionario per contenere i canali outlier per ogni finestra temporale
    outlier_channels = {}
    found_outlier = False

    # Loop su ogni finestra temporale unica
    for time_window in df_slopes['id_twindow'].unique():
        # Filtra i dati per la finestra temporale corrente
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]
        
        # Calcola la media e la deviazione standard di ZSlope per tutti i canali
        mean_zslope = df_time_window.groupby('chan')['Zslope'].mean()
        std_zslope = df_time_window.groupby('chan')['Zslope'].std()

        # Identifica i canali con valore medio di ZSlope superiore alla soglia di 3 sigma
        outliers = mean_zslope[mean_zslope > threshold * std_zslope].index.tolist()

        # Salva gli outlier se presenti
        if outliers:
            outlier_channels[time_window] = outliers
            found_outlier = True  # Almeno un outlier trovato

    return outlier_channels, found_outlier



def find_outlier_channels_by_twindow_v3(df_slopes, threshold=3):
    """
    Identifica i canali che hanno una media di ZSlope superiore alla media globale ± threshold * std 
    in ogni finestra temporale, e stampa lo Z-score medio per ogni canale outlier.
    
    Parametri:
    - df_slopes: DataFrame con colonne 'id_twindow', 'chan' e 'Zslope'.
    - threshold: Soglia in deviazioni standard per considerare un canale anomalo (default=3).
    
    Ritorna:
    - Dizionario con finestre temporali come chiavi e liste di canali outlier come valori.
    - Booleano True se almeno una finestra temporale contiene outlier, altrimenti False.
    """
    outlier_channels = {}
    found_outlier = False

    for time_window in df_slopes['id_twindow'].unique():
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]

        # Calcola la media Zslope per ciascun canale
        mean_zslope_per_chan = df_time_window.groupby('chan')['Zslope'].mean()

        # Calcola media e std globale tra canali
        global_mean = mean_zslope_per_chan.mean()
        global_std = mean_zslope_per_chan.std()

        # Calcola Z-score normalizzato per ciascun canale
        zscore = (mean_zslope_per_chan - global_mean) / global_std

        # Identifica outlier sopra o sotto soglia
        outliers = zscore[abs(zscore) > threshold]

        if not outliers.empty:
            outlier_channels[time_window] = outliers.index.tolist()
            found_outlier = True

            print(f"\n[Time window: {time_window}] Canali outlier (Z-score > ±{threshold}):")
            for chan in outliers.index:
                print(f" - Canale: {chan}, Z-score medio: {zscore[chan]:.2f}")

    return outlier_channels, found_outlier


def run_ica_filtering(EPOCHS, n_components=None, manualCheck=True, threshold_percentile=75, subPath='4.postICA', saveNote='postICA'):
    
    #ic_labels, ica = runICA(EPOCHS)    
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2  # Autovalori
    threshold = np.percentile(eigenvalues, threshold_percentile)
    excluded_components = np.where(eigenvalues <= threshold)[0].tolist()
    print(f"Componenti escluse (eigenvalue < {threshold:.4f}): {excluded_components}")
    ica.exclude = excluded_components
    postICA_raw = ica.apply(EPOCHS.copy())
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))
    fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
    fig1.savefig(f'{experiment_dir}\\{subPath}\\{sub}_excluded_ICAs_{saveNote}.png')
    plt.close()
    fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
    fig2.savefig(f'{experiment_dir}\\{subPath}\\{sub}_included_ICAs_{saveNote}.png')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]  # Indici sopra soglia
    below_threshold = np.where(eigenvalues < threshold)[0]   # Indici sotto soglia
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(f'{experiment_dir}\\{subPath}\\{sub}_eigenvalueDist_{saveNote}.png')
    plt.close()
    if manualCheck: 
        ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
        postICA_raw_bis = ica.apply(postICA_raw.copy())  
        postICA_clean = postICA_raw_bis
        ica.exclude = excluded_components

    if not manualCheck: 
        postICA_clean = postICA_raw

    return postICA_clean, ica


def run_ica_filtering_v2(EPOCHS, n_components=None, manualCheck=True, threshold_percentile=75,
                          subPath='4.postICA', saveNote='postICA', experiment_dir='.', sub='subject'):

    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(os.path.join(experiment_dir, subPath), exist_ok=True)

    # ICA decomposition
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    # 1. Auto-tagging delle componenti
    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other']
    auto_excluded = [i for i, label in enumerate(labels) if label in artifact_tags]

    print(f"[Auto-tagging] Componenti escluse per label ICLabel: {auto_excluded}")

    # 2. Filtro per soglia su autovalori
    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
    threshold = np.percentile(eigenvalues, threshold_percentile)
    low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()

    print(f"[Autovalori] Componenti escluse (eigenvalue < {threshold:.4f}): {low_eigen_excluded}")

    # 3. Unione esclusioni
    excluded_components = sorted(set(auto_excluded + low_eigen_excluded))
    ica.exclude = excluded_components

    # 4. Applica ICA per rimuovere componenti escluse
    postICA_raw = ica.apply(EPOCHS.copy())

    # 5. Salvataggio grafici
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
    fig1.savefig(os.path.join(experiment_dir, subPath, f'{sub}_excluded_ICAs_{saveNote}.png'))
    plt.close()

    fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
    fig2.savefig(os.path.join(experiment_dir, subPath, f'{sub}_included_ICAs_{saveNote}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(os.path.join(experiment_dir, subPath, f'{sub}_eigenvalueDist_{saveNote}.png'))
    plt.close()

    # 6. Controllo manuale finale
    if manualCheck:
        try:
            import tmspath_utils_adj
            ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
            postICA_clean = ica.apply(postICA_raw.copy())
        except ImportError:
            print("tmspath_utils_adj non disponibile. Salto ispezione manuale.")
            postICA_clean = postICA_raw
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica



def run_ica_artist_ext_only(EPOCHS, n_components=None, ext_threshold_uv=30, manualCheck=True, subPath='4.postICA', saveNote='postICA'):
    import os
    os.makedirs(f'{experiment_dir}\\{subPath}', exist_ok=True)

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    # Estrai le sorgenti ICA
    
    sources = ica.get_sources(EPOCHS).get_data() * 1e6  # μV
    
    if sources.ndim == 3:
        # EPOCHS: (n_components, n_epochs, n_times)
        max_abs = np.max(np.abs(sources), axis=(1, 2))
    elif sources.ndim == 2:
        # RAW: (n_components, n_times)
        max_abs = np.max(np.abs(sources), axis=1)
    else:
        raise ValueError("Formato sorgenti ICA non riconosciuto.")
    
    excluded_components = np.where(max_abs > ext_threshold_uv)[0].tolist()


    print(f"Componenti escluse (criterio EXT: |amp| > {ext_threshold_uv}μV in qualsiasi punto): {excluded_components}")
    ica.exclude = excluded_components

    # Applica ICA
    postICA_raw = ica.apply(EPOCHS.copy())

    # Salva i plot
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    if manualCheck:
        ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
        postICA_clean = ica.apply(postICA_raw.copy())
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica


def run_ica_artist_tms_events(raw, events, n_components=None, 
                              ext_threshold_uv=30, 
                              window_ms=50, 
                              manualCheck=True,
                              subPath='4.postICA', 
                              saveNote='postICA',
                              demean_between_events=False):
    import os
    import numpy as np
    import mne

    os.makedirs(f'{experiment_dir}\\{subPath}', exist_ok=True)

    sfreq = raw.info['sfreq']
    n_samples_window = int(window_ms / 1000 * sfreq)
    lag_ms = json_data['baseline_cor_tmin']*-1000
    lag_samples = int(lag_ms / 1000 * sfreq)

    # ----------- Rimozione media opzionale su intervalli ritardati -----------
    if demean_between_events:
        eeg_data = raw.get_data(picks='eeg') * 1e6  # μV
        eeg_picks = mne.pick_types(raw.info, eeg=True)

        for i in range(len(events) - 1):
            start = events[i, 0] - lag_samples
            end = events[i + 1, 0] - lag_samples
            if start < 0 or end > eeg_data.shape[1] or start >= end:
                continue
            segment = eeg_data[:, start:end]
            segment_mean = np.mean(segment, axis=1, keepdims=True)
            eeg_data[:, start:end] -= segment_mean

        raw._data[eeg_picks] = eeg_data / 1e6  # torna in Volt

    # ----------- ICA -----------
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(raw)

    sources = ica.get_sources(raw).get_data() * 1e6  # sorgenti in μV

    n_components = sources.shape[0]
    excluded_components = []

    for ic in range(n_components):
        ic_signal = sources[ic]
        for ev in events:
            onset = ev[0]
            end = onset + n_samples_window
            if end >= len(ic_signal):
                continue
            seg = ic_signal[onset:end]
            if np.any(np.abs(seg) > ext_threshold_uv):
                excluded_components.append(ic)
                break

    excluded_components = sorted(set(excluded_components))
    print(f"Componenti escluse (criterio EXT: ±{ext_threshold_uv}μV entro {window_ms}ms da eventi): {excluded_components}")
    ica.exclude = excluded_components

    postICA_raw = ica.apply(raw.copy())

    all_components = set(np.arange(n_components))
    remaining_components = list(all_components - set(excluded_components))

    if manualCheck:
        ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
        postICA_clean = ica.apply(postICA_raw.copy())
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica


def plot_customTEP(EPOCHS, subDir, key, FIGSIZE):
    # Calcolo e configurazione dei dati
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)  # Grafico più largo per una migliore visualizzazione

    mat = np.mean(EPOCHS.get_data(), axis=0).T
    times = EPOCHS.times
    
    # Traccia la linea del segnale
    b = ax.plot(
        times, 
        mat, 
        c='blue', 
        #label='REF', 
        alpha=0.7, 
        linewidth=1.5  # Linea più visibile
    )
    
    # Linee verticali
    ax.axvline(
        x=json_data['pulse_artifact_rej_timewindow_min'], 
        linewidth=2.5, 
        c='black', 
        alpha=0.8, 
        linestyle='--', 
        label='Pulse Artifact Start'
    )
    ax.axvline(
        x=json_data['pulse_artifact_rej_timewindow_max'], 
        linewidth=2.5, 
        c='black', 
        alpha=0.8, 
        linestyle='--', 
        label='Pulse Artifact End'
    )
    ax.axvline(
        x=0,
        linewidth=2.5, 
        c='red', 
        alpha=0.8, 
        linestyle='-.', 
        label='Offset Detrend Start'
    )
    ax.axvline(
        x=json_data['detrend_modeTimeWindowOffset']  + (json_data['detrend_offsetOddSamples'] * 1e-3), 
        linewidth=2.5, 
        c='red', 
        alpha=0.8, 
        linestyle='-.', 
        label='Offset Detrend End'
    )

    # Ombreggiatura per la finestra del Offset
    ax.fill_between(
        times, 
        mat.min(),  # Valore minimo del segnale
        mat.max(),  # Valore massimo del segnale
        where=(
            (times >= 0) & 
            (times <= json_data['detrend_modeTimeWindowOffset']  + (json_data['detrend_offsetOddSamples'] * 1e-3))
        ),
        color='red', 
        alpha=0.3, 
        label='Offset Detrend Window'
    )
    
    # Ombreggiatura per la finestra del pulse artifact
    ax.fill_between(
        times, 
        mat.min(),  # Valore minimo del segnale
        mat.max(),  # Valore massimo del segnale
        where=(
            (times >= json_data['pulse_artifact_rej_timewindow_min']) & 
            (times <= json_data['pulse_artifact_rej_timewindow_max'])
        ),
        color='gray', 
        alpha=0.3, 
        label='Pulse Artifact Rejection Window'
    )

    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(mat.min() + 0 * abs(mat.min()), mat.max() + 0 * abs(mat.max()))  # Margini extra per evitare tagli
    ax.set_xlabel("Time (s)", fontweight='bold')
    ax.set_ylabel("Amplitude (μV)", fontweight='bold')
    ax.tick_params(axis='both', which='major')
    ax.set_title(f'{sub} - {key}')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    #fig.tight_layout()
    fig.savefig(f'{experiment_dir}\\{subPath}\\butterflyPaper_{key}.png')
    plt.close(fig)

    """
    fig, ax = plt.subplots(1,1, figsize=(6, 3))

    mat = np.mean(EPOCHS.get_data(), axis=0).T
    for i in range(evoked_com.get_data().shape[0]):
        b=ax.plot(EPOCHS.times, EPOCHS.get_data()[i, :], c='b', alpha=0.5)
    ax.axvline(0, linewidth=10, c='k', alpha=0.5)
    ax.set_title(sub)
    
    fig.savefig(f'{experiment_dir}{sub}_butterfly_asPaper_{key}.png', dpi=300)
    plt.close(fig)
    """

def basicPlots(EPOCHS, json_data, experiment_dir, sub, key='epochs', subPath='1.basic', figsize=FIGSIZE, show=False, do_psdtopomap=False):
    fig = EPOCHS.average().plot(show=show, spatial_colors=True)  
    fig.set_size_inches(figsize[0], figsize[1])
    fig.savefig(f'{experiment_dir}\\{subPath}\\tep_{key}.png')
    plt.close()
    
    fig = EPOCHS.average().plot_topo(show=show)
    fig.set_size_inches(figsize[0], figsize[1])
    fig.savefig(f'{experiment_dir}\\{subPath}\\topo_{key}.png')
    plt.close()
    
    fig = EPOCHS.plot_psd(method='welch', 
                          fmin=EPOCHS.info['highpass'], 
                          fmax=EPOCHS.info['lowpass'], 
                          xscale='log',
                          show=show)
    fig.set_size_inches(figsize[0], figsize[1])
    fig.savefig(f'{experiment_dir}\\{subPath}\\PSD_{key}.png')
    plt.close()

    if do_psdtopomap:
        fig = EPOCHS.plot_psd_topomap(method='welch', cmap='turbo',
                            fmin=EPOCHS.info['highpass'], 
                            fmax=EPOCHS.info['lowpass'], 
                            show=show,
                            normalize=True)
        fig.set_size_inches(figsize[0], figsize[1])
        fig.savefig(f'{experiment_dir}\\{subPath}\\PSD_topomap_{key}.png')
        plt.close()

    #plot_customTEP(EPOCHS, subPath, key, FIGSIZE)

def runICA(detrendedEpochs):

    # find the maximum number of independent components
    # as the number of good channels - 1 because of average referencing
    
    n_components = len(detrendedEpochs.ch_names) - len(detrendedEpochs.info['bads']) - 1
    print(n_components)
    json_data['n_components'] = n_components
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')
    
    # set ICA parameters
    ica=mne.preprocessing.ICA(n_components=n_components,
        max_iter="auto",
        method="infomax",
        random_state=220986,
        fit_params=dict(extended=True),
    )
    
    # perform ICA decomposition
    ica=ica.fit(detrendedEpochs)
    # tagging components
    ic_labels = label_components(detrendedEpochs, ica, method="iclabel")
    """
    brain
    muscle artifact
    eye blink
    heart beat
    line noise
    channel noise
    other
    """
    for idx, i  in enumerate(ic_labels['labels']):
        print(idx+1, i)
    
    return ic_labels, ica

import os
from pathlib import Path
import numpy as np
import pandas as pd
from fooof import FOOOF
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from mne.filter import filter_data

def extract_psd_features(epochs, note, experiment_dir, json_data):
    save_dir = Path(experiment_dir) / '7.FOOOF' / note
    save_dir.mkdir(parents=True, exist_ok=True)

    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    n_trials, n_channels, n_times = data.shape

    # Concatenazione trial → (n_channels, n_times * n_trials)
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, n_times * n_trials)

    freqs, psds = welch(data_concat, fs=sfreq, average='mean', nperseg=1024)

    results = []

    # === Freq range usato per il fit
    freq_range = (json_data['l_freq'], json_data['h_freq']*2)

    for ch_idx, ch_name in enumerate(epochs.ch_names):
        fm = FOOOF(aperiodic_mode='fixed')
        fm.fit(freqs, psds[ch_idx])#, freq_range=freq_range)

        # Plot limitato al freq_range
        figp = fm.plot(freq_range=freq_range, 
                save_fig=True,
                file_name=f'chan_{ch_name}_{note}',
                file_path=save_dir)
        plt.close()

        intercept, slope = fm.aperiodic_params_
        n_resonances = len(fm.peak_params_)
        fit_error = fm.error_
        fit_r2 = fm.r_squared_

        freq_pos = freqs[freqs > 0]
        aperiodic_psd = intercept + np.log10(1 / freq_pos**slope)
        area_psd = np.trapz(aperiodic_psd, dx=freqs[1] - freqs[0])

        snr_index_broad = (area_psd / slope) / area_psd if slope != 0 and area_psd != 0 else np.nan
        snr_index_eq = 1 / slope if slope != 0 else np.nan

        freq_mask_band = (freq_pos >= json_data['l_freq']) & (freq_pos <= json_data['h_freq'])
        area_psd_narrowband = np.trapz(aperiodic_psd[freq_mask_band], dx=freqs[1] - freqs[0])
        snr_index_narrow = (area_psd_narrowband / slope) / area_psd if slope != 0 and area_psd != 0 else np.nan

        results.append({
            'channel': ch_name,
            'channel_index': ch_idx,
            'intercept': intercept,
            'slope': slope,
            'fiterror': fit_error,
            'r2': fit_r2,
            'n_resonances': n_resonances,
            'area_psd': area_psd,
            'area_psd_narrow': area_psd_narrowband,
            'snr_index_broad': snr_index_broad,
            'snr_index_eq': snr_index_eq,
            'snr_index_narrow': snr_index_narrow,
        })

    # === Salva risultati
    df = pd.DataFrame(results)
    df.to_csv(save_dir / f"{note}.csv", index=False)
    df.to_pickle(save_dir / f"{note}.pkl")

    # === Plot riepilogativo
    fig, axs = plt.subplots(2, 2, figsize=(16, 21))
    fig.suptitle(note)

    sns.barplot(y="channel", x="slope", data=df, ax=axs[0, 0], color="k")
    axs[0, 0].set_title("Aperiodic Slope per Channel")
    axs[0, 0].set_xlabel("Slope (log)")
    axs[0, 0].set_ylabel("Channel")
    axs[0, 0].set_xlim(0.01, 10)
    axs[0, 0].set_xscale("log")
    for x in [0.5, 1, 2, 3, 4, 5]:
        axs[0, 0].axvline(x=x, color='red', linestyle='-', linewidth=2)

    sns.barplot(y="channel", x="n_resonances", data=df, ax=axs[0, 1], color="k")
    axs[0, 1].set_title("Number of Resonances per Channel")
    axs[0, 1].set_xlabel("n Peaks (log)")
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xlim(0.01, 100)
    axs[0, 1].set_xscale("log")

    sns.barplot(y="channel", x="snr_index_narrow", data=df, ax=axs[1, 0], color="k")
    axs[1, 0].set_title("SNR Narrowband per Channel")
    axs[1, 0].set_xlabel("SNR")
    axs[1, 0].set_ylabel("Channel")
    axs[1, 0].set_xlim(0, 0.5)

    sns.barplot(y="channel", x="fiterror", data=df, ax=axs[1, 1], color="k")
    axs[1, 1].set_title("Fit Error per Channel")
    axs[1, 1].set_xlabel("Error")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xlim(0, 0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "summary_plots_hbar.png", dpi=150)
    plt.close()

    print(f"✅ FOOOF PSD features salvate e plottate in: {save_dir}")
    return df
    
def save_bad_epochs_and_channels(info_string, json_data):
    marker_epochs = "The following epochs were marked as bad and are dropped:"
    bad_epochs = ""
    
    if marker_epochs in info_string:
        print('i', info_string.find(marker_epochs))
        start = info_string.find(marker_epochs) + len(marker_epochs) + 1
        bad_epochs = info_string[start:].split("\n")[0].strip()
        print('f', info_string[start:].split("\n")[0].strip())
    
    json_data['bad_trials']=bad_epochs
    
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


def plotTrialTepVariability(epochs, json_data, experiment_dir, sub, chanNAME='AF3', operator=np.mean, save=False, 
                            figsize=FIGSIZE, parDir='preDetrend'):
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=False, sharex=True)
    i = np.where(np.array(epochs.ch_names)==chanNAME)[0][0]
    ax.plot(epochs.times, epochs.get_data()[:, i, :].T, c='b', linewidth=1, alpha=0.5)
    ax.plot(epochs.times, operator(epochs.get_data()[:, i, :], axis=0), c='r', label=str(operator), linewidth=5)
    ax.set_title(f'{epochs.ch_names[i]}')
    ax.axvline(0, color='k', linestyle='--')
    #ax.set_xlim(json_data['epochs_timewindow_min'], json_data['epochs_timewindow_max'] )
    ax.legend()
    ax.set_xlabel('Times [s]')
    ax.set_ylabel('Amplitude [V]')   
    plt.tight_layout()
    if save:
        saveNote = f'tepVar_{chanNAME}'
        fig.savefig(f'{experiment_dir}/3.trials/{parDir}/{sub}_{saveNote}.png')
    plt.close(fig)



def computeTimeMasks(epochs, chan, trial, json_data, do_plot=False, offset=0.20):
    
    par = offset
    #print(json_data['pulse_artifact_rej_timewindow_max']*par)

    maskPreOffset = np.logical_and(epochs.times>=epochs.times.min(), 
                                   epochs.times<json_data['detrend_minTimeWindowOffset'])
    
    tempMaskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                    epochs.times<=par)

    
    if len(epochs.get_data().shape)==2: 
        sig = epochs.get_data()[chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[chan, :]
        
    if len(epochs.get_data().shape)>2: 
        sig = epochs.get_data()[trial, chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[trial, chan, :]

    extreme=json_data['detrendExtremeTechinque']
    if extreme=='max':
        tMaxOffsetIndex=np.argmax(abs(sig))
    if extreme=='derivative':
        diff = np.diff(sig)
        # Trova transizione da negativo a positivo (zero-crossing della derivata)
        zero_crossings = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0]
        if len(zero_crossings) > 0:
            tMaxOffsetIndex = zero_crossings[0] + 1  # +1 perché diff è più corto di 1
        else:
            # fallback: se non trovi un punto di inversione, usa massimo assoluto
            tMaxOffsetIndex = np.argmax(np.abs(sig))

    if tMaxOffsetIndex==0: tMaxOffsetIndex=1
    if tMaxOffsetIndex==len(tempMaskOffset): tMaxOffsetIndex=len(tempMaskOffset)-1
   
    if do_plot:
        plt.figure(figsize=(13, 11))
        plt.plot(epochs.times[tempMaskOffset], sig)
        plt.plot(epochs.times, sigAll)
        plt.title(f'{sub}-{epochs.ch_names[chan]}-trial{trial}')
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f'peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}')
        plt.axvline(x=offset, label=f'maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}')
        plt.legend(loc='upper right')
        plt.savefig(f'{experiment_dir}/2.detrend/test_maskTest_{sub}_{chan}_{trial}.png')
        plt.close()
        
    maskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                epochs.times<=epochs.times[tempMaskOffset][tMaxOffsetIndex])   
    
    maskPostOffset = np.logical_and(epochs.times>epochs.times[tempMaskOffset][tMaxOffsetIndex], 
                                    epochs.times<=epochs.times.max())

    return maskPreOffset, maskOffset, maskPostOffset

def computeTimeMasks_old(epochs, chan, trial, do_plot=False, offset=0.20):
    
    par = offset
    #print(json_data['pulse_artifact_rej_timewindow_max']*par)

    maskPreOffset = np.logical_and(epochs.times>=epochs.times.min(), 
                                   epochs.times<json_data['detrend_minTimeWindowOffset'])
    
    tempMaskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                    epochs.times<=par)

    
    if len(epochs.get_data().shape)==2: 
        sig = epochs.get_data()[chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[chan, :]
        
    if len(epochs.get_data().shape)>2: 
        sig = epochs.get_data()[trial, chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[trial, chan, :]
        

    tMaxOffsetIndex=np.argmax(abs(sig))
    
    if tMaxOffsetIndex==0: tMaxOffsetIndex=1
    if tMaxOffsetIndex==len(tempMaskOffset): tMaxOffsetIndex=len(tempMaskOffset)-1

    #print(tMaxOffsetIndex)
    
    if do_plot:
        plt.figure(figsize=(13, 11))
        plt.plot(epochs.times[tempMaskOffset], sig)
        plt.plot(epochs.times, sigAll)
        plt.title(f'{sub}-{epochs.ch_names[chan]}-trial{trial}')
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f'peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}')
        plt.axvline(x=offset, label=f'maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}')
        plt.legend(loc='upper right')
        plt.savefig(f'{experiment_dir}/2.detrend/test/maskTest_{sub}_{chan}_{trial}.png')
        plt.close()
        
    maskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                epochs.times<=epochs.times[tempMaskOffset][tMaxOffsetIndex])   
    
    maskPostOffset = np.logical_and(epochs.times>epochs.times[tempMaskOffset][tMaxOffsetIndex], 
                                    epochs.times<=epochs.times.max())

    return maskPreOffset, maskOffset, maskPostOffset

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
    df_slopes['Zslope'] = np.zeros(df_slopes['slope'].shape[0])
    for time_label in timeMaskLabels:
        mask = df_slopes['id_twindow'] == time_label
        df_slopes.loc[mask, 'Zslope'] = scipy.stats.zscore(df_slopes.loc[mask, 'slope'].values)

    # add plot distance from seed vs slope

    return df_slopes

import os
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def computeSlopesPlot_v3(df_slopes, sub,
                         saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend',
                      zvalue=True, json_data=None, experiment_dir='.'):
    
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    
    # Dizionario e flag
    outlier_channels_by_twindow = {}
    found_outlier = False
    df_outliers_list = []

    # ANOVA per ciascuna finestra
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)

    # Plot swarm/pointplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        has_outliers = any(abs(data_subset[VAR] - mean_val) > 3 * std_val)
        title_color = "red" if has_outliers else "black"

        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-7, 7) if zvalue else ax[idx].set_xlim(data_subset[VAR].min(), data_subset[VAR].max())

        if json_data is not None:
            thr = json_data.get('detrend_slopeThr', 3)
            ax[idx].axvline(x=-thr, alpha=0.25, linewidth=10, c='r')
            ax[idx].axvline(x=thr, alpha=0.25, linewidth=10, c='r')

        sns.swarmplot(data=data_subset, x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1)
        sns.pointplot(data=data_subset, estimator=np.mean, x=VAR, y='chan', ax=ax[idx], color='r', alpha=1)
        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')

        
        # --- Calcolo media e outlier per canale ---
        mean_per_chan = data_subset.groupby("chan")[VAR].mean()
        global_mean = mean_per_chan.mean()
        global_std = mean_per_chan.std()
        z_scores = (mean_per_chan - global_mean) / global_std
        outlier_mask = abs(z_scores) > 3

        df_summary = pd.DataFrame({
            'id_twindow': label,
            'chan': mean_per_chan.index,
            'mean_' + VAR: mean_per_chan.values,
            'Zscore': z_scores.values,
        })
        df_summary['is_outlier'] = abs(df_summary['Zscore']) > 3
        
        df_outliers_list.append(df_summary)
        
        if df_summary['is_outlier'].any():
            outlier_channels_by_twindow[label] = df_summary[df_summary['is_outlier']]['chan'].tolist()
            found_outlier = True


    # Salva plot swarm/pointplot
    out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # KDE plot per canali seed
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])] if json_data else df_slopes
    timeMaskLabels = df_slopes['id_twindow'].unique()
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_slopes_seed, x=VAR, hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)

    seed_list = json_data["seedChans"] if json_data and "seedChans" in json_data else "ALL"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}')
    histo_path = os.path.join(experiment_dir, subPath, f'histo_{VAR}_{saveNote}.png')
    plt.savefig(histo_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Unisci risultati per finestra in unico DataFrame
    df_outliers_all = pd.concat(df_outliers_list, ignore_index=True)

    return outlier_channels_by_twindow, found_outlier, df_outliers_all




import os
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def computeSlopesPlot_v2(df_slopes,
                         sub,
                         saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend', zvalue=True, json_data=None, experiment_dir='.'):
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    outlier_channels = {}

    # Calcolo ANOVA per ogni time window
    anova_results = []
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)
        anova_results.append({"Time Window": time_window, "Statistic": anova_stat, "p-value": anova_p})

    df_anova_results = pd.DataFrame(anova_results)

    # Plot a 3 pannelli
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        has_outliers = any(abs(data_subset[VAR] - mean_val) > 3 * std_val)
        title_color = "red" if has_outliers else "black"

        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-7, 7) if zvalue else ax[idx].set_xlim(data_subset[VAR].min(), data_subset[VAR].max())

        if json_data is not None:
            thr = json_data.get('detrend_slopeThr', 3)
            ax[idx].axvline(x=-thr, alpha=0.25, linewidth=10, c='r')
            ax[idx].axvline(x=thr, alpha=0.25, linewidth=10, c='r')

        sns.swarmplot(data=data_subset, x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1)
        sns.pointplot(data=data_subset, estimator=np.mean, x=VAR, y='chan', ax=ax[idx], color='r', alpha=1)

        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')

        # --- Calcolo outlier medi per canale in questa finestra ---
        mean_per_chan = data_subset.groupby("chan")[VAR].mean()
        global_mean = mean_per_chan.mean()
        global_std = mean_per_chan.std()
        z_scores = (mean_per_chan - global_mean) / global_std
        outliers = z_scores[abs(z_scores) > 3]

        if not outliers.empty:
            outlier_channels[label] = [(chan, round(z_scores[chan], 2)) for chan in outliers.index]

    # Salvataggio del plot swarm+pointplot
    out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # KDE plot sui canali seed (o tutti se seedChans mancante)
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])] if json_data else df_slopes
    timeMaskLabels = df_slopes['id_twindow'].unique()
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_slopes_seed, x=VAR, hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)

    p_text = "p<0.05" if p_value < 0.05 else "p=ns"
    seed_list = json_data["seedChans"] if json_data and "seedChans" in json_data else "ALL"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}')

    histo_path = os.path.join(experiment_dir, subPath, f'histo_{VAR}_{saveNote}.png')
    plt.savefig(histo_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'p_value': p_value, 'F': anova_stat, 'outlier_channels': outlier_channels}


def computeSlopesPlot(df_slopes, 
                      json_data, experiment_dir, sub,
                      saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend', zvalue=True):
    
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']

    # Calcolo ANOVA per ogni time window
    anova_results = []
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)
        anova_results.append({"Time Window": time_window, "Statistic": anova_stat, "p-value": anova_p})

    # Convertire in DataFrame
    df_anova_results = pd.DataFrame(anova_results)

    # Plot dei pointplot per ogni timeMaskLabel con ANOVA p-value nei titoli
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
    
        # Calcola media e deviazione standard
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        
        # Determina se ci sono valori oltre 3 sigma
        #has_outliers = any(abs(data_subset[VAR] - mean_val) > threshold * std_val)
        title_color = "black" #red" if has_outliers else "black"
        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-3.5*2, 3.5*2)
        ax[idx].axvline(x=-json_data['detrend_slopeThr'], alpha=0.25, linewidth=10, c='r')
        ax[idx].axvline(x=json_data['detrend_slopeThr'], alpha=0.25, linewidth=10, c='r')
        
        # Crea il pointplot
        sns.swarmplot(
            data=data_subset,
            x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1, #errorbar='se'
        )

        # Crea il pointplot
        sns.pointplot(
            data=data_subset,
            estimator=np.mean,
            x=VAR, y='chan', ax=ax[idx], color='r', alpha=1, #errorbar='se'
        )

        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')
        out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}_{label}.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_subset.to_csv(out_path, index=False)


    # Salvataggio del primo plot
    fig.savefig(f'{experiment_dir}\\{subPath}\\{VAR}_{saveNote}.png')
    plt.close(fig)

    # Seleziona la variabile di interesse
    VAR = 'Zslope' if zvalue else 'slope'
    timeMaskLabels = df_slopes['id_twindow'].unique()
    plt.figure(figsize=(8, 5))
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])]
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)
    sns.kdeplot(data=df_slopes_seed, x='Zslope', hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)
    #plt.title(f"ANOVA: F={anova_stat:.3f}, p={p_value:.3e} \n {json_data['seedChans']}") #\n {multimodal_text}")
    p_text = "p<0.05" if p_value < 0.05 else "p=ns"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={json_data['seedChans']}')
    plt.savefig(f'{experiment_dir}\\{subPath}\\histo_{VAR}_{saveNote}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('p_value', p_value, 'F', anova_stat)

    return df_anova_results



def generate_noise_from_distribution(time_series, model='Gaussian', n_samples=1000):
    """
    Generate noise samples from a specified distribution based on a time series.

    Parameters:
    - time_series: The input time series (numpy array or pandas Series).
    - model: The distribution to sample from ('Gaussian', 'Exponential', 'Laplace', 
             'Poisson', 'Rayleigh', 'Gamma', 'StudentT', 'Uniform').
    - n_samples: The number of noise samples to generate.

    Returns:
    - noise_samples: The generated noise samples.
    """
    
    # Compute parameters based on the time series
    if isinstance(time_series, pd.Series):
        time_series = time_series.values  # Convert pandas Series to numpy array for easier handling
    
    # Common parameters
    mean = np.mean(time_series)
    std = np.std(time_series)*0.5
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    median = np.median(time_series)
    
    # Now, depending on the chosen model, we compute the parameters and generate samples
    if model == 'Gaussian':
        # Gaussian distribution (Normal distribution)
        noise_samples = np.random.normal(loc=mean, scale=std, size=n_samples)
    
    elif model == 'Exponential':
        # Exponential distribution (mean = 1/lambda, use std for scale)
        lambda_param = 1 / std  # Use standard deviation as a proxy for rate
        noise_samples = np.random.exponential(scale=1/lambda_param, size=n_samples)
    
    elif model == 'Laplace':
        # Laplace distribution (mean = median, scale = std)
        noise_samples = np.random.laplace(loc=median, scale=std, size=n_samples)
    
    elif model == 'Poisson':
        # Poisson distribution (lambda = mean of the time series)
        lambda_param = mean  # Average rate of occurrence
        noise_samples = np.random.poisson(lam=lambda_param, size=n_samples)
    
    elif model == 'Rayleigh':
        # Rayleigh distribution (scale = std)
        noise_samples = np.random.rayleigh(scale=std, size=n_samples)
    
    elif model == 'Gamma':
        # Gamma distribution (shape = (mean^2)/(std^2), scale = (std^2)/mean)
        shape = (mean**2) / (std**2)
        scale = (std**2) / mean
        noise_samples = np.random.gamma(shape, scale, size=n_samples)
    
    elif model == 'StudentT':
        # Student's t-distribution (df = 2 for heavy tails, scale = std)
        df = 2  # degrees of freedom (use a small df for heavy tails)
        noise_samples = np.random.standard_t(df, size=n_samples) * std + mean
    
    elif model == 'Uniform':
        # Uniform distribution (min = min_val, max = max_val)
        noise_samples = np.random.uniform(low=min_val, high=max_val, size=n_samples)
    
    else:
        raise ValueError(f"Model '{model}' is not supported. Choose from ['Gaussian', 'Exponential', 'Laplace', 'Poisson', 'Rayleigh', 'Gamma', 'StudentT', 'Uniform']")

    return noise_samples


# no Exponential, Poisson, Rayleigh, 'Gamma', 'StudentT', 
supported_models = ['Exponential', 'Gaussian', 'Laplace', 'Uniform']

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
                tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
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
            tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
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


def plot_detrend_example_v3(sub,   
                            typeOffsetRise,
                            typeOffsetDecay,
                            OPTPARS_A,
                            OPTPARS_B,
                            OPTPARS_C,
                            chan, epoch_idx, times, TEP, trend_line_A, trend_line_B, trend_line_C,
                            tep_agg, timeMask, fitConstraint,
                         #detrendMode, orderPreOffset, orderOffset,
                         #orderPostOffset_temp, popt_C=None, popt_B=None, 
                         markerChan=False,
                         experiment_dir=None, PROB=1.0):

    # Decide number of subplots and layout
    if markerChan and (epoch_idx in [1, 10, 20, 30, 40]):
        fig, ax = plt.subplots(2, 2, figsize=(18, 11))
        ax = ax.flatten()
    elif not markerChan and np.random.rand() < PROB * 0:
        fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    else:
        return  # Skip plotting

    min_offset_time = times[timeMask[1]].min()
    max_offset_time = times[timeMask[1]].max()
    title = (
        f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
#        f'orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n'
        f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n'
        f'typeOffsetRise={typeOffsetRise} with pars={OPTPARS_B} \n'
        f'typeOffsetDecay={typeOffsetDecay} with pars={OPTPARS_C} \n'     
    )
    filename = f'{sub}_{chan}_{epoch_idx}_{fitConstraint}.png'

    fig.suptitle(title)

    # First subplot: full fit
    ax[0].plot(times, TEP, 'k', label='Original', marker='+')
    ax[0].plot(times[timeMask[0]], trend_line_A, c='b', label='fit prestimulus', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[1]], trend_line_B, c='orange', label='fit artifacts', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[2]], trend_line_C, c='g', label='fit poststimulus', alpha=.5, linewidth=4)
    ax[0].set_title('Fit')

    # Second subplot: detrended
    ax[1].plot(times, tep_agg, c='r', marker='+')
    ax[1].set_title('TEP Detrended')

    if len(ax) > 2:
        # Third subplot: artifact zoom
        ax[2].plot(times[timeMask[1]], TEP[timeMask[1]], 'k+', label='Original')
        ax[2].plot(times[timeMask[1]], trend_line_B, 'orange', linewidth=3, label='Exp Fit')
        ax[2].axvline(0)
        ax[2].set_title('Artifact Window Focus')
        ax[2].legend()

        # Fourth subplot: prestimulus zoom
        ax[3].plot(times[timeMask[0]], TEP[timeMask[0]], 'k+', label='Original')
        ax[3].plot(times[timeMask[0]], trend_line_A, 'orange', linewidth=3, label='Exp Fit')
        ax[3].axvline(0)
        ax[3].set_title('Prestimulus Window Focus')
        ax[3].legend()

    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.65 if len(ax) > 2 else -1, 1), borderaxespad=0.)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, '2.detrend', 'examples', filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_detrend_example(sub, chan, epoch_idx, times, TEP, trend_line_A, trend_line_B, trend_line_C,
                         tep_agg, timeMask, 
                         detrendMode, fitConstraint, orderPreOffset, orderOffset,
                         orderPostOffset_temp, popt_C=None, popt_B=None, 
                         markerChan=False,
                         experiment_dir=None, PROB=1.0):

    # Decide number of subplots and layout
    if markerChan and (epoch_idx in [1, 10, 20]):
        fig, ax = plt.subplots(2, 2, figsize=(18, 11))
        ax = ax.flatten()
    elif not markerChan and np.random.rand() < PROB * 1:
        fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    else:
        return  # Skip plotting

    min_offset_time = times[timeMask[1]].min()
    max_offset_time = times[timeMask[1]].max()

    # Common title logic
    if detrendMode == 'poly':
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
            f'orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f}'
        )
        filename = f'{sub}_{chan}_{epoch_idx}_{fitConstraint}_{orderPostOffset_temp}.png'
        
    if detrendMode in ['singleDecay', 'doubleDecay']:
        """
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
            f'orderPre={orderPreOffset} | Art exp={orderOffset} with pars={popt_B}\n if popt_C != None else noExp but Lagrange with order={orderOffset}\n'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n '
            f'postArt exp={detrendMode} with pars={popt_C}'
        )
        """
        if popt_C is not None:
            exp_info = f'Art exp={orderOffset} with pars={popt_B}\n'
            postart_info = f'postArt exp={detrendMode} with pars={popt_C}'
        else:
            exp_info = f'noExp but Lagrange with order={orderOffset}\n'
            postart_info = 'no postArt exp fit'
        
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n'
            f'fitConstrain={fitConstraint} \n'
            f'orderPre={orderPreOffset} | {exp_info}'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n'
            f'{postart_info}'
        )
        filename = f'{sub}_detrendExample_{chan}_{epoch_idx}_{fitConstraint}_{detrendMode}.png'

    fig.suptitle(title)

    # First subplot: full fit
    ax[0].plot(times, TEP, 'k', label='Original', marker='+')
    ax[0].plot(times[timeMask[0]], trend_line_A, c='b', label='fit prestimulus', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[1]], trend_line_B, c='orange', label='fit artifacts', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[2]], trend_line_C, c='g', label='fit poststimulus', alpha=.5, linewidth=4)
    ax[0].set_title('Fit')

    # Second subplot: detrended
    ax[1].plot(times, tep_agg, c='r', marker='+')
    ax[1].set_title('TEP Detrended')

    if len(ax) > 2:
        # Third subplot: artifact zoom
        ax[2].plot(times[timeMask[1]], TEP[timeMask[1]], 'k+', label='Original')
        ax[2].plot(times[timeMask[1]], trend_line_B, 'orange', linewidth=3, label='Exp Fit')
        ax[2].axvline(0)
        ax[2].set_title('Artifact Window Focus')
        ax[2].legend()

        # Fourth subplot: prestimulus zoom
        ax[3].plot(times[timeMask[0]], TEP[timeMask[0]], 'k+', label='Original')
        ax[3].plot(times[timeMask[0]], trend_line_A, 'orange', linewidth=3, label='Exp Fit')
        ax[3].axvline(0)
        ax[3].set_title('Prestimulus Window Focus')
        ax[3].legend()

    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.65 if len(ax) > 2 else -1, 1), borderaxespad=0.)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, '2.detrend', 'examples', filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tabStat(df_tabStat, experiment_dir):
    # Crea la cartella per i plot
    output_dir = os.path.join(experiment_dir, '2.detrend', 'statDetrend')
    os.makedirs(output_dir, exist_ok=True)

    # Assicurati che la colonna totale esista
    df_tabStat['MSE_total'] = df_tabStat[['mseA', 'mseB', 'mseC']].sum(axis=1)

    # --- 1. Boxplot per segmento ---
    df_melted = df_tabStat.melt(
        id_vars=['chan', 'epoch_idx'],
        value_vars=['mseA', 'mseB', 'mseC'],
        var_name='Segment',
        value_name='MSE'
    )
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melted, x='Segment', y='MSE')
    plt.title('Distribuzione MSE per segmento')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_MSE_per_segment.png'), dpi=300)
    plt.close()

    # --- 2. Heatmap MSE totale ---
    pivot = df_tabStat.pivot(index='chan', columns='epoch_idx', values='MSE_total')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='viridis')
    plt.title('Heatmap MSE totale per canale ed epoca')
    plt.xlabel('Epoch')
    plt.ylabel('Canale')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_MSE_total.png'), dpi=300)
    plt.close()

    # --- 3. Line plot per epoca ---
    plt.figure(figsize=(10, 5))
    for chan in df_tabStat['chan'].unique():
        df_chan = df_tabStat[df_tabStat['chan'] == chan]
        plt.plot(df_chan['epoch_idx'], df_chan['MSE_total'], label=chan)
    plt.title('MSE totale per epoca per canale')
    plt.xlabel('Epoch')
    plt.ylabel('MSE totale')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lineplot_MSE_total_per_epoch.png'), dpi=300)
    plt.close()

    # --- 4. Barplot per canale ---
    df_grouped = df_tabStat.groupby('chan')['MSE_total'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_grouped, x='chan', y='MSE_total')
    plt.title('MSE medio per canale')
    plt.ylabel('MSE totale (media su epoche)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplot_MSE_per_channel.png'), dpi=300)
    plt.close()

    print(f"[INFO] Plot salvati in: {output_dir}")





def apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
    k = oddSamples / 1000  # da ms a secondi

    times = EPOCHS.times
    precorrectionMask = np.logical_and(times >= times[timeMask[0]].min(),
                                       times < (times[timeMask[0]].max() - k))
    correctionMask = np.logical_and(times >= (times[timeMask[1]].min() - k),
                                    times <= (times[timeMask[1]].max() + k))

    if correctMode == 'moving_average':
        tep_flat = tep[precorrectionMask].flatten()
        window_size = oddSamples
        if len(tep_flat) >= window_size:
            new_samples = np.array([
                np.mean(tep_flat[max(0, i - window_size//2):i + window_size//2])
                for i in range(len(tep_flat))
            ])[-1]
        else:
            new_samples = np.mean(tep_flat) if len(tep_flat) > 0 else 0
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'median':
        new_samples = np.median(tep[precorrectionMask].flatten())
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'zeros':
        num_samples = sum(correctionMask)
        tep_agg[correctionMask] = np.zeros(num_samples)

    elif correctMode == 'resample':
        num_samples = sum(correctionMask)
        new_samples = resample(tep[precorrectionMask].flatten(), num=num_samples)
        tep_agg[correctionMask] = new_samples

    elif correctMode in supported_models:
        num_samples = sum(correctionMask)
        new_samples = generate_noise_from_distribution(
            tep[precorrectionMask].flatten(),
            model=correctMode,
            n_samples=num_samples
        )
        tep_agg[correctionMask] = new_samples

    return tep_agg



def plot_slope_resonances(PSTATS, PSTATS2, saveNote='pol_degree_estimate', subPath='2.detrend'):

    neural_params_dfs = []
    offset_params_dfs = []
    
    for entry in PSTATS:
        integer_value = entry[0]  # Numero intero (grado del polinomio)
        #offset_data = entry[1]    # Offset con p-values
        df_data = entry[2]        # DataFrame con dati EEG
        mse_value = entry[3]      # Valore MSE
        
        # Creazione del DataFrame dei parametri neurali
        df_neural = df_data.copy()
        df_neural['pol_degree'] = integer_value
        neural_params_dfs.append(df_neural)
        
        # Creazione del DataFrame dei parametri di offset
        #df_offset = pd.DataFrame(offset_data, columns=['offset_type', 'p_value'])
        #df_offset['pol_degree'] = integer_value
        #df_offset['mse'] = mse_value
        #offset_params_dfs.append(df_offset)
    
    df_neural_params = pd.concat(neural_params_dfs, ignore_index=True)
    #df_offset_params = pd.concat(offset_params_dfs, ignore_index=True)
    #min_resonance_row = df_neural_params.loc[df_neural_params['n_resonances'].idxmin()]
    #pol_degree_min_resonances = min_resonance_row['pol_degree']
    df_mean_resonances = df_neural_params.groupby('pol_degree')['n_resonances'].mean()
    pol_degree_min_resonances = np.argmin(df_neural_params.groupby('pol_degree')['n_resonances'].mean())+1
    
    fig1, ax1 = plt.subplots(figsize=(13, 6))
    sns.swarmplot(data=df_neural_params, x='pol_degree', y='slope', ax=ax1, color='b', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='slope', ax=ax1, color='b')
    ax1.set_ylabel('exp of 1/f^exp', color='b')
    ax1.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_slope.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Number of Resonances', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_resonances.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # F and p da PSTATS2
    df = pd.DataFrame(PSTATS2, columns=['Polynomial Degree', 'Stats'])
    df = df.join(pd.json_normalize(df['Stats'])).drop(columns=['Stats'])
    pol_degree_min_F = df['Polynomial Degree'][np.argmin(df['F'])]

    df = pd.DataFrame(PSTATS2, columns=['Polynomial Degree', 'Stats'])
    df = df.join(pd.json_normalize(df['Stats'])).drop(columns=['Stats'])
    fig3, ax1 = plt.subplots(figsize=(13, 6))
    sns.lineplot(data=df, x='Polynomial Degree', y='p_value', ax=ax1, color='b', label='p-value')
    ax1.set_ylabel('p-value (log scale)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Polynomial Degree', y='F', ax=ax2, color='r', label='F-statistic')
    ax2.set_ylabel('F-statistic (log scale)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_title(f"Estimated Pol Degree (min F)={pol_degree_min_F}")
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_fp.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='fiterror', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_fitError.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='r2', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_r2.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    return df_neural_params, pol_degree_min_resonances, pol_degree_min_F






















    
 