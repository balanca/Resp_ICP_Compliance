import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal
import scipy.interpolate
import physio
import pycns
import sys
from tools import *
from overview_data_pycns import get_patient_list
from configuration import *
from params_multi_projects_jobs import *
import jobtools


# DETECT RESP JOB

def detect_resp(sub, **p):
    """
    Detect respiration cycles and ventilation-control periods from a subject's CO2 signal.

    This function loads the subject's CO2 stream, computes respiration cycles using
    physio.compute_respiration(), extracts timing information, computes breathing
    frequency and its sliding variability, and finally identifies periods of
    controlled ventilation based on a threshold (automatic or user-provided).
    """
    raw_folder = data_path / sub # path to subject's raw data folder
    cns_reader = pycns.CnsReader(raw_folder) # load CNS reader for raw signals
    co2_stream = cns_reader.streams['CO2'] # select CO2 stream
    datetimes = co2_stream.get_times(as_second=False) # get timestamps as datetime
    times = co2_stream.get_times(as_second=True) # get timestamps in seconds
    raw_co2 = co2_stream.get_data(with_times=False, apply_gain=True) # load CO2 values
    srate = 1/(np.median(np.diff(times))) # compute sampling rate
    co2, resp_cycles = physio.compute_respiration(raw_co2, srate, parameter_preset='human_co2') # compute respiration cycles
    resp_cycles['inspi_time'] = times[resp_cycles['inspi_index']] # timestamp of inspiration onset
    resp_cycles['expi_time'] = times[resp_cycles['expi_index']] # timestamp of expiration onset
    resp_cycles['next_inspi_time'] = times[resp_cycles['next_inspi_index']] # next inspiration
    resp_cycles['inspi_date'] = datetimes[resp_cycles['inspi_index']].astype('datetime64[ns]') # insp. datetime
    resp_cycles['expi_date'] = datetimes[resp_cycles['expi_index']].astype('datetime64[ns]') # exp. datetime
    resp_cycles['next_inspi_date'] = datetimes[resp_cycles['next_inspi_index']].astype('datetime64[ns]') # next insp. datetime
    resp_cycles['cycle_freq_cpm'] = resp_cycles['cycle_freq'] * 60 # convert freq to cycles per minute
    resp_cycles['sliding_variability_cpm'] = resp_cycles['cycle_freq_cpm'].rolling(p['N_cycles_sliding_sd'], center=True).std().bfill().ffill() # sliding SD of freq

    if p['threshold_controlled_ventilation_sd_cpm'] == 'auto': # auto threshold mode
        bins = np.arange(0, 1, 0.025) # histogram bins
        count, bins = np.histogram(resp_cycles['sliding_variability_cpm'], bins) # variability distribution
        d1 = np.gradient(count) # first derivative of histogram
        minimums = np.where((d1[:-1] < 0) & (d1[1:] >= 0))[0] # find troughs in derivative
        if minimums.size > 0:
            first_minimum = minimums[0] # first trough
            thresh = bins[1:][first_minimum] # threshold = variability at trough
        else:
            thresh = 0 # fallback if no minimum found
    else:
        thresh = p['threshold_controlled_ventilation_sd_cpm'] # use provided threshold

    resp_cycles['is_ventilation_controlled'] = (resp_cycles['sliding_variability_cpm'] < thresh).astype(int) # detect low-variability periods
    resp_cycles['is_ventilation_controlled'] = resp_cycles['is_ventilation_controlled'].rolling(p['N_cycles_sliding_ventilation_bool'], center=True).median().bfill().ffill() # smooth flag
    resp_cycles.loc[resp_cycles['is_ventilation_controlled'] == 0.5, 'is_ventilation_controlled'] = 0 # resolve ambiguous 0.5 values

    return xr.Dataset(resp_cycles) # return xr.Dataset with respiration metrics


def test_detect_resp(sub):
    print(sub)
    ds = detect_resp(sub, **detect_resp_params).to_dataframe()
    print(ds)

detect_resp_job = jobtools.Job(precomputedir, 'detect_resp', detect_resp_params, detect_resp)
jobtools.register_job(detect_resp_job)


# DETECT ECG JOB

def detect_ecg(sub, **p):
    """
    Detect ECG peaks and extract their timing information from a subject's ECG signal.

    This function loads the subject's ECG stream, applies peak detection using
    physio.compute_ecg(), and returns peak timing both in seconds and datetime format.
    """
    raw_folder = data_path / sub # path to subject's raw data folder
    cns_reader = pycns.CnsReader(raw_folder) # load CNS reader for raw signals
    ecg_stream = cns_reader.streams['ECG_II'] # select ECG lead II stream
    datetimes = ecg_stream.get_times(as_second=False) # get timestamps as datetime
    times = ecg_stream.get_times(as_second=True) # get timestamps in seconds
    raw_ecg = ecg_stream.get_data(with_times=False, apply_gain=False) # load ECG values
    srate = 1/(np.median(np.diff(times))) # compute sampling rate
    parameters = physio.get_ecg_parameters('human_ecg') # load default ECG parameters
    if sub == 'P112':
        parameters['peak_clean']['min_interval_ms'] = 300.0 # subject-specific adjustment
    ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameters=parameters) # compute ECG + peaks
    ecg_peaks['peak_time'] = times[ecg_peaks['peak_index']] # peak time in seconds
    ecg_peaks['peak_date'] = datetimes[ecg_peaks['peak_index']].astype('datetime64[ns]') # peak time as datetime
    return xr.Dataset(ecg_peaks) # return xr.Dataset with ECG peak information


def test_detect_ecg(sub):
    print(sub)
    ds = detect_ecg(sub, **detect_ecg_params).to_dataframe()
    print(ds)

detect_ecg_job = jobtools.Job(precomputedir, 'detect_ecg', detect_ecg_params, detect_ecg)
jobtools.register_job(detect_ecg_job)


# DETECT ICP JOB

def detect_icp(sub, **p):
    """
    Detect intracranial pressure (ICP) peaks and troughs from the ICP stream.

    This function loads the subject's ICP stream, computes peaks and troughs using
    compute_icp(), and returns an xarray dataset containing timing information
    for peaks, troughs, and next troughs.
    """
    raw_folder = data_path / sub # path to subject's raw data folder
    cns_reader = pycns.CnsReader(raw_folder) # load CNS reader
    stream = cns_reader.streams['ICP'] # select ICP stream
    datetimes = stream.get_times(as_second=False) # timestamps as datetime
    times = stream.get_times(as_second=True) # timestamps in seconds
    srate_sig = 1/(np.median(np.diff(times))) # compute sampling rate
    raw_sig = stream.get_data(with_times=False, apply_gain=True) # load ICP signal
    detections = compute_icp( # compute ICP peaks and troughs
        raw_sig, srate_sig,
        date_vector=datetimes,
        lowcut=p['lowcut'],
        highcut=p['highcut'],
        order=p['order'],
        ftype=p['ftype'],
        peak_prominence=p['peak_prominence'],
        h_distance_s=p['h_distance_s'],
        rise_amplitude_limits=p['rise_amplitude_limits'],
        amplitude_at_trough_low_limit=p['amplitude_at_trough_low_limit']
    )
    detections['trough_time'] = times[detections['trough_ind']] # trough times in seconds
    detections['peak_time'] = times[detections['peak_ind']] # peak times in seconds
    detections['next_trough_time'] = times[detections['next_trough_ind']] # next trough times
    return xr.Dataset(detections) # return as xarray dataset


def test_detect_icp(sub):
    print(sub)
    ds = detect_icp(sub, **detect_icp_params).to_dataframe()
    print(ds)

detect_icp_job = jobtools.Job(precomputedir, 'detect_icp', detect_icp_params, detect_icp)
jobtools.register_job(detect_icp_job)


# P2/P1 RATIO JOB

def ratio_P1P2(sub, **p):
    """
    Compute the P2/P1 ratio from ICP pulses using a SubPeakDetector plugin (from Sophysa / Donatien Legé).

    This function loads the subject's ICP stream, optionally downsamples it,
    computes the P2/P1 ratio for each pulse in sliding windows to save memory,
    and returns an xarray dataset with ratio values and their corresponding dates.
    """
    cns_reader = pycns.CnsReader(data_path / sub) # load CNS reader
    stream_name = 'ICP'
    icp_stream = cns_reader.streams[stream_name] # select ICP stream

    # Add the plugin directory to sys.path
    plugin_dir = base_folder / 'package_P2_P1'
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))

    # Import necessary plugin modules
    from p2p1.subpeaks import SubPeakDetector

    # Define local function to compute P2/P1 from ICP signal
    def icp_to_P2P1(icp_sig, icp_dates, srate):
        sd = SubPeakDetector(all_preds=False)
        srate_detect = int(np.round(srate))
        sd.detect_pulses(signal=icp_sig, fs=srate_detect)
        onsets_inds, ratio_P1P2_vector = sd.compute_ratio()
        onsets_inds[onsets_inds >= icp_dates.size] = icp_dates.size - 1
        onsets_dates = icp_dates[onsets_inds]
        if onsets_dates.size == ratio_P1P2_vector.size + 1:
            onsets_dates = onsets_dates[:-1]
        elif onsets_dates.size == ratio_P1P2_vector.size - 1:
            ratio_P1P2_vector = ratio_P1P2_vector[:-1]
        return list(ratio_P1P2_vector), list(onsets_dates)

    datetimes = icp_stream.get_times() # get datetime stamps
    times = icp_stream.get_times(as_second=True) # times in seconds
    srate = 1/(np.median(np.diff(times))) # compute sampling rate
    raw_signal = icp_stream.get_data(with_times=False, apply_gain=True) # load ICP signal
    raw_signal[np.isnan(raw_signal)] = np.nanmedian(raw_signal) # replace NaNs

    if p['down_sample']: # optional downsampling
        down_sample_factor = 2
        raw_signal = scipy.signal.decimate(raw_signal, q=down_sample_factor)
        datetimes = datetimes[::down_sample_factor]

    total_load_size = raw_signal.size
    win_load_size = int(p['win_compute_duration_hours'] * 3600 * srate) # window size in samples
    n_wins = total_load_size // win_load_size # number of windows

    ratio_P1P2_vector_list_values = [] # store P2/P1 values
    ratio_P1P2_vector_list_dates = [] # store corresponding dates

    start = 0
    for i in range(n_wins): # loop over windows to save memory
        stop = start + win_load_size
        local_icp = raw_signal[start:stop]
        local_dates = datetimes[start:stop]

        try:
            ratio_P1P2_vector, onsets_dates = icp_to_P2P1(local_icp, local_dates, srate)
        except:
            start = stop
            continue

        ratio_P1P2_vector_list_values.extend(ratio_P1P2_vector)
        ratio_P1P2_vector_list_dates.extend(onsets_dates)

        start = stop

    try: # process any remaining signal
        last_values, last_dates = icp_to_P2P1(raw_signal[stop:], datetimes[stop:], srate)
        ratio_P1P2_vector_list_values.extend(last_values)
        ratio_P1P2_vector_list_dates.extend(last_dates)
    except:
        pass

    ratio_P1P2_da = xr.DataArray( # create DataArray
        data=ratio_P1P2_vector_list_values,
        dims=['date'],
        coords={'date': np.array(ratio_P1P2_vector_list_dates, dtype='datetime64[ns]')}
    )
    ds = xr.Dataset()
    ds['ratio_P1P2'] = ratio_P1P2_da # assign DataArray
    return ds # return dataset


def test_ratio_P1P2(sub):
    print(sub)
    ds = ratio_P1P2(sub, **ratio_P1P2_params)
    print(ds['ratio_P1P2'])

ratio_P1P2_job = jobtools.Job(precomputedir, 'ratio_P1P2', ratio_P1P2_params, ratio_P1P2)
jobtools.register_job(ratio_P1P2_job)


# HEART AND RESP SPECTRAL AMPLITUDES IN ICP

def compute_heart_resp_spectral_ratio_in_icp(icp, srate, wsize_secs=50, resp_fband=(0.12, 0.6), heart_fband=(0.8, 2.5), rolling_N_time=5):
    """
    Compute the ratio of heart to respiratory spectral amplitude in ICP.

    This function calculates the spectrogram of the ICP signal, extracts the
    maximum amplitudes and frequencies in the respiration and heart bands,
    applies a rolling median to smooth the results, and returns the ratio of
    heart to respiration amplitude over time.
    """
    nperseg = int(wsize_secs * srate) # segment length in samples
    nfft = int(nperseg) # FFT length

    # Compute ICP spectrogram
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs=srate, nperseg=nperseg, nfft=nfft)

    # For a sinusoid of physical amplitude A, the PSD peak is PSD_peak ~ A^2 / (2 * df), where
    # df is the frequency resolution of the spectrogram (srate / nfft = 1 / wsize_secs).
    # The amplitude is therefore A = sqrt(2 * df * PSD_peak), not sqrt(2 * PSD_peak): without
    # the df factor the amplitude was underestimated by a constant sqrt(wsize_secs) (~7.75 for 60 s).
    df = 1/wsize_secs
    Sxx_icp = np.sqrt(Sxx_icp * 2 * df)  # PSD -> amplitude (mmHg)

    da = xr.DataArray( # create DataArray
        data=Sxx_icp,
        dims=['freq', 'time'],
        coords={'freq': freqs, 'time': times_spectrum_s}
    )

    # Respiratory amplitude and frequency
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1], :].max('freq').rolling(time=rolling_N_time).median().bfill('time').ffill('time') # amplitude of the resp spectral peak
    resp_freq = da.loc[resp_fband[0]:resp_fband[1], :].idxmax('freq').rolling(time=rolling_N_time).median().bfill('time').ffill('time')  # frequency of the resp spectral peak

    # Heart amplitude and frequency
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1], :].max('freq').rolling(time=rolling_N_time).median().bfill('time').ffill('time') # amplitude of the heart spectral peak
    heart_freq = da.loc[heart_fband[0]:heart_fband[1], :].idxmax('freq').rolling(time=rolling_N_time).median().bfill('time').ffill('time') # frequency of the heart spectral peak

    # Compute heart/respiration amplitude ratio
    ratio_heart_resp = heart_amplitude / resp_amplitude

    res = { # return results as dictionary
        'times': times_spectrum_s,
        'heart_in_icp_spectrum': heart_amplitude.values,
        'heart_freq_from_icp': heart_freq,
        'resp_in_icp_spectrum': resp_amplitude.values,
        'resp_freq_from_icp': resp_freq,
        'ratio_heart_resp_in_icp_spectrum': ratio_heart_resp.values
    }

    return res


def heart_resp_in_icp(sub, **p):
    """
    Compute heart and respiration spectral features from ICP signal.

    This function loads the subject's ICP stream, calculates the spectrogram,
    extracts heart and respiratory amplitudes and frequencies using
    compute_heart_resp_spectral_ratio_in_icp(), maps the spectrogram times to
    datetimes, and returns an xarray dataset with all features.
    """
    cns_reader = pycns.CnsReader(data_path / sub) # load CNS reader
    stream_name = 'ICP'
    icp_stream = cns_reader.streams[stream_name] # select ICP stream
    raw_times_referenced = icp_stream.get_times(as_second=True) # times in seconds
    srate = 1/(np.median(np.diff(raw_times_referenced))) # compute sampling rate
    raw_signal, dates = icp_stream.get_data(with_times=True, apply_gain=True) # load ICP signal with timestamps
    raw_times = np.arange(raw_signal.size) / srate # time vector

    # Compute heart/resp spectral ratio in ICP
    res = compute_heart_resp_spectral_ratio_in_icp(
        raw_signal,
        srate,
        wsize_secs=p['spectrogram_win_size_secs'],
        resp_fband=p['resp_fband'],
        heart_fband=p['heart_fband'],
        rolling_N_time=p['rolling_N_time_spectrogram'],
    )

    datetimes = dates[np.searchsorted(raw_times, res['times'])] # map times to datetimes

    df_res = pd.DataFrame(res) # convert to DataFrame
    df_res['times'] = datetimes.copy()
    df_res = df_res.rename(columns={'times': 'datetime'}).set_index('datetime')

    da = xr.DataArray( # create DataArray with feature x datetime
        data=df_res.values.T,
        dims=['feature', 'datetime'],
        coords={'feature': df_res.columns.tolist(), 'datetime': df_res.index}
    )

    ds = xr.Dataset()
    ds['heart_resp_in_icp'] = da # assign DataArray
    return ds # return Dataset


def test_heart_resp_in_icp(sub):
    print(sub)
    ds = heart_resp_in_icp(sub, **heart_resp_in_icp_params)
    print(ds)

heart_resp_in_icp_job = jobtools.Job(precomputedir, 'heart_resp_in_icp', heart_resp_in_icp_params, heart_resp_in_icp)
jobtools.register_job(heart_resp_in_icp_job)


# ICP BY RESP CYCLE

def icp_by_resp_cycle(sub, **p):
    """
    Compute ICP signal aligned to respiratory cycles.

    This function loads the ICP stream, applies a bandpass filter, loads
    respiration cycles, optionally applies mono- or bi-segment deformation,
    and returns the ICP signal aligned to respiratory cycles as an xarray dataset.
    """
    cns_reader = pycns.CnsReader(data_path / sub) # load CNS reader
    icp_stream = cns_reader.streams['ICP'] # select ICP stream
    times_icp = icp_stream.get_times(as_second=True) # times in seconds
    srate_icp = 1/(np.median(np.diff(times_icp))) # compute sampling rate
    raw_icp = icp_stream.get_data(with_times=False, apply_gain=True) # load signal
    icp_filtered = iirfilt(raw_icp, srate_icp, highcut=p['highcut'], order=p['order'], ftype=p['ftype']) # filter signal

    resp_cycles = detect_resp_job.get(sub).to_dataframe() # load respiration cycles
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_icp[0]) & (resp_cycles['next_inspi_time'] < times_icp[-1])] # restrict cycles
    med_cycle_ratio = resp_cycles['cycle_ratio'].median() # median inspiration ratio

    # Define cycle times and segment ratios
    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values
        segment_ratios = med_cycle_ratio

    # Deform ICP signal into cycle template
    icp_by_resp_cycle = physio.deform_traces_to_cycle_template(
        data=icp_filtered,
        times=times_icp,
        cycle_times=cycle_times,
        segment_ratios=segment_ratios,
        points_per_cycle=p['points_per_cycle']
    )

    da = xr.DataArray( # create DataArray with cycle x phase
        data=icp_by_resp_cycle,
        dims=['cycle_date', 'phase'],
        coords={
            'cycle_date': resp_cycles['inspi_date'].values,
            'phase': np.linspace(0, 1, p['points_per_cycle'])
        },
        attrs={'cycle_ratio': segment_ratios}
    )

    ds = xr.Dataset()
    ds['icp_by_resp_cycle'] = da # assign DataArray
    return ds # return Dataset


def test_icp_by_resp_cycle(sub):
    print(sub)
    ds = icp_by_resp_cycle(sub, **icp_by_resp_cycle_params)
    print(ds)

icp_by_resp_cycle_job = jobtools.Job(precomputedir, 'icp_by_resp_cycle', icp_by_resp_cycle_params, icp_by_resp_cycle)
jobtools.register_job(icp_by_resp_cycle_job)


# ICP PULSE AMPLITUDE BY RESP CYCLE

def pulse_by_resp_cycle(sub, detect_pulse_job, **p):
    """
    Compute pulse amplitude aligned to respiratory cycles.

    This function loads CO2 times, respiration cycles, and detected pulse
    amplitudes, interpolates pulse amplitude on CO2 times, optionally applies
    mono- or bi-segment deformation, and returns an xarray DataArray of pulse
    amplitude per respiratory cycle.
    """
    cns_reader = pycns.CnsReader(data_path / sub) # load CNS reader
    times_co2_referenced = cns_reader.streams['CO2'].get_times(as_second=True) # CO2 times in seconds
    resp_cycles = detect_resp_job.get(sub).to_dataframe() # load respiration cycles
    detections = detect_pulse_job.get(sub).to_dataframe() # load pulse detections

    # Crop CO2 times to pulse detection range
    times_co2_referenced_cropped = times_co2_referenced[
        (times_co2_referenced >= detections['trough_time'].values[0]) &
        (times_co2_referenced < detections['next_trough_time'].values[-1])
    ]

    # Interpolate pulse amplitude on CO2 times
    f = scipy.interpolate.interp1d(
        detections['peak_time'],
        detections['rise_amplitude'],
        fill_value=(min(detections['rise_amplitude']), max(detections['rise_amplitude'])),
        bounds_error=False,
        kind='linear'
    )
    xnew = times_co2_referenced_cropped.copy()
    pulse_amplitude_interpolated = f(xnew)

    # Restrict respiration cycles to cropped CO2 times
    resp_cycles = resp_cycles[
        (resp_cycles['inspi_time'] > times_co2_referenced_cropped[0]) &
        (resp_cycles['next_inspi_time'] < times_co2_referenced_cropped[-1])
    ]

    assert p['segmentation_deformation'] in ['mono', 'bi'], 'should be "mono" or "bi"'

    # Define cycle times and segment ratios
    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time', 'next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time', 'expi_time', 'next_inspi_time']].values
        segment_ratios = resp_cycles['cycle_ratio'].median()

    # Deform pulse amplitude into respiratory cycle template
    pulse_amplitude_by_resp = physio.deform_traces_to_cycle_template(
        data=pulse_amplitude_interpolated,
        times=times_co2_referenced_cropped,
        cycle_times=cycle_times,
        segment_ratios=segment_ratios,
        points_per_cycle=p['points_per_cycle']
    )

    da = xr.DataArray( # create DataArray with cycle x phase
        data=pulse_amplitude_by_resp,
        dims=['cycle_date', 'phase'],
        coords={
            'cycle_date': resp_cycles['inspi_date'].values,
            'phase': np.linspace(0, 1, p['points_per_cycle'])
        },
        attrs={'cycle_ratio': segment_ratios}
    )

    return da # return DataArray


def icp_pulse_by_resp_cycle(sub, **p):
    """
    Compute ICP pulse amplitude aligned to respiratory cycles.

    This function uses pulse_by_resp_cycle() with ICP pulse detections and
    returns an xarray dataset containing the ICP pulse amplitude per respiratory cycle.
    """
    da = pulse_by_resp_cycle(sub, detect_icp_job, **p) # compute ICP pulse by resp cycle
    ds = xr.Dataset()
    ds['icp_pulse_by_resp_cycle'] = da # assign DataArray
    return ds # return Dataset


def test_icp_pulse_by_resp_cycle(sub):
    print(sub)
    ds = icp_pulse_by_resp_cycle(sub, **icp_pulse_by_resp_cycle_params)
    print(ds['icp_pulse_by_resp_cycle'])

icp_pulse_by_resp_cycle_job = jobtools.Job(precomputedir, 'icp_pulse_by_resp_cycle', icp_pulse_by_resp_cycle_params, icp_pulse_by_resp_cycle)
jobtools.register_job(icp_pulse_by_resp_cycle_job)


# COMPUTE

def compute_all():
    # Upstream jobs needed by anais_jobs.py, in dependency order. Jobs are cached in
    # precomputedir; force_recompute=False only computes the missing subjects.
    icp_co2_subs = [(sub,) for sub in get_patient_list(['ICP', 'CO2'])]
    jobtools.compute_job_list(detect_resp_job, icp_co2_subs, force_recompute=False, engine='loop')
    jobtools.compute_job_list(detect_icp_job, icp_co2_subs, force_recompute=False, engine='loop')
    jobtools.compute_job_list(ratio_P1P2_job, icp_co2_subs, force_recompute=False, engine='loop')
    jobtools.compute_job_list(heart_resp_in_icp_job, icp_co2_subs, force_recompute=False, engine='loop')
    jobtools.compute_job_list(icp_by_resp_cycle_job, icp_co2_subs, force_recompute=False, engine='loop')
    jobtools.compute_job_list(icp_pulse_by_resp_cycle_job, icp_co2_subs, force_recompute=False, engine='loop')
    # heart rate, used by the enrichment step of the notebook
    ecg_subs = [(sub,) for sub in get_patient_list(['ICP', 'CO2', 'ECG_II'])]
    jobtools.compute_job_list(detect_ecg_job, ecg_subs, force_recompute=False, engine='loop')

if __name__ == "__main__":
    # test_detect_resp('P0044')
    # test_icp_pulse_by_resp_cycle('P0044')
    compute_all()
