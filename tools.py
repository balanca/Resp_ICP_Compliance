import pandas as pd
import pycns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
from configuration import *
from scipy import signal
import physio
import matplotlib.dates as mdates
from pycns import CnsReader
import scipy
from matplotlib.colors import Normalize
import itertools
import seaborn as sns

def compute_resphrv_amplitude_fast(resp_cycles, ecg_peaks, limits = [30, 200]):
    ecg_peaks2 = ecg_peaks.copy()
    columns = ['peak_value','trough_value','decay_amplitude']
    resphrv_amplitudes_by_cycle = pd.DataFrame(index = resp_cycles.index, columns=columns, dtype = float)
    r_peak_phase = physio.time_to_cycle(ecg_peaks['peak_time'].values, cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values, segment_ratios = None)
    ecg_peaks2['resp_cycle_ind'] = r_peak_phase - r_peak_phase % 1
    diffs = ecg_peaks2.groupby('resp_cycle_ind')['peak_time'].diff()
    ecg_peaks2['hr_bpm'] = 60.0 / diffs
    if not limits is None:
        bads_hr_mask = (ecg_peaks2['hr_bpm'] < limits[0]) | (ecg_peaks2['hr_bpm'] > limits[1])
        ecg_peaks2.loc[bads_hr_mask, 'hr_bpm'] = np.nan  
    grouped = ecg_peaks2.groupby('resp_cycle_ind')['hr_bpm']
    resphrv_amplitude_min = grouped.min()
    resphrv_amplitude_max = grouped.max()
    resphrv_amplitude_delta = resphrv_amplitude_max - resphrv_amplitude_min
    resphrv_amplitudes_by_cycle.loc[resphrv_amplitude_delta.index, 'peak_value'] = resphrv_amplitude_max
    resphrv_amplitudes_by_cycle.loc[resphrv_amplitude_delta.index, 'trough_value'] = resphrv_amplitude_min
    resphrv_amplitudes_by_cycle.loc[resphrv_amplitude_delta.index, 'decay_amplitude'] = resphrv_amplitude_delta

    count_peaks_by_cycle = ecg_peaks2.groupby('resp_cycle_ind')['hr_bpm'].count() < 3
    not_enough_r_peaks_cycle_inds = count_peaks_by_cycle.index[count_peaks_by_cycle]
    if not_enough_r_peaks_cycle_inds.size > 0:
        resphrv_amplitudes_by_cycle.loc[not_enough_r_peaks_cycle_inds,:] = np.nan
    return resphrv_amplitudes_by_cycle

def get_abp_stream_name_from_reader(reader):
    stream_names = list(reader.streams.keys())
    abp_stream_names = [name for name in stream_names if len(name) == 3 and name[0] == 'A']
    if len(abp_stream_names) == 2:
        abp_stream_name = 'ART'
    elif len(abp_stream_names) == 1:
        abp_stream_name = abp_stream_names[0]
    elif len(abp_stream_names) == 0:
        abp_stream_name = None
    return abp_stream_name

def compute_nanmedian_mad(data, axis=0):
    """
    Compute median and mad

    Parameters
    ----------
    data: np.array
        An array
    axis: int (default 0)
        The axis 
    Returns
    -------
    med: 
        The median
    mad: 
        The mad
    """
    med = np.nanmedian(data, axis=axis)
    mad = np.nanmedian(np.abs(data - med), axis=axis) / 0.6744897501960817
    return med, mad

def get_datetime_edges_from_stream(stream):
    d0 = stream.index['datetime'][0]
    sample_interval_us = stream.index[-1]['sample_interval_integer'] + (stream.index[-1]['sample_interval_fract'] / 2 **32)
    d1 = stream.index['datetime'][-1] + ((stream.shape[0] - stream.index[-1]['sample_ind']) * sample_interval_us).astype('timedelta64[us]')
    return d0, d1

def get_stream_index_datetime_edges(sub, stream_name = None):
    raw_folder = data_path / sub
    cns_reader = CnsReader(raw_folder)
    if stream_name is None:
        min_ = min([cns_reader.streams[name].index["datetime"][0] for name in cns_reader.streams.keys()])
        max_ = max([cns_reader.streams[name].index["datetime"][-1] for name in cns_reader.streams.keys()])
    else:
        min_ = cns_reader.streams[stream_name].index["datetime"][0]
        max_ = cns_reader.streams[stream_name].index["datetime"][-1]
    return min_, max_

def filter_events(events_df, pattern, mode = 'without', output_mode = 'df'):
    """
    Filter events dataframe based one a string pattern used to keep or reject it.
    Parameters:
        events_df : pd.DataFrame , containing the events with a column 'name' on which the patterns will be filtered
        pattern : str , set the pattern that will be used to filter the column 'name'
        mode : str , 'with' or 'without' to keep or reject the events corresponding to the pattern, resepectively. Default = 'without' = rejection mode.
        output_mode : str , set the output of the function as being a dataframe already filtered if set to 'df' or a boolean mask if set to 'df'. Default = 'df'
    """
    assert mode in ['with','without'], f"'{mode}' search mode not possible.'mode' parameter should be set by 'with' or 'without'"
    assert output_mode in ['df','mask'], f"'{output_mode}' output_mode not possible.'output_mode' parameter should be set by 'df' or 'mask'"
    if mode == 'without':
        mask = events_df['name'].apply(lambda x:False if pattern in x else True)
    elif mode == 'with':
        mask = events_df['name'].apply(lambda x:True if pattern in x else False)
    if output_mode == 'df':
        return events_df[mask].reset_index(drop = True)
    elif output_mode == 'mask':
        return mask

def iqr_interval(a, round_=2, format = 'str'):
    q25, q75 = np.nanquantile(a, 0.25), np.nanquantile(a, 0.75)
    if format == 'str':
        return f'[{q25.round(round_)}, {q75.round(round_)}]' # "If your audience is international, the comma format may be more intuitive,  it's best to include a space inside the brackets for readability"
    elif format == 'tuple':
        return (q25, q75)
    
def med_iqr(a, round_=2):
    q25, q50, q75 = np.nanquantile(a, 0.25), np.nanquantile(a, 0.50), np.nanquantile(a, 0.75)
    return f'{q50.round(round_)} [{q25.round(round_)}, {q75.round(round_)}]' # "If your audience is international, the comma format may be more intuitive,  it's best to include a space inside the brackets for readability"

def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(a*np.log(a))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def notch_filter(sig, srate, bandcut = (48,52), order = 4, ftype = 'butter', show = False, axis = -1):

    """
    IIR-Filter to notch/cut 50 Hz of signal
    """

    band = [bandcut[0], bandcut[1]]
    Wn = [e / srate * 2 for e in band]
    sos = signal.iirfilter(order, Wn, analog=False, btype='bandstop', ftype=ftype, output='sos')
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if show:
        w, h = signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

def get_patient_dates(patient): # to remove but keep this old func that recruites the new one for continuity of codes
    return get_stream_index_datetime_edges(patient)

def get_metadata(sub = None):
    """
    Inputs
        sub : str id of patient to get its metadata or None if all metadata. Default is None
    Ouputs 
        pd.DataFrame or pd.Series
    """
    if sub is None:
        return pd.read_excel(metadata_file)
    else:
        return pd.read_excel(metadata_file).set_index('ID_pseudo').loc[sub,:]

def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = -1):

    """
    IIR-Filter of signal
    -------------------
    Inputs : 
    - sig : 1D numpy vector
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig


def plot_frequency_response(srate, lowcut=None, highcut=None, order = 4, ftype = 'butter'):
    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    w, h = signal.sosfreqz(sos,fs=srate)
    fig, ax = plt.subplots()
    ax.plot(w, np.abs(h))
    ax.set_title('Frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    plt.show()

def get_amp(sig, axis = -1):
    analytic_signal = signal.hilbert(sig, axis = axis)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def crosscorrelogram(a, b, bins):
    """
    Lazy implementation of crosscorrelogram.
    """
    diff = a[:, np.newaxis] - b[np.newaxis, :]
    count, bins = np.histogram(diff, bins)
    return count, bins

def get_rate_variablity(cycles, rate_bins, bin_size_min, colname_date, colname_time, units):
    times = cycles[colname_date].values

    start = times[0]
    stop = times[-1]
    delta = np.timedelta64(int(bin_size_min*60), 's')
    time_bins = np.arange(start, stop, delta)

    rate_dist = np.zeros((time_bins.size - 1, rate_bins.size - 1)) * np.nan
    rate = np.zeros(time_bins.size - 1) * np.nan
    rate_variability = np.zeros(time_bins.size - 1) * np.nan

    for i in range(time_bins.size - 1):

        t0, t1 = time_bins[i], time_bins[i+1]

        keep = (cycles[colname_date] > t0) & (cycles[colname_date] < t1)
        cycles_keep = cycles[keep]

        if cycles_keep.shape[0] < 2:
            continue

        d = np.diff(cycles_keep[colname_time].values)
        if units == 'Hz':
            r = 1 / d
        elif  units == 'bpm':
            r = 60 / d
        else:
            raise ValueError(f'bad units {units}')

        count, bins = np.histogram(r, bins=rate_bins, density=True)
        rate_dist[i, :] = count
        rate[i], rate_variability[i] = physio.compute_median_mad(r)

    results = dict(
    time_bins=time_bins,
    rate_bins=rate_bins,
    rate_dist=rate_dist,
    rate=rate,
    rate_variability=rate_variability,
    units=units,
    )
    return results


def plot_variability(results, ratio_saturation=4, ax=None, plot_type = '2d', color='red'):
    globals().update(results)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    if plot_type == '2d':
        
        im = ax.imshow(rate_dist.T, origin='lower', aspect='auto', interpolation='None',
                 extent=[mdates.date2num(time_bins[0]), mdates.date2num(time_bins[-1]),
                         rate_bins[0], rate_bins[-1]])
        ax.plot(mdates.date2num(time_bins[:-1]), rate, color=color)
        ax.set_ylabel(f'rate [{units}]')

        im.set_clim(0, np.nanmax(rate_dist) / ratio_saturation)
    
    elif plot_type == '1d':
        ax.plot(mdates.date2num(time_bins[:-1]), rate_variability, color=color)
        ax.set_ylabel(f'rate variability [{units}]')  
    return ax

def compute_icp(raw_icp, srate, date_vector = None, lowcut = 0.08, highcut = 10, order = 4, ftype = 'butter', peak_prominence = 15, h_distance_s = 0.5, rise_amplitude_limits = (0,20), amplitude_at_trough_low_limit = -10, verbose = False, show = False):
    icp_filt = iirfilt(raw_icp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    maximums,_ = scipy.signal.find_peaks(icp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    minimums,_ = scipy.signal.find_peaks(-icp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    if minimums[0] > maximums[0]: # first point detected has to be a minimum
        maximums = maximums[1:] # so remove the first maximum if is before first minimum
    if maximums[-1] > minimums[-1]: # last point detected has to be a minimum
        maximums = maximums[:-1] # so remove the last maximum if is after last minimum

    peak_index = maximums
    trough_index = minimums[np.searchsorted(minimums, peak_index) - 1]

    detection = pd.DataFrame()
    detection['trough_ind'] = trough_index
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = trough_index[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peak_index
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['next_trough_ind'] = detection['next_trough_ind'].astype(int)
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']

    detection['amplitude_at_trough'] = raw_icp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_icp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_icp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']

    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']].astype('datetime64[ns]')
        detection['peak_date'] =  date_vector[detection['peak_ind']].astype('datetime64[ns]')
        detection['next_trough_date'] =  date_vector[detection['next_trough_ind']].astype('datetime64[ns]')
    
    #cleaning
    detection_clean = detection.copy()
    detection_clean = detection_clean[(detection_clean['amplitude_at_trough'] >= amplitude_at_trough_low_limit)]
    detection_clean = detection_clean[(detection_clean['rise_amplitude'] >= rise_amplitude_limits[0]) & (detection_clean['rise_amplitude'] <= rise_amplitude_limits[1]) ]

    if verbose:
        print("{n_removed} abp cycles were removed by cleaning".format(n_removed = detection.shape[0] - detection_clean.shape[0]))

    if show:
        t = np.arange(raw_icp.size) / srate
        fig, ax = plt.subplots()
        ax.plot(t, raw_icp)
        ax.scatter(t[detection_clean['trough_ind']], raw_icp[detection_clean['trough_ind']], color = 'r')
        ax.scatter(t[detection_clean['peak_ind']], raw_icp[detection_clean['peak_ind']], color = 'g')
        plt.show()

    return detection_clean.reset_index(drop = True)

def compute_abp(raw_abp, srate, date_vector = None, lowcut = 0.5, highcut = 10, order = 1, ftype = 'bessel', peak_prominence = 15, h_distance_s = 0.3, rise_amplitude_limits = (15,200), amplitude_at_trough_low_limit = 20, range_first_derivative = 3, compute_cardiac_output = True, show = False, verbose = False):
    assert not np.any(np.isnan(raw_abp)), 'Nans in ABP sig'
    abp_filt = iirfilt(raw_abp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    maximums,_ = scipy.signal.find_peaks(abp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    minimums,_ = scipy.signal.find_peaks(-abp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    if minimums[0] > maximums[0]: # first point detected has to be a minimum
        maximums = maximums[1:] # so remove the first maximum if is before first minimum
    if maximums[-1] > minimums[-1]: # last point detected has to be a minimum
        maximums = maximums[:-1] # so remove the last maximum if is after last minimum

    peak_index = maximums
    trough_index = minimums[np.searchsorted(minimums, peak_index) - 1]

    detection = pd.DataFrame()
    detection['trough_ind'] = trough_index
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = trough_index[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peak_index
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['next_trough_ind'] = detection['next_trough_ind'].astype(int)
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']

    detection['amplitude_at_trough'] = raw_abp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_abp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_abp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']

    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']].astype('datetime64[ns]')
        detection['peak_date'] =  date_vector[detection['peak_ind']].astype('datetime64[ns]')
        detection['next_trough_date'] =  date_vector[detection['next_trough_ind']].astype('datetime64[ns]')
    
    #cleaning
    detection_clean = detection.copy()
    detection_clean = detection_clean[(detection_clean['amplitude_at_trough'] >= amplitude_at_trough_low_limit)]
    detection_clean = detection_clean[(detection_clean['rise_amplitude'] >= rise_amplitude_limits[0]) & (detection_clean['rise_amplitude'] <= rise_amplitude_limits[1]) ]
    detection_clean = detection_clean.reset_index(drop = True)


    if compute_cardiac_output:
        # dicrotic

        first_derivative = np.gradient(abp_filt)
        second_derivative = np.gradient(first_derivative)

        mask_amp = (abp_filt > np.quantile(abp_filt, 0.25)) & (abp_filt < np.quantile(abp_filt, 0.75))
        mask_1st_derivative = (first_derivative > -range_first_derivative) & (first_derivative < range_first_derivative) # search where first derivative is approachnig 0
        mask_2nd_derivative = second_derivative > 0 # search positive inflexion point

        mask = mask_amp & mask_1st_derivative & mask_2nd_derivative
        crossings = detect_cross(mask, 0.5)
        crossings['rises'] = crossings['rises'] + 1

        detection_clean['dicrotic_notch_ind'] = np.nan
        detection_clean['auc_cardiac_output'] = np.nan

        for i, row in detection_clean.iterrows():
            peak_ind = int(row['peak_ind'])
            next_trough_ind = int(row['next_trough_ind'])

            local_crossings = crossings[(crossings['rises'] > peak_ind) & (crossings['decays'] < next_trough_ind)]

            if local_crossings.shape[0] > 0:
                local_crossings = local_crossings.iloc[0,:]

                first_derivation_dicrotic_zone = first_derivative[local_crossings['rises']:local_crossings['decays']]
                change_derivative_sign_in_dicrotic_zone = (first_derivation_dicrotic_zone[:-1] <= 0) & (first_derivation_dicrotic_zone[1:] > 0)
                if np.any(change_derivative_sign_in_dicrotic_zone):
                    dicrotic_notch_ind = np.where(change_derivative_sign_in_dicrotic_zone)[0][0] + local_crossings['rises']
                else:
                    dicrotic_notch_ind = local_crossings['rises']

                local_sig_cardiac_output = raw_abp[peak_ind:dicrotic_notch_ind] - row['amplitude_at_trough']
                detection_clean.loc[i, 'dicrotic_notch_ind'] = dicrotic_notch_ind
                detection_clean.loc[i,'auc_cardiac_output'] = np.trapz(local_sig_cardiac_output)
        detection_clean['dicrotic_notch_time'] = detection_clean['dicrotic_notch_ind'] / srate
        detection_clean['dicrotic_notch_ind'] = detection_clean['dicrotic_notch_ind'].astype('Int64')
        detection_clean['dicrotic_notch_amplitude'] = raw_abp[detection_clean['dicrotic_notch_ind']]

        if not date_vector is None:
            detection_clean['dicrotic_notch_date'] =  date_vector[detection_clean['dicrotic_notch_ind']].astype('datetime64[ns]')
        
    if verbose:
        print("{n_removed} abp cycles were removed by cleaning".format(n_removed = detection.shape[0] - detection_clean.shape[0]))

    if show:
        t = np.arange(raw_abp.size) / srate
        fig, ax = plt.subplots()
        ax.plot(t, raw_abp)
        ax.scatter(t[detection_clean['trough_ind']], raw_abp[detection_clean['trough_ind']], color = 'r')
        ax.scatter(t[detection_clean['peak_ind']], raw_abp[detection_clean['peak_ind']], color = 'g')
        plt.show()

    # return detection_clean.reset_index(drop = True), mask, crossings
    return detection_clean.reset_index(drop = True)

def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def morlet_family(srate, f_start, f_stop, n_steps, n_cycles, duration_s = 'auto'):
    """
    Create a family of morlet wavelets
    
    ------------------------------
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_steps : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet. If it is an integer, all wavelets have this same number of cycles, else if a tuple, number of cycles increase linearly from the first to the second element.
    duration_s : duration in seconds of the kernel
    """
    freqs = np.linspace(f_start,f_stop,n_steps)
    if isinstance(n_cycles, int):
        n_cycles_vector = np.ones(freqs.shape) * n_cycles
    elif isinstance(n_cycles, tuple):
        n_cycles_vector = np.ones(freqs.shape) * np.linspace(n_cycles[0], n_cycles[1], freqs.size)
    duration_s = 2 * n_cycles_vector[0] * (srate / freqs[0]) 
    tmw = np.arange(-duration_s/2,duration_s/2,1/srate)
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        n_cycles
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles_vector[i], freq = fi)
    return tmw, n_cycles_vector, freqs, mw_family

def morlet_power(sig, srate, f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2, duration_s = 'auto'):
    """
    Compute time-frequency matrix by convoluting wavelets on a signal
    
    ------------------------------
    Inputs =
    - sig : the signal (1D np vector)
    - srate : sampling rate
    - f_start : lowest frequency of the wavelet family
    - f_stop : highest frequency of the wavelet family
    - n_steps : number of frequencies from f_start to f_stop
    - n_cycles : number of waves in the wavelet
    - amplitude_exponent : amplitude values extracted from the length of the complex vector will be raised to this exponent factor (default = 2 = V**2 as unit)

    Outputs = 
    - freqs : frequency 1D np vector
    - power : 2D np array , axis 0 = freq, axis 1 = time

    """
    _, _, freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_steps = n_steps, n_cycles = n_cycles, duration_s=duration_s)
    sigs = np.tile(sig, (n_steps,1))
    tf = signal.fftconvolve(sigs, family, mode = 'same', axes = 1)
    power = np.abs(tf) ** amplitude_exponent
    return freqs , power


def smooth_per_freq(matrix, srate, sigma_s_per_freq):
    """Row-by-row smoothing with a different sigma_s for each frequency."""

    n_freqs, n_times = matrix.shape
    output = np.zeros_like(matrix, dtype=matrix.dtype)
    
    for i in range(n_freqs):
        sigma_samples = sigma_s_per_freq[i] * srate
        half = int(4 * sigma_samples)
        t = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
        kernel /= kernel.sum()
        output[i] = scipy.signal.fftconvolve(matrix[i], kernel, mode='same')
    
    return output

def morlet_coherence(sig1, sig2, srate, f_start, f_stop, n_steps, n_cycles, sigma_s = 'auto', smooth_factor = 5):
    _, n_cycles_vector, freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_steps = n_steps, n_cycles = n_cycles, duration_s='auto')
    x_tf = scipy.signal.fftconvolve(np.tile(sig1, (n_steps,1)), family, mode = 'same', axes = 1)
    y_tf = scipy.signal.fftconvolve(np.tile(sig2, (n_steps,1)), family, mode = 'same', axes = 1)
    Sxx = np.abs(x_tf)**2
    Syy = np.abs(y_tf)**2
    Sxy = x_tf * np.conj(y_tf)
    sigma_s = smooth_factor * n_cycles_vector / freqs
    Sxx_smooth = smooth_per_freq(Sxx, srate, sigma_s)
    Syy_smooth = smooth_per_freq(Syy, srate, sigma_s)
    Sxy_smooth = smooth_per_freq(Sxy.real, srate, sigma_s) + 1j * smooth_per_freq(Sxy.imag, srate, sigma_s)
    Cxy = np.abs(Sxy_smooth)**2 / (Sxx_smooth * Syy_smooth + 1e-10)
    coords = dict(data  = ['Sxx','Syy','Cxy'], freq = freqs, time = np.arange(sig1.size) / srate)
    da = xr.DataArray(data = np.nan , dims = coords.keys(), coords= coords)
    da.loc['Sxx'] = Sxx_smooth
    da.loc['Syy'] = Syy_smooth
    da.loc['Cxy'] = Cxy
    return da

def compute_spectrum_log_slope(spectrum, freqs, freq_range = [1,40], show = False):
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f_log = np.log(freqs[mask])
    spectrum_log = np.log(spectrum[mask])

    res = scipy.stats.linregress(f_log, spectrum_log)
    a = res.slope
    
    if show:
        b = res.intercept
        fit_log = a * f_log + b
        fit = np.exp(a * f_log + b)
        
        fig, axs = plt.subplots(nrows = 2, figsize = (8,6))
        ax = axs[0]
        ax.plot(f_log, spectrum_log)
        ax.plot(f_log,  fit_log)
        ax.set_title('Slope : {:.3f}'.format(a))
        
        ax = axs[1]
        ax.semilogy(freqs[mask], spectrum[mask])
        ax.semilogy(freqs[mask],  fit)
        plt.show()
    
    return a

def load_one_eeg_chan(eeg_stream, chan, win_load_duration_hours = 0.1, apply_gain=True):
    assert chan in eeg_stream.channel_names, f'EEG stream do not have channel {chan}'
    srate = eeg_stream.sample_rate
    
    total_load_size = eeg_stream.shape[0]
    win_load_size = int(win_load_duration_hours * 3600 * srate)
    n_wins = total_load_size // win_load_size
    
    chan_ind = eeg_stream.channel_names.index(chan)
    
    
    chan_sig = np.zeros(total_load_size)
    start = 0
    for i in range(n_wins):
        stop = start + win_load_size
        chan_sig[start:stop] = eeg_stream.get_data(isel = slice(start,stop), apply_gain = apply_gain)[:,chan_ind]
        start = stop
    chan_sig[stop:] = eeg_stream.get_data(isel = slice(stop,None))[:,chan_ind]
    return chan_sig, srate

def attribute_subplots(element_list, nrows, ncols):
    assert nrows * ncols >= len(element_list), f'Not enough subplots planned ({nrows*ncols} subplots but {len(element_list)} elements)'
    subplots_pos = {}
    counter = 0
    for r in range(nrows):
        for c in range(ncols):
            if counter == len(element_list):
                break
            subplots_pos[f'{element_list[counter]}'] = [r,c]
            counter += 1
    return subplots_pos  

def get_mcolors():
    from matplotlib.colors import TABLEAU_COLORS
    return list(TABLEAU_COLORS.keys())

def detect_cross(sig, thresh):
    rises, = np.where((sig[:-1] <=thresh) & (sig[1:] >thresh)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=thresh) & (sig[1:] <thresh)) # detect where sign inversion from + to -
    if rises.size > 0 and decays.size > 0:
        if rises[0] > decays[0]: # first point detected has to be a rise
            decays = decays[1:] # so remove the first decay if is before first rise
        if rises[-1] > decays[-1]: # last point detected has to be a decay
            rises = rises[:-1] # so remove the last rise if is after last decay
        return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T
    else:
        return None

def compute_prx(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    icp = df_sigs['icp'].values
    abp = df_sigs['abp'].values
    dates = df_sigs['dates'].values
    
    wsize_inds = int(srate * wsize_mean_secs)

    starts = np.arange(0, icp.size, wsize_inds)

    icp_down_mean = np.zeros(starts.size)
    abp_down_mean = np.zeros(starts.size)

    for i, start in enumerate(starts):
        stop = start + wsize_inds
        if stop > icp.size:
            break
        icp_down_mean[i] = np.mean(icp[start:stop])
        abp_down_mean[i] = np.mean(abp[start:stop])

    dates = dates[::wsize_inds]
    
    corrs_wsize_secs = wsize_corr_mins * 60
    n_samples_win = int(corrs_wsize_secs / wsize_mean_secs)

    n_samples_between_starts = n_samples_win - int(overlap_corr_prop * n_samples_win)

    start_win_inds = np.arange(0, icp_down_mean.size, n_samples_between_starts)

    prx_r = np.zeros(start_win_inds.size)
    prx_pval = np.zeros(start_win_inds.size)
    for i, start_win_ind in enumerate(start_win_inds):
        stop_win_ind = start_win_ind + n_samples_win
        if stop_win_ind > icp_down_mean.size:
            stop_win_ind = icp_down_mean.size
        actual_win_size = stop_win_ind - start_win_ind
        if actual_win_size > 2: # in case when last window too short to compute correlation ...
            res = scipy.stats.pearsonr(icp_down_mean[start_win_ind:stop_win_ind], abp_down_mean[start_win_ind:stop_win_ind])
            prx_r[i] = res.statistic
            prx_pval[i] = res.pvalue
        else: # ... fill last value with pre-last value
            prx_r[i] = prx_r[i-1]
            prx_pval[i] = prx_pval[i-1]
            
        
    dates = dates[start_win_inds]
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates

def compute_prx_and_keep_nans(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys() # get all stream names

    # check if ABP or ART stream in available streams
    if 'ABP' in all_streams: 
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data' # check if ICP stream in available streams
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names]) # compute srate for upsampling based on the most sampled stream
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate) # load ICP and blood pressure streams with same datetime basis
    icp = ds['ICP'].values # icp : dataset to numpy
    abp = ds[abp_name].values # abp : dataset to numpy
    dates = ds['times'].values # datetimes : dataset to numpy

    wsize_inds = int(srate * wsize_mean_secs) # compute window size in points for the local averaging

    start_mean_inds = np.arange(0, icp.size, wsize_inds) # compute start inds
    stop_mean_inds = start_mean_inds + wsize_inds # stop inds = start inds + window size in points
    stop_mean_inds[-1] = icp.size - 1 # last stop ind is replaced by the size of original signal to not slice too far

    icp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean icp signal
    abp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean abp signal
    dates_local_mean = dates[stop_mean_inds] # compute date vector of local means by slicing original dates by stop inds
    
    for i, start, stop in zip(np.arange(start_mean_inds.size), start_mean_inds, stop_mean_inds): # loop over start and stop inds
        icp_down_local_mean[i] = np.mean(icp[start:stop]) # compute local mean icp (return Nan if Nan in the window)
        abp_down_local_mean[i] = np.mean(abp[start:stop]) # compute local mean abp (return Nan if Nan in the window)

    n_samples_by_corr_win = int(wsize_corr_mins * 60 / wsize_mean_secs) # compute number of points by correlation window (seconds durations of corr win / seconds duration of local mean win) (= 30 if corr win = 5 mins and local mean win = 10 secs)
    n_samples_between_start_prx_inds = n_samples_by_corr_win - int(overlap_corr_prop * n_samples_by_corr_win) # = 6 if overlap = 80% and n_samples_by_corr_win = 30
    start_prx_inds = np.arange(0, icp_down_local_mean.size, n_samples_between_start_prx_inds) # compute start inds of prx

    prx_r = np.zeros(start_prx_inds.size) # initialize prx vector of shape start_prx_inds.size
    prx_pval = np.zeros(start_prx_inds.size) # initialize prx pval vector of shape start_prx_inds.size
    dates_prx = [] # initialize a list to store datetimes of prx computing
    for i, start_win_ind in enumerate(start_prx_inds): # loop over start inds
        stop_win_ind = start_win_ind + n_samples_by_corr_win  # compute stop ind = start ind + n_samples_by_corr_win
        if stop_win_ind >= icp_down_local_mean.size: # if stop win index higher that size of local mean sig ...
            stop_win_ind = icp_down_local_mean.size # ... computing window will end at the last local mean sig point
            dates_prx.append(dates_local_mean[stop_win_ind-1]) # add a datetime corresponding to local mean date vector sliced with current stop ind - 1
        else:
            dates_prx.append(dates_local_mean[stop_win_ind]) # add a datetime corresponding to local mean date vector sliced with current stop ind
        actual_win_size = stop_win_ind - start_win_ind # compute the window size in points
        if actual_win_size > 2: # check if window size has at least two points to correlate ...
            icp_sig_win = icp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean icp sig
            abp_sig_win = abp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean abp sig
            if np.any(np.isnan(icp_sig_win)) or np.any(np.isnan(abp_sig_win)): # check if nan in the slice of local mean sig and fill with nan if it is the case
                prx_r[i] = np.nan
                prx_pval[i] = np.nan
            elif np.std(icp_sig_win) == 0 or np.std(abp_sig_win) == 0: # if icp or abp is constant
                prx_r[i] = np.nan
                prx_pval[i] = np.nan
            else: # if no nan, compute pearson correlation from scipy
                res = scipy.stats.pearsonr(icp_sig_win, abp_sig_win)
                prx_r[i] = res.statistic
                prx_pval[i] = res.pvalue
        else: # ... else fill with a nan if no two points available to correlate
            prx_r[i] = np.nan
            prx_pval[i] = np.nan
    
    dates_prx = np.array(dates_prx).astype('datetime64')
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates_prx

def compute_homemade_prx(cns_reader, win_size_rolling_mins = 5, highcut_Hz=0.1, ftype = 'bessel', order = 4):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    df_sigs['icp'] = iirfilt(df_sigs['icp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    df_sigs['abp'] = iirfilt(df_sigs['abp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    down_samp_compute = int(srate / (highcut_Hz * 20))
    down_samp_compute = 1 if down_samp_compute < 1 else down_samp_compute
    new_srate = srate / down_samp_compute
    df_sigs = df_sigs.iloc[::down_samp_compute]
    corr = df_sigs['abp'].rolling(int(win_size_rolling_mins * 60 * srate)).corr(df_sigs['icp'])
    corr.index = df_sigs['dates']
    return corr


def init_da(coords, name = None, values = np.nan):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.full(shape, values)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def compute_suppression_ratio(sig, srate, threshold_µV = 5, win_size_sec_epoch = 0.240, win_size_sec_moving = 63):
    mask_not_suppressed = np.abs(sig) > threshold_µV
    t = np.arange(sig.size) / srate
    start_wins = np.arange(0, t[-1], win_size_sec_epoch)
    rows = []
    for i, start_win in enumerate(start_wins):
        stop_win = start_win + win_size_sec_epoch
        start_win_ind = int(start_win * srate)
        stop_win_ind = start_win_ind + int(win_size_sec_epoch * srate)
        if stop_win_ind > sig.size:
            break
        mask_not_suppressed_win = mask_not_suppressed[start_win_ind:stop_win_ind]
        is_suppressed = 0 if np.sum(mask_not_suppressed_win) > 0 else 1
        rows.append([start_win, stop_win, is_suppressed])
    df_suppression = pd.DataFrame(rows, columns = ['start_t','stop_t','is_suppressed'])

    if t[-1] < win_size_sec_moving:
        suppression_ratio = df_suppression['is_suppressed'].mean()
        start_wins = np.array([0])
    else:
        start_wins = np.arange(0, t[-1], win_size_sec_moving)
        suppression_ratio = np.zeros(start_wins.size)
        for i, start_win in enumerate(start_wins):
            stop_win = start_win + win_size_sec_moving
            if stop_win > t[-1]:
                stop_win = t[-1]
            local_df_suppression = df_suppression[(df_suppression['start_t'] >= start_win) & (df_suppression['start_t'] < stop_win)]
            suppression_ratio[i] = local_df_suppression['is_suppressed'].mean()
    suppression_ratio *= 100 # transform ratio into percentage
    return suppression_ratio, start_wins

def compute_suppression_ratio_homemade(sig, srate, threshold_µV = 5, lowcut=0.5, highcut = 40):
    sig_filtered = iirfilt(sig, srate, lowcut, highcut)
    sig_amp = get_amp(sig_filtered)
    return (np.sum(sig_amp < threshold_μV) / sig_amp.size) * 100

def compute_spectral_entropy(power, normalized = True):
    # Normalize the power spectrum
    power /= np.sum(power)
    # Compute entropy
    entropy = -np.sum(power * np.log2(power))
    if normalized:
        entropy = entropy / np.log2(power.size)
    return entropy

def get_crest_line(freqs, Sxx, freq_axis = 0):
    argmax_freqs = np.argmax(Sxx, axis = freq_axis)
    fmax_freqs = np.apply_along_axis(lambda i:freqs[i], axis = freq_axis, arr = argmax_freqs)
    return fmax_freqs

def load_overview_data():
    return pd.read_excel(base_folder / 'overview_data_pycns.xlsx')

def compute_bins(series, nbins = 50, n_mads = 6):
    a = series.values
    max_ = np.nanmax(a)
    min_ = np.nanmin(a)
    name = series.name
    if np.any(np.isnan(a)):
        a = a[~np.isnan(a)]
    med, mad = physio.compute_median_mad(a)
    inf = med - n_mads * mad
    sup = med + n_mads * mad
    if sup > max_:
        sup = max_
    if inf < min_:
        inf = min_
    if name != 'PRx':
        bins = np.linspace(inf if inf > 0 else 0, sup, nbins)
    else:
        bins = np.linspace(inf, sup, nbins)
    return bins

def get_significance(p_value):
    if p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return "ns"

def pairplot_homemade(data, kind = 'hist', mapper_clean_name = None, savefile = None, **kwargs):
    p_labels = kwargs.get("p_labels", {})  # Extract p_labels if provided
    p_ticks = kwargs.get("p_ticks", {})  # Extract p_ticks if provided
    metrics = data.columns.tolist()
    if mapper_clean_name is None:
        mapper_clean_name = {k:v for k,v in zip(metrics,metrics)}
    combinations = [i for i in itertools.combinations(metrics, 2)]

    nrows = len(data.columns)
    ncols = nrows

    norm = Normalize(-1, 1)
    cmap = plt.get_cmap('seismic')

    figsize = (nrows * 3, ncols * 3)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize, constrained_layout = True)

    is_done = pd.DataFrame(index = data.columns, columns = data.columns, dtype = 'bool')
    is_done[:] = False
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            row_metric = data.columns[r]
            col_metric = data.columns[c]
            if row_metric == col_metric and not is_done.iloc[r,c]:
                bins_x = compute_bins(data[row_metric])
                ax.hist(data[row_metric], bins=bins_x, color = 'k')
                ax.set_xlabel(mapper_clean_name[row_metric], **p_labels)
                ax.set_ylabel('Count', **p_labels)
                ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), **p_ticks)
                ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), **p_ticks)
            else:
                if not is_done.iloc[r,c] and not is_done.iloc[c,r]: 
                    bins_x = compute_bins(data[row_metric])
                    bins_y = compute_bins(data[col_metric])
                    if kind == 'hist':
                        sns.histplot(data = data, x = row_metric, y = col_metric, ax = ax, bins = (bins_x,bins_y), color = 'k', pthresh=0., pmax=1)
                    elif kind == 'scatter':
                        sns.scatterplot(data = data, x = row_metric, y = col_metric, ax = ax)
                    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), **p_ticks)
                    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), **p_ticks)
                else:
                    data_sel = data[[row_metric, col_metric]].dropna()
                    res = scipy.stats.spearmanr(data_sel[row_metric], data_sel[col_metric])
                    r_coef = res.statistic
                    p = res.pvalue * len(combinations)
                    p = 1 if p > 1 else p
                    s = 'R : {r_coef}\np : {p}'.format(r_coef=round(r_coef, 3), p = get_significance(p))
                    ax.text(0.5, 0.5, s, ha = 'center', weight = 'bold', fontsize = 20, va = 'center')
                    ax.set_facecolor(cmap(norm(r_coef)))
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_xlabel(mapper_clean_name[row_metric], **p_labels)
                ax.set_ylabel(mapper_clean_name[col_metric], **p_labels)
            is_done.iloc[r,c] = True
    if not savefile is None:
        fig.savefig(savefile, bbox_inches = 'tight', dpi = 500)
    plt.show()

if __name__ == "__main__":

    # compute_prx_and_keep_nans(CnsReader(data_path / 'MF12'))

    # compute_prx(CnsReader(data_path / 'MF12'))

    # print(iqr_interval([np.random.randn(100)]))
    # print(med_iqr([np.random.randn(100)], round_=2))
    morlet_family(srate = 10, f_start = 3, f_stop = 20, n_steps = 10, n_cycles = (4, 15))
