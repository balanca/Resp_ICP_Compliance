import pycns
import numpy as np
import string
import jobtools
import pandas as pd
from configuration import *
import xarray as xr


def get_patients_list_raw(verbose = False):
    patients = [str(x).split('/')[-1] for x in data_path.iterdir() if x.is_dir()]
    if verbose:
        print(patients)
    return patients

def count_sd_events(event_file):
    """Number of SD events in an Events.xml file (0 if the file is missing or has no event).

    The guard on the empty DataFrame is needed since pandas 3: on an empty Series,
    .apply() keeps the original dtype (str) instead of returning booleans, and the
    logical 'or' between two StringArray raises a TypeError.
    """
    if not event_file.is_file():
        return 0
    evs = pd.DataFrame(pycns.read_events_xml(event_file, time_zone='Europe/Paris'))
    if evs.shape[0] == 0:
        return 0
    names = evs['name'].astype(str)
    mask = (names.str.contains('SD') | names.str.contains('sd')) & ~names.str.contains('?', regex=False)
    return int(np.sum(mask))

get_patient_durations_by_stream_params = {}

def get_patient_durations_by_stream(patient, **p):
    n_sd = count_sd_events(data_path / patient / 'Events.xml')
    raw_folder = data_path / patient
    uppercase = list(string.ascii_uppercase)
    patient_type = 'SD_ICU' if patient[1] in uppercase else 'non_SD_ICU'
    durations = pd.DataFrame(index = [0], columns = ['patient','patient_type','N_SDs','stream','duration_mins','duration_hours','duration_days','n_chans','ecog_type'])
    durations['patient'] = patient
    durations['patient_type'] = patient_type
    try:
        cns_reader = pycns.CnsReader(raw_folder)
    except:
        return durations
    else:
        stream_keys = cns_reader.streams.keys()
        counter = 0
        if len(stream_keys) > 0:
            for stream_name in stream_keys:
                dates = cns_reader.streams[stream_name].get_times()
                start = dates[0]
                stop = dates[-1]
                start = np.datetime64(start, 'us')
                stop = np.datetime64(stop, 'us')
                duration_us = np.timedelta64(stop - start)
                duration_mins = duration_us.astype('timedelta64[m]').astype(int)
                duration_hours = duration_mins / 60
                duration_days = duration_mins / (60 * 24)
                n_chans = None
                ecog_type = None
                row = [patient, patient_type, n_sd, stream_name, duration_mins, duration_hours, duration_days, n_chans, ecog_type]
                durations.loc[counter,:] = row
                counter += 1
                if stream_name == 'EEG': 
                    ch_names  = cns_reader.streams[stream_name].channel_names      
                    ecog_chans = [chan for chan in ch_names if 'ECoG' in chan]
                    scalp_chans = [chan for chan in ch_names if not chan in ecog_chans]
                    eeg_types = ['Scalp','ECoG'] if len(ecog_chans) > 0 and len(scalp_chans) > 0 else ['Scalp']
                    for eeg_type in eeg_types:
                        if eeg_type == 'Scalp':
                            n_chans = len(scalp_chans)
                            ecog_type = None
                        elif eeg_type == 'ECoG':
                            n_chans = len(ecog_chans)
                            if n_chans == 8:
                                ecog_type = 'depth'
                            elif n_chans == 6:
                                ecog_type = 'strip'
                        row = [patient, patient_type, n_sd, eeg_type, duration_mins, duration_hours, duration_days,n_chans,ecog_type]
                        durations.loc[counter,:] = row
                        counter += 1
        durations['N_SDs'] = durations['N_SDs'].astype(int)
        return xr.Dataset(durations)

def test_get_patient_durations_by_stream(patient):
    print(patient)
    ds = get_patient_durations_by_stream(patient, **get_patient_durations_by_stream_params)
    print(ds)
    print(ds.to_dataframe())

get_patient_durations_by_stream_job = jobtools.Job(precomputedir, 'get_patient_durations_by_stream', get_patient_durations_by_stream_params, get_patient_durations_by_stream)

jobtools.register_job(get_patient_durations_by_stream_job)

detailed_view_streams_params = {
    'save':True,
    'patient_list':get_patients_list_raw()
}

def detailed_view_streams(key, **p):
    subs = get_patients_list_raw()
    concat = [get_patient_durations_by_stream_job.get(sub).to_dataframe() for sub in subs]
    detailed_view = pd.concat(concat).reset_index(drop = True)
    detailed_view['duration_mins'] = detailed_view['duration_mins'].astype('float64')
    detailed_view['N_SDs'] = detailed_view['N_SDs'].astype('float64')
    detailed_view['duration_hours'] = detailed_view['duration_hours'].astype('float64')
    detailed_view['duration_days'] = detailed_view['duration_days'].astype('float64')
    detailed_view['n_chans'] = detailed_view['n_chans'].astype('float64')
    return xr.Dataset(detailed_view)

def test_detailed_view_streams():
    ds = detailed_view_streams('concat', **detailed_view_streams_params)
    # ds.to_netcdf(base_folder / 'test.nc')
    print(ds.to_dataframe())

detailed_view_streams_job = jobtools.Job(precomputedir, 'detailed_view_streams', detailed_view_streams_params, detailed_view_streams)

jobtools.register_job(detailed_view_streams_job)

def get_patient_list(stream_selection = None, patient_type = None, threshold_duration_mins = 120, threshold_N_SDs = 0, verbose = False):
    """
    Function aiming to get a list of patient based on various criteria (stream availability, patient belonging to SD ICU or not, minimal stream duration).
    
    ----------
    Parameters
    ----------
    - stream_selection : None or list or str
        If None, no selection based of this criteria. Select all pycns readable patients if set to 'readable'. Select pycns readable patients with ICP and ABP available if ['ICP','ABP'] for example. Default = None
    - patient_type : None or str
        If None, no selection is applied. If 'SD_ICU' or 'non_SD_ICU', select patients based on patients belonging to this criteria. Default = None
    - threshold_duration_mins : float or int
        If 0, no selection is applied. Else, apply a selection based on streams with durations of at least this parameter. Default = 120 minutes
    - threshold_N_SDs : int
        Filter patient based on number of SD manually detected by Baptiste in events
    - verbose : bool
        If True, print some informations about the amount of patients selected from the total list. Default = False.

    -------
    Returns
    -------
    - sub_list : List of patients
    """
    detailed_view = detailed_view_streams_job.get('concat').to_dataframe()
    all_patients = list(detailed_view['patient'].unique())

    if threshold_N_SDs > 0:
        patient_n_sds_selected = list(detailed_view[detailed_view['N_SDs'] >= threshold_N_SDs]['patient'].unique())

    if patient_type is None:
        sub_list = all_patients
    else:
        detailed_view = detailed_view[detailed_view['patient_type'] == patient_type]

    
    if stream_selection is None:
        sub_list = list(detailed_view['patient'].unique())

    elif stream_selection == 'readable': 
        vcount = detailed_view['patient'].value_counts()
        mask_vcount = vcount > 1
        sub_list = list(mask_vcount.index[mask_vcount])
    else:
        mask = (detailed_view['stream'].isin(stream_selection)) & (detailed_view['duration_mins']>=threshold_duration_mins)
        masked_df = detailed_view[mask]
        vcount = masked_df['patient'].value_counts()
        sub_list = list(vcount.index[vcount == len(stream_selection)])

    if threshold_N_SDs > 0:
        sub_list = [sub for sub in sub_list if sub in patient_n_sds_selected]

    if not stream_selection is None:
        if 'CO2' in stream_selection:
            flat_co2_subs = ['P0061','P0095']   # flat CO2 trace, no detectable cycle
            sub_list = [s for s in sub_list if not s in flat_co2_subs]

    if verbose:
        n_before_selection = len(all_patients)
        n_after_selection = len(sub_list)
        print(f'Selection based on streams {stream_selection} and patient of type {patient_type} with at least {threshold_duration_mins} minutes of recording and at least {threshold_N_SDs} SDs kept {n_after_selection} patients from {n_before_selection} initially')
    return sub_list

# RUN JOB

def compute_all():
    jobtools.compute_job_list(get_patient_durations_by_stream_job, [(sub,) for sub in get_patients_list_raw()], force_recompute=False, engine='joblib', n_jobs = 10)
    jobtools.compute_job_list(detailed_view_streams_job, [('concat',)], force_recompute=True, engine = 'loop')

if __name__ == '__main__':
    # test_get_patient_durations_by_stream('P0044')
    # test_detailed_view_streams()
    # print(get_patient_list(['EEG','CO2'], verbose = True))
    compute_all()
