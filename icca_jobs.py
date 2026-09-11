import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import jobtools
from configuration import *
from icca_tools import *
from tools import get_metadata

# PARAMS

icca_biology_params = {
}
icca_clinical_params = {
}
icca_pse_tt_params = {
    'use_human_weight':True
}
icca_medication_tt_params = {
}
icca_csf_params = {
}
resample_bio_params = {
    'icca_biology_params':icca_biology_params,
    'interpolation_kind':'linear',
}
resample_clinical_params = {
    'icca_clinical_params':icca_clinical_params,
    'interpolation_kind':'previous',
}
resample_pse_tt_params = {
    'icca_pse_tt_params':icca_pse_tt_params,
    'interpolation_kind':'previous',
    'n_mins_considering_zero_dose':130, # when this amount of minutes has passed since last available dose, a zero dose is initialized until next available dose
}

qualitative_sedation_level_params = {
    'resample_pse_tt_params':resample_pse_tt_params,
    'sedative_names':['Thiopenthal','Midazolam','Propofol'], # sedative treatments to search
    'edges_datetime_mode':'hospit_date', # hospit_date | sedative_date. I prefer hospit_date but warning to period between start hospit and start sedative where no data available will be filled as set level.
    'bfill_start':True, # backward fill if True and hospit_date mode the start sedative date stage level toward the start hospit
}

# ICCA BIO
def icca_bio(sub, **p):
    ds = load_biology_in_dataset(sub)
    return ds

def test_icca_bio(sub):
    print(sub)
    ds = icca_bio(sub, **icca_biology_params)
    print(ds)

icca_bio_job = jobtools.Job(precomputedir, 'icca_bio', icca_biology_params, icca_bio)
jobtools.register_job(icca_bio_job)

# ICCA CLINICAL
def icca_clinical(sub, **p):
    ds = load_clinical_in_dataset(sub)
    return ds

def test_icca_clinical(sub):
    print(sub)
    ds = icca_clinical(sub, **icca_clinical_params)
    print(ds)

icca_clinical_job = jobtools.Job(precomputedir, 'icca_clinical', icca_clinical_params, icca_clinical)
jobtools.register_job(icca_clinical_job)

# ICCA PSE TT
def icca_pse_tt(sub, **p):
    ds = load_PSE_treatment_in_dataset(sub, use_human_weight=p['use_human_weight'])
    return ds

def test_icca_pse_tt(sub):
    print(sub)
    ds = icca_pse_tt(sub, **icca_pse_tt_params)
    print(ds)

icca_pse_tt_job = jobtools.Job(precomputedir, 'icca_pse_tt', icca_pse_tt_params, icca_pse_tt)
jobtools.register_job(icca_pse_tt_job)

# ICCA MEDICATION TT
def icca_medication_tt(sub, **p):
    ds = load_medication_treatment_in_dataset(sub)
    return ds

def test_icca_medication_tt(sub):
    print(sub)
    ds = icca_medication_tt(sub, **icca_medication_tt_params)
    print(ds)

icca_medication_tt_job = jobtools.Job(precomputedir, 'icca_medication_tt', icca_medication_tt_params, icca_medication_tt)
jobtools.register_job(icca_medication_tt_job)

# ICCA CSF 
def icca_csf(sub, **p):
    ds = load_csf_in_dataset(sub)
    return ds

def test_icca_csf(sub):
    print(sub)
    ds = icca_csf(sub, **icca_csf_params)
    print(ds)

icca_csf_job = jobtools.Job(precomputedir, 'icca_csf', icca_csf_params, icca_csf)
jobtools.register_job(icca_csf_job)

# Resample func minute by minute

def resample_clinical_or_bio(ds, **p): 
    new_ds = xr.Dataset()
    names = list(ds.keys())
    for name in names:
        print(name)
        da = ds[name]
        data = da.values
        if not np.issubdtype(data.dtype, np.number):
            # print(f'{name} is of type {data.dtype} so not resampled')
            new_da = da.copy()
        else:
            initial_datetimes = da[da.dims[0]].values
            d_start, d_stop = initial_datetimes[0], initial_datetimes[-1]
            initial_times = (initial_datetimes - d_start).astype(float) / 1e9 / 60 

            new_times = np.arange(initial_times[0], initial_times[-1] + 1, 1)
            new_datetimes = np.arange(initial_datetimes[0], initial_datetimes[-1] + np.timedelta64(1, 'm'), np.timedelta64(1, 'm'))

            f = scipy.interpolate.interp1d(initial_times, data, fill_value="extrapolate", kind = p['interpolation_kind'])
            new_data = f(new_times)

            new_da = xr.DataArray(data = new_data, coords = {f'datetime_{name}':new_datetimes}, attrs = da.attrs)
        new_ds[name] = new_da
    return new_ds


# RESAMPLE CLINICAL 

def resample_clinical(sub, **p):
    new_ds = resample_clinical_or_bio(icca_clinical_job.get(sub), **p)
    return new_ds

def test_resample_clinical(sub):
    print(sub)
    ds = resample_clinical(sub, **resample_clinical_params)
    print(ds)

resample_clinical_job = jobtools.Job(precomputedir, 'resample_clinical', resample_clinical_params, resample_clinical)
jobtools.register_job(resample_clinical_job)

# RESAMPLE BIO

def resample_bio(sub, **p):
    new_ds = resample_clinical_or_bio(icca_bio_job.get(sub), **p)
    return new_ds

def test_resample_bio(sub):
    print(sub)
    ds = resample_bio(sub, **resample_bio_params)
    print(ds)

resample_bio_job = jobtools.Job(precomputedir, 'resample_bio', resample_bio_params, resample_bio)
jobtools.register_job(resample_bio_job)

# RESAMPLE PSE TT

def resample_pse_tt(sub, **p):
    ds = icca_pse_tt_job.get(sub)
    new_ds = xr.Dataset()
    names = list(ds.keys())
    for name in names:
        da = ds[name]

        dose = da.values
        initial_datetimes = da[da.dims[0]].values
        d_start, d_stop = initial_datetimes[0], initial_datetimes[-1]
        initial_times = (initial_datetimes - d_start).astype(float) / 1e9 / 60 

        mask_where_zeros_should_be = np.diff(initial_times) > p['n_mins_considering_zero_dose']
        mask_where_zeros_should_be = np.append(mask_where_zeros_should_be, False)
        dates_where_zero_should_be = initial_datetimes[mask_where_zeros_should_be] + np.timedelta64(p['n_mins_considering_zero_dose'], 'm')
        dose_with_zeros = dose.copy()
        inds_where_insert_zeros = np.searchsorted(initial_datetimes, dates_where_zero_should_be)
        dose_with_zeros = np.insert(dose_with_zeros,inds_where_insert_zeros , 0)
        initial_datetimes_with_zeros =  np.insert(initial_datetimes, inds_where_insert_zeros, dates_where_zero_should_be)
        initial_times_with_zeros = (initial_datetimes_with_zeros - d_start).astype(float) / 1e9 / 60 

        new_times = np.arange(initial_times_with_zeros[0], initial_times_with_zeros[-1] + 1, 1)
        new_datetimes = np.arange(initial_datetimes_with_zeros[0], initial_datetimes_with_zeros[-1] + np.timedelta64(1, 'm'), np.timedelta64(1, 'm'))

        f = scipy.interpolate.interp1d(initial_times_with_zeros, dose_with_zeros, fill_value="extrapolate", kind = p['interpolation_kind'])
        new_dose = f(new_times)

        new_da = xr.DataArray(data = new_dose, coords = {f'datetime_{name}':new_datetimes}, attrs = da.attrs)

        new_ds[name] = new_da
    return new_ds

def test_resample_pse_tt(sub):
    print(sub)
    ds = resample_pse_tt(sub, **resample_pse_tt_params)
    print(ds)

resample_pse_tt_job = jobtools.Job(precomputedir, 'resample_pse_tt', resample_pse_tt_params, resample_pse_tt)
jobtools.register_job(resample_pse_tt_job)

# QUALITATIVE SEDATION LEVEL

def qualitative_sedation_level(sub, **p):
    sedatives = p['sedative_names']
    ds = resample_pse_tt_job.get(sub)
    possible_tts = list(ds.keys())
    possible_sedatives = [s for s in possible_tts if s in sedatives]

    has_sedatives = True if len(possible_sedatives) > 0 else False
    
    assert p['edges_datetime_mode'] in ['hospit_date','sedative_date'], "'edges_datetime_mode' should be one['hospit_date','sedative_date']"

    if has_sedatives:
        if p['edges_datetime_mode'] == 'hospit_date':
            meta = get_metadata(sub)
            min_datetime_pse = min([ds[name][ds[name].dims[0]].values[0] for name in possible_sedatives])
            max_datetime_pse = max([ds[name][ds[name].dims[0]].values[-1] for name in possible_sedatives])
            min_datetime_hospit = np.datetime64(meta['entree_rea'])
            max_datetime_hospit = np.datetime64(meta['date_sortie_rea'])
            min_datetime = min([min_datetime_pse, min_datetime_hospit]) # should not have to deal with such situation but due to round errors by baptiste to set entree rea it can happens 
            max_datetime = max([max_datetime_pse, max_datetime_hospit])  # should not have to deal with such situation but due to round errors by baptiste to set sortie rea it can happens
        elif p['edges_datetime_mode'] == 'sedative_date':
            min_datetime = min([ds[name][ds[name].dims[0]].values[0] for name in possible_sedatives])
            max_datetime = max([ds[name][ds[name].dims[0]].values[-1] for name in possible_sedatives])
    else: # force hospit date in the case where no sedative
        print(f'No sedative drug detected in patient {sub} so qualitative level is a zero vector from start to stop hospit stay')
        meta = get_metadata(sub)
        min_datetime = np.datetime64(meta['entree_rea'])
        max_datetime = np.datetime64(meta['date_sortie_rea'])

    minute_delta_t = np.timedelta64(1 ,'m')
    datetimes_global = np.arange(min_datetime, max_datetime + minute_delta_t, minute_delta_t)

    if has_sedatives:
        bool_global = np.zeros((len(possible_sedatives), datetimes_global.size), dtype = bool)

        for i, name in enumerate(possible_sedatives):
            da = ds[name]
            datetimes = da[da.dims[0]].values
            dose = da.values
            non_zero_dose = dose != 0
            datetimes_non_zero = datetimes[non_zero_dose]
            bool_tt = np.isin(datetimes_global, datetimes_non_zero)
            bool_global[i,:] = bool_tt

        qualitative_sedation_level = np.sum(bool_global, axis = 0)
        qualitative_sedation_level[qualitative_sedation_level >= 2] = 2
        if 'Thiopenthal' in possible_sedatives: # force stage 2 of sedation where Thiopenthal is ... even if alone
            da = ds['Thiopenthal']
            datetimes_thiopenthal = da['datetime_Thiopenthal'].values
            dose_thiopenthal = da.values
            non_zero_dose_thiopenthal = dose_thiopenthal != 0
            datetimes_non_zero_thiopenthal = datetimes_thiopenthal[non_zero_dose_thiopenthal]
            mask_where_thiopenthal = np.isin(datetimes_global, datetimes_non_zero_thiopenthal)
            qualitative_sedation_level[mask_where_thiopenthal] = 2 # force stage 2 of sedation where Thiopenthal is ... even if alone
    else:
        qualitative_sedation_level = np.zeros(datetimes_global.size)
    
    da = xr.DataArray(data = qualitative_sedation_level, coords = {'datetime':datetimes_global.astype('datetime64[ns]')})
    if has_sedatives and p['bfill_start'] and p['edges_datetime_mode'] == 'hospit_date':
        da.loc[:min_datetime_pse] = da.loc[min_datetime_pse]
    ds = xr.Dataset()
    ds['qualitative_sedation_level'] = da
    return ds

def test_qualitative_sedation_level(sub):
    print(sub)
    ds = qualitative_sedation_level(sub, **qualitative_sedation_level_params)
    print(ds['qualitative_sedation_level'])

qualitative_sedation_level_job = jobtools.Job(precomputedir, 'qualitative_sedation_level', qualitative_sedation_level_params, qualitative_sedation_level)
jobtools.register_job(qualitative_sedation_level_job)


# get patient list function for icca treatments

def get_patient_list_icca_treatments(mode, tt_name = None, verbose = False):
    """
    Function aiming to get a list of patient based on various icca treatments criteria    
    ----------
    Parameters
    ----------
    - mode : str
        Should be 'pse' (for pousse-seringue electrique) or 'medication' for one times treatments
    - tt_name : list or None
        List of one or several treatment names used to filter (boolean AND) patients than do have ALL these treatments. If None, return all patients for the selected mode.
    - verbose : bool
        If True, print some available treatments for the mode (pse or medication)
    -------
    Returns
    -------
    - sub_list : List of patients
    """
    icca_subs = get_icca_subs()
    icca_subs_tt = [s for s in icca_subs if (icca_path / s / f'{s}_ICCA_treatment_anonymous.xlsx').exists()]
    pse_bug_subs = ['P1','HA1']
    assert mode in ['pse','medication'], "mode should be 'pse' or 'medication'"
    dict_tt = {}
    all_tt_names = []
    if mode == 'pse':
        job = icca_pse_tt_job
    elif mode == 'medication':
        job = icca_medication_tt_job
    for sub in icca_subs_tt:
        if mode == 'pse' and sub in pse_bug_subs:
            continue
        ds = job.get(sub)
        tt_names = list(ds.keys())
        dict_tt[sub] = tt_names
        all_tt_names.extend(tt_names)
        
    if verbose:
        print(list(set(all_tt_names)))
        
    if tt_name is None:
        return_subs = list(dict_tt.keys())
    else:
        assert isinstance(tt_name,list), "tt_name argument should be of type 'list' even if composed of one element"
        return_subs = []
        for tt in tt_name:
            assert tt in all_tt_names, f'"{tt}" not available'
            return_subs.extend([s for s in dict_tt.keys() if tt in dict_tt[s]])
        return_subs = [s for s in return_subs if return_subs.count(s) == len(tt_name)]
        return_subs = list(set(return_subs))
    assert len(return_subs) > 0, 'no patient available for this condition of treatments'
    return return_subs

def compute_all():
    icca_subs = get_icca_subs()


    # sub_keys = [(sub,) for sub in icca_subs]
    # jobtools.compute_job_list(icca_bio_job, sub_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(icca_clinical_job, sub_keys, force_recompute=True, engine='loop')

    # no_pse_subs = ['P1','HA1']
    # pse_tt_keys = [(sub,) for sub in icca_subs if not sub in no_pse_subs]
    # jobtools.compute_job_list(icca_pse_tt_job, pse_tt_keys , force_recompute=True, engine='loop')
    # jobtools.compute_job_list(icca_medication_tt_job, sub_keys , force_recompute=True, engine='loop')

    # csf_subs = get_csf_subs()
    # csf_keys = [(sub,) for sub in csf_subs]
    # jobtools.compute_job_list(icca_csf_job, csf_keys,  force_recompute=True, engine='loop')

    # sub_keys = [(sub,) for sub in icca_subs]
    # jobtools.compute_job_list(resample_clinical_job, sub_keys , force_recompute=True, engine='loop')

    sub_keys = [(sub,) for sub in icca_subs]
    jobtools.compute_job_list(resample_bio_job, sub_keys , force_recompute=True, engine='loop')

    # no_pse_subs = ['P1','HA1']
    # pse_tt_keys = [(sub,) for sub in icca_subs if not sub in no_pse_subs]
    # jobtools.compute_job_list(resample_pse_tt_job, pse_tt_keys , force_recompute=True, engine='loop')

    # no_pse_subs = ['P1','HA1']
    # pse_tt_keys = [(sub,) for sub in icca_subs if not sub in no_pse_subs]
    # jobtools.compute_job_list(qualitative_sedation_level_job, pse_tt_keys , force_recompute=True, engine='loop')


if __name__ == "__main__":
    # test_icca_bio('DV4') 
    # test_icca_clinical('MF12')
    # test_icca_pse_tt('P111') # P1 (no PSE TT), HA1 (no PSE tt) 
    # test_icca_medication_tt('P18')
    # test_icca_csf('PL20')
    test_resample_bio('P64')
    # test_resample_clinical('MF12')
    # test_resample_pse_tt('MF12')
    # test_qualitative_sedation_level('MF12')

    # compute_all()

    # print(len(get_patient_list_icca_treatments('pse', ['Sufentanil','Rémifentanil'])))
