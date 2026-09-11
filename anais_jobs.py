from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from custom_view import Respi_Rate
import pycns
import physio
from configuration import *
from tools import *
from multi_projects_jobs import *
from icca_tools import *
from icca_jobs import *
from overview_data_pycns import get_patient_list

# outputs are written next to this file (df_for_stats/ and figures/ are not distributed)
base_dir = Path(__file__).resolve().parent

# job params

ylims = {
    'icp_in_resp':None,
    'heart_resp_in_icp':(0,40),
    'P2P1':(0.5,2.5),
    'icp':(0,50),
    'abp':(50, 200),
    'ppc':(20,130),
    'co2':None,
    'ventilation':None,
    'pco2_art':(20,50),
    'sedation':None,
    'csf':(0,100),
}

fig_overview_params = {
    'rolling_P2P1':300,
    'rolling_icp_abp':600,
    'resp_wsize_in_mins':3, 
    'ratio_sat':4, 
    'rate_bins_resp':(5, 30, 0.5), 
    'plot_type': '2d',
    'quantile_saturation_2d':0.025,
    'ylims':ylims,
    'with_icca':True, # add the ICCA rows (arterial pCO2, sedation level, drained CSF)
}

fig_overview_no_icca_params = dict(fig_overview_params, with_icca=False)

design_matrix_anais_params = {
    'icp_by_resp_cycle_params':icp_by_resp_cycle_params,
    'icp_pulse_by_resp_cycle_params':icp_pulse_by_resp_cycle_params,
    'ratio_P1P2_params':ratio_P1P2_params,
    'qualitative_sedation_level_params':qualitative_sedation_level_params,
    'delta_minutes':60,
    'rolling_P2P1':300,
    'rolling_icp_abp':600,
}

## TOOLS
def get_abp_name(reader):
    # the choice is based on the availability of the _Mean stream, not of the raw stream:
    # P0136 does have 'ABP' but no 'ABP_Mean', whereas 'ART_Mean' is available.
    all_streams = reader.streams.keys()
    assert 'ICP_Mean' in all_streams, 'No ICP_Mean stream in data'
    for abp_name in ['ABP', 'ART']:
        if f'{abp_name}_Mean' in all_streams:
            return abp_name
    raise NotImplementedError('No blood pressure _Mean stream in data')


def get_common_srate(reader, stream_names):
    # since pycns d927bc3, export_to_xarray decimates when sample_rate <= the stream srate,
    # and decimate(q=1) raises a ValueError. We force the interpolation branch by asking
    # for the smallest float strictly greater than the max of the srates.
    srate = max([reader.streams[stream_name].sample_rate for stream_name in stream_names])
    return np.nextafter(srate, np.inf)


def get_start_stop_datetimes(start_date, stop_date, delta_minutes):
    starts = np.arange(start_date.astype('datetime64[m]')+1, stop_date.astype('datetime64[m]')-1, np.timedelta64(delta_minutes, 'm')).astype('datetime64[ns]')
    stops = starts + np.timedelta64(delta_minutes, 'm')
    return starts[:-1], stops[:-1]


# jobs
## fig overview job
def fig_overview(sub, **p):
    """
    One overview figure per patient: respiratory modulation of ICP, heart and respiratory
    amplitudes in the ICP spectrum, P2/P1 ratio, ICP, ABP, CPP, respiratory rate and
    ventilation mode, plus the ICCA rows (arterial pCO2, sedation level, drained CSF)
    when p['with_icca'] is True.
    """
    reader = pycns.CnsReader(data_path / sub)

    icp_by_resp_cycle = icp_by_resp_cycle_job.get(sub)['icp_by_resp_cycle']
    p2_p1_da = ratio_P1P2_job.get(sub)['ratio_P1P2'].rolling(date = p['rolling_P2P1']).median('date').bfill('date').ffill('date')
    heart_resp_in_icp = heart_resp_in_icp_job.get(sub)['heart_resp_in_icp']
    abp_name = get_abp_name(reader)
    stream_names = ['ICP_Mean',f'{abp_name}_Mean']
    srate = get_common_srate(reader, stream_names)
    ds = reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    icp = ds['ICP_Mean'].rolling(times = p['rolling_icp_abp']).median('times').bfill('times').ffill('times')
    abp = ds[f'{abp_name}_Mean'].rolling(times = p['rolling_icp_abp']).median('times').bfill('times').ffill('times')
    ppc = abp - icp
    resp_cycles = detect_resp_job.get(sub).to_dataframe()

    meta = get_metadata(sub=sub)
    gcs_sortie = meta['GCS_sortie']                                         
    mrs_sortie = meta['mRS_sortie']                                                                      
    mrs_6 = meta['mRS_6mois']
    motif = meta['motif']
    duree = meta['durée']
    age = (meta['entree_rea'] - meta['ddn']).days / 365.25

    d0 = icp_by_resp_cycle['cycle_date'].values[0]
    d1 = icp_by_resp_cycle['cycle_date'].values[-1]

    dict_das = {
        'icp_in_resp':icp_by_resp_cycle,
        'heart_resp_in_icp':heart_resp_in_icp,
        'P2P1':p2_p1_da,
        'icp':icp,
        'abp':abp,
        'ppc':ppc,
        'co2':resp_cycles,
        'ventilation':resp_cycles,
    }

    ylabels = {
        'icp_in_resp':'Phase',
        'heart_resp_in_icp':'mmHg',
        'P2P1':'Score',
        'icp':'mmHg',
        'abp':'mmHg',
        'ppc':'mmHg',
        'co2':'cpm',
        'ventilation':'mode',
        'pco2_art':'mmHg',
        'sedation':'Level',
        'csf':'mL',
    }

    if p['with_icca']:
        dict_das['pco2_art'] = icca_bio_job.get(sub)['pCO2 artériel'] * 7.50062 # kPa -> mmHg
        dict_das['sedation'] = qualitative_sedation_level_job.get(sub)['qualitative_sedation_level']
        # drained CSF volume, only for patients with an external ventricular drain in ICCA
        dict_das['csf'] = icca_csf_job.get(sub)['Vol vidé (E_S)'] if sub in get_csf_subs() else None
        suffix = ''
    else:
        suffix = '_no_icca'

    ylims = p['ylims']

    nrows = len(dict_das)
    figsize = (12,nrows * 3)

    fig, axs = plt.subplots(nrows = nrows, figsize=figsize , constrained_layout = True, sharex = False)
    fig.suptitle(f'{sub}\nage : {int(round(age,0))} - diagnosis : {motif} - duration : {duree} days\nGCS at discharge : {gcs_sortie} - mRS at discharge : {mrs_sortie} - mRS at 6 months : {mrs_6}')

    for i, name in enumerate(dict_das):
        ax = axs[i]
        ax.set_title(name)
        ax.set_xlim(d0, d1)
        ax.set_ylabel(ylabels[name])
        if not ylims[name] is None:
            ax.set_ylim(ylims[name])

        da = dict_das[name]

        if not da is None:
            if da.ndim == 1:
                ax.plot(da[da.dims[0]], da.values, color = 'k')
                if name in ['pco2_art','csf']: # sparse measurements: mark the samples
                    ax.scatter(da[da.dims[0]], da.values, color = 'r')
            if name == 'icp_in_resp':
                da = da - da.median('phase')
                da = da[::10,::2]
                vmin = da.quantile(p['quantile_saturation_2d'])
                vmax = da.quantile(1 - p['quantile_saturation_2d'])
                vmin = vmin if abs(vmin) > abs(vmax) else -vmax
                vmax = vmax if abs(vmax) > abs(vmin) else -vmin
                ax.pcolormesh(da[da.dims[0]].values, da['phase'].values, da.values.T, vmin = vmin, vmax=vmax, cmap = 'seismic')
                ax.axhline(icp_by_resp_cycle.attrs['cycle_ratio'], color = 'g', lw = 2)
            if name == 'icp':
                ax.axhline(20, color = 'r', lw = 2)
            if name == 'co2':
                rate_bins_resp = np.arange(p['rate_bins_resp'][0], p['rate_bins_resp'][1], p['rate_bins_resp'][2])
                view = Respi_Rate(resp_cycles, resp_wsize_in_mins = p['resp_wsize_in_mins'], ratio_sat = p['ratio_sat'], rate_bins_resp = rate_bins_resp, plot_type = p['plot_type'])
                view.plot(ax, d0, d1)
            if name == 'ventilation':
                local_resp_features = resp_cycles[(resp_cycles['inspi_date'] > d0) & (resp_cycles['inspi_date'] < d1)]
                local_mode = local_resp_features['is_ventilation_controlled'].values
                datetimes = local_resp_features['inspi_date'].values
                ax.plot(datetimes, local_mode, color = 'k', lw = 2)
                ax.set_ylim(-0.1, 1.1)
                ax.set_yticks([0,1], ['assisted','controlled'])
            if name == 'heart_resp_in_icp':
                alpha = 0.7
                local_da = da.loc[:,d0:d1].rolling(datetime = 60, center = True).median('datetime').bfill('datetime').ffill('datetime')
                ax.plot(local_da['datetime'].values, local_da.loc['heart_in_icp_spectrum'], color = 'm', alpha = alpha)
                ax.set_ylabel('Heart in icp\n(mmHg)', color = 'm')
                ax2 = ax.twinx()
                ax2.plot(local_da['datetime'].values, local_da.loc['resp_in_icp_spectrum'], color = 'g', alpha = alpha)
                ax2.set_ylim(0,15)
                ax2.set_ylabel('Resp in icp\n(mmHg)', color = 'g')

    save_folder = base_dir / 'figures' / 'overview'
    save_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_folder / f'{sub}{suffix}.png', dpi = 300, bbox_inches = 'tight')
    plt.close(fig)
    return xr.Dataset()

def test_fig_overview(sub):
    print(sub)
    ds = fig_overview(sub, **fig_overview_params)

fig_overview_job = jobtools.Job(precomputedir, 'fig_overview', fig_overview_params, fig_overview)
fig_overview_no_icca_job = jobtools.Job(precomputedir, 'fig_overview_no_icca', fig_overview_no_icca_params, fig_overview)


## design matrix job

def design_matrix_anais(sub, **p):
    """
    One row per window of p['delta_minutes'] (60 min) over the whole monitoring of a patient:
    phase of the ICP maximum within the respiratory cycle (and its inspiration/expiration
    label), respiratory modulation of ICP, P2/P1, ICP, ICP pulse amplitude, ABP, CPP,
    sedation level, respiratory rate and variability, ventilation mode, time since ICU
    admission.
    """
    reader = pycns.CnsReader(data_path / sub)

    icp_by_resp_cycle = icp_by_resp_cycle_job.get(sub)['icp_by_resp_cycle']
    p2_p1_da = ratio_P1P2_job.get(sub)['ratio_P1P2'].rolling(date = p['rolling_P2P1']).median('date').bfill('date').ffill('date')
    # ICP pulse amplitude (peak - trough of each cardiac pulse, in mmHg), deformed onto the
    # respiratory cycle: matrix (cycle_date x phase)
    icp_pulse_by_resp_cycle = icp_pulse_by_resp_cycle_job.get(sub)['icp_pulse_by_resp_cycle']

    abp_name = get_abp_name(reader)
    stream_names = ['ICP_Mean',f'{abp_name}_Mean']
    srate = get_common_srate(reader, stream_names)
    ds = reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    icp = ds['ICP_Mean'].rolling(times = p['rolling_icp_abp']).median('times').bfill('times').ffill('times')
    abp = ds[f'{abp_name}_Mean'].rolling(times = p['rolling_icp_abp']).median('times').bfill('times').ffill('times')
    ppc = abp - icp
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    # arterial pCO2 is rarely available -> not loaded
    sedation = qualitative_sedation_level_job.get(sub)['qualitative_sedation_level']

    meta = get_metadata(sub=sub)
    date_entree_rea = meta['entree_rea']
    motif = meta['motif']

    d0 = icp_by_resp_cycle['cycle_date'].values[0]
    d1 = icp_by_resp_cycle['cycle_date'].values[-1]

    start_dates, stop_dates = get_start_stop_datetimes(d0, d1, p['delta_minutes'])

    rows = []

    for start_date, stop_date in zip(start_dates, stop_dates):
        days_after_start_hospit = (start_date - date_entree_rea).total_seconds() / (3600 * 24)
        local_resp_cycles = resp_cycles[(resp_cycles['inspi_date'] > start_date) & (resp_cycles['expi_date'] < stop_date)]
        local_med_cycle_ratio = local_resp_cycles['cycle_ratio'].median()
        local_icp_by_resp_cycle = icp_by_resp_cycle.loc[start_date:stop_date,:]
        resp_in_icp = float((local_icp_by_resp_cycle.max('phase') - local_icp_by_resp_cycle.min('phase')).median('cycle_date'))
        max_icp_phase = float(local_icp_by_resp_cycle.median('cycle_date').idxmax('phase'))
        if not np.isnan(max_icp_phase):
            label_max_icp_phase = 'inspiration' if max_icp_phase <= local_med_cycle_ratio else 'expiration'
        else:
            label_max_icp_phase = np.nan
        local_P2P1 = np.nanmedian(p2_p1_da.loc[start_date:stop_date].values)
        local_icp = np.nanmedian(icp.loc[start_date:stop_date].values)
        # level of the ICP pulse amplitude over the window, in mmHg.
        # median over (cycle_date, phase): this is the amplitude itself, not its
        # modulation by respiration (which would be max('phase') - min('phase')).
        local_icp_pulse_amplitude = np.nanmedian(icp_pulse_by_resp_cycle.loc[start_date:stop_date,:].values)
        local_abp = np.nanmedian(abp.loc[start_date:stop_date].values)
        local_ppc = np.nanmedian(ppc.loc[start_date:stop_date].values)
        local_sedation = round(np.nanmedian(sedation.loc[start_date:stop_date].values),0)
        local_resp_rate, local_resp_rate_variability = physio.compute_median_mad(local_resp_cycles['cycle_freq_cpm'].values)
        local_is_ventilation_controlled = local_resp_cycles['is_ventilation_controlled'].median()
        if local_is_ventilation_controlled > 0.5:
            local_is_ventilation_controlled = 'controlled'
        else:
            local_is_ventilation_controlled = 'assisted'
        
        d = dict(
            max_icp_phase=max_icp_phase,
            label_max_icp_phase=label_max_icp_phase,
            resp_in_icp=resp_in_icp,
            P2P1 = local_P2P1,
            icp=local_icp,
            icp_pulse_amplitude = local_icp_pulse_amplitude,
            abp=local_abp,
            ppc=local_ppc,
            sedation = local_sedation,
            resp_rate = local_resp_rate,
            resp_variability = local_resp_rate_variability,
            ventilation_mode = local_is_ventilation_controlled,
            time = days_after_start_hospit,
            subject=sub,
            motif=motif,
            )

        rows.append(d)

    df = pd.DataFrame(rows)
    ds = xr.Dataset(df)
    return ds

def test_design_matrix_anais(sub):
    print(sub)
    ds = design_matrix_anais(sub, **design_matrix_anais_params)
    print(ds.to_dataframe())

design_matrix_anais_job = jobtools.Job(precomputedir, 'design_matrix_anais', design_matrix_anais_params, design_matrix_anais)
jobtools.register_job(design_matrix_anais_job)


# RUN

def get_run_keys(tuple_formatted = True):
    subs = [s for s in get_patient_list(['ICP','CO2'])]

    # excluded patients: P0075 (progression to brain death), P0076 (unreliable data),
    # P0085 and P0150 (cycle detection failure), P0155 (no ICCA data), P0015 (no data),
    # P0098 (too short), P0112 (signal problem), P0030 P0061 P0095 P0119 P0130_2
    # (no detectable respiratory cycle), P0065, P0113, P0042, P0022, P0057_old
    remove_subs = ['P0065', 'P0113', 'P0075', 'P0076', 'P0085', 'P0155', 'P0015',
                   'P0098', 'P0030', 'P0061', 'P0095', 'P0119', 'P0130_2', 'P0112', 'P0150', 'P0042', 'P0022', 'P0057_old']

    subs = [s for s in subs if not s in remove_subs]

    if tuple_formatted:
        return [(s,) for s in subs]
    else:
        return subs


def compute_all():
    run_keys = get_run_keys()
    # jobtools.compute_job_list(fig_overview_job, run_keys, force_recompute=True, engine='joblib', n_jobs = 10)
    # jobtools.compute_job_list(fig_overview_no_icca_job, run_keys, force_recompute=True, engine='joblib', n_jobs = 10)
    jobtools.compute_job_list(design_matrix_anais_job, run_keys, force_recompute=False, engine='loop')


def concat_and_save(save = False):
    """
    Concatenate the design matrices of all subjects into one DataFrame (one row per 60-min window).
    With save=True, write df_for_stats/anais_design_matrix_bb.xlsx.
    """
    # the precomputed file is read directly instead of going through job.get(): a missing
    # subject would trigger an online recomputation, which returns None if it fails and
    # crashes .to_dataframe(). Here it is skipped and reported.
    concat = []
    missing = []
    empty = []
    for sub in get_run_keys(False):
        filename = design_matrix_anais_job.get_filename(sub)
        if not filename.is_file():
            missing.append(sub)
            continue
        ds = xr.open_dataset(filename)
        if len(ds.dims) == 0 or ds.sizes.get('dim_0', 0) == 0:
            # empty design matrix (subject too short): to_dataframe() would raise
            # "no valid index for a 0-dimensional object"
            empty.append(sub)
            continue
        df = ds.to_dataframe()
        df['subject'] = sub
        concat.append(df)

    if len(missing) > 0:
        print(f'{len(missing)} subjects without design matrix, skipped: {missing}')
    if len(empty) > 0:
        print(f'{len(empty)} subjects with an empty design matrix, skipped: {empty}')
    assert len(concat) > 0, 'no design matrix to concatenate'

    concat = pd.concat(concat)
    concat = concat.dropna(subset = ['max_icp_phase','resp_in_icp']).reset_index(drop = True)
    print(f"{len(concat)} windows kept from {concat['subject'].nunique()} subjects")
    if save:
        save_folder = base_dir / 'df_for_stats'
        save_folder.mkdir(exist_ok=True)
        concat.to_excel(save_folder / 'anais_design_matrix_bb.xlsx')
    return concat

if __name__ == "__main__":
    # test_fig_overview('P0044')
    # test_design_matrix_anais('P0044')

    # compute_all()

    print(concat_and_save(save = True))
