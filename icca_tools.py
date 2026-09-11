import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from configuration import *

def get_icca_subs():
    icca_subs = [str(sub).split('/')[-1] for sub in icca_path.iterdir()]
    icca_subs = [sub for sub in icca_subs if not '.' in sub]
    return icca_subs

def get_csf_subs():
    csf_subs = [str(path).split('/')[-2] for path in icca_path.glob('*/*csf*.xlsx')]
    return list(set(csf_subs))

def get_icca_treatments_id_mapper():
    return pd.read_excel(icca_path / 'intervention_id_mapper_treatments.xlsx', index_col = 0)

def load_icca_raw(sub, dtype, time_zone = 'Europe/Paris'):
    """
    sub : str
    dtype : str - clinical or treatment or biology or csf
    """
    sub_folder = icca_path / sub
    file = [f for f in sub_folder.glob(f'*{dtype}*.xlsx')][0]
    df = pd.read_excel(file)
    df['date_gmt'] = df['Date'].dt.tz_localize(time_zone, ambiguous='NaT').dt.tz_convert('GMT').values
    return df

def load_treatment(sub, name = None, administration_type = None, use_human_weight = True, verbose = False, show = False):
    df = load_icca_raw(sub, 'treatment')
    mapper_tt = get_icca_treatments_id_mapper()
    df['short_label'] = df['intervention Id'].map(mapper_tt['name'])
    df['administration_type'] = df['intervention Id'].map(mapper_tt['administration_type'])
    df = df[df['administration_type'].notnull()].reset_index(drop = True)
    if verbose:
        print('labels', df['short_label'].unique().tolist())
    clinical = load_icca_raw(sub, 'clinical')
    median_weight = clinical[clinical['Unite'] == 'kg']['Valeur'].apply(lambda x:float(x.replace(',','.')) if type(x) is str else float(x)).median()
    if name is None:
        return df
    else:
        df = df[df['short_label'] == name]
        possible_administration_types = df['administration_type'].unique()
        if verbose:
            print('possible administrations', possible_administration_types)
        if administration_type is None:
            return df
        elif administration_type == 'PSE':
            assert administration_type in possible_administration_types, f'"{administration_type}" not available for this treatment'
            df = df[df['administration_type'] == administration_type]
            df = df.sort_values(by = 'date_gmt')
            unit = df[df['Parametre court'] == 'Conc']['Unite'].unique()[0].split('/')[0]
            t_unit = df[df['Parametre court'] == 'Débit adm']['Unite'].unique()[0].split('/')[1]
            unit = f'{unit}/{t_unit}'
            concentrations = df[df['Parametre court'] == 'Conc']['value Number'].values
            flows = df[df['Parametre court'] == 'Débit adm']['value Number'].values
            if np.unique(concentrations).size == 1: # if concentration do not vary
                concentrations = concentrations[0] # take the first concentration to compute doses
            else: # else check than same number of concentrations and flows values are available
                assert concentrations.size == flows.size, f'Not the same amount of concentration and flow values (conc : {concentrations.size}, flow : {flows.size}, datetimes : {datetimes.size})'
            datetimes = df[df['Parametre court'] == 'Débit adm']['date_gmt'].to_numpy()
            doses = concentrations * flows
            if use_human_weight:
                doses = doses / median_weight
                unit = f'{unit}/kg'
            if show:
                fig, ax = plt.subplots()
                ax.plot(datetimes, doses)
                ax.scatter(datetimes, doses, color = 'k')
                ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
                ax.set_ylabel(f'{name} - {administration_type} ({unit})')
                plt.show()
            return {'datetimes':datetimes,'values':doses, 'unit':unit}
        elif administration_type == 'medication':
            assert administration_type in possible_administration_types, f'"{administration_type}" not available for this treatment'
            df = df[df['administration_type'] == administration_type]
            df = df.sort_values(by = 'date_gmt')
            datetimes = df['date_gmt'].values
            unit = df['Unite'].unique()[0]
            doses = df['value Number'].values
            if use_human_weight:
                doses = doses / median_weight
                unit = f'{unit}/kg'
            if show:
                fig, ax = plt.subplots()
                ax.plot(datetimes, doses)
                ax.scatter(datetimes, doses, color = 'k')
                ax.set_ylabel(f'{name} - {administration_type} ({unit})')
                ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
                plt.show()
            return {'datetimes':datetimes,'values':doses, 'unit':unit}
        
def load_clinical(sub, name = None, verbose = False, show = False):
    df = load_icca_raw(sub, 'clinical')
    df['short_label'] = df['Parametre court'].copy()
    if verbose:
        print(df['short_label'].unique().tolist())
    if name is None:
        return df
    else:
        if name == 'Pupille Gauche':
            name = 'Gauche'
        elif name == 'Pupille Droite':
            name = 'Droite'
        elif name == 'Taille Pupille Gauche':
            name = 'Taille Gauche'
        elif name == 'Taille Pupille Droite':
            name = 'Taille Droite'
        df = df[df['short_label'] == name]
        df = df.sort_values(by = 'date_gmt')
        unit = df['Unite'].unique()[0]
        df_sel = df[['date_gmt','Valeur']].dropna()
        datetimes = df_sel['date_gmt'].values
        values = df_sel['Valeur']
        # print(values)
        try:
            values = values.values.astype(float)
        except:
            if 'Dextro' in name or 'Poids' in name:
                values = values.apply(lambda x:float(x.replace(',','.'))).values
            else:
                values = values.values

        if show:
            fig, ax = plt.subplots()
            ax.plot(datetimes, values)
            ax.scatter(datetimes, values, color = 'k')
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
            ax.set_ylabel(f'{name} ({unit})')
            plt.show()

        return {'datetimes':datetimes,'values':values, 'unit':unit}


def load_biology(sub, name = None, verbose = False, show = False):
    df = load_icca_raw(sub, 'biology')
    df['short_label'] = df['Parametre long'].apply(lambda x:x.split('.')[-1])
    df['unit'] = df['Parametre long'].apply(lambda x:x.split('.')[0].split(' ')[-1])
    if verbose:
        print(df['short_label'].unique().tolist())
    if name is None:
        return df
    else:
        df = df[df['short_label'] == name]
        df = df.sort_values(by = 'date_gmt')
        unit = df['unit'].unique()[0]
        df_sel = df[['date_gmt','Valeur num']].dropna()
        datetimes = df_sel['date_gmt'].values
        values = df_sel['Valeur num'].values
  
        if show:
            fig, ax = plt.subplots()
            ax.plot(datetimes, values)
            ax.scatter(datetimes, values, color = 'k')
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
            ax.set_ylabel(f'{name} {unit}')
            plt.show()

        return {'datetimes':datetimes,'values':values, 'unit':unit}
    
def load_csf(sub, show = False):
    df = load_icca_raw(sub, 'csf')
    df = df.sort_values(by = 'date_gmt')
    unit = df['Unite'].unique()[0]
    name = df['Parametre court'].unique()[0]
    df_sel = df[['date_gmt','Valeur']].dropna()
    datetimes = df_sel['date_gmt'].values
    values = df_sel['Valeur'].values
    
    if show:
        fig, ax = plt.subplots()
        ax.plot(datetimes, values)
        ax.scatter(datetimes, values, color = 'k')
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
        ax.set_ylabel(f'{name} ({unit})')
        plt.show()

    return {'datetimes':datetimes,'values':values, 'unit':unit}

def load_icca(sub, dtype, name, administration_type = None, use_human_weight = True, verbose = False, show = False):
    if dtype == 'treatment':
        res = load_treatment(sub, name, administration_type, use_human_weight, verbose, show)
    elif dtype == 'biology':
        res = load_biology(sub, name, verbose, show)
    elif dtype == 'clinical':
        res = load_clinical(sub, name, verbose, show)
    elif dtype == 'csf':
        res = load_csf(sub, verbose, show)
    return res


def get_PSE_conc_unit_mapping(compute = False):
    if compute:
        icca_subs = get_icca_subs()
        dtype = 'treatment'
        concat = []
        for sub in icca_subs:
            df_sub = load_icca_raw(sub, dtype)
            df_sub['admin_type'] = df_sub['short Label'].apply(lambda x:'PSE' if 'PSE' in x else 'medication')
            keep_mask = (df_sub['admin_type'] == 'PSE') & (df_sub['Parametre court'] == 'Conc')
            df_sub = df_sub[keep_mask]
            res = df_sub.drop_duplicates(subset = ['short Label','value Number','Unite'])[['short Label','value Number','Unite']].reset_index(drop = True)
            res['sub'] = sub
            concat.append(res)
        res_all = pd.concat(concat)
        pse_possibilities = res_all[res_all['short Label'] != 'Adm. PSE']
        df_return = pse_possibilities.drop_duplicates(['short Label','value Number','Unite']).drop(columns = ['sub']).reset_index(drop = True).drop_duplicates(['value Number','Unite'])
        df_return['short_label'] = df_return['short Label'].apply(lambda x:x.split(':')[-1][1:])
    else:
        df_return = pd.read_excel(base_folder / 'icca_review' / 'PSE_conc_unit_mapping.xlsx', index_col = 0)
    return df_return

def load_PSE_treatment(sub, name = None, use_human_weight = True, verbose = False, show = False):
    df = load_icca_raw(sub, 'treatment')
    df['administration_type'] = df['short Label'].apply(lambda x:'PSE' if 'PSE' in x else 'medication')
    df = df[df['administration_type'] == 'PSE'].reset_index(drop = True)
    assert df.shape[0] != 0, f'no PSE treatment available in subject {sub}'

    mapping_conc_unit = get_PSE_conc_unit_mapping(compute = False).set_index(['Unite','value Number'])

    for i, row in df.iterrows():
        if row['Parametre court'] == 'Conc':
            df.loc[i,'short_label'] = mapping_conc_unit.loc[(row['Unite'],row['value Number']),'short_label']
    for i, row in df.iterrows():
        if row['Parametre court'] == 'Débit adm':
            date_deb = row['date_gmt']
            if i-1 >= 0:
                if df.loc[i-1,'Parametre court'] == 'Conc' and df.loc[i-1,'date_gmt'] == date_deb:
                    df.loc[i,'short_label'] = df.loc[i-1,'short_label']
            if i+1 <= df.index[-1]:
                if df.loc[i+1,'Parametre court'] == 'Conc' and df.loc[i+1,'date_gmt'] == date_deb:
                    df.loc[i,'short_label'] = df.loc[i+1,'short_label']
    assert df['short_label'].isna().sum() == 0

    if verbose:
        print('possible names :', df['short_label'].unique().tolist())
    
    if name is None:
        return df
    else:
        assert name in df['short_label'].tolist(), f'name {name} not available'
        df = df[df['short_label'] == name].reset_index(drop = True)
        df = df.sort_values(by = 'date_gmt')
        unit = df[df['Parametre court'] == 'Conc']['Unite'].unique()[0].split('/')[0]
        t_unit = df[df['Parametre court'] == 'Débit adm']['Unite'].unique()[0].split('/')[1]
        unit = f'{unit}/{t_unit}'
        concentrations = df[df['Parametre court'] == 'Conc']['value Number'].values
        flows = df[df['Parametre court'] == 'Débit adm']['value Number'].values
        if np.unique(concentrations).size == 1: # if concentration do not vary
            concentrations = concentrations[0] # take the first concentration to compute doses
        else: # else check than same number of concentrations and flows values are available
            assert concentrations.size == flows.size, f'Not the same amount of contration and flow values (conc : {concentrations.size}, flow : {flows.size}, datetimes : {datetimes.size})'
        datetimes = df[df['Parametre court'] == 'Débit adm']['date_gmt'].to_numpy()
        doses = concentrations * flows
        if use_human_weight:
            clinical = load_icca_raw(sub, 'clinical')
            median_weight = clinical[clinical['Unite'] == 'kg']['Valeur'].apply(lambda x:float(x.replace(',','.'))).median()
            doses = doses / median_weight
            unit = f'{unit}/kg'
        if show:
            fig, ax = plt.subplots()
            ax.plot(datetimes, doses)
            ax.scatter(datetimes, doses, color = 'k')
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation = 90)
            ax.set_ylabel(f'{name} - PSE ({unit})')
            plt.show()
        mask_na_datetimes = np.isnan(datetimes)
        return {'datetimes':datetimes[~mask_na_datetimes],'values':doses[~mask_na_datetimes], 'unit':unit}


def load_PSE_treatment_in_dataset(sub, use_human_weight = True):
    df = load_treatment(sub)
    df = df[df['administration_type'] == 'PSE'].reset_index(drop = True)
    assert df.shape[0] != 0, f'no PSE treatment available in subject {sub}'

    clinical = load_icca_raw(sub, 'clinical')
    weights = clinical[clinical['Unite'] == 'kg']['Valeur'].dropna()
    median_weight = weights.apply(lambda x:float(x.replace(',','.'))).median()
    names = df['short_label'].unique().tolist()
    ds = xr.Dataset()
    for name in names:
        df_name = df[df['short_label'] == name].reset_index(drop = True)
        df_name = df_name.sort_values(by = 'date_gmt')
        if df_name['Parametre court'].unique().size <= 1:
            print(f'{sub} : {name} PSE tt has some values but is skipped because lacks Conc or Débit adm parameter in raw data')
            continue # skip this treatment if Conc or Débit adm parameter is lacking to compute dose
        unit = df_name[df_name['Parametre court'] == 'Conc']['Unite'].unique()[0].split('/')[0]
        t_unit = df_name[df_name['Parametre court'] == 'Débit adm']['Unite'].unique()[0].split('/')[1]
        unit = f'{unit}/{t_unit}'
        concentrations = df_name[df_name['Parametre court'] == 'Conc']['value Number'].values
        datetimes_concentrations = df_name[df_name['Parametre court'] == 'Conc']['date_gmt'].to_numpy().astype('datetime64[ns]')
        flows = df_name[df_name['Parametre court'] == 'Débit adm']['value Number'].values
        datetimes_flows = df_name[df_name['Parametre court'] == 'Débit adm']['date_gmt'].to_numpy().astype('datetime64[ns]')
        if np.unique(concentrations).size == 1: # if concentration do not vary
            concentrations = concentrations[0] # take the first concentration to compute doses
            doses = concentrations * flows
            datetimes = datetimes_flows.copy()
        else: # else check than same number of concentrations and flows values are available
            if concentrations.size == flows.size:
                doses = concentrations * flows
                datetimes = datetimes_concentrations.copy()
            else: # if not the same amount of concentrations and flow values, match them by date and remove the not matchable
                datetimes, concentrations_keep_inds, flows_keep_inds = np.intersect1d(datetimes_concentrations, datetimes_flows, return_indices=True, assume_unique = False)
                concentrations = concentrations[concentrations_keep_inds]
                flows = flows[flows_keep_inds]
                doses = concentrations * flows
        if use_human_weight:
            doses = doses / median_weight
            unit = f'{unit}/kg'
        if '/' in name:
            name = name.replace('/','_')
        mask_na_datetimes = np.isnan(datetimes)
        da_name = xr.DataArray(data = doses[~mask_na_datetimes], dims = [f'datetime_{name}'], coords = {f'datetime_{name}':datetimes[~mask_na_datetimes]}, attrs = {'unit':unit})
        ds[name] = da_name
    return ds

def load_medication_treatment_in_dataset(sub):
    df = load_treatment(sub)
    df = df[df['administration_type'] == 'medication'].reset_index(drop = True)
    assert df.shape[0] != 0, f'no medication treatment available in subject {sub}'

    names = df['short_label'].unique().tolist()
    ds = xr.Dataset()
    for name in names:
        df_name = df[df['short_label'] == name].reset_index(drop = True)
        df_name = df_name.sort_values(by = 'date_gmt')
        unit = df_name['Unite'].unique()[0]
        doses = df_name['value Number'].values
        datetimes = df_name['date_gmt'].to_numpy()
        if '/' in name:
            name = name.replace('/','_')
        da_name = xr.DataArray(data = doses, dims = [f'datetime_{name}'], coords = {f'datetime_{name}':datetimes}, attrs = {'unit':unit})
        ds[name] = da_name
    return ds

def load_biology_in_dataset(sub):
    df_start = load_icca_raw(sub, 'biology')
    df_start['short_label'] = df_start['Parametre long'].apply(lambda x:x.split('.')[-1])
    df_start['unit'] = df_start['Parametre long'].apply(lambda x:x.split('.')[0].split(' ')[-1])
    names = df_start['short_label'].unique().tolist()
    ds = xr.Dataset()
    for name in names:
        df = df_start[df_start['short_label'] == name]
        df = df.sort_values(by = 'date_gmt')
        unit = df['unit'].unique()[0]
        df_sel = df[['date_gmt','Valeur num']].dropna()
        datetimes = df_sel['date_gmt'].values
        values = df_sel['Valeur num'].values
        if '/' in name:
            name = name.replace('/','_')
        da_name = xr.DataArray(data = values, dims = [f'datetime_{name}'], coords = {f'datetime_{name}':datetimes}, attrs = {'unit':unit})
        ds[name] = da_name
    return ds

def load_clinical_in_dataset(sub):
    df_start = load_icca_raw(sub, 'clinical')
    df_start['short_label'] = df_start['Parametre court'].copy()
    names = df_start['short_label'].unique().tolist()
    ds = xr.Dataset()
    for name in names:
        df = df_start[df_start['short_label'] == name]
        df = df.sort_values(by = 'date_gmt')
        unit = df['Unite'].unique()[0]
        df_sel = df[['date_gmt','Valeur']].dropna()
        datetimes = df_sel['date_gmt'].values
        values = df_sel['Valeur']
        try:
            values = values.values.astype(float)
        except:
            if 'Dextro' in name or 'Poids' in name:
                values = values.apply(lambda x:float(x.replace(',','.'))).values
            else:
                values = values.values
        if '/' in name:
            name = name.replace('/','_')
        da_name = xr.DataArray(data = values, dims = [f'datetime_{name}'], coords = {f'datetime_{name}':datetimes}, attrs = {'unit':unit})
        ds[name] = da_name
    return ds

def load_csf_in_dataset(sub):
    df = load_icca_raw(sub, 'csf')
    df = df.sort_values(by = 'date_gmt')
    unit = df['Unite'].unique()[0]
    name = df['Parametre court'].unique()[0]
    name = name.replace('/', '_')
    df_sel = df[['date_gmt','Valeur']].dropna()
    datetimes = df_sel['date_gmt'].values
    values = df_sel['Valeur'].values
    da_name = xr.DataArray(data = values, dims = [f'datetime_{name}'], coords = {f'datetime_{name}':datetimes}, attrs = {'unit':unit})
    ds = xr.Dataset()
    ds[name] = da_name
    return ds

if __name__ == "__main__":
    # print(load_PSE_treatment_in_dataset('P111'))
    print(load_icca_raw('P0022','biology'))
    # print(load_PSE_treatment_in_dataset('P70'))
    # print(load_medication_treatment_in_dataset('GA9'))
    # print(load_treatment('P12', name = 'Simvastatine', administration_type='medication', verbose = True))
    # print(get_icca_treatments_id_mapper())
    # print(get_icca_subs())
    # print(load_PSE_treatment_in_dataset('P64'))
    # print(get_csf_subs())
    # print(len(get_icca_subs()))
    # print(len(get_csf_subs()))
    # print(len([s for s in get_icca_subs() if not s in get_csf_subs()]))