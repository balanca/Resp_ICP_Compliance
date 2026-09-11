import numpy as np


def iqr_interval(a, round_=2, format = 'str'):
    q25, q75 = np.nanquantile(a, 0.25), np.nanquantile(a, 0.75)
    if format == 'str':
        return f'[{q25.round(round_)}, {q75.round(round_)}]' # "If your audience is international, the comma format may be more intuitive,  it's best to include a space inside the brackets for readability"
    elif format == 'tuple':
        return (q25, q75)


def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001:
        stars = '***'
    else:
        stars = 'ns'
    return stars


def readable_pval(pval):
    if pval < 0.01 and pval >= 0.001:
        return ' < 0.01'
    elif pval < 0.001 and pval >= 0.0001:
        return ' < 0.001'
    elif pval < 0.0001:
        return ' < 0.0001'
    else:
        return f' = {round(pval, 3)}'
