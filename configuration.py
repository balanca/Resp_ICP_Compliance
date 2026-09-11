import sys,os
import getpass
from pathlib import Path

"""
if getpass.getuser() in ('samuel.garcia', 'valentin.ghibaudo','baptiste.balanca', 'gwendan.percevault','hidayat.benbelaid','hugo.ardaillon','yacine.medjahdi') and  sys.platform.startswith('linux'):
    # base_folder = '/home/valentin/smb4k/CRNLDATA/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_folder = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_data = '/crnldata/REA_NEURO_MULTI_ICU/'

elif sys.platform.startswith('win') and getpass.getuser() in ('baptiste.balanca'):
    base_folder = 'N:/baptiste.balanca/Neuro_rea_monitorage/'
    base_data = 'N:/REA_NEURO_MULTI_ICU/'
    
elif sys.platform.startswith('win') and getpass.getuser() in ('gwenp'):
    base_folder = 'n:/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_data = 'N:/REA_NEURO_MULTI_ICU/'

elif sys.platform.startswith('win') and getpass.getuser() in ('vatgh'):
    base_folder = 'n:/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_data = 'N:/REA_NEURO_MULTI_ICU/'
    
elif sys.platform.startswith('darwin') and getpass.getuser() in ('gwendanpercevault'): 
    base_folder = 'Volumes/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_data = 'Volumes/REA_NEURO_MULTI_ICU/'
"""

base_folder = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
base_data = '/crnldata/REA_NEURO_MULTI_ICU/'

base_folder = Path(base_folder)
base_data = Path(base_data)
data_path = base_data / 'raw_data'
icca_path = base_data / 'data_ICCA'

# precomputedir = base_folder / 'precompute' # precomputedir on crnldata
#base_mnt_data = Path('/mnt/data/valentinghibaudo/')
#precomputedir = base_mnt_data / 'Neuro_rea_monitorage' / 'precompute' # mnt/data/

# precompute directory (job cache)
base_mnt_data = Path('/mnt/data/NEURO_REA_MONITORAGE/')
precomputedir = base_mnt_data / 'precompute' # mnt/data/
metadata_file = base_data / "liste_monito_multi_18_06_2026.xlsx"


if __name__ == '__main__':
    print(base_folder, base_folder.exists())
    print(data_path, data_path.exists())
    print(precomputedir, precomputedir.exists())
    print(icca_path, precomputedir.exists())
    print(metadata_file, precomputedir.exists())
