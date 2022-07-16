# 個人名ごとにdataset再構築

import shutil
from pathlib import Path
from tqdm import tqdm

wd = Path.cwd()
ds_t = wd / '..' / '01a_signdata_dataset' / 'dataset' / 'true'
ds_f = wd / '..' / '01a_signdata_dataset' / 'dataset' / 'false'
new_ds = wd / 'dataset_individual'
new_ds.mkdir(exist_ok=True)

for subj, dir in enumerate(tqdm(sorted(ds_t.iterdir(), key=lambda x: int(x.name)))):
  (tmp := (new_ds / f'{subj}' / 'dataset')).mkdir(exist_ok=True, parents=True)
  (t_d := (tmp / 'true')).mkdir(exist_ok=True)
  for bmp in tqdm(dir.glob('*.bmp')):
    shutil.copy(bmp, t_d)

for subj, dir in enumerate(tqdm(sorted(ds_f.iterdir(), key=lambda x: int(x.name)))):
  (f_d := (new_ds / f'{subj}' / 'dataset' / 'false')).mkdir(exist_ok=True)
  for bmp in tqdm(dir.glob('*.bmp')):
    shutil.copy(bmp, f_d)
