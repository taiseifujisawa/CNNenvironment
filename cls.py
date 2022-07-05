from pathlib import Path
import shutil

# put this file in the parent directory

wd = Path.cwd() / 'test_results' / 'failure'
(sn := wd / 'same_name').mkdir(exist_ok=True)
(dn := wd / 'different_name').mkdir(exist_ok=True)
(ft := wd / 'same_name' / 'f_as_t').mkdir(exist_ok=True)
(tf := wd / 'same_name' / 't_as_f').mkdir(exist_ok=True)
(mt := wd / 'different_name' / 'missed_t').mkdir(exist_ok=True)
(mf := wd / 'different_name' / 'missed_f').mkdir(exist_ok=True)
number_of_subjects = 16

for f in wd.glob('*.png'):
    pred, ans = (fname := f.name.split('_'))[1], fname[2]
    if f.stem[-3:] == 'cam':
        pred = int(pred[1:])
        ans = int(ans[1:])
    else:
        pred = int(pred[1:])
        ans = int(ans[1:-4])
    if pred - ans == number_of_subjects:
        shutil.copy(f, tf)
    elif pred - ans == -number_of_subjects:
        shutil.copy(f, ft)
    elif ans < number_of_subjects:
        shutil.copy(f, mt)
    else:
        shutil.copy(f, mf)
