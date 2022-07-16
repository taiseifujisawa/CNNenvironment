from pathlib import Path
import shutil
from tqdm import tqdm

wd = Path.cwd() / 'dataset_individual'
save_dir = wd / 'all'
save_dir.mkdir(exist_ok=True)

for i, dir in enumerate(tqdm(wd.iterdir())):
    if dir.glob(r'[0-9]*'):
        result_dir = dir / 'test_results'
        for imgdir in result_dir.glob(r'[0-1]'):
            for img in imgdir.iterdir():
                shutil.copy(img, save_dir / f'{i}_{str(img.name)}')


