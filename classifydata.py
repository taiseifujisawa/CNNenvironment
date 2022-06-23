from pathlib import Path
from subject_number import subject_number_dict
from csv_to_img import csv2img
from tqdm import tqdm

cwd = Path.cwd()
RELATIVE_CSV_PATH = 'サインテスト２/Csv'
TRUE_FOLDER = cwd / 'true'
FALSE_FOLDER = cwd / 'false'
TRUE_FOLDER.mkdir(exist_ok=True)
FALSE_FOLDER.mkdir(exist_ok=True)

# flame size
X_MAX = 13000
Y_MAX = 2200
# downsampling rate(minimum: 1)
DSR = 10
# target dpi
RESOLUTION = 640 ** 2
# target padding dpi
PADDED_X = 2000
PADDED_Y = 500

# get all subjects' directories
dirs = [dir / RELATIVE_CSV_PATH for dir in cwd.iterdir() if dir.is_dir() \
  and dir not in [TRUE_FOLDER, FALSE_FOLDER, cwd / '__pycache__', cwd / '.git',\
    cwd / 'row', cwd / 'down_sampled', cwd / 'trimmed', cwd / 'cam', cwd / 'dataset']]

# count 0 to 7
i = 0
for dir in tqdm(dirs):
  # for each csv file in subjects' directories, make image out of itself
  for csv in tqdm([f for f in dir.iterdir() if f.is_file()]):
    filename = csv.name.split('-')
    subject_no = int(filename[1])
    try:
      name_no = subject_number_dict[filename[2]]
    except Exception as e:
      msg = f'[ERROR] An error occurred with "{str(csv)}".{str(e)} does not exist in the dictionary.\n'
      print(msg)
      with open("errorlog.txt", 'a', encoding='utf-8') as f1,\
        open("errorfile.txt", 'a', encoding='utf-8') as f2:
        f1.write(msg)
        f2.write(f'{str(csv)}\n')
      continue

    authenticity = filename[3] if len(filename) == 5 else filename[4]
    assert authenticity in ['True', 'False'], "authenticity is neither 'True' nor 'False'"

    if authenticity == 'True':
      sign_no = int(filename[3]) - 1
      output_filename = TRUE_FOLDER / f'{subject_no}-{name_no}-{sign_no * 8 + i}.bmp'
      i = i + 1 if i < 7 else 0   # inclement 8 times and then initialize to 0 next time
    elif authenticity == 'False':
      output_filename = FALSE_FOLDER / f'{subject_no}-{name_no}-{i}.bmp'
      i = i + 1 if i < 7 else 0   # inclement 7 times and then initialize to 0 next time

    csv2img(csv, output_filename, X_MAX, Y_MAX, DSR, RESOLUTION, PADDED_X, PADDED_Y, False)
