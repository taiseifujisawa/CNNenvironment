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

# get all subjects' directories
dirs = [dir / RELATIVE_CSV_PATH for dir in cwd.iterdir() if dir.is_dir() \
  and dir not in [TRUE_FOLDER, FALSE_FOLDER, cwd / '__pycache__']]

# count 0 to 7
i = 0
for dir in tqdm(dirs):
  dir = dirs[1]
  # for each csv file in subjects' directories, make image out of itself
  for csv in tqdm([f for f in dir.iterdir() if f.is_file()]):
    with open(csv, 'r', encoding="shift-JIS") as f:
      subject_no = int(f.readline()[13:])
      second_line = f.readline().split('-')
      name_no = subject_number_dict[second_line[0][17:].rstrip()]
      authenticity = f.readline()[15:].rstrip()
      assert authenticity in ['True', 'False']

    if authenticity == 'True':
      sign_no = int(second_line[1]) - 1
      output_filename = TRUE_FOLDER / f'{subject_no}-{name_no}-{sign_no * 8 + i}.bmp'
      i = i + 1 if i < 7 else 0   # inclement 8 times and then initialize to 0 next time
    elif authenticity == 'False':
      output_filename = FALSE_FOLDER / f'{subject_no}-{name_no}-{i}.bmp'
      i = i + 1 if i < 7 else 0   # inclement 7 times and then initialize to 0 next time

    csv2img(csv, output_filename, X_MAX, Y_MAX, DSR, RESOLUTION)
