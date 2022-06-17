import pandas as pd
import numpy as np
import csv
import cv2
from pathlib import Path
import shutil
from downsampling import downsampling
import codecs

def csv2nparray(csvpath, rate, x_max, y_max):
  # read CSV
  with codecs.open(csvpath, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    lines = [row for row in reader]
  # cut surplus
  column_name = lines[3]
  data = lines[4:]
  # make all data int
  int_data = [list(map(int, column)) for column in data]
  # make data flame
  df = pd.DataFrame(data=int_data, columns = column_name)
  df.set_index("Time", inplace=True)

  # np.array all whose components are zero
  arr = np.ones((x_max, y_max))
  # extract writing spot
  df_writing = df[df["Thickness"] != 0]
  # append writing spot
  for x, y in zip(df_writing["X cood."], df_writing["Y cood."]):
    arr = downsampling(arr, x, y, rate, x_max, y_max)

  return arr

def changedpi(img, resolation):
  h, w = img.shape
  scale = (resolation / (w * h)) **0.5
  changed_img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

  return changed_img
"""
def trimming(img):
  up, down, left, right = -1, -1, -1, -1
  for i, column in enumerate(img):
    if np.all(column == 1) and up == -1:
      pass
    elif np.all(column == 1) and up != -1:
      down = i
    else:
      if up == -1:
        up = i
      else:
        continue
      for j, element in enumerate(column):
        if element == 1 and left == -1:
          pass
        elif element == 1 and left != -1:
          right = j
        elif left == -1:
          left = j
        elif left > j:
          left = j
        else:
          pass

  return img[up:down, left:right]
"""
def csv2img(input_file_path: Path, output_file_path: Path, x_max: int, y_max: int, DSR: int, resolution: int):

  # convert csv into np.ndarray
  arr = csv2nparray(input_file_path, DSR, x_max, y_max)

  #arr_trimed = trimming(arr)

  # postprocessing
  arr *= 255
  # make image out of np.ndarray as a temporary file
  # cv2は日本語のパスに対応していないので、Pathオブジェクトが使えない
  # cwdを'./'と書いて一時的ファイルとして保存し、移動する
  cv2.imwrite(f'./tmp_unchanged{output_file_path.suffix}', arr.T)
  cv2.imwrite(f'./tmp_changed{output_file_path.suffix}', changedpi(arr, resolution).T)
  # make folders on cwd
  rowdata_folder = Path(output_file_path.parent / 'row' )
  down_sampleddata_folder = Path(output_file_path.parent / 'down_sampled')
  if not rowdata_folder.exists():
    rowdata_folder.mkdir(exist_ok=True)
  if not down_sampleddata_folder.exists():
    down_sampleddata_folder.mkdir(exist_ok=True)
  # move image to the right place
  shutil.move(f'./tmp_unchanged{output_file_path.suffix}',\
    str(rowdata_folder / f'{output_file_path.stem}{output_file_path.suffix}'))
  shutil.move(f'./tmp_changed{output_file_path.suffix}',\
    str(down_sampleddata_folder / f'{output_file_path.stem}_dpichanged{output_file_path.suffix}'))

  #cv2.imwrite(OUTPUT_FILE_NAME + 'trimmed.png', changedpi(arr_trimed, RESOLATION).T)

def main():
  # filepath
  INPUT_FILE_PATH = Path('C:/Users/Taisei/development/署名データ_copy/宮原/サインテスト２/Csv/Sign-2016-宮原悠司-1-True-1.csv')
  OUTPUT_FILE_PATH = Path('./out.bmp')
  # flame size
  X_MAX = 13000
  Y_MAX = 2200
  # downsampling rate(minimum: 1)
  DSR = 10
  # target dpi
  RESOLUTION = 640 ** 2

  csv2img(INPUT_FILE_PATH, OUTPUT_FILE_PATH, X_MAX, Y_MAX, DSR, RESOLUTION)

if __name__ == '__main__':
  main()
