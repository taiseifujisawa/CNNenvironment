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


def trimming(img):
  for i, column in enumerate(img):
    if 0 in column:
      left = i
      break
    else:
      pass
  for i, column in enumerate(img[::-1]):
    if 0 in column:
      right = -i
      break
    else:
      pass
  img_T = img.T
  for i, column in enumerate(img_T):
    if 0 in column:
      up = i
      break
    else:
      pass
  for i, column in enumerate(img_T[::-1]):
    if 0 in column:
      down = -i
      break
    else:
      pass

  return img[left:right, up:down]


def padding(img, pad_x, pad_y):
  w, h = img.shape
  add_updown = int((pad_x - w) / 2)
  add_leftright = int((pad_y - h) / 2)
  padded_img = cv2.copyMakeBorder(img, add_updown + 1, add_updown + 1,\
    add_leftright + 1, add_leftright + 1, cv2.BORDER_CONSTANT, value=1)

  return padded_img[0:pad_x, 0:pad_y]


def csv2img(input_file_path: Path, output_file_path: Path, x_max: int, y_max: int,\
  DSR: int, resolution: int, pad_x: int, pad_y: int, overwrite=True):

  # convert csv into np.ndarray
  arr = csv2nparray(input_file_path, DSR, x_max, y_max)
  arr_dpichanged = changedpi(arr, resolution)
  arr_trimed = trimming(arr)
  arr_trimed_dpichanged = changedpi(arr_trimed, resolution)
  arr_padded = padding(arr_trimed_dpichanged, pad_x, pad_y)

  # postprocessing
  arr *= 255
  arr_dpichanged *= 255
  arr_padded *= 255

  ### make "row"
  # make folders on cwd
  rowdata_folder = Path(output_file_path.parent / 'row' )
  if not rowdata_folder.exists():
    rowdata_folder.mkdir(exist_ok=True)
  # make image out of np.ndarray as a temporary file
  # cv2は日本語のパスに対応していないので、Pathオブジェクトが使えない
  # cwdを'./'と書いて一時的ファイルとして保存し、移動する
  if overwrite or not (rowdata_folder / f'{output_file_path.stem}{output_file_path.suffix}').exists():
    cv2.imwrite(f'./tmp_unchanged{output_file_path.suffix}', arr.T)
    # move image to the right place
    shutil.move(f'./tmp_unchanged{output_file_path.suffix}',\
      str(rowdata_folder / f'{output_file_path.stem}{output_file_path.suffix}'))

  ### make "down_sampled"
  down_sampleddata_folder = Path(output_file_path.parent / 'down_sampled')
  if not down_sampleddata_folder.exists():
    down_sampleddata_folder.mkdir(exist_ok=True)
  if overwrite or not (down_sampleddata_folder / f'{output_file_path.stem}_dpichanged{output_file_path.suffix}').exists():
    cv2.imwrite(f'./tmp_changed{output_file_path.suffix}', arr_dpichanged.T)
    shutil.move(f'./tmp_changed{output_file_path.suffix}',\
      str(down_sampleddata_folder / f'{output_file_path.stem}_dpichanged{output_file_path.suffix}'))

  ### make "trimmed"
  trimmeddata_folder = Path(output_file_path.parent / 'trimmed')
  if not trimmeddata_folder.exists():
    trimmeddata_folder.mkdir(exist_ok=True)
  if overwrite or not (trimmeddata_folder / f'{output_file_path.stem}_trimmed{output_file_path.suffix}').exists():
    cv2.imwrite(f'./tmp_trimmed{output_file_path.suffix}', arr_padded.T)
    shutil.move(f'./tmp_trimmed{output_file_path.suffix}',\
      str(trimmeddata_folder / f'{output_file_path.stem}_trimmed{output_file_path.suffix}'))


def main():
  # filepath
  #INPUT_FILE_PATH = Path(r'C:\Users\Taisei\development\01b_signdata_copy\宮原\サインテスト２\Csv\Sign-2016-宮原悠司-1-True-1.csv')
  INPUT_FILE_PATH = r'C:\Users\taise\Documents\大世のドキュメント\横浜国立大学\中田研\csv2img\坪井\サインテスト２\Csv\Sign-2022-坪井陽人-1-True-3.csv'
  OUTPUT_FILE_PATH = Path('./out.bmp')
  # flame size
  X_MAX = 13000
  Y_MAX = 2200
  # downsampling rate(minimum: 1)
  DSR = 10
  # target dpi
  RESOLUTION = 640 ** 2
  # target padding dpi
  PADDED_X = 2000
  PADDED_Y = 300

  csv2img(INPUT_FILE_PATH, OUTPUT_FILE_PATH, X_MAX, Y_MAX, DSR, RESOLUTION, PADDED_X, PADDED_Y)


if __name__ == '__main__':
  main()
