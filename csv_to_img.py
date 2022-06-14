import pandas as pd
import numpy as np
import csv
import cv2
from downsampling import downsampling


def csv2nparray(csvpath, rate, x_max, y_max):
  # read CSV
  with open(csvpath, 'r') as f:
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
        else:
          pass

  return img[up:down, left:right]

def main():
  # filepath
  #FILE_PATH = 'C:/Users/Taisei/development/test/サインテスト２/Csv'
  #FILE_NAME = 'Sign-1-藤澤大世-1-True-1.csv'
  FILE_PATH = './csv'
  FILE_NAME = 'Sign-1-藤澤大世-1-True-1.csv'
  OUTPUT_FILE_NAME = './out.png'
  # flame size
  X_MAX = 13000
  Y_MAX = 2200
  # downsampling rate(minimum: 1)
  DSR = 10
  # target dpi
  RESOLATION = 640 ** 2

  # convert csv into np.ndarray
  arr = csv2nparray(FILE_PATH + '/' + FILE_NAME, DSR, X_MAX, Y_MAX)
  arr_trimed = trimming(arr)

  # postprocessing
  arr *= 255

  # make image out of np.ndarray
  cv2.imwrite(OUTPUT_FILE_NAME, arr.T)
  cv2.imwrite(OUTPUT_FILE_NAME + 'dpichanged.png', changedpi(arr, RESOLATION).T)
  cv2.imwrite(OUTPUT_FILE_NAME + 'trimmed.png', changedpi(arr_trimed, RESOLATION).T)

if __name__ == '__main__':
  main()
