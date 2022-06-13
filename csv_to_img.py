import pandas as pd
import numpy as np
import csv
import cv2
import os
from downsampling import downsampling


def csv2img():
  # filepath
  FILE_PATH = 'C:/Users/Taisei/development/test/サインテスト２/Csv'
  FILE_NAME = 'Sign-1-藤澤大世-1-True-1.csv'
  OUTPUT_FILE_NAME = 'out.png'
  # flame size
  X_MAX = 13000
  Y_MAX = 2200
  # downsampling rate(minimum: 1)
  DSR = 1

  # read CSV
  with open(FILE_PATH + '/' + FILE_NAME, 'r') as f:
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
  arr = np.ones((X_MAX, Y_MAX))
  # extract writing spot
  df_writing = df[df["Thickness"] != 0]
  # append writing spot
  for x, y in zip(df_writing["X cood."], df_writing["Y cood."]):
    arr = downsampling(arr, x, y, DSR, X_MAX, Y_MAX)

  # after processing
  arr *= 255
  # make np.array image
  cv2.imwrite(OUTPUT_FILE_NAME, arr.T)

def main():
  csv2img()

if __name__ == '__main__':
  main()
