import pandas as pd
import numpy as np
import csv
import cv2
#from downsampling import downsampling


def downsampling(array, x, y, rate, x_max, y_max):
  def leftlimit():
    if x - rate < 0:
      return 1
    else:
      return 0
  def rightlimit():
    if x + rate >= x_max:
      return 2
    else:
      return 0
  def uplimit():
    if y - rate < 0:
      return 4
    else:
      return 0
  def downlimit():
    if y + rate >= y_max:
      return 8
    else:
      return 0
  def limitcase():
    return leftlimit() + rightlimit() + uplimit() + downlimit()

  if limitcase() == 0:
    array[x-rate:x, y-rate:y] = 0   # leftabove
    array[x:x+rate, y-rate:y] = 0   # rightabove
    array[x-rate:x, y:y+rate] = 0   # leftbelow
    array[x:x+rate, y:y+rate] = 0   # rightbelow
  if limitcase() == 1:
    array[:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 2:
    array[x-rate:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 3:
    array[:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 4:
    array[x-rate:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 5:
    array[:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[:x, y:y+rate] = 0
    array[x:x+rate, y:y+rate] = 0
  if limitcase() == 6:
    array[x-rate:x, :y] = 0
    array[x:, :y] = 0
    array[x-rate:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 7:
    array[:x, :y] = 0
    array[x:, :y] = 0
    array[:x, y:y+rate] = 0
    array[x:, y:y+rate] = 0
  if limitcase() == 8:
    array[x-rate:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[x-rate:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 9:
    array[:x, y-rate:y] = 0
    array[x:x+rate, y-rate:y] = 0
    array[:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 10:
    array[x-rate:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[x-rate:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 11:
    array[:x, y-rate:y] = 0
    array[x:, y-rate:y] = 0
    array[:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 12:
    array[x-rate:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[x-rate:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 13:
    array[:x, :y] = 0
    array[x:x+rate, :y] = 0
    array[:x, y:] = 0
    array[x:x+rate, y:] = 0
  if limitcase() == 14:
    array[x-rate:x, :y] = 0
    array[x:, :y] = 0
    array[x-rate:x, y:] = 0
    array[x:, y:] = 0
  if limitcase() == 15:
    array[:x, :y] = 0
    array[x:, :y] = 0
    array[:x, y:] = 0
    array[x:, y:] = 0

  return array

# filepath
FILE_PATH = 'C:/Users/Taisei/development/test/サインテスト２/Csv'
FILE_NAME = 'Sign-1-藤澤大世-1-True-1.csv'
# flame size
X_MAX = 13000
Y_MAX = 2200
# downsampling rate
DSR = 1

def main():
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
  cv2.imwrite(FILE_PATH + '/' + FILE_NAME + '.png', arr)

if __name__ == '__main__':
  main()
