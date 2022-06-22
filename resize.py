from pathlib import Path
import cv2
from tqdm import tqdm

RESOLUTION = 1000 * 250

cwd = Path.cwd() / '../01b_signdata_copy_all'
true = cwd / 'true/trimmed'
false = cwd / 'false/trimmed'
true_new = cwd / 'true/resized'
false_new = cwd / 'false/resized'
true_new.mkdir(exist_ok=True)
false_new.mkdir(exist_ok=True)

for i, bmp in enumerate(tqdm(true.glob('*.bmp'))):
  im = cv2.imread(str(bmp))
  h, w = im.shape[:2]
  scale = (RESOLUTION / (w * h)) ** 0.5
  img_reshaped = cv2.resize(im, dsize=None, fx=scale, fy=scale)
  cv2.imwrite(str(true_new / f'{i}.bmp'), img_reshaped)

for i, bmp in enumerate(tqdm(true.glob('*.bmp'))):
  im = cv2.imread(str(bmp))
  h, w = im.shape[:2]
  scale = (RESOLUTION / (w * h)) ** 0.5
  img_reshaped = cv2.resize(im, dsize=None, fx=scale, fy=scale)
  cv2.imwrite(str(false_new / f'{i}.bmp'), img_reshaped)
