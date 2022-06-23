# 新型
# cwdにdatasetディレクトリを作り、その中にtrue, falseディレクトリを作り、
# さらにその下に被験者のディレクトリを作り、名前を整理して保存
# subject_number_dictが参加被験者のみ(藤澤、洞口消去)になっている必要あり
# classifydataの後に使う

from pathlib import Path
import cv2
from tqdm import tqdm
from subject_number import subject_number_dict

dict_of_label_subjectno = {v: i for i, v in enumerate(subject_number_dict.values())}

RESOLUTION = 1000 * 250

cwd = Path.cwd()
data_dir = cwd / 'dataset'
data_dir.mkdir(exist_ok=True)
true_t = cwd / 'true/trimmed'
false_t = cwd / 'false/trimmed'
true = 'true'
false = 'false'

d_t = data_dir / true
d_t.mkdir(exist_ok=True)
d_f = data_dir / false
d_f.mkdir(exist_ok=True)

for i in range(len(dict_of_label_subjectno)):
  (d_t / f'{i}').mkdir(exist_ok=True)
  (d_f / f'{i}').mkdir(exist_ok=True)

for i, bmp in enumerate(tqdm(true_t.glob('*.bmp'))):
  im = cv2.imread(str(bmp))
  h, w = im.shape[:2]
  scale = (RESOLUTION / (w * h)) ** 0.5
  img_reshaped = cv2.resize(im, dsize=None, fx=scale, fy=scale)

  subject = bmp.name.split('-')[1]
  subject = dict_of_label_subjectno[int(subject)]

  cv2.imwrite(str(data_dir / 'true' / f'{subject}/{i}.bmp'), img_reshaped)

for i, bmp in enumerate(tqdm(false_t.glob('*.bmp'))):
  im = cv2.imread(str(bmp))
  h, w = im.shape[:2]
  scale = (RESOLUTION / (w * h)) ** 0.5
  img_reshaped = cv2.resize(im, dsize=None, fx=scale, fy=scale)

  subject = bmp.name.split('-')[1]
  subject = dict_of_label_subjectno[int(subject)]

  cv2.imwrite(str(data_dir / 'false' / f'{subject}/{i}.bmp'), img_reshaped)
