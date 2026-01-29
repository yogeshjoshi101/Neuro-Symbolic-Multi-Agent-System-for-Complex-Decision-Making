import csv, os
from collections import Counter
from PIL import Image

labels = []
with open('perception_data/labels.csv') as f:
    next(f)
    for line in f:
        fn, lab = line.strip().split(',')
        labels.append(int(lab))
print("class counts:", Counter(labels))

im_dir = os.path.join('perception_data', 'images')
files = os.listdir(im_dir)

for i in range(6):
    fn = files[i]
    im = Image.open(os.path.join(im_dir, fn))
    print(fn, im.size)
    im.show()
