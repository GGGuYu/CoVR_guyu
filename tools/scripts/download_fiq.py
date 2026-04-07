import os
import urllib.request
import argparse

### This is a script to download the fashion-iq dataset

# python download_image_data.py --split=0
# python download_image_data.py --split=1
# python download_image_data.py --split=2

# 使用 argparse 替代 tf.app.flags
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=0, help='split')
FLAGS = parser.parse_args()

readpath = ['/media/bd/PSSD/datasets/fashion-iq/image_tag_dataset/image_url/asin2url.dress.txt', \
            '/media/bd/PSSD/datasets/fashion-iq/image_tag_dataset/image_url/asin2url.shirt.txt', \
            '/media/bd/PSSD/datasets/fashion-iq/image_tag_dataset/image_url/asin2url.toptee.txt']

savepath = ['/media/bd/PSSD/datasets/fashion-iq/image_data/dress', \
            '/media/bd/PSSD/datasets/fashion-iq/image_data/shirt', \
            '/media/bd/PSSD/datasets/fashion-iq/image_data/toptee']

missing_file = ['/media/bd/PSSD/datasets/fashion-iq/missing_dress.log', \
                '/media/bd/PSSD/datasets/fashion-iq/missing_shirt.log', \
                '/media/bd/PSSD/datasets/fashion-iq/missing_toptee.log']

k = FLAGS.split

with open(missing_file[k], 'a') as f:
  missing = 0
  file = open(readpath[k], "r")
  lines = file.readlines()
  print(len(lines))
  for i in range(len(lines)):
    try:
      line = lines[i].replace('\n','').split(' \t ')
      url = line[1]
      imgpath = os.path.join(savepath[k], line[0]+'.jpg')
      urllib.request.urlretrieve(url, imgpath)
    except:
      missing += 1
      f.write(imgpath)
      print(imgpath)
      pass

print("missing %d." % missing)