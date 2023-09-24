# https://gist.github.com/mk-minchul/90170e3a71fef08c85e6ee76197cdc52
# Run `pip install mxnet` (if necessary, run 'pip install tqdm Pillow opencv-python')
# Run `python convert.py --rec_path ./faces_webface_112x112' (note the '.' at the end of the command)
# Find images in png format inside `./faces_webface_112x112/imgs` folder.

from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import cv2
import numbers

def save_rec_to_img_dir(rec_path, swap_color_channel=False, save_as_png=True):

    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if swap_color_channel:
            # this option saves the image in the right color.
            # but the training code uses PIL (RGB)
            # and validation code uses Cv2 (BGR)
            # so we want to turn this off to deliberately swap the color channel order.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()

        if save_as_png:
            img_save_path = label_path/'{}.png'.format(idx)
            img.save(img_save_path)
        else:
            img_save_path = label_path/'{}.jpg'.format(idx)
            img.save(img_save_path, quality=95)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='./faces_emore', type=str)
    parser.add_argument("--swap_color_channel", action='store_true')

    args = parser.parse_args()
    rec_path = Path(args.rec_path)
    # unfolds train.rec to image folders
    save_rec_to_img_dir(rec_path, swap_color_channel=args.swap_color_channel)