# https://gist.github.com/mk-minchul/90170e3a71fef08c85e6ee76197cdc52
# Run `pip install mxnet` (if necessary, run 'pip install tqdm Pillow opencv-python')
# Run `python convert.py --rec_path ./faces_webface_112x112' (note the '.' at the end of the command)
# Find images in png format inside `./faces_webface_112x112/imgs` folder.

import os, sys
from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import cv2
import numbers
import glob
import pandas as pd


# Class adapted from https://github.com/mk-minchul/dcface/blob/master/dcface/convert/record.py
class RecordReader():
    '''
    def __init__(self, root='/mckim/temp/temp_recfiles'):
        path_imgidx = os.path.join(root, 'file.idx')
        path_imgrec = os.path.join(root, 'file.rec')
        self.root = root
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, 'list.txt')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}
    '''

    def __init__(self, path_imgidx, path_imgrec):
        root = os.path.dirname(path_imgrec)
        self.root = root
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, 'list.txt')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        image = mx.image.imdecode(binary).asnumpy()
        path = self.index_to_path[index]
        return image, path

    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)

    def export(self, save_root, save_as_png=True):
        num_imgs = len(self.index_to_path.keys())
        for idx in self.index_to_path.keys():
            image, path = self.read_by_index(idx)
            img_save_path = os.path.join(save_root, path)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

            if save_as_png:
                img_dir = os.path.dirname(img_save_path)
                img_name = os.path.basename(img_save_path)
                img_name = img_name.split('.')[0] + '.png'
                img_save_path = os.path.join(img_dir, img_name)
            
            print(f'Extracting {idx}/{num_imgs} - \'{img_save_path}\'', end='\r')
            cv2.imwrite(img_save_path, image[:,:,::-1])

        print('')

    def existing_keys(self):
        return self.path_to_index.keys()

    def load_done_list(self):
        donelist_path = os.path.join(self.root, 'done_list.txt')
        if os.path.isfile(donelist_path):
            donelist = pd.read_csv(donelist_path, header=None, sep='\t')
            donelist.columns = ['type', 'path']
            return set(donelist['path'].values)
        else:
            return None


def save_rec_to_img_dir(rec_path, swap_color_channel=False, save_as_png=True):
    assert os.path.exists(rec_path), f'Error, path not found: \'{rec_path}\''
    if os.path.isdir(rec_path):
        rec_path = rec_path.rstrip('/')
        rec_file_path = glob.glob(os.path.join(rec_path, '*.rec'))
        assert len(rec_file_path) > 0, f'Error, no file \'*.rec\' found in path \'{rec_path}\''
        rec_file_path = rec_file_path[0]
    else:
        rec_file_path = rec_path
        rec_path = os.path.dirname(rec_path)
    idx_file_path = rec_file_path.replace('.rec', '.idx')
    
    rec_path = Path(rec_path)

    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()

    print(f'Reading files \'{idx_file_path}\' and \'{rec_file_path}\'')
    # imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    imgrec = mx.recordio.MXIndexedRecordIO(idx_file_path, rec_file_path, 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    print(f'save_path: \'{save_path}\'')

    if header.flag == 0:
        # reader = RecordReader(root=args.rec_path)
        reader = RecordReader(path_imgidx=idx_file_path, path_imgrec=rec_file_path)
        reader.export(save_path, save_as_png)

    elif header.flag == 2:
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

    print('Finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='./faces_emore', type=str)
    parser.add_argument("--swap_color_channel", action='store_true')

    args = parser.parse_args()

    # rec_path = Path(args.rec_path)
    # save_rec_to_img_dir(rec_path, swap_color_channel=args.swap_color_channel)

    save_rec_to_img_dir(args.rec_path, swap_color_channel=args.swap_color_channel)