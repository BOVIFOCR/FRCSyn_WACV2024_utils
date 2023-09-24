import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
from retinaface.retinaface import RetinaFace
from insightface.utils import face_align


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace', help='the dir your dataset of face which need to crop')
    parser.add_argument('--output_path', type=str, default='/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace_crop', help='the dir the cropped faces of your dataset where to save')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=int, default=112, help='the size of the face to save, the size x%2==0, and width equal height')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold for face detection')
    args = parser.parse_args()
    return args


def draw_bbox(img, bbox):
    result_img = img.copy()
    x, y, width, height = bbox
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(result_img, (int(round(x)), int(round(y))),
                              (int(round(x + width)), int(round(y + height))), color, thickness)
    return result_img


def draw_lmks(img, lmks):
    result_img = img.copy()
    for l in range(lmks.shape[0]):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(result_img, (int(round(lmks[l][0])), int(round(lmks[l][1]))), 1, color, 2)
    return result_img


def crop_align_face(args):
    input_dir = args.input_path
    output_dir = args.output_path
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    # detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    detector = RetinaFace('./retinaface/model/retinaface-R50/R50', 0, args.gpu, 'net3')

    count_no_find_face = 0
    count_crop_images = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root, dirs, files in tqdm(os.walk(input_dir)):
        for file_name in files:
            
            if not file_name.split('.')[-1] == 'jpg' \
                and not file_name.split('.')[-1] == 'jpeg' \
                and not file_name.split('.')[-1] == 'png':
                continue

            output_root = root.replace(input_dir, output_dir)
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            file_path = os.path.join(root, file_name)
            print(f'Reading {file_path}')
            face_img = cv2.imread(file_path)
            
            print(f'Detecting face...')
            ret = detector.detect(face_img, args.thresh, scales=[1.0], do_flip=False)
            
            if ret is None:
                print('%s do not find face'%file_path)
                count_no_find_face += 1
                continue
            bbox, points = ret
            if bbox.shape[0] == 0:
                print('%s do not find face'%file_path)
                count_no_find_face += 1
                continue

            for i in range(bbox.shape[0]):
                bbox_ = bbox[i, 0:4]
                # points_ = points[i, :].reshape((2, 5)).T
                points_ = points[i, :].reshape((5, 2))

                face_img_copy = draw_bbox(face_img, bbox_)
                face_img_copy = draw_lmks(face_img, points_)
                face_name = '%s_bbox.png'%(file_name.split('.')[0])
                file_path_bbox_save = os.path.join(output_root, face_name)
                print(f'Saving {file_path_bbox_save}')
                cv2.imwrite(file_path_bbox_save, face_img_copy)

                face = face_align.norm_crop(face_img, landmark=points_, image_size=args.face_size)

                face_name = '%s.png'%(file_name.split('.')[0])
                
                file_path_save = os.path.join(output_root, face_name)
                print(f'Saving {file_path_save}')
                cv2.imwrite(file_path_save, face)
                print('-------------')
                break   # take only the most confident face in image

            count_crop_images += 1

    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)

if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)