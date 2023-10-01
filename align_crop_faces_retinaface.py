# Dataset BUPT-BalancedFace
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000 --output_path /datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_112x112 --thresh 0.8 --scales [0.5]

# Dataset FFHQ
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/2_FFHQ/images1024x1024 --output_path /datasets2/frcsyn_wacv2024/datasets/real/2_FFHQ/images_crops_112x112 --thresh 0.8 --scales [0.5]

# Dataset AgeDB
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/4_AgeDB/03_Protocol_Images --output_path /datasets2/frcsyn_wacv2024/datasets/real/4_AgeDB/03_Protocol_Images_crops_112x112 --thresh 0.5 --scales [1.0]

import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import ast
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
    parser.add_argument('--scales', type=str, default='[1.0]', help='the scale to resize image before detecting face')
    parser.add_argument('--draw_bbox_lmk', action='store_true', help='')
    args = parser.parse_args()

    args.scales = ast.literal_eval(args.scales)
    return args


def draw_bbox(img, bbox):
    result_img = img.copy()
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(result_img, (int(round(x1)), int(round(y1))),
                              (int(round(x2)), int(round(y2))), color, thickness)
    return result_img


def draw_lmks(img, lmks):
    result_img = img.copy()
    for l in range(lmks.shape[0]):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(result_img, (int(round(lmks[l][0])), int(round(lmks[l][1]))), 1, color, 2)
    return result_img


def get_all_files_in_path(folder_path, file_extension='.jpg'):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(file_extension):
                file_list.append(os.path.join(root, filename))
    file_list.sort()
    return file_list


def crop_align_face(args):
    input_dir = args.input_path.rstrip('/')
    output_dir = args.output_path.rstrip('/')
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    det_path = './retinaface/model/retinaface-R50/R50'
    print(f'\nLoading face detector \'{det_path}\'...')
    detector = RetinaFace(det_path, 0, args.gpu, 'net3')

    count_no_find_face = 0
    count_crop_images = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ext = '.jpg'
    print(f'\nSearching \'{ext}\' files in path \'{input_dir}\'...')
    all_img_paths = get_all_files_in_path(input_dir, ext)
    assert len(all_img_paths) > 0, f'Error: no files found with extention {ext} in path \'{input_dir}\''
    # for i, img_path in enumerate(all_img_paths):
    #     print(f'{i}/{len(all_img_paths)-1} - path: {img_path}')
    # print('len(all_img_paths):', len(all_img_paths))
    # sys.exit(0)

    for i, input_img_path in enumerate(all_img_paths):
        print(f'{i}/{len(all_img_paths)}\nReading {input_img_path} ...')
        face_img = cv2.imread(input_img_path)

        print(f'Detecting face...')
        ret = detector.detect(face_img, args.thresh, args.scales, do_flip=False)

        bbox, points = ret
        if bbox.shape[0] == 0:
            print('%s do not find face'%input_img_path)
            count_no_find_face += 1
            continue

        confidences = [bbox[idx, 4] for idx in range(bbox.shape[0])]
        print('Confidences:', confidences)

        for bbox_idx in range(bbox.shape[0]):
            bbox_ = bbox[bbox_idx, 0:4]
            points_ = points[bbox_idx, :].reshape((5, 2))

            print(f'Aligning and cropping to size {args.face_size}x{args.face_size} ...')
            face = face_align.norm_crop(face_img, landmark=points_, image_size=args.face_size)

            # face_name = '%s.png'%(file_name.split('.')[0])
            output_img_path = input_img_path.replace(input_dir, output_dir)
            face_name = output_img_path.split('/')[-1].split('.')[0] + '.png'
            output_img_path = os.path.join(os.path.dirname(output_img_path), face_name)
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            
            if args.draw_bbox_lmk:
                face_img_copy = draw_bbox(face_img, bbox_)
                face_img_copy = draw_lmks(face_img_copy, points_)
                face_name = '%s_bbox.png'%(input_img_path.split('/')[-1].split('.')[0])
                file_path_bbox_save = os.path.join(os.path.dirname(output_img_path), face_name)
                print(f'Saving {file_path_bbox_save}')
                cv2.imwrite(file_path_bbox_save, face_img_copy)

            print(f'Saving {output_img_path} ...')
            cv2.imwrite(output_img_path, face)
            print('-------------')
            break   # take only the most confident face in image

        count_crop_images += 1

    print('-------------------------------')
    print('Finished')
    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)


if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)
