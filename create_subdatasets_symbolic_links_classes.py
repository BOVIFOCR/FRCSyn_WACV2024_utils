import os
import numpy as np
import sys
import argparse
import ast


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output', help='')
    parser.add_argument('--output_path', type=str, default='/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace', help='')
    # parser.add_argument('--classes_list', type=str, default='/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/CASIA-WebFace_100_bigger_classes.txt', help='')
    parser.add_argument('--classes_list', type=str, default='', help='')
    parser.add_argument('--num_classes', type=str, default='100', help='')
    # parser.add_argument('--num_classes', type=str, default='[100,5000]', help='')
    
    args = parser.parse_args()
    if type(args.num_classes) is str:
        if '[' in args.num_classes and ']' in args.num_classes:
            args.num_classes = ast.literal_eval(args.num_classes)
        else:
            args.num_classes = [int(args.num_classes)]
    return args


def load_classes_names_from_file(path_file):
    with open(path_file, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

        if len(lines) > 0 and '/' in lines[0]:
            for i in range(len(lines)):
                lines[i] = lines[i].split('/')[-1]
    return lines


def concat_path_into_strings(path='', str_list=['']):
    for i in range(len(str_list)):
        str_list[i] = os.path.join(path, str_list[i])
    return str_list


def get_subfolders(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs


def make_symbolic_links_folders(src_path='', folders_names=[''], limit_folders=10, dst_path=''):
    assert len(folders_names) >= limit_folders
    assert os.path.exists(src_path)
    if not os.path.exists(dst_path):
        print('Making destination folder \'' + dst_path + '\' ...')
        os.makedirs(dst_path)

    print('Making symbolic links in \'' + dst_path + '\' ...')
    for i, folder_name in enumerate(folders_names[:limit_folders]):
        src_folder_path = src_path + '/' + folder_name
        dst_folder_path = dst_path + '/' + folder_name
        command = 'ln -s ' + src_folder_path + ' ' + dst_folder_path
        print('%d/%d - %s' % (i+1, limit_folders, dst_folder_path), end='\r')
        os.system(command)
    print()



if __name__ == '__main__':
    
    args = getArgs()
    # print('args:', args)
    # sys.exit(0)

    if args.classes_list == '':
        print('Searching all subfolders in \'' + args.input_path + '\' ...')
        subfolders_list = get_subfolders(args.input_path)    
    else:
        subfolders_list = load_classes_names_from_file(args.classes_list)
    print('len(subfolders_list):', len(subfolders_list))
    # print('subfolders_list:', subfolders_list)
    # sys.exit(0)


    if args.output_path == '':
        args.output_path = args.input_path.split('/')[-1]


    for num_class in args.num_classes:
        tgt_folder_name = args.input_path.split('/')[-1]
        path_target_symb_links = os.path.join(args.output_path, tgt_folder_name+'_'+str(num_class)+'class')
        print('\nMaking %s symbolic links...' % (num_class))
        make_symbolic_links_folders(args.input_path, subfolders_list, num_class, path_target_symb_links)

