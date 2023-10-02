import os
import sys
import time
import datetime
import argparse


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, default='/datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000', help='')
    parser.add_argument('--ext1', type=str, default='.jpg', help='')
    parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('--path2', type=str, default='/datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_112x112', help='')
    parser.add_argument('--ext2', type=str, default='.png', help='')
    parser.add_argument('--save_list', type=str, default='', help='')

    args = parser.parse_args()

    return args


def get_all_files_in_path(folder_path, file_extension='.jpg', pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            if pattern in path_file and path_file.endswith(file_extension):
                file_list.append(path_file)
    file_list.sort()
    return file_list


def add_string_end_file(file_path, string_to_add):
    string_to_add += '\n'
    try:
        with open(file_path, 'a') as file:
            file.write(string_to_add)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(string_to_add)


def check_corresponding_files(args):
    args.path1 = args.path1.rstrip('/')
    args.path2 = args.path2.rstrip('/')

    print(f'\nSearching \'{args.ext1}\' files with pattern \'{args.str_pattern}\' in path \'{args.path1}\'...')
    all_img_paths1 = get_all_files_in_path(args.path1, args.ext1, args.str_pattern)
    assert len(all_img_paths1) > 0, f'No files found with extention {args.ext1} in path \'{args.path1}\''
    print(f'{len(all_img_paths1)} files found\n')
    # for i, img_path in enumerate(all_img_paths):
    #     print(f'img_path: {img_path}')

    # if args.save_list != '':
    #     open(args.save_list, 'w')

    non_corresponding_files = 0

    for i, img_path1 in enumerate(all_img_paths1):
        img_path2 = img_path1.replace(args.path1, args.path2).replace(args.ext1, args.ext2)
        print(f'\rChecking image {i+1}/{len(all_img_paths1)}', end='')
        # print(f'img_path1: {img_path1}')
        # print(f'img_path2: {img_path2}')

        if not os.path.isfile(img_path2):
            non_corresponding_files += 1
            print(f'\n{non_corresponding_files} non-corresponding file found:')
            print(f'img_path1: {img_path1}')
            print(f'img_path2: {img_path2}')
            if args.save_list != '':
                print(f'    adding \'img_path1\' to file \'{args.save_list}\'')
                add_string_end_file(args.save_list, img_path1)

            print('----------')

    print('\n---------------------')
    print('Finished')
    print(f'{non_corresponding_files} non-corresponding files found!')
    if non_corresponding_files > 0 and args.save_list != '':
        print(f'    Paths saved in file \'{args.save_list}\'')
    print('')



if __name__ == '__main__':
    args = getArgs()
    check_corresponding_files(args)
