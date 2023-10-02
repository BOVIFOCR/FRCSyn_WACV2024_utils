import os
import sys
import time
import datetime
import argparse


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', help='')
    parser.add_argument('--sample_list', type=str, default='', help='')
    parser.add_argument('--protocol', type=str, default='', help='')
    parser.add_argument('--save_list', type=str, default='', help='')

    args = parser.parse_args()

    return args


def add_string_end_file(file_path, string_to_add):
    string_to_add += '\n'
    try:
        with open(file_path, 'a') as file:
            file.write(string_to_add)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(string_to_add)


def get_all_paths_from_file(file_path, pattern='', sort=False):
    with open(file_path, 'r') as file:
        all_lines = [line.strip() for line in file.readlines()]
        valid_lines = []
        for i, line in enumerate(all_lines):
            if '\\' in line:
                line = line.replace('\\', '/')
            if '\x00' in line:
                line = line.replace('\x00', '/000')
            if pattern in line:
                valid_lines.append(line)
        if sort:
            valid_lines.sort()
        # print('all_lines:', all_lines)
        # sys.exit(0)
        return valid_lines


def add_substring(paths, substring, position='begin'):   # position='end'
    paths_updt = []
    for i, path in enumerate(paths):
        if position == 'begin':
            path_updt = substring.rstrip('/') + '/' + path.lstrip('/')
        elif position == 'end':
            path_updt = path.rstrip('/') + '/' + substring.lstrip('/')
        paths_updt.append(path_updt)
        paths_updt.sort()
    return paths_updt


def remove_substring(paths, substring):
    paths_updt = []
    for i, path in enumerate(paths):
        path_updt = path.replace(substring, '').lstrip('/')
        # print('paths_updt:', paths_updt)
        paths_updt.append(path_updt)
    return paths_updt


def check_list1_into_list2(list1, list2):
    samples_from_list1_in_list2 = []
    for i, sample_list1 in enumerate(list1):
        for j, sample_list2 in enumerate(list2):
            if sample_list1 in sample_list2:
                samples_from_list1_in_list2.append(sample_list1)
    samples_from_list1_in_list2 = list(set(samples_from_list1_in_list2))
    samples_from_list1_in_list2.sort()
    return samples_from_list1_in_list2


def add_string_end_file(file_path, string_to_add):
    string_to_add += '\n'
    try:
        with open(file_path, 'a') as file:
            file.write(string_to_add)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(string_to_add)


def save_list_to_file(file_path, string_list):
    with open(file_path, 'w') as file:
        for i, line in enumerate(string_list):
            file.write(line + '\n')


def check_samples_in_protocol(args):
    print(f'\nLoading paths from file \'{args.sample_list}\' ...')
    samples_paths = get_all_paths_from_file(args.sample_list)
    protocol_paths = get_all_paths_from_file(args.protocol)
    # print('samples_paths:', samples_paths)
    # print('protocol_paths:', protocol_paths)
    # sys.exit(0)

    samples_paths_updt = remove_substring(samples_paths, args.input_path)
    # print('samples_paths_updt:', samples_paths_updt)
    # sys.exit(0)

    print(f'Checking samples in protocol ...')
    samples_in_protocol = check_list1_into_list2(samples_paths_updt, protocol_paths)
    print(f'    {len(samples_in_protocol)} samples found in protocol')
    # print('samples_in_protocol:', samples_in_protocol)

    if len(samples_in_protocol) > 0:
        samples_in_protocol_updt = add_substring(samples_in_protocol, args.input_path, 'begin')
        # print('samples_in_protocol_updt:', samples_in_protocol_updt)
        if args.save_list != '':
            path_samples_in_protocol = args.save_list
        else:
            path_samples_in_protocol = os.join('/'.join(args.sample_list.split('/')[:-1]), 'samples_no_face_detected_in_test_protocol.txt')
        print(f'    Saving samples paths in file \'{path_samples_in_protocol}\' ...')
        save_list_to_file(path_samples_in_protocol, samples_in_protocol_updt)
        # for i, sample_updt in enumerate(samples_in_protocol_updt):
        #     add_string_end_file(path_samples_in_protocol, sample_updt)

    print('\nFinished!')



if __name__ == '__main__':
    args = getArgs()
    check_samples_in_protocol(args)
