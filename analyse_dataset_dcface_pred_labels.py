import csv
import argparse

def load_csv_as_list_of_dicts(file_path):
    data = []
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # print(row)
            path = row['path']                # 'White/Female/9999/46.jpg'
            path_race   = path.split('/')[0]  # 'White'
            path_gender = path.split('/')[1]  # 'Female'
            path_label  = path.split('/')[2]  # '9999'
            path_sample = path.split('/')[3]  # '46.jpg'

            row['path_race']   = path_race
            row['path_gender'] = path_gender
            row['path_label']  = path_label
            row['path_sample']  = path_sample

            data.append(row)
    return data


def count_differences_by_combination(data):
    differences = {
        'Female_Asian': 0,
        'Female_Black': 0,
        'Female_Indian': 0,
        'Female_Other': 0,
        'Female_White': 0,
        'Male_Asian': 0,
        'Male_Black': 0,
        'Male_Indian': 0,
        'Male_Other': 0,
        'Male_White': 0,
    }

    for entry in data:
        gender = entry['gender']
        race = entry['race']
        label = entry['label']
        path_gender = entry['path_gender']
        path_race = entry['path_race']
        path_label = entry['path_label']

        if gender != path_gender:
            # differences[f'{path_gender}_{path_race}'] += 1
            differences[f'{gender}_{race}'] += 1
        if race != path_race:
            # differences[f'{path_gender}_{path_race}'] += 1
            differences[f'{gender}_{race}'] += 1

    return differences


def main():
    parser = argparse.ArgumentParser(description='Load a CSV file into a list of dictionaries.')
    parser.add_argument('-file_path', type=str, help='Path to the CSV file')
    parser.add_argument('-rows', type=int, default=-1, help='')
    parser.add_argument('-print', action='store_true', help='')
    
    args = parser.parse_args()

    file_path = args.file_path
    print(f'Loading file \'{args.file_path}\'...')
    dcface_data = load_csv_as_list_of_dicts(file_path)
    
    if args.rows > -1:
        dcface_data = dcface_data[:args.rows]
    
    if args.print:
        for row in range(len(dcface_data)):
            print(f'row {row}/{len(dcface_data)}: {dcface_data[row]}')
            if row+1 == args.rows:
                break
    print()

    diffs = count_differences_by_combination(dcface_data)
    print(f'len(dcface_data): {len(dcface_data)}')
    print('diffs:', diffs)

if __name__ == '__main__':
    main()
