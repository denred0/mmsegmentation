from pathlib import Path
import shutil


def get_all_files_in_folder(root_dir, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(list(root_dir.glob(f"*{t}")))
    return sorted(files_grabbed)


def main(label_name):
    root_initial = Path('prepare_dataset/laser/' + label_name + '/00_usable')
    type_png = '.png'
    files_initial = get_all_files_in_folder(root_initial, type_png)

    print(f"Count initial files {len(files_initial)}")

    root_exist = Path('data/data_new/images')
    files_already_exist = get_all_files_in_folder(root_exist, type_png)

    for i_file in files_initial:
        exist = False
        for e_file in files_already_exist:
            if i_file.stem == e_file.stem:
                exist = True
        if not exist:
            shutil.copy(Path(root_initial).joinpath(i_file.stem + type_png),
                        Path('prepare_dataset/for_mark/laser/' + label_name))

    files_transfered = get_all_files_in_folder(Path('prepare_dataset/for_mark/laser/' + label_name), type_png)
    print(f"Count transfered files {len(files_transfered)}")

if __name__ == '__main__':
    label_name = 'рисунок'
    main(label_name=label_name)
