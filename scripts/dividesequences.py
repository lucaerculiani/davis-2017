import argparse
import logging
import shutil
from pathlib import Path

import numpy as np



def main(cmdline):

    min_seq_len = cmdline.sequences
    if cmdline.length is not None:
        min_seq_len *= cmdline.length

    gen = gen_get_items(cmdline.dataset, cmdline.outpath, min_seq_len)
    
    if cmdline.soft_links:
        copy_fn = softlink
    else:
        copy_fn = copy
    
    for input_p, output_p in gen:
        divide_seq(input_p,
                   output_p,
                   cmdline.length,
                   cmdline.sequences,
                   copy_fn)


def get_images(path):
    return list(path.glob("*.jpg"))

def gen_get_items(base_path, out_path, img_n):

    for seq_path in (sub for sub in base_path.iterdir() if sub.is_dir()):
        for seq_id_path in (sub for sub in seq_path.iterdir() if sub.is_dir()):
            found_img_n = len(get_images(seq_id_path))
            if found_img_n >= img_n:
                outdir = out_path / (seq_path.name + '_' + seq_id_path.name)
                outdir.mkdir(exist_ok=True, parents=False)
                yield seq_id_path, outdir
            else:
                info = "discarded subpath {}, found only {} images"
                info = info.format(seq_id_path.relative_to(base_path),
                                   found_img_n)
                logging.info(info)


def divide_seq(input_p, output_p, s_length, s_number, copy_fn):
    images = sorted(get_images(input_p))

    splits = np.array_split(images, s_number)

    if s_length is not None:
        splits = [elem[:s_length] for elem in splits[:-1]] + \
                 [splits[-1][-s_length:]]

    for seq_id, seq_elems in zip(range(len(splits)), splits):
        seq_id_folder = output_p / str(seq_id)
        seq_id_folder.mkdir(exist_ok=True)

        for elem in seq_elems:
            dest = seq_id_folder / elem.name
            copy_fn(elem,dest)


def copy(elem, dest):
    shutil.copyfile(str(elem), str(dest))

def softlink(elem, dest):
    dest.symlink_to(elem)

def abs_path(path):
    return Path(path).resolve()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=abs_path,
                        help="directory containing the davis full res dataset")
    parser.add_argument("outpath", type=abs_path,
                        help="output directory")
    parser.add_argument("-l", '--length', type=int, default=None,
                        help="lenght of every sequence")
    parser.add_argument("-n", '--sequences', type=int, default=2,
                        help="number of sequences to create")
    parser.add_argument("-s", '--soft-links', action='store_true',
                        help="create softlinks instead of copies")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
