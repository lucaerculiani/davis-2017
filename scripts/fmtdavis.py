import argparse
from pathlib import Path

import scipy
from skimage import io
from PIL import Image
from joblib import Parallel, delayed
import numpy as np

import davis
# davis.io.imread_indexed


_ANN_REL_PATH = Path("Annotations/Full-Resolution/")
_IMG_REL_PATH = Path("JPEGImages/Full-Resolution/")

_TRAIN_REL_PATH= Path("ImageSets/2017/train.txt")
_VAL_REL_PATH= Path("ImageSets/2017/val.txt")




def main(cmdline):
    gen = gen_get_elems(cmdline.dataset, cmdline.outpath)

    if cmdline.workers == 0:
        for input_tuple in gen:
            process_seq(*input_tuple)

    else:
        par = Parallel(n_jobs=cmdline.workers)
        par(delayed(process_seq)(*input_tuple) for input_tuple in gen)



def gen_get_elems(base_path, out_path):

    ann_p = base_path / _ANN_REL_PATH
    img_p = base_path / _IMG_REL_PATH

    for sub_d in img_p.iterdir():
        name = sub_d.name
        outdir = out_path / name
        outdir.mkdir(exist_ok=True)
        yield sub_d, ann_p / name, outdir



def process_seq(img_d, ann_d, dest_d):


    ann_l =  sorted(ann_d.glob("*.png"))
    img_l = list(img_d / ( str(ann.name).rstrip("png") + "jpg")
                 for ann in ann_l)

    annotations = np.array([davis.io.imread_indexed(str(ann))[0]
                            for ann in ann_l])

    n_labels = np.unique(annotations)
    n_labels = n_labels[n_labels != 0]

    for lab_id in n_labels:
        out_f = dest_d / str(lab_id)
        out_f.mkdir(exist_ok=True)

    for ann, img_path in zip(annotations, img_l):
        objs = scipy.ndimage.find_objects(ann)
        img = io.imread(str(img_path))

        for ann_rng, ann_id in zip(objs,np.arange(1, len(objs) + 1)):
            
            if ann_rng is not None:
                formatted = cut_image(img, ann, ann_id, ann_rng)
                out_path = dest_d / str(ann_id) / img_path.name
                formatted.save(str(out_path))






def cut_image(image, annotation, ann_id, ann_rng, res_edge=240):

    pad_color = np.array((127, 127,127)).astype(np.uint8)
    final_img = np.tile(pad_color, image.shape[0:2] + (1,))
    final_img[annotation == ann_id, ...] = image[annotation == ann_id, ...]

    final_img = final_img[ann_rng]
    squared = make_square(Image.fromarray(final_img))
    
    resized = squared.resize((res_edge, res_edge))

    return resized


    
def make_square(im,fill_color=(127, 127, 127)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    return new_im 



def abs_path(path):
    return Path(path).resolve()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=abs_path,
                        help="directory containing the davis full res dataset")
    parser.add_argument("outpath", type=abs_path,
                        help="output directory")
    parser.add_argument("-w", "--workers", type=int, default=0, 
                        help="number of worker processes")
    args = parser.parse_args()
    main(args)
