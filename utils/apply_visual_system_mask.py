import sys
import os
import numpy as np
from load_confounds import Params36
from nilearn.masking import apply_mask
from nilearn.image import clean_img

mask_path = "data/external/visual_mask.nii.gz"
dataset_path = "data/friends"
derivatives_path = os.path.join(dataset_path, "derivatives/fmriprep-20.1.0/fmriprep")

i = int(sys.argv[1])

with open("utils/remaining_files_to_mask.txt", "r") as f:
    file_list = [line.rstrip('\n') for line in f]

for file_path in file_list[10*i:10*(i+1)]:
    file_name = os.path.split(file_path)[1][:-6]+"npy"
    conf = Params36().load(file_path)

    cleaned_img = clean_img(file_path, confounds=conf)
    masked_data = apply_mask(cleaned_img, mask_path)

    np.save(os.path.join("data/preprocessed/fmri/", file_name), masked_data)
    print("masked date saved at {}".format(os.path.join("data/preprocessed/fmri/", file_name)))
    del(conf)
    del(cleaned_img)
    del(masked_data)

