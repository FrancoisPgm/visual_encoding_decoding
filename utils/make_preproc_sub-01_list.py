import os

preproc_dir = "data/preprocessed/fmri"

preproc_sub01 = [os.path.join(preproc_dir, p) for p in os.listdir(preproc_dir) if "sub-01" in p]

with open("utils/sub-01_preproc_list.txt", "w") as f:
    for path in preproc_sub01:
        f.write(path+"\n")
