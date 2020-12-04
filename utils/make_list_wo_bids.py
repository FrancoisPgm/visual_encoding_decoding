import os

base_path = "data/friends/derivatives/fmriprep-20.1.0/fmriprep"
file_paths = []
out_path = "utils/s1s2_wobids.txt"


for i in range(1,7):
    for sesdir in os.listdir(os.path.join(base_path, "sub-0{}".format(i))):
        if "ses-" in sesdir:
            func_dir = os.path.join(base_path, "sub-0{}".format(i), sesdir, "func")
            print(func_dir)
            for filename in os.listdir(func_dir):
                if "task-s01" in filename or "task-s02" in filename:
                    if "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in filename:
                        file_paths.append(os.path.join(func_dir, filename))


with open(out_path, "w") as f:
    for path in file_paths:
        f.write(path+"\n")

print("Paths written to {}.".format(out_path))
