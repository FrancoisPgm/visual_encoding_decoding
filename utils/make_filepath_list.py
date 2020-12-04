from bids import BIDSLayout
import os

dataset_path = "data/friends"
derivatives_path = os.path.join(dataset_path, "derivatives/fmriprep-20.1.0/fmriprep")
out_path = "utils/s1_file_paths.txt"

layout=BIDSLayout(dataset_path, derivatives=derivatives_path)
layout.save("data/external/pybids_cache")
print(layout.get_subjects())
file_paths = layout.get(subject="", session="", task=["s01", "s02"], suffix="^bold$", extension="nii.gz",
                        scope="derivatives", regex_search=True, return_type="file")

print(file_paths)
with open(out_path, "w") as f:
    for path in file_paths:
        f.write(path+"\n")

print("Paths written to {}.".format(out_path))
