"""The ISL DWPose safetensors format. read_dw_pose replaces pickle.load and
write_dw_pose replaces pickle.dump for dw_pose files; the returned sequence keeps
the old seq[f]['predictions'][0][k]['keypoints'] indexing."""

from .check import check_dw_pose
from .legacy import legacy_to_sequence, load_legacy_pickle
from .readwrite import read_dw_pose, write_dw_pose
from .sequence import NUM_KEYPOINTS, DwPoseSequence
