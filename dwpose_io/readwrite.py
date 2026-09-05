"""Reader and writer for the safetensors file: the six arrays of a DwPoseSequence
stored as tensors under their attribute names, plus one metadata entry, 'dwpose',
holding a JSON string with format_version, keypoint_layout, coord_space,
video_hash, width and height. A single entry with sorted JSON keys keeps the
writer byte-deterministic; safetensors itself writes multi-key metadata in
random order."""

import json

from safetensors import safe_open
from safetensors.numpy import save, save_file

from .sequence import COORD_SPACE, FORMAT_VERSION, KEYPOINT_LAYOUT, TENSOR_NAMES, DwPoseSequence

METADATA_KEY = 'dwpose'


def sequence_to_tensors(seq):
    """Split a DwPoseSequence into the tensor dict and the metadata dict that
    safetensors stores."""
    seq.validate()
    tensors = {name: getattr(seq, name) for name in TENSOR_NAMES}
    fields = {
        'format_version': FORMAT_VERSION,
        'keypoint_layout': KEYPOINT_LAYOUT,
        'coord_space': COORD_SPACE,
        'video_hash': seq.video_hash,
        'width': seq.width,
        'height': seq.height,
    }
    return tensors, {METADATA_KEY: json.dumps(fields, sort_keys=True)}


def write_dw_pose(seq, path):
    tensors, metadata = sequence_to_tensors(seq)
    save_file(tensors, path, metadata=metadata)


def serialize_dw_pose(seq):
    """The exact bytes write_dw_pose puts on disk."""
    tensors, metadata = sequence_to_tensors(seq)
    return save(tensors, metadata=metadata)


def read_metadata(handle):
    meta = handle.metadata()
    assert meta is not None and set(meta.keys()) == {METADATA_KEY}, f'(read_metadata): metadata keys {sorted(meta.keys()) if meta else None}, expected [{METADATA_KEY!r}]'
    fields = json.loads(meta[METADATA_KEY])
    for key, expected in (('format_version', FORMAT_VERSION), ('keypoint_layout', KEYPOINT_LAYOUT), ('coord_space', COORD_SPACE)):
        assert fields.get(key) == expected, f'(read_metadata): {key} is {fields.get(key)!r}, expected {expected!r}'
    return fields['video_hash'], fields['width'], fields['height']


def read_dw_pose(path):
    """Drop-in replacement for pickle.load on a dw_pose pickle: the returned
    DwPoseSequence answers seq[f]['predictions'][0][k]['keypoints'] and friends."""
    with safe_open(path, framework='numpy') as handle:
        assert set(handle.keys()) == set(TENSOR_NAMES), f'(read_dw_pose): tensors {sorted(handle.keys())}, expected {sorted(TENSOR_NAMES)}'
        video_hash, width, height = read_metadata(handle)
        arrays = {name: handle.get_tensor(name) for name in TENSOR_NAMES}
    seq = DwPoseSequence(**arrays, video_hash=video_hash, width=width, height=height)
    seq.validate()
    return seq
