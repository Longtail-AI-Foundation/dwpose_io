"""Reader and writer for the safetensors file: the six arrays of a DwPoseSequence
stored as tensors under their attribute names, plus one metadata entry, 'dwpose',
holding a JSON string with format_version, keypoint_layout, coord_space,
video_hash, width and height. A single entry with sorted JSON keys keeps the
writer byte-deterministic; safetensors itself writes multi-key metadata in
random order."""

import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save, save_file

from .sequence import COORD_SPACE, FORMAT_VERSION, KEYPOINT_LAYOUT, PERSON_ARRAY_NAMES, TENSOR_NAMES, DwPoseSequence

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


def read_frame_range(handle, frames):
    """The six arrays of the frames a slice selects, read with get_slice so only
    that span of the file is loaded. The slice is clamped like a list slice and
    its step must be 1; the result matches DwPoseSequence.frame_range."""
    assert isinstance(frames, slice), f'(read_frame_range): frames is {type(frames).__name__}, expected slice'
    offsets = handle.get_slice('frame_offsets')
    num_frames = offsets.get_shape()[0] - 1
    start, stop, step = frames.indices(num_frames)
    assert step == 1, f'(read_frame_range): slice step {step}, expected 1'
    stop = max(stop, start)
    frame_offsets = offsets[start:stop + 1]
    p0, p1 = int(frame_offsets[0]), int(frame_offsets[-1])

    def take(name, a, b, dtype):
        """Rows a:b of a tensor. safetensors refuses an empty slice that starts at
        the end of a tensor, so an empty range is built instead of sliced."""
        tensor = handle.get_slice(name)
        if a == b:
            return np.zeros((0,) + tuple(tensor.get_shape()[1:]), dtype=dtype)
        return tensor[a:b]

    arrays = {name: take(name, p0, p1, np.float32) for name in PERSON_ARRAY_NAMES}
    arrays['frame_offsets'] = frame_offsets - p0
    arrays['timestamps_ms'] = take('timestamps_ms', start, stop, np.float64)
    return arrays


def read_dw_pose(path, frames=None):
    """Drop-in replacement for pickle.load on a dw_pose pickle: the returned
    DwPoseSequence answers seq[f]['predictions'][0][k]['keypoints'] and friends.
    With frames, a slice of frame indices, only those frames are read from the
    file and the result equals read_dw_pose(path)[frames], so the memory cost is
    the size of the range rather than the size of the file."""
    with safe_open(path, framework='numpy') as handle:
        assert set(handle.keys()) == set(TENSOR_NAMES), f'(read_dw_pose): tensors {sorted(handle.keys())}, expected {sorted(TENSOR_NAMES)}'
        video_hash, width, height = read_metadata(handle)
        if frames is None:
            arrays = {name: handle.get_tensor(name) for name in TENSOR_NAMES}
        else:
            arrays = read_frame_range(handle, frames)
    seq = DwPoseSequence(**arrays, video_hash=video_hash, width=width, height=height)
    seq.validate()
    return seq
