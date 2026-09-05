"""Checker proving a safetensors file is a faithful conversion of a dw_pose
pickle. (a) reading the file and serializing it again reproduces its bytes;
(b) every stored array equals the pickle values cast to the stored dtype bit for
bit, with frame count, per-frame person count and person order preserved;
(c) walking the read sequence with the old pickle indexing yields the same values
and the tuple and float32 types consumers rely on. Stops at the first mismatch,
naming frame, person and field. Run as `python -m dwpose_io.check` to convert
one hardcoded hash into ~/dw_pose_test and check it."""

import os
import os.path as osp
import pickle

import numpy as np

from .legacy import legacy_frame_person_count, legacy_frame_timestamp, legacy_person_arrays, legacy_to_sequence, load_legacy_pickle
from .readwrite import read_dw_pose, serialize_dw_pose, write_dw_pose
from .sequence import NUM_KEYPOINTS


def same_bits(name, got, expected, where):
    """Assert two arrays agree in dtype, shape and every byte, naming the first
    differing element."""
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.dtype == expected.dtype, f'(same_bits): {where} {name} dtype {got.dtype}, expected {expected.dtype}'
    assert got.shape == expected.shape, f'(same_bits): {where} {name} shape {got.shape}, expected {expected.shape}'
    diff = np.flatnonzero(np.frombuffer(got.tobytes(), np.uint8) != np.frombuffer(expected.tobytes(), np.uint8))
    if len(diff) == 0:
        return
    e = int(diff[0]) // got.dtype.itemsize
    raise AssertionError(f'(same_bits): {where} {name}[{e}] is {got.ravel()[e]!r}, expected {expected.ravel()[e]!r}')


def cast_through_float32(value):
    """Round a nested list of floats to float32, the format's precision."""
    return np.asarray(value, dtype=np.float64).astype(np.float32)


def check_arrays_against_legacy(seq, frames):
    """Property (b)."""
    assert seq.num_frames == len(frames), f'(check_arrays_against_legacy): {seq.num_frames} frames, pickle has {len(frames)}'
    i = 0
    for f, frame in enumerate(frames):
        where = f'frame {f}'
        same_bits('timestamp', seq.timestamps_ms[f], np.float64(legacy_frame_timestamp(frame)), where)
        count = legacy_frame_person_count(frame)
        stored = int(seq.frame_offsets[f + 1] - seq.frame_offsets[f])
        assert stored == count, f'(check_arrays_against_legacy): {where} holds {stored} persons, pickle has {count}'
        for k, person in enumerate(frame['predictions'][0]):
            where = f'frame {f} person {k}'
            keypoints, scores, box, score = legacy_person_arrays(person)
            same_bits('keypoints', seq.keypoints[i], keypoints, where)
            same_bits('keypoint_scores', seq.keypoint_scores[i], scores, where)
            same_bits('bbox', seq.bbox[i], box, where)
            same_bits('bbox_score', seq.bbox_score[i], score, where)
            i += 1
    assert i == seq.num_persons, f'(check_arrays_against_legacy): walked {i} persons, file holds {seq.num_persons}'


def check_person_view(person, src, where):
    keypoints = person['keypoints']
    assert isinstance(keypoints, np.ndarray) and keypoints.dtype == np.float32 and keypoints.shape == (NUM_KEYPOINTS, 2), f'(check_person_view): {where} keypoints is {type(keypoints).__name__} {getattr(keypoints, "dtype", None)} {getattr(keypoints, "shape", None)}'
    same_bits('keypoints', keypoints, cast_through_float32(src['keypoints']), where)
    scores = person['keypoint_scores']
    assert isinstance(scores, np.ndarray) and scores.dtype == np.float32 and scores.shape == (NUM_KEYPOINTS,), f'(check_person_view): {where} keypoint_scores is {type(scores).__name__} {getattr(scores, "dtype", None)} {getattr(scores, "shape", None)}'
    same_bits('keypoint_scores', scores, cast_through_float32(src['keypoint_scores']), where)
    box = person['bbox']
    assert type(box) is tuple and len(box) == 1, f'(check_person_view): {where} bbox is {type(box).__name__} of length {len(box)}, expected 1-tuple'
    x1, y1, x2, y2 = box[0]
    same_bits('bbox', np.asarray([x1, y1, x2, y2], dtype=np.float32), cast_through_float32(src['bbox'][0]), where)
    score = person['bbox_score']
    assert type(score) is np.float32, f'(check_person_view): {where} bbox_score is {type(score).__name__}, expected numpy.float32'
    same_bits('bbox_score', score, src['bbox_score'], where)


def check_view_against_legacy(seq, frames):
    """Property (c)."""
    assert len(seq) == len(frames), f'(check_view_against_legacy): len(seq) is {len(seq)}, pickle has {len(frames)} frames'
    assert sum(1 for _ in seq) == len(frames), '(check_view_against_legacy): iterating seq gave a different count than len(seq)'
    for f, src in enumerate(frames):
        frame = seq[f]
        where = f'frame {f}'
        assert 'predictions' in frame and 'timestamp' in frame, f'(check_view_against_legacy): {where} keys {list(frame.keys())}'
        timestamp = frame['timestamp']
        assert type(timestamp) is float, f'(check_view_against_legacy): {where} timestamp is {type(timestamp).__name__}, expected float'
        same_bits('timestamp', np.float64(timestamp), np.float64(legacy_frame_timestamp(src)), where)
        preds = frame['predictions']
        assert type(preds) is list and len(preds) == 1, f'(check_view_against_legacy): {where} predictions is {type(preds).__name__} of length {len(preds)}, expected list of length 1'
        persons = preds[0]
        count = legacy_frame_person_count(src)
        assert len(persons) == count, f'(check_view_against_legacy): {where} view holds {len(persons)} persons, pickle has {count}'
        for k, person_src in enumerate(src['predictions'][0]):
            check_person_view(persons[k], person_src, f'frame {f} person {k}')


def check_dw_pose(pkl_path, st_path):
    """Assert properties (a), (b) and (c) for one pickle and its conversion,
    returning the read sequence."""
    seq = read_dw_pose(st_path)
    with open(st_path, 'rb') as fp:
        file_bytes = fp.read()
    assert serialize_dw_pose(seq) == file_bytes, f'(check_dw_pose): re-serializing {st_path} does not reproduce its bytes'
    frames = load_legacy_pickle(pkl_path, seq.video_hash)
    check_arrays_against_legacy(seq, frames)
    check_view_against_legacy(seq, frames)
    return seq


if __name__ == '__main__':
    import pandas as pd
    ISL_ROOT = '/home/ubuntu/general-purpose/ISL'
    VIDEO_HASH = '00029cc0d4716bac03ac2f8130f2e940'
    OUT_DIR = osp.expanduser('~/dw_pose_test')
    pkl_path = osp.join(ISL_ROOT, 'dw_pose', f'{VIDEO_HASH}.pkl')
    st_path = osp.join(OUT_DIR, f'{VIDEO_HASH}.safetensors')

    meta = pd.read_csv(osp.join(ISL_ROOT, 'metadata', 'video_metadata.csv'))
    row = meta[meta.hash == VIDEO_HASH]
    assert len(row) == 1, f'(__main__): {len(row)} metadata rows for {VIDEO_HASH}, expected 1'
    frames = load_legacy_pickle(pkl_path, VIDEO_HASH)
    seq = legacy_to_sequence(frames, VIDEO_HASH, int(row.width.iloc[0]), int(row.height.iloc[0]))
    os.makedirs(OUT_DIR, exist_ok=True)
    write_dw_pose(seq, st_path)
    check_dw_pose(pkl_path, st_path)
    print(f'(__main__): {seq!r}; pkl {osp.getsize(pkl_path) / 1e6:.1f} MB -> safetensors {osp.getsize(st_path) / 1e6:.1f} MB; all checks passed')
