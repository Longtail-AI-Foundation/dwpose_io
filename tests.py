"""Tests for dwpose_io on synthetic data. Run from this directory:
    python tests.py
"""

import json
import os.path as osp
import pickle
import tempfile
from bisect import bisect_left

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

import dwpose_io as dw
from dwpose_io.check import cast_through_float32, check_view_against_legacy, same_bits
from dwpose_io.readwrite import METADATA_KEY, sequence_to_tensors, serialize_dw_pose
from dwpose_io.sequence import TENSOR_NAMES


def make_sequence(counts, seed=0, video_hash='abc123', width=1920, height=1080):
    """Random DwPoseSequence with counts[f] persons in frame f."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=np.int64)
    n, t = int(counts.sum()), len(counts)
    frame_offsets = np.zeros(t + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(counts)
    seq = dw.DwPoseSequence(
        keypoints=rng.normal(500, 800, (n, dw.NUM_KEYPOINTS, 2)).astype(np.float32),
        keypoint_scores=rng.uniform(-0.05, 1.3, (n, dw.NUM_KEYPOINTS)).astype(np.float32),
        bbox=rng.uniform(0, 2000, (n, 4)).astype(np.float32),
        bbox_score=rng.uniform(0, 1, (n,)).astype(np.float32),
        frame_offsets=frame_offsets,
        timestamps_ms=np.cumsum(rng.uniform(16.0, 42.0, t)).astype(np.float64),
        video_hash=video_hash,
        width=width,
        height=height,
    )
    seq.validate()
    return seq


def make_legacy_frames(counts, seed=0):
    """Frames in the exact pickle layout, with float64 values that are not
    float32-representable so the cast is exercised, and the 'visualization' key."""
    rng = np.random.default_rng(seed)
    frames = []
    for count in counts:
        persons = []
        for _ in range(count):
            persons.append({
                'keypoints': rng.normal(500, 800, (dw.NUM_KEYPOINTS, 2)).tolist(),
                'keypoint_scores': rng.uniform(-0.05, 1.3, dw.NUM_KEYPOINTS).tolist(),
                'bbox': (rng.uniform(0, 2000, 4).tolist(),),
                'bbox_score': np.float32(rng.uniform(0, 1)),
            })
        frames.append({'predictions': [persons], 'timestamp': float(rng.uniform(0, 1e6)), 'visualization': None})
    return frames


def assert_sequences_equal(a, b):
    for name in TENSOR_NAMES:
        same_bits(name, getattr(a, name), getattr(b, name), 'sequence')
    assert (a.video_hash, a.width, a.height) == (b.video_hash, b.width, b.height), f'(assert_sequences_equal): metadata {(a.video_hash, a.width, a.height)} vs {(b.video_hash, b.width, b.height)}'


def expect_assertion(fn, *fragments):
    try:
        fn()
    except AssertionError as e:
        for fragment in fragments:
            assert fragment in str(e), f'(expect_assertion): {fragment!r} not in {str(e)!r}'
        return
    raise AssertionError(f'(expect_assertion): {fn.__name__} did not raise')


def expect_error(fn, error_type):
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f'(expect_error): {fn.__name__} did not raise {error_type.__name__}')


def convert_frames_to_files(frames, tmp):
    pkl_path, st_path = osp.join(tmp, 'x.pkl'), osp.join(tmp, 'x.safetensors')
    with open(pkl_path, 'wb') as fp:
        pickle.dump(frames, fp)
    dw.write_dw_pose(dw.legacy_to_sequence(frames, 'abc123', 1920, 1080), st_path)
    return pkl_path, st_path


def rewrite_with(st_path, mutate):
    """Rewrite a safetensors file after applying mutate(tensors) in place."""
    with safe_open(st_path, framework='numpy') as handle:
        tensors = {k: handle.get_tensor(k).copy() for k in handle.keys()}
        metadata = handle.metadata()
    mutate(tensors)
    save_file(tensors, st_path, metadata=metadata)


def test_roundtrip_file():
    seq = make_sequence([1, 0, 3, 2, 0, 0, 6])
    with tempfile.TemporaryDirectory() as tmp:
        path = osp.join(tmp, 'a.safetensors')
        dw.write_dw_pose(seq, path)
        assert_sequences_equal(dw.read_dw_pose(path), seq)


def test_empty_sequence():
    seq = make_sequence([])
    assert seq.num_frames == 0 and seq.num_persons == 0 and len(seq) == 0 and list(seq) == []
    with tempfile.TemporaryDirectory() as tmp:
        path = osp.join(tmp, 'empty.safetensors')
        dw.write_dw_pose(seq, path)
        assert_sequences_equal(dw.read_dw_pose(path), seq)
    assert dw.legacy_to_sequence([], 'abc123', 1920, 1080).num_frames == 0


def test_write_is_deterministic():
    seq = make_sequence([2, 1, 4])
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = osp.join(tmp, '1.safetensors'), osp.join(tmp, '2.safetensors')
        dw.write_dw_pose(seq, p1)
        dw.write_dw_pose(seq, p2)
        with open(p1, 'rb') as f1, open(p2, 'rb') as f2:
            b1, b2 = f1.read(), f2.read()
        assert b1 == b2, '(test_write_is_deterministic): two writes differ'
        assert serialize_dw_pose(seq) == b1, '(test_write_is_deterministic): serialize differs from file'


def test_legacy_indexing():
    counts = [1, 0, 3, 2, 0, 6]
    frames = make_legacy_frames(counts)
    seq = dw.legacy_to_sequence(frames, 'abc123', 1920, 1080)
    assert len(seq) == 6 and seq.num_persons == 12
    check_view_against_legacy(seq, frames)
    same_bits('keypoints', seq[2]['predictions'][0][1]['keypoints'], cast_through_float32(frames[2]['predictions'][0][1]['keypoints']), 'frame 2 person 1')
    assert seq[-1]['timestamp'] == frames[5]['timestamp']
    assert 'predictions' in seq[0] and 'timestamp' in seq[0] and 'visualization' not in seq[0]
    assert list(seq[0].keys()) == ['predictions', 'timestamp']
    assert len(seq[0]['predictions']) == 1
    assert len(seq[1]['predictions'][0]) == 0 and not seq[1]['predictions'][0]
    assert [type(p['bbox_score']) for p in seq[2]['predictions'][0]] == [np.float32] * 3
    assert len(seq[2]['predictions'][0][1:]) == 2
    sub = seq[2:5]
    assert len(sub) == 3 and sub.num_persons == 5 and sub[0]['timestamp'] == seq[2]['timestamp']
    assert len(seq[4:100]) == 2 and len(seq[3:2]) == 0 and len(seq[-2:]) == 2
    expect_error(lambda: seq[6], IndexError)
    expect_error(lambda: seq[2]['predictions'][0][3], IndexError)
    expect_error(lambda: seq[0]['nope'], KeyError)
    expect_error(lambda: seq[2]['predictions'][0][0]['nope'], KeyError)
    timestamps = [frame['timestamp'] for frame in frames]
    probe = sorted(timestamps)[3]
    assert bisect_left(seq, probe, key=lambda frame: frame['timestamp']) == bisect_left(frames, probe, key=lambda frame: frame['timestamp'])


def test_in_place_assignment():
    seq = dw.legacy_to_sequence(make_legacy_frames([1, 0, 3, 2]), 'abc123', 1920, 1080)
    person = seq[2]['predictions'][0][1]
    new_keypoints = np.random.default_rng(7).normal(0, 1, (dw.NUM_KEYPOINTS, 2)).tolist()
    person['keypoints'] = new_keypoints
    same_bits('keypoints', seq.keypoints[2], cast_through_float32(new_keypoints), 'row 2')
    person['bbox'] = ([1.0, 2.0, 3.0, 4.0],)
    same_bits('bbox', seq.bbox[2], np.array([1, 2, 3, 4], dtype=np.float32), 'row 2')
    person['bbox_score'] = 0.25
    assert seq.bbox_score[2] == np.float32(0.25)
    seq[0]['timestamp'] = 5.0
    assert seq.timestamps_ms[0] == 5.0
    sub = seq[2:4]
    sub[0]['predictions'][0][0]['keypoint_scores'] = np.zeros(dw.NUM_KEYPOINTS)
    assert not seq.keypoint_scores[1].any(), '(test_in_place_assignment): write through a slice did not reach the parent'
    expect_assertion(lambda: person.__setitem__('bbox', [1.0, 2.0, 3.0, 4.0]), 'bbox', '1-tuple')
    expect_assertion(lambda: person.__setitem__('keypoints', np.zeros((5, 2))), 'keypoints shape')
    expect_assertion(lambda: person.__setitem__('nope', 1), 'unknown key')
    expect_assertion(lambda: seq[0].__setitem__('timestamp', 3), 'expected float')
    expect_assertion(lambda: seq[0].__setitem__('predictions', []), 'only timestamp')


def test_delete_person():
    seq = dw.legacy_to_sequence(make_legacy_frames([1, 0, 3, 2]), 'abc123', 1920, 1080)
    original_scores = seq.bbox_score.copy()
    del seq[2]['predictions'][0][1]
    assert seq.num_persons == 5 and seq.frame_offsets.tolist() == [0, 1, 1, 3, 5]
    same_bits('bbox_score', seq.bbox_score, np.delete(original_scores, 2), 'after one delete')
    for k in sorted([1, 0], reverse=True):
        del seq[2]['predictions'][0][k]
    assert seq.num_persons == 3 and seq.frame_offsets.tolist() == [0, 1, 1, 1, 3] and len(seq[2]['predictions'][0]) == 0
    same_bits('bbox_score', seq.bbox_score, original_scores[[0, 4, 5]], 'after emptying frame 2')
    sub = seq[3:4]
    del sub[0]['predictions'][0][0]
    assert sub.num_persons == 1 and seq.num_persons == 3, '(test_delete_person): delete through a slice must not touch the parent'
    expect_error(lambda: seq[3]['predictions'][0].__delitem__(2), IndexError)
    with tempfile.TemporaryDirectory() as tmp:
        path = osp.join(tmp, 'a.safetensors')
        dw.write_dw_pose(seq, path)
        assert_sequences_equal(dw.read_dw_pose(path), seq)


def test_check_dw_pose_passes_on_conversion():
    frames = make_legacy_frames([1, 0, 3, 2, 0, 6], seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        assert dw.check_dw_pose(pkl_path, st_path).num_persons == 12


def test_check_dw_pose_detects_corruption():
    frames = make_legacy_frames([1, 0, 3, 2], seed=4)
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        rewrite_with(st_path, lambda t: t['keypoints'].__setitem__((2, 7, 1), t['keypoints'][2, 7, 1] + 1.0))
        expect_assertion(lambda: dw.check_dw_pose(pkl_path, st_path), 'frame 2 person 1', 'keypoints[15]')

        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        rewrite_with(st_path, lambda t: t['timestamps_ms'].__setitem__(3, 0.0))
        expect_assertion(lambda: dw.check_dw_pose(pkl_path, st_path), 'frame 3', 'timestamp')

        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        rewrite_with(st_path, lambda t: t['bbox_score'].__setitem__(5, np.float32(0.5)))
        expect_assertion(lambda: dw.check_dw_pose(pkl_path, st_path), 'frame 3 person 1', 'bbox_score')

        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        rewrite_with(st_path, lambda t: t['frame_offsets'].__setitem__(2, 2))
        expect_assertion(lambda: dw.check_dw_pose(pkl_path, st_path), 'frame 1', 'persons')

        pkl_path, st_path = convert_frames_to_files(frames, tmp)
        with open(pkl_path, 'wb') as fp:
            pickle.dump(frames[:-1], fp)
        expect_assertion(lambda: dw.check_dw_pose(pkl_path, st_path), 'frames')


def test_metadata_is_checked_on_read():
    seq = make_sequence([1, 2])
    with tempfile.TemporaryDirectory() as tmp:
        path = osp.join(tmp, 'a.safetensors')
        dw.write_dw_pose(seq, path)
        with safe_open(path, framework='numpy') as handle:
            metadata = handle.metadata()
        expected = {'coord_space': 'display', 'format_version': '1', 'height': 1080, 'keypoint_layout': 'coco_wholebody_133', 'video_hash': 'abc123', 'width': 1920}
        assert metadata == {METADATA_KEY: json.dumps(expected, sort_keys=True)}, f'(test_metadata_is_checked_on_read): {metadata}'
        tensors, _ = sequence_to_tensors(seq)
        save_file(tensors, path, metadata={METADATA_KEY: json.dumps({**expected, 'format_version': '0'})})
        expect_assertion(lambda: dw.read_dw_pose(path), 'format_version')
        save_file(tensors, path, metadata={METADATA_KEY: json.dumps({**expected, 'coord_space': 'encoded'})})
        expect_assertion(lambda: dw.read_dw_pose(path), 'coord_space')
        save_file(tensors, path)
        expect_assertion(lambda: dw.read_dw_pose(path), 'metadata')
        save_file(tensors, path, metadata={METADATA_KEY: json.dumps(expected), 'extra': 'x'})
        expect_assertion(lambda: dw.read_dw_pose(path), 'metadata keys')
        save_file({k: v for k, v in tensors.items() if k != 'bbox'}, path, metadata={METADATA_KEY: json.dumps(expected)})
        expect_assertion(lambda: dw.read_dw_pose(path), 'tensors')


def test_legacy_to_sequence_rejects_malformed():
    def with_person(mutate):
        frames = make_legacy_frames([2])
        mutate(frames[0]['predictions'][0][1])
        return lambda: dw.legacy_to_sequence(frames, 'abc123', 1920, 1080)

    bad_preds = make_legacy_frames([1])
    bad_preds[0]['predictions'] = [bad_preds[0]['predictions'][0], []]
    expect_assertion(lambda: dw.legacy_to_sequence(bad_preds, 'abc123', 1920, 1080), 'predictions', 'length 2')
    expect_assertion(with_person(lambda p: p['keypoints'].pop()), 'keypoints shape')
    expect_assertion(with_person(lambda p: p.__setitem__('bbox', p['bbox'][0])), 'bbox', '1-tuple')
    expect_assertion(with_person(lambda p: p.__setitem__('bbox_score', float(p['bbox_score']))), 'bbox_score', 'numpy.float32')
    expect_assertion(with_person(lambda p: p.__setitem__('extra', 1)), 'person keys')
    bad_ts = make_legacy_frames([1])
    bad_ts[0]['timestamp'] = 40
    expect_assertion(lambda: dw.legacy_to_sequence(bad_ts, 'abc123', 1920, 1080), 'timestamp', 'expected float')
    expect_assertion(lambda: dw.legacy_to_sequence(make_legacy_frames([2]), 'abc123', np.int64(1920), 1080), 'width')


def test_validate_rejects_bad_arrays():
    seq = make_sequence([1, 2, 1])
    seq.frame_offsets = np.array([0, 1, 3, 5], dtype=np.int64)
    expect_assertion(seq.validate, 'frame_offsets[-1]')
    seq.frame_offsets = np.array([0, 2, 1, 4], dtype=np.int64)
    expect_assertion(seq.validate, 'non-decreasing')
    seq.frame_offsets = np.array([1, 2, 3, 4], dtype=np.int64)
    expect_assertion(seq.validate, 'frame_offsets[0]')
    seq = make_sequence([1, 2, 1])
    seq.keypoints = seq.keypoints.astype(np.float64)
    expect_assertion(seq.validate, 'keypoints dtype')
    seq = make_sequence([1, 2, 1])
    seq.keypoints = np.asfortranarray(seq.keypoints)
    expect_assertion(seq.validate, 'C-contiguous')


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print(f'(tests): {name} ok')
    print(f'(tests): {len(tests)} passed')
