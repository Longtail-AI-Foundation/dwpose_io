"""Conversion from the dw_pose pickle layout. A pickle is a list with one dict
per frame; each dict has 'predictions' holding a single list of person dicts,
'timestamp' as a float in ms, and an always-None 'visualization' that is
ignored. A person dict has 'keypoints' as a list of 133 [x, y] float lists,
'keypoint_scores' as a list of 133 floats, 'bbox' as a 1-tuple holding one
[x1, y1, x2, y2] float list, and 'bbox_score' as a numpy float32."""

import csv
import functools
import pickle

import numpy as np

from .sequence import NUM_KEYPOINTS, DwPoseSequence

LEGACY_PERSON_KEYS = {'keypoints', 'keypoint_scores', 'bbox', 'bbox_score'}
VIDEO_HASHES_CSV = '/home/ubuntu/general-purpose/ISL/metadata/video_hashes.csv'
ALPHABET_PATH_PREFIX = 'videos/islrtc/alphabet_IMG_4651/'


def legacy_frame_person_count(frame):
    """Number of persons in one pickle frame, after checking that 'predictions' is
    a list holding exactly one list."""
    assert isinstance(frame, dict), f'(legacy_frame_person_count): frame is {type(frame).__name__}, expected dict'
    preds = frame['predictions']
    assert type(preds) is list and len(preds) == 1, f'(legacy_frame_person_count): predictions is {type(preds).__name__} of length {len(preds)}, expected list of length 1'
    assert type(preds[0]) is list, f'(legacy_frame_person_count): predictions[0] is {type(preds[0]).__name__}, expected list'
    return len(preds[0])


def legacy_frame_timestamp(frame):
    ts = frame['timestamp']
    assert type(ts) is float, f'(legacy_frame_timestamp): timestamp is {type(ts).__name__}, expected float'
    return ts


def legacy_person_arrays(person):
    """(keypoints [133, 2], keypoint_scores [133], bbox [4], bbox_score) as float32
    from one pickle person dict. Only the first three are cast; bbox_score is
    already float32 in the pickles."""
    assert isinstance(person, dict), f'(legacy_person_arrays): person is {type(person).__name__}, expected dict'
    assert set(person.keys()) == LEGACY_PERSON_KEYS, f'(legacy_person_arrays): person keys {sorted(person.keys())}, expected {sorted(LEGACY_PERSON_KEYS)}'
    keypoints = np.asarray(person['keypoints'], dtype=np.float64)
    assert keypoints.shape == (NUM_KEYPOINTS, 2), f'(legacy_person_arrays): keypoints shape {keypoints.shape}, expected {(NUM_KEYPOINTS, 2)}'
    scores = np.asarray(person['keypoint_scores'], dtype=np.float64)
    assert scores.shape == (NUM_KEYPOINTS,), f'(legacy_person_arrays): keypoint_scores shape {scores.shape}, expected {(NUM_KEYPOINTS,)}'
    box = person['bbox']
    assert type(box) is tuple and len(box) == 1, f'(legacy_person_arrays): bbox is {type(box).__name__} of length {len(box)}, expected 1-tuple'
    box = np.asarray(box[0], dtype=np.float64)
    assert box.shape == (4,), f'(legacy_person_arrays): bbox[0] shape {box.shape}, expected (4,)'
    score = person['bbox_score']
    assert type(score) is np.float32, f'(legacy_person_arrays): bbox_score is {type(score).__name__}, expected numpy.float32'
    return keypoints.astype(np.float32), scores.astype(np.float32), box.astype(np.float32), score


@functools.lru_cache(maxsize=1)
def alphabet_hashes():
    """The 26 video hashes whose path in video_hashes.csv starts with
    ALPHABET_PATH_PREFIX. Their pickles were written by
    scripts/backfill_alphabet_dw_pose.py, which stores 'bbox' as a flat
    [x1, y1, x2, y2] list instead of the 1-tuple every other pickle has."""
    with open(VIDEO_HASHES_CSV) as fp:
        hashes = {row['hash'] for row in csv.DictReader(fp) if row['path'].startswith(ALPHABET_PATH_PREFIX)}
    assert len(hashes) == 26, f'(alphabet_hashes): found {len(hashes)} alphabet hashes, expected 26'
    return hashes


def load_legacy_pickle(path, video_hash):
    """pickle.load of a dw_pose pickle. For the alphabet hashes only, each
    person's flat bbox list is wrapped into the 1-tuple layout the rest of this
    module expects; every other pickle is returned untouched."""
    with open(path, 'rb') as fp:
        frames = pickle.load(fp)
    if video_hash not in alphabet_hashes():
        return frames
    for frame in frames:
        for person in frame['predictions'][0]:
            box = person['bbox']
            assert type(box) is list and len(box) == 4, f'(load_legacy_pickle): alphabet {video_hash} bbox is {type(box).__name__} of length {len(box)}, expected flat list of 4'
            person['bbox'] = (box,)
    return frames


def legacy_to_sequence(frames, video_hash, width, height):
    """Build a DwPoseSequence from the frames of a dw_pose pickle. Keypoints,
    scores and boxes are rounded to float32; timestamps stay float64."""
    assert type(frames) is list, f'(legacy_to_sequence): frames is {type(frames).__name__}, expected list'
    counts = np.array([legacy_frame_person_count(frame) for frame in frames], dtype=np.int64)
    frame_offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(counts)
    n = int(frame_offsets[-1])
    keypoints = np.empty((n, NUM_KEYPOINTS, 2), dtype=np.float32)
    keypoint_scores = np.empty((n, NUM_KEYPOINTS), dtype=np.float32)
    bbox = np.empty((n, 4), dtype=np.float32)
    bbox_score = np.empty((n,), dtype=np.float32)
    timestamps_ms = np.empty((len(frames),), dtype=np.float64)
    i = 0
    for f, frame in enumerate(frames):
        timestamps_ms[f] = legacy_frame_timestamp(frame)
        for person in frame['predictions'][0]:
            keypoints[i], keypoint_scores[i], bbox[i], bbox_score[i] = legacy_person_arrays(person)
            i += 1
    assert i == n, f'(legacy_to_sequence): filled {i} persons, expected {n}'
    seq = DwPoseSequence(keypoints, keypoint_scores, bbox, bbox_score, frame_offsets, timestamps_ms, video_hash, width, height)
    seq.validate()
    return seq
