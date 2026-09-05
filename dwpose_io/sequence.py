"""In-memory form of one video's DWPose detections, with the indexing of the old
pickles. Persons are concatenated in frame order and frame f owns the slice
frame_offsets[f]:frame_offsets[f + 1] of the person axis, so a frame with no
detections owns an empty slice and the order of persons inside a frame is the
order DWPose emitted them. Coordinates are pixels in the display frame of the
video (what cv2.VideoCapture returns), keypoints follow the 133-point
COCO-WholeBody layout, and timestamps are the cv2 CAP_PROP_POS_MSEC value of
each frame kept as float64 because downstream code bisects on them.
"""

from dataclasses import dataclass

import numpy as np

NUM_KEYPOINTS = 133
KEYPOINT_LAYOUT = 'coco_wholebody_133'
COORD_SPACE = 'display'
FORMAT_VERSION = '1'
PERSON_ARRAY_NAMES = ('keypoints', 'keypoint_scores', 'bbox', 'bbox_score')
TENSOR_NAMES = PERSON_ARRAY_NAMES + ('frame_offsets', 'timestamps_ms')
FRAME_KEYS = ('predictions', 'timestamp')


def normalize_index(index, length, who):
    """Turn a possibly negative integer index into 0..length-1, raising IndexError
    past either end the way a list does."""
    assert isinstance(index, (int, np.integer)) and not isinstance(index, bool), f'({who}): index {index!r} is {type(index).__name__}, expected int'
    index = int(index)
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError(f'({who}): index {index} out of range for length {length}')
    return index


@dataclass(eq=False, repr=False)
class DwPoseSequence:
    """Six arrays plus metadata: keypoints float32 [N, 133, 2], keypoint_scores
    float32 [N, 133], bbox float32 [N, 4] as x1 y1 x2 y2, bbox_score float32 [N],
    frame_offsets int64 [T + 1], timestamps_ms float64 [T], video_hash, and the
    display width and height in pixels. Indexing mirrors the old pickle:
    seq[f]['predictions'][0][k]['keypoints'] is person k of frame f, seq[a:b] is a
    sub-sequence sharing the arrays, and len(seq) is the frame count."""
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    bbox: np.ndarray
    bbox_score: np.ndarray
    frame_offsets: np.ndarray
    timestamps_ms: np.ndarray
    video_hash: str
    width: int
    height: int

    @property
    def num_frames(self):
        return len(self.timestamps_ms)

    @property
    def num_persons(self):
        return len(self.bbox_score)

    def validate(self):
        n, t = self.num_persons, self.num_frames
        expected = {
            'keypoints': (np.float32, (n, NUM_KEYPOINTS, 2)),
            'keypoint_scores': (np.float32, (n, NUM_KEYPOINTS)),
            'bbox': (np.float32, (n, 4)),
            'bbox_score': (np.float32, (n,)),
            'frame_offsets': (np.int64, (t + 1,)),
            'timestamps_ms': (np.float64, (t,)),
        }
        for name, (dtype, shape) in expected.items():
            arr = getattr(self, name)
            assert isinstance(arr, np.ndarray), f'(DwPoseSequence::validate): {name} is {type(arr).__name__}, expected ndarray'
            assert arr.dtype == dtype, f'(DwPoseSequence::validate): {name} dtype {arr.dtype}, expected {np.dtype(dtype)}'
            assert arr.shape == shape, f'(DwPoseSequence::validate): {name} shape {arr.shape}, expected {shape}'
            assert arr.flags.c_contiguous, f'(DwPoseSequence::validate): {name} is not C-contiguous'
        assert self.frame_offsets[0] == 0, f'(DwPoseSequence::validate): frame_offsets[0] is {self.frame_offsets[0]}, expected 0'
        assert self.frame_offsets[-1] == n, f'(DwPoseSequence::validate): frame_offsets[-1] is {self.frame_offsets[-1]}, expected {n} persons'
        assert (np.diff(self.frame_offsets) >= 0).all(), '(DwPoseSequence::validate): frame_offsets must be non-decreasing'
        assert isinstance(self.video_hash, str) and self.video_hash, f'(DwPoseSequence::validate): bad video_hash {self.video_hash!r}'
        assert isinstance(self.width, int) and self.width > 0, f'(DwPoseSequence::validate): bad width {self.width!r}'
        assert isinstance(self.height, int) and self.height > 0, f'(DwPoseSequence::validate): bad height {self.height!r}'

    def __repr__(self):
        return f'DwPoseSequence({self.video_hash}, {self.num_frames} frames, {self.num_persons} persons, {self.width}x{self.height})'

    def person_slice(self, f):
        """(first, last + 1) person index of frame f."""
        return int(self.frame_offsets[f]), int(self.frame_offsets[f + 1])

    def person_slice_range(self, start, stop):
        """(first, last + 1) person index covered by frames start:stop."""
        return int(self.frame_offsets[start]), int(self.frame_offsets[stop])

    def frame_of_person(self, i):
        """Frame owning person row i: the last offset that is <= i."""
        return int(np.searchsorted(self.frame_offsets, i, side='right')) - 1

    def delete_person(self, i):
        """Remove person row i from every person array and lower every offset after
        the owning frame by one. Costs a copy of the person arrays."""
        assert 0 <= i < self.num_persons, f'(DwPoseSequence::delete_person): person {i} out of range for {self.num_persons} persons'
        f = self.frame_of_person(i)
        for name in PERSON_ARRAY_NAMES:
            setattr(self, name, np.delete(getattr(self, name), i, axis=0))
        self.frame_offsets = self.frame_offsets.copy()
        self.frame_offsets[f + 1:] -= 1
        self.validate()

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        return (FrameView(self, f) for f in range(self.num_frames))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.frame_range(index)
        return FrameView(self, normalize_index(index, self.num_frames, 'DwPoseSequence::__getitem__'))

    def frame_range(self, index):
        """Sub-sequence for a slice of frames, clamped like a list slice. The person
        and timestamp arrays are views into this sequence, frame_offsets is rebased."""
        start, stop, step = index.indices(self.num_frames)
        assert step == 1, f'(DwPoseSequence::frame_range): slice step {step}, expected 1'
        stop = max(stop, start)
        p0, p1 = self.person_slice_range(start, stop)
        sub = DwPoseSequence(
            keypoints=self.keypoints[p0:p1],
            keypoint_scores=self.keypoint_scores[p0:p1],
            bbox=self.bbox[p0:p1],
            bbox_score=self.bbox_score[p0:p1],
            frame_offsets=self.frame_offsets[start:stop + 1] - self.frame_offsets[start],
            timestamps_ms=self.timestamps_ms[start:stop],
            video_hash=self.video_hash,
            width=self.width,
            height=self.height,
        )
        sub.validate()
        return sub


class FrameView:
    """Dict-like view of one frame with the two keys the pickles are read by:
    'predictions' is a one-element list holding the frame's persons and
    'timestamp' is the frame time in ms as a float."""

    def __init__(self, seq, f):
        self.seq, self.f = seq, f

    def keys(self):
        return list(FRAME_KEYS)

    def __iter__(self):
        return iter(FRAME_KEYS)

    def __len__(self):
        return len(FRAME_KEYS)

    def __contains__(self, key):
        return key in FRAME_KEYS

    def __getitem__(self, key):
        if key == 'predictions':
            return [PersonListView(self.seq, self.f)]
        if key == 'timestamp':
            return float(self.seq.timestamps_ms[self.f])
        raise KeyError(key)

    def __setitem__(self, key, value):
        assert key == 'timestamp', f'(FrameView::__setitem__): only timestamp can be assigned, got {key!r}'
        assert type(value) is float, f'(FrameView::__setitem__): timestamp is {type(value).__name__}, expected float'
        self.seq.timestamps_ms[self.f] = value

    def __repr__(self):
        return f'FrameView(frame {self.f} of {self.seq!r})'


class PersonListView:
    """List-like view of the persons of one frame: len, indexing, slicing to a
    list, iteration, and del, which removes the person from the sequence."""

    def __init__(self, seq, f):
        self.seq, self.f = seq, f

    def __len__(self):
        p0, p1 = self.seq.person_slice(self.f)
        return p1 - p0

    def __iter__(self):
        return (PersonView(self.seq, i) for i in range(*self.seq.person_slice(self.f)))

    def __getitem__(self, index):
        p0, p1 = self.seq.person_slice(self.f)
        if isinstance(index, slice):
            return [PersonView(self.seq, p0 + k) for k in range(*index.indices(p1 - p0))]
        return PersonView(self.seq, p0 + normalize_index(index, p1 - p0, 'PersonListView::__getitem__'))

    def __delitem__(self, index):
        p0, p1 = self.seq.person_slice(self.f)
        self.seq.delete_person(p0 + normalize_index(index, p1 - p0, 'PersonListView::__delitem__'))

    def __repr__(self):
        return f'PersonListView({len(self)} persons in frame {self.f} of {self.seq!r})'


class PersonView:
    """Dict-like view of one person. 'keypoints' float32 [133, 2] and
    'keypoint_scores' float32 [133] are views into the sequence arrays, 'bbox' is
    a 1-tuple holding a float32 [4] view (x1 y1 x2 y2) as in the pickles, and
    'bbox_score' is a numpy float32. Assignment writes into the arrays, rounding
    the given values to float32."""

    def __init__(self, seq, i):
        self.seq, self.i = seq, i

    def keys(self):
        return list(PERSON_ARRAY_NAMES)

    def __iter__(self):
        return iter(PERSON_ARRAY_NAMES)

    def __len__(self):
        return len(PERSON_ARRAY_NAMES)

    def __contains__(self, key):
        return key in PERSON_ARRAY_NAMES

    def __getitem__(self, key):
        if key == 'bbox':
            return (self.seq.bbox[self.i],)
        if key in PERSON_ARRAY_NAMES:
            return getattr(self.seq, key)[self.i]
        raise KeyError(key)

    def __setitem__(self, key, value):
        assert key in PERSON_ARRAY_NAMES, f'(PersonView::__setitem__): unknown key {key!r}'
        if key == 'bbox':
            assert type(value) is tuple and len(value) == 1, f'(PersonView::__setitem__): bbox is {type(value).__name__} of length {len(value)}, expected 1-tuple'
            value = value[0]
        target = getattr(self.seq, key)
        arr = np.asarray(value, dtype=np.float32)
        assert arr.shape == target.shape[1:], f'(PersonView::__setitem__): {key} shape {arr.shape}, expected {target.shape[1:]}'
        target[self.i] = arr

    def __repr__(self):
        return f'PersonView(person {self.i} of {self.seq!r})'
