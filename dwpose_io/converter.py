import os
import os.path as osp
import pickle

import numpy as np

from .legacy import legacy_frame_person_count, legacy_frame_timestamp, legacy_person_arrays, legacy_to_sequence, load_legacy_pickle
from .readwrite import read_dw_pose, serialize_dw_pose, write_dw_pose
from .sequence import NUM_KEYPOINTS
from .check import check_dw_pose
from tqdm import tqdm
import multiprocessing as mpr

BAD_POSE_PICKLES = {
    '8c9903ac10b32c65b2c961d38fef9605',  # qRBFFtMmprI.webm, 4h, 19.8 GB pickle truncated
}

def pmap(function, items, chunksize=None) :
    """ parallel mapper using Pool with progress bar """
    cpu_count = 30
    if chunksize is None :
        chunksize = len(items) // (cpu_count * 5)
    chunksize = max(1, chunksize)
    with mpr.Pool(cpu_count) as p :
        mapper = p.imap(function, items, chunksize=chunksize)
        return list(tqdm(mapper, total=len(items)))

def fn (data) : 
    ISL_ROOT = '/home/ubuntu/general-purpose/ISL'
    OUT_DIR = osp.expanduser('~/dw_pose_test')
    bn_pkl_path = data
    pkl_path = osp.join(ISL_ROOT, 'dw_pose', bn_pkl_path)
    st_path = osp.join(OUT_DIR, bn_pkl_path.replace('pkl', 'safetensors')) 
    if osp.exists(st_path) : 
        return
    meta = pd.read_csv(osp.join(ISL_ROOT, 'metadata', 'video_metadata.csv'))
    video_hash = bn_pkl_path.split('.')[0]
    if video_hash in BAD_POSE_PICKLES :
        return
    row = meta[meta.hash == video_hash]
    if len(row) == 0 : 
        return
    assert len(row) == 1, f'(__main__): {len(row)} metadata rows for {video_hash}, expected 1'
    frames = load_legacy_pickle(pkl_path, video_hash)
    seq = legacy_to_sequence(frames, video_hash, int(row.width.iloc[0]), int(row.height.iloc[0]))
    os.makedirs(OUT_DIR, exist_ok=True)
    write_dw_pose(seq, st_path)
    check_dw_pose(pkl_path, st_path)
    print(f'(__main__): {seq!r}; pkl {osp.getsize(pkl_path) / 1e6:.1f} MB -> safetensors {osp.getsize(st_path) / 1e6:.1f} MB; all checks passed')

if __name__ == '__main__':
    import pandas as pd
    ISL_ROOT = '/home/ubuntu/general-purpose/ISL'
    OUT_DIR = osp.expanduser('~/dw_pose_test')
    files = os.listdir(osp.join(ISL_ROOT, 'dw_pose'))
    files = [f for f in files if not osp.exists(osp.join(OUT_DIR, f.replace('pkl', 'safetensors')))]
    print(f'(__main__): {len(files)} pkl files without safetensors output')
    list(pmap(fn, files))
