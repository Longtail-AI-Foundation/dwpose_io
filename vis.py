"""Side-by-side keypoint overlays from the pickle and the safetensors copy of the
same dw_pose file. The two sources differ only in the loader called; every
drawing function is shared. Writes /tmp/dw_pose_vis/<category>/<hash>_f<frame>.png
and an index.html, 20 random hashes per source category."""

import glob
import os
import os.path as osp
import pickle
import random
import subprocess

import cv2
import numpy as np
import pandas as pd

from dwpose_io import load_legacy_pickle, read_dw_pose

ISL_ROOT = '/home/ubuntu/general-purpose/ISL'
PKL_DIR = osp.join(ISL_ROOT, 'dw_pose')
ST_DIR = osp.expanduser('~/dw_pose_test')
OUT_DIR = '/tmp/dw_pose_vis'
VIDEO_HASHES = osp.join(ISL_ROOT, 'metadata', 'video_hashes.csv')
PER_CATEGORY = 20
VIDEOS_PER_CATEGORY = 2
MAX_VIDEO_FRAMES = 300
SEED = 0
SCORE_THRESHOLD = 0.3


def load_pkl(video_hash):
    return load_legacy_pickle(osp.join(PKL_DIR, f'{video_hash}.pkl'), video_hash)


def load_st(video_hash):
    return read_dw_pose(osp.join(ST_DIR, f'{video_hash}.safetensors'))


def category_of(path):
    """Second path component, except that the alphabet_IMG_4651 clips filed
    under islrtc count as their own category 'alphabet', and paths of the form
    incoming/<set>/videos/... are the category <set>."""
    parts = path.split('/')
    if parts[0] == 'incoming':
        assert parts[2] == 'videos', f'(category_of): path {path} is not incoming/<set>/videos/...'
        return parts[1]
    assert parts[0] == 'videos', f'(category_of): path {path} does not start with videos/'
    if parts[1] == 'islrtc' and parts[2] == 'alphabet_IMG_4651':
        return 'alphabet'
    return parts[1]


def sample_hashes():
    """{category: [(hash, video_path), ...]} with up to PER_CATEGORY random
    hashes per category, drawn only from hashes that have both a pickle and a
    safetensors file."""
    table = pd.read_csv(VIDEO_HASHES)
    have_pkl = {f[:-4] for f in os.listdir(PKL_DIR) if f.endswith('.pkl')}
    have_st = {f[:-12] for f in os.listdir(ST_DIR) if f.endswith('.safetensors')}
    by_category = {}
    for path, video_hash in zip(table['path'], table['hash']):
        if video_hash in have_pkl and video_hash in have_st:
            by_category.setdefault(category_of(path), {})[video_hash] = osp.join(ISL_ROOT, path)
    rng = random.Random(SEED)
    sampled = {}
    for category, paths in sorted(by_category.items()):
        chosen = rng.sample(sorted(paths), min(PER_CATEGORY, len(paths)))
        sampled[category] = [(h, paths[h]) for h in chosen]
    return sampled


def draw_frame(image, frame):
    """Draw every person of one frame onto image, reading the frame with the
    pickle indexing only: keypoints green above SCORE_THRESHOLD and red below,
    the bbox in yellow."""
    for person in frame['predictions'][0]:
        x1, y1, x2, y2 = person['bbox'][0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
        for (x, y), score in zip(person['keypoints'], person['keypoint_scores']):
            color = (0, 255, 0) if score > SCORE_THRESHOLD else (0, 0, 255)
            cv2.circle(image, (int(round(x)), int(round(y))), 2, color, -1)
    return image


def label(image, text):
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def grab_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'(grab_frame): cannot open {video_path}'
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, image = cap.read()
    cap.release()
    assert success, f'(grab_frame): cannot read frame {frame_index} of {video_path}'
    return image


def panel(image, frame_pkl, frame_st, video_hash, f):
    """The same video frame overlaid twice, pickle on the left and safetensors
    on the right, through the same draw_frame."""
    left = label(draw_frame(image.copy(), frame_pkl), f'pkl {video_hash[:8]} f{f}')
    right = label(draw_frame(image.copy(), frame_st), f'safetensors {video_hash[:8]} f{f}')
    return np.concatenate([left, right], axis=1)


def write_video(video_path, frames_pkl, frames_st, video_hash, out_path):
    """The panel for the first min(len(frames_pkl), MAX_VIDEO_FRAMES) frames of
    the video, piped into ffmpeg as H.264 so a browser can play it."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'(write_video): cannot open {video_path}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert h % 2 == 0, f'(write_video): {video_path} height {h} is odd, H.264 needs even'
    count = min(len(frames_pkl), MAX_VIDEO_FRAMES)
    assert count > 0, f'(write_video): {video_hash} has no frames'
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{2 * w}x{h}', '-r', f'{fps}', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', out_path]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for f in range(count):
        success, image = cap.read()
        assert success, f'(write_video): cannot read frame {f} of {video_path}'
        proc.stdin.write(panel(image, frames_pkl[f], frames_st[f], video_hash, f).tobytes())
    cap.release()
    proc.stdin.close()
    assert proc.wait() == 0, f'(write_video): ffmpeg failed on {out_path}'


def write_index(rows):
    """index.html with one section per category and one panel per row; rows is
    [(category, png path relative to OUT_DIR)] in category order."""
    html = ['<html><body style="background:#111;color:#eee;font-family:sans-serif">']
    current = None
    for category, rel in rows:
        if category != current:
            html.append(f'<h2>{category}</h2>')
            current = category
        tag = 'video controls' if rel.endswith('.mp4') else 'img'
        html.append(f'<div>{rel}</div><{tag} src="{rel}" style="max-width:100%"></{tag.split()[0]}><br>')
    html.append('</body></html>')
    with open(osp.join(OUT_DIR, 'index.html'), 'w') as fp:
        fp.write('\n'.join(html))


def clear_outputs():
    """Remove every png and mp4 from an earlier run so the folders hold only
    what the new index references."""
    for name in glob.glob(osp.join(OUT_DIR, '*', '*.png')) + glob.glob(osp.join(OUT_DIR, '*', '*.mp4')):
        os.remove(name)


def main():
    clear_outputs()
    rng = random.Random(SEED)
    rows = []
    for category, items in sample_hashes().items():
        os.makedirs(osp.join(OUT_DIR, category), exist_ok=True)
        for j, (video_hash, video_path) in enumerate(items):
            frames_pkl = load_pkl(video_hash)
            frames_st = load_st(video_hash)
            assert len(frames_pkl) == len(frames_st), f'(main): {video_hash} has {len(frames_pkl)} pkl frames and {len(frames_st)} safetensors frames'
            if len(frames_pkl) == 0:
                print(f'(main): {category} {video_hash} is empty, skipped')
                continue
            if j < VIDEOS_PER_CATEGORY:
                rel_video = osp.join(category, f'{video_hash}.mp4')
                write_video(video_path, frames_pkl, frames_st, video_hash, osp.join(OUT_DIR, rel_video))
                rows.append((category, rel_video))
                print(rel_video)
            f = rng.randrange(len(frames_pkl))
            rel = osp.join(category, f'{video_hash}_f{f}.png')
            image = grab_frame(video_path, f)
            cv2.imwrite(osp.join(OUT_DIR, rel), panel(image, frames_pkl[f], frames_st[f], video_hash, f))
            rows.append((category, rel))
            print(rel)
    write_index(rows)


if __name__ == '__main__':
    main()
