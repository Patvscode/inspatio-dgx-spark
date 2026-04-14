"""Functional decord mock using pyAV for ARM64 compatibility.

Drop-in replacement for decord.VideoReader. InSpatio-World calls:
  - decord.bridge.set_bridge('torch')
  - vr = decord.VideoReader(uri=path, height=-1, width=-1)
  - frames = vr.get_batch(range(len(vr)))

This mock implements all three using pyAV (FFmpeg-based, works on all platforms).
"""
import av
import torch
import numpy as np


class bridge:
    @staticmethod
    def set_bridge(name):
        pass


class VideoReader:
    """Drop-in replacement for decord.VideoReader using pyAV."""

    def __init__(self, uri, height=-1, width=-1, **kwargs):
        self.uri = uri
        self.frames = []
        container = av.open(str(uri))
        for frame in container.decode(video=0):
            self.frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        self._len = len(self.frames)

    def __len__(self):
        return self._len

    def get_batch(self, indices):
        batch = [self.frames[i] for i in indices]
        return torch.from_numpy(np.stack(batch))


class AVVideoReader(VideoReader):
    pass
