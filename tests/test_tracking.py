
import unittest

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker
from pipedet.solver.hooks import Detection, Recorder, ImageWriter, MOTWriter
from pipedet.structure.large_image import LargeImage

_root_images = "/home/appuser/data/facing_via_mirror/3840_2160_60fps/minimum/20200124_022_minimum/frames"
_root_output_images = "/home/appuser/src/pipedet/tests/demo_tracking_result/frames_for_tracking"
_root_output_mot = "/home/appuser/src/pipedet/tests/demo_tracking_result/"

class TestTrackingFrameLoader(unittest.TestCase):
    def test_loader(self):
        loader = TrackingFrameLoader(root_images=_root_images)
        loader_iter = loader
        # loader_iter = iter(loader)
        data = next(loader_iter)
        self.assertIsInstance(data, LargeImage)

class TestIoUTracker(unittest.TestCase):
    def test_tracking(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_images)
        hooks = [
            Detection(0.5, 0.5),
            Recorder(),
            ImageWriter(root_output_images=_root_output_images),
            MOTWriter(root_output_mot=_root_output_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track(0, 335)

