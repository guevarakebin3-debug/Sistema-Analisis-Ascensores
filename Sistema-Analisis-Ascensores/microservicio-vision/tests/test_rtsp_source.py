import os
import unittest
from unittest.mock import patch

import numpy as np

from cameras.rtsp_source import RTSPSource


class FakeCapture:
    def __init__(self, opened=True, received=True, frame=None):
        self.opened = opened
        self.received = received
        self.frame = frame if frame is not None else np.zeros((2, 3, 3))
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.received, self.frame

    def release(self):
        self.released = True


class RTSPSourceTests(unittest.TestCase):
    def test_rejects_non_rtsp_url(self):
        with self.assertRaises(ValueError):
            RTSPSource("https://camera.example/stream")

    def test_requires_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CAMERA_URL"):
                RTSPSource.from_environment("CAMERA_URL")

    @patch("cameras.rtsp_source.cv2.VideoCapture")
    def test_opens_reads_and_releases_capture(self, video_capture):
        frame = np.ones((2, 3, 3))
        capture = FakeCapture(frame=frame)
        video_capture.return_value = capture

        with RTSPSource("rtsp://camera.example/live") as source:
            self.assertIs(source.read(), frame)

        video_capture.assert_called_once_with("rtsp://camera.example/live")
        self.assertTrue(capture.released)

    @patch("cameras.rtsp_source.cv2.VideoCapture")
    def test_raises_if_stream_cannot_be_opened(self, video_capture):
        capture = FakeCapture(opened=False)
        video_capture.return_value = capture

        with self.assertRaisesRegex(ConnectionError, "No se pudo abrir"):
            RTSPSource("rtsp://camera.example/live").open()

        self.assertTrue(capture.released)

    @patch("cameras.rtsp_source.cv2.VideoCapture")
    def test_raises_if_stream_does_not_return_a_frame(self, video_capture):
        video_capture.return_value = FakeCapture(received=False)
        source = RTSPSource("rtsp://camera.example/live").open()

        with self.assertRaisesRegex(ConnectionError, "no entregó"):
            source.read()

        source.close()


if __name__ == "__main__":
    unittest.main()
