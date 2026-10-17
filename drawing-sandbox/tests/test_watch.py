import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from drawing_sandbox import server as server_module


class WatchViewerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.previous_root = server_module._RECORDINGS
        server_module._RECORDINGS = Path(self.folder.name)
        run = server_module._RECORDINGS / "watch-test"
        (run / "steps").mkdir(parents=True)
        (run / "steps/000-drawing.png").write_bytes(b"png-frame")
        (run / "manifest.json").write_text(json.dumps({
            "run_id": "watch-test",
            "composition": {"title": "Watch test"},
            "updated_at": "now",
            "final_step": 0,
            "steps": [{
                "index": 0,
                "label": "Accepted",
                "recorded_at": "now",
                "png": "steps/000-drawing.png",
                "final": True,
                "metrics": {"expanded_shapes": 4},
                "provenance": {"context_revision": 0, "agent_id": "nemotron", "skill": "add_objects"},
            }],
        }))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        server_module._RECORDINGS = self.previous_root
        self.folder.cleanup()

    def test_read_only_viewer_metadata_and_frame(self):
        with urlopen(self.base + "/watch?run=watch-test") as response:
            self.assertIn(b"Shared canvas", response.read())
        with urlopen(self.base + "/api/watch?run=watch-test") as response:
            data = json.load(response)
        self.assertEqual(data["step_count"], 1)
        self.assertEqual(data["steps"][0]["skill"], "add_objects")
        with urlopen(self.base + "/api/watch/frame?run=watch-test&step=0") as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertEqual(response.read(), b"png-frame")

    def test_watch_rejects_path_traversal(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/api/watch?run=../watch-test")
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()


if __name__ == "__main__":
    unittest.main()
