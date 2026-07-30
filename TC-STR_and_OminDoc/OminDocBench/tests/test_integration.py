import json
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from omnidocbench.core import Settings, infer_page, parse_processor


class Handler(BaseHTTPRequestHandler):
    payload = None
    done_value = True

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        Handler.payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "response": "# parsed",
                "done": Handler.done_value,
                "done_reason": "stop",
                "eval_count": 2,
                "eval_duration": 1000000000,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        pass


class IntegrationTests(unittest.TestCase):
    def test_processor_parser_is_strict_and_canonical(self):
        self.assertEqual(
            parse_processor("NAME ID SIZE PROCESSOR\nexact:tag abc 1 GB 100% GPU", "exact:tag"),
            "100% GPU",
        )
        self.assertNotEqual(
            parse_processor("exact:tag abc 1 GB 90% GPU / 10% CPU", "exact:tag"),
            "100% GPU",
        )

    def test_ollama_request_has_exact_common_options_and_think_false(self):
        Handler.done_value = True
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        settings = Settings.load()
        settings.raw["inference"]["endpoint"] = f"http://127.0.0.1:{server.server_port}"
        settings.raw["inference"]["timeout_seconds"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            image = pathlib.Path(tmp) / "x.png"
            image.write_bytes(b"fake")
            response, errors = infer_page(settings, {"ollama_tag": "exact:tag"}, image)
        server.shutdown()
        server.server_close()
        self.assertFalse(errors)
        self.assertEqual(response["response"], "# parsed")
        self.assertFalse(Handler.payload["stream"])
        self.assertFalse(Handler.payload["think"])
        self.assertEqual(Handler.payload["options"], settings.raw["inference"]["options"])
        self.assertEqual(Handler.payload["model"], "exact:tag")
        self.assertEqual(len(Handler.payload["images"]), 1)

    def test_incomplete_non_stream_response_is_rejected(self):
        Handler.done_value = False
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        settings = Settings.load()
        settings.raw["inference"]["endpoint"] = f"http://127.0.0.1:{server.server_port}"
        settings.raw["inference"]["timeout_seconds"] = 2
        settings.raw["inference"]["max_attempts"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            image = pathlib.Path(tmp) / "x.png"
            image.write_bytes(b"fake")
            response, errors = infer_page(settings, {"ollama_tag": "exact:tag"}, image)
        server.shutdown()
        server.server_close()
        Handler.done_value = True
        self.assertIsNone(response)
        self.assertEqual(len(errors), 1)
        self.assertIn("incomplete non-stream response", errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
