#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import time


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools" / "openai_varprompt_bench"
sys.path.insert(0, str(ROOT))

from tilepo.compiler import compile_plan  # noqa: E402


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_bench_runs_fake_openai_server_and_bootstraps_tilepo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        server = root / "fake_openai_server.py"
        server.write_text(
            textwrap.dedent(
                """
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
                import argparse
                import json

                class Handler(BaseHTTPRequestHandler):
                    def _send(self, payload):
                        encoded = json.dumps(payload).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(encoded)))
                        self.end_headers()
                        self.wfile.write(encoded)

                    def do_GET(self):
                        if self.path == "/v1/models":
                            self._send({"data": [{"id": "tilemem-active"}]})
                        else:
                            self.send_error(404)

                    def do_POST(self):
                        self._send({"choices": [{"text": "ok"}]})

                    def log_message(self, *_):
                        pass

                parser = argparse.ArgumentParser()
                parser.add_argument("--port", type=int, required=True)
                args = parser.parse_args()
                ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
                """
            )
        )
        manifest = compile_plan(ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem", root / "compiled").manifest_path
        out = root / "row.jsonl"
        log = root / "bench.log"
        marker = root / "bootstrap.json"
        port = _free_port()
        proc = subprocess.run(
            [
                sys.executable,
                str(BENCH),
                "--out",
                str(out),
                "--log",
                str(log),
                "--system",
                "C",
                "--run-name",
                "fake",
                "--model",
                "fake",
                "--served-model-name",
                "tilemem-active",
                "--request-count",
                "1",
                "--warmup-request-count",
                "0",
                "--output-tokens",
                "1",
                "--startup-timeout-sec",
                "20",
                "--request-timeout-sec",
                "5",
                "--evidence-level",
                "real",
                "--port",
                str(port),
                "--runtime-dir",
                str(root / "runtime"),
                "--native-tmp-dir",
                str(root / "native"),
                "--plugin-out",
                str(root / "plugin.json"),
                "--extra-env",
                "TILEPO_ENABLE=1",
                "--extra-env",
                f"TILEPO_MANIFEST={manifest}",
                "--extra-env",
                "TILEPO_MODE=serve",
                "--extra-env",
                "TILEPO_BACKEND=cuda,tilelang,kt_fallback",
                "--extra-env",
                f"TILEPO_BOOTSTRAP_MARKER={marker}",
                "--extra-env",
                "TILEPO_RUN_ID=fake-run",
                "--server-command",
                sys.executable,
                str(server),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        row = json.loads(out.read_text())
        assert row["status"] == "success"
        assert row["simulated"] is False
        marker_data = json.loads(marker.read_text())
        assert marker_data["enabled"] is True
        assert marker_data["run_id"] == "fake-run"
        assert marker_data["hot_backend_probe"]["status"] == "success"


def test_warmup_latency_is_not_counted_in_measured_latency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        server = root / "fake_slow_warmup_server.py"
        server.write_text(
            textwrap.dedent(
                """
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
                import argparse
                import json
                import time

                count = 0

                class Handler(BaseHTTPRequestHandler):
                    def _send(self, payload):
                        encoded = json.dumps(payload).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(encoded)))
                        self.end_headers()
                        self.wfile.write(encoded)

                    def do_GET(self):
                        self._send({"data": [{"id": "tilemem-active"}]})

                    def do_POST(self):
                        global count
                        count += 1
                        if count == 1:
                            time.sleep(0.25)
                        self._send({"choices": [{"text": "ok"}]})

                    def log_message(self, *_):
                        pass

                parser = argparse.ArgumentParser()
                parser.add_argument("--port", type=int, required=True)
                args = parser.parse_args()
                ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
                """
            )
        )
        out = root / "row.jsonl"
        log = root / "bench.log"
        port = _free_port()
        started = time.perf_counter()
        proc = subprocess.run(
            [
                sys.executable,
                str(BENCH),
                "--out",
                str(out),
                "--log",
                str(log),
                "--system",
                "B",
                "--run-name",
                "fake-warmup",
                "--model",
                "fake",
                "--served-model-name",
                "tilemem-active",
                "--request-count",
                "1",
                "--warmup-request-count",
                "1",
                "--output-tokens",
                "1",
                "--startup-timeout-sec",
                "20",
                "--request-timeout-sec",
                "5",
                "--evidence-level",
                "real",
                "--port",
                str(port),
                "--server-command",
                sys.executable,
                str(server),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        row = json.loads(out.read_text())
        assert row["warmup_latency_ms"] >= 200.0
        assert row["p95_latency_ms"] < 200.0
        assert (time.perf_counter() - started) * 1000.0 > row["p95_latency_ms"]


def main() -> None:
    test_bench_runs_fake_openai_server_and_bootstraps_tilepo()
    test_warmup_latency_is_not_counted_in_measured_latency()
    print("openai_varprompt_bench tests passed")


if __name__ == "__main__":
    main()
