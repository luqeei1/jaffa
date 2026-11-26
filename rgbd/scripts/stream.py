"""Minimal MJPEG-over-HTTP server for viewing annotated frames in a browser (e.g. over SSH).

    server = MjpegServer(port=8080)   # starts serving in a background thread
    server.publish(bgr_image)         # call every frame
    server.close()

Open http://<host>:<port>/ in a browser. Uses only the standard library + cv2.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"
PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marco person detection</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;min-height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream" alt="live stream"></body></html>"""


class MjpegServer:
    def __init__(self, port=8080, host="0.0.0.0", quality=80):
        self.quality = quality
        self._jpeg = None
        self._cond = threading.Condition()
        self._closed = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for detection output
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            jpeg = server._wait_for_frame(last)
                            if server._closed:
                                return
                            if jpeg is None:  # no frame published yet
                                continue
                            last = jpeg
                            self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                             f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return  # browser tab closed
                else:
                    self.send_error(404)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self):
        return self._httpd.server_address[1]

    def urls(self):
        """URLs worth printing: localhost (for SSH/VS Code port forwarding) and this machine's LAN address."""
        urls = [f"http://localhost:{self.port}/"]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
                urls.append(f"http://{s.getsockname()[0]}:{self.port}/")
        except OSError:
            pass
        return urls

    def publish(self, bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._cond.notify_all()

    def _wait_for_frame(self, last):
        """Block until there is a frame newer than `last` (or 5 s pass, which re-sends it as a keep-alive)."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._jpeg is not None and self._jpeg is not last),
                                timeout=5.0)
            return self._jpeg

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
