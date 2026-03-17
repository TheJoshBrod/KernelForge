#!/usr/bin/env python3
import argparse
import mimetypes
import posixpath
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND_PREFIXES = (
    "/walker/",
    "/function/",
    "/user/",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/admin",
    "/static/",
)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _is_backend_request(path: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in BACKEND_PREFIXES)


class KernelForgeUiHandler(BaseHTTPRequestHandler):
    api_base_url = ""
    dist_dir = Path(".")

    def do_GET(self) -> None:
        self._handle_request()

    def do_HEAD(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_PUT(self) -> None:
        self._handle_request()

    def do_PATCH(self) -> None:
        self._handle_request()

    def do_DELETE(self) -> None:
        self._handle_request()

    def do_OPTIONS(self) -> None:
        self._handle_request()

    def _handle_request(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        request_path = parsed.path or "/"

        if _is_backend_request(request_path):
            self._proxy_request()
            return

        file_path = self._resolve_static_path(request_path)
        if file_path is None:
            self._serve_index()
            return

        if file_path.is_file():
            self._serve_file(file_path)
            return

        if "." in Path(request_path).name:
            self.send_error(404, "File not found")
            return

        self._serve_index()

    def _resolve_static_path(self, request_path: str) -> Path | None:
        normalized = posixpath.normpath(urllib.parse.unquote(request_path))
        if normalized in ("", "."):
            normalized = "/"

        if normalized == "/":
            return None

        relative = normalized.lstrip("/")
        candidate = (self.dist_dir / relative).resolve()
        try:
            candidate.relative_to(self.dist_dir)
        except ValueError:
            return self.dist_dir / "__missing__"
        return candidate

    def _serve_index(self) -> None:
        self._serve_file((self.dist_dir / "index.html").resolve())

    def _serve_file(self, file_path: Path) -> None:
        try:
            content = file_path.read_bytes()
        except OSError:
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def _proxy_request(self) -> None:
        target_url = f"{self.api_base_url}{self.path}"
        body = None
        length = self.headers.get("Content-Length")
        if length:
            body = self.rfile.read(int(length))

        headers = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "host":
                continue
            headers[key] = value

        request = urllib.request.Request(
            target_url,
            data=body,
            headers=headers,
            method=self.command,
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in HOP_BY_HOP_HEADERS:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            for key, value in error.headers.items():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
        except Exception as error:
            message = f"UI proxy error: {error}\n".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(message)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[ui] {self.address_string()} - {format % args}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel Forge desktop UI server")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--api-base-url", required=True)
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        print(f"Kernel Forge UI server could not find {index_path}", file=sys.stderr)
        return 1

    KernelForgeUiHandler.api_base_url = args.api_base_url.rstrip("/")
    KernelForgeUiHandler.dist_dir = dist_dir

    server = ThreadingHTTPServer((args.bind, args.port), KernelForgeUiHandler)
    print(f"KFORGE_UI_SERVER=http://{args.bind}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
