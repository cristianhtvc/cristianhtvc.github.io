from __future__ import annotations

import argparse
import cgi
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from content_review import review_markdown
from upload_note import ROOT, load_data, upload_markdown


ADMIN_ROOT = ROOT / "admin"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    return "text/html; charset=utf-8"


class UploadHandler(BaseHTTPRequestHandler):
    server_version = "ORLUpload/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[upload] {self.address_string()} - {format % args}")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/config":
            data = load_data()
            venues = [
                {
                    "id": venue["id"],
                    "title": venue["title"],
                    "years": [year_block["year"] for year_block in venue["years"]],
                }
                for venue in data["offline_rl"]["venues"]
            ]
            self.send_json({"ok": True, "venues": venues})
            return

        route = self.path.split("?", 1)[0]
        if route in {"/", "/admin"}:
            route = "/index.html"
        safe_name = route.lstrip("/")
        if not safe_name or ".." in Path(safe_name).parts:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        target = ADMIN_ROOT / safe_name
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type(target))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in {"/api/review", "/api/upload"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            filename, markdown, replace = self.read_upload()
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        review = review_markdown(markdown)
        if self.path == "/api/review":
            self.send_json({"ok": review.passed, "review": review.to_dict()})
            return

        if not review.passed:
            self.send_json({"ok": False, "review": review.to_dict()}, HTTPStatus.UNPROCESSABLE_ENTITY)
            return

        try:
            result = upload_markdown(markdown, filename, replace=replace, build=True, publish=True)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc), "review": review.to_dict()}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_json({"ok": True, "result": result.to_dict()})

    def read_upload(self) -> tuple[str, str, bool]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("没有收到上传内容。")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("文件过大，请控制在 4 MB 以内。")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(length),
            },
        )
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            raise ValueError("请选择一个 `.md` 文件。")
        filename = Path(file_item.filename).name
        if not filename.lower().endswith(".md"):
            raise ValueError("只能上传 Markdown `.md` 文件。")
        raw = file_item.file.read()
        try:
            markdown = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("文件必须使用 UTF-8 编码。") from exc
        replace = str(form.getfirst("replace", "")).lower() in {"1", "true", "on", "yes"}
        return filename, markdown, replace


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local ORL note upload server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), UploadHandler)
    print(f"ORL 上传后台已启动：http://{args.host}:{args.port}/")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n上传后台已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
