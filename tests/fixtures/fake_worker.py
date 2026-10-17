"""A subprocess HTTP worker implementing llama-server's Unix listener contract."""

import argparse
import json
import os
import socket
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--alias", required=True)
args, _ = parser.parse_known_args()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, body, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(b'{"status":"ok"}')

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        result = {"model": args.alias, "pid": os.getpid(), "input": payload}
        if payload.get("stream"):
            self.reply(
                f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n".encode(),
                "text/event-stream",
            )
        else:
            self.reply(json.dumps(result).encode())


class UnixHTTPServer(HTTPServer):
    address_family = socket.AF_UNIX
    server_bind = socketserver.TCPServer.server_bind


print("offloaded 10/33 layers to GPU", flush=True)
with UnixHTTPServer(args.host, Handler) as server:
    server.serve_forever()
