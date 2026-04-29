"""Forward remote TensorBoard port to local localhost via Paramiko.

Defaults:
- reads cloud host credentials from repo-root .env
- forwards local 16006 -> remote 127.0.0.1:6006
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
from pathlib import Path

import paramiko


def load_env(repo_root: Path) -> dict[str, str]:
    env_path = repo_root / ".env"
    env: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class Handler(socketserver.BaseRequestHandler):
    chain_host = "127.0.0.1"
    chain_port = 6006
    transport: paramiko.Transport | None = None

    def handle(self):
        assert self.transport is not None
        chan = self.transport.open_channel(
            "direct-tcpip",
            (self.chain_host, self.chain_port),
            self.request.getpeername(),
        )
        try:
            while True:
                rlist, _, _ = select.select([self.request, chan], [], [])
                if self.request in rlist:
                    data = self.request.recv(1024)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in rlist:
                    data = chan.recv(1024)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            chan.close()
            self.request.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--local-port", type=int, default=16006)
    parser.add_argument("--remote-port", type=int, default=6006)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    env = load_env(repo_root)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=env["CLOUD_HOST"],
        port=int(env["CLOUD_PORT"]),
        username=env["CLOUD_USER"],
        password=env["CLOUD_PASSWORD"],
        timeout=20,
    )

    transport = client.get_transport()
    assert transport is not None

    class ForwardServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    Handler.chain_port = args.remote_port
    Handler.transport = transport

    server = ForwardServer(("127.0.0.1", args.local_port), Handler)
    print(f"Forwarding http://127.0.0.1:{args.local_port} -> remote 127.0.0.1:{args.remote_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        client.close()


if __name__ == "__main__":
    main()
