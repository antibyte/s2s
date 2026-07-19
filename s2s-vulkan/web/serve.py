#!/usr/bin/env python3
"""
HTTPS static server + WSS→WS reverse proxy for the s2s lab UI.

Browsers only allow getUserMedia (microphone) in a secure context:
  - https://…  or  http://localhost / http://127.0.0.1

From another machine you need HTTPS. This server:
  1) Serves the UI over HTTPS (self-signed cert, auto-generated)
  2) Proxies wss://host:port/ws  →  ws://BACKEND (avoids mixed content)

Usage:
  python serve.py --host 0.0.0.0 --port 9999 --backend 127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import socket
import ssl
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
CERT_DIR = WEB_DIR / ".certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


def local_ips() -> list[str]:
    names = {"localhost", "127.0.0.1", "::1"}
    try:
        hostname = socket.gethostname()
        names.add(hostname)
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            if ":" not in addr:  # ipv4
                names.add(addr)
    except OSError:
        pass
    # Best-effort: all non-loopback IPv4s Windows reports via UDP trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        names.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(names)


def ensure_certs() -> None:
    if CERT_FILE.is_file() and KEY_FILE.is_file():
        return
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("Installing cryptography for self-signed certs…", file=sys.stderr)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "s2s-lab")]
    )
    san: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName("*.local"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    for ip in local_ips():
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            san.append(x509.DNSName(ip))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Wrote self-signed cert → {CERT_FILE}", file=sys.stderr)


def ensure_aiohttp() -> None:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("Installing aiohttp…", file=sys.stderr)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])


async def main_async(host: str, port: int, backend: str) -> None:
    ensure_aiohttp()
    ensure_certs()

    from aiohttp import ClientSession, WSMsgType, web

    backend_ws = f"ws://{backend}"

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_DIR / "index.html")

    async def ws_proxy(request: web.Request) -> web.WebSocketResponse:
        """Browser wss://…/ws  ↔  plain ws://backend"""
        client = web.WebSocketResponse(heartbeat=30.0, max_msg_size=8 * 1024 * 1024)
        await client.prepare(request)

        try:
            async with ClientSession() as session:
                async with session.ws_connect(
                    backend_ws,
                    heartbeat=30.0,
                    max_msg_size=8 * 1024 * 1024,
                ) as upstream:

                    async def client_to_upstream() -> None:
                        async for msg in client:
                            if msg.type == WSMsgType.BINARY:
                                await upstream.send_bytes(msg.data)
                            elif msg.type == WSMsgType.TEXT:
                                await upstream.send_str(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break

                    async def upstream_to_client() -> None:
                        async for msg in upstream:
                            if msg.type == WSMsgType.BINARY:
                                await client.send_bytes(msg.data)
                            elif msg.type == WSMsgType.TEXT:
                                await client.send_str(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break

                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(client_to_upstream()),
                            asyncio.create_task(upstream_to_client()),
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
        except Exception as e:
            print(f"ws proxy error: {e}", file=sys.stderr)
            if not client.closed:
                await client.close(code=1011, message=str(e).encode()[:120])
        return client

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_proxy)
    app.router.add_static("/", WEB_DIR, show_index=False)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port, ssl_context=ssl_ctx)
    await site.start()

    ips = [ip for ip in local_ips() if not ip.startswith("127.")]
    print("", file=sys.stderr)
    print("s2s lab HTTPS ready (self-signed — accept the browser warning once)", file=sys.stderr)
    print(f"  local:   https://127.0.0.1:{port}", file=sys.stderr)
    for ip in ips:
        print(f"  network: https://{ip}:{port}", file=sys.stderr)
    print(f"  WSS:     wss://<host>:{port}/ws  →  {backend_ws}", file=sys.stderr)
    print("  Mic works on remote devices because the page is HTTPS.", file=sys.stderr)
    print("", file=sys.stderr)

    while True:
        await asyncio.sleep(3600)


def main() -> None:
    p = argparse.ArgumentParser(description="HTTPS + WSS proxy for s2s web lab")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument(
        "--backend",
        default="127.0.0.1:8765",
        help="s2s-vulkan websocket host:port (plain WS)",
    )
    args = p.parse_args()
    try:
        asyncio.run(main_async(args.host, args.port, args.backend))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
