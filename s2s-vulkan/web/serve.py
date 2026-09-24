#!/usr/bin/env python3
"""
HTTPS static server + WSS→WS reverse proxy for the s2s lab UI.

Browsers only allow getUserMedia (microphone) in a secure context:
  - https://…  or  http://localhost / http://127.0.0.1

This local helper:
  1) Serves the UI over HTTPS (self-signed cert, auto-generated)
  2) Proxies wss://host:port/ws  →  ws://BACKEND (avoids mixed content)

Usage:
  python serve.py --host 127.0.0.1 --port 9999 --backend 127.0.0.1:8765
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
from urllib.parse import urlsplit

WEB_DIR = Path(__file__).resolve().parent
CERT_DIR = WEB_DIR / ".certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


def valid_local_browser_origin(host: str, origin: str | None) -> bool:
    """Keep the loopback HTTPS helper unavailable through DNS rebinding."""
    try:
        authority = urlsplit(f"https://{host}")
        authority.port
    except ValueError:
        return False
    if (
        authority.hostname not in {"127.0.0.1", "localhost", "::1"}
        or authority.username is not None
        or authority.path
        or authority.query
        or authority.fragment
    ):
        return False
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == host.lower()
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
    )


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


def backend_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def main_async(host: str, port: int, backend: str, backend_ws_path: str) -> None:
    if host == "localhost":
        host = "127.0.0.1"
    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("the standalone Lab server must bind to loopback; use AuraGo /speech-lab/ remotely")
    ensure_aiohttp()
    ensure_certs()

    from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

    if backend_ws_path and not backend_ws_path.startswith("/"):
        raise ValueError("--backend-ws-path must be empty or start with '/'")
    backend_ws = f"ws://{backend}{backend_ws_path}"
    backend_http = f"http://{backend}"
    if ":" in backend and not backend.startswith("["):
        b_host, b_port_s = backend.rsplit(":", 1)
        b_port = int(b_port_s)
    else:
        b_host, b_port = backend, 8765

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_DIR / "index.html")

    async def health(_request: web.Request) -> web.Response:
        up = backend_reachable(b_host, b_port)
        body = {
            "proxy": "ok",
            "backend": backend,
            "backend_up": up,
        }
        return web.json_response(body, status=200 if up else 503)

    async def api_proxy(request: web.Request) -> web.StreamResponse:
        """Proxy the complete Lab API to the same backend as the WebSocket."""
        tail = request.match_info.get("tail", "")
        target = f"{backend_http}/api/{tail}"
        if request.query_string:
            target = f"{target}?{request.query_string}"
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in hop_by_hop and name.lower() not in {"host", "origin"}
        }
        try:
            timeout = ClientTimeout(total=None, sock_connect=5, sock_read=None)
            async with ClientSession(timeout=timeout) as session:
                async with session.request(
                    request.method,
                    target,
                    headers=headers,
                    data=request.content if request.can_read_body else None,
                    allow_redirects=False,
                ) as upstream:
                    response_headers = {
                        name: value
                        for name, value in upstream.headers.items()
                        if name.lower() not in hop_by_hop
                    }
                    response = web.StreamResponse(
                        status=upstream.status,
                        reason=upstream.reason,
                        headers=response_headers,
                    )
                    await response.prepare(request)
                    async for chunk in upstream.content.iter_chunked(64 * 1024):
                        await response.write(chunk)
                    await response.write_eof()
                    return response
        except Exception as error:
            return web.json_response(
                {
                    "error": f"Lab API backend {backend} is unavailable",
                    "detail": str(error),
                },
                status=502,
            )

    async def ws_proxy(request: web.Request) -> web.StreamResponse:
        """Browser wss://…/ws  ↔  plain ws://backend.

        Connect upstream *before* accepting the browser WS so a dead s2s
        returns HTTP 502 instead of open→immediate close thrash in the UI.
        """
        peer = request.remote or "?"
        if not backend_reachable(b_host, b_port):
            print(
                f"ws deny {peer}: backend {backend} not reachable",
                file=sys.stderr,
                flush=True,
            )
            return web.Response(
                status=502,
                text=f"s2s backend {backend} is down — start s2s-vulkan on that port",
            )

        client = web.WebSocketResponse(heartbeat=30.0, max_msg_size=8 * 1024 * 1024)
        await client.prepare(request)
        print(f"ws open  {peer} → {backend_ws}", file=sys.stderr, flush=True)

        try:
            timeout = ClientTimeout(total=None, sock_connect=5, sock_read=None)
            async with ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    backend_ws,
                    heartbeat=30.0,
                    max_msg_size=8 * 1024 * 1024,
                    autoclose=True,
                    autoping=True,
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
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    for t in done:
                        try:
                            exc = t.exception()
                            if exc:
                                print(f"ws relay {peer}: {exc}", file=sys.stderr, flush=True)
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            pass
        except Exception as e:
            print(f"ws proxy error {peer}: {e}", file=sys.stderr, flush=True)
            if not client.closed:
                await client.close(code=1011, message=str(e).encode()[:120])
        finally:
            print(f"ws close {peer}", file=sys.stderr, flush=True)
            if not client.closed:
                await client.close()
        return client

    @web.middleware
    async def local_browser_guard(request: web.Request, handler):
        if not valid_local_browser_origin(request.host, request.headers.get("Origin")):
            return web.Response(status=403, text="Lab origin rejected")
        return await handler(request)

    app = web.Application(middlewares=[local_browser_guard])
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/healthz", health)
    app.router.add_get("/ws", ws_proxy)
    app.router.add_route("*", "/api/{tail:.*}", api_proxy)
    app.router.add_static("/", WEB_DIR, show_index=False)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port, ssl_context=ssl_ctx)
    await site.start()

    print("", file=sys.stderr)
    print("s2s lab local HTTPS ready (self-signed)", file=sys.stderr)
    print(f"  local:   https://127.0.0.1:{port}", file=sys.stderr)
    print(f"  WSS:     wss://<host>:{port}/ws  →  {backend_ws}", file=sys.stderr)
    print(f"  API:     https://<host>:{port}/api/ → {backend_http}/api/", file=sys.stderr)
    print(f"  health:  https://<host>:{port}/health", file=sys.stderr)
    print("  Remote browser access: AuraGo /speech-lab/", file=sys.stderr)
    if not backend_reachable(b_host, b_port):
        print(
            f"  WARNING: backend {backend} is NOT listening right now",
            file=sys.stderr,
        )
    print("", file=sys.stderr)

    while True:
        await asyncio.sleep(3600)


def main() -> None:
    p = argparse.ArgumentParser(description="HTTPS + WSS proxy for s2s web lab")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument(
        "--backend",
        default="127.0.0.1:8765",
        help="s2s Lab backend host:port",
    )
    p.add_argument(
        "--backend-ws-path",
        default="",
        help="upstream WebSocket path, for example /ws for the Docker web gateway",
    )
    args = p.parse_args()
    try:
        asyncio.run(
            main_async(args.host, args.port, args.backend, args.backend_ws_path)
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
