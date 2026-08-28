# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded transparent framing shared with the local VLCB1 host."""

import asyncio
import json
import struct

REQUEST_HEADER = struct.Struct("!5sHI")
RESPONSE_HEADER = struct.Struct("!I")
MAGIC = b"VLCB1"
MAX_SIGNATURE_BYTES = 256
MAX_PAYLOAD_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 16 * 1024


class SidecarProtocolError(ValueError):
    pass


async def read_request_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(REQUEST_HEADER.size)
    magic, signature_length, payload_length = REQUEST_HEADER.unpack(header)
    if magic != MAGIC:
        raise SidecarProtocolError("invalid request magic")
    if not 0 < signature_length <= MAX_SIGNATURE_BYTES:
        raise SidecarProtocolError("invalid signature length")
    if not 0 < payload_length <= MAX_PAYLOAD_BYTES:
        raise SidecarProtocolError("invalid payload length")
    body = await reader.readexactly(signature_length + payload_length)
    return header + body


async def read_response_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(RESPONSE_HEADER.size)
    (length,) = RESPONSE_HEADER.unpack(header)
    if not 0 < length <= MAX_RESPONSE_BYTES:
        raise SidecarProtocolError("invalid response length")
    return header + await reader.readexactly(length)


def error_frame(code: str) -> bytes:
    body = json.dumps(
        {"kind": "error", "code": code}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return RESPONSE_HEADER.pack(len(body)) + body
