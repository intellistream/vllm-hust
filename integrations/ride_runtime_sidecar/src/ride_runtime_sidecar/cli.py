# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio

from ride_runtime_sidecar.config import load_config
from ride_runtime_sidecar.server import RemoteControlSidecar


async def _run(config_path: str) -> None:
    sidecar = RemoteControlSidecar(load_config(config_path))
    await sidecar.start()
    try:
        await sidecar.serve_forever()
    finally:
        await sidecar.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.config))
