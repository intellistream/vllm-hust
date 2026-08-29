# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect and validate installed vLLM-HUST extension Bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.plugins.installed import (
    InstalledExtensionBundle,
    discover_installed_extension_bundles,
)
from vllm.plugins.startup import (
    parse_component_permission_allowlist,
    resolve_extension_startup,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _bundle_payload(bundle: InstalledExtensionBundle) -> dict[str, Any]:
    descriptor = bundle.descriptor
    return {
        "bundle_id": descriptor.bundle_id,
        "bundle_version": descriptor.bundle_version,
        "host_api_range": descriptor.host_api_range,
        "distribution": bundle.distribution_name,
        "distribution_version": bundle.distribution_version,
        "manifest_path": str(bundle.manifest_path),
        "components": [
            {
                "component_id": component.component_id,
                "contracts": [contract.value for contract in component.contracts],
                "execution_planes": [
                    plane.value for plane in component.execution_planes
                ],
                "isolation": component.isolation.value,
                "permissions": [
                    permission.value for permission in component.permissions
                ],
            }
            for component in descriptor.components
        ],
    }


def _one_installed_bundle(bundle_id: str) -> InstalledExtensionBundle:
    bundles = discover_installed_extension_bundles((bundle_id,))
    if not bundles:
        raise ValueError(f"installed extension Bundle {bundle_id!r} was not found")
    return bundles[0]


class PluginSubcommand(CLISubcommand):
    """The ``plugin`` inspection subcommand."""

    name = "plugin"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.plugin_action == "list":
            bundles = discover_installed_extension_bundles()
            if args.json:
                print(json.dumps([_bundle_payload(bundle) for bundle in bundles]))
                return
            for bundle in bundles:
                print(
                    f"{bundle.bundle_id}\t{bundle.descriptor.bundle_version}\t"
                    f"{bundle.distribution_name}=={bundle.distribution_version}"
                )
            return

        bundle = _one_installed_bundle(args.bundle_id)
        if args.plugin_action == "inspect":
            print(json.dumps(_bundle_payload(bundle), indent=2, sort_keys=True))
            return
        if args.plugin_action == "validate":
            permissions = parse_component_permission_allowlist(args.allow_permission)
            resolution = resolve_extension_startup(
                (bundle.manifest_path,),
                enabled_bundle_ids=(bundle.bundle_id,),
                allowed_permissions=permissions,
            )
            print(json.dumps(asdict(resolution.diagnostics()), sort_keys=True))
            return
        raise ValueError(f"unsupported plugin action: {args.plugin_action!r}")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Inspect installed vLLM-HUST extension Bundles.",
        )
        actions = parser.add_subparsers(required=True, dest="plugin_action")

        list_parser = actions.add_parser("list", help="List installed Bundles.")
        list_parser.add_argument("--json", action="store_true")

        inspect_parser = actions.add_parser(
            "inspect", help="Show one installed Bundle descriptor."
        )
        inspect_parser.add_argument("bundle_id")

        validate_parser = actions.add_parser(
            "validate", help="Validate one installed Bundle against this host."
        )
        validate_parser.add_argument("bundle_id")
        validate_parser.add_argument(
            "--allow-permission",
            action="append",
            default=[],
            help="Allow one declared capability during validation; repeat as needed.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [PluginSubcommand()]
