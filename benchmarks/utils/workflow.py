# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.utils.genai import run_concurrency_sweep
from benchmarks.utils.plot import generate_plots
from benchmarks.utils.vanilla_client import VanillaBackendClient
from deploy.utils.dynamo_deployment import DynamoDeploymentClient


async def deploy_and_wait(client: DynamoDeploymentClient, manifest_path: str) -> None:
    await client.create_deployment(manifest_path)
    await client.wait_for_deployment_ready(timeout=1800)


async def teardown(client) -> None:
    try:
        if hasattr(client, "stop_port_forward"):
            client.stop_port_forward()
        await client.delete_deployment()
    except Exception:
        pass


async def run_benchmark_workflow(
    namespace: str,
    agg_manifest: str = None,
    disagg_manifest: str = None,
    vanilla_manifest: str = None,
    isl: int = 200,
    std: int = 10,
    osl: int = 200,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    output_dir: str = "benchmarks/results",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Deploy and benchmark aggregated (if specified)
    if agg_manifest:
        print("🚀 Starting aggregated deployment benchmark...")
        agg_name = Path(agg_manifest).stem
        agg_client = DynamoDeploymentClient(
            namespace=namespace, deployment_name=agg_name
        )
        await deploy_and_wait(agg_client, agg_manifest)
        try:
            print("Starting aggregated concurrency sweep!", flush=True)
            print(
                "This may take several minutes - running through multiple concurrency levels...",
                flush=True,
            )
            run_concurrency_sweep(
                service_url=agg_client.port_forward_frontend(quiet=True),
                model_name=model,
                isl=isl,
                osl=osl,
                stddev=std,
                output_dir=Path(output_dir) / "agg",
            )
            agg_client.stop_port_forward()
        finally:
            await teardown(agg_client)
        print("✅ Aggregated deployment benchmark completed!")
    else:
        print("⏭️  Skipping aggregated deployment (not specified)")

    # Deploy and benchmark disaggregated (if specified)
    if disagg_manifest:
        print("🚀 Starting disaggregated deployment benchmark...")
        disagg_name = Path(disagg_manifest).stem
        disagg_client = DynamoDeploymentClient(
            namespace=namespace, deployment_name=disagg_name
        )
        await deploy_and_wait(disagg_client, disagg_manifest)
        try:
            print("Starting disaggregated concurrency sweep!", flush=True)
            run_concurrency_sweep(
                service_url=disagg_client.port_forward_frontend(quiet=True),
                model_name=model,
                isl=isl,
                osl=osl,
                stddev=std,
                output_dir=Path(output_dir) / "disagg",
            )
            disagg_client.stop_port_forward()
        finally:
            await teardown(disagg_client)
        print("✅ Disaggregated deployment benchmark completed!")
    else:
        print("⏭️  Skipping disaggregated deployment (not specified)")

    # Deploy and benchmark vanilla backend (if specified)
    if vanilla_manifest:
        print("🚀 Starting vanilla backend deployment benchmark...")
        vanilla_client = VanillaBackendClient(namespace=namespace)
        await vanilla_client.create_deployment(vanilla_manifest)
        await vanilla_client.wait_for_deployment_ready(timeout=1800)
        try:
            print("Starting vanilla backend concurrency sweep!", flush=True)
            run_concurrency_sweep(
                service_url=vanilla_client.port_forward_frontend(quiet=True),
                model_name=model,
                isl=isl,
                osl=osl,
                stddev=std,
                output_dir=Path(output_dir) / "vanilla",
            )
        finally:
            await teardown(vanilla_client)
        print("✅ Vanilla backend deployment benchmark completed!")
    else:
        print("⏭️  Skipping vanilla backend deployment (not specified)")

    # Generate plots across outputs (only for available data)
    print("📊 Generating plots...")
    generate_plots(base_output_dir=Path(output_dir))
