"""
Decentralized AI Inference Module

This module enables running large language models across distributed GPU/CPU resources
contributed by multiple participants worldwide. It implements:
- Pipeline parallelism for splitting large models across nodes
- Contribution tracking for fair token distribution
- Simple CLI interface for non-technical users
"""

from hivemind.inference.pipeline import PipelineParallelRunner, ModelChunkProvider
from hivemind.inference.contribution import ContributionTracker, TokenRewardCalculator
from hivemind.inference.discovery import LayerDiscoveryProtocol, ResourceRegistry
from hivemind.inference.cli import run_inference_cli

__all__ = [
    "PipelineParallelRunner",
    "ModelChunkProvider", 
    "ContributionTracker",
    "TokenRewardCalculator",
    "LayerDiscoveryProtocol",
    "ResourceRegistry",
    "run_inference_cli",
]
