"""
Decentralized AI Inference Module

This module enables running large language models across distributed GPU/CPU resources
contributed by multiple participants worldwide. It implements:
- Pipeline parallelism for splitting large models across nodes
- Performance optimizations (speculative execution, adaptive compression, checkpointing)
- Contribution tracking for fair token distribution
- Smart scheduling based on real-time performance metrics
- Simple CLI interface for non-technical users
"""

from hivemind.inference.pipeline import PipelineParallelRunner, ModelChunkProvider
from hivemind.inference.contribution import ContributionTracker, TokenRewardCalculator
from hivemind.inference.discovery import LayerDiscoveryProtocol, ResourceRegistry
from hivemind.inference.performance import (
    PerformanceMonitor,
    AdaptiveCompressor,
    SpeculativeExecutor,
    CheckpointManager,
    SmartScheduler
)
from hivemind.inference.cli import run_inference_cli

__all__ = [
    "PipelineParallelRunner",
    "ModelChunkProvider", 
    "ContributionTracker",
    "TokenRewardCalculator",
    "LayerDiscoveryProtocol",
    "ResourceRegistry",
    "PerformanceMonitor",
    "AdaptiveCompressor",
    "SpeculativeExecutor",
    "CheckpointManager",
    "SmartScheduler",
    "run_inference_cli",
]
