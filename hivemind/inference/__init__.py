"""
Decentralized AI Inference Module

This module enables running large language models across distributed GPU/CPU resources
contributed by multiple participants worldwide. It implements:
- Pipeline parallelism for splitting large models across nodes
- Performance optimizations (speculative execution, adaptive compression, checkpointing)
- Contribution tracking for fair token distribution
- Smart scheduling based on real-time performance metrics
- Simple CLI interface for non-technical users
- GHOST PROTOCOL: Revolutionary performance layer with time-travel speculation,
  quantum semantic compression, and liquid topology for breakthrough speed
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

# Ghost Protocol: Next-generation performance engine
try:
    from hivemind.inference.ghost import (
        ChronoExecutor,
        SpeculativeState,
        NeuralEntropyCoder,
        SharedCodebook,
        MeshOrchestrator,
        TaskPacket
    )
    GHOST_AVAILABLE = True
except ImportError:
    GHOST_AVAILABLE = False
    ChronoExecutor = None
    SpeculativeState = None
    NeuralEntropyCoder = None
    SharedCodebook = None
    MeshOrchestrator = None
    TaskPacket = None

__all__ = [
    # Core pipeline
    "PipelineParallelRunner",
    "ModelChunkProvider", 
    
    # Contribution & rewards
    "ContributionTracker",
    "TokenRewardCalculator",
    
    # Discovery
    "LayerDiscoveryProtocol",
    "ResourceRegistry",
    
    # Performance optimizations
    "PerformanceMonitor",
    "AdaptiveCompressor",
    "SpeculativeExecutor",
    "CheckpointManager",
    "SmartScheduler",
    
    # Ghost Protocol (if available)
    "ChronoExecutor",
    "SpeculativeState",
    "NeuralEntropyCoder",
    "SharedCodebook",
    "MeshOrchestrator",
    "TaskPacket",
    
    # CLI
    "run_inference_cli",
]
