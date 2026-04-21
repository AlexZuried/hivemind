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
from hivemind.inference.mist_node import MistNode, MistConfig, run_mist_node
from hivemind.inference.cli import run_inference_cli

# Ghost Protocol imports
from hivemind.inference.ghost import (
    ChronoExecutor, SpeculativeState,
    NeuralEntropyCoder, SharedCodebook,
    MeshOrchestrator, TaskPacket,
    NeuralZipper, ZipperConfig, AdaptivePredictor,
    ShadowValidator, ConsensusEngine, ShadowConfig, NodeReputation, ShadowConsensusValidator,
    HolographicContextManager, AnticipatoryRouter, NeuroPlasticTopology
)

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
    "NeuralZipper",
    "ZipperConfig",
    "AdaptivePredictor",
    "ShadowValidator",
    "ConsensusEngine",
    "ShadowConfig",
    "NodeReputation",
    "ShadowConsensusValidator",
    "HolographicContextManager",
    "AnticipatoryRouter",
    "NeuroPlasticTopology",
    
    # Mist Protocol
    "MistNode",
    "MistConfig",
    "run_mist_node",
    
    # CLI
    "run_inference_cli",
]
