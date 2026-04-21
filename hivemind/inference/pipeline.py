"""
Pipeline Parallelism for Decentralized Inference

Enables splitting large language models across multiple nodes globally,
allowing consumer hardware to collectively run massive models.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import torch
import torch.nn as nn

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from hivemind.moe.server.dht_handler import get_experts
from hivemind.moe.expert_uid import ExpertUID, ExpertInfo
from hivemind.p2p import P2P, PeerID
from hivemind.utils import get_logger, get_dht_time
from hivemind.inference.discovery import LayerDiscoveryProtocol

logger = get_logger(__name__)


@dataclass
class ModelChunkConfig:
    """Configuration for a model chunk provider"""
    model_name: str
    layer_start: int
    layer_end: int
    device: str = "cuda"
    hidden_dim: int = 4096
    expert_cls: str = "transformer"


class ModelChunkProvider:
    """
    A node that contributes GPU/CPU resources by hosting a chunk of a large model.
    
    This allows users with consumer hardware to contribute specific layers
    of a large model (e.g., Kimi K2.6) to the global compute pool.
    """
    
    def __init__(
        self,
        dht: DHT,
        config: ModelChunkConfig,
        initial_peers: List[str] = None,
        update_period: float = 30.0,
        expiration: float = 120.0
    ):
        self.dht = dht
        self.config = config
        self.initial_peers = initial_peers or []
        self.update_period = update_period
        self.expiration = expiration
        
        # Load the model chunk
        self.model_chunk = self._load_model_chunk()
        self.expert_uid = self._generate_expert_uid()
        
        # Track contributions
        self._compute_times: List[float] = []
        self._tokens_processed: int = 0
        self._lock = threading.Lock()
        
        logger.info(
            f"ModelChunkProvider ready: {config.model_name} "
            f"layers {config.layer_start}-{config.layer_end} on {config.device}"
        )
    
    def _load_model_chunk(self) -> nn.Module:
        """Load the specific chunk of the model this node will host"""
        from hivemind.moe.server.layers import name_to_block
        
        if self.config.expert_cls not in name_to_block:
            raise ValueError(f"Unknown expert class: {self.config.expert_cls}")
            
        model_chunk = name_to_block[self.config.expert_cls](self.config.hidden_dim)
        model_chunk = model_chunk.to(self.config.device)
        return model_chunk
    
    def _generate_expert_uid(self) -> ExpertUID:
        """Generate unique expert UID for this model chunk"""
        return f"{self.config.model_name}.layer.{self.config.layer_start}.{self.config.layer_end}"
    
    def advertise(self):
        """Advertise this model chunk to the DHT network"""
        from hivemind.moe.server.dht_handler import declare_experts
        
        # Store compute capability in DHT
        key = f"compute:{self.config.model_name}:{self.config.layer_start}-{self.config.layer_end}"
        value = {
            "peer_id": self.dht.peer_id.to_base58(),
            "layer_start": self.config.layer_start,
            "layer_end": self.config.layer_end,
            "device_type": "gpu" if "cuda" in self.config.device else "cpu",
            "hidden_dim": self.config.hidden_dim,
            "timestamp": get_dht_time()
        }
        
        self.dht.store(key, value, expiration_time=get_dht_time() + self.expiration)
        logger.debug(f"Advertised model chunk: {key}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process incoming hidden states through this model chunk.
        
        :param hidden_states: Input tensor from previous layer
        :returns: Output tensor to send to next layer
        """
        start_time = time.time()
        
        with torch.no_grad():
            hidden_states = hidden_states.to(self.config.device)
            output = self.model_chunk(hidden_states)
        
        compute_time = time.time() - start_time
        
        with self._lock:
            self._compute_times.append(compute_time)
            self._tokens_processed += hidden_states.shape[0]
        
        return output.cpu()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this provider's contributions"""
        with self._lock:
            avg_compute_time = sum(self._compute_times) / len(self._compute_times) if self._compute_times else 0
            return {
                "expert_uid": self.expert_uid,
                "total_compute_time": sum(self._compute_times),
                "avg_compute_time": avg_compute_time,
                "tokens_processed": self._tokens_processed,
                "device": self.config.device
            }


class PipelineParallelRunner:
    """
    Runs inference by chaining multiple model chunks across different nodes.
    
    This enables running very large models (like Kimi K2.6) by splitting
    the pipeline across multiple consumer GPUs worldwide.
    """
    
    def __init__(self, dht: DHT, model_name: str, p2p: P2P = None):
        self.dht = dht
        self.model_name = model_name
        self.p2p = p2p
        self.discovery = LayerDiscoveryProtocol(dht, model_name)
        self._topology: Optional[List[ExpertInfo]] = None
        
    async def discover_topology(self, max_layers: int = 100) -> List[ExpertInfo]:
        """
        Discover available model chunks and assemble the inference pipeline.
        
        :param max_layers: Maximum number of layers to discover
        :returns: Ordered list of experts forming the pipeline
        """
        self._topology = await self.discovery.assemble_pipeline(max_layers)
        
        if not self._topology:
            raise RuntimeError(f"No compute resources found for model {self.model_name}")
        
        logger.info(
            f"Assembled pipeline with {len(self._topology)} chunks "
            f"from {len(set(e.peer_id for e in self._topology))} unique peers"
        )
        
        return self._topology
    
    def _get_p2p(self) -> P2P:
        """Get P2P instance, creating one if necessary"""
        if self.p2p is None:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.p2p = loop.run_until_complete(self.dht.replicate_p2p())
        return self.p2p
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text by passing through the distributed pipeline.
        
        :param prompt: Input text prompt
        :param max_tokens: Maximum tokens to generate
        :returns: Generated text
        """
        if self._topology is None:
            await self.discover_topology()
        
        # Tokenize input (simplified - in practice use actual tokenizer)
        hidden_states = self._encode_prompt(prompt)
        
        # Pass through each layer in the pipeline
        p2p = self._get_p2p()
        for expert_info in self._topology:
            expert = RemoteExpert(expert_info, p2p)
            hidden_states = expert(hidden_states)
        
        # Decode output
        generated_text = self._decode_output(hidden_states)
        return generated_text
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Convert prompt to initial hidden states (placeholder)"""
        # In practice, this would use the model's actual tokenizer and embeddings
        batch_size = 1
        seq_len = len(prompt.split())
        hidden_dim = 4096  # Should match model config
        return torch.randn(batch_size, seq_len, hidden_dim)
    
    def _decode_output(self, hidden_states: torch.Tensor) -> str:
        """Convert final hidden states to text (placeholder)"""
        # In practice, this would use the model's actual decoder
        return "[Generated text placeholder]"
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the current pipeline"""
        if not self._topology:
            return {"status": "not_initialized"}
        
        peer_ids = [str(e.peer_id) for e in self._topology]
        unique_peers = set(peer_ids)
        
        return {
            "model_name": self.model_name,
            "total_layers": len(self._topology),
            "unique_peers": len(unique_peers),
            "peer_distribution": {peer: peer_ids.count(peer) for peer in unique_peers}
        }


async def run_pipeline_example():
    """Example usage of pipeline parallelism"""
    from hivemind.dht import DHT
    
    # Initialize DHT
    dht = DHT(start=True)
    
    # Create pipeline runner
    runner = PipelineParallelRunner(dht, "kimi-k2.6")
    
    try:
        # Discover available compute resources
        topology = await runner.discover_topology()
        print(f"Found {len(topology)} model chunks")
        
        # Run inference
        result = await runner.generate("What is quantum computing?", max_tokens=50)
        print(f"Generated: {result}")
        
    finally:
        dht.shutdown()
