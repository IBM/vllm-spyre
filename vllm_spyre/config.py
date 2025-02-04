import hashlib
import operator
from dataclasses import dataclass, field
from typing import Any, List, Optional

from vllm.logger import init_logger

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)

_POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: str = "generate"  # The runner type to launch for the model.

    # Maximum number of tokens to be processed in a single iteration.
    max_num_batched_tokens: int = field(default=None)  # type: ignore

    # Maximum number of sequences to be processed in a single iteration.
    max_num_seqs: int = 128

    # Maximum length of a sequence (including prompt and generated text).
    max_model_len: int = 8192

    # The number of slots to allocate per sequence per
    # step, beyond the known token ids. This is used in speculative
    # decoding to store KV activations of tokens which may or may not be
    # accepted.
    num_lookahead_slots: int = 0

    # Apply a delay (of delay factor multiplied by previous
    # prompt latency) before scheduling next prompt.
    delay_factor: float = 0.0

    # If True, prefill requests can be chunked based
    # on the remaining max_num_batched_tokens.
    enable_chunked_prefill: bool = False

    is_multimodal_model: bool = False

    # NOTE: The following multimodal encoder budget will be initialized to
    # max_num_batched_tokens and overridden in case max multimodal embedding
    # size is larger.
    # TODO (ywang96): Make these configurable.
    # Multimodal encoder compute budget, only used in V1
    max_num_encoder_input_tokens: int = field(default=None)  # type: ignore

    # Multimodal encoder cache size, only used in V1
    encoder_cache_size: int = field(default=None)  # type: ignore

    # Whether to perform preemption by swapping or
    # recomputation. If not specified, we determine the mode as follows:
    # We use recomputation by default since it incurs lower overhead than
    # swapping. However, when the sequence group has multiple sequences
    # (e.g., beam search), recomputation is not currently supported. In
    # such a case, we use swapping instead.
    preemption_mode: Optional[str] = None

    num_scheduler_steps: int = 1

    multi_step_stream_outputs: bool = False

    # Private API. If used, scheduler sends delta data to
    # workers instead of an entire data. It should be enabled only
    # when SPMD worker architecture is enabled. I.e.,
    # VLLM_USE_RAY_SPMD_WORKER=1
    send_delta_data: bool = False

    # The scheduling policy to use. "fcfs" (default) or "priority".
    policy: str = "fcfs"

    chunked_prefill_enabled: bool = field(init=False)

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(self.max_model_len, 2048)
                else:
                    # This value is chosen to have a balance between ITL
                    # and TTFT. Note it is not optimized for throughput.
                    self.max_num_batched_tokens = 2048
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(self.max_model_len, 2048)

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
        from vllm.platforms import current_platform
        self.spyre_scheduling_enabled = current_platform.get_device_name(
        ) == "spyre"
        if self.spyre_scheduling_enabled:
            # load warmup shapes and sort by "speed"
            wup_prompt_lens = envs_spyre.VLLM_SPYRE_WARMUP_PROMPT_LENS or []
            wup_batch_sizes = envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES or []
            if len(wup_prompt_lens) != len(wup_batch_sizes):
                raise RuntimeError(
                    "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                    "VLLM_SPYRE_WARMUP_BATCH_SIZES must have equal length")
            if self.runner_type == "pooling":
                wup_new_tokens = [0] * len(wup_prompt_lens)
            else:
                wup_new_tokens = envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS or []
                if len(wup_new_tokens) != len(wup_prompt_lens):
                    raise RuntimeError(
                        "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                        "VLLM_SPYRE_WARMUP_NEW_TOKENS must have equal length")

            print("[SchedulerConfig] VLLM_SPYRE_WARMUP_PROMPT_LENS =",
                  wup_prompt_lens)
            print("[SchedulerConfig] VLLM_SPYRE_WARMUP_NEW_TOKENS =",
                  wup_new_tokens)
            print("[SchedulerConfig] VLLM_SPYRE_WARMUP_BATCH_SIZES =",
                  wup_batch_sizes)

            self.spyre_warmup_shapes = tuple(
                sorted([{
                    'prompt_length': pl,
                    'new_tokens': nt,
                    'batch_size': bs
                } for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens,
                                        wup_batch_sizes)],
                       key=operator.itemgetter('batch_size', 'prompt_length')))
        self._verify_args()

    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1
