from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch

from vllm.config import (
    ContrastiveDecodingConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler,
    SpecDecodeStochasticBaseSampler,
)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler,
)
from vllm.sequence import (
    VLLM_INVALID_TOKEN_ID,
    CompletionSequenceGroupOutput,
    ExecuteModelRequest,
    HiddenStates,
    SequenceGroupMetadata,
    get_all_seq_ids_and_request_ids,
)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.contrast_decode.contrast_model_runner import ContrastModelRunner

from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


def create_contrastive_worker(*args, **kwargs) -> "ContrastiveDecodeWorker":
    assert "contrastive_decoding_config" in kwargs
    contrastive_decoding_config: ContrastiveDecodingConfig = kwargs.get(
        "contrastive_decoding_config"
    )
    assert contrastive_decoding_config is not None

    contrastive_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = ContrastModelRunner
    base_worker = Worker(*args, **kwargs)

    contrastive_worker_kwargs.update(
        parallel_config=contrastive_decoding_config.parallel_config,
    )

    contrastive_decode_worker = ContrastiveDecodeWorker.create_worker(
        base_worker=base_worker,
        worker_kwargs=contrastive_worker_kwargs,
        positive_model_config=contrastive_decoding_config.positive_model_config,
        negative_model_config=contrastive_decoding_config.negative_model_config,
        sampler_alpha=contrastive_decoding_config.sampler_alpha,
    )

    return contrastive_decode_worker


class ContrastiveDecodeWorker(LoraNotSupportedWorkerBase):

    @classmethod
    def create_worker(
        cls,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_model_config: Optional[ModelConfig],
        negative_model_config: Optional[ModelConfig],
        sampler_alpha: float,
    ) -> "ContrastiveDecodeWorker":
        """
        Create a ContrastiveDecodeWorker from the given arguments.
        """
        assert (
            positive_model_config is not None or negative_model_config is not None
        ), "Either positive_model_config or negative_model_config must be specified."

        if positive_model_config is None:
            positive_worker = None
        else:
            positive_worker_kwargs = worker_kwargs.copy()
            positive_worker_kwargs.update(
                model_config=positive_model_config,
            )
            positive_worker = MultiStepWorker(**positive_worker_kwargs)

        if negative_model_config is None:
            negative_worker = None
        else:
            negative_worker_kwargs = worker_kwargs.copy()
            negative_worker_kwargs.update(
                model_config=negative_model_config,
            )
            negative_worker = MultiStepWorker(**negative_worker_kwargs)

        # decode_sampler = ContrastiveSampler(
        #     alpha=sampler_alpha,
        # )

        return cls(
            base_worker=base_worker,
            worker_kwargs=worker_kwargs,
            positive_worker=positive_worker,
            negative_worker=negative_worker,
            sampler_alpha=sampler_alpha,
            # decode_sampler=decode_sampler,
        )

    def __init__(
        self,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_worker: Optional[WorkerBase],
        negative_worker: Optional[WorkerBase],
        sampler_alpha: float,
        # decode_sampler: ContrastiveSamplerBase,
    ):
        self.base_worker = base_worker
        self.worker_kwargs = worker_kwargs
        self.positive_worker = positive_worker
        self.negative_worker = negative_worker
        # self.decode_sampler = decode_sampler
        self.sampler_alpha = sampler_alpha
        self.sampler = Sampler()

    def init_device(self) -> None:
        self.base_worker.init_device()
        if self.positive_worker is not None:
            self.positive_worker.init_device()
        if self.negative_worker is not None:
            self.negative_worker.init_device()

        self.base_worker.load_model()
        if self.positive_worker is not None:
            self.positive_worker.load_model()
        if self.negative_worker is not None:
            self.negative_worker.load_model()

        # self._metrics.init_gpu_tensors(self.rank)
        self.decode_sampler.init_gpu_tensors(self.rank)

    def load_model(self, *args, **kwargs):
        pass

    @torch.inference_mode()
    def execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform contrastive decoding on the input batch."""
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        if execute_model_req is None:
            """
            This signals that there's no more requests to process for now.
            All workers are running infinite loop with broadcast_tensor_dict,
            and it stops the loop when the driver broadcasts an empty input.
            Send an empty input to notify all other workers to stop their
            execution loop.
            """
            broadcast_tensor_dict({}, src=0)
            return []

        disable_all_contrastive_decoding = (
            self._should_disable_all_contrastive_decoding(execute_model_req)
        )
        boardcast_dict = dict(
            disable_all_contrastive_decoding=disable_all_contrastive_decoding,
        )
        broadcast_tensor_dict(boardcast_dict, src=self._driver_rank)

        if disable_all_contrastive_decoding:
            return self._run_no_contrastive_decoding(execute_model_req)

        return self._run_contrastive_decoding(execute_model_req)

    def _should_disable_all_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> bool:
        """
        Determine if all contrastive decoding should be disabled.
        """
        # TODO: Implement this
        return False

    def _run_no_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model without contrastive decoding.
        """
        sampler_output = self.base_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        sampler_output.sampled_token_ids = None
        sampler_output.sampled_token_probs = None
        sampler_output.logprobs = None
        sampler_output.logits = None
        return [sampler_output]

    def _run_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model with contrastive decoding.
        """
        base_sampler_output = self.base_worker.execute_model(execute_model_req)
        if self.positive_worker is not None:
            positive_sampler_output = self.positive_worker.execute_model(execute_model_req)
        else:
            positive_sampler_output = []
        if self.negative_worker is not None:
            negative_sampler_output = self.negative_worker.execute_model(execute_model_req)
        else:
            negative_sampler_output = []

        generators = self.base_worker.model_runner.get_generators(
            execute_model_req.finished_requests_ids)
        
        input_tokens_tensor, seq_lens, query_lens = self._prepare_input_tensors(
            execute_model_req.seq_group_metadata_list,
        )

        sampling_metadata = SamplingMetadata.prepare(
            execute_model_req.seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            self.base_worker.model_runner.pin_memory,
            generators,
        )

        contrastive_sampler_output = self._create_contrastive_sampler_output(
            sampling_metadata,
            base_sampler_output,
            positive_sampler_output,
            negative_sampler_output,
        )
        return contrastive_sampler_output

    def _create_contrastive_sampler_output(
        self,
        sampling_metadata: SamplingMetadata,
        base_sampler_output: List[SamplerOutput],
        positive_sampler_output: List[SamplerOutput],
        negative_sampler_output: List[SamplerOutput],
    ) -> List[SamplerOutput]:
        """
        Create a contrastive sampler output.
        """
        # Sample the next token.
        logits = base_sampler_output[0].logits
        if self.positive_worker:
            logits = logits + self.sampler_alpha * positive_sampler_output[0].logits
        if self.negative_worker:
            logits = logits - self.sampler_alpha * negative_sampler_output[0].logits

        output: SamplerOutput = self.base_worker.model_runner.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return [output]

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        if not seq_group_metadata_list:
            return torch.empty(0, device=self.device), [], []

        input_tokens: List[int] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_data in seq_group_metadata.seq_data.values():
                seq_data_len = seq_data.get_len()
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                    seq_len = min(
                        seq_data_len,
                        context_len + seq_group_metadata.token_chunk_size)
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                    seq_lens.append(seq_len)
                    input_tokens.extend(tokens)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    input_tokens.append(seq_data.get_last_token_id())
                    query_lens.append(1)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        return input_tokens_tensor, seq_lens, query_lens
    
    @cached_property
    def vocab_size(self) -> int:
        return self.base_worker.vocab_size

    @property
    def rank(self) -> int:
        return self.base_worker.rank

    @property
    def device(self) -> torch.device:
        return self.base_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0
