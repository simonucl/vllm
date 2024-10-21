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
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScorer,
    SpeculativeScores,
)
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import (
    Timer,
    create_logprobs_output,
    create_sequence_group_output,
    get_all_num_logprobs,
    get_sampled_token_logprobs,
    nvtx_range,
    split_batch_by_proposal_len,
)
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

    kwargs["model_runner_cls"] = TargetModelRunner
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

        decode_sampler = ContrastiveSampler(
            alpha=sampler_alpha,
        )

        return cls(
            base_worker=base_worker,
            worker_kwargs=worker_kwargs,
            positive_worker=positive_worker,
            negative_worker=negative_worker,
            sampler_alpha=sampler_alpha,
            decode_sampler=decode_sampler,
        )

    def __init__(
        self,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_worker: Optional[WorkerBase],
        negative_worker: Optional[WorkerBase],
        decode_sampler: ContrastiveSamplerBase,
    ):
        self.base_worker = base_worker
        self.worker_kwargs = worker_kwargs
        self.positive_worker = positive_worker
        self.negative_worker = negative_worker
        self.decode_sampler = decode_sampler

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

    @nvtx_range("contrastive_decode_worker.execute_model")
    def _execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Execute the model for a single step.
        """
        pass

    def _should_disable_all_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> bool:
        """
        Determine if all contrastive decoding should be disabled.
        """
        # TODO: Implement this
        return False

    @nvtx_range("contrastive_decode_worker._run_no_contrastive_decoding")
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
        return [sampler_output]

    @nvtx_range("contrastive_decode_worker._run_contrastive_decoding")
    def _run_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model with contrastive decoding.
        """
        base_sampler_output = self.base_worker.execute_model(execute_model_req)
        assert len(base_sampler_output) == 1
        base_sampler_output = base_sampler_output[0]

        positive_sampler_output = self.positive_worker.execute_model(execute_model_req)
        assert len(positive_sampler_output) == 1
        positive_sampler_output = positive_sampler_output[0]

        negative_sampler_output = self.negative_worker.execute_model(execute_model_req)
        assert len(negative_sampler_output) == 1
        negative_sampler_output = negative_sampler_output[0]

        contrastive_sampler_output = self._create_contrastive_sampler_output(
            base_sampler_output,
            positive_sampler_output,
            negative_sampler_output,
        )
        return contrastive_sampler_output

    def _create_contrastive_sampler_output(
        self,
        base_sampler_output: SamplerOutput,
        positive_sampler_output: SamplerOutput,
        negative_sampler_output: SamplerOutput,
    ) -> List[SamplerOutput]:
        """
        Create a contrastive sampler output.
        """
        

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
