# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionWithCrossConSTConfig(FairseqDataclass):
    alpha: float = field(
        default=0.0,
        metadata={"help": "alpha for crossconst, 0 means no crossconst"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_with_crossconst", dataclass=CrossEntropyCriterionWithCrossConSTConfig)
class CrossEntropyCriterionWithCrossConST(FairseqCriterion):
    def __init__(self, task, sentence_avg, alpha=0.0):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.alpha = alpha

    def cross_const(self, model, omega, sample, reduce):

        prob = F.softmax(omega, dim=-1)
        valid_indices = (sample["target"] != self.padding_idx)
        src_lengths = valid_indices.sum(dim=-1)
        pad_lengths = (sample["target"] == self.padding_idx).sum(dim=-1)
        src_tokens_list = []
        for i in range(sample["target"].size(0)):
            src_tokens_list.append(
                F.pad(
                    sample["target"][i][:src_lengths[i].item()],
                    (pad_lengths[i].item(), 0),
                    value=self.padding_idx
                ).unsqueeze(dim=0)
            )
        src_tokens = torch.cat(src_tokens_list)

        encoder_out = model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths)
        decoder_out = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out,
            lang_id=0)

        loss = F.kl_div(
            input=F.log_softmax(decoder_out[0], dim=-1),
            target=prob, reduction='none')
        loss = loss.sum(dim=-1)
        loss = loss * valid_indices.float()
        if reduce:
            loss = loss.sum()

        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        if model.training:
            loss += self.alpha * self.cross_const(model, net_output[0], sample, reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
