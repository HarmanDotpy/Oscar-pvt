# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import torch
import torch.nn.functional as F

# from transformers.pytorch_transformers.modeling_bert import (BertConfig,
#         load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
#         BertPreTrainedModel)
# from transformers.pytorch_transformers.modeling_utils import (PreTrainedModel,
#     WEIGHTS_NAME, TF_WEIGHTS_NAME)
# from transformers.pytorch_transformers.file_utils import cached_path
import transformers
from transformers.models.bert.modeling_bert import (BertConfig,
        load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertPreTrainedModel)
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = BERT_PRETRAINED_MODEL_ARCHIVE_LIST # Hack, see https://github.com/huggingface/transformers/issues/5842
# from transformers.modeling_utils import (PreTrainedModel,
#     WEIGHTS_NAME, TF_WEIGHTS_NAME)
from transformers.modeling_utils import PreTrainedModel

# # from transformers.file_utils import cached_path # cached path is replaced with cached file
# from transformers.utils import cached_file

logger = logging.getLogger()

#TODO: make this compatible with latest transformers
class CaptionPreTrainedModel(BertPreTrainedModel):
    """ Expand base class for image captioning modeling.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'bert'

    def __init__(self, config, *inputs, **kwargs):
        super(CaptionPreTrainedModel, self).__init__(config, *inputs, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling
        and beam-search.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.

            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)  # do greedy decoding without beam search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, do_sample=True, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[0][i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, num_beams=3)  # generate sequences using greedy beam search decoding (3 beams)
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences using using greedy search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, "`bos_token_id` should be a positive integer."
        assert isinstance(pad_token_id, int) and pad_token_id >= 0, "`pad_token_id` should be a positive integer."
        assert isinstance(eos_token_ids, (list, tuple)) and (
            e >= 0 for e in eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        if num_return_sequences != 1:
            for i in range(len(output)):
                output[i] = output[i].view(batch_size, num_return_sequences, -1)
        return output

    def _decode_step(self, input_ids, past):
        model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        token_len = outputs[0].shape[1]
        if self.od_labels_len == 0:
            next_token_idx = token_len - 1
        else:
            if token_len == 2:
                assert self._do_output_past(outputs)
                next_token_idx = 1
            else:
                next_token_idx = token_len - self.od_labels_len - 1

        next_token_logits = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
        assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

        # if model has past, then set the past variable to speed up decoding
        if self._do_output_past(outputs):
            past = outputs[1]
        return next_token_logits, past

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1
            assert outputs[0].shape[1] == token_len

            next_token_logits = outputs[0][:, next_token_idx, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            #for t in input_ids:
                #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        num_keep_best = self.num_keep_best
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        TOPN_PER_BEAM = 2

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert outputs[0].shape[1] == token_len
            scores = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
            assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [TOPN_PER_BEAM] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=TOPN_PER_BEAM)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, TOPN_PER_BEAM)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, TOPN_PER_BEAM).to(next_words.device)
                next_words = next_words.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 1st position
                    reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]

        return decoded, logprobs


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


#TODO: for now, I am removing the old from_pretrained and adding the from_pretrained function since it seems to be almost the same as pretrainedModel, so doesn't seem to be needed
class ImgPreTrainedModel(PreTrainedModel):
    """ Base class for all models. Handle loading/storing model config and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(ImgPreTrainedModel, self).__init__(config, *inputs, **kwargs)


    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path , *model_args, **kwargs):
    #     r"""
    #     Instantiate a pretrained pytorch model from a pre-trained model configuration.

    #     The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
    #     the model, you should first set it back in training mode with `model.train()`.

    #     The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
    #     pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    #     task.

    #     The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
    #     weights are discarded.

    #     Parameters:
    #         pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
    #             Can be either:

    #                 - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
    #                   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
    #                   user or organization name, like `dbmdz/bert-base-german-cased`.
    #                 - A path to a *directory* containing model weights saved using
    #                   [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
    #                 - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    #                   this case, `from_tf` should be set to `True` and a configuration object should be provided as
    #                   `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    #                   PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
    #                 - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
    #                   `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
    #                   `True`.
    #                 - `None` if you are both providing the configuration and state dictionary (resp. with keyword
    #                   arguments `config` and `state_dict`).
    #         model_args (sequence of positional arguments, *optional*):
    #             All remaining positional arguments will be passed to the underlying model's `__init__` method.
    #         config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
    #             Can be either:

    #                 - an instance of a class derived from [`PretrainedConfig`],
    #                 - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

    #             Configuration for the model to use instead of an automatically loaded configuration. Configuration can
    #             be automatically loaded when:

    #                 - The model is a model provided by the library (loaded with the *model id* string of a pretrained
    #                   model).
    #                 - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
    #                   save directory.
    #                 - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    #                   configuration JSON file named *config.json* is found in the directory.
    #         state_dict (`Dict[str, torch.Tensor]`, *optional*):
    #             A state dictionary to use instead of a state dictionary loaded from saved weights file.

    #             This option can be used if you want to create a model from a pretrained configuration but load your own
    #             weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
    #             [`~PreTrainedModel.from_pretrained`] is not a simpler option.
    #         cache_dir (`Union[str, os.PathLike]`, *optional*):
    #             Path to a directory in which a downloaded pretrained model configuration should be cached if the
    #             standard cache should not be used.
    #         from_tf (`bool`, *optional*, defaults to `False`):
    #             Load the model weights from a TensorFlow checkpoint save file (see docstring of
    #             `pretrained_model_name_or_path` argument).
    #         from_flax (`bool`, *optional*, defaults to `False`):
    #             Load the model weights from a Flax checkpoint save file (see docstring of
    #             `pretrained_model_name_or_path` argument).
    #         ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
    #             Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
    #             as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
    #             checkpoint with 3 labels).
    #         force_download (`bool`, *optional*, defaults to `False`):
    #             Whether or not to force the (re-)download of the model weights and configuration files, overriding the
    #             cached versions if they exist.
    #         resume_download (`bool`, *optional*, defaults to `False`):
    #             Whether or not to delete incompletely received files. Will attempt to resume the download if such a
    #             file exists.
    #         proxies (`Dict[str, str]`, *optional*):
    #             A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
    #             'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
    #         output_loading_info(`bool`, *optional*, defaults to `False`):
    #             Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
    #         local_files_only(`bool`, *optional*, defaults to `False`):
    #             Whether or not to only look at local files (i.e., do not try to download the model).
    #         use_auth_token (`str` or *bool*, *optional*):
    #             The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
    #             when running `huggingface-cli login` (stored in `~/.huggingface`).
    #         revision (`str`, *optional*, defaults to `"main"`):
    #             The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
    #             git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
    #             identifier allowed by git.
    #         mirror (`str`, *optional*):
    #             Mirror source to accelerate downloads in China. If you are from China and have an accessibility
    #             problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
    #             Please refer to the mirror site for more information.
    #         _fast_init(`bool`, *optional*, defaults to `True`):
    #             Whether or not to disable fast initialization.

    #             <Tip warning={true}>

    #             One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
    #             4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
    #             [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

    #             </Tip>

    #         > Parameters for big model inference

    #         low_cpu_mem_usage(`bool`, *optional*):
    #             Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
    #             This is an experimental feature and a subject to change at any moment.
    #         torch_dtype (`str` or `torch.dtype`, *optional*):
    #             Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
    #             will be automatically derived from the model's weights.
    #         device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
    #             A map that specifies where each submodule should go. It doesn't need to be refined to each
    #             parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
    #             same device.

    #             To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
    #             more information about each option see [designing a device
    #             map](https://hf.co/docs/accelerate/main/big_modeling#designing-a-device-map).
    #         max_memory (`Dict`, *optional*):
    #             A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
    #             GPU and the available CPU RAM if unset.
    #         offload_folder (`str` or `os.PathLike`, *optional*):
    #             If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
    #         offload_state_dict (`bool`, *optional*):
    #             If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
    #             RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
    #             `True` when there is some disk offload.
    #         load_in_8bit (`bool`, *optional*, defaults to `False`):
    #             If `True`, will convert the loaded model into mixed-8bit quantized model. To use this feature please
    #             install `bitsandbytes` compiled with your CUDA version by running `pip install -i
    #             https://test.pypi.org/simple/ bitsandbytes-cudaXXX` where XXX is your CUDA version (e.g. 11.6 = 116).
    #             Make also sure that you have enough GPU RAM to store half of the model size since the 8bit modules are
    #             not compiled and adapted for CPUs.
    #         int8_threshold (`float`, *optional*, defaults to 6):
    #             Works together with `load_in_8bit`. This corresponds to the outlier threshold for outlier detection as
    #             described in `GPT3.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper. Any hidden
    #             states value that is above this threshold will be considered an outlier and the operation on those
    #             values will be done in fp16. Values are usually normally distributed, that is, most values are in the
    #             range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently
    #             distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8
    #             quantization works well for values of magnitude ~5, but beyond that, there is a significant performance
    #             penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models
    #             (small models, fine-tuning).
    #         subfolder (`str`, *optional*, defaults to `""`):
    #             In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
    #             specify the folder name here.

    #         kwargs (remaining dictionary of keyword arguments, *optional*):
    #             Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
    #             `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
    #             automatically loaded:

    #                 - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    #                   underlying model's `__init__` method (we assume all relevant updates to the configuration have
    #                   already been done)
    #                 - If a configuration is not provided, `kwargs` will be first passed to the configuration class
    #                   initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
    #                   corresponds to a configuration attribute will be used to override said attribute with the
    #                   supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    #                   will be passed to the underlying model's `__init__` function.

    #     <Tip>

    #     Passing `use_auth_token=True`` is required when you want to use a private model.

    #     </Tip>

    #     <Tip>

    #     Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
    #     use this method in a firewalled environment.

    #     </Tip>

    #     Examples:

    #     ```python
    #     >>> from transformers import BertConfig, BertModel

    #     >>> # Download model and configuration from huggingface.co and cache.
    #     >>> model = BertModel.from_pretrained("bert-base-uncased")
    #     >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
    #     >>> model = BertModel.from_pretrained("./test/saved_model/")
    #     >>> # Update configuration during loading.
    #     >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    #     >>> assert model.config.output_attentions == True
    #     >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
    #     >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
    #     >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
    #     >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
    #     >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
    #     ```

    #     * `low_cpu_mem_usage` algorithm:

    #     This is an experimental function that loads the model using ~1x model size CPU memory

    #     Here is how it works:

    #     1. save which state_dict keys we have
    #     2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
    #     3. after the model has been instantiated switch to the meta device all params/buffers that
    #     are going to be replaced from the loaded state_dict
    #     4. load state_dict 2nd time
    #     5. replace the params/buffers from the state_dict

    #     Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

    #     """
    #     config = kwargs.pop("config", None)
    #     state_dict = kwargs.pop("state_dict", None)
    #     cache_dir = kwargs.pop("cache_dir", None)
    #     from_tf = kwargs.pop("from_tf", False)
    #     from_flax = kwargs.pop("from_flax", False)
    #     ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
    #     force_download = kwargs.pop("force_download", False)
    #     resume_download = kwargs.pop("resume_download", False)
    #     proxies = kwargs.pop("proxies", None)
    #     output_loading_info = kwargs.pop("output_loading_info", False)
    #     local_files_only = kwargs.pop("local_files_only", False)
    #     use_auth_token = kwargs.pop("use_auth_token", None)
    #     revision = kwargs.pop("revision", None)
    #     trust_remote_code = kwargs.pop("trust_remote_code", None)
    #     _ = kwargs.pop("mirror", None)
    #     from_pipeline = kwargs.pop("_from_pipeline", None)
    #     from_auto_class = kwargs.pop("_from_auto", False)
    #     _fast_init = kwargs.pop("_fast_init", True)
    #     torch_dtype = kwargs.pop("torch_dtype", None)
    #     low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
    #     device_map = kwargs.pop("device_map", None)
    #     max_memory = kwargs.pop("max_memory", None)
    #     offload_folder = kwargs.pop("offload_folder", None)
    #     offload_state_dict = kwargs.pop("offload_state_dict", False)
    #     load_in_8bit = kwargs.pop("load_in_8bit", False)
    #     int8_threshold = kwargs.pop("int8_threshold", 6.0)
    #     subfolder = kwargs.pop("subfolder", "")
    #     commit_hash = kwargs.pop("_commit_hash", None)

    #     if trust_remote_code is True:
    #         logger.warning(
    #             "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
    #             " ignored."
    #         )
    #     if device_map is not None:
    #         if low_cpu_mem_usage is None:
    #             low_cpu_mem_usage = True
    #         elif not low_cpu_mem_usage:
    #             raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

    #     if low_cpu_mem_usage:
    #         # low_cpu_mem_usage requires PyTorch >= 1.9 to have the meta device.
    #         require_version_core("torch>=1.9")

    #         if is_deepspeed_zero3_enabled():
    #             raise ValueError(
    #                 "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
    #             )
    #         elif not is_accelerate_available():
    #             raise ImportError(
    #                 "Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`"
    #             )

    #     if load_in_8bit:
    #         if not (is_accelerate_available() and is_bitsandbytes_available()):
    #             raise ImportError(
    #                 "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of"
    #                 " bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or"
    #                 " pip install bitsandbytes` "
    #             )
    #         if torch_dtype == "auto" or torch_dtype != torch.float16:
    #             # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
    #             torch_dtype = torch.float16
    #             logger.info("Loading the model in mixed int8 - forcing the weights to be casted in float16")
    #         if device_map is None:
    #             raise ValueError(
    #                 "A device map needs to be passed to run convert models into mixed-int8 format. Please run"
    #                 "`.from_pretrained` with `device_map='auto'`"
    #             )
    #         if from_tf or from_flax:
    #             raise ValueError(
    #                 "Converting into mixed 8-bit weights from tf/flax weights is currently not supported, please make"
    #                 " sure the weights are in PyTorch format."
    #             )

    #     from_pt = not (from_tf | from_flax)

    #     user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    #     if from_pipeline is not None:
    #         user_agent["using_pipeline"] = from_pipeline

    #     if is_offline_mode() and not local_files_only:
    #         logger.info("Offline mode: forcing local_files_only=True")
    #         local_files_only = True

    #     # Load config if we don't provide a configuration
    #     if not isinstance(config, PretrainedConfig):
    #         config_path = config if config is not None else pretrained_model_name_or_path
    #         config, model_kwargs = cls.config_class.from_pretrained(
    #             config_path,
    #             cache_dir=cache_dir,
    #             return_unused_kwargs=True,
    #             force_download=force_download,
    #             resume_download=resume_download,
    #             proxies=proxies,
    #             local_files_only=local_files_only,
    #             use_auth_token=use_auth_token,
    #             revision=revision,
    #             subfolder=subfolder,
    #             _from_auto=from_auto_class,
    #             _from_pipeline=from_pipeline,
    #             **kwargs,
    #         )
    #     else:
    #         model_kwargs = kwargs

    #     if commit_hash is None:
    #         commit_hash = getattr(config, "_commit_hash", None)

    #     # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    #     # index of the files.
    #     is_sharded = False
    #     sharded_metadata = None
    #     # Load model
    #     loading_info = None

    #     if pretrained_model_name_or_path is not None:
    #         pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    #         is_local = os.path.isdir(pretrained_model_name_or_path)
    #         if is_local:
    #             if from_tf and os.path.isfile(
    #                 os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
    #             ):
    #                 # Load from a TF 1.0 checkpoint in priority if from_tf
    #                 archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
    #             elif from_tf and os.path.isfile(
    #                 os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
    #             ):
    #                 # Load from a TF 2.0 checkpoint in priority if from_tf
    #                 archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
    #             elif from_flax and os.path.isfile(
    #                 os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
    #             ):
    #                 # Load from a Flax checkpoint in priority if from_flax
    #                 archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
    #             elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)):
    #                 # Load from a PyTorch checkpoint
    #                 archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
    #             elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)):
    #                 # Load from a sharded PyTorch checkpoint
    #                 archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)
    #                 is_sharded = True
    #             # At this stage we don't have a weight file so we will raise an error.
    #             elif os.path.isfile(
    #                 os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
    #             ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
    #                 raise EnvironmentError(
    #                     f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
    #                     "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
    #                     "weights."
    #                 )
    #             elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
    #                 raise EnvironmentError(
    #                     f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
    #                     "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
    #                     "weights."
    #                 )
    #             else:
    #                 raise EnvironmentError(
    #                     f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
    #                     f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
    #                 )
    #         elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
    #             archive_file = pretrained_model_name_or_path
    #             is_local = True
    #         elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
    #             if not from_tf:
    #                 raise ValueError(
    #                     f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
    #                     "from_tf to True to load from this checkpoint."
    #                 )
    #             archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
    #             is_local = True
    #         elif is_remote_url(pretrained_model_name_or_path):
    #             filename = pretrained_model_name_or_path
    #             resolved_archive_file = download_url(pretrained_model_name_or_path)
    #         else:
    #             # set correct filename
    #             if from_tf:
    #                 filename = TF2_WEIGHTS_NAME
    #             elif from_flax:
    #                 filename = FLAX_WEIGHTS_NAME
    #             else:
    #                 filename = WEIGHTS_NAME

    #             try:
    #                 # Load from URL or cache if already cached
    #                 cached_file_kwargs = dict(
    #                     cache_dir=cache_dir,
    #                     force_download=force_download,
    #                     proxies=proxies,
    #                     resume_download=resume_download,
    #                     local_files_only=local_files_only,
    #                     use_auth_token=use_auth_token,
    #                     user_agent=user_agent,
    #                     revision=revision,
    #                     subfolder=subfolder,
    #                     _raise_exceptions_for_missing_entries=False,
    #                     _commit_hash=commit_hash,
    #                 )
    #                 resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

    #                 # Since we set _raise_exceptions_for_missing_entries=False, we don't get an expection but a None
    #                 # result when internet is up, the repo and revision exist, but the file does not.
    #                 if resolved_archive_file is None and filename == WEIGHTS_NAME:
    #                     # Maybe the checkpoint is sharded, we try to grab the index name in this case.
    #                     resolved_archive_file = cached_file(
    #                         pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
    #                     )
    #                     if resolved_archive_file is not None:
    #                         is_sharded = True
    #                 if resolved_archive_file is None:
    #                     # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
    #                     # message.
    #                     has_file_kwargs = {
    #                         "revision": revision,
    #                         "proxies": proxies,
    #                         "use_auth_token": use_auth_token,
    #                     }
    #                     if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
    #                         raise EnvironmentError(
    #                             f"{pretrained_model_name_or_path} does not appear to have a file named"
    #                             f" {WEIGHTS_NAME} but there is a file for TensorFlow weights. Use `from_tf=True` to"
    #                             " load this model from those weights."
    #                         )
    #                     elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
    #                         raise EnvironmentError(
    #                             f"{pretrained_model_name_or_path} does not appear to have a file named"
    #                             f" {WEIGHTS_NAME} but there is a file for Flax weights. Use `from_flax=True` to load"
    #                             " this model from those weights."
    #                         )
    #                     else:
    #                         raise EnvironmentError(
    #                             f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME},"
    #                             f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
    #                         )
    #             except EnvironmentError:
    #                 # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
    #                 # to the original exception.
    #                 raise
    #             except Exception:
    #                 # For any other exception, we throw a generic error.
    #                 raise EnvironmentError(
    #                     f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
    #                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
    #                     f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
    #                     f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
    #                     f" {FLAX_WEIGHTS_NAME}."
    #                 )

    #         if is_local:
    #             logger.info(f"loading weights file {archive_file}")
    #             resolved_archive_file = archive_file
    #         else:
    #             logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
    #     else:
    #         resolved_archive_file = None

    #     # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    #     if is_sharded:
    #         # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
    #         resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
    #             pretrained_model_name_or_path,
    #             resolved_archive_file,
    #             cache_dir=cache_dir,
    #             force_download=force_download,
    #             proxies=proxies,
    #             resume_download=resume_download,
    #             local_files_only=local_files_only,
    #             use_auth_token=use_auth_token,
    #             user_agent=user_agent,
    #             revision=revision,
    #             subfolder=subfolder,
    #             _commit_hash=commit_hash,
    #         )

    #     # load pt weights early so that we know which dtype to init the model under
    #     if from_pt:
    #         if not is_sharded and state_dict is None:
    #             # Time to load the checkpoint
    #             state_dict = load_state_dict(resolved_archive_file)

    #         # set dtype to instantiate the model under:
    #         # 1. If torch_dtype is not None, we use that dtype
    #         # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
    #         #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    #         # we also may have config.torch_dtype available, but we won't rely on it till v5
    #         dtype_orig = None
    #         if torch_dtype is not None:
    #             if isinstance(torch_dtype, str):
    #                 if torch_dtype == "auto":
    #                     if is_sharded and "dtype" in sharded_metadata:
    #                         torch_dtype = sharded_metadata["dtype"]
    #                     elif not is_sharded:
    #                         torch_dtype = get_state_dict_dtype(state_dict)
    #                     else:
    #                         one_state_dict = load_state_dict(resolved_archive_file[0])
    #                         torch_dtype = get_state_dict_dtype(one_state_dict)
    #                         del one_state_dict  # free CPU memory
    #                 else:
    #                     raise ValueError(
    #                         f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
    #                     )
    #             dtype_orig = cls._set_default_torch_dtype(torch_dtype)

    #         if is_sharded:
    #             loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
    #         else:
    #             loaded_state_dict_keys = [k for k in state_dict.keys()]
    #         if low_cpu_mem_usage:
    #             state_dict = None

    #     config.name_or_path = pretrained_model_name_or_path

    #     # Instantiate model.
    #     init_contexts = [no_init_weights(_enable=_fast_init)]

    #     if is_deepspeed_zero3_enabled():
    #         import deepspeed

    #         logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
    #         init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
    #     elif load_in_8bit or low_cpu_mem_usage:
    #         init_contexts.append(init_empty_weights())

    #     with ContextManagers(init_contexts):
    #         model = cls(config, *model_args, **model_kwargs)

    #     if load_in_8bit:
    #         from .utils.bitsandbytes import get_key_to_not_convert, replace_8bit_linear

    #         logger.info("Detected 8-bit loading: activating 8-bit loading for this model")

    #         # We never convert lm_head or any last modules for numerical stability reasons
    #         modules_to_not_convert = get_key_to_not_convert(model)
    #         model = replace_8bit_linear(model, threshold=int8_threshold, modules_to_not_convert=modules_to_not_convert)

    #     if isinstance(device_map, str):
    #         if model._no_split_modules is None:
    #             raise ValueError(f"{model.__class__.__name__} does not support `device_map='{device_map}'` yet.")
    #         no_split_modules = model._no_split_modules
    #         if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
    #             raise ValueError(
    #                 "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
    #                 "'sequential'."
    #             )
    #         elif device_map in ["balanced", "balanced_low_0"] and get_balanced_memory is None:
    #             raise ValueError(f"`device_map={device_map}` requires a source install of Accelerate.")
    #         if device_map != "sequential" and get_balanced_memory is not None:
    #             max_memory = get_balanced_memory(
    #                 model,
    #                 max_memory=max_memory,
    #                 no_split_module_classes=no_split_modules,
    #                 dtype=torch_dtype,
    #                 low_zero=(device_map == "balanced_low_0"),
    #             )
    #         # Make sure tied weights are tied before creating the device map.
    #         model.tie_weights()
    #         device_map = infer_auto_device_map(
    #             model,
    #             no_split_module_classes=no_split_modules,
    #             dtype=torch_dtype if not load_in_8bit else torch.int8,
    #             max_memory=max_memory,
    #         )

    #         if load_in_8bit:
    #             # The LM head can stay on disk / CPU
    #             device_map_without_lm_head = {
    #                 key: device_map[key] for key in device_map.keys() if key != modules_to_not_convert
    #             }
    #             if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
    #                 raise ValueError("8-bit operations on `bitsandbytes` are not supported under CPU!")
    #             del device_map_without_lm_head

    #     if from_tf:
    #         if resolved_archive_file.endswith(".index"):
    #             # Load from a TensorFlow 1.X checkpoint - provided by original authors
    #             model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
    #         else:
    #             # Load from our TensorFlow 2.0 checkpoints
    #             try:
    #                 from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

    #                 model, loading_info = load_tf2_checkpoint_in_pytorch_model(
    #                     model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
    #                 )
    #             except ImportError:
    #                 logger.error(
    #                     "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
    #                     " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
    #                     " instructions."
    #                 )
    #                 raise
    #     elif from_flax:
    #         try:
    #             from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

    #             model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
    #         except ImportError:
    #             logger.error(
    #                 "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
    #                 " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
    #                 " installation instructions."
    #             )
    #             raise
    #     elif from_pt:

    #         # restore default dtype
    #         if dtype_orig is not None:
    #             torch.set_default_dtype(dtype_orig)

    #         model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
    #             model,
    #             state_dict,
    #             loaded_state_dict_keys,  # XXX: rename?
    #             resolved_archive_file,
    #             pretrained_model_name_or_path,
    #             ignore_mismatched_sizes=ignore_mismatched_sizes,
    #             sharded_metadata=sharded_metadata,
    #             _fast_init=_fast_init,
    #             low_cpu_mem_usage=low_cpu_mem_usage,
    #             device_map=device_map,
    #             offload_folder=offload_folder,
    #             offload_state_dict=offload_state_dict,
    #             dtype=torch_dtype,
    #             load_in_8bit=load_in_8bit,
    #         )

    #     # make sure token embedding weights are still tied if needed
    #     model.tie_weights()

    #     # Set model in evaluation mode to deactivate DropOut modules by default
    #     model.eval()

    #     # Dispatch model with hooks on all devices if necessary
    #     if device_map is not None:
    #         dispatch_model(model, device_map=device_map, offload_dir=offload_folder)

    #     if output_loading_info:
    #         if loading_info is None:
    #             loading_info = {
    #                 "missing_keys": missing_keys,
    #                 "unexpected_keys": unexpected_keys,
    #                 "mismatched_keys": mismatched_keys,
    #                 "error_msgs": error_msgs,
    #             }
    #         return model, loading_info

    #     return model



# class ImgPreTrainedModel(PreTrainedModel):
#     """ Base class for all models. Handle loading/storing model config and
#         a simple interface for dowloading and loading pretrained models.
#     """

#     def __init__(self, config, *inputs, **kwargs):
#         super(ImgPreTrainedModel, self).__init__(config, *inputs, **kwargs)

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
#         r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

#             The model is set in evaluation mode by default using `model.eval()` (Dropout modules are desactivated)
#             To train the model, you should first set it back in training mode with `model.train()`

#         Params:
#             **pretrained_model_name_or_path**: either:
#                 - a string with the `shortcut name` of a pre-trained model to load from cache
#                     or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
#                 - a path to a `directory` containing a configuration file saved
#                     using the `save_pretrained(save_directory)` method.
#                 - a path or url to a tensorflow index checkpoint `file` (e.g. `./tf_model/model.ckpt.index`).
#                     In this case, ``from_tf`` should be set to True and a configuration object should be
#                     provided as `config` argument. This loading option is slower than converting the TensorFlow
#                     checkpoint in a PyTorch model using the provided conversion scripts and loading
#                     the PyTorch model afterwards.
#             **model_args**: (`optional`) Sequence:
#                 All remaning positional arguments will be passed to the underlying model's __init__ function
#             **config**: an optional configuration for the model to use instead of an automatically loaded configuation.
#                 Configuration can be automatically loaded when:
#                 - the model is a model provided by the library (loaded with a `shortcut name` of a pre-trained model), or
#                 - the model was saved using the `save_pretrained(save_directory)` (loaded by suppling the save directory).
#             **state_dict**: an optional state dictionnary for the model to use instead of a state dictionary loaded
#                 from saved weights file.
#                 This option can be used if you want to create a model from a pretrained configuraton but load your own weights.
#                 In this case though, you should check if using `save_pretrained(dir)` and `from_pretrained(save_directory)` is not
#                 a simpler option.
#             **cache_dir**: (`optional`) string:
#                 Path to a directory in which a downloaded pre-trained model
#                 configuration should be cached if the standard cache should not be used.
#             **output_loading_info**: (`optional`) boolean:
#                 Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
#             **kwargs**: (`optional`) dict:
#                 Dictionary of key, values to update the configuration object after loading.
#                 Can be used to override selected configuration parameters. E.g. ``output_attention=True``.

#                - If a configuration is provided with `config`, **kwargs will be directly passed
#                  to the underlying model's __init__ method.
#                - If a configuration is not provided, **kwargs will be first passed to the pretrained
#                  model configuration class loading function (`PretrainedConfig.from_pretrained`).
#                  Each key of **kwargs that corresponds to a configuration attribute
#                  will be used to override said attribute with the supplied **kwargs value.
#                  Remaining keys that do not correspond to any configuration attribute will
#                  be passed to the underlying model's __init__ function.

#         Examples::

#             >>> model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
#             >>> model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
#             >>> model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
#             >>> assert model.config.output_attention == True
#             >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
#             >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
#             >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

#         """
#         config = kwargs.pop('config', None)
#         state_dict = kwargs.pop('state_dict', None)
#         cache_dir = kwargs.pop('cache_dir', None)
#         from_tf = kwargs.pop('from_tf', False)
#         output_loading_info = kwargs.pop('output_loading_info', False)

#         # Load config
#         if config is None:
#             config, model_kwargs = cls.config_class.from_pretrained(
#                 pretrained_model_name_or_path, *model_args,
#                 cache_dir=cache_dir, return_unused_kwargs=True,
#                 **kwargs
#             )
#         else:
#             model_kwargs = kwargs

#         # Load model
#         if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
#             archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
#         elif os.path.isdir(pretrained_model_name_or_path):
#             if from_tf:
#                 # Directly load from a TensorFlow checkpoint
#                 archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
#             else:
#                 archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
#         else:
#             if from_tf:
#                 # Directly load from a TensorFlow checkpoint
#                 archive_file = pretrained_model_name_or_path + ".index"
#             else:
#                 archive_file = pretrained_model_name_or_path
#         # redirect to the cache, if necessary
#         try:
#             resolved_archive_file = cached_file(archive_file, cache_dir=cache_dir)
#         except EnvironmentError:
#             if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
#                 logger.error(
#                     "Couldn't reach server at '{}' to download pretrained weights.".format(
#                         archive_file))
#             else:
#                 logger.error(
#                     "Model name '{}' was not found in model name list ({}). "
#                     "We assumed '{}' was a path or url but couldn't find any file "
#                     "associated to this path or url.".format(
#                         pretrained_model_name_or_path,
#                         ', '.join(cls.pretrained_model_archive_map.keys()),
#                         archive_file))
#             return None
#         if resolved_archive_file == archive_file:
#             logger.info("loading weights file {}".format(archive_file))
#         else:
#             logger.info("loading weights file {} from cache at {}".format(
#                 archive_file, resolved_archive_file))

#         # Instantiate model.
#         model = cls(config, *model_args, **model_kwargs)

#         if state_dict is None and not from_tf:
#             state_dict = torch.load(resolved_archive_file, map_location='cpu')

#         if from_tf:
#             # Directly load from a TensorFlow checkpoint
#             return cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'

#         # Convert old format to new format if needed from a PyTorch state_dict
#         old_keys = []
#         new_keys = []
#         for key in state_dict.keys():
#             new_key = None
#             if 'gamma' in key:
#                 new_key = key.replace('gamma', 'weight')
#             if 'beta' in key:
#                 new_key = key.replace('beta', 'bias')
#             if new_key:
#                 old_keys.append(key)
#                 new_keys.append(new_key)
#         for old_key, new_key in zip(old_keys, new_keys):
#             state_dict[new_key] = state_dict.pop(old_key)

#         # Load from a PyTorch state_dict
#         missing_keys = []
#         unexpected_keys = []
#         error_msgs = []
#         # copy state_dict so _load_from_state_dict can modify it
#         metadata = getattr(state_dict, '_metadata', None)
#         state_dict = state_dict.copy()
#         if metadata is not None:
#             state_dict._metadata = metadata

#         def load(module, prefix=''):
#             local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#             module._load_from_state_dict(
#                 state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
#             for name, child in module._modules.items():
#                 if child is not None:
#                     load(child, prefix + name + '.')

#         # Make sure we are able to load base models as well as derived models (with heads)
#         start_prefix = ''
#         model_to_load = model
#         if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#             start_prefix = cls.base_model_prefix + '.'
#         if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#             model_to_load = getattr(model, cls.base_model_prefix)

#         load(model_to_load, prefix=start_prefix)
#         if len(missing_keys) > 0:
#             logger.info("Weights of {} not initialized from pretrained model: {}".format(
#                 model.__class__.__name__, missing_keys))
#         if len(unexpected_keys) > 0:
#             logger.info("Weights from pretrained model not used in {}: {}".format(
#                 model.__class__.__name__, unexpected_keys))
#         if len(error_msgs) == 2 and "size mismatch for cls.seq_relationship.weight" in error_msgs[0]:
#             logger.info('Error(s) in loading state_dict for {}:\n\t{}'.format(
#                 model.__class__.__name__, "\n\t".join(error_msgs)))
#         elif len(error_msgs) > 0:
#             raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
#                                model.__class__.__name__, "\n\t".join(error_msgs)))

#         if hasattr(model, 'tie_weights'):
#             model.tie_weights()  # make sure word embedding weights are still tied

#         # Set model in evaluation mode to desactivate DropOut modules by default
#         model.eval()

#         if output_loading_info:
#             loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
#             return model, loading_info

#         return model