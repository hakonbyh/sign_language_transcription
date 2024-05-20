import logging
from itertools import groupby

import torch
from torch.nn.utils.rnn import pad_sequence

from .unpad import unpad_padded


class Decode(object):
    def __init__(self, tokenizer, blank_id):
        self.tokenizer = tokenizer
        self.blank_id = blank_id

    def decode_targets(self, tokenized_targets):
        decoded_sentences = []
        for target in tokenized_targets:
            decoded_tokens = [
                self.tokenizer.decode([int(gloss_id)]) for gloss_id in target
            ]
            decoded_sentence = "".join(
                [
                    word[2:] if word.startswith("##") else " " + word
                    for word in decoded_tokens
                ]
            ).strip()
            decoded_sentences.append(decoded_sentence)
        return decoded_sentences

    def decode(self, nn_output):
        sequence_lengths = list(map(len, nn_output))

        nn_output = pad_sequence(nn_output, True)
        logging.info(f"nn_output: {nn_output}")
        index_list = torch.argmax(nn_output, axis=2)
        logging.info(f"index_list: {index_list}")
        batchsize, lgt = index_list.shape
        index_list = unpad_padded(index_list, sequence_lengths)
        logging.info(f"unpadded index_list: {index_list}")
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            logging.info(f"filtered: {filtered}")
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            decoded_tokens = [
                self.tokenizer.decode([int(gloss_id)]) for gloss_id in max_result
            ]
            decoded_sentence = "".join(
                [
                    word[2:] if word.startswith("##") else " " + word
                    for word in decoded_tokens
                ]
            ).strip()
            ret_list.append(decoded_sentence)
        return ret_list


class BeamSearchDecoder:
    def __init__(self, blank_id, beam_width=10):
        self.blank_id = blank_id
        self.beam_width = beam_width

    def _get_next_top_k(self, probs, sequences, beam_width):
        next_top_probs, next_top_idx = probs.topk(beam_width, dim=1)
        next_top_log_probs = torch.log(next_top_probs)

        new_sequences = []
        for idx in range(len(sequences)):
            current_seq = sequences[idx]
            for k in range(beam_width):
                new_seq = current_seq + [next_top_idx[idx][k].item()]
                new_sequences.append((new_seq, next_top_log_probs[idx][k].item()))

        return new_sequences

    def decode(self, probs):
        sequences = [([], 0.0) for _ in range(self.beam_width)]

        for t in range(probs.size(0)):
            all_candidates = []
            for seq, score in sequences:
                seq_prob = probs[t]
                seq_score = score + torch.log(seq_prob)
                candidates = self._get_next_top_k(seq_score, [seq], self.beam_width)
                all_candidates.extend(candidates)

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

            sequences = ordered[: self.beam_width]

        return sequences[0][0]

    def batch_decode(self, nn_output):
        results = []
        for log_probs in nn_output:
            decoded_seq = self.decode(torch.exp(log_probs))
            results.append(decoded_seq)
        return results
