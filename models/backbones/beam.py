"""
Manage beam search info structure.
Heavily borrowed from OpenNMT-py.
For code in OpenNMT-py, please check the following link (maybe in oldest version):
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch

class Constants():
    def __init__(self):
        self.PAD_WORD = '<|endoftext|>'
        self.UNK_WORD = '<|endoftext|>'
        self.BOS_WORD = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        self.EOS_WORD = "<|endoftext|>"

    @classmethod
    def from_tokenizer(cls, tokenizer):  # cls=Constance
        instance = cls()
        vocab = tokenizer.get_vocab()  # Accessing vocabulary

        # Using 'convert_tokens_to_ids' to get token IDs
        instance.PAD = vocab.get(instance.EOS_WORD, tokenizer.eos_token_id)
        instance.UNK = vocab.get(instance.UNK_WORD, tokenizer.unk_token_id)
        instance.BOS = vocab.get(instance.BOS_WORD, tokenizer.bos_token_id)
        instance.EOS = vocab.get(instance.EOS_WORD, tokenizer.eos_token_id)
        # print(instance.PAD, instance.UNK, instance.BOS, instance.EOS)

        return instance

class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False, tokenizer=None, prompt=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)  # token index

        self.size = size  # beam size
        self._done = False  # end symbol
        # The score for each interface on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)  # each beam score
        self.all_scores = []  # each step beam score
        self.max_length = 50

        if prompt == None:
            # The backpointers at each time-step.
            self.prev_ks = []  # recod the best score
            # The outputs at each time-step.
            self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]
        else:
            self.next_ys = []
            self.prev_ks = []
            for item in prompt:
                self.prev_ks.append(list(range(size)))
                self.next_ys.append(torch.full((size, ), item, dtype=torch.long, device=device))
            self.prev_ks = self.prev_ks[:-1]
    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob, word_length=None):

        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.constants.EOS:
            self._done = True
        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))  # The sequence of words obtained by backtracking is in reverse order, turning it into an order
