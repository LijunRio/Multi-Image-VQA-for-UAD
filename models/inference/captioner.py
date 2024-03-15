import datetime

import torch
from dateutil import tz
from pytorch_lightning import LightningModule
from transformers import GPT2Tokenizer

from constant import *
from models.backbones.beam import Beam

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Captioner(LightningModule):
    def __init__(self,
                 model,
                 prompt,
                 max_words=100,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_words = max_words
        self.beam_size = kwargs['beam_size']
        self.image_type = kwargs['image_type']
        self.model_type = kwargs['model_type']
        now = datetime.datetime.now(tz.tzlocal())
        extension = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.hyp_path = os.path.join(VQA_GEN_DIR, f"{extension}_res.txt")

    def share_step(self, batch):
        self.model.eval()
        all_result_lists = []
        all_caption_lists = []
        all_question_lists = []
        all_filename_lists = []
        img0 = batch['img0']
        img1 = batch['img1']
        img2 = batch['img2']
        output_caption_ids = batch['caption']
        prompts = batch['prompt']
        file_name = batch['filename']
        prompt_ids = [self.tokenizer.encode(pt) for pt in prompts]

        bs = len(img0)

        with torch.no_grad():
            qfeat = self.model.encode(img0, img1, img2)
            device = img0.device
            if self.model_type =='KQFormer_concat':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)
            elif self.model_type == 'MI_average_vit' or self.model_type == 'MI_channel_vit':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 197, 768) # VIT
                encoder_mask = torch.ones((bs * self.beam_size, 197)).to(device)
            elif self.model_type == 'MI_average_res'or self.model_type == 'MI_channel_res':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 362, 768) # ResNet
                encoder_mask = torch.ones((bs * self.beam_size, 362)).to(device)
            elif self.model_type == 'MI_concat': # concate
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)
            elif self.model_type =='KQFormer_channel':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)

            else:
                raise ValueError('please enter a valid model type!')


            # -- Prepare beams
            inst_dec_beams = [Beam(self.beam_size, device=device, tokenizer=self.tokenizer, prompt=prompt_ids[idx]) for idx in range(bs)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(bs))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(len(prompt_ids[0]), self.max_words + 1):
                active_inst_idx_list = beam_decode_step(self.model, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, self.beam_size, device,
                                                        qfeat, encoder_mask)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                qfeat, encoder_mask, inst_idx_to_position_map = collate_active_info(qfeat, encoder_mask, inst_idx_to_position_map, active_inst_idx_list, self.beam_size, device)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_list = [batch_hyp[i][0] for i in range(bs)]
        caption_list = output_caption_ids.cpu().detach().numpy()


        result_ids = result_list[0][len(prompt_ids[0])-1:] # [[0,1,23,...]]
        decoded_res = self.tokenizer.decode(result_ids, skip_special_tokens=True)
        gt_ids = caption_list[0] # [[0,1,23,...]]
        decoded_gt = self.tokenizer.decode(gt_ids, skip_special_tokens=True)


        all_result_lists.append(decoded_res)
        all_caption_lists.append(decoded_gt)
        all_question_lists.append(prompts[0])
        all_filename_lists.append(file_name[0])

            # Save pure results
        with open(self.hyp_path, "a", encoding='utf-8') as writer:
            for i in range(len(all_result_lists)):
                name = all_filename_lists[i]
                Q = all_question_lists[i]
                A_p = all_result_lists[i]
                A_gt = all_caption_lists[i]
                writer.write(name + '\t' + Q + '\t' + A_gt + '\t' + A_p + '\n')
        return {'res': all_result_lists, 'ref': all_caption_lists}

    def test_step(self, batch, batch_idx):
        return self.share_step(batch)
    # def test_step(self, batch, batch_idx):
    #     res =  self.share_step(batch)
    #     return res


    def forward(self, batch):
        self.model.eval()
        all_result_lists = []
        all_caption_lists = []
        all_question_lists = []
        all_filename_lists = []
        img0 = batch['img0']
        img1 = batch['img1']
        img2 = batch['img2']
        output_caption_ids = batch['caption']
        prompts = batch['prompt']
        file_name = batch['filename']
        prompt_ids = [self.tokenizer.encode(pt) for pt in prompts]

        bs = len(img0)

        with torch.no_grad():
            qfeat = self.model.encode(img0, img1, img2)
            device = img0.device
            if self.model_type =='KQFormer_concat':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)
            elif self.model_type == 'MI_average_vit' or self.model_type == 'MI_channel_vit':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 197, 768) # VIT
                encoder_mask = torch.ones((bs * self.beam_size, 197)).to(device)
            elif self.model_type == 'MI_average_res'or self.model_type == 'MI_channel_res':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 362, 768) # ResNet
                encoder_mask = torch.ones((bs * self.beam_size, 362)).to(device)
            elif self.model_type == 'MI_concat': # concate
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)
            elif self.model_type =='KQFormer_channel':
                qfeat = qfeat.repeat(1, self.beam_size, 1).view(bs * self.beam_size, 96, 768)
                encoder_mask = torch.ones((bs * self.beam_size, 96)).to(device)

            else:
                raise ValueError('please enter a valid model type!')


            # -- Prepare beams
            inst_dec_beams = [Beam(self.beam_size, device=device, tokenizer=self.tokenizer, prompt=prompt_ids[idx]) for idx in range(bs)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(bs))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(len(prompt_ids[0]), self.max_words + 1):
                active_inst_idx_list = beam_decode_step(self.model, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, self.beam_size, device,
                                                        qfeat, encoder_mask)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                qfeat, encoder_mask, inst_idx_to_position_map = collate_active_info(qfeat, encoder_mask, inst_idx_to_position_map, active_inst_idx_list, self.beam_size, device)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_list = [batch_hyp[i][0] for i in range(bs)]
        caption_list = output_caption_ids.cpu().detach().numpy()


        result_ids = result_list[0][len(prompt_ids[0])-1:] # [[0,1,23,...]]
        decoded_res = self.tokenizer.decode(result_ids, skip_special_tokens=True)
        gt_ids = caption_list[0] # [[0,1,23,...]]
        decoded_gt = self.tokenizer.decode(gt_ids, skip_special_tokens=True)


        all_result_lists.append(decoded_res)
        all_caption_lists.append(decoded_gt)
        all_question_lists.append(prompts[0])
        all_filename_lists.append(file_name[0])

            # Save pure results
        with open(self.hyp_path, "a", encoding='utf-8') as writer:
            for i in range(len(all_result_lists)):
                name = all_filename_lists[i]
                Q = all_question_lists[i]
                A_p = all_result_lists[i]
                A_gt = all_caption_lists[i]
                writer.write(name + '\t' + Q + '\t' + A_gt + '\t' + A_p + '\n')
        return {'res': all_result_lists, 'ref': all_caption_lists}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def beam_decode_step(model, inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm, device, encoder_output, encoder_mask, decoder_length=None):


    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, encoder_output, encoder_mask):
        # next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)
        next_decoder_ids = next_decoder_ids.view(-1, next_decoder_ids.shape[-1])
        # next_decoder_mask = next_decoder_mask.view(-1, next_decoder_mask.shape[-1])
        dec_output = model.decode(input_ids=next_decoder_ids, encoder_output=encoder_output)
        dec_output = dec_output['logits'][:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, encoder_output, encoder_mask)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collate_active_info(encoder_output, encoder_mask, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)  # batch size
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_encoder_output = collect_active_part(encoder_output, active_inst_idx, n_prev_active_inst, n_bm)
    active_encoder_mask = collect_active_part(encoder_mask, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return active_encoder_output, active_encoder_mask, active_inst_idx_to_position_map

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores