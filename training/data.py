import torch
import jsonlines
from math import ceil
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Chameleon/Anole special tokens
PAD_TOKEN = 1
SEP_TOKEN = 8710
IMAGE_START_TOKEN = 8197
IMAGE_END_TOKEN = 8196


class DistillDataset(Dataset):
    """Pairs each task sample with an image-generation sample used for distillation.

    The image-generation stream is looped or truncated so that every task sample is
    paired with one image-generation sample.
    """

    def __init__(self, task_filepath, img_gen_filepath, max_length=2048):
        self.task_data = []
        with jsonlines.open(task_filepath) as reader:
            for obj in reader:
                self.task_data.append(torch.tensor(obj['tokens'], dtype=torch.long)[:max_length])
        self.task_len = len(self.task_data)

        self.img_gen_data = []
        with jsonlines.open(img_gen_filepath) as reader:
            for obj in reader:
                self.img_gen_data.append(torch.tensor(obj['image_tokens'], dtype=torch.long)[:max_length])

        if len(self.img_gen_data) > self.task_len:
            self.img_gen_data = self.img_gen_data[:self.task_len]
        elif len(self.img_gen_data) < self.task_len:
            repeats = ceil(self.task_len / len(self.img_gen_data))
            self.img_gen_data = (self.img_gen_data * repeats)[:self.task_len]

    def __len__(self):
        return self.task_len

    def __getitem__(self, idx):
        return {
            'tokens': self.task_data[idx],
            'img_gen_data': self.img_gen_data[idx],
        },


def _build_labels(batch_inputs):
    """Mask everything up to and including the separator so loss is on the answer only."""
    labels = list(batch_inputs)
    if len((labels[0] == SEP_TOKEN).nonzero()) != 0:
        new_labels = []
        for label in labels:
            loc_sep = (label == SEP_TOKEN).nonzero()[0]
            new_labels.append(
                torch.cat([torch.tensor([-100] * (loc_sep + 1)), label[loc_sep + 1:]], dim=-1)
            )
        labels = new_labels
    return pad_sequence(labels, batch_first=True, padding_value=-100)


def _modality_ids(batch_inputs_padded):
    """Per-token modality: 1 inside an image span (inclusive of markers), 0 for text."""
    modality_ids_list = []
    for seq in batch_inputs_padded:
        mod_ids = torch.zeros_like(seq)
        in_image = False
        for idx, token in enumerate(seq):
            token_val = token.item()
            if token_val == IMAGE_START_TOKEN:
                in_image = True
                mod_ids[idx] = 1
            elif token_val == IMAGE_END_TOKEN:
                mod_ids[idx] = 1
                in_image = False
            elif in_image:
                mod_ids[idx] = 1
            else:
                mod_ids[idx] = 0
        modality_ids_list.append(mod_ids)
    return torch.stack(modality_ids_list, dim=0)


def distill_collate_fn(batch, task_id=-1):
    batch_inputs = [item[0]['tokens'] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=PAD_TOKEN)

    labels = _build_labels(batch_inputs)

    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != PAD_TOKEN, 1)

    modality_ids = _modality_ids(batch_inputs_padded)

    img_gen_tokens = [item[0]['img_gen_data'] for item in batch]
    img_gen_tokens_padded = pad_sequence(img_gen_tokens, batch_first=True, padding_value=PAD_TOKEN)

    img_gen_attention_masks = torch.zeros_like(img_gen_tokens_padded, dtype=torch.long)
    img_gen_attention_masks = img_gen_attention_masks.masked_fill(img_gen_tokens_padded != PAD_TOKEN, 1)

    img_gen_token_type_ids = torch.ones_like(img_gen_tokens_padded)

    return {
        'input_ids': batch_inputs_padded,
        'attention_mask': attention_masks,
        'labels': labels,
        'task_id': task_id,
        'token_type_ids': modality_ids,
        'img_gen_input_ids': img_gen_tokens_padded,
        'img_gen_attention_mask': img_gen_attention_masks,
        'img_gen_token_type_ids': img_gen_token_type_ids,
    }
