
# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class InferenceIterator(object):
    def __init__(self, data, batch_size, shuffle=False, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    @staticmethod
    def pad_data(batch_data):
        batch_text_indices = []
        batch_text_mask = []

        max_len = max([len(t['text_indices']) for t in batch_data])
        for item in batch_data:
            text_indices = item['text_indices']
            # 0-padding because 0 stands for 'O'
            text_padding = [0] * (max_len - len(text_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_text_mask.append([1] * len(text_indices) + text_padding)

        return {
                'text_indices': torch.tensor(batch_text_indices),
                'text_mask': torch.tensor(batch_text_mask, dtype=torch.bool),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
