# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from inference_iterator import InferenceIterator
from data_utils import ABSADataReader, build_tokenizer, build_embedding_matrix, ABSADataReaderInference
from models import CMLA, HAST, OTE
import boto3
import csv


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
       
        absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
        self.tokenizer = build_tokenizer(data_dir=opt.data_dir)
        embedding_matrix = build_embedding_matrix(opt.data_dir, self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.idx2tag, self.idx2polarity = absa_data_reader.reverse_tag_map, absa_data_reader.reverse_polarity_map
        self.model = opt.model_class(embedding_matrix, opt, self.idx2tag, self.idx2polarity).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        # self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()

        # get a handle on s3
        session = boto3.Session(
            aws_access_key_id='XXXXXXXXXXXX',
            aws_secret_access_key='XXXXXXXX',
            region_name='XXXXXXXX')

        self.s3 = session.resource('s3')
        self.bucket = self.s3.Bucket('surveybuddy-responses')  # example: energy_market_procesing

        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        text_indices = self.tokenizer.text_to_sequence(text)
        text_mask = [1] * len(text_indices)
        t_sample_batched = {
            'text_indices': torch.tensor([text_indices]),
            'text_mask': torch.tensor([text_mask], dtype=torch.uint8),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
        
        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred]

    def evaluateList(self,textList):
        absa_data_reader = ABSADataReaderInference(data_dir='textResponses.xlsx', textList=textList, bucket=self.bucket);
        dataset = absa_data_reader.get_dataset(self.tokenizer)
        self.inference_data_loader = InferenceIterator(data=dataset, batch_size=opt.batch_size)

        t_triplets_pred_all = None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.inference_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                _, _, t_triplets_pred = self.model.inference(t_inputs)

                if t_triplets_pred_all is None:

                    # t_ap_spans_pred_all = t_ap_spans_pred
                    # t_op_spans_pred_all = t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred
                else:

                    # t_ap_spans_pred_all = t_ap_spans_pred_all + t_ap_spans_pred
                    # t_op_spans_pred_all = t_op_spans_pred_all + t_op_spans_pred
                    t_triplets_pred_all = t_triplets_pred_all + t_triplets_pred


        return t_triplets_pred


if __name__ == '__main__':
    dataset = 'rest16'
    # set your trained models here
    model_state_dict_paths = {
        'ote': 'state_dict/ote_'+dataset+'.pkl',
    }
    model_classes = {
        'ote': OTE,
    }
    input_colses = {
        'ote': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
    }
    data_dirs = {
        'laptop14': 'datasets/14lap',
        'rest14': 'datasets/14rest',
        'rest15': 'datasets/15rest',
        'rest16': 'datasets/16rest',
    }
    class Option(object): pass
    opt = Option()
    opt.dataset = dataset
    opt.model_name = 'ote'
    opt.eval_cols = ['ap_spans', 'op_spans','triplets']
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.target_cols = target_colses[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 4
    opt.batch_size = 32
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)

    #rest
    #text = 'Great food but the service was dreadful !'
    # text = 'the atmosphere is attractive , but a little uncomfortable .'
    #laptop
    # text = 'I am pleased with the fast log on , speedy WiFi connection and the long battery life ( > 6 hrs ) .'

    textList = [
        'Great food but the service was dreadful !',
        'the atmosphere is attractive , but a little uncomfortable .'
    ]
    tripletsList = inf.evaluateList(textList)

    for (text,triplets) in zip(textList,tripletsList):
        words = text.split()
        polarity_map = {0:'N', 1:'NEU', 2:'NEG', 3:'POS'}
        for triplet in triplets:
            ap_beg, ap_end, op_beg, op_end, p = triplet
            ap = ' '.join(words[ap_beg:ap_end+1])
            op = ' '.join(words[op_beg:op_end+1])
            polarity = polarity_map[p]
            print(f'{ap}, {op}, {polarity}')

