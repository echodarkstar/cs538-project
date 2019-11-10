import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F


from transformers import RobertaForSequenceClassification, RobertaTokenizer

def get_predictions(probs):
    inferences = ['Contradiction', 'Neutral', 'Entailment']

    confidence, pred = torch.max(probs, 1)

    print(list(zip(confidence, pred, [inferences[k] for k in pred])))

    return confidence, pred

def infer(tokenizer, model, batch_of_pairs, labels):

    batch_inputs =  [torch.tensor(tokenizer.encode(*pair,add_special_tokens=True)) for pair in batch_of_pairs]
    
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value = -1)
    mask = torch.zeros_like(padded_batch)
    mask[padded_batch!=-1] = 1
    padded_batch[padded_batch==-1] = 0

    #Masking isn't happening properly. Wait for PR to be accepted.
    #https://github.com/huggingface/transformers/issues/1761

    # import pdb
    # pdb.set_trace()

    outputs = model(padded_batch, labels=labels, attention_mask = mask)
    loss, logits = outputs[:2]
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)

    return probs
    
    import pdb
    pdb.set_trace()

def run():
    parser = ArgumentParser()

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    # model.to('cuda')
    model.eval()
    batch_of_pairs = [
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
        ['potatoes are awesome.', 'I like to run.'],
        ['Mars is very far from earth.', 'Mars is very close.'],
    ]

    labels = torch.tensor([[0], [2], [1], [0]])

    # batch_of_pairs = [["Roberta is a heavily optimized version of BERT", "Roberta is not very optimized"]]
    # labels = torch.tensor([0]).unsqueeze(0)
    probs = infer(tokenizer, model, batch_of_pairs, labels)
    confidence, pred = get_predictions(probs)

if __name__=='__main__':
    run()