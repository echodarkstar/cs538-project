import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import importlib
interact = importlib.import_module("transfer-learning-conv-ai.interact")

def get_predictions(probs):
    inferences = ['Contradiction', 'Neutral', 'Entailment']

    confidence, pred = torch.max(probs, 1)

    print(list(zip(confidence, pred, [inferences[k] for k in pred])))

    return confidence, pred

def infer(tokenizer, model, batch_of_pairs, labels, device):

    batch_inputs =  [torch.tensor(tokenizer.encode(*pair,add_special_tokens=True)) for pair in batch_of_pairs]
    
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value = -1).to(device)
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

def run_inference():
    batch_of_pairs = [
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
        ['potatoes are awesome.', 'I like to run.'],
        ['Mars is very far from earth.', 'Mars is very close.'],
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    labels = torch.tensor([[0], [2], [1], [0]]).to(device)
    # batch_of_pairs = [["Roberta is a heavily optimized version of BERT", "Roberta is not very optimized"]]
    # labels = torch.tensor([0]).unsqueeze(0)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    model.to(device) 
    model.eval()

    probs = infer(tokenizer, model, batch_of_pairs, labels, device)
    confidence, pred = get_predictions(probs)

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = interact.download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (interact.GPT2Tokenizer, interact.GPT2LMHeadModel) if args.model == 'gpt2' else (interact.OpenAIGPTTokenizer, interact.OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    interact.add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = interact.get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    personality_sentences = tokenizer.decode(chain(*personality)).split('. ')
    history = []

    #Inference model
    inf_tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    inf_model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    inf_model.to(args.device) 
    inf_model.eval()


    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = interact.sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)

        #Each pair is (personality, output). Instead of having a batch, find sentence similarity using some distance metric
        #to find most relevant personality pair out of all
        batch_of_pairs = [[personality_sent, out_text] for personality_sent in personality_sentences]

        labels = torch.tensor([[0], [2], [1], [0], [0]]).to(args.device)
        probs = infer(inf_tokenizer, inf_model, batch_of_pairs, labels, args.device)
        confidence, pred = get_predictions(probs)

if __name__=='__main__':
    run()