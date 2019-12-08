import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from pprint import pprint
import warnings
import pickle

import torch
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import importlib
from copy import deepcopy
import numpy as np

interact = importlib.import_module("transfer-learning-conv-ai.interact")
attacks = importlib.import_module("universal-triggers.attacks")
trigger_utils = importlib.import_module("universal-triggers.utils")

######### TRIGGER RELATED FUNCTIONS #################
#From create_adv_token.py in universal-triggers

# returns the wordpiece embedding weight matrix
def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 40483: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 40483: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(trigger_utils.extract_grad_hook)

# Gets the loss of the target_tokens using the triggers as the context
def get_loss(language_model, batch_size, trigger, target, device='cuda'):
    # context is trigger repeated batch size
    tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    mask_out = -1 * torch.ones_like(tensor_trigger) # we zero out the loss for the trigger tokens
    lm_input = torch.cat((tensor_trigger, target), dim=1) # we feed the model the trigger + target texts
    mask_and_target = torch.cat((mask_out, target), dim=1) # has -1's + target texts for loss computation
    lm_input[lm_input == -1] = 1   # put random token of 1 at end of context (its masked out)
    loss = language_model(lm_input, labels=mask_and_target)[0]
    return loss

# creates the batch of target texts with -1 placed at the end of the sequences for padding (for masking out the loss).
def make_target_batch(tokenizer, device, target_texts):
    # encode items and get the max length
    encoded_texts = []
    max_len = 0
    for target_text in target_texts:
        encoded_target_text = tokenizer.encode(target_text)
        encoded_texts.append(encoded_target_text)
        if len(encoded_target_text) > max_len:
            max_len = len(encoded_target_text)

    # pad tokens, i.e., append -1 to the end of the non-longest ones
    for indx, encoded_text in enumerate(encoded_texts):
        if len(encoded_text) < max_len:
            encoded_texts[indx].extend([-1] * (max_len - len(encoded_text)))

    # convert to tensors and batch them up
    target_tokens_batch = None
    for encoded_text in encoded_texts:
        target_tokens = torch.tensor(encoded_text, device=device, dtype=torch.long).unsqueeze(0)
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
        else:
            target_tokens_batch = torch.cat((target_tokens, target_tokens_batch), dim=0)
    return target_tokens_batch

######### INFERENCE RELATED FUNCTIONS #################

def get_predictions(probs):
    #For roberta-mnli, the order of inferences is ['Contradiction','Neutral','Entailment']
    #If you're using mnli, change this accordingly while printing results as well
    inferences = ['Contradiction', 'Entailment', 'Neutral']
    confidence, pred = torch.max(probs, 1)
    #print(pred)
    #print(list(zip(confidence, pred, [inferences[k] for k in pred])))

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
    
def run_inference(sentence1,sentence2):
    batch_of_pairs = [
        [sentence1,sentence2]
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = torch.zeros([len(batch_of_pairs)],dtype=torch.long).to(device)
    # batch_of_pairs = [["Roberta is a heavily optimized version of BERT", "Roberta is not very optimized"]]
    # labels = torch.tensor([0]).unsqueeze(0)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-dnli')
    model = RobertaForSequenceClassification.from_pretrained('roberta-dnli')

    model.to(device)
    model.eval()

    probs = infer(tokenizer, model, batch_of_pairs, labels, device)
    confidence, pred = get_predictions(probs)
    class_mapping = {0:'Contradiction',1:'Entailment',2:'Neutral'}
    print("Confidence: "+str(confidence.item()))
    print("Predicted Class: "+str(class_mapping[pred.item()]))
    print("Sentence 1: "+sentence1)
    print("Sentence 2: "+sentence2)

########## LOGGING UTIL #################################

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

######### FUNCTION TO RUN EVERYTHING #################

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)",
                        choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--create_trigger", action='store_true', help="Creates triggers")
    parser.add_argument("--split_output", action='store_true', help="Splits output to check inference")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of times a dialogue instance happens (each dialogue ends on breakage). This is for consistency evaluation.")
    parser.add_argument("--max_utterance", type=int, default=20, help="Max number of utterances in a dialogue. End dialogue if this (or breakage) is reached, whichever happens first")
    parser.add_argument("--log_file", type=str, default="output", help="Name of output log")
    parser.add_argument("--query_type", type=str, default="basic", help="Type of input that is fed to the model")
    parser.add_argument("--run_inference", type=bool, default=False, help="Run inference flag")
    parser.add_argument("--sentence1", type=str, default="", help="Sentence1 to run inference on")  
    parser.add_argument("--sentence2", type=str, default="", help="Sentence2 to run inference on")  
    args = parser.parse_args()

    
    logger = setup_logger('conversation_logger', args.log_file + ".log")
    cont_logger = setup_logger('cont_logger', args.log_file + "_cont.log")
    # ent_logger = setup_logger('ent_logger', args.log_file + "_ent.log")
    # neut_logger = setup_logger('neut_logger', args.log_file + "_neut.log")
    logger.info(pformat(args))
    if args.run_inference == True:
        run_inference(args.sentence1,args.sentence2)
    else:
        if args.model_checkpoint == "":
            if args.model == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                args.model_checkpoint = interact.download_pretrained_model()

        #This seeding makes sense for reproducibility
        #But if we want to make repeated trials and calculate some aggreate measure
        #then randomness should be preserved
        #this means personality sampling will also happen randomly which is nice
        #Only seed when you aren't conducting consistency evaluation
        if args.num_trials == 1:
            random.seed(args.seed)
            torch.random.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (interact.GPT2Tokenizer, interact.GPT2LMHeadModel) if args.model == 'gpt2' else (interact.OpenAIGPTTokenizer, interact.OpenAIGPTLMHeadModel)
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        model = model_class.from_pretrained(args.model_checkpoint)
        model.eval()
        model.to(args.device)
        interact.add_special_tokens_(model, tokenizer)

        add_hooks(model) # add gradient hooks to embeddings
        embedding_weight = get_embedding_weight(model) # save the word embedding matrix

        logger.info("Sample a personality")
        dataset = interact.get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        personality = random.choice(personalities)
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
        personality_sentences = tokenizer.decode(chain(*personality)).split('. ')
        history = []

        #Target text should be contradictions to the personality sentences
        #For now it is hardcoded.
        target_texts = [
            "i hate vinyl records",
            "vinyl records are the worst",
            "i am still",
            "i am very lazy"
            "i do not fix airplanes",
            "i can't fix the world",
            "the world cannot be fixed"
        ]

        # batch and pad the target tokens
        target_tokens = make_target_batch(tokenizer, args.device, target_texts)

        #Inference model
        inf_tokenizer = RobertaTokenizer.from_pretrained('roberta-dnli')
        inf_model = RobertaForSequenceClassification.from_pretrained('roberta-dnli')

        inf_model.to(args.device) 
        inf_model.eval()

        # import pdb
        # pdb.set_trace()

    if args.create_trigger:
        for _ in range(10): # different random restarts of the trigger
            total_vocab_size = 40483  # total number of subword pieces in the GPT-2 model
            trigger_token_length = 6  # how many subword pieces in the trigger
            batch_size = target_tokens.shape[0]

            # sample random initial trigger
            trigger_tokens = np.random.randint(total_vocab_size, size=trigger_token_length)
            print(tokenizer.decode(trigger_tokens))

            # get initial loss for the trigger
            model.zero_grad()
            loss = get_loss(model, batch_size, trigger_tokens, target_tokens, args.device)
            best_loss = loss
            counter = 0
            end_iter = False
            for _ in range(50):  # this many updates of the entire trigger sequence
                for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                    if end_iter:  # no loss improvement over whole sweep -> continue to new random restart
                        continue

                    # Get average gradient w.r.t. the triggers
                    loss.backward()
                    averaged_grad = torch.sum(trigger_utils.extracted_grads[0], dim=0)
                    # import pdb
                    # pdb.set_trace()
                    averaged_grad = averaged_grad[token_to_flip].unsqueeze(0)

                    # Use hotflip (linear approximation) attack to get the top num_candidates
                    candidates = attacks.hotflip_attack(averaged_grad, embedding_weight,
                                                        [trigger_tokens[token_to_flip]], 
                                                        increase_loss=False, num_candidates=100)[0]

                    # try all the candidates and pick the best
                    curr_best_loss = 999999
                    curr_best_trigger_tokens = None
                    for cand in candidates:
                        # replace one token with new candidate
                        candidate_trigger_tokens = deepcopy(trigger_tokens)
                        candidate_trigger_tokens[token_to_flip] = cand

                        # get loss, update current best if its lower loss
                        curr_loss = get_loss(model, batch_size, candidate_trigger_tokens,
                                            target_tokens, args.device)
                        if curr_loss < curr_best_loss:
                            curr_best_loss = curr_loss
                            curr_best_trigger_tokens = deepcopy(candidate_trigger_tokens)

                    # Update overall best if the best current candidate is better
                    if curr_best_loss < best_loss:
                        counter = 0 # used to exit early if no improvements in the trigger
                        best_loss = curr_best_loss
                        trigger_tokens = deepcopy(curr_best_trigger_tokens)
                        print("Loss: " + str(best_loss.data.item()))
                        print(tokenizer.decode(trigger_tokens) + '\n')

                    # if you have gone through all trigger_tokens without improvement, end iteration
                    elif counter == len(trigger_tokens):
                        print("\nNo improvement, ending iteration")
                        end_iter = True
                    # If the loss didn't get better, just move to the next word.
                    else:
                        counter = counter + 1

                    # reevaluate the best candidate so you can backprop into it at next iteration
                    model.zero_grad()
                    loss = get_loss(model, batch_size, trigger_tokens, target_tokens, args.device)

            # Print final trigger and get 10 samples from the model
            print("Loss: " + str(best_loss.data.item()))
            print(tokenizer.decode(trigger_tokens))
    else:
        if args.num_trials !=1:
            total_utterances = 0
            num_breaks = 0
            queries = [
                'what is your job?',
                'how many children do you have?',
                'what kind of music do you like?',
                'where do you work?',
                'are you single?',
                'how do you feel?',
                'what is your favourite hobby?',
                'what do you like to do?',
                'do you have any pets?',
                'do you like animals?'
            ]
            
            for trial in range(args.num_trials):
                personality = random.choice(personalities)
                logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
                personality_sentences = tokenizer.decode(chain(*personality)).split('. ')
                history = []
                num_to_break = 0
                while True:
                    if args.query_type == 'permute':
                        queries = []
                        # for i in range(10):
                        #     #queries = [' '.join(random.sample(i.split(), len(i.split()))) for i in personality_sentences]
                        #     queries.append(' '.join(random.sample(list(set(' '.join(personality_sentences).replace('.','').split())), random.randint(5,8))))
                        # queries = list(set(queries))
                        temp = ' '.join(personality_sentences)
                        queries = [' '.join(random.sample(list(set(temp.split())), len(list(set(temp.split())))))]

                    raw_text = random.choice(queries)
                    logger.info("B:  %s", raw_text)
                    while not raw_text:
                        print('Prompt should not be empty!')
                        raw_text = input(">>> ")
                    history.append(tokenizer.encode(raw_text))
                    with torch.no_grad():
                        out_ids = interact.sample_sequence(personality, history, tokenizer, model, args)
                    history.append(out_ids)
                    history = history[-(2*args.max_history+1):]
                    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    logger.info("A:  %s", out_text)
                    #print(out_text)
                    num_to_break += 1
                    #Each pair is (personality, output). Instead of having a batch, find sentence similarity using some distance metric
                    #to find most relevant personality pair out of all
                    if args.split_output:
                        out_text_list = out_text.split('.')
                        out_text_list = list(filter(None, out_text_list))
                        batch_of_pairs = [[personality_sent, o_text] for personality_sent in personality_sentences for o_text in out_text_list if '?' not in o_text]

                    else:
                        if '?' in out_text:
                            continue
                        batch_of_pairs = [[personality_sent, out_text] for personality_sent in personality_sentences]
                    # labels = torch.tensor([[0], [2], [1], [0], [0]]).to(args.device)
                    labels = torch.zeros(len(batch_of_pairs),1).long().to(args.device)
                    probs = infer(inf_tokenizer, inf_model, batch_of_pairs, labels, args.device)
                    confidence, pred = get_predictions(probs)
                    # print("#"*5,"Contradiction","#"*5)
                    contradiction_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==0]
                    # import pdb
                    # pdb.set_trace()
                    #Break dialogue if there is a contradiction
                    if len(contradiction_pairs) != 0:
                        num_breaks += 1
                        total_utterances += num_to_break
                        logger.info("%s", contradiction_pairs)
                        logger.info("Confidence %s", [conf for conf, pr in zip(confidence, pred) if pr == 0])
                        logger.info("Breakage at  %s", num_to_break)
                        logger.info("#"*15)
                        cont_logger.info("CONTRADICTIONS")
                        cont_logger.info("%s", contradiction_pairs)
                        cont_logger.info("Confidence %s", [prob for prob, pr in zip(probs, pred) if pr == 0])
                        cont_logger.info("Contradiction confidence %s", [conf for conf, pr in zip(confidence, pred) if pr == 0])
                        cont_logger.info("Breakage at  %s", num_to_break)
                        cont_logger.info("#"*15)
                        # dump_data = {
                        #     'sentences': contradiction_pairs,
                        #     'confidences': [prob for prob, pr in zip(probs, pred) if pr == 0],
                        #     'contra_conf': [conf for conf, pr in zip(confidence, pred) if pr == 0]
                        # }
                        # cont_logger.info("NEUTRAL")
                        # neutral_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==1]
                        # cont_logger.info("%s", neutral_pairs)
                        # cont_logger.info("Confidence %s", [conf for conf, pr in zip(confidence, pred) if pr == 1])
                        # cont_logger.info("#"*15)
                        # cont_logger.info("ENTAILMENT")
                        # entailment_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==2]
                        # cont_logger.info("%s", entailment_pairs)
                        # cont_logger.info("Confidence %s", [conf for conf, pr in zip(confidence, pred) if pr == 2])
                        # cont_logger.info("#"*15)
                        break
                    #Break dialogue if max utterance limit is reached
                    if num_to_break >= args.max_utterance:
                        total_utterances += num_to_break
                        logger.info("Max utterance reached")
                        logger.info("#"*15)
                        break
                    # pprint(contradiction_pairs)
                    # print("#"*5,"Entailment","#"*5)
                    # neutral_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==1]
                    # neut_logger.info("%s", neutral_pairs)
                    # neut_logger.info("Confidence %s", [conf for conf, pr in zip(confidence, pred) if pr == 1])
                    # neut_logger.info("#"*15)
                    # pprint(neutral_pairs)
                    # print("#"*5,"Neutral","#"*5)
                    # entailment_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==2]
                    # pprint(entailment_pairs)
            avg_breakpoint = total_utterances / args.num_trials
            logger.info("Number of total breaks in {} trials is {}".format(args.num_trials, num_breaks))
            logger.info("Average breakpoint over {} trials is {}".format(args.num_trials,avg_breakpoint))
        else:
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
                if args.split_output:
                    out_text_list = out_text.split('.')
                    out_text_list = list(filter(None, out_text_list))
                    batch_of_pairs = [[personality_sent, o_text] for personality_sent in personality_sentences for o_text in out_text_list]

                else:
                    batch_of_pairs = [[personality_sent, out_text] for personality_sent in personality_sentences]
                
                # labels = torch.tensor([[0], [2], [1], [0], [0]]).to(args.device)
                labels = torch.zeros(len(batch_of_pairs),1).long().to(args.device)
                probs = infer(inf_tokenizer, inf_model, batch_of_pairs, labels, args.device)
                confidence, pred = get_predictions(probs)
                print("#"*5,"Contradiction","#"*5)
                contradiction_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==0]
                pprint(contradiction_pairs)
                print("#"*5,"Entailment","#"*5)
                neutral_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==1]
                pprint(neutral_pairs)
                print("#"*5,"Neutral","#"*5)
                entailment_pairs = [val for val, ind in zip(batch_of_pairs,pred) if ind==2]
                pprint(entailment_pairs)
                # pprint(list(zip(batch_of_pairs, pred)))


if __name__=='__main__':
    run()