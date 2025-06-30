import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from pdata import image_transform

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from transformers import AutoTokenizer
from models.clip_encoder import CLIPVisionTower
from transformers import CLIPImageProcessor
from llava.llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

import os
from omegaconf import DictConfig, ListConfig, OmegaConf
from openai import OpenAI

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    # _512x512
    parser.add_argument("--data_root", type=str, default="path/to/uni_c_tokens_data")
    
    parser.add_argument("--concept", type=str, default="bo")
    parser.add_argument("--ckpt_name", type=str, default="test_train_s3")
    parser.add_argument("--epoch_to_load", type=int, default=15)
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nums_new_token_i_stage_1", type=int, default=16)
    parser.add_argument("--nums_new_token_i_stage_2", type=int, default=8)
    parser.add_argument("--nums_new_token_i_stage_3", type=int, default=8)    
    return parser.parse_args()


def init_deepseek(api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client
CLIENT = init_deepseek("xxx")

def use_deepseek(client, sys_prompt, prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content


def run_one_test_item(config,
                      test_item, 
                      gt_dict, 
                      pred_dict, 
                      question_dict,
                      system_prompt, 
                      model, 
                      vq_model, 
                      device, 
                      uni_prompting, 
                      top_k):
    with torch.no_grad():
        image_path = test_item["image"]
        test_type = test_item["type"]
        question = system_prompt + test_item["conversations"][0]["value"].replace("<image>\n", "")
        ## processing
        image_ori = Image.open(image_path).convert("RGB")
        # tranforming the image to the required resolution
        image = image_transform(image_ori, resolution = config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)

        image_tokens_mmu = vq_model.get_code(image)
        image_tokens = image_tokens_mmu + len(uni_prompting.text_tokenizer)

        input_ids = uni_prompting.text_tokenizer(['USER: ' + question + ' ASSISTANT:'])['input_ids']
        input_ids = torch.tensor(input_ids).to(device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

        cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
                                    max_new_tokens=100, top_k=top_k,
                                    eot_token=uni_prompting.sptids_dict['<|eot|>'])

        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0].strip()
        
        if gt_dict.get(test_type) is None:
            gt_dict[test_type] = []
            pred_dict[test_type] = []
            question_dict[test_type] = []
        
        gt_dict[test_type].append(test_item["conversations"][1]["value"])
        pred_dict[test_type].append(text)
        question_dict[test_type].append({
            "question": question,
            "image": image_path
        })
    

def calculate_bleu(reference, candidate):
    # Convert string to character list (suitable for Chinese)
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    
    # Use smoothing function and lower-order n-gram
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                weights=(0.7, 0.3), # Use only 1-gram to 3-gram
                smoothing_function=smoothie)
    
    return bleu_score
    

def ds_score_2_sentences(query, gt, pred):
    sys_prompt = f"""You are a score evaluator. Given a question, an answer, and a predicted answer, you need to give a score.
    The range is <0, 0.5, 1.0>, where 0 means the answer is completely irrelevant, 1 means the answer is completely relevant, and 0.5 means the answer is partially relevant.
    You need to ignore grammar and only focus on whether the correct content is answered.
    For example, if asked 'What color dress do you like?', the answer is 'I like blue dresses',
    and the predicted answer is 'blue' or 'the favorite color is blue', your evaluation score should be 1.0 in both cases.
    """
    prompt = f"Question: {query}, Answer: {gt}, Predicted answer: {pred}. Please give a score and don't make any other responses."
    score_str = use_deepseek(CLIENT, sys_prompt, prompt)
    score = float(score_str.split("Score::")[-1].strip())
    assert score >= 0 and score <= 1.0, f"Invalid score: {score_str}"
    return score
    

def calculate_score(query, gt, pred, test_type):
    if test_type == "rec":
        if gt == "No.":
            if "no" in pred.lower():
                return {"no_recall": 1.0, "accuracy": 1.0}
            else:
                return {"no_recall": 0.0, "accuracy": 0.0}
        else:
            if "yes" in pred.lower():
                return {"recall": 1.0, "accuracy": 1.0}
            else:
                return {"recall": 0.0, "accuracy": 0.0}
    elif "choice" in test_type:
        if gt in pred:
            return {"accuracy": 1.0}
        else:
            return {"accuracy": 0.0}
    elif test_type == "vqa":
        return {"bleu": calculate_bleu(gt, pred), "ds-score": ds_score_2_sentences(query, gt, pred)}
    elif test_type == "text_only":
        return {"bleu": calculate_bleu(gt, pred), "ds-score": ds_score_2_sentences(query, gt, pred)}
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    

def statistic(gt_dict, pred_dict, question_dict):
    results = {}
    types = list(gt_dict.keys())
    for test_type in types:
        results[test_type] = {}
        results[test_type]["data"] = []
        
        scores = [] # [{"xx": 0.1, "yy": 0.2}, {"xx": 0.3, "zz": 0.4}, {"xx": 0.3, "zz": 0.1}, ]
        for gt, pred, question_data in zip(gt_dict[test_type], pred_dict[test_type], question_dict[test_type]):
            score = calculate_score(question_data["question"], gt, pred, test_type)
            scores.append(score)
            results[test_type]["data"].append({
                "question": question_data["question"],
                "image": question_data["image"],
                "gt": gt,
                "pred": pred,
                "score": score
            })
        # Calculate mean scores for each key
        scores_mean = {}
        all_keys = set(key for score in scores for key in score.keys())
        for key in all_keys:
            values = [score[key] for score in scores if key in score]
            scores_mean[key] = sum(values) / len(values)
        results[test_type]["score"] = scores_mean
    
    return results
    

def main():
    args = get_test_args()
    config = OmegaConf.load(args.config_file)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    data_root = args.data_root
    concept = args.concept
    ckpt_name = args.ckpt_name
    epoch2load = args.epoch_to_load

    ckpt_path = os.path.join("saves", concept, ckpt_name)
    ckpt_embed_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_embed.pt")
    ckpt_lm_head_weight_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_weight.pt")
    ckpt_lm_head_bias_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_bias.pt")

    nums_new_token_i_stage_1 = args.nums_new_token_i_stage_1
    nums_new_token_i_stage_2 = args.nums_new_token_i_stage_2
    nums_new_token_i_stage_3 = args.nums_new_token_i_stage_3
    
    new_tokens_total = nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3
    new_tokens_total = [f"<{concept}>"] + [f"<token_{i}>" for i in range(new_tokens_total)]
    num_new_tokens_total = len(new_tokens_total)  # 21
    sks_token = [f"<{concept}>"]
    
    new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
    new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_i_stage_2)]
    new_tokens_stage_3 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1 + nums_new_token_i_stage_2, nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3)]
    

    # Known original parameters
    # Text token count (ID 0-50304)
    original_text_vocab_size = len(tokenizer)  
    # Image token count (original ID 50305-58497)
    original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)

    original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498

    # New parameters
    new_text_vocab_size = original_text_vocab_size + num_new_tokens_total  # 50305 + 17 = 50322
    new_total_vocab = original_total_vocab + num_new_tokens_total          # 58498 + 17 = 58515

    # ------------------------------
    # Step 1: Modify the Tokenizer's vocabulary
    # ------------------------------

    # Add new tokens to positions 50305-50321
    num_new_tokens = tokenizer.add_tokens(new_tokens_total)
    new_token_ids_total = tokenizer.convert_tokens_to_ids(new_tokens_total)
    print("New token IDs:", new_token_ids_total)  # Should output 50305-50321
    sks_token_id = tokenizer.convert_tokens_to_ids(sks_token)
    print("sks_token_id:", sks_token_id)  # Should output <concept> token ID
    stage_1_token_ids = tokenizer.convert_tokens_to_ids(new_tokens_stage_1)  # Should output 50305-50320
    stage_2_token_ids = tokenizer.convert_tokens_to_ids(new_tokens_stage_2)  # Should output 50305-50320
    stage_3_token_ids = tokenizer.convert_tokens_to_ids(new_tokens_stage_3)  # Should output 50321-50322
    print("stage_1_token_ids:", stage_1_token_ids)  # Should output 50305-50320
    print("stage_2_token_ids:", stage_2_token_ids)  # Should output 50305-50320
    print("stage_3_token_ids:", stage_3_token_ids)  # Should output 50321-50322
    
    # ------------------------------
    # Step 2: Adjust model weights
    # ------------------------------
    with torch.no_grad():
        # Get embedding layer weights
        embeddings = model.showo.get_input_embeddings().weight.data
        
        # Expand embedding layer (58498 -> 58515)
        model.showo.resize_token_embeddings(new_total_vocab)
        # new_embeddings = model.showo.get_input_embeddings().weight.data

        # Move original Image Token weights back by 17 positions
        original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
        model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights
        
        # Initialize new token weights (using original last 17 text tokens)
        if os.path.exists(ckpt_embed_path):
            ckpt_embed_weight = torch.load(ckpt_embed_path)
            with torch.no_grad():
                model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_embed_weight.to(model.showo.get_input_embeddings().weight.device)
        else:
            raise ValueError("Embedding weights do not exist!")
            
        # new_text_weights = embeddings[original_text_vocab_size - num_new_tokens : original_text_vocab_size].clone()
        # model.showo.get_input_embeddings().weight.data[original_text_vocab_size : new_text_vocab_size] = new_text_weights
        # print(model.showo.lm_head.weight.data.shape[1])
        # Handle lm_head (assuming weight sharing with embedding layer)
        if model.showo.lm_head.weight.data.shape[0] == new_total_vocab:
            # Expand lm_head weights
            lm_head = model.showo.lm_head
            new_lm_head = torch.nn.Linear(
                lm_head.in_features, 
                new_total_vocab, 
                bias=hasattr(lm_head, 'bias')
            )
            new_lm_head.weight.data = lm_head.weight.data.clone()
            new_lm_head.weight.data[new_text_vocab_size:new_total_vocab] = lm_head.weight.data[original_text_vocab_size:original_total_vocab]
            
            if os.path.exists(ckpt_lm_head_weight_path):
                ckpt_lm_head_weight = torch.load(ckpt_lm_head_weight_path)
                with torch.no_grad():
                    new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_weight.to(new_lm_head.weight.device)
            else:
                raise ValueError("lm_head weights do not exist!")
            # new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = lm_head.weight.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                
                if os.path.exists(ckpt_lm_head_bias_path):
                    ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                    with torch.no_grad():
                        new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_bias.to(new_lm_head.weight.device)
                else:
                    raise ValueError("lm_head bias do not exist!")
                # new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = lm_head.bias.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

    config.model.showo.llm_vocab_size = len(tokenizer) - 10
    # ------------------------------

    # ------------------------------
    system_personalized_prompt = f"<{concept}> is "
    for i in range(nums_new_token_i_stage_1):
        system_personalized_prompt += f"<token_{i}>"
    for i in range(nums_new_token_i_stage_1 + nums_new_token_i_stage_2,
                    nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3):
        system_personalized_prompt += f"<token_{i}>"
        if i == nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3 - 1:
            system_personalized_prompt += ".\n"
    system_prompt = system_personalized_prompt
    # ------------------------------

    test_data_file = os.path.join(data_root, "test_data", f"{concept}.json")
    with open(test_data_file, "r") as f:
        test_data = json.load(f)
    
    gt_dict = {}
    pred_dict = {}
    question_dict = {}
    for test_item in test_data:
        run_one_test_item(config,
                          test_item, 
                          gt_dict, 
                          pred_dict, 
                          question_dict,
                          system_prompt, 
                          model, 
                          vq_model, 
                          device, 
                          uni_prompting, 
                          top_k)
    
    results_json = statistic(gt_dict, pred_dict, question_dict)
    results_json_path = os.path.join("logs", concept, ckpt_name)
    os.makedirs(results_json_path, exist_ok=True)
    results_json_path = os.path.join(results_json_path, f"epoch_{epoch2load}_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"Results saved to {results_json_path}")


if __name__ == "__main__":
    main()
