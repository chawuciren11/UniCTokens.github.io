import argparse
from typing import Union
from pdata import get_personalized_mmu_dataloader, get_personalized_t2i_dataloader, get_concept_info, get_concept_all_training_images
from lightning.pytorch.utilities import CombinedLoader

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from transformers import AutoTokenizer
from llava.llava import conversation as conversation_lib
from peft import LoraConfig, get_peft_model

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

import os
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.cluster import KMeans


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    parser.add_argument("--data_root", type=str, default="path/to/uni_c_tokens_data")
    
    parser.add_argument("--concept", type=str, default="bo")
    parser.add_argument("--task_name", type=str, default="test_train_s1")
    
    parser.add_argument("--need_new_tokens", default=False, action="store_true")
    parser.add_argument("--need_lora", default=False, action="store_true")
    parser.add_argument("--t2i_data", default=False, action="store_true")
    parser.add_argument("--mmu_data", default=False, action="store_true")
    parser.add_argument("--more_t2i_data", default=False, action="store_true")
    parser.add_argument("--image_size", type=int, default=512)
    
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--t2i_bsz", type=int, default=1)
    parser.add_argument("--mmu_bsz", type=int, default=4)
    parser.add_argument("--save_training_image", default=False, action="store_true")
    parser.add_argument("--need_init", default=False, action="store_true")
    parser.add_argument("--init_by_images", default=False, action="store_true")

    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--interval_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nums_new_token_i", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", nargs='+', default=["fc1", "k_proj", "v_proj", "q_proj", "fc2"])
    return parser.parse_args()


def setup_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(args.device)
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(args.device)
    return tokenizer, uni_prompting, vq_model, model


def get_nums_new_token_i_k_means_embeddings(args, concept, model, vq_model, new_text_vocab_size, nums_new_token_i):
    pixel_values_list = get_concept_all_training_images(concept, resolution=args.image_size)
    batched_pixel_values = torch.stack(pixel_values_list, dim=0).to(args.device)
    batched_image_token_ids = vq_model.get_code(batched_pixel_values)
    batched_image_token_ids = batched_image_token_ids + new_text_vocab_size
    with torch.no_grad():
        batched_image_embeddings = model.showo.get_input_embeddings()(batched_image_token_ids).to(args.device) # [batch_size, n, 2048]
    # [batch_size, n, 2048] -> [batch_sizexn, 2048]
    image_embeddings = batched_image_embeddings.view(-1, batched_image_embeddings.shape[-1])  # Flatten batch_size
    image_embeddings_npy = image_embeddings.cpu().numpy()  # Convert to numpy array
    print("image_embeddings_npy shape:", image_embeddings_npy.shape)  # [batch_size * n, 2048]
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=nums_new_token_i, random_state=0)
    kmeans.fit(image_embeddings_npy)  # Fit the k-means model
    
    # [nums_new_token_i, 2048]
    cluster_centers = torch.tensor(kmeans.cluster_centers_, device=args.device, dtype=model.showo.get_input_embeddings().weight.dtype)  # Convert to tensor
    return cluster_centers
    

def update_tokens(args, 
                  concept, 
                  tokenizer, 
                  model, 
                  vq_model, 
                  init_words, 
                  nums_new_token_i=16):
    new_tokens = [f"<{concept}>"] + [f"<token_{i}>" for i in range(nums_new_token_i)]
    num_new_tokens = len(new_tokens)  # 17
    # Known original parameters
    # Number of text tokens (ID 0-50304)
    original_text_vocab_size = len(tokenizer) 
    # Number of Image Tokens (Original IDs 50305-58497)
    original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)

    original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498
    
    # New parameters
    new_text_vocab_size = original_text_vocab_size + num_new_tokens  # 50305 + 17 = 50322
    new_total_vocab = original_total_vocab + num_new_tokens          # 58498 + 17 = 58515

    # ------------------------------
    # Step 1: Modify the Tokenizer's vocabulary list
    # ------------------------------

    # Add new token to the position of 50305-50321
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("New token ID:", new_token_ids)  # should print 50305-50321
    
    # ------------------------------
    # Step 2: Adjust the model's weights
    # ------------------------------
    with torch.no_grad():
        # Get embedded layer weights
        embeddings = model.showo.get_input_embeddings().weight.data
        
        # Expand embedded layer (58498 -> 58515)
        model.showo.resize_token_embeddings(new_total_vocab)

        # Shift the original Image Token weight backward by 17 positions.
        original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
        model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights
        
        # lm_head
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
            # new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = lm_head.weight.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                # new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = lm_head.bias.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

        # Initialize the weights of a new token
        if init_words is not None and isinstance(init_words, list):
            print(f"Initialize the new token using specified words: {init_words}")
            
            all_init_embeds = []
            for init_word in init_words:
                if init_word is not None:
                    init_token_ids = tokenizer(init_word, add_special_tokens=False).input_ids
                    if len(init_token_ids) > 0:
                        all_init_embeds.append(torch.mean(embeddings[init_token_ids], dim=0))
            
            if all_init_embeds:
                avg_init_embed = torch.mean(torch.stack(all_init_embeds), dim=0)
                
                concept_token_id = new_token_ids[0]  # <concept> token's ID
                other_token_ids = new_token_ids[1:]  # <token_x> token's ID
                
                if init_words and init_words[0] is not None:
                    first_word_ids = tokenizer(init_words[0], add_special_tokens=False).input_ids
                    if len(first_word_ids) > 0:
                        concept_embed = torch.mean(embeddings[first_word_ids], dim=0)
                        model.showo.get_input_embeddings().weight.data[concept_token_id] = concept_embed
                        print(f"Token {new_tokens[0]} initialized with '{init_words[0]}'")
                    else:
                        model.showo.get_input_embeddings().weight.data[concept_token_id] = avg_init_embed
                        print(f"Token {new_tokens[0]} is initialized with the average embedding vector of all words")
                else:
                    model.showo.get_input_embeddings().weight.data[concept_token_id] = avg_init_embed
                    print(f"Token {new_tokens[0]} is initialized with the average embedding vector of all words")
                
                # <token_x>
                for i, token_id in enumerate(other_token_ids):
                    model.showo.get_input_embeddings().weight.data[token_id] = avg_init_embed
                    print(f"Token {new_tokens[i+1]} initialized with the average embedding vector of all words")
            else:
                print("Warning: No valid initialization words, the new token will remain randomly initialized")
        elif isinstance(init_words, str):
            print("Initialize new token with image")
            first_word_ids = tokenizer(init_words, add_special_tokens=False).input_ids
            concept_token_id = new_token_ids[0]  # <concept> token's ID
            
            concept_embed = torch.mean(embeddings[first_word_ids], dim=0)
            model.showo.get_input_embeddings().weight.data[concept_token_id] = concept_embed
            print(f"Token {new_tokens[0]} initialized with '{init_words}'")
            
            # For all <token_x>, use image embedding -> k-means -> token
            if len(new_tokens) > 1:
                other_token_ids = new_token_ids[1:]  # <token_x> series token IDs
                nums_new_token_i_k_means_embeddings = get_nums_new_token_i_k_means_embeddings(
                    args, concept, model, vq_model, new_text_vocab_size, nums_new_token_i
                )
                
                existing_norm = embeddings.norm(dim=1).mean()
                nums_new_token_i_k_means_embeddings_norm = nums_new_token_i_k_means_embeddings.norm(dim=1, keepdim=True)
                nums_new_token_i_k_means_embeddings = nums_new_token_i_k_means_embeddings / nums_new_token_i_k_means_embeddings_norm * nums_new_token_i_k_means_embeddings_norm
                with torch.no_grad():
                    model.showo.get_input_embeddings().weight.data[other_token_ids] = nums_new_token_i_k_means_embeddings

    index_no_updates = torch.ones((new_total_vocab,), dtype=torch.bool)
    index_no_updates[new_token_ids] = False
    
    with torch.no_grad():
        orig_embeds = model.showo.get_input_embeddings().weight.data.clone()
        orig_lm_head_weight = model.showo.lm_head.weight.data.clone()
        orig_lm_head_bias = model.showo.lm_head.bias.data.clone()
        
    return tokenizer, model, orig_embeds, orig_lm_head_weight, orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids


def apply_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,  
        lora_alpha=args.lora_alpha,  
        lora_dropout=args.lora_dropout,  
        task_type="CAUSAL_LM",  
        target_modules = args.lora_target_modules 
    )
    
    model.showo = get_peft_model(model.showo, lora_config)
    
    return model


def prepare_inputs_and_labels(
        mask_id,
        config,
        vq_model,
        uni_prompting,
        mask_schedule,
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
):

    image_tokens = vq_model.get_code(pixel_values_or_image_ids)
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

    # create MLM mask and labels
    input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=is_train,
    )
    input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')

    return input_ids, labels, mask_prob, image_tokens


def main():
    args = get_test_args()
    
    config = OmegaConf.load(args.config_file)
    tokenizer, uni_prompting, vq_model, model = setup_model(args, config)
    
    data_root = args.data_root
    concept = args.concept
    save_path = os.path.join("saves", concept, args.task_name)
    os.makedirs(save_path, exist_ok=True)
    
    # set up training arch
    if args.need_new_tokens:
        if args.need_init:
            class_concept, concept_info, _ = get_concept_info(concept)
            init_words = [class_concept] + concept_info.split(" ")
            if args.init_by_images:
                init_words = class_concept
        else:
            init_words = None
        tokenizer, model, orig_embeds, orig_lm_head_weight, \
        orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids \
        = update_tokens(args, concept, tokenizer, model, vq_model, init_words, args.nums_new_token_i)
    if args.need_lora:
        model = apply_lora(model, args)
    
    # set up parameters
    vq_model.requires_grad_ = False
    vq_model.eval()
    model.train()

    for name, param in model.named_parameters():
        
        if args.need_lora and args.need_new_tokens:
            if "lora" in name or "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif args.need_lora:
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif args.need_new_tokens:
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(
                trainable_params, # for optimize the embeddings and the head
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
                eps=1e-6,
            )
    
    for names, p in model.named_parameters():
        if p.requires_grad:
            print(f"{names} requires_grad") # embed_token, lm_head will be updated
            
    lora_params = list(filter(lambda kv: "lora" in kv[0], model.named_parameters()))
    lora_params_num = sum(p.numel() for n, p in lora_params)
    print(f"LoRA parameters: {lora_params_num}")
    trainable_params_num = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_params_num}")
    
    # set up dataset
    if args.t2i_data:
        t2i_dataloader = get_personalized_t2i_dataloader(data_root, 
                                                         concept, 
                                                         tokenizer, 
                                                         args.image_size, 
                                                         batch_size=args.t2i_bsz, 
                                                         num_workers=0, 
                                                         max_length=128, 
                                                         more_data = args.more_t2i_data, 
                                                         inited = args.need_init)
    if args.mmu_data:
        mmu_dataloader = get_personalized_mmu_dataloader(data_root, 
                                                         concept, 
                                                         tokenizer, 
                                                         args.image_size, 
                                                         batch_size=args.mmu_bsz, 
                                                         num_workers=0, 
                                                         max_length=128, 
                                                         new_tokens=args.need_new_tokens,
                                                         stage = 1)
        
    if args.t2i_data and args.mmu_data:
        iterables = {
            'mmu_flow': mmu_dataloader,
            't2i_flow': t2i_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    elif args.t2i_data:
        iterables = {
            't2i_flow': t2i_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    elif args.mmu_data:
        iterables = {
            'mmu_flow': mmu_dataloader
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
    else:
        raise ValueError("No dataset loaded")
    
    combined_dataloader_list = list(combined_dataloader)

    # misc setting
    model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    model.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    mask_id = model.mask_token_id
    if args.need_lora:
        mask_dtype = model.showo.base_model.model.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.get_input_embeddings().weight.dtype
    if args.need_new_tokens:
        model.output_size = new_total_vocab
    
    # start training
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}")
        loss_list = []
        if args.t2i_data:
            loss_t2i_list = []
        if args.mmu_data:
            loss_mmu_list = []
        for batch, batch_idx, dataloader_idx in tqdm(combined_dataloader_list):
            if args.t2i_data:
                batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
                pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["conditions"]
                pixel_values = pixel_values.to(args.device)
                input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori = prepare_inputs_and_labels(mask_id,
                                                                                        config,
                                                                                        vq_model,
                                                                                        uni_prompting,
                                                                                        mask_schedule,
                                                                                        pixel_values,
                                                                                        texts,
                                                                                        is_train=True,)
                attention_mask_t2i = create_attention_mask_predict_next(input_ids_t2i,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True,
                                                                    return_inverse_mask=True)
                attention_mask_t2i = attention_mask_t2i.to(mask_dtype)
            if args.mmu_data:
                batch_size_mmu = batch["mmu_flow"]["images"].shape[0]
                pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"],
                                                            batch["mmu_flow"]["input_ids"],
                                                            batch["mmu_flow"]["labels"])
                # print(input_ids_mmu.shape)
                # print(input_ids_mmu)
                pixel_values_mmu = pixel_values_mmu.to(args.device, non_blocking=True)
                input_ids_mmu = input_ids_mmu.to(args.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                
                input_ids_mmu = torch.cat([
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(args.device),
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(args.device),
                            image_tokens_mmu,
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(args.device),
                            input_ids_mmu,
                        ], dim=1).long()

                labels_mmu = torch.cat([
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(args.device),
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(args.device),
                            torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(args.device),
                            labels_mmu.to(args.device)
                        ], dim=1).long()
                
                
                attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to(args.device),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                attention_mask_mmu = attention_mask_mmu.to(mask_dtype)

            if args.t2i_data and args.mmu_data:
                attention_mask = torch.cat([attention_mask_t2i, attention_mask_mmu], dim=0)
                input_ids = torch.cat([input_ids_t2i, input_ids_mmu], dim=0)
                labels = torch.cat([labels_t2i, labels_mmu], dim=0)
            elif args.t2i_data:
                attention_mask = attention_mask_t2i
                input_ids = input_ids_t2i
                labels = labels_t2i
                batch_size_mmu = 0
            elif args.mmu_data:
                attention_mask = attention_mask_mmu
                input_ids = input_ids_mmu
                labels = labels_mmu
                batch_size_t2i = 0
            else:
                raise ValueError("No dataset loaded")
            
            optimizer.zero_grad()
            logits, loss_t2i, loss_lm, loss_mmu = model(
                        input_ids=input_ids,
                        input_embeddings=None,
                        attention_mask=attention_mask,
                        labels=labels,
                        label_smoothing=0.0,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=0,
                        batch_size_mmu=batch_size_mmu,
                        max_seq_length=128,
                    )
            # logits: [B_t2i+B_mmu, len, 58515(model.showo.lm_head)] 
            if args.t2i_data and args.mmu_data:
                loss = 0.8 * loss_t2i + 0.2 * loss_mmu
            elif args.t2i_data:
                loss = loss_t2i
            elif args.mmu_data:
                loss = loss_mmu
            
            loss.backward()
            optimizer.step()
            if args.t2i_data and args.save_training_image:
                with torch.no_grad():
                    # Extract the first B_t2i elements from the batch, 
                    # the last config.model.showo.num_vq_tokens elements from the length, 
                    # and the last config.model.showo.codebook_size elements from the 'd' vector, to form the image logits.
                    image_logits = logits[:batch_size_t2i, -config.model.showo.num_vq_tokens:, -config.model.showo.codebook_size:]
                    assert image_logits.shape == (batch_size_t2i, config.model.showo.num_vq_tokens, config.model.showo.codebook_size)
                    # Get the category id through the position of the maximum value [batch_size_t2i, config.model.showo.num_vq_tokens]
                    image_gen_ids = torch.argmax(image_logits, dim=-1) 
                    assert image_gen_ids.shape == (batch_size_t2i, config.model.showo.num_vq_tokens)
                    image_gen_ids = torch.clamp(image_gen_ids, max=config.model.showo.codebook_size - 1, min=0)
                    training_gen_images = vq_model.decode_code(image_gen_ids) # [batch_size_t2i, 3, size, size]
                    training_gen_images = torch.clamp((training_gen_images + 1.0) / 2.0, min=0.0, max=1.0)
                    training_gen_images *= 255.0
                    training_gen_images = training_gen_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    pil_images = [Image.fromarray(image) for image in training_gen_images]
                    for i, img in enumerate(pil_images):
                        save_training_image_dir = os.path.join(save_path, "training_images", f"epoch_{epoch+1}", f"batch_{batch_idx}_loss{loss_t2i.item()}")
                        os.makedirs(save_training_image_dir, exist_ok=True)
                        img.save(os.path.join(save_training_image_dir, f"{i}_t2i.png"))
                   
            loss_list.append(loss.item())
            if args.t2i_data:
                loss_t2i_list.append(loss_t2i.item())
            if args.mmu_data:
                loss_mmu_list.append(loss_mmu.item())
            
            if args.need_new_tokens:
                model.showo.get_input_embeddings().weight.data[index_no_updates] = orig_embeds[index_no_updates]
                model.showo.lm_head.weight.data[index_no_updates] = orig_lm_head_weight[index_no_updates]
                model.showo.lm_head.bias.data[index_no_updates] = orig_lm_head_bias[index_no_updates]

        if args.t2i_data and args.mmu_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_t2i: {np.mean(loss_t2i_list)}, loss_mmu: {np.mean(loss_mmu_list)}")
        elif args.t2i_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_t2i: {np.mean(loss_t2i_list)}")
        elif args.mmu_data:
            print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}, loss_mmu: {np.mean(loss_mmu_list)}")
        else:
            raise ValueError("No dataset loaded")
        
        if (epoch+1) % args.interval_epochs == 0:
            if args.need_new_tokens:
                save_path_embed = os.path.join(save_path, f"epoch_{epoch+1}_embed.pt")
                save_path_lm_head_weight = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_weight.pt")
                save_path_lm_head_bias = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_bias.pt")
                
                torch.save(model.showo.get_input_embeddings().weight.data[new_token_ids], save_path_embed)
                torch.save(model.showo.lm_head.weight.data[new_token_ids], save_path_lm_head_weight)
                torch.save(model.showo.lm_head.bias.data[new_token_ids], save_path_lm_head_bias)
            if args.need_lora:
                model.showo.save_pretrained(os.path.join(save_path, f"epoch_{epoch+1}_lora_model"))


if __name__ == "__main__":
    main()
