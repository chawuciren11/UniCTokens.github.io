import argparse
from typing import Union

from sklearn.cluster import KMeans
from pdata import get_personalized_mmu_dataloader, get_personalized_t2i_dataloader, get_concept_info, image_transform
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

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

import os
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    # _512x512
    parser.add_argument("--data_root", type=str, default="path/to/uni_c_tokens_data")
    
    parser.add_argument("--concept", type=str, default="bingbing")
    parser.add_argument("--task_name", type=str, default="test_train_s3")
    
    parser.add_argument("--pre_trained_ckpt_name", type=str, default="test_train_s2")
    parser.add_argument("--t2i_data", default=False, action="store_true")
    parser.add_argument("--mmu_data", default=False, action="store_true")
    parser.add_argument("--more_t2i_data", default=False, action="store_true")
    parser.add_argument("--image_size", type=int, default=512)
    
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--t2i_bsz", type=int, default=1)
    parser.add_argument("--mmu_bsz", type=int, default=1)
    parser.add_argument("--save_training_image", default=False, action="store_true")
    parser.add_argument("--l2_lambda", type=float, default=0.0, help="identifier l2 norm regularization")
    parser.add_argument("--caption_training", default=False, action="store_true")

    parser.add_argument("--interval_epochs", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--epoch_to_load", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nums_new_token_i_stage_1", type=int, default=16)
    parser.add_argument("--nums_new_token_i_stage_2", type=int, default=8)
    parser.add_argument("--nums_new_token_i_stage_3", type=int, default=8)
    parser.add_argument("--less_t2i_data", default=False, action="store_true")

    return parser.parse_args()


def get_nums_new_token_i_k_means_embeddings(args, concept, model, vq_model, new_text_vocab_size, nums_new_token_i):
    diff_masks_dir = os.path.join("saves", concept, args.pre_trained_ckpt_name, "diff_masks")
    images = [img for img in os.listdir(diff_masks_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = [Image.open(os.path.join(diff_masks_dir, img)).convert("RGB") for img in images]
    pixel_values_list = [image_transform(image, args.image_size, normalize=True, flip_augmentation=False, crop_augmentation=False) for image in images]
    batched_pixel_values = torch.stack(pixel_values_list, dim=0).to(args.device)
    with torch.no_grad():
        batched_image_token_ids = vq_model.get_code(batched_pixel_values)
        batched_image_token_ids = batched_image_token_ids + new_text_vocab_size
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


def t2i_inite_emd(args, concept, config, tokenizer, model, vq_model):
    if args.nums_new_token_i_stage_3 == 0:
        return None
    return get_nums_new_token_i_k_means_embeddings(args, concept, model, vq_model, len(tokenizer), args.nums_new_token_i_stage_3)


def setup_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(args.device)
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(args.device)
    return tokenizer, uni_prompting, vq_model, model


def update_tokens_load_from_pretrained(concept, tokenizer, model, 
                                       pre_trained_ckpt_name, epoch_to_load, nums_new_token_i_stage_1, 
                                       nums_new_token_stage_2, nums_new_token_stage_3, t2i_inited_emd=None):
    ckpt_path = os.path.join("saves", concept, pre_trained_ckpt_name)
    ckpt_embed_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_embed.pt")
    ckpt_lm_head_weight_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_lm_head_weight.pt")
    ckpt_lm_head_bias_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_lm_head_bias.pt")

    new_tokens_total = nums_new_token_i_stage_1 + nums_new_token_stage_2 + nums_new_token_stage_3
    new_tokens_total = [f"<{concept}>"] + [f"<token_{i}>" for i in range(new_tokens_total)]
    num_new_tokens_total = len(new_tokens_total)  # 21
    sks_token = [f"<{concept}>"]
    
    new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
    new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_stage_2)]
    new_tokens_stage_3 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1 + nums_new_token_stage_2, nums_new_token_i_stage_1 + nums_new_token_stage_2 + nums_new_token_stage_3)]
    
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
        
        if os.path.exists(ckpt_embed_path):
            ckpt_embed_weight = torch.load(ckpt_embed_path)
            with torch.no_grad():
                model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size-nums_new_token_stage_3] = ckpt_embed_weight.to(model.showo.get_input_embeddings().weight.device)
                if t2i_inited_emd is not None:
                    
                    model.showo.get_input_embeddings().weight.data[new_text_vocab_size-nums_new_token_stage_3:new_text_vocab_size] = t2i_inited_emd
        else:
            raise ValueError("Embedding weights do not exist!")
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
                    new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size-nums_new_token_stage_3] = ckpt_lm_head_weight.to(new_lm_head.weight.device)
            else:
                raise ValueError("lm_head weights do not exist!")

            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                
                if os.path.exists(ckpt_lm_head_bias_path):
                    ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                    with torch.no_grad():
                        new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size-nums_new_token_stage_3] = ckpt_lm_head_bias.to(new_lm_head.weight.device)
                else:
                    raise ValueError("lm_head bias do not exist!")   
                         
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

    index_no_updates = torch.ones((new_total_vocab,), dtype=torch.bool)
    index_no_updates[sks_token_id] = False
    index_no_updates[stage_3_token_ids] = False
    
    # adj_tokens = [f"<token_{i}>" for i in range(num_new_tokens_total)]
    # adj_token_ids = tokenizer.convert_tokens_to_ids(adj_tokens) # shape: [16]
    
    
    # ------------------------------
    # Verification
    # ------------------------------
    # Check new token IDs

    # Check new ID of an original Image Token
    sample_image_token = tokenizer.convert_ids_to_tokens(original_text_vocab_size)  # Original ID 50305
    print(f"Concept Token '{sample_image_token}' new ID:", tokenizer.convert_tokens_to_ids(sample_image_token))  # Should output 50322

    # Check embedding layer shape
    print("Embedding layer size:", model.showo.get_input_embeddings().weight.shape)  # Should show torch.Size([58515, 2048])

    # Check positions and count of True in index_no_updates, True should be new token ids
    print("False positions in index_no_updates:", torch.nonzero(~index_no_updates).squeeze())  # Should output 50305-50321
    print("True count in index_no_updates:", torch.sum(index_no_updates))  # Should output 58498

    with torch.no_grad():
        orig_embeds = model.showo.get_input_embeddings().weight.data.clone()
        orig_lm_head_weight = model.showo.lm_head.weight.data.clone()
        orig_lm_head_bias = model.showo.lm_head.bias.data.clone()
        
    return tokenizer, model, orig_embeds, orig_lm_head_weight, orig_lm_head_bias, \
           index_no_updates, new_total_vocab, new_token_ids_total, sks_token_id


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
    t2i_inited_emd = t2i_inite_emd(args, args.concept, config, tokenizer, model, vq_model)
    
    data_root = args.data_root
    concept = args.concept
    save_path = os.path.join("saves", concept, args.task_name)
    os.makedirs(save_path, exist_ok=True)
    
    total_new_token_i = args.nums_new_token_i_stage_2 + args.nums_new_token_i_stage_3
    
    # set up training arch

    tokenizer, model, orig_embeds, orig_lm_head_weight, \
    orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids, sks_token_id \
    = update_tokens_load_from_pretrained(concept, tokenizer, 
                                         model, args.pre_trained_ckpt_name, 
                                         args.epoch_to_load, 
                                         args.nums_new_token_i_stage_1,
                                         args.nums_new_token_i_stage_2, 
                                         args.nums_new_token_i_stage_3, t2i_inited_emd)

    # set up parameters
    vq_model.requires_grad_ = False
    vq_model.eval()
    model.train()

    for name, param in model.named_parameters():
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
                weight_decay=1e-3,
                eps=1e-08,
            )
    
    for names, p in model.named_parameters():
        if p.requires_grad:
            print(f"{names} requires_grad") # embed_token, lm_head will be updated
            
    # Count all trainable parameters
    trainable_params_num = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_params_num}")
    
    # set up dataset
    if args.t2i_data:
        load_caption_training_path = os.path.join("saves", concept, args.pre_trained_ckpt_name, "captions.json")
        t2i_dataloader = get_personalized_t2i_dataloader(data_root, 
                                                         concept, 
                                                         tokenizer, 
                                                         args.image_size, 
                                                         batch_size=args.t2i_bsz, 
                                                         num_workers=0, 
                                                         max_length=128, 
                                                         nums_new_token_i=args.nums_new_token_i_stage_1 + args.nums_new_token_i_stage_2,
                                                         more_data = args.more_t2i_data, 
                                                         inited = True, 
                                                         system_prompt_t2i = False, 
                                                         caption_training = args.caption_training, 
                                                         load_caption_training_path = load_caption_training_path,
                                                         less_t2i_data=args.less_t2i_data,
                                                         load_cropped_image=False)
    if args.mmu_data:
        mmu_dataloader = get_personalized_mmu_dataloader(data_root, 
                                                         concept, 
                                                         tokenizer, 
                                                         args.image_size, 
                                                         batch_size=args.mmu_bsz, 
                                                         num_workers=0, 
                                                         max_length=128, 
                                                         new_tokens=True,
                                                         stage = 3,
                                                         nums_new_token_i_stage_1=args.nums_new_token_i_stage_1,
                                                         nums_new_token_i_stage_2=args.nums_new_token_i_stage_2,
                                                         nums_new_token_i_stage_3=args.nums_new_token_i_stage_3,)

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
    mask_dtype = model.showo.get_input_embeddings().weight.dtype
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
            # logits: [B_t2i+B_mmu, len, 58515(model.showo.lm_head classification count)] 
            if args.t2i_data and args.mmu_data:
                loss = 0.8 * loss_t2i + 0.2 * loss_mmu
            elif args.t2i_data:
                loss = loss_t2i
            elif args.mmu_data:
                loss = loss_mmu
            print(f"Loss: {loss.item()}")
                
            # ====== Add L2 regularization, penalize differences between updatable parts and original parameters ======
            l2_lambda = args.l2_lambda  # Hyperparameter, adjustable as needed
            # Only calculate regularization for updatable tokens (i.e., ~index_no_updates part)
            l2_reg_embed = torch.sum((model.showo.get_input_embeddings().weight[sks_token_id] - orig_embeds[sks_token_id]) ** 2)
            l2_reg_lm_weight = torch.sum((model.showo.lm_head.weight[sks_token_id] - orig_lm_head_weight[sks_token_id]) ** 2)
            l2_reg_lm_bias = torch.sum((model.showo.lm_head.bias[sks_token_id] - orig_lm_head_bias[sks_token_id]) ** 2)
            print(f"L2 Regularization - Embed: {l2_reg_embed.item()}, ")
            print(f"L2 Regularization - LM Weight: {l2_reg_lm_weight.item()}, ")
            print(f"L2 Regularization - LM Bias: {l2_reg_lm_bias.item()}, ")
            loss = loss + l2_lambda * (l2_reg_embed + l2_reg_lm_weight + l2_reg_lm_bias)
            # ================================================================
            
            
            loss.backward()
            optimizer.step()
            if args.t2i_data and args.save_training_image and (epoch+1) % 10 == 0:
                with torch.no_grad():
                    # Extract the first B_t2i from batch, last config.model.showo.num_vq_tokens from len, last config.model.showo.codebook_size from d, as image logits
                    image_logits = logits[:batch_size_t2i, -config.model.showo.num_vq_tokens:, -config.model.showo.codebook_size:]
                    assert image_logits.shape == (batch_size_t2i, config.model.showo.num_vq_tokens, config.model.showo.codebook_size)
                    # Get classification id through maximum value position [batch_size_t2i, config.model.showo.num_vq_tokens]
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
            save_path_embed = os.path.join(save_path, f"epoch_{epoch+1}_embed.pt")
            save_path_lm_head_weight = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_weight.pt")
            save_path_lm_head_bias = os.path.join(save_path, f"epoch_{epoch+1}_lm_head_bias.pt")
            
            torch.save(model.showo.get_input_embeddings().weight.data[new_token_ids], save_path_embed)
            torch.save(model.showo.lm_head.weight.data[new_token_ids], save_path_lm_head_weight)
            torch.save(model.showo.lm_head.bias.data[new_token_ids], save_path_lm_head_bias)


if __name__ == "__main__":
    main()
