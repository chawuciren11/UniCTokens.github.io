import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import random
import json
# import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
import argparse

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    # _512x512
    parser.add_argument("--data_root", type=str, default="path/to/uni_c_tokens_data")
    
    parser.add_argument("--concept", type=str, default="bo")
    parser.add_argument("--ckpt_name", type=str, default="test_train_s3")
    parser.add_argument("--epoch_to_load", type=int, default=15)
    parser.add_argument("--nums_new_token_i_stage_1", type=int, default=16)
    parser.add_argument("--nums_new_token_i_stage_2", type=int, default=8)
    parser.add_argument("--nums_new_token_i_stage_3", type=int, default=8)
    parser.add_argument("--num_gen_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="t2i_saved")
    
    parser.add_argument("--inverse_prompt", default=False, action="store_true")
    parser.add_argument("--pers_system_prompt", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--t2i_batch_size", type=int, default=20)
    
    return parser.parse_args()


def main():
    args = get_test_args()
    config = OmegaConf.load(args.config_file)
    device = torch.device(args.device)
    
    config.mode = 't2i'
    config.batch_size = 2
    config.generation_timesteps = 50
    config.guidance_scale = 5
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    # load from users passed arguments
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    
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
    model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    config.model.showo.llm_vocab_size = len(tokenizer) - 10
    adj_tokens_str = "".join([f"<token_{i}>" for i in range(nums_new_token_i_stage_1 + nums_new_token_i_stage_2)])
    adj_tokens_str = adj_tokens_str + f" <{concept}>"
            
    t2i_conditions_file = os.path.join(data_root, "concept/test", concept, "t2i_conditions.json")
    with open(t2i_conditions_file, "r") as f:
        t2i_conditions = json.load(f)
    
    for t2i_condition in t2i_conditions:
        save_dir = os.path.join(args.output_dir, concept, ckpt_name, str(epoch2load), t2i_condition)
        os.makedirs(save_dir, exist_ok=True)
        print("Save dir:", save_dir)
        condition =  t2i_condition
        if args.inverse_prompt:
            condition = condition.replace(f"<{concept}>", adj_tokens_str)
        print("Condition:", condition)
        
        for i in tqdm(range(0, args.num_gen_images, args.t2i_batch_size)):
            image_tokens = torch.ones((args.t2i_batch_size, config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id    # shape [batch, num_vq_tokens] [1, 256], fill with mask token
            conditions = [condition] * args.t2i_batch_size
            
            input_ids, _ = uni_prompting((conditions, image_tokens), 't2i_gen')   # [1, 387]

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * args.t2i_batch_size, image_tokens), 't2i_gen')
                # [1, 387], == [PAD] * 126 + <|t2i|> + <|endoftext|> + <|endoftext|> + <|soi|> + [MASK] * 256 + <|eoi|> ## no prompt
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),    # [2, 387]
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None
            # attention_mask [2, 1, 387, 387]

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
                
            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]
            
            for j in range(args.t2i_batch_size):
                gen_image = pil_images[j]
                gen_image.save(os.path.join(save_dir, f"{args.t2i_batch_size * i + j}.png"))
            # print(f"Saved image {i} to {save_dir}")


if __name__ == "__main__":
    main()