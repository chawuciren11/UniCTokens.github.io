import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
# import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from pdata import get_concept_info
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

from omegaconf import DictConfig, ListConfig, OmegaConf
import matplotlib.pyplot as plt
import math

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    parser.add_argument("--data_root", type=str, default="path/to/uni_c_tokens_data")
    
    parser.add_argument("--concept", type=str, default="bingbing")
    parser.add_argument("--ckpt_name", type=str, default="test_train_s2")
    parser.add_argument("--epoch2load", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="saves")
    parser.add_argument("--nums_new_token_i_stage_1", type=int, default=16)
    parser.add_argument("--nums_new_token_i_stage_2", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_to_generate", type=int, default=20)
    
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--num_generate_stps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5)
    parser.add_argument("--start_stp", type=int, default=10)
    parser.add_argument("--end_stp", type=int, default=35)
    return parser.parse_args()


def setup_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(args.device)
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(args.device)
    return tokenizer, uni_prompting, vq_model, model


def main():
    args = get_test_args()
    
    config = OmegaConf.load('configs/showo_demo_512x512.yaml')
    config.mode = 't2i'
    config.generation_timesteps = args.num_generate_stps
    config.guidance_scale = args.guidance_scale
    
    device = args.device
    tokenizer, uni_prompting, vq_model, model = setup_model(args, config)
    vq_model.eval()
    model.eval()

    # load from users passed arguments
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    
    concept = args.concept
    ckpt_name = args.ckpt_name
    epoch2load = args.epoch2load
    save_path = args.save_path
    
    ckpt_path = os.path.join(save_path, concept, ckpt_name)
    ckpt_embed_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_embed.pt")
    ckpt_lm_head_weight_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_weight.pt")
    ckpt_lm_head_bias_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_bias.pt")
    
    nums_new_token_i = args.nums_new_token_i_stage_1 + args.nums_new_token_i_stage_2
    new_tokens = [f"<{concept}>"] + [f"<token_{i}>" for i in range(nums_new_token_i)]
    num_new_tokens = len(new_tokens)
    # Text token count (ID 0-50304)
    original_text_vocab_size = len(tokenizer)  
    # Image token count (original ID 50305-58497)
    original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)

    original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498

    # New parameters
    new_text_vocab_size = original_text_vocab_size + num_new_tokens  # 50305 + 17 = 50322
    new_total_vocab = original_total_vocab + num_new_tokens          # 58498 + 17 = 58515

    # ------------------------------
    # Step 1: Modify the Tokenizer's vocabulary
    # ------------------------------

    # Add new tokens to positions 50305-50321
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("New token IDs:", new_token_ids)  # Should output 50305-50321

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
                model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_embed_weight[:num_new_tokens].to(model.showo.get_input_embeddings().weight.device)
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
                    new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_weight[:num_new_tokens].to(new_lm_head.weight.device)
            else:
                raise ValueError("lm_head weights do not exist!")
            # new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = lm_head.weight.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                
                if os.path.exists(ckpt_lm_head_bias_path):
                    ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                    with torch.no_grad():
                        new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_bias[:num_new_tokens].to(new_lm_head.weight.device)
                else:
                    raise ValueError("lm_head bias do not exist!")
                # new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = lm_head.bias.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
            
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

    index_no_updates = torch.ones((new_total_vocab,), dtype=torch.bool)
    index_no_updates[new_token_ids] = False
    # ------------------------------
    # Verification
    # ------------------------------
    # Check new token IDs
    print("New text token IDs:", [tokenizer.convert_tokens_to_ids(t) for t in new_tokens])  # Should output 50305-50321

    # Check new ID of an original Image Token
    sample_image_token = tokenizer.convert_ids_to_tokens(original_text_vocab_size)  # Original ID 50305
    print(f"Concept Token '{sample_image_token}' new ID:", tokenizer.convert_tokens_to_ids(sample_image_token))  # Should output 50322

    # Check embedding layer shape
    print("Embedding layer size:", model.showo.get_input_embeddings().weight.shape)  # Should show torch.Size([58515, 2048])

    # Check positions and count of True in index_no_updates, True should be new token ids
    print("False positions in index_no_updates:", torch.nonzero(~index_no_updates).squeeze())  # Should output 50305-50321
    print("True count in index_no_updates:", torch.sum(index_no_updates))  # Should output 58498
    
    model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
    config.model.showo.llm_vocab_size = len(tokenizer) - 10
    
    tokens = "".join([f"<token_{i}>" for i in range(nums_new_token_i)])
    if get_concept_info(concept)[2] == "human":
        prompt = f"A photo of {tokens} <{concept}>'s face in detail."
    else:
        prompt = f"A photo of {tokens} <{concept}> in the middle of the photo."
    
    print(prompt)
    
    for i in tqdm(range(args.num_to_generate)):
        image_tokens = torch.ones((1, config.model.showo.num_vq_tokens),
                                dtype=torch.long, device=device) * mask_token_id    # shape [batch, num_vq_tokens] [1, 256], fill with mask token
        input_ids, _ = uni_prompting(([prompt], image_tokens), 't2i_gen')   # [1, 387]
        
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * 1, image_tokens), 't2i_gen')
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
        
        steps_to_return = [args.start_stp-1, args.end_stp-1]
        with torch.no_grad():
            gen_token_ids_dict = model.t2i_generate_return_multi_steps_result(
                steps_to_return=steps_to_return,
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
        
        final_gen_token_ids = gen_token_ids_dict[config.training.generation_timesteps-1].clone()
        final_gen_token_ids = torch.clamp(final_gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        images_final_step = vq_model.decode_code(final_gen_token_ids)  # shape ([1, 3, 512, 512])
        stored_mask = None
        
        with torch.no_grad():
            for step, gen_token_ids_now_step in gen_token_ids_dict.items():
                gen_token_ids_now_step = torch.clamp(gen_token_ids_now_step, max=config.model.showo.codebook_size - 1, min=0)
                images_now_step = vq_model.decode_code(gen_token_ids_now_step)
                
                diff_mask = (gen_token_ids_now_step != final_gen_token_ids).float()
                
                if step == args.start_stp - 1:
                    stored_mask = diff_mask.clone()
                if step == args.end_stp - 1:
                    saving_mask = (diff_mask.clone() != stored_mask.clone()).float()
                    h = int(math.sqrt(diff_mask.shape[1]))
                    w = h
                    diff_mask = diff_mask.reshape(1, 1, h, w)
                    diff_mask = torch.nn.functional.interpolate(
                        diff_mask, 
                        size=(images_now_step.shape[2], images_now_step.shape[3]),
                        mode='nearest'
                    )
                    diff_mask = diff_mask.expand(-1, 3, -1, -1)
                    diff_mask = torch.where(
                        diff_mask == 1.0,
                        torch.ones_like(images_now_step),  # White color
                        images_now_step
                    )
                    diff_mask = torch.clamp((diff_mask + 1.0) / 2.0, min=0.0, max=1.0)
                    diff_mask = diff_mask * 255.0
                    diff_mask = diff_mask.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
                    save_image_dir = os.path.join(ckpt_path, "diff_masks")
                    os.makedirs(save_image_dir, exist_ok=True)
                    save_image_path = os.path.join(save_image_dir, f"{i}.png")
                    Image.fromarray(diff_mask).save(save_image_path)
  
  
if __name__ == "__main__":
    main()
