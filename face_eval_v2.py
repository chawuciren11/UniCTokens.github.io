import clip
import torch
import os
import numpy as np
import cv2

import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from insightface.app import FaceAnalysis

# from ldm.models.diffusion.ddim import DDIMSampler
def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


class FACEEvaluator(object):
    def __init__(self, device) -> None:
        self.device = device
        self.app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    @torch.no_grad()
    def encode_image(self, image_path: str, norm:bool = True) -> torch.Tensor:
        face_image = Image.open(image_path).convert("RGB")
        face_image = resize_img(face_image)
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info['embedding']   # np ndarrary (512,)
        
        face_emb = torch.tensor(face_emb).to(self.device)   #shape (512,)
        if norm:
            face_emb /= face_emb.norm(dim=-1, keepdim=True)
        return face_emb
    
    def get_images_features(self, images_path):
        if isinstance(images_path, str):
            images_path = [images_path]

        images = []
        for image_path in images_path:
            try:
                face_emb = self.encode_image(image_path)
            except Exception as e:
                continue
            images.append(face_emb)
        images = torch.stack(images).to(self.device)    #   shape (N, 512)
        return images   # shape (N, 512)

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_images_features(src_images) # shape (N, 512)
        gen_img_features = self.get_images_features(generated_images) # shape (M, 512)
        sim = (src_img_features @ gen_img_features.T) # shape (N, M)
        sim = sim.mean(dim=1) # shape (N,)
        sim = sim.mean().item() # shape ()
        return sim    

class SHOWO_P_FACEEvaluator(FACEEvaluator):
    def __init__(self, 
                 device, 
                 data_root='/home/hpyky/show_data',
                 work_dir='/home/hpyky/Show-o',
                 save_dir='t2i_saved',
                 resized_size=512) -> None:
        super().__init__(device)
        self.data_root = data_root
        self.gen_saved_dir = os.path.join(work_dir, save_dir)
        self.results = {}
        self.resized_size = resized_size
    
    def evaluate_concept_ref2ref(self, concept):
        
        ref_images_dir = os.path.join(self.data_root, "concept/train", concept)
        
        train_images_paths_file = os.path.join(self.data_root, "concept/train/train_images.json")
        with open(train_images_paths_file, 'r') as f:
            train_images_paths = json.load(f)
        train_images_paths_for_concept = train_images_paths[concept]
        src_images_path = [os.path.join(ref_images_dir, img_path) for img_path in train_images_paths_for_concept]
                
        avg_similarity = self.img_to_img_similarity(src_images_path, src_images_path)
        return avg_similarity
    
    def evaluate_concept(self, concept, ckpt_name, epoch2load):
        
        self.results[concept] = 0
        
        ref_images_dir = os.path.join(self.data_root, "concept/train", concept)
        
        train_images_paths_file = os.path.join(self.data_root, "concept/train/train_images.json")
        with open(train_images_paths_file, 'r') as f:
            train_images_paths = json.load(f)
        train_images_paths_for_concept = train_images_paths[concept]
        src_images_path = [os.path.join(ref_images_dir, img_path) for img_path in train_images_paths_for_concept]
        
        generated_images_dir = os.path.join(self.gen_saved_dir, concept, ckpt_name, f"{epoch2load}")
        generated_images_dir = os.path.join(generated_images_dir, f"A photo of <{concept}>.")
        if not os.path.exists(generated_images_dir):
            raise ValueError(f"Generated images directory {generated_images_dir} does not exist.")
        
        generated_images_paths = os.listdir(generated_images_dir)
        generated_images_paths = [os.path.join(generated_images_dir, img_path) for img_path in generated_images_paths]
        sim_samples_to_img = self.img_to_img_similarity(src_images_path, generated_images_paths)
        self.results[concept] = sim_samples_to_img
        print(f"Concept: {concept}, Similarity: {sim_samples_to_img}")
        return sim_samples_to_img
 
    def save_results(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {save_path}")
        
    def print_avg_results(self):
        # print global average results of all concepts
        sims = []
        for concept, sim in self.results.items():
            sims.append(sim)
        avg_sim = sum(sims) / len(sims)
        print(f"Average Similarity: {avg_sim}")
        
        
if __name__ == '__main__':
    model = SHOWO_P_FACEEvaluator(device='cuda', work_dir='./')
    ckpt_name = "test_train_s3"
    epoch2load = 20
    
    # !! If only one concept is to be tested, the concept can be directly specified, 
    # !! concept_list_file = None, concepts = ["concept_name"].
    concept_list_file = "path/to/uni_c_tokens_data/concepts_list.json"
    with open(concept_list_file, 'r') as f:
        concepts = json.load(f)
    sims = []
    for concept in tqdm(concepts):
        # sim = model.evaluate_concept_ref2ref(concept)
        # sims.append(sim)
        # print(f"Concept: {concept}, Similarity: {sim}")
        model.evaluate_concept(concept, ckpt_name, epoch2load)
    model.save_results(f"face_eval_results_{ckpt_name}_{epoch2load}.json")
    model.print_avg_results()


