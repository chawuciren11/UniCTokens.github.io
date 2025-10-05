from eval.api import MultiModalChatSession
import json
import requests
import base64
import requests
from PIL import Image
import io
import json
import os
import re
import random
from tqdm import tqdm


class GPTEvaluator(object):
    def __init__(self, system_prompt) -> None:
        self.t2i_saved_dir = "./t2i_saved"
        self.data_root = "path/to/uni_c_tokens_data"
        self.num_per_condition = 4
        self.session = MultiModalChatSession(system_prompt)
        self.results = {}
    
    def evaluate_concept(self, concept, ckpt_name, epoch2load):
        
        self.results[concept] = {}

        t2i_images_dir = os.path.join(self.t2i_saved_dir, concept, ckpt_name, f"{epoch2load}")
        
        concept_info_file = os.path.join(self.data_root, "concept/train", concept, "info.json")
        with open(concept_info_file, 'r') as f:
            info = json.load(f)
        info_str = f"name: <{concept}>\n"
        info_str += f"info: {info['info']}" + "\n" + "extra_info: "
        for idx in range(len(info["extra_info"])):
            if idx != len(info["extra_info"]) - 1:
                info_str += info["extra_info"][idx] + " "
            else:
                info_str += info["extra_info"][idx] + "\n"
        
        concept_test_conditions_file = os.path.join(self.data_root, "concept/test", concept, "t2i_conditions.json")
        with open(concept_test_conditions_file, 'r') as f:
            concept_test_conditions = json.load(f)
        concept_test_conditions = concept_test_conditions[1:]
        for condition in concept_test_conditions:
            self.results[concept][condition] = {}
            t2i_images_dir_condition = os.path.join(t2i_images_dir, condition)
            if not os.path.exists(t2i_images_dir_condition):
                raise ValueError(f"t2i_images_dir_condition {t2i_images_dir_condition} does not exist")
            t2i_images = [img for img in os.listdir(t2i_images_dir_condition) if img.endswith(".png")]
            t2i_images_path = [os.path.join(t2i_images_dir_condition, img) for img in t2i_images]
            images_to_evaluate = random.sample(t2i_images_path, min(self.num_per_condition, len(t2i_images_path)))
            for img_path in images_to_evaluate:
                prompt = "Concept Information:\n" + info_str + "\n" + "Generated image prompt:\n" + condition + "\n" + "Please give a score according to the rules, no explanation needed.\n" + "Score:"
                
                score = self.session.chat(prompt, image_paths=[img_path])
                try:
                    score = float(score.strip())
                except:
                    continue
                print(f"Concept: {concept}, Condition: {condition}, Image: {img_path}, Score: {score}")
                self.results[concept][condition][img_path] = score
            self.results[concept][condition]["avg_score"] = sum(self.results[concept][condition].values()) / len(self.results[concept][condition])
        self.results[concept]["avg_score"] = sum([self.results[concept][condition]["avg_score"] for condition in self.results[concept]]) / len(self.results[concept])
 
    def save_results(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {save_path}")
        
    def print_avg_results(self):
        # print global average results of all concepts
        avg_score = 0
        for concept, item in self.results.items():
            avg_score += item["avg_score"]
        avg_score /= len(self.results)
        print(f"Global average score: {avg_score}")
        
        
if __name__ == '__main__':
    model = GPTEvaluator(system_prompt="""
                         You are an expert in evaluating model-generated images. This evaluation mainly assesses whether T2I generated images match the given concept description's extra_info.
                         The concept description format is as follows:
                         name: <concept name>
                         info: <concept description>
                         extra_info: <additional information>
                         You need to evaluate whether the generated image matches the description based on extra_info and the prompt used to generate the image, giving a score of 0, 0.5, or 1, where 0 means does not match, 0.5 means partially matches, and 1 means matches.
                         For example: If the extra_info of concept <bingbing> is that he has a red dress, and the generated image prompt is "A photo of <bingbing> wearing her dress.", if there is a red dress in the image, you should give a score of 1, otherwise give a score of 0.
                         You only need to give the score, no explanation needed.
                         """)
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
    model.save_results(f"t2i_results_{ckpt_name}_{epoch2load}.json")
    model.print_avg_results()
    