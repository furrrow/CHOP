from dataclasses import dataclass
import glob
import os
import json
import argparse
from typing import Optional
import yaml

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path

from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model
from policy_sources.omnivla.inference.run_omnivla_modified import Inference
from policy_sources.visualnav_transformer.deployment.src.utils import transform_images
from policy_sources.visualnav_transformer.train.vint_train.training.train_utils import get_action
from PIL import Image as PILImage

@dataclass
class FrameItem:
    frame_idx: int
    image: np.ndarray
    laserscan: Optional[np.ndarray] = None
    pos: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    cum_distance: float = 0.0
    goal_idx: int = -1

class InferenceConfigOriginal:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000
    vla_path: str = "./weights/omnivla-finetuned-cast"   
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

class InferenceConfigFinetuned:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./weights/omnivla-finetuned-chop"   
    resume_step: Optional[int] = 222500
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
class BaseEvaluator():
    def __init__(self, bag_dir: str, output_path: str, test_train_split_path: str, model: str, downsample_factor: int = 6, 
                 sample_goals: bool = True, max_distance: float = 20.0, min_distance: float = 2.0):
        self.bag_dir = bag_dir
        self.bridge = CvBridge()
        self.test_train_split_path = test_train_split_path
        self.output_path = output_path
        self.downsample_factor = downsample_factor
        self.model_name = model
        
        self.sample_goals = sample_goals
        self.max_distance = max_distance
        self.min_distance = min_distance

        self.frames: list[FrameItem] = []

        parser = argparse.ArgumentParser(description="Config to model")
        parser.add_argument(
            "--config",
            "-c",
            default=f"configs/chop_{self.model_name}_vnt.yaml",
            type=str,
            help="Path to the config file in config folder",
        )
        args = parser.parse_args()

        if self.model_name in {"vint", "gnm", "nomad"}:
            with open("configs/chop_default_vnt.yaml", "r") as f:
                default_config = yaml.safe_load(f)

            config = default_config

            with open(args.config, "r") as f:
                user_config = yaml.safe_load(f)

            config.update(user_config)
        else:
            with open("configs/chop_omnivla.yaml", "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        self.context_frames = self.config.get("context_size", 0)
        self._eval_name = "base_eval"
        self.log_file_path = None
        self.log_file_path_ft = None
        self.bag_name = None

        self.vla_config = InferenceConfigOriginal()
        self.vla_config_finetuned = InferenceConfigFinetuned()

    def _open_output_files(self):
        output_dir = os.path.join(self.output_path, self._eval_name)
        os.makedirs(output_dir, exist_ok=True)

        self.log_file_path = os.path.join(output_dir, f"{self._eval_name}_results.log")
        self.log_file_path_ft = os.path.join(output_dir, f"{self._eval_name}_results_finetuned.log")

    def load_model(self, finetuned: bool = True, ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        noise_scheduler = None
        
        if self.config.get("model_type") in {"vint", "gnm", "nomad"}:
            ckpt_path = self.config["chop_finetuned_path"] if finetuned else self.config["pretrained_model_path"]
            model = deployment_load_model(str(ckpt_path), self.config, device)

            if self.model_name == "nomad":
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config["num_diffusion_iters"],
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    prediction_type='epsilon'
                )

        elif self.model_name == "omnivla":
            if finetuned:
                vla_config = self.vla_config_finetuned
            else:
                vla_config = self.vla_config

            model = Inference(save_dir="./inference",
                            ego_frame_mode=True,
                            save_images=False, 
                            radians=True,
                            vla_config=vla_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        if model is None:
            raise RuntimeError("Model failed to initialize.")

        # Some wrappers (e.g., OmnivLA Inference) are not nn.Modules; guard attribute usage.
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        return model, noise_scheduler

    def run_inference(self, model, frame: FrameItem, goal_frame: FrameItem, noise_scheduler=None):
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_idx = max(0, frame.frame_idx - self.context_frames)
        context_imgs = [f.image[:, :, ::-1] for f in self.frames[start_idx:frame.frame_idx + 1]]  # BGR -> RGB
        context_pil = [PILImage.fromarray(img) for img in context_imgs]
        goal_pil = PILImage.fromarray(goal_frame.image[:, :, ::-1])

        if self.model_name in {"vint", "gnm"}:
            obs_tensor = transform_images(context_pil, self.config["image_size"])
            goal_tensor = transform_images(goal_pil, self.config["image_size"])
            obs_tensor = obs_tensor.to(device)
            goal_tensor = goal_tensor.to(device)
            with torch.no_grad():
                _, action_pred = model(obs_tensor, goal_tensor)
            path_xy = action_pred[0, :, :2].detach().cpu().numpy()
        elif self.model_name == "nomad":
            if noise_scheduler is None:
                raise RuntimeError("Noise scheduler required for NoMaD inference.")
            obs_images = transform_images(context_pil, self.config["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            goal_tensor = transform_images(goal_pil, self.config["image_size"], center_crop=False).to(device)
            mask = torch.zeros(1, device=device).long()

            obsgoal_cond = model('vision_encoder', obs_img=obs_images, goal_img=goal_tensor, input_goal_mask=mask)
            obs_cond = obsgoal_cond

            num_diffusion_iters = self.config["num_diffusion_iters"]
            noise_scheduler.set_timesteps(num_diffusion_iters)
            with torch.no_grad():
                noisy_action = torch.randn((1, self.config["len_traj_pred"], 2), device=device)
                naction = noisy_action
                for k in noise_scheduler.timesteps:
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = get_action(naction)
            path_xy = naction[0, :, :2].detach().cpu().numpy()
        elif self.model_name == "omnivla":
            cur_img = PILImage.fromarray(frame.image[:, :, ::-1]) #BGR to RGB
            cur_pos = frame.pos
            cur_yaw = frame.yaw

            goal_img = PILImage.fromarray(goal_frame.image[:, :, ::-1])
            goal_pos = goal_frame.pos
            goal_yaw = goal_frame.yaw

            model.update_current_state(cur_img, cur_pos, cur_yaw)
            model.update_goal(goal_image_PIL=goal_img, 
                                    goal_utm=goal_pos,
                                    goal_compass=goal_yaw, 
                                    lan_inst_prompt=None)
            model.run()
            waypoints = model.waypoints.reshape(-1, model.waypoints.shape[-1])
            path_xy = waypoints[:, :2] * model.metric_waypoint_spacing  # Convert to meters
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Normalize to shape (N, 2) for downstream metrics
        path_xy = np.asarray(path_xy)
        if path_xy.ndim == 3:
            path_xy = path_xy.reshape(-1, path_xy.shape[-1])
        if path_xy.shape[-1] > 2:
            path_xy = path_xy[:, :2]

        return path_xy
    
    def sample_goal_indices(self):
        goal_dist = np.random.uniform(self.min_distance, self.max_distance, size=len(self.frames))
        for i in range(self.context_frames, len(self.frames)-1):
            cur_dist = self.frames[i].cum_distance
            for j in range(i + 1, len(self.frames)):
                next_dist = self.frames[j].cum_distance
                if next_dist - cur_dist >= goal_dist[i]:
                    self.frames[i].goal_idx = j
                    break
            else:
                self.frames[i].goal_idx = j

    def process_bag(self):
        # Placeholder for bag processing logic
        return

    def analyze_bag(self):
        # Placeholder for analysis logic
        return 

    def log_metrics(self, evaluations, finetuned: bool = True):
        if self.log_file_path is None or self.log_file_path_ft is None:
            self._open_output_file()

        if finetuned:
            log_file_path = self.log_file_path_ft
        else:
            log_file_path = self.log_file_path

        mean, std = self.calculate_statistics(evaluations)
        if mean is None or std is None:
            return

        bag_stem = Path(self.bag_name).stem
        evaluations = [float(val) for val in evaluations] if evaluations else []

        with open(log_file_path, "a") as f:
            json_record = {
                "bag": bag_stem,
                "frame_count": len(self.frames),
                "data": evaluations,
                "mean": mean,
                "std": std,
            }
            f.write(json.dumps(json_record) + "\n")  # one object per line


            print(f"[INFO] Evaluation results for {self.bag_name} {self._eval_name}: {mean:.4f} ± {std:.4f}\n")

    def calculate_statistics(self, evaluations: list):
        if not evaluations:
            print("[WARNING] No evaluations to calculate statistics.")
            return None, None

        mean = float(np.mean(evaluations))
        std = float(np.std(evaluations))
        return mean, std

    def run(self):
        bag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))

        if not bag_files:
            print(f"[ERROR] No .bag files found in {self.bag_dir}")
            return
        else:
            print(f"[INFO] Found {len(bag_files)} .bag files in {self.bag_dir}")
        with open(self.test_train_split_path, 'r') as f:
            test_train_bags = json.load(f)

        for bp in bag_files:
            self.bag_name = Path(os.path.basename(bp)).stem
            if test_train_bags.get(self.bag_name, "train") == "train":
                print(f"[INFO] Skipping training bag: {self.bag_name}")
                continue
            
            print(f"\n[INFO] Evaluating bag: {self.bag_name}")
            self.process_bag(bp)
            eval_finetuned = self.analyze_bag(finetuned=True)
            eval_pretrained = self.analyze_bag(finetuned=False)
            self.log_metrics(eval_pretrained, finetuned=False)
            self.log_metrics(eval_finetuned, finetuned=True)
            self.sample_goals = True

        print(f"\n[DONE] Annotations written to {self.output_path}")
