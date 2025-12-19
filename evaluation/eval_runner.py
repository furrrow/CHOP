"""
Shared evaluation runner that processes each bag once, runs inference once per
checkpoint (finetuned / baseline), and caches the resulting trajectories on a
frame list. Metric-specific evaluators can then consume the cached paths
instead of re-running the model.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path

import json
import os
from typing import Optional, Dict, List
import glob
from collections import defaultdict

import numpy as np
import rosbag

from evaluation.proximity_evaluate import ProximityEvaluator, FrameItem as ProxFrame
from datasets.preprocess_scand_a_chop import _resample_path

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
    path_ft: Optional[np.ndarray] = None
    path_bl: Optional[np.ndarray] = None
    path_gt: Optional[np.ndarray] = None
    laserscan: Optional[np.ndarray] = None
    pos: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    cum_distance: float = 0.0
    goal_idx: int = -1

class EvalRunner:
    def __init__(
        self,
        bag_dir: str,
        output_path: str,
        test_train_split_path: str,
        pref_annotations_path: str,
        model: str,
        downsample_factor: int = 6,
        sample_goals: bool = True,
        max_distance: float = 20.0,
        min_distance: float = 2.0,
        scand: bool = True,
    ):
        self.bag_dir = bag_dir
        self.bag_name = None
        self.pref_annotations_path = pref_annotations_path
        self.pref_annotations = None

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Reuse proximity evaluator for bag processing and model inference.
        self.evaluator = ProximityEvaluator(
            bag_dir=bag_dir,
            output_path=output_path,
            test_train_split_path=test_train_split_path,
            model=model,
            downsample_factor=downsample_factor,
            sample_goals=sample_goals,
            max_distance=max_distance,
            min_distance=min_distance,
            scand=scand,
        )

    def _get_timestamps_from_expert_annotations(self):
        self.pref_file = os.path.join(self.pref_annotations_path, f"{self.bag_name}.json")
        with open(self.pref_file, "r") as f:
            self.pref_annotations = json.load(f)
        timestamps = []
        for key in self.pref_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps

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
    
    def _attach_paths(
        self, frames: List[FrameWithPaths], variant_paths: Dict[int, np.ndarray], finetuned: bool
    ):
        """Attach cached paths onto frames keyed by frame_idx."""
        for frame in frames:
            path = variant_paths.get(frame.frame_idx)
            if path is None:
                continue
            if finetuned:
                frame.path_ft = path
            else:
                frame.path_bl = path

    def _save_paths_json(self, bag_name: str, frames: List[FrameWithPaths]):
        payload = []
        for f in frames:
            payload.append(
                {
                    "frame_idx": f.frame_idx,
                    "goal_idx": f.goal_idx,
                    "pos": f.pos.tolist(),
                    "yaw": float(f.yaw),
                    "goal_pos": frames[f.goal_idx].pos.tolist() if f.goal_idx != -1 else None,
                    "path_ft": f.path_ft.tolist() if f.path_ft is not None else None,
                    "path_bl": f.path_bl.tolist() if f.path_bl is not None else None,
                }
            )

        out_file = self.output_path / f"{Path(bag_name).stem}_paths.json"
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved cached paths to {out_file}")

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

    def preprocess_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.frames = []

        self.timestamps = self._get_timestamps_from_expert_annotations()

        print(f"\n=== Processing {self.bag_name} ===")

        if "Jackal" in self.bag_name:
            self.image_topic = "/camera/rgb/image_raw/compressed"
            self.laserscan_topic = "/velodyne_2dscan"
            self.odom_topic = "/jackal_velocity_controller/odom"
        elif "Spot" in self.bag_name:
            self.image_topic = "/image_raw/compressed"
            self.laserscan_topic = "/scan"
            self.odom_topic = "/odom"
        print(f"[INFO] Using image topic: {self.image_topic}")

        # print(self.timestamps[:10])
        skip_count = 0
        timestamp_counter = 0
        with rosbag.Bag(self.bag_path, "r") as bag:

            count = 0
            scan_data = None
            last_pos = None
            pos = None
            yaw = None
            cum_distance = 0.0
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.laserscan_topic, self.odom_topic])):
                # print(int(str(t)), int(self.timestamps[timestamp_counter]), timestamp_counter, len(self.timestamps))
                if(int(str(t)) > int(self.timestamps[timestamp_counter]) and pos is None):
                    timestamp_counter += 1
                    skip_count += 1
                if topic == self.odom_topic:
                    pos, yaw = self.process_odom(msg)
                elif topic == self.image_topic:
                    cv_img = self.process_image(msg)
                elif topic == self.laserscan_topic:
                    scan_data = self.process_laserscan_msg(msg)

                if str(t) == str(self.timestamps[timestamp_counter]):
                    if scan_data is not None and pos is not None and yaw is not None:
                        if last_pos is None:
                            cum_distance = 0.0
                        else:
                            cum_distance += np.linalg.norm(pos - last_pos)
                        last_pos = pos
                        ranges, angle_min, angle_increment, range_min, range_max = scan_data
                        gt_path = self.pref_annotations.get("annotations_by_stamp", {}).get(str(t), {}).get("paths", None).get("0", None).get("points", None)
                        if gt_path is not None:
                            gt_path = np.array(gt_path, dtype=np.float32)
                            gt_path = _resample_path(gt_path, self.num_points + 1)[1:]  # Resample and drop origin
                        self.frames.append(
                                    FrameItem(
                                        frame_idx=count,
                                        image=cv_img,
                                        laserscan=ranges,
                                        angle_min=angle_min,
                                        angle_increment=angle_increment,
                                        range_min=range_min,
                                        range_max=range_max,
                                        path_gt=gt_path,
                                        pos=pos,
                                        yaw=yaw,
                                        cum_distance=cum_distance
                                    )
                                )
                        timestamp_counter += 1
                        count += 1
                if timestamp_counter >= len(self.timestamps):
                    break
            if self.sample_goals:
                self.sample_goal_indices()
                self.sample_goals = False

        print(f"[INFO] Loaded {len(self.frames)} frames from bag after skipping {skip_count} frames.")
        if not self.frames:
            print("[WARN] No frames after undersampling.")
            return
        
    def run(self):
        bag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        pref_files = sorted(glob.glob(os.path.join(self.pref_annotations_path, "*.json")))
        pref_dict = defaultdict(bool)
        for pf in pref_files:
            pref_dict[Path(os.path.basename(pf)).stem] = True

        if not bag_files:
            print(f"[ERROR] No .bag files found in {self.bag_dir}")
            return
        else:
            print(f"[INFO] Found {len(bag_files)} .bag files in {self.bag_dir}")
        with open(self.test_train_split_path, 'r') as f:
            test_train_bags = json.load(f)

        for bp in bag_files:
            self.bag_name = Path(os.path.basename(bp)).stem
            if test_train_bags.get(self.bag_name, "train") == "train" or not pref_dict.get(self.bag_name, False):
                print(f"[INFO] Skipping training bag: {self.bag_name}")
                continue
            print(f"[INFO] Processing bag: {self.bag_name}")
            self.preprocess_bag(bp)
           

            # Persist cached paths for downstream metrics.
            self._save_paths_json(bag_name)

if __name__ == "__main__":
    runner = EvalRunner(
        bag_dir="/path/to/bags",
        output_path="./outputs/evals/cached_paths",
        test_train_split_path="./data/annotations/test-train-split.json",
        pref_annotations_path="./data/annotations/preferences",
        model="omnivla",
        downsample_factor=6,
        sample_goals=True,
        min_distance=2.0,
        max_distance=20.0,
        scand=True,
    )
    runner.run()
