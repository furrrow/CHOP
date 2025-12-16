import glob
import os
import json
import argparse
import yaml

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

import rosbag
from sensor_msgs.msg import LaserScan
from evaluation.metrics.obs_proximity import min_clearance_to_obstacles_ls
from pathlib import Path
from cv_bridge import CvBridge

from policy_sources.visualnav_transformer.train.vint_train.models.gnm.gnm import GNM
from policy_sources.visualnav_transformer.train.vint_train.models.vint.vint import ViNT
from policy_sources.visualnav_transformer.train.vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model

class BaseEvaluator:
    def __init__(self, bag_dir: str, output_path: str, test_train_split_path: str, bags_to_skip: str,
                 model: str, fov_angle: float = 90.0, downsample_factor: int = 6, sample_goals: bool = True,
                 max_distance: float = 20.0, min_distance: float = 2.0):
        self.bag_dir = bag_dir
        self.bridge = CvBridge()
        self.test_train_split_path = test_train_split_path
        self.bags_to_skip = bags_to_skip
        self.output_path = output_path
        self.downsample_factor = downsample_factor
        self.model_name = model
        
        self.sample_goals = sample_goals
        self.fov_angle = fov_angle
        self.max_distance = max_distance
        self.min_distance = min_distance

        self.frames: list[FrameItem] = []

        parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
        parser.add_argument(
            "--config",
            "-c",
            default=f"configs/chop_{self.model_name}_vnt.yaml",
            type=str,
            help="Path to the config file in train_config folder",
        )
        args = parser.parse_args()

        with open("configs/chop_default_vnt.yaml", "r") as f:
            default_config = yaml.safe_load(f)

        config = default_config

        with open(args.config, "r") as f:
            user_config = yaml.safe_load(f)

        config.update(user_config)
        self.config = config
        self.context_frames = self.config.get("context_size", 0)

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = self.config.get("pretrained_model_path")
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path

        model = None
        state_dict = None

        if self.config.get("model_type") in {"vint", "gnm", "nomad"}:
            model = deployment_load_model(str(ckpt_path), self.config, device)

            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

        elif self.model_name == "omnivla":
            model = Inference(save_dir="./inference",
                            ego_frame_mode=True,
                            save_images=False, 
                            radians=True,
                            vla_config=InferenceConfig())
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        if model is None:
            raise RuntimeError("Model failed to initialize.")

        model.to(device)
        model.eval()
        model.requires_grad_(False)
        return model

    def process_laserscan_msg(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        # Replace NaNs with inf
        ranges = np.nan_to_num(ranges, nan=np.inf)
        # Clip very large values (optional)
        ranges[ranges > self.max_distance] = np.inf

        return ranges
    
    def process_odom(self, msg):        
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        return pos
    
    def preprocess_data(self):
        return 

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

    def process_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.frames = []

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

        with rosbag.Bag(bag_path, "r") as bag:

            count = 0
            cam_frames = 0
            scan_points = None
            last_pos = None
            pos = None
            cum_distance = 0.0
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.laserscan_topic, self.odom_topic])):
                if topic == self.image_topic:
                    if cam_frames % self.downsample_factor == 0:

                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        if scan_points is not None and pos is not None:
                            if last_pos == None:
                                cum_distance = 0.0
                            else:
                                cum_distance += np.linalg.norm(pos - last_pos)
                            last_pos = pos
                            self.frames.append(FrameItem(frame_idx=count, image=cv_image, laserscan=scan_points, cum_distance=cum_distance))
                            count += 1
                    cam_frames += 1

                if topic == self.laserscan_topic:
                    scan_points = self.process_laserscan_msg(msg)

                if topic == self.odom_topic:
                    pos = self.process_odom(msg)

            print(f"[INFO] Processed {len(self.frames)} frames from {self.bag_name}")
            
            if self.sample_goals:
                self.sample_goal_indices()
                self.sample_goals = False

    def analyze_bag(self):
        print(f"[INFO] Analyzing obstacle proximity for {self.bag_name}")
        min_clearances = []
        
        for i, frame in enumerate(self.frames):
            if frame.goal_idx == -1:
                continue

            
            path_xy = np.array(path_xy)

            min_clearance = min_clearance_to_obstacles_ls(
                path_xy=path_xy,
                laserscan=frame.laserscan,
                angle_increment=0.005,
                angle_min=-np.pi,
                angle_max=np.pi,
                range_min=0.2,
                range_max=self.max_distance,
            )

            if min_clearance != np.inf:
                min_clearances.append(min_clearance)
        return min_clearances
                
    def run(self):
        bag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))

        if not bag_files:
            print(f"[ERROR] No .bag files found in {self.bag_dir}")
            return
        
        with open(self.test_train_split_path, 'r') as f:
            test_train_bags = json.load(f)

        for bp in bag_files:
            bag_name = os.path.basename(bp)
            if test_train_bags.get(bag_name, "train") == "train":
                continue
            
            self.process_bag(bp)
            min_clearance_1 = self.analyze_bag()
            min_clearance_2 = self.analyze_bag()
            self.sample_goals = True

        print(f"\n[DONE] Annotations written to {self.output_path}")

if __name__ == "__main__":
    evaluator = ProximityEvaluator(
        bag_dir="/path/to/bags",
        output_path="/path/to/output.json",
        test_train_split_path="/path/to/split.json",
        bags_to_skip="",
        model="vint",
        fov_angle=90.0,
        downsample_factor=6,
        sample_goals=True,
        max_distance=20.0,
    )
    evaluator.run()
