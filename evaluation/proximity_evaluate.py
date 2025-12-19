from typing import Optional
from tqdm import tqdm

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import rosbag
from evaluation.metrics.obs_proximity import min_clearance_to_obstacles_ls
from evaluation.base_evaluator import BaseEvaluator

@dataclass
class FrameItem:
    frame_idx: int
    image: np.ndarray
    laserscan: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    pos: np.ndarray
    yaw: float
    cum_distance: float = 0.0
    goal_idx: int = -1

class ProximityEvaluator(BaseEvaluator):
    def __init__(self, bag_dir: str, output_path: str, test_train_split_path: str, model: str, fov_angle: float = 90.0, downsample_factor: int = 6, 
                 sample_goals: bool = True, max_distance: float = 20.0, min_distance: float = 2.0, scand : bool = True):
        super().__init__(bag_dir, output_path, test_train_split_path, model, 
                         downsample_factor, sample_goals, max_distance, min_distance)
        
        self._eval_name = "proximity"
        if scand:
            self._eval_name += "_scand"

        self.fov_angle = fov_angle
        self.frames: list[FrameItem] = []
        self._open_output_files()

    def process_laserscan_msg(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        range_min = float(msg.range_min)
        range_max = min(float(msg.range_max), self.max_distance)

        # Replace NaNs with inf
        ranges = np.nan_to_num(ranges, nan=np.inf)
        # Clip very large values
        ranges[ranges > range_max] = np.inf

        # Apply FOV mask (centered at 0 yaw)
        if self.fov_angle < 360.0:
            angles = angle_min + np.arange(len(ranges)) * angle_increment
            half_fov = np.deg2rad(self.fov_angle) / 2.0
            mask = (angles >= -half_fov) & (angles <= half_fov)
            ranges = np.where(mask, ranges, np.inf)

        return ranges, angle_min, angle_increment, range_min, range_max

    def process_odom(self, msg):
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        return pos, yaw

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
            scan_data = None
            last_pos = None
            pos = None
            yaw = None
            cum_distance = 0.0
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.laserscan_topic, self.odom_topic])):
                if topic == self.image_topic:
                    if cam_frames % self.downsample_factor == 0:

                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        if scan_data is not None and pos is not None and yaw is not None:
                            if last_pos is None:
                                cum_distance = 0.0
                            else:
                                cum_distance += np.linalg.norm(pos - last_pos)
                            last_pos = pos
                            ranges, angle_min, angle_increment, range_min, range_max = scan_data
                            self.frames.append(
                                FrameItem(
                                    frame_idx=count,
                                    image=cv_image,
                                    laserscan=ranges,
                                    angle_min=angle_min,
                                    angle_increment=angle_increment,
                                    range_min=range_min,
                                    range_max=range_max,
                                    pos=pos,
                                    yaw=yaw,
                                    cum_distance=cum_distance,
                                )
                            )
                            count += 1
                    cam_frames += 1

                if topic == self.laserscan_topic:
                    scan_data = self.process_laserscan_msg(msg)

                if topic == self.odom_topic:
                    pos, yaw = self.process_odom(msg)

            print(f"[INFO] Processed {len(self.frames)} frames from {self.bag_name}")
            
            if self.sample_goals:
                self.sample_goal_indices()
                self.sample_goals = False

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing obstacle proximity for {self.bag_name}")
        min_clearances = []
        model, noise_scheduler = self.load_model(finetuned=finetuned)

        for frame in tqdm(self.frames, desc=f"Eval {self.bag_name}", unit="frame"):
            if frame.goal_idx == -1:
                continue
            
            goal_frame = self.frames[frame.goal_idx]
            path_xy = self.run_inference(model, frame, goal_frame, noise_scheduler=noise_scheduler)

            angle_max = frame.angle_min + (len(frame.laserscan) - 1) * frame.angle_increment
            min_clearance = min_clearance_to_obstacles_ls(
                path_xy=path_xy,
                laserscan=frame.laserscan,
                angle_increment=frame.angle_increment,
                angle_min=frame.angle_min,
                angle_max=angle_max,
                range_min=frame.range_min,
                range_max=frame.range_max,
            )

            if min_clearance != np.inf:
                min_clearances.append(min_clearance)
        return min_clearances

if __name__ == "__main__":
    evaluator = ProximityEvaluator(
        bag_dir="/media/beast-gamma/Media/Datasets/SCAND/rosbags/",
        output_path="./outputs/evals",
        test_train_split_path="./data/annotations/test-train-split.json",
        model="omnivla",
        fov_angle=90.0,
        downsample_factor=6,
        sample_goals=True,
        min_distance=2.0,
        max_distance=20.0,
        scand=True
    )
    evaluator.run()
