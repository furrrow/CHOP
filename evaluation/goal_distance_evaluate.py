from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import rosbag

from evaluation.base_evaluator import BaseEvaluator


@dataclass
class FrameItem:
    frame_idx: int
    image: np.ndarray
    angle_min: float
    yaw: float
    cum_distance: float = 0.0
    goal_idx: int = -1


class GoalDistanceEvaluator(BaseEvaluator):
    """
    Evaluates how far the model's final predicted waypoint is from the sampled goal.
    Only uses camera + odom topics
    """

    def __init__(
        self,
        bag_dir: str,
        output_path: str,
        test_train_split_path: str,
        model: str,
        downsample_factor: int = 6,
        sample_goals: bool = True,
        max_distance: float = 20.0,
        min_distance: float = 2.0,
        scand: bool = True,
    ):
        super().__init__(
            bag_dir,
            output_path,
            test_train_split_path,
            model,
            downsample_factor,
            sample_goals,
            max_distance,
            min_distance,
        )

        self._eval_name = "goal_distance"
        if scand:
            self._eval_name += "_scand"

        self.frames: list[FrameItem] = []
        self._open_output_files()

    def process_odom(self, msg):
        quaternion = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        return pos, yaw

    def process_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        self.frames = []

        print(f"\n=== Processing {self.bag_name} ===")

        # Topic selection
        if "Jackal" in self.bag_name:
            self.image_topic = "/camera/rgb/image_raw/compressed"
            self.odom_topic = "/jackal_velocity_controller/odom"
        elif "Spot" in self.bag_name:
            self.image_topic = "/image_raw/compressed"
            self.odom_topic = "/odom"
        else:
            raise ValueError(f"Unrecognized platform for bag {self.bag_name}")

        print(f"[INFO] Using image topic: {self.image_topic}")

        with rosbag.Bag(bag_path, "r") as bag:
            count = 0
            cam_frames = 0
            last_pos = None
            pos = None
            yaw = None
            cum_distance = 0.0

            for _, (topic, msg, _) in enumerate(
                bag.read_messages(topics=[self.image_topic, self.odom_topic])
            ):
                if topic == self.image_topic:
                    if cam_frames % self.downsample_factor == 0:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        if pos is not None and yaw is not None:
                            if last_pos is None:
                                cum_distance = 0.0
                            else:
                                cum_distance += np.linalg.norm(pos - last_pos)
                            last_pos = pos

                            self.frames.append(
                                FrameItem(
                                    frame_idx=count,
                                    image=cv_image,
                                    pos=pos,
                                    yaw=yaw,
                                    cum_distance=cum_distance,
                                )
                            )
                            count += 1
                    cam_frames += 1

                if topic == self.odom_topic:
                    pos, yaw = self.process_odom(msg)

            print(f"[INFO] Processed {len(self.frames)} frames from {self.bag_name}")

            if self.sample_goals:
                self.sample_goal_indices()
                self.sample_goals = False

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing goal distance for {self.bag_name}")
        distances = []
        model, noise_scheduler = self.load_model(finetuned=finetuned)

        for frame in self.frames:
            if frame.goal_idx == -1:
                continue

            goal_frame = self.frames[frame.goal_idx]
            path_xy = self.run_inference(
                model, frame, goal_frame, noise_scheduler=noise_scheduler
            )

            if path_xy is None or len(path_xy) == 0:
                continue

            final_local = np.array(path_xy[-1])
            # Rotate from robot frame to world frame using current yaw, then translate.
            c, s = np.cos(frame.yaw), np.sin(frame.yaw)
            rot = np.array([[c, -s], [s, c]])
            final_global = frame.pos + rot @ final_local

            dist_to_goal = float(np.linalg.norm(final_global - goal_frame.pos))
            distances.append(dist_to_goal)

        return distances


if __name__ == "__main__":
    evaluator = GoalDistanceEvaluator(
        bag_dir="/path/to/bags",
        output_path="./outputs/evals",
        test_train_split_path="./data/annotations/test-train-split.json",
        model="omnivla",
        downsample_factor=6,
        sample_goals=True,
        min_distance=2.0,
        max_distance=20.0,
        scand=True,
    )
    evaluator.run()
