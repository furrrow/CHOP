#!/usr/bin/env python3
import math
import threading
from typing import List, Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation as R
from deployment.utils.transformations import start_to_current

class PathManagerNode(Node):
    def __init__(self):
        super().__init__("path_manager")

        self.qos_profile  = QoSProfile(
                                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST,  
                                depth=15  
                            )
        # ---- Params ----
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("behind_margin", 0.09)    # pop if x < -margin
        self.declare_parameter("reach_radius", 0.1)     # pop if dist < radius (extra robustness)

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.behind_margin = float(self.get_parameter("behind_margin").value)
        self.reach_radius = float(self.get_parameter("reach_radius").value)

        # ---- State ----
        self._lock = threading.Lock()

        # latest homogeneous from base to world
        self._current_T_w : Optional[np.ndarray] = None
        # snapshot homogeneous from base to world
        self._start_T_w: Optional[np.ndarray] = None

        # path points stored in START robot frame (what model outputs)
        self._path_start_xy: np.ndarray = np.empty((0, 2))

        # ---- ROS I/O ----
        self.create_subscription(Odometry, "/odom", self.on_odom, self.qos_profile)
        self.create_subscription(Empty, "/started", self.on_started, self.qos_profile)
        self.create_subscription(Path, "/path", self.on_path, self.qos_profile)
        self.create_subscription(Empty, "/req_goal", self.on_req_goal, self.qos_profile)

        self.pub_next_goal = self.create_publisher(PoseStamped, "/next_goal", 10)
        self.pub_active_path = self.create_publisher(Path, "/active_path", 10)

        self.get_logger().info(
            f"PathManagerNode running. base_frame={self.base_frame}, "
            f"behind_margin={self.behind_margin}, reach_radius={self.reach_radius}"
        )

    # ---------------- callbacks ----------------

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        c, s = np.cos(yaw), np.sin(yaw)
        with self._lock:
            self._current_T_w = np.eye(3)
            self._current_T_w[:2, :2] = [[c, -s], [s, c]]
            self._current_T_w[:2, 2]  = [p.x, p.y]

    def on_started(self, _msg: Empty):
        with self._lock:
            if self._current_T_w is None:
                self.get_logger().warn("Got /started but no odom yet; cannot snapshot start pose.")
                return
            self._start_T_w = self._current_T_w.copy()
        self.get_logger().info("Saved start odom pose from /started.")

    def on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty /path.")
            with self._lock:
                self._path_start_xy = np.empty((0, 2))
            return

        with self._lock:
            if self._start_T_w is None:
                self.get_logger().warn("Received /path but no /started pose saved; ignoring.")
                return
            # store points in start frame (model frame)
            self._path_start_xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])

        # Rebase to current frame + drop behind + publish
        self._drop_behind_and_publish()

    def on_req_goal(self, _msg: Empty):
        # Planner says it reached the current goal -> pop front and publish next
        with self._lock:
            if self._path_start_xy.size != 0:
                self._path_start_xy = self._path_start_xy[1:]
        self._drop_behind_and_publish()

    # ---------------- core logic ----------------

    def _drop_behind_and_publish(self):
        with self._lock:
            if self._path_start_xy.size == 0:
                return
            T_s = self._start_T_w
            T_c = self._current_T_w
            pts_start = self._path_start_xy.copy()

        if T_s is None or T_c is None:
            return
        
        pts_cur = start_to_current(T_s, T_c, pts_start)  # (N,2)

        dists = np.linalg.norm(pts_cur, axis=1)
        behind = pts_cur[:, 0] < -self.behind_margin
        reached = dists < self.reach_radius
        keep = ~(behind | reached)

        with self._lock:
            self._path_start_xy = self._path_start_xy[keep, :]
        pts_cur = pts_cur[keep, :]

        # current frame -> world frame
        pts_w = []
        T_w_c = T_c
        pts_c_h = np.vstack([pts_cur.T, np.ones(pts_cur.shape[0])])  # (3,N)
        pts_w   = (T_w_c @ pts_c_h)[:2, :].T  # (N,2)

        self._publish_if_available(pts_w)

    def _publish_if_available(self, pts_world: np.ndarray):

        if pts_world.size == 0:
            return
        gx, gy = pts_world[0]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.world_frame
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.pub_next_goal.publish(goal)

        path_msg = Path()
        path_msg.header = goal.header
        for x_w, y_w in pts_world:
            ps = PoseStamped()
            ps.header = goal.header
            ps.pose.position.x = float(x_w)
            ps.pose.position.y = float(y_w)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_active_path.publish(path_msg)

def main():
    rclpy.init()
    node = PathManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
