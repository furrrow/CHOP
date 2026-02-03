import json
import math
from typing import Optional
import numpy as np
import cv2

def load_calibration(json_path: str):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if data is None or "H_cam_bl" not in data:
        raise ValueError(f"Missing H_cam_bl in {json_path}")

    h = data["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [ 0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll),  math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3]  = np.array([xt, yt, zt], dtype=np.float64)

    fx = data["Intrinsics"]["fx"]
    fy = data["Intrinsics"]["fy"]
    cx = data["Intrinsics"]["cx"]
    cy = data["Intrinsics"]["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam


def overlay_path(pts_cur: np.ndarray, img: Optional[np.ndarray] = None, cam_matrix: Optional[np.ndarray] = None,
                 T_cam_from_base: Optional[np.ndarray] = None):
    if pts_cur.size == 0:
        return
    if cam_matrix is None or T_cam_from_base is None:
        return
    if img is None:
        return

    # Points in base frame -> camera frame -> pixels
    pts_3d = np.hstack([pts_cur, np.zeros((pts_cur.shape[0], 1))])  # z=0 in base frame
    R_cb = T_cam_from_base[:3, :3]
    t_cb = T_cam_from_base[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cb)

    img_pts, _ = cv2.projectPoints(pts_3d, rvec, t_cb, cam_matrix, None)
    img_pts = img_pts.reshape(-1, 2)

    # Keep points in front of camera and inside image
    pts_cam = (R_cb @ pts_3d.T + t_cb.reshape(3, 1)).T
    valid_z = pts_cam[:, 2] > 0
    h, w = img.shape[:2]
    valid_xy = (
        (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
        (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
    )
    keep = valid_z & valid_xy
    if not keep.any():
        return

    pts_pix = img_pts[keep].astype(int)
    overlay = img.copy()
    if len(pts_pix) >= 2:
        cv2.polylines(overlay, [pts_pix], isClosed=False, color=(0, 0, 255), thickness=2)
    else:
        for pt in pts_pix:
            cv2.circle(overlay, tuple(pt), radius=3, color=(0, 0, 255), thickness=-1)

    return overlay
