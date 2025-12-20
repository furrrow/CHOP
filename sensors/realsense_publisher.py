import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import pyrealsense2 as rs
# import cv2
import numpy as np
"""
https://answers.ros.org/question/359029/

requires python 3.8, a different conda env is created for this script.

"""

class CustomPublisher(Node):
    def __init__(self):
        super().__init__('custom_publisher')

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        # print(device.sensors)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        self.pipeline = pipeline

        # ros
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    pass
                    # continue
                else:

                    # Convert images to numpy arrays
                    # depth_image = np.asanyarray(depth_frame.get_data())
                    frame = np.asanyarray(color_frame.get_data())
                    self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame))
                    self.get_logger().info(f'image size {frame.shape}')

        finally:

            # Stop streaming
            self.pipeline.stop()
        


def main(args=None):
    rclpy.init(args=args)
    custom_publisher = CustomPublisher()
    rclpy.spin(custom_publisher)
    custom_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()