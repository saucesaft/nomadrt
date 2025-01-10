#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class VideoPublisher(Node):
    def __init__(self, video_file):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.video_file = video_file

        # Open the video file
        if not os.path.exists(video_file):
            self.get_logger().error(f"Video file '{video_file}' does not exist!")
            rclpy.shutdown()
            return

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file '{video_file}'!")
            rclpy.shutdown()
            return

        # Set a timer to publish frames at the correct frame rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timer_period = 1.0 / fps  # Timer period in seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(f"Publishing video from '{video_file}' at {fps} FPS")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video file reached.")
            rclpy.shutdown()
            return

        try:
            # Convert OpenCV frame to ROS2 Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        self.get_logger().info("VideoPublisher node shutdown complete.")


def main(args=None):
    rclpy.init(args=args)

    video_file = 'dashcam.webm'  # Replace with the path to your video file
    node = VideoPublisher(video_file)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Video Publisher Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
