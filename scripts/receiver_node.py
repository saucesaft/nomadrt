#!/usr/bin/python3

# ros deps
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# opencv
import cv2

# nomadrt
from nomadrt.trt_model import NomadTRT

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('receiver')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Replace with the appropriate topic name
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.get_logger().info('Image Subscriber Node Initialized')

    def image_callback(self, msg):
        self.get_logger().info('Image received')
        try:
            # Convert ROS2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image (optional)
            cv2.imshow('Received Image', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    model = NomadTRT()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Image Subscriber Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
