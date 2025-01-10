#!/usr/bin/python3

# ros deps
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# python deps
import time
from collections import deque

# ml deps
import cv2
import numpy as np

# nomadrt
from nomadrt.trt_model import NomadTRT
import nomadrt.utils as preproc

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('receiver')
        
        self.declare_parameter('model_path', 'weights')
        self.model = NomadTRT( self.get_parameter('model_path').get_parameter_value().string_value, self.get_logger() )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Replace with the appropriate topic name
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.trajectory_publisher = self.create_publisher(Path, '/nomadrt/trajectory', 10)

        self.bridge = CvBridge()

        self.image_size = (96 ,96)
        self.context_size = 4
        self.dq = deque( maxlen=self.context_size )
        self.queue_timer = self.create_timer(1/30, self.timer_callback)

        # TODO change goal to be an actual image
        goal_shape = self.model.encoder_session.inputs['goal_img']['shape']
        self.goal = np.random.randint(2, size=goal_shape)
        self.mask = np.zeros(1, dtype=int).repeat(goal_shape[0])

        self.get_logger().info('receiver node start')

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img = preproc.center_crop_and_resize(img, self.image_size)
            img = preproc.prepare_image(img)

            self.dq.append(img)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def timer_callback(self):

        if len(self.dq) < self.context_size:
            return
        
        # img = self.dq[-1]

        obs = np.concatenate(self.dq, axis=1)
        # self.get_logger().info(f"Observation ready {obs.shape}")

        start_t = time.process_time() 
        dist, traj = self.model.predict(obs, self.goal, self.mask)

        traj = traj[:, :5, :2]
 
        self.get_logger().info(f"Inference done {time.process_time() - start_t}s")
        # self.get_logger().info(f"Distance: {dist[-1]}")
        # self.get_logger().info(f"Trajectory: {traj[-1]}")

        choosen_path = traj[-1] # choose the best path with heuristics TODO
        
        msg = Path()
        msg.header.frame_id = "world"

        for section in choosen_path:
            p = PoseStamped()
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = section[0]
            p.pose.position.y = section[1]
            p.pose.position.z = 0.0

            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = 0.0
            p.pose.orientation.w = 0.0

            msg.poses.append(p)

        self.trajectory_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

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
