# Import the rclpy library, the ROS 2 client library for Python
import rclpy
from rclpy.node import Node
# Import the Node class from rclpy.node


class MyFirstNode(Node):
    """
    A simple ROS 2 Node that prints a message periodically.
    """

    def __init__(self):
        # Call the constructor of the parent Node class and give it a name
        super().__init__("my_first_node")

        # Create a timer that calls the timer_callback function every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("MyFirstNode has been started!")

    def timer_callback(self):
        # Log a message to the console
        self.get_logger().info("Hello from my_first_node!")


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create an instance of the node
    node = MyFirstNode()

    try:
        # "Spin" the node, which allows it to process callbacks (like the timer)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This block will be executed when you press Ctrl+C
        pass
    finally:
        # Cleanly destroy the node and shut down rclpy
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
