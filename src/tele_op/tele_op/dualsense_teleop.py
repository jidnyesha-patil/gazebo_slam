import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class keyboard_teleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.twist_publisher = self.create_publisher(Twist,'cmd_vel',1)
        self.sub = self.create_subscription(Joy,"/joy",self.tele_op_map,1)
        self.z = 0.0
        self.reset = False

    def tele_op_map(self,msg):
        joystick_values = msg.axes
        y = -joystick_values[0]  #sideways
        x = joystick_values[1]    #front/back

        rot_z = joystick_values[3]/4


        if msg.buttons[4]:
            self.reset = True
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
            self.twist_publisher.publish(cmd_vel)

        if msg.buttons[5]:
            self.reset = False
            

        if(not self.reset):
            cmd_vel = Twist()
            cmd_vel.linear.x = x
            cmd_vel.linear.y = y
            cmd_vel.angular.z = rot_z
            self.twist_publisher.publish(cmd_vel)
            

def main():
    rclpy.init()
    node = keyboard_teleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
