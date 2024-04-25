#depth image
#rgb to grayscale the input rgb image
#k matrix once _-- init
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from message_filters import TimeSynchronizer,Subscriber
import cv2
from cv_bridge import CvBridge,CvBridgeError

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odom_node')

        self.K = None # Camera Calibration matrix

        # Subscribe to depth camera - RGB and depth frames
        self.rgb_subscriber = Subscriber(self,Image,'/depth_camera/image_raw')
        self.depth_subscriber = Subscriber(self,Image,'/depth_camera/depth/image_raw')
        self.sync_subscriber = TimeSynchronizer([self.rgb_subscriber,self.depth_subscriber],10)
        self.sync_subscriber.registerCallback(self.frame_callback)
        self.camera_info_subscriber = self.create_subscription(CameraInfo,'/depth_camera/camera_info',self.get_camera_info,10)

        self.bridge=CvBridge()

        self.rgb_frames=[]
        self.depth_frames=[]

    def get_camera_info(self,info_msg):
        if self.K is not None:
            self.K = info_msg.k # float64 [9] data type
            self.get_logger().info(f'K: {self.K}')

    def frame_callback(self,rgb_msg,depth_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg,'bgr8')
        except CvBridgeError as e1:
            self.get_logger().info(f'RGB frame CV Bridge failed : {e1}')
        try:
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg,'32FC1')
            self.depth_frame=depth_frame
            self.get_logger().info(f'{depth_frame[240][320]}')
        except CvBridgeError as e2:
            self.get_logger().info(f'Depth frame CV Bridge failed : {e2}')

        if rgb_frame is not None:
            cv2.namedWindow("rgb",cv2.WINDOW_NORMAL)
            cv2.imshow("rgb",rgb_frame)
        if depth_frame is not None:       
            cv2.namedWindow("depth",cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("depth", self.on_mouse_click)
            cv2.imshow("depth",depth_frame)
        if cv2.waitKey(1)==ord('q'):
            raise SystemExit
    def on_mouse_click(self, event, x, y, flags, param):
        """
        Callback function for mouse click events.

        Args:
        - event: The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        - x: The x-coordinate of the mouse click.
        - y: The y-coordinate of the mouse click.
        - flags: Additional flags (not used).
        - param: Additional parameters (not used).

        Returns:
        - None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Print the depth value at the clicked point
            depth_value = self.depth_frame[y][x]
            self.get_logger().info(f"Depth value at point ({x}, {y}): {depth_value}")

def main(args=None):
    rclpy.init(args=args)
    vo = VisualOdometryNode()
    try:
        rclpy.spin(vo)
    except (SystemExit,KeyboardInterrupt):
        rclpy.logging.get_logger("Quitting").info('Done')
    vo.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()