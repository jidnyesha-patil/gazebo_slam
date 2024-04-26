#depth image
#rgb to grayscale the input rgb image
#k matrix once _-- init
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from message_filters import TimeSynchronizer,Subscriber
import cv2
from cv_bridge import CvBridge,CvBridgeError
import numpy as np
from matplotlib import pyplot as plt

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odom_node')

        self.K = np.array([[434.31002800449204,0.0,320.5],[0.0,434.31002800449204,240.5],[0.0,0.0,1.0]]) # Camera Calibration matrix 

        # Subscribe to depth camera - RGB and depth frames
        self.rgb_subscriber = Subscriber(self,Image,'/depth_camera/image_raw')
        self.depth_subscriber = Subscriber(self,Image,'/depth_camera/depth/image_raw')
        self.sync_subscriber = TimeSynchronizer([self.rgb_subscriber,self.depth_subscriber],10)
        self.sync_subscriber.registerCallback(self.frame_callback)
        # self.camera_info_subscriber = self.create_subscription(CameraInfo,'/depth_camera/camera_info',self.get_camera_info,10)
        self.path_publisher = self.create_publisher(Path,'/trajectory',10)
        self.bridge=CvBridge()

        self.curr_img_frame=None
        self.curr_depth_frame=None
        self.prev_img_frame=None
        self.prev_depth_frame=None
        self.kp_array=[]
        self.des_array=[]
        self.matches_array=[]
        self.curr_idx=0
        self.frame_count=0

        self.robot_pose = np.zeros((1,4,4))
        
        self.trajectory = np.zeros((3, 1))

        # Create SIFT and FLANN matcher
        self.sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        # Path message
        self.path_msg = Path()

    # def get_camera_info(self,info_msg):
    #     if self.K is not None:
    #         self.K = info_msg.k # float64 [9] data type
    #         self.get_logger().info(f'K: {self.K}')

    def frame_callback(self,rgb_msg,depth_msg):
        if self.frame_count %4 == 0:
            try:
                img_frame = cv2.cvtColor(self.bridge.imgmsg_to_cv2(rgb_msg,'bgr8'),cv2.COLOR_BGR2GRAY)
            except CvBridgeError as e1:
                self.get_logger().info(f'RGB frame CV Bridge failed : {e1}')
            try:
                depth_frame = self.bridge.imgmsg_to_cv2(depth_msg,'32FC1')
                self.depth_frame=depth_frame
                # self.get_logger().info(f'{depth_frame[240][320]}')
            except CvBridgeError as e2:
                self.get_logger().info(f'Depth frame CV Bridge failed : {e2}')

            # if img_frame is not None:
            #     cv2.namedWindow("grayscale",cv2.WINDOW_NORMAL)
            #     cv2.imshow("grayscale",img_frame)
            # if depth_frame is not None:       
            #     cv2.namedWindow("depth",cv2.WINDOW_NORMAL)
            #     normalized_depth = cv2.normalize(depth_frame,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            #     cv2.imshow("depth",normalized_depth)
            
            # if cv2.waitKey(1)==ord('q'):
            #     raise SystemExit
            
            self.curr_img_frame=img_frame
            self.curr_depth_frame=depth_frame

            # TO - DO : Feature extraction
            kp,des = self.extract_frame_features(img_frame)
            self.kp_array.append(kp)
            self.des_array.append(des)

            #if not first frame
            if self.prev_img_frame is not None:
                #Feature Matching
                match = self.match_feature(self.des_array[self.curr_idx-1],self.des_array[self.curr_idx])
                self.matches_array.append(match)

                #Estimate Motion
                rmat, tvec, image1_points, image2_points = self.estimate_motion(match,self.kp_array[self.curr_idx-1],self.kp_array[self.curr_idx],self.K, self.prev_depth_frame)
                
                # Update Trajectory
                self.update_trajectory(rmat,tvec)

            self.curr_idx +=1
            self.prev_depth_frame=self.curr_depth_frame
            self.prev_img_frame = self.curr_img_frame

        self.frame_count += 1

    def extract_frame_features(self,image):
        
        kp,des = self.sift.detectAndCompute(image,None)
        
        return kp,des

    def match_feature(self,des1,des2):
        des1 = np.float32(des1)
        des2 = np.float32(des2)

        match_1 = self.flann.knnMatch(des1,des2,k=2)
        
        good_matches = []
        for m,n in match_1:
            if m.distance < 0.6*n.distance:
                good_matches.append(m)

        return good_matches
    
    def estimate_motion(self,match, kp1, kp2, k, depth1=None):
        """
        Estimate camera motion from a pair of subsequent image frames

        Arguments:
        match -- list of matched features from the pair of images
        kp1 -- list of the keypoints in the first image
        kp2 -- list of the keypoints in the second image
        k -- camera calibration matrix 
        
        Optional arguments:
        depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

        Returns:
        rmat -- recovered 3x3 rotation numpy matrix
        tvec -- recovered 3x1 translation numpy vector
        image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                        coordinates of the i-th match in the image coordinate system
        image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                        coordinates of the i-th match in the image coordinate system
                
        """
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        image1_points = []
        image2_points = []
        
        objectpoints = []
        
        # Iterate through the matched features
        for m in match:
            # Get the pixel coordinates of features f[k - 1] and f[k]
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            
            # Get the scale of features f[k - 1] from the depth map
            s = depth1[int(v1), int(u1)]
            
            # Check for valid scale values
            if s < 8.0:
                # Transform pixel coordinates to camera coordinates using the pinhole camera model
                p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))
                
                # Save the results
                image1_points.append([u1, v1])
                image2_points.append([u2, v2])
                objectpoints.append(p_c)
            
        # Convert lists to numpy arrays
        objectpoints = np.vstack(objectpoints)
        imagepoints = np.array(image2_points)
        
        # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
        _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        return rmat, tvec, image1_points, image2_points
    
    def update_trajectory(self,rmat,tvec):
        current_pose = np.eye(4)
        current_pose[0:3, 0:3] = rmat
        current_pose[0:3, 3] = tvec.T

        # Build the robot's pose from the initial position by multiplying previous and current poses
        robot_pose = self.robot_pose[-1] @ np.linalg.inv(current_pose)
        # self.get_logger().info(f'Pose:{robot_pose}')
        self.robot_pose=np.append(self.robot_pose,robot_pose.reshape(1,4,4),axis=0)
        # Calculate current camera position from origin
        position = self.robot_pose[self.curr_idx] @ np.array([0., 0., 0., 1.])

        # Build trajectory
        self.trajectory=np.append(self.trajectory,position[0:3].reshape(3,1),axis=1)
        self.publish_path(position)
        self.get_logger().info(f'Trajectory obtained {self.trajectory}')

    def publish_path(self,position):
        now_time = self.get_clock().now().to_msg()
        self.path_msg.header.frame_id='odom'
        self.path_msg.header.stamp = now_time
        this_pose = PoseStamped()
        this_pose.header.frame_id='odom'
        this_pose.header.stamp=now_time
        this_pose.pose.position.x=position[0]
        this_pose.pose.position.y=position[1]
        this_pose.pose.position.z=position[2]
        self.path_msg.poses.append(this_pose)

        self.path_publisher.publish(self.path_msg)

    def visualize_matches(image1, kp1, image2, kp2, match):
        """
        Visualize corresponding matches in two images

        Arguments:
        image1 -- the first image in a matched image pair
        kp1 -- list of the keypoints in the first image
        image2 -- the second image in a matched image pair
        kp2 -- list of the keypoints in the second image
        match -- list of matched features from the pair of images

        Returns:
        image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
        """
        image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None,flags=2)
        plt.figure(figsize=(16, 6), dpi=100)
        plt.imshow(image_matches)

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