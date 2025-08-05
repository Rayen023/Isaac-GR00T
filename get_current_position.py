from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
import time

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"

robot_config = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    cameras={
        "wrist": OpenCVCameraConfig(
            index_or_path=0, 
            width=640, 
            height=480, 
            fps=30,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.ROTATE_180
        ),
        "front": OpenCVCameraConfig(
            index_or_path=2, 
            width=640, 
            height=480, 
            fps=30,
            color_mode=ColorMode.RGB,
        )
    }
)

robot = SO101Follower(robot_config)

while True:
    print("Connecting to the robot..., Do not move the robot.")
    robot.connect()
    observation = robot.get_observation()
    
    print("\n=== ROBOT STATE ===")
    robot_state_keys = list(robot._motors_ft.keys())
    print(f"Available state keys: {robot_state_keys}")
    
    for key in robot_state_keys:
        if key in observation:
            print(f"{key}: {observation[key]:.4f}")
    
    print("\n=== CAMERA INFO ===")
    camera_keys = list(robot_config.cameras.keys())
    print(f"Available cameras: {camera_keys}")
    
    for cam_key in camera_keys:
        if cam_key in observation:
            img_shape = observation[cam_key].shape
            print(f"{cam_key} camera: {img_shape}")
    
    print(f"\nTotal observation keys: {list(observation.keys())}")
    robot.disconnect()
    print("Robot Disconnected. Move to the next position.")
    print("-" * 30)
    time.sleep(10)
        
