import cv2
import numpy as np

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="my_leader_arm_1",
)

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
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

observation = robot.get_observation()
for key, value in observation.items():
    if isinstance(value, list):
        print(f"{key}: {len(value)} items")
    else:
        print(f"{key}: {value}")

print(observation.items())

print(observation["wrist"].shape)  # (480, 640, 3)
print(observation["front"].shape)  # (480, 640, 3)

while True:
    observation = robot.get_observation()

    # Get camera images (already processed with correct colors and rotation)
    wrist = observation["wrist"]
    front = observation["front"]

    # Display the images
    cv2.imshow("Wrist View", wrist)
    cv2.imshow("Up View", front)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    action = teleop_device.get_action()
    print(f"Action: {action}")
    
    robot.send_action(action)

# Clean up
cv2.destroyAllWindows()
