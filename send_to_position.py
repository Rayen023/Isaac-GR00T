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

# Target position based on your example


print("Connecting to robot...")
robot.connect()

target_position = {
    'shoulder_pan.pos': -3.2353,
    'shoulder_lift.pos': -99.0678,
    'elbow_flex.pos': 99.3657,
    'wrist_flex.pos': 74.1432,
    'wrist_roll.pos': -1.1966,
    'gripper.pos': 1
}
robot.send_action(target_position)
time.sleep(1)
target_position = {
    'shoulder_pan.pos': -3.2353,
    'shoulder_lift.pos': -99.0678,
    'elbow_flex.pos': 99.3657,
    'wrist_flex.pos': 74.1432,
    'wrist_roll.pos': -1.1966,
    'gripper.pos': 60
}
robot.send_action(target_position)
time.sleep(0.6)

target_position = {
    'shoulder_pan.pos': -3.2353,
    'shoulder_lift.pos': -99.0678,
    'elbow_flex.pos': 99.3657,
    'wrist_flex.pos': 74.1432,
    'wrist_roll.pos': -1.1966,
    'gripper.pos': 1
}
robot.send_action(target_position)
time.sleep(0.6)
target_position = {
    'shoulder_pan.pos': -3.2353,
    'shoulder_lift.pos': -99.0678,
    'elbow_flex.pos': 99.3657,
    'wrist_flex.pos': 74.1432,
    'wrist_roll.pos': -1.1966,
    'gripper.pos': 60
}
robot.send_action(target_position)
time.sleep(0.6)
target_position = {
    'shoulder_pan.pos': -3.2353,
    'shoulder_lift.pos': -99.0678,
    'elbow_flex.pos': 99.3657,
    'wrist_flex.pos': 74.1432,
    'wrist_roll.pos': -1.1966,
    'gripper.pos': 1
}
robot.send_action(target_position)
time.sleep(0.6)


robot.disconnect()
