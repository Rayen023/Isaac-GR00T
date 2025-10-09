from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
import time
import csv
from datetime import datetime
from pathlib import Path


def create_trajectory_csv():
    """
    Creates a new CSV file with a timestamp in its name for recording trajectory data.
    Returns the file path and CSV writer.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.csv"
    filepath = Path(filename)
    
    # Define column headers based on action structure
    fieldnames = [
        'timestamp',
        'shoulder_pan.pos',
        'shoulder_lift.pos',
        'elbow_flex.pos',
        'wrist_flex.pos',
        'wrist_roll.pos',
        'gripper.pos'
    ]
    
    file = open(filepath, 'w', newline='')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    print(f"Created trajectory file: {filename}")
    return file, writer, filepath


def is_position_different(current_action, previous_action, threshold=0.5):
    """
    Check if the current action is significantly different from the previous one.
    
    Args:
        current_action: Dictionary with current joint positions
        previous_action: Dictionary with previous joint positions
        threshold: Minimum difference in degrees to consider as a change
    
    Returns:
        True if position is different enough to record, False otherwise
    """
    if previous_action is None:
        return True
    
    # Check each joint position
    for key in current_action.keys():
        if key in previous_action:
            diff = abs(current_action[key] - previous_action[key])
            if diff > threshold:
                return True
    
    return False


def save_action_to_csv(writer, action, timestamp):
    """
    Saves an action to the CSV file.
    
    Args:
        writer: CSV DictWriter object
        action: Dictionary with joint positions
        timestamp: Current timestamp
    """
    row = {
        'timestamp': timestamp,
        'shoulder_pan.pos': action.get('shoulder_pan.pos', 0),
        'shoulder_lift.pos': action.get('shoulder_lift.pos', 0),
        'elbow_flex.pos': action.get('elbow_flex.pos', 0),
        'wrist_flex.pos': action.get('wrist_flex.pos', 0),
        'wrist_roll.pos': action.get('wrist_roll.pos', 0),
        'gripper.pos': action.get('gripper.pos', 0)
    }
    writer.writerow(row)


teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="my_leader_arm_1",
)

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"

robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,)


robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

# Create CSV file for recording trajectory
csv_file, csv_writer, csv_filepath = create_trajectory_csv()
previous_action = None
recorded_count = 0
total_count = 0

try : 
    while True:        
        action = teleop_device.get_action()
        total_count += 1
        
        # Check if this action is different enough from the previous one
        if is_position_different(action, previous_action, threshold=0.5):
            current_time = time.time()
            save_action_to_csv(csv_writer, action, current_time)
            csv_file.flush()  # Ensure data is written immediately
            recorded_count += 1
            previous_action = action.copy()
            print(f"Recorded position {recorded_count}/{total_count}: {action}")
        
        robot.send_action(action)
        time.sleep(0.04) # in seconds so 40ms
        
        
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    print(f"\nRecording complete!")
    print(f"Total positions checked: {total_count}")
    print(f"Positions recorded: {recorded_count}")
    print(f"Compression ratio: {recorded_count}/{total_count} = {recorded_count/total_count*100:.1f}%")
    print(f"Trajectory saved to: {csv_filepath}")
    
    csv_file.close()
    print("Disconnecting...")
    robot.disconnect()
    teleop_device.disconnect()
