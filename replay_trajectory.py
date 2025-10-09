from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import csv
import time

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_calibrated_follower_arm8"
CSV_FILE = "fara.csv"

robot_config = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID)
robot = SO101Follower(robot_config)
robot.connect()

try:
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        prev_time = None
        
        for row in reader:
            action = {
                'shoulder_pan.pos': float(row['shoulder_pan.pos']),
                'shoulder_lift.pos': float(row['shoulder_lift.pos']),
                'elbow_flex.pos': float(row['elbow_flex.pos']),
                'wrist_flex.pos': float(row['wrist_flex.pos']),
                'wrist_roll.pos': float(row['wrist_roll.pos']),
                'gripper.pos': float(row['gripper.pos'])
            }
            
            # Simulate same timing as recorded
            # current_time = float(row['timestamp'])
            # if prev_time is not None:
            #     time.sleep(current_time - prev_time)
            # prev_time = current_time
            time.sleep(0.04) # in seconds so 40ms
            
            robot.send_action(action)
            print(f"Sent: {action}")

except KeyboardInterrupt:
    print("\nStopped")
finally:
    robot.disconnect()
