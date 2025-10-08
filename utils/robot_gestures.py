"""
Robot gesture utilities for SO101 Follower robot.
Contains functions for various robot movements and greetings.
"""
import time


def say_hello(policy_client, delay=0.5):
    LOOKING_UP = {
        'shoulder_pan.pos': 45.9559,
        'shoulder_lift.pos': -98.5593,
        'elbow_flex.pos': 99.5469,
        'wrist_flex.pos': -57.1367,
        'wrist_roll.pos': 3.6386,
        'gripper.pos': 0.9253
    }
    
    LOOKING_LEFT = {
        'shoulder_pan.pos': -49.3382,
        'shoulder_lift.pos': -98.4746,
        'elbow_flex.pos': 99.5469,
        'wrist_flex.pos': -43.6876,
        'wrist_roll.pos': 3.4921,
        'gripper.pos': 0.9914
    }
    
    LOOKING_RIGHT = {
        'shoulder_pan.pos': 45.9559,
        'shoulder_lift.pos': -98.5593,
        'elbow_flex.pos': 99.5469,
        'wrist_flex.pos': -57.1367,
        'wrist_roll.pos': 3.6386,
        'gripper.pos': 0.9253
    }
    
    RESET_POSITION = {
        "shoulder_pan.pos": -0.5882352941176521,
        "shoulder_lift.pos": -98.38983050847457,
        "elbow_flex.pos": 99.45627548708654,
        "wrist_flex.pos": 74.40347071583514,
        "wrist_roll.pos": 3.3943833943834107,
        "gripper.pos": 1.0575016523463316
    }
    
    robot = policy_client.robot
    print("Robot saying hello!")
    
    # 1. Look up and wave
    print("Looking up...")
    robot.send_action(LOOKING_UP)
    time.sleep(delay)
    
    # Open gripper
    print("Waving (opening gripper)...")
    looking_up_open = LOOKING_UP.copy()
    looking_up_open['gripper.pos'] = 60
    robot.send_action(looking_up_open)
    time.sleep(delay)
    
    # Close gripper
    print("Waving (closing gripper)...")
    robot.send_action(LOOKING_UP)
    time.sleep(delay)
    
    # Open gripper again
    robot.send_action(looking_up_open)
    time.sleep(delay)
    
    # Close gripper
    robot.send_action(LOOKING_UP)
    time.sleep(delay)
    
    # 2. Look left and wave
    print("Looking left...")
    robot.send_action(LOOKING_LEFT)
    time.sleep(delay)
    
    # Open gripper
    print("Waving (opening gripper)...")
    looking_left_open = LOOKING_LEFT.copy()
    looking_left_open['gripper.pos'] = 60
    robot.send_action(looking_left_open)
    time.sleep(delay)
    
    # Close gripper
    print("Waving (closing gripper)...")
    robot.send_action(LOOKING_LEFT)
    time.sleep(delay)
    
    # Open gripper again
    robot.send_action(looking_left_open)
    time.sleep(delay)
    
    # Close gripper
    robot.send_action(LOOKING_LEFT)
    time.sleep(delay)
    
    # 3. Look right and wave
    print("Looking right...")
    robot.send_action(LOOKING_RIGHT)
    time.sleep(delay)
    
    # Open gripper
    print("Waving (opening gripper)...")
    looking_right_open = LOOKING_RIGHT.copy()
    looking_right_open['gripper.pos'] = 60
    robot.send_action(looking_right_open)
    time.sleep(delay)
    
    # Close gripper
    print("Waving (closing gripper)...")
    robot.send_action(LOOKING_RIGHT)
    time.sleep(delay)
    
    # Open gripper again
    robot.send_action(looking_right_open)
    time.sleep(delay)
    
    # Close gripper
    robot.send_action(LOOKING_RIGHT)
    time.sleep(delay)
    
    # 4. Return to rest position
    print("Returning to rest position...")
    robot.send_action(RESET_POSITION)
    time.sleep(delay)
    


def draw_letter_f(policy_client, delay=1.0):
    BOTTOM_POINT = {
        'shoulder_pan.pos': -18.5294,
        'shoulder_lift.pos': -99.0678,
        'elbow_flex.pos': 91.5723,
        'wrist_flex.pos': 33.6226,
        'wrist_roll.pos': 3.7363,
        'gripper.pos': 1.0575
    }
    
    TOP_LEFT_CORNER = {
        'shoulder_pan.pos': -18.6765,
        'shoulder_lift.pos': -43.1356,
        'elbow_flex.pos': 9.2886,
        'wrist_flex.pos': 48.0260,
        'wrist_roll.pos': 0.5617,
        'gripper.pos': 1.1897
    }
    
    TOP_RIGHT_CORNER = {
        'shoulder_pan.pos': 17.6471,
        'shoulder_lift.pos': -26.0169,
        'elbow_flex.pos': -15.7227,
        'wrist_flex.pos': 52.1041,
        'wrist_roll.pos': 3.7851,
        'gripper.pos': 0.8592
    }
    
    MIDDLE_LEFT_CORNER = {
        'shoulder_pan.pos': -19.1912,
        'shoulder_lift.pos': -67.6271,
        'elbow_flex.pos': 45.9900,
        'wrist_flex.pos': 33.7093,
        'wrist_roll.pos': 3.7363,
        'gripper.pos': 1.0575
    }
    
    MIDDLE_RIGHT_CORNER = {
        'shoulder_pan.pos': 15.0735,
        'shoulder_lift.pos': -67.7119,
        'elbow_flex.pos': 45.9900,
        'wrist_flex.pos': 33.7093,
        'wrist_roll.pos': 3.7363,
        'gripper.pos': 1.0575
    }
    
    robot = policy_client.robot
    print("Robot drawing the letter F!")
    
    # 1. Start at bottom point
    print("Moving to bottom point (starting position)...")
    robot.send_action(BOTTOM_POINT)
    time.sleep(delay)
    
    # 2. Move to top left corner
    print("Drawing upward stroke to top left corner...")
    robot.send_action(TOP_LEFT_CORNER)
    time.sleep(delay)
    
    # 3. Move to top right corner
    print("Drawing top horizontal stroke to top right corner...")
    robot.send_action(TOP_RIGHT_CORNER)
    time.sleep(delay)
    
    # 4. Return to top left corner
    print("Returning to top left corner...")
    robot.send_action(TOP_LEFT_CORNER)
    time.sleep(delay)
    
    # 5. Move down to middle left corner
    print("Moving down to middle left corner...")
    robot.send_action(MIDDLE_LEFT_CORNER)
    time.sleep(delay)
    
    # 6. Move to middle right corner
    print("Drawing middle horizontal stroke to middle right corner...")
    robot.send_action(MIDDLE_RIGHT_CORNER)
    time.sleep(delay)
    
    # 7. Return to middle left corner
    print("Returning to middle left corner...")
    robot.send_action(MIDDLE_LEFT_CORNER)
    time.sleep(delay)
    
    # 8. Move down to bottom point
    print("Moving down to bottom point (finishing position)...")
    robot.send_action(BOTTOM_POINT)
    time.sleep(delay)
