import requests
import time
import base64

import numpy as np

vla_main_server_endpoint = "http://{YOUR_SERVER_IP}:8778/compute_hidden_states"
rdt_head_server_endpoint = "http://127.0.0.1:8777/act"

def prepare_fake_observation_for_vla(prompt, resize_size=(224, 224)):
    """Create fake observation data for VLA server testing."""
    # Create fake RGB images with correct data type (uint8)
    fake_cam_high = np.random.randint(0, 256, (resize_size[0], resize_size[1], 3), dtype=np.uint8)
    fake_cam_left_wrist = np.random.randint(0, 256, (resize_size[0], resize_size[1], 3), dtype=np.uint8)
    
    fake_state = np.random.randn(7).astype(np.float32) 
    
    vla_observation = {
        "full_image": fake_cam_high,
        "wrist_image": fake_cam_left_wrist,
        "state": fake_state,
        "instruction": prompt,
    }
    
    return vla_observation

def convert_observation_to_serializable(observation):
    observation["full_image"] = observation["full_image"].tolist()
    observation["wrist_image"] = observation["wrist_image"].tolist()
    observation["state"] = observation["state"].tolist()
    return observation

def get_action_with_timing_control(observation, vla_endpoint, rdt_endpoint):
    try:
        # observation_old = observation.copy()
        # 将 observation 转换为可序列化的格式
        observation = convert_observation_to_serializable(observation)


        print("Triggering VLA computation...")
        vla_response = requests.post(vla_endpoint, json=observation, timeout=10.0)
        print("VLA server response:", vla_response.json())

        print("Requesting action from RDT server...")
        response = requests.post(rdt_endpoint, json=observation, timeout=15.0)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"RDT server error: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Request failed: {e}"}

def decode_numpy_array(encoded_array):
    if "__numpy__" in encoded_array:
        data = base64.b64decode(encoded_array["__numpy__"])
        return np.frombuffer(data, dtype=encoded_array["dtype"]).reshape(encoded_array["shape"])
    return encoded_array


vla_observation = prepare_fake_observation_for_vla(prompt="Test action retrieval")

actions = get_action_with_timing_control(
    observation=vla_observation,
    vla_endpoint=vla_main_server_endpoint,
    rdt_endpoint=rdt_head_server_endpoint
)

if isinstance(actions, list) and len(actions) > 0 and "__numpy__" in actions[0]:
    decoded_actions = [decode_numpy_array(action) for action in actions]

    formatted_actions = [np.array2string(action, formatter={'float_kind': lambda x: f"{x:.2f}"}, separator=', ') for action in decoded_actions]
    print("Decoded actions:")
    for action in formatted_actions:
        print(action)
else:
    print("Retrieved actions:", actions)
