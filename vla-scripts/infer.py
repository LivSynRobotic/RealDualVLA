import requests
import time
import base64
import threading
import numpy as np

# 频率配置
RDT_INTERVAL_SEC = 0.050   # RDT 每 50ms 请求一次
VLA_INTERVAL_SEC = 0.170   # VLA 每 170ms 触发一次
PRINT_EVERY = 5            # 解码后每多少次打印详细数组
TOTAL_RDT_CYCLES = 15      # 总共发起多少次 RDT 请求（保持你之前的 15 次要求）

def prepare_fake_observation_for_vla(prompt, resize_size=(224, 224)):
    fake_cam_high = np.random.randint(0, 256, (resize_size[0], resize_size[1], 3), dtype=np.uint8)
    fake_cam_left_wrist = np.random.randint(0, 256, (resize_size[0], resize_size[1], 3), dtype=np.uint8)
    fake_state = np.random.randn(7).astype(np.float32)
    return {
        "full_image": fake_cam_high,
        "wrist_image": fake_cam_left_wrist,
        "state": fake_state,
        "instruction": prompt,
    }

def convert_observation_to_serializable(observation):
    return {
        "full_image": observation["full_image"].tolist(),
        "wrist_image": observation["wrist_image"].tolist(),
        "state": observation["state"].tolist(),
        "instruction": observation["instruction"],
    }

def trigger_vla_async(serializable_obs, vla_endpoint):
    def _post():
        try:
            resp = requests.post(vla_endpoint, json=serializable_obs, timeout=10.0)
            print(f"[VLA][async] status={resp.status_code}")
        except Exception as e:
            print(f"[VLA][async] error: {e}")
    threading.Thread(target=_post, daemon=True).start()

def request_rdt_action(serializable_obs, rdt_endpoint):
    try:
        resp = requests.post(rdt_endpoint, json=serializable_obs, timeout=15.0)
        if resp.status_code == 200:
            return resp.json()
        else:
            err = {"error": f"RDT server error {resp.status_code}", "text": resp.text[:120]}
            print(err)
            return err
    except Exception as e:
        err = {"error": f"RDT request failed: {e}"}
        print(err)
        return err

def decode_base64_numpy(d):
    try:
        raw = base64.b64decode(d["__numpy__"])
        return np.frombuffer(raw, dtype=d["dtype"]).reshape(d["shape"])
    except Exception as e:
        print(f"[decode] 失败: {e}")
        return None

def try_decode_actions(actions):
    if (isinstance(actions, list) and len(actions) == 1
        and isinstance(actions[0], dict)
        and "__numpy__" in actions[0]):
        return decode_base64_numpy(actions[0])
    return None

# 端点配置
vla_main_server_endpoint = "http://159.75.78.68:8778/compute_hidden_states"
rdt_head_server_endpoint = "http://127.0.0.1:8777/act"

def main_loop():
    print(f"开始：RDT 每 {RDT_INTERVAL_SEC*1000:.0f} ms，请求；VLA 每 {VLA_INTERVAL_SEC*1000:.0f} ms 触发；总 RDT 次数 {TOTAL_RDT_CYCLES}")
    iteration = 0
    last_vla_time = 0.0  # 初始化一个早期时间戳
    next_rdt_start = time.perf_counter()

    try:
        for _ in range(TOTAL_RDT_CYCLES):
            now = time.perf_counter()

            # 采集（或真实）观测
            observation = prepare_fake_observation_for_vla(prompt="Test action retrieval")
            serializable_obs = convert_observation_to_serializable(observation)

            # 判断是否需要触发 VLA
            if now - last_vla_time >= VLA_INTERVAL_SEC:
                trigger_vla_async(serializable_obs, vla_main_server_endpoint)
                last_vla_time = now

            # RDT 每 50ms 一次（本循环即一次）
            actions = request_rdt_action(serializable_obs, rdt_head_server_endpoint)

            decoded = try_decode_actions(actions)
            if decoded is not None:
                if iteration % PRINT_EVERY == 0:
                    print(f"[RDT] decoded shape={decoded.shape}")
                    np.set_printoptions(precision=3, suppress=True)
                    print(decoded)
                else:
                    print(f"[RDT] action shape={decoded.shape} (skip detail)")
            else:
                short = str(actions)
                if len(short) > 160:
                    short = short[:160] + "..."
                print(f"[RDT] raw: {short}")

            iteration += 1

            # 精确节拍控速到 50ms
            next_rdt_start += RDT_INTERVAL_SEC
            sleep_time = next_rdt_start - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 超时跳过补偿，防止连锁塌陷
                next_rdt_start = time.perf_counter()
    except KeyboardInterrupt:
        print("\n手动中断。")
    finally:
        print(f"结束，RDT 请求次数={iteration}，VLA 触发次数≈{int((iteration*RDT_INTERVAL_SEC)/VLA_INTERVAL_SEC)}(理论值)")

if __name__ == "__main__":
    main_loop()
