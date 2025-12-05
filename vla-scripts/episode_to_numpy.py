# episode_to_numpy.py
import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image

def parse_episode(tfrecord_file, take_n=1):
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    episodes = []

    for raw_record in dataset.take(take_n):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        is_first = np.array(features["steps/is_first"].int64_list.value, dtype=np.int64)
        is_last = np.array(features["steps/is_last"].int64_list.value, dtype=np.int64)
        is_terminal = np.array(features["steps/is_terminal"].int64_list.value, dtype=np.int64)

        action = np.array(features["steps/action"].float_list.value, dtype=np.float32)
        state = np.array(features["steps/observation/state"].float_list.value, dtype=np.float32)

        steps = len(is_first)
        action_dim = action.shape[0] // steps
        state_dim = state.shape[0] // steps

        action = action.reshape(steps, action_dim)
        state = state.reshape(steps, state_dim)

        instructions = [b.decode("utf-8") for b in features["steps/language_instruction"].bytes_list.value]

        cam_high = [tf.io.decode_jpeg(b).numpy() for b in features["steps/observation/cam_high"].bytes_list.value]
        cam_left_wrist = [tf.io.decode_jpeg(b).numpy() for b in features["steps/observation/cam_left_wrist"].bytes_list.value]

        episode = {
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "actions": action,
            "states": state,
            "instruction": instructions[0] if instructions else "",
            "images": {
                "cam_high": np.stack(cam_high),
                "cam_left_wrist": np.stack(cam_left_wrist),
            },
        }
        episodes.append(episode)

    return episodes


def save_episode_images(ep, out_dir, cams=("cam_high", "cam_left_wrist"), max_frames=5):
    """将 episode 中的图像保存为 PNG 文件。"""
    os.makedirs(out_dir, exist_ok=True)
    for cam in cams:
        imgs = ep["images"][cam]
        n = min(max_frames, imgs.shape[0])
        for i in range(n):
            img = imgs[i]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            Image.fromarray(img).save(os.path.join(out_dir, f"{cam}_{i:03d}.png"))


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    parser = argparse.ArgumentParser(description="将 TFRecord episode 解析为 numpy，并可导出图像。")
    parser.add_argument("tfrecord_file", help="TFRecord 文件路径")
    parser.add_argument("--print-all", action="store_true", help="打印完整信息")
    parser.add_argument("--save-images", action="store_true", help="保存图像到本地")
    parser.add_argument("--max-frames", type=int, default=5, help="每个相机最多保存的帧数")
    parser.add_argument("--out-dir", type=str, default=None, help="图像输出目录（默认 images/<tfrecord名>）")
    args = parser.parse_args()

    filename = args.tfrecord_file
    print_all = args.print_all

    episodes = parse_episode(filename, take_n=1)
    ep = episodes[0]

    print("指令:", ep["instruction"])

    if args.save_images:
        base = os.path.splitext(os.path.basename(filename))[0]
        out_dir = args.out_dir or os.path.join("images", base)
        save_episode_images(ep, out_dir, max_frames=args.max_frames)
        print(f"已保存前 {min(args.max_frames, ep['images']['cam_high'].shape[0])} 帧到: {out_dir}")

    if print_all:
        print("\n动作 shape:", ep["actions"].shape)
        # print(ep["actions"])

        print("\n状态 shape:", ep["states"].shape)
        # print(ep["states"])

        print("高位相机图像 shape:", ep["images"]["cam_high"].shape)
        print(ep["images"]["cam_high"][:1])
        print("手腕相机图像 shape:", ep["images"]["cam_left_wrist"].shape)

        print("\n是否终止 flags:", ep["is_terminal"])
    else:
        print("\n动作 shape:", ep["actions"].shape)
        print(ep["actions"][:5])   # 前5步

        print("\n状态 shape:", ep["states"].shape)
        print(ep["states"][:5])    # 前5步

        print("高位相机图像 shape:", ep["images"]["cam_high"].shape)
        print("手腕相机图像 shape:", ep["images"]["cam_left_wrist"].shape)

        print("\n是否终止 flags (前20):", ep["is_terminal"][:20])
