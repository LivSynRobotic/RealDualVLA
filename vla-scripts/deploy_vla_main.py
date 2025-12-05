"""
deploy_vla_main.py

Starts the VLA main server (Sys2), which computes last_hidden_states from observations
and sends them to the RDT Action Head server (Sys1).
"""

import base64
import json
import traceback
import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import json_numpy
import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch.nn.functional as F
# Apply JSON numpy patch for serialization
json_numpy.patch()

from experiments.robot.openvla_utils import (
    get_processor,
    get_proprio_projector,
    get_vla,
    normalize_proprio,
    prepare_images_for_vla,
)
from prismatic.vla.constants import PROPRIO_DIM

def resize_images_for_vla(pixel_values, target_size=(224, 224)):
    """将图像从384x384调整到224x224用于VLA"""
    
    # pixel_values shape: (B, C, H, W)
    if pixel_values.shape[-2:] != target_size:
        # 使用双线性插值调整图像尺寸到224x224
        resized = F.interpolate(
            pixel_values, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        return resized
    return pixel_values


class VLAMainServer:
    def __init__(self, cfg: "DeployVlaMainConfig") -> None:
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)
        
        assert cfg.unnorm_key in self.vla.norm_stats, f"Un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
        
        self.latest_payload: Optional[Dict[str, Any]] = None
        self.payload_lock = threading.Lock()
        self.is_computing = False
        self.compute_lock = threading.Lock()
        
        self.request_count = 0
        self.compute_count = 0
        self.cache_update_count = 0
        self.stats_lock = threading.Lock()
        
        self.should_stop = False
        self.compute_thread = threading.Thread(target=self._background_compute_loop, daemon=True)
        self.compute_thread.start()

    def _background_compute_loop(self) -> None:
        loop_count = 0
        
        while not self.should_stop:
            loop_count += 1
            try:
                with self.payload_lock:
                    current_payload = self.latest_payload
                    current_payload_id = id(current_payload) if current_payload else None
                    
                if current_payload is None:
                    threading.Event().wait(0.1)
                    continue
                    
                with self.compute_lock:
                    if self.is_computing:
                        threading.Event().wait(0.1)
                        continue
                    self.is_computing = True
                
                with self.stats_lock:
                    self.compute_count += 1
                    compute_id = self.compute_count
                
                start_time = time.time()
                
                self._compute_hidden_states(current_payload, compute_id)
                
                compute_time = time.time() - start_time
                
                with self.payload_lock:
                    latest_payload_id = id(self.latest_payload) if self.latest_payload else None
                    if latest_payload_id == current_payload_id:
                        self.latest_payload = None
                        
            except Exception:
                print(traceback.format_exc())
            finally:
                with self.compute_lock:
                    self.is_computing = False
                
                threading.Event().wait(0.05)

    def _compute_hidden_states(self, payload: Dict[str, Any], compute_id: int) -> None:
        try:
            if "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

            all_images = [observation["full_image"]]
            if self.cfg.num_images_in_input > 1:
                all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])

            all_images = prepare_images_for_vla(all_images, self.cfg)
            primary_image = all_images.pop(0)

            inputs = self.processor(prompt, primary_image).to(self.device, dtype=torch.bfloat16)

            if all_images:
                all_wrist_inputs = [
                    self.processor(prompt, image_wrist).to(self.device, dtype=torch.bfloat16) 
                    for image_wrist in all_images
                ]
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
                inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

            proprio = None
            if self.cfg.use_proprio:
                proprio_norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
                proprio = normalize_proprio(observation["state"], proprio_norm_stats)

            # Resize from 384x384 to 224x224 for VLA model
            inputs["pixel_values"] = resize_images_for_vla(inputs["pixel_values"], target_size=(224, 224))


            # Get last hidden states from VLA
            print(f"🧠 [Compute #{compute_id}] Computing hidden states with VLA model...")
            inference_start = time.time()
            with torch.inference_mode():
                text_hidden_states, actions_hidden_states = self.vla.get_last_hidden_states(
                    **inputs,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    use_film=self.cfg.use_film,
                )
            inference_time = time.time() - inference_start
            print(f"⚡ [Compute #{compute_id}] VLA inference completed in {inference_time:.3f}s")

            text_hidden_states = text_hidden_states.to(dtype=torch.float32)
            actions_hidden_states = actions_hidden_states.to(dtype=torch.float32)

            hidden_states_np = {
                "text_hidden_states": text_hidden_states.cpu().numpy(),
                "actions_hidden_states": actions_hidden_states.cpu().numpy(),
            }
            serialized_states = base64.b64encode(json_numpy.dumps(hidden_states_np).encode("utf-8")).decode("utf-8")

            try:
                response = requests.post(
                    self.cfg.action_head_server_url,
                    json={"hidden_states": serialized_states},
                    timeout=5.0
                )
                if response.status_code == 200:
                    print(f"✅ [Compute #{compute_id}] Hidden states sent successfully to action head server")
                else:
                    print(f"⚠️ [Compute #{compute_id}] Failed to send hidden states, status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ [Compute #{compute_id}] Failed to send hidden states: {e}")


        except Exception:
            print(traceback.format_exc())

    def get_server_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            with self.stats_lock:
                self.request_count += 1
                request_id = self.request_count
            
            with self.payload_lock:
                self.latest_payload = payload.copy()
                with self.stats_lock:
                    self.cache_update_count += 1
            
            return JSONResponse({
                "status": "ok", 
                "message": f"Request #{request_id} received and cached for processing.",
                "request_id": request_id
            })
            
        except Exception:
            print(traceback.format_exc())
            return JSONResponse({
                "status": "error", 
                "message": "Failed to cache request."
            }, status_code=500)

    def stop(self) -> None:
        self.should_stop = True
        if self.compute_thread.is_alive():
            self.compute_thread.join(timeout=5.0)

    def run(self, host: str = "0.0.0.0", port: int = 8778) -> None:
        self.app = FastAPI()
        self.app.post("/compute_hidden_states")(self.get_server_action)
        
        import signal
        def signal_handler(signum, frame):
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            self.stop()


@dataclass
class DeployVlaMainConfig:
    host: str = "0.0.0.0"
    port: int = 8778
    action_head_server_url: str = "http://127.0.0.1:8777/update_hidden_states"
    pretrained_checkpoint: Union[str, Path] = ""
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@draccus.wrap()
def main(cfg: DeployVlaMainConfig) -> None:
    server = VLAMainServer(cfg)
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    main()