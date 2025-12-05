"""
deploy_rdt_head.py

Starts the RDT Action Head server (Sys1), which receives client requests and
last_hidden_states from the VLA main server (Sys2) to compute the final action.
"""

import base64
import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import json_numpy
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Apply JSON numpy patch for serialization
json_numpy.patch()

from experiments.robot.openvla_utils import (
    _get_rdt_action,
    get_rdt_action_head,
    prepare_rdt_image_inputs,
)
from prismatic.vla.constants import ACTION_DIM

# === Server Interface ===
class RDTHeadServer:
    def __init__(self, cfg: "DeployRdtHeadConfig") -> None:
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load RDT Action Head
        self.action_head = get_rdt_action_head(cfg)
        
        # Load SiglipVisionTower
        from fast_models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        self.siglip_vision_encoder = SiglipVisionTower(
            vision_tower=cfg.pretrained_vision_encoder_name_or_path,
            args=None
        )
        self.siglip_vision_encoder.vision_tower.to(self.device, dtype=torch.bfloat16)

        # State management for hidden states
        self.latest_hidden_states = None
        
        self.lock = threading.Lock()
        self._hs_cv = threading.Condition(self.lock)
        self.last_update_time = None

    def update_hidden_states(self, payload: Dict[str, Any]) -> JSONResponse:
        """Update hidden states from VLA main server (Sys2)"""
        try:
            serialized_states = payload["hidden_states"]
            numpy_states = json_numpy.loads(base64.b64decode(serialized_states))
            
            with self._hs_cv:
                self.latest_hidden_states = {
                    "text_hidden_states": torch.from_numpy(numpy_states["text_hidden_states"].copy()).to(self.device, dtype=torch.bfloat16),
                    "actions_hidden_states": torch.from_numpy(numpy_states["actions_hidden_states"].copy()).to(self.device, dtype=torch.bfloat16),
                }
                self.last_update_time = time.time()
                # Wake up waiting threads
                self._hs_cv.notify_all()
            
            logging.info("Successfully updated hidden states from VLA server")
            return JSONResponse({"status": "ok"})
            
        except Exception:
            logging.error(f"Failed to update hidden states: {traceback.format_exc()}")
            return JSONResponse({"status": "error", "message": "Failed to update hidden states"}, status_code=500)

    def get_server_action(self, payload: Dict[str, Any]) -> JSONResponse:
        """Compute action using RDT head and latest hidden states"""
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload

            # Wait for hidden states to be available
            with self._hs_cv:
                if self.latest_hidden_states is None:
                    logging.info("Waiting for hidden states from the server...")
                    start = time.time()
                    last_log_time = start
                    while self.latest_hidden_states is None:
                        elapsed = time.time() - start
                        remaining = self.cfg.hidden_states_wait_timeout - elapsed
                        if remaining <= 0:
                            break
                        wait_interval = min(1.0, remaining)
                        self._hs_cv.wait(timeout=wait_interval)
                        if self.latest_hidden_states is not None:
                            break
                        now = time.time()
                        if now - last_log_time >= 3.0:
                            logging.info(
                                f"Still waiting for hidden states... Elapsed: {elapsed:.2f}s, Remaining: {max(self.cfg.hidden_states_wait_timeout - (now - start), 0):.2f}s"
                            )
                            last_log_time = now
                if self.latest_hidden_states is None:
                    logging.error("Timeout waiting for hidden states from the server.")
                    return JSONResponse({
                        "error": f"Timeout waiting for hidden states (>{self.cfg.hidden_states_wait_timeout}s). Ensure VLA main server is pushing updates."
                    }, status_code=504)

                # Check for staleness
                if self.last_update_time and (time.time() - self.last_update_time) > 5.0:
                    logging.warning(f"Hidden states are stale (age: {time.time() - self.last_update_time:.2f}s)")

                text_hidden_states = self.latest_hidden_states["text_hidden_states"].clone()
                actions_hidden_states = self.latest_hidden_states["actions_hidden_states"].clone()
                update_age = time.time() - self.last_update_time if self.last_update_time else float('inf')

            logging.info(f"Using hidden states (age: {update_age:.3f}s)")
            
            # Prepare inputs for _get_rdt_action
            pixel_values = prepare_rdt_image_inputs(
                observation=observation,
                cfg=self.cfg
            )
            
            # Extract proprio if available (raw; normalization, if needed, should match VLA server side)
            proprio = None
            if self.cfg.use_proprio:
                proprio = observation.get("state")
                if proprio is not None:
                    if isinstance(proprio, list):
                        import numpy as np
                        proprio = np.array(proprio)
                else:
                    logging.info("Proprio is None")
            
            start_time = time.time()
            action = _get_rdt_action(
                action_head=self.action_head,
                pixel_values=pixel_values,
                cfg=self.cfg,
                text_hidden_states=text_hidden_states,
                actions_hidden_states=None,
                proprio=proprio,
                siglip_vision_encoder=self.siglip_vision_encoder,
            )
            end_time = time.time()
            inference_time = end_time - start_time
            logging.info(f"Inference completed in {inference_time:.3f} seconds")
            logging.info(f"Successfully computed action with shape: {action.shape}")
            
            # Convert to list format for client
            action_list = [action[i] for i in range(len(action))]
            
            if double_encode:
                return JSONResponse(json_numpy.dumps(action_list))
            else:
                return JSONResponse(action_list)

        except Exception:
            logging.error(f"Failed to compute action: {traceback.format_exc()}")
            return JSONResponse({
                "status": "error", 
                "message": "Failed to compute action with RDT head."
            }, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        self.app.post("/update_hidden_states")(self.update_hidden_states)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployRdtHeadConfig:
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8777

    # RDT Head Parameters
    use_rdt: bool = True
    rdt_config_path: Optional[str] = None
    pretrained_rdt_path: Optional[str] = None
    pretrained_vision_encoder_name_or_path: str = "google/siglip-so400m-patch14-224"
    
    use_proprio: bool = True
    num_images_in_input: int = 2
    center_crop: bool = True

    # 仅传入用于反归一化的 key
    unnorm_key: Optional[str] = None

    hidden_states_wait_timeout: float = 20.0


@draccus.wrap()
def main(cfg: DeployRdtHeadConfig) -> None:
    server = RDTHeadServer(cfg)
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    main()