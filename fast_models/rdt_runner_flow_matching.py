import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_models.hub_mixin import CompatiblePyTorchModelHubMixin
from fast_models.rdt.model import RDT


class RDTRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config,
                 lang_token_dim, img_token_dim, state_token_dim,
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None,
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner, self).__init__()
        # Create flow matching model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'],
            in_features=lang_token_dim,
            out_features=hidden_size
        ).to(dtype)
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'],
            in_features=img_token_dim,
            out_features=hidden_size
        ).to(dtype)
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,
            out_features=hidden_size
        ).to(dtype)

        # Flow matching configuration
        flow_matching_config = config.get('flow_matching', {})
        self.sigma_min = flow_matching_config.get('sigma_min', 1e-4)
        self.num_inference_steps = flow_matching_config.get('num_inference_steps', 10)
        self.ode_solver = flow_matching_config.get('ode_solver', 'euler')  # 'euler' or 'rk4'

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print("Flow Matching params: %e" % sum(
            [p.numel() for p in self.model.parameters()] +
            [p.numel() for p in self.lang_adaptor.parameters()] +
            [p.numel() for p in self.img_adaptor.parameters()] +
            [p.numel() for p in self.state_adaptor.parameters()]))

        self.action_predictor = None

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)

        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def _ensure_action_predictor(self, feature_dim, device, dtype):
        if self.action_predictor is None:
            from prismatic.vla.constants import ACTION_DIM
            self.action_predictor = nn.Linear(feature_dim, ACTION_DIM).to(device, dtype=dtype)
        return self.action_predictor

    def get_velocity_field(self, x_t, t, lang_cond, lang_attn_mask, img_cond,
                          state_traj, action_mask, ctrl_freqs):
        """
        Compute the velocity field v_t(x_t) for flow matching
        
        x_t: current state at time t
        t: time tensor (batch_size,) or scalar
        
        Note: We use the convention where t=1 is noise and t=0 is the target distribution
        """
        # Prepare state-action trajectory
        action_traj = torch.cat([x_t, action_mask], dim=2)
        action_traj = self.state_adaptor(action_traj)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        # Predict the velocity field
        if t.dim() == 0:  # scalar
            t = t.unsqueeze(0).expand(x_t.shape[0])
        
        velocity = self.model(state_action_traj, ctrl_freqs,
                            t.to(x_t.device),
                            lang_cond, img_cond, lang_mask=lang_attn_mask)
        
        return velocity

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond,
                           state_traj, action_mask, ctrl_freqs, actions_hidden_states=None):
        '''
        Flow matching sampling using ODE solver
        Convention: t=1 is noise, t=0 is the target distribution
        
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        actions_hidden_states: (batch_size, chunk_len * action_dim, hidden_dim), optional

        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        
        # Initialize x_1 from noise (or from actions_hidden_states)
        if actions_hidden_states is not None:
            batch_size = actions_hidden_states.shape[0]
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
            rearranged_actions_hidden_states = actions_hidden_states.reshape(
                batch_size, NUM_ACTIONS_CHUNK, -1
            )
            
            feature_dim = rearranged_actions_hidden_states.shape[-1]
            action_predictor = self._ensure_action_predictor(feature_dim, device, dtype)
            
            initial_action = action_predictor(rearranged_actions_hidden_states)
            
            if initial_action.shape[1] != self.pred_horizon:
                if initial_action.shape[1] < self.pred_horizon:
                    padding_len = self.pred_horizon - initial_action.shape[1]
                    last_action = initial_action[:, -1:, :].repeat(1, padding_len, 1)
                    x_t = torch.cat([initial_action, last_action], dim=1)
                else:
                    x_t = initial_action[:, :self.pred_horizon, :]
            else:
                x_t = initial_action
                
            # Add noise for flow matching initialization
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        else:
            # Start from pure noise at t=1
            x_t = torch.randn(
                size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                dtype=dtype, device=device)

        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)

        # ODE integration from t=1 to t=0 (noise to data)
        dt = -1.0 / self.num_inference_steps  # negative step
        t = 1.0  # start from t=1
        
        for step in range(self.num_inference_steps):
            if self.ode_solver == 'euler':
                # Euler method
                t_tensor = torch.tensor(t, dtype=dtype, device=device)
                v_t = self.get_velocity_field(
                    x_t, t_tensor, lang_cond, lang_attn_mask, img_cond,
                    state_traj, action_mask_expanded, ctrl_freqs
                )
                x_t = x_t + dt * v_t
                
            elif self.ode_solver == 'rk4':
                # Runge-Kutta 4th order
                t_tensor = torch.tensor(t, dtype=dtype, device=device)
                k1 = self.get_velocity_field(
                    x_t, t_tensor, lang_cond, lang_attn_mask, img_cond,
                    state_traj, action_mask_expanded, ctrl_freqs
                )
                
                t_half = t + 0.5 * dt
                t_half_tensor = torch.tensor(t_half, dtype=dtype, device=device)
                k2 = self.get_velocity_field(
                    x_t + 0.5 * dt * k1, t_half_tensor,
                    lang_cond, lang_attn_mask, img_cond,
                    state_traj, action_mask_expanded, ctrl_freqs
                )
                
                k3 = self.get_velocity_field(
                    x_t + 0.5 * dt * k2, t_half_tensor,
                    lang_cond, lang_attn_mask, img_cond,
                    state_traj, action_mask_expanded, ctrl_freqs
                )
                
                t_next = t + dt
                t_next_tensor = torch.tensor(t_next, dtype=dtype, device=device)
                k4 = self.get_velocity_field(
                    x_t + dt * k3, t_next_tensor,
                    lang_cond, lang_attn_mask, img_cond,
                    state_traj, action_mask_expanded, ctrl_freqs
                )
                
                x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Update time: t decreases from 1.0 to 0.0
            t = t + dt
            x_t = x_t.to(dtype)

        # Apply action mask
        x_t = x_t * action_mask_expanded

        return x_t

    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs
                    ) -> torch.Tensor:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        # Sample random time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # x_0 is data (at t=0), x_1 is noise (at t=1)
        x_0 = action_gt
        x_1 = torch.randn_like(action_gt)
        
        # Flow matching: x_t = (1-t) * x_0 + t * x_1
        # When t=0: x_t = x_0 (data)
        # When t=1: x_t = x_1 (noise)
        t_expanded = t.view(batch_size, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Target velocity: v_t = dx_t/dt = x_1 - x_0
        # This points from data (t=0) to noise (t=1)
        # But we integrate backward, so the sign is correct!
        target_velocity = x_1 - x_0
        
        # Prepare state tokens 
        state_tokens = state_tokens.to(dtype=torch.bfloat16, device=device)
        action_mask = action_mask.to(dtype=torch.bfloat16)
        x_t = x_t.to(dtype=torch.bfloat16)

        state_tokens_with_mask = torch.cat([state_tokens, action_mask], dim=2)
        
        # Adapt conditions
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens_with_mask)

        action_traj = torch.cat([x_t, action_mask.expand(-1, x_t.shape[1], -1)], dim=2)
        action_traj = self.state_adaptor(action_traj)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)
        
        # Predict velocity field
        pred_velocity = self.model(state_action_traj, ctrl_freqs, 
                                   t, lang_cond, img_cond, 
                                   lang_mask=lang_attn_mask)

        # Flow matching loss: MSE between predicted and target velocity
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss
    

    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, actions_hidden_states=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim)
        ctrl_freqs: (batch_size,), control frequency for each sample.
        actions_hidden_states: (batch_size, chunk_len * action_dim, hidden_dim), optional

        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = state_tokens.to(action_mask.device)
        state_tokens = state_tokens.to(dtype=torch.bfloat16)
        action_mask = action_mask.to(dtype=torch.bfloat16)

        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)

        # Run flow matching sampling
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond,
            state_traj, action_mask, ctrl_freqs,
            actions_hidden_states=actions_hidden_states
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
