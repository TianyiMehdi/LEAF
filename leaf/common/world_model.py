from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._flow_model = layers.mlp(cfg.latent_dim + cfg.task_dim + cfg.action_dim + 1, 2*[cfg.mlp_dim], cfg.action_dim)
		self._flow_log_std = nn.Parameter(torch.zeros(cfg.action_dim))
		if self.cfg.use_v_instead_q:
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		else:
			self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.register_buffer("expl_log_std_min", torch.tensor(cfg.expl_log_std_min))
		self.register_buffer("expl_log_std_dif", torch.tensor(cfg.expl_log_std_max) - self.expl_log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z, task, expl=False):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.

		expl: whether to use expl_log_std_min and expl_log_std_dif to calculate log_std.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		if expl:
			log_std = math.log_std(log_std, self.expl_log_std_min, self.expl_log_std_dif)
		else:
			log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		if self.cfg.bmpc:
			mean = torch.tanh(mean)
			action = (mean + eps * log_std.exp()).clamp(-1,1)
		else:
			action = mean + eps * log_std.exp()
			mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info
	
	def _estimate_value_with_action_rollout(self, z, a_0, task, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		action = a_0
		for t in range(horizon):
			reward = math.two_hot_inv(self.reward(z, action, task), self.cfg)
			z = self.next(z, action, task)
			with torch.no_grad():
				# Disable score guidance in inner flow_pi to avoid infinite recursion (flow_step -> _estimate_value -> flow_pi -> flow_step -> ...)
				action, _ = self.flow_pi(z, task, apply_score_guidance=False)
			G = G + discount * reward
			# discount *= (
			# 	self.discount[torch.tensor(task)]
			# 	if self.cfg.multitask
			# 	else self.discount
			# )
			discount *= 0.8
		if self.cfg.use_v_instead_q:
			return G + discount * self.V(z, task, return_type="avg")
		else:
			action, _ = self.flow_pi(z, task, apply_score_guidance=False)
			return G + discount * self.Q(z, action, task, return_type='avg')
	
	def flow_step(self, state, action, time_start, time_end, task):
		"""
		Integrate the learned Phase I velocity field with a midpoint step.
		"""
		velocity_start = self._flow_model(torch.cat([state, action, time_start], dim=-1))
		intermediate_state = action + velocity_start * (time_end - time_start)/2
		intermediate_state = torch.where((time_start == 0).expand_as(action), intermediate_state + action, intermediate_state)

		velocity_mid = self._flow_model(torch.cat([state, intermediate_state, time_start + (time_end - time_start)/2], dim=-1))
		action_t = action + velocity_mid * (time_end - time_start)
		return action_t

	def gradient_step(self, state, action, task):
		"""Phase II: directly refine an action using a look-ahead return gradient."""
		with torch.enable_grad():
			action_detach = action.detach().requires_grad_(True)
			value = self._estimate_value_with_action_rollout(state, action_detach, task, horizon=self.cfg.horizon)
			action_grad = torch.autograd.grad(value, action_detach, create_graph=True)[0]
		action_t = action + action_grad * self.cfg.score_guidance_coef
		return action_t
	
	def flow_pi(self, z, task, expl=False, apply_score_guidance=None):
		"""Sample Phase I flow; append Phase II only when guidance is enabled."""
		if apply_score_guidance is None:
			apply_score_guidance = self.cfg.use_score_guidance
		num_steps = self.cfg.num_flow_steps
		num_grad_steps = int(0.1 / self.cfg.score_guidance_coef)
		device = z.device
		dtype = z.dtype
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		time_start = torch.zeros(*z.shape[:-1], 1, device=device, dtype=dtype)
		time_step = 1.0 / num_steps
		actions = torch.normal(0, 1, size=(*z.shape[:-1], self.cfg.action_dim), device=device, dtype=dtype)
		actions = torch.clamp(actions, -1.0, 1.0)

		for t in range(num_steps):
			time_end = time_start + time_step
			actions = self.flow_step(z, actions, time_start, time_end, task)
			time_start = time_end
		
		if apply_score_guidance:
			for t in range(num_grad_steps):
				actions = self.gradient_step(z, actions, task)
		
		if self.cfg.multitask:
			actions = actions * self._action_masks[task]

		actions = actions.clamp(-1.0, 1.0)
		info = TensorDict({
			"mean": actions,
			"log_std": 1.0,
			"action_prob": 1.,
			"entropy": 0.0,
			"scaled_entropy": 0.0,
		})
		return actions, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
		assert not self.cfg.use_v_instead_q

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

	def V(self, z, task, return_type='avg', target=False, detach=False):
		"""
		Predict state value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled V-values.
			- `avg`: return the average of two randomly subsampled V-values.
			- `all`: return all V-values.
		`target` specifies whether to use the target V-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
		assert self.cfg.use_v_instead_q

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		if target:
			vnet = self._target_Qs
		elif detach:
			vnet = self._detach_Qs
		else:
			vnet = self._Qs
		out = vnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		V = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return V.min(0).values
		return V.sum(0) / 2
