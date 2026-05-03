"""MPS-accelerated model wrapper for RLlib."""

import torch
from gymnasium.spaces import Space
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

MPS_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else None


class MPSFullyConnectedNetwork(TorchModelV2, torch.nn.Module):
    """Fully-connected network that offloads the forward pass to Apple MPS.

    Wraps RLlib's ``FullyConnectedNetwork`` so that all tensor computations
    execute on the Metal Performance Shaders (MPS) device when available,
    then moves results back to CPU before returning them to RLlib.  If MPS is
    not available the model transparently falls back to CPU.

    This is necessary because RLlib's rollout workers assume CPU-resident
    tensors; the model therefore acts as a device-translation shim rather than
    changing any algorithmic behaviour.

    Parameters
    ----------
    obs_space : Space
        Gymnasium observation space passed through from RLlib.
    action_space : Space
        Gymnasium action space passed through from RLlib.
    num_outputs : int
        Number of output logits (action dimensions).
    model_config : ModelConfigDict
        RLlib model configuration dictionary (e.g. ``fcnet_hiddens``).
    name : str
        Unique model name used by RLlib for weight registration.
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        self.device = MPS_DEVICE or torch.device("cpu")
        self._base_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_base"
        )
        self._base_model.to(self.device)
        self._last_value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """Run the forward pass on MPS and return CPU tensors.

        Moves the observation tensor to the MPS device, delegates to the
        wrapped ``FullyConnectedNetwork``, caches the value estimate for the
        subsequent ``value_function()`` call, and moves all outputs back to
        CPU so that RLlib's sampling infrastructure can consume them without
        device mismatches.

        Parameters
        ----------
        input_dict : dict[str, TensorType]
            RLlib input dictionary; must contain the key ``"obs"``.
        state : list[TensorType]
            Recurrent state tensors (empty list for feed-forward models).
        seq_lens : TensorType
            Tensor of sequence lengths used for recurrent models.

        Returns
        -------
        tuple[TensorType, list[TensorType]]
            ``(action_logits, new_state)`` where ``action_logits`` is a
            CPU tensor of shape ``[B, num_outputs]`` and ``new_state`` is
            the (possibly empty) updated recurrent state.
        """
        obs = input_dict["obs"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        obs_mps = obs.to(self.device)
        input_dict_mps = {**input_dict, "obs": obs_mps, "obs_flat": obs_mps}

        output, new_state = self._base_model(input_dict_mps, state, seq_lens)

        # Cache value for value_function() call
        self._last_value = self._base_model.value_function()

        # Return to CPU for RLlib
        return output.cpu(), new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value estimate cached during the last ``forward()`` call.

        RLlib calls this method immediately after ``forward()`` to retrieve the
        critic (baseline) value for PPO/APPO advantage computation.  The value
        tensor was computed on MPS during the forward pass and is moved to CPU
        here to match the device expected by RLlib.

        Returns
        -------
        TensorType
            1-D CPU tensor of shape ``[B]`` containing per-sample value
            estimates for the most recent batch.

        Raises
        ------
        ValueError
            If ``forward()`` has not been called yet (no cached value).
        """
        if self._last_value is None:
            raise ValueError("forward() must be called before value_function()")
        return self._last_value.cpu()
