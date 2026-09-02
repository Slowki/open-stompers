"""Feed-forward for RSL-based policies."""

from itertools import pairwise
from pathlib import Path
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
import orbax.checkpoint as checkpointing
from flax import nnx
from jaxtyping import ArrayLike, Float

from open_stompers_modeling.checkpoint import read_checkpoint, restore_parameters
from open_stompers_modeling.export import export_onnx


class MlpActor(nnx.Module):
    """MLP-based policy with ELU hidden layers and linear action outputs."""

    #: Model kind stored in checkpoint metadata.
    KIND: ClassVar[str] = "rsl_actor"

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, rngs: nnx.Rngs) -> Self:
        """Create an actor and restore weights from a project or Orbax checkpoint."""
        checkpoint_path = Path(path).resolve()
        if (checkpoint_path / "metadata").is_dir():
            saved = read_checkpoint(checkpoint_path, kind=cls.KIND)
            description = saved["metadata"]["model"]
            actor = cls(**description, rngs=rngs)
            graphdef, parameters = nnx.split(actor, nnx.Param)
            return nnx.merge(graphdef, restore_parameters(parameters, saved["parameters"]))

        parameters = checkpointing.PyTreeCheckpointer().restore(checkpoint_path)
        layers = parameters["layers"]
        ordered_layers = [layers[index] for index in range(len(layers))]
        dimensions = [int(ordered_layers[0]["kernel"].shape[0])]
        dimensions.extend(int(layer["kernel"].shape[1]) for layer in ordered_layers)
        actor = cls(
            observation_size=dimensions[0],
            action_size=dimensions[-1],
            hidden_sizes=tuple(dimensions[1:-1]),
            rngs=rngs,
        )
        graphdef, template = nnx.split(actor, nnx.Param)
        actor_parameters = {
            "layers": {
                str(index): {
                    "kernel": jnp.asarray(layer["kernel"]),
                    "bias": jnp.asarray(layer["bias"]),
                }
                for index, layer in enumerate(ordered_layers)
            }
        }
        return nnx.merge(graphdef, restore_parameters(template, actor_parameters))

    def __init__(
        self,
        observation_size: int = 345,
        action_size: int = 20,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Create the policy layers.

        Args:
            observation_size: Width of the prepared observation vector.
            action_size: Number of action means.
            hidden_sizes: Widths of the ELU hidden layers.
            rngs: Random streams for parameter initialization.
        """
        dimensions = (observation_size, *hidden_sizes, action_size)
        if min(dimensions) < 1:
            raise ValueError("Actor dimensions must be positive")
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.layers = nnx.List(
            [
                nnx.Linear(input_size, output_size, precision=jax.lax.Precision.HIGHEST, rngs=rngs)
                for input_size, output_size in pairwise(dimensions)
            ]
        )

    def __call__(
        self, observations: Float[ArrayLike, "*batch observations"]
    ) -> Float[jax.Array, "*batch actions"]:
        """Compute action means from prepared observation vectors.

        Args:
            observations: Flattened observations with optional leading batch axes.

        Returns:
            Unclipped action means with the same leading batch axes.
        """
        hidden = jnp.asarray(observations, dtype=jnp.float32)
        if hidden.ndim < 1 or hidden.shape[-1] != self.observation_size:
            raise ValueError("Observation width does not match the actor")
        for layer in self.layers[:-1]:
            hidden = jax.nn.elu(layer(hidden))
        return self.layers[-1](hidden)

    def act_inference(
        self, observations: Float[ArrayLike, "*batch observations"]
    ) -> Float[jax.Array, "*batch actions"]:
        """Compute deterministic action means without sampling exploration noise.

        Args:
            observations: Prepared observation vectors.

        Returns:
            Unclipped action means.
        """
        return self(observations)

    def export_onnx(self, path: str | Path) -> Path:
        """Export the deterministic actor with a dynamic batch dimension."""
        graphdef, parameters = nnx.split(self, nnx.Param)

        def apply(observations: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, parameters)(observations)

        return export_onnx(
            apply,
            path,
            [jax.ShapeDtypeStruct(("batch", self.observation_size), jnp.float32)],
            input_names=["observations"],
            output_names=["actions"],
            model_name="rsl_mlp_actor",
        )
