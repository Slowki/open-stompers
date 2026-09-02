# Coding Guidance for AI

## Compatibility

- In general do not worry about maintaining backwards compatibility for the code you are writing.

## Python

- Use `uv` for interacting with Python
- Use Google-style docstrings with Markdown content
- Conform to PEP8
- Use modern Python features like match statements and the walrus operator to decrease code verbosity

### JAX

- Inputs to JAX functions should be `jax.typing.ArrayLike` and convert to a `jax.Array` using `jnp.asarray` internally unless they specifically require a concrete `jax.Array`. Use `jaxtyping` to annotate the shape and type of `Array`s and `ArrayLike`s.

```python
import jax
from jax import numpy as jnp
from jaxtyping import Float, ArrayLike

def my_function(image: Float[ArrayLike, "C H W"]) -> Float[jax.Array, "C H W"]:
    """Function to operate on `image`.

    Args:
        image: An image in channel, height, width order.

    Returns:
        An image in channel, height, width order.
    """
```

- Outputs from JAX functions should be `jaxtyping.Array` or some other JAX-compatible type like a PyTree.
- Neural networks should be modeled using Flax NNX (`flax.nnx`).
- Complex inputs and outputs should be represented by using a flax dataclass or a PyTree.

```python
from flax import struct
import jax

@struct.dataclass
class MyThing:
    """Data for my thing."""
    #: A field representing some data.
    a_field: jax.Array
```

## Rust

- Use Rust 2024
- Use `thiserror::Error` when making defining error types.
- Use `uom` for well typed unit tests.
- Follow best practice for writing Rustdoc comments, example:

```rust
/// Compute the square root of a value.
///
/// # Arguments
///
/// * `value` - The value to compute the square of.
pub const fn square_root(value: i32) -> i32 {
    // ...
}
```

## After making a change

After you are done making a change run `tools/fix` to run the formatters and linters.
