# Experimental Triton shape stubs

These source-only stubs model a small subset of Triton 3.9. They are not
published or enabled automatically. Add this directory and
`../pyrefly-shape-extensions` to your project's Pyrefly `search_path` to use them.

`tensor[Shape, DType]` tracks dimensions and the Triton dtype. For example,
`tl.arange(0, 128)` produces `tensor[[128], dtype[Literal["int32"]]]`.
The stubs cover `arange`, `full`, `to`, addition, comparison with `<`, indexing,
and ordinary pointer loads and stores. Memory masks, fallback values, and stored
values must broadcast to the pointer shape. Stores permit numeric dtype casts.
In Triton 3.9, a store returns a scalar tensor with dtype `void`.

See [vector_add.py](examples/vector_add.py) for inferred types and an invalid
load. Keep static type aliases inside `TYPE_CHECKING` blocks, as that example
does. Triton's runtime classes do not implement these generic annotations.

Mixed numeric dtypes retain the output shape but use gradual dtype information.
Only `int1`, `int32`, and `float32` arithmetic is modeled. Kernel launch arguments,
block pointers, dtype promotion, and compiler restrictions such as power-of-two
extents are outside this prototype. `constexpr` parameters remain gradual until
their values are known statically.

Run the static checks from the repository root. They require no Triton or GPU:

```bash
python3 tensor-shapes/pyrefly-triton-stubs/run_pyrefly.py
```

The runtime cross-checks use Triton 3.9.0 and NumPy. They exercise Triton's real
semantic rules with its CPU interpreter and NumPy-backed pointers, without a GPU:

```bash
python3 tensor-shapes/pyrefly-triton-stubs/run_runtime_tests.py
```
