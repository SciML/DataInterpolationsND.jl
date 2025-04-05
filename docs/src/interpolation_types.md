# Interpolation types

```@example tutorial
using NDInterpolations
using Random
using Plots
Random.seed!(2)

t1 = cumsum(0.5 .+ rand(10))
t2 = cumsum(0.5 .+ rand(10))

t1_eval = range(first(t1), last(t1); length = 100)
t2_eval = range(first(t2), last(t2); length = 100)

u = rand(10, 10)
out = zeros(100, 100)
nothing # hide
```

## Linear Interpolation

```@example tutorial
interp_dims = (
    LinearInterpolationDimension(t1; t_eval = t1_eval),
    LinearInterpolationDimension(t2; t_eval = t2_eval)
)
interp = NDInterpolation(u, interp_dims)
eval_grid!(out, interp)
heatmap(out)
```

## Constant Interpolation

```@example tutorial
interp_dims = (
    ConstantInterpolationDimension(t1; t_eval = t1_eval),
    ConstantInterpolationDimension(t2; t_eval = t2_eval)
)
interp = NDInterpolation(u, interp_dims)
eval_grid!(out, interp)
heatmap(out)
```