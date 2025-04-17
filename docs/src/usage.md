## Single point evaluation

### Setup

```@example tutorial
using NDInterpolations
using Random
Random.seed!(1)

interp_dims = (
    LinearInterpolationDimension(cumsum(0.5 .+ rand(5))),
    LinearInterpolationDimension(cumsum(0.5 .+ rand(10)))
)

t_eval_1 = range(
    first(interp_dims[1].t),
    last(interp_dims[1].t),
    length = 100
)
t_eval_2 = range(
    first(interp_dims[2].t),
    last(interp_dims[2].t),
    length = 100
)
```

### Scalar output

```@example tutorial
using Plots

u = rand(5, 10)
itp = NDInterpolation(u, interp_dims)
out = itp.(t_eval_1, t_eval_2')
heatmap(out)
```

### Vector output, out of place

```@example tutorial
u = reshape(u, 5, 10, 1)
itp = NDInterpolation(u, interp_dims)
out = itp.(t_eval_1, t_eval_2')
heatmap(map(only, out))
```

### vector output, in place

```@example tutorial
out = zeros(100, 100)
for I in CartesianIndices(out)
    i, j = Tuple(I)
    itp(view(out, i, j:j), t_eval_1[i], t_eval_2[j])
end
heatmap(out)
```

## Single point derivative evaluation

### Partial derivative w.r.t. first input (ForwardDiff)

```@example tutorial
using ForwardDiff

u = reshape(u, 5, 10)
itp = NDInterpolation(u, interp_dims)
∂₁itp = (t1, t2) -> ForwardDiff.derivative(t_ -> itp(t_, t2), t1)
out = ∂₁itp.(t_eval_1, t_eval_2')
heatmap(out)
```

### Partial derivative w.r.t. first input (analytic)

```@example tutorial
out = itp.(t_eval_1, t_eval_2'; derivative_orders = (1, 0))
heatmap(out)
```

## Multiple point evaluation (using KA)

### Unstructured multi-point scalar evaluation (out of place)

```@example tutorial
using LinearAlgebra

interp_dims = (
    LinearInterpolationDimension(interp_dims[1].t; t_eval = t_eval_1),
    LinearInterpolationDimension(interp_dims[2].t; t_eval = t_eval_2)
)
itp = NDInterpolation(u, interp_dims)

out = eval_unstructured(itp)
heatmap(diagm(out))
```

### Grid vector evaluation (in place)

```@example tutorial
u = reshape(u, 5, 10, 1)
itp = NDInterpolation(u, interp_dims)
out = zeros(100, 100, 1)
eval_grid!(out, itp)
heatmap(out[:, :, 1])
```
