# Interpolation types

```@example tutorial
using DataInterpolationsND
using Random
using Plots
Random.seed!(2)

t1 = cumsum(0.5 .+ rand(10))
t2 = cumsum(0.5 .+ rand(10))

t1_eval = collect(range(first(t1), last(t1); length = 100))
t2_eval = collect(range(first(t2), last(t2); length = 100))

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

## Extrapolation

Each dimension accepts one `extrapolation::ExtrapolationType.T` keyword. Its mode
applies on both sides of that dimension; different dimensions can use different modes. The
[`ExtrapolationType`](@ref) namespace follows DataInterpolations.jl's mode names;
it is provided by DataInterpolationsND without a runtime dependency on DataInterpolations.

| Mode         | Outside the grid of a dimension                                      |
|:------------ |:-------------------------------------------------------------------- |
| `None`       | throws a [`DataInterpolationsND.ExtrapolationError`](@ref)           |
| `Constant`   | holds the boundary value                                             |
| `Linear`     | continues with the boundary value and the boundary slope             |
| `Extension`  | continues the polynomial of the boundary interval                    |
| `Periodic`   | wraps the query point periodically onto the grid                     |
| `Reflective` | reflects the query point back onto the grid                          |

The defaults are `Linear` for linear dimensions, `Constant` for constant dimensions,
and `Extension` for spline dimensions. For linear dimensions `Linear` and `Extension`
coincide. `Constant`, `Linear` and `Extension` all hold the edge for constant dimensions.
NURBS supports every mode except `Linear`, which is rejected when constructing an
interpolation with `NURBSWeights`.

### Holding the boundary value

`ExtrapolationType.Constant` holds the value at the boundary of that dimension.
Other dimensions continue to interpolate independently. Derivatives along a held
axis are zero strictly outside its grid; at the boundary they use the interpolation's
derivative. Mixed derivatives are zero if they differentiate along a held axis.

```@example tutorial
soc = [0.0, 0.5, 1.0]
temperatures = [0.0, 25.0]
voltages = [3.0 3.1; 3.5 3.6; 3.9 4.0]
voltage = NDInterpolation(
    voltages,
    (
        LinearInterpolationDimension(soc; extrapolation = ExtrapolationType.Constant),
        LinearInterpolationDimension(temperatures; extrapolation = ExtrapolationType.Constant),
    )
)
(voltage(1.2, 25.0), voltage(-0.2, 0.0), voltage(0.5, 40.0))
```

### Linear extrapolation in several dimensions

`ExtrapolationType.Linear` matches DataInterpolations.jl's linear extrapolation in 1D.
If only axis ``i`` is out of range, with ``x_b`` the query point clamped onto the grid,

```math
f(x) = f(x_b) + \frac{\partial f}{\partial x_i}(x_b)\,(x_i - x_{b,i}).
```

If several axes are out of range, the one-dimensional rules compose as a tensor product.
At a corner where axes ``i`` and ``j`` are both out of range this gives

```math
f(x) = f(x_b) + \partial_i f(x_b)\,\Delta_i + \partial_j f(x_b)\,\Delta_j
    + \partial_i \partial_j f(x_b)\,\Delta_i \Delta_j,
\qquad \Delta_k = x_k - x_{b,k},
```

so the extrapolant grows bilinearly in corners. The same composition applies when
dimensions use different modes, for example `Constant` along one axis and `Linear`
along another.

Partial derivatives are consistent with the extrapolant: first partials are continuous
across the boundary, the first partial along an out-of-range `Linear` axis is the
boundary slope, and higher partials along it are zero.

```@example tutorial
spline = NDInterpolation(
    [sin(i) + cos(j) + 0.1i * j for i in 1:5, j in 1:5],
    (
        BSplineInterpolationDimension([0.0, 0.5, 1.0], 3; extrapolation = ExtrapolationType.Linear),
        BSplineInterpolationDimension([0.0, 0.5, 1.0], 3; extrapolation = ExtrapolationType.Linear),
    )
)
x_b = (1.0, 1.0)
Δ = (0.5, 0.25)
∂(orders) = spline(x_b; derivative_orders = orders)
corner = ∂((0, 0)) + ∂((1, 0)) * Δ[1] + ∂((0, 1)) * Δ[2] + ∂((1, 1)) * Δ[1] * Δ[2]
spline((x_b .+ Δ)...) ≈ corner
```

### Periodic and reflective extrapolation

`Periodic` and `Reflective` map the query point onto the grid before evaluating, so
they also work with multi-point evaluation. On mirrored copies of a `Reflective`
dimension, odd derivatives along that dimension change sign.

```@example tutorial
angles = range(0, 2π; length = 9)
periodic = NDInterpolation(
    sin.(angles),
    LinearInterpolationDimension(collect(angles); extrapolation = ExtrapolationType.Periodic)
)
periodic(1.0) ≈ periodic(1.0 + 4π)
```

### Rejecting extrapolation

With `ExtrapolationType.None`, evaluating outside the grid of that dimension throws a
[`DataInterpolationsND.ExtrapolationError`](@ref). For multi-point evaluation this check
happens in [`eval_grid`](@ref) and [`eval_unstructured`](@ref), not when constructing
the dimension.

```@example tutorial
strict = NDInterpolation(
    [1.0, 2.0, 4.0],
    LinearInterpolationDimension([0.0, 1.0, 2.0]; extrapolation = ExtrapolationType.None)
)
try
    strict(2.5)
catch err
    err
end
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

## BSpline Interpolation

```@example tutorial
interp_dims = (
    BSplineInterpolationDimension(t1, 2; t_eval = t1_eval),
    BSplineInterpolationDimension(t2, 2; t_eval = t2_eval)
)
u_bspline = rand(11, 11) # per dimension this is `sum(multiplicities) - degree - 1 = 11`
u_bspline[1:10, 1:10] = u
interp = NDInterpolation(u_bspline, interp_dims)
eval_grid!(out, interp)
heatmap(out)
```

## NURBS Interpolation

```@example tutorial
weights = rand(11, 11)
interp = NDInterpolation(u_bspline, interp_dims; cache = NURBSWeights(weights))
eval_grid!(out, interp)
heatmap(out)
```
