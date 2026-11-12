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

With `ExtrapolationType.Linear`, the one-dimensional rule
``f(x) = f(x_b) + \partial_i f(x_b)\,(x_i - x_{b,i})``, with ``x_b`` the query point
clamped onto the grid, composes as a tensor product over the out-of-range axes. In a
corner where axes ``i`` and ``j`` are both out of range,

```math
f(x) = f(x_b) + \partial_i f(x_b)\,\Delta_i + \partial_j f(x_b)\,\Delta_j
    + \partial_i \partial_j f(x_b)\,\Delta_i \Delta_j,
\qquad \Delta_k = x_k - x_{b,k}.
```

```@example tutorial
mode = ExtrapolationType.Linear
spline = NDInterpolation(
    [sin(i) + cos(j) + 0.1i * j for i in 1:5, j in 1:5],
    ntuple(_ -> BSplineInterpolationDimension([0.0, 0.5, 1.0], 3; extrapolation = mode), 2)
)
∂(orders) = spline((1.0, 1.0); derivative_orders = orders)
spline(1.5, 1.25) ≈ ∂((0, 0)) + ∂((1, 0)) * 0.5 + ∂((0, 1)) * 0.25 + ∂((1, 1)) * 0.5 * 0.25
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
