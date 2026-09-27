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

The defaults are `Linear` for linear dimensions, `Constant` for constant dimensions,
and `Extension` for spline dimensions. `Linear` extends the boundary tangent;
`Extension` continues the boundary polynomial. All three modes hold the edge for
constant dimensions. NURBS supports `Constant` and `Extension`; `Linear` is rejected
when constructing an interpolation with `NURBSWeights`.

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
