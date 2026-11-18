# API

## Structs

```@docs
NDInterpolation
LinearInterpolationDimension
ConstantInterpolationDimension
BSplineInterpolationDimension
NURBSWeights
```

## Extrapolation

Every dimension constructor accepts `extrapolation::ExtrapolationType.T`, which
applies the selected mode on both sides of that dimension. The defaults are
`Linear` for linear dimensions, `Constant` for constant dimensions, and `Extension`
for spline dimensions. See the extrapolation section of
[Interpolation types](interpolation_types.md#Extrapolation) for the multi-dimensional semantics.

```@docs
ExtrapolationType
ExtrapolationType.T
ExtrapolationType.None
ExtrapolationType.Constant
ExtrapolationType.Linear
ExtrapolationType.Extension
ExtrapolationType.Periodic
ExtrapolationType.Reflective
DataInterpolationsND.ExtrapolationError
```

## Multi-point evaluation

```@docs
eval_unstructured
eval_unstructured!
eval_grid
eval_grid!
```
