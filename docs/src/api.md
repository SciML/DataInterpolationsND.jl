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
for spline dimensions.

```@docs
ExtrapolationType
ExtrapolationType.T
ExtrapolationType.Constant
ExtrapolationType.Linear
ExtrapolationType.Extension
```

## Multi-point evaluation

```@docs
eval_unstructured
eval_unstructured!
eval_grid
eval_grid!
```
