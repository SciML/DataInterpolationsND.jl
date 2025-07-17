# Splines

To be expanded. A nice thing to demonstrate is the plotting of the basis functions:

```@example tutorial
using DataInterpolationsND
using Plots

itp_dim = BSplineInterpolationDimension(
    [1.0, 2.0, 3.0, 5.0, 8.0], 3; t_eval = collect(range(1.0, 8.0; length = 100))
)
plot(itp_dim)
```
