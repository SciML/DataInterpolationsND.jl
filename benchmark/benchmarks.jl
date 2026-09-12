using BenchmarkTools
using DataInterpolationsND
using DataInterpolationsND: EmptyCache

const SUITE = BenchmarkGroup()

n1d = 100
t1 = collect(range(0.0, 1.0; length = n1d))
u1 = sin.(2π .* t1)
t_eval_1d = collect(range(0.05, 0.95; length = 50))

n2d = 20
t2 = collect(range(0.0, 1.0; length = n2d))
u2 = [sin(2π * x) * cos(2π * y) for x in t2, y in t2]
pts_2d = [(0.37, 0.62), (0.11, 0.89), (0.55, 0.45), (0.78, 0.21), (0.33, 0.77)]

SUITE["1d"] = BenchmarkGroup()
SUITE["1d"]["linear_construct"] = @benchmarkable NDInterpolation($u1, LinearInterpolationDimension($t1))
itp_lin = NDInterpolation(u1, LinearInterpolationDimension(t1))
SUITE["1d"]["linear_eval"] = @benchmarkable $itp_lin.($t_eval_1d)
SUITE["1d"]["constant_construct"] = @benchmarkable NDInterpolation($u1, ConstantInterpolationDimension($t1))
itp_con = NDInterpolation(u1, ConstantInterpolationDimension(t1))
SUITE["1d"]["constant_eval"] = @benchmarkable $itp_con.($t_eval_1d)
u1_bs = sin.(2π .* range(0.0, 1.0; length = n1d + 1))
SUITE["1d"]["bspline_construct"] = @benchmarkable NDInterpolation($u1_bs, BSplineInterpolationDimension($t1, 2))
itp_bs = NDInterpolation(u1_bs, BSplineInterpolationDimension(t1, 2))
SUITE["1d"]["bspline_eval"] = @benchmarkable $itp_bs.($t_eval_1d)
SUITE["1d"]["bspline_derivative"] = @benchmarkable $itp_bs.($t_eval_1d; derivative_orders = (1,))

SUITE["2d"] = BenchmarkGroup()
dims_lin = (LinearInterpolationDimension(t2), LinearInterpolationDimension(t2))
SUITE["2d"]["linear_construct"] = @benchmarkable NDInterpolation($u2, $dims_lin)
itp_2d = NDInterpolation(u2, dims_lin)
SUITE["2d"]["linear_eval"] = @benchmarkable foreach(p -> $itp_2d(p), $pts_2d)
SUITE["2d"]["eval_grid"] = @benchmarkable eval_grid($itp_2d)

u2_bs = [sin(2π * x) * cos(2π * y) for x in range(0.0, 1.0; length = n2d + 1), y in range(0.0, 1.0; length = n2d + 1)]
dims_bs = (BSplineInterpolationDimension(t2, 2), BSplineInterpolationDimension(t2, 2))
SUITE["2d"]["bspline_construct"] = @benchmarkable NDInterpolation($u2_bs, $dims_bs)
itp_2d_bs = NDInterpolation(u2_bs, dims_bs)
SUITE["2d"]["bspline_eval"] = @benchmarkable foreach(p -> $itp_2d_bs(p), $pts_2d)
SUITE["2d"]["bspline_derivative"] = @benchmarkable foreach(
    p -> $itp_2d_bs(p; derivative_orders = (1, 0)), $pts_2d
)
