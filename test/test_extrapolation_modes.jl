using DataInterpolationsND
using DataInterpolationsND: ExtrapolationError
import DataInterpolations
using ForwardDiff
using Test

const ET = ExtrapolationType

cubic_2d(mode) = NDInterpolation(
    [sin(i) + cos(j) + 0.1i * j for i in 1:5, j in 1:5],
    ntuple(_ -> BSplineInterpolationDimension([0.0, 0.5, 1.0], 3; extrapolation = mode), 2)
)

@testset "None throws outside the grid" begin
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]
    di = DataInterpolations.LinearInterpolation(u, t)
    interp = NDInterpolation(u, LinearInterpolationDimension(t; extrapolation = ET.None))
    @test interp.([0.0, 0.35, 1.1]) ≈ di.([0.0, 0.35, 1.1])
    @test_throws ExtrapolationError(1, true) interp(-0.1)
    @test_throws ExtrapolationError(1, false) interp(1.2; derivative_orders = (1,))
    # Multi point evaluation checks when evaluating, not when constructing
    interp = NDInterpolation(
        u, LinearInterpolationDimension(t; extrapolation = ET.None, t_eval = [0.5, 1.5])
    )
    @test_throws ExtrapolationError(1, false) eval_grid(interp)
end

@testset "Periodic and Reflective match DataInterpolations in 1D" begin
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]
    # Several periods away on both sides, avoiding knots after folding
    ts = [-2.35, -0.4, 1.63, 3.33]
    for (mode, di_mode) in (
            (ET.Periodic, DataInterpolations.ExtrapolationType.Periodic),
            (ET.Reflective, DataInterpolations.ExtrapolationType.Reflective),
        )
        di = DataInterpolations.LinearInterpolation(u, t; extrapolation = di_mode)
        interp = NDInterpolation(u, LinearInterpolationDimension(t; extrapolation = mode, t_eval = ts))
        derivs = DataInterpolations.derivative.(Ref(di), ts)
        @test interp.(ts) ≈ di.(ts)
        @test eval_grid(interp) ≈ di.(ts)
        @test [interp(x; derivative_orders = (1,)) for x in ts] ≈ derivs
        @test eval_grid(interp; derivative_orders = (1,)) ≈ derivs
    end
end

@testset "2D Linear extrapolation in a corner" begin
    # Both axes out of range: the tensor product picks up the mixed partial term
    interp = cubic_2d(ET.Linear)
    x_b = (1.0, 1.0)
    ∂(orders) = interp(x_b; derivative_orders = orders)
    for Δ in ((0.5, 0.25), (3.0, 0.1))
        @test interp((x_b .+ Δ)...) ≈
            ∂((0, 0)) + ∂((1, 0)) * Δ[1] + ∂((0, 1)) * Δ[2] + ∂((1, 1)) * Δ[1] * Δ[2]
    end
end

@testset "Derivative continuity at the boundary" begin
    interp = cubic_2d(ET.Linear)
    ε = 1.0e-9
    for other in (0.3, 2.0), orders in ((1, 0), (0, 1))
        @test interp(1.0 - ε, other; derivative_orders = orders) ≈
            interp(1.0 + ε, other; derivative_orders = orders) atol = 1.0e-4
    end
    # Outside a Linear axis the slope is the boundary slope and higher derivatives vanish
    @test interp(5.0, 0.3; derivative_orders = (1, 0)) ≈ interp(1.0, 0.3; derivative_orders = (1, 0))
    @test interp(5.0, 0.3; derivative_orders = (2, 0)) == 0
end

@testset "Default extrapolation is unchanged" begin
    # Reference values computed with the default extrapolation before `None`, `Periodic`
    # and `Reflective` were added
    x = [0.0, 0.4, 1.0, 1.5]
    y = [-1.0, 0.0, 2.0]
    u = [sin(3a) + cos(2b) + a * b for a in x, b in y]
    u_bspline = [sin(i) + cos(2j) + 0.3i * j for i in 1:5, j in 1:4]
    pts = [(-0.7, -1.6), (0.7, 2.9), (2.2, -1.6)]
    reference = (
        (
            (LinearInterpolationDimension(x), LinearInterpolationDimension(y)), u, ET.Linear,
            [-1.776903338918074, 1.1687962967613097, -7.329475232155476],
            [0.7300977149180659, 1.581801536821068, -3.837300251449927],
        ),
        (
            (ConstantInterpolationDimension(x), ConstantInterpolationDimension(y)), u, ET.Constant,
            [-0.4161468365471424, 1.0783954651036145, -2.893676954212239], zeros(3),
        ),
        (
            (BSplineInterpolationDimension(x, 2), BSplineInterpolationDimension(y, 2)), u_bspline, ET.Extension,
            [0.17589843696628493, 3.0019646126046506, -2.1437708073234982],
            [2.268234543424896, 1.2605521236270993, -0.58594816414654],
        ),
    )
    for (dims, U, mode, vals, d10) in reference
        @test all(dim -> dim.extrapolation == mode, dims)
        interp = NDInterpolation(U, dims)
        @test [interp(p...) for p in pts] ≈ vals rtol = 1.0e-12
        @test [interp(p...; derivative_orders = (1, 0)) for p in pts] ≈ d10 rtol = 1.0e-12
    end
end

@testset "Heterogeneous input tuples in 3D" begin
    # `make_out` failed to dispatch on Julia 1.10 for three or more dimensions when the
    # inputs have different types
    t = [0.0, 0.5, 1.0]
    interp = NDInterpolation(
        [i + 2j + 3k + i * j * k for i in 1:3, j in 1:3, k in 1:3],
        ntuple(_ -> LinearInterpolationDimension(t), 3)
    )
    @test interp(0.3, 1, 0.7f0) ≈ interp(0.3, 1.0, 0.7)
end

@testset "Unsigned grid below the grid" begin
    interp = NDInterpolation(
        [10.0, 20.0], LinearInterpolationDimension(UInt[1, 2]; extrapolation = ET.Constant)
    )
    @test interp(-1) == 10.0
end

@testset "ForwardDiff through extrapolation" begin
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]
    for mode in (ET.Linear, ET.Periodic, ET.Reflective), x in (-0.4, 1.63)
        interp = NDInterpolation(u, LinearInterpolationDimension(t; extrapolation = mode))
        @test ForwardDiff.derivative(interp, x) ≈ interp(x; derivative_orders = (1,))
    end
    interp = NDInterpolation(u, LinearInterpolationDimension(t; extrapolation = ET.None))
    @test_throws ExtrapolationError ForwardDiff.derivative(interp, 1.2)
end

@testset "Vector valued output" begin
    t = [0.0, 0.3, 0.7, 1.0]
    u = [sin(i) + cos(j) for i in 1:4, j in 1:4]
    for mode in (ET.Linear, ET.Reflective)
        dims = (
            LinearInterpolationDimension(t; extrapolation = mode, t_eval = [-0.45, 1.4]),
            LinearInterpolationDimension(t; t_eval = [0.2, 1.9]),
        )
        scalar = NDInterpolation(u, dims)
        vector = NDInterpolation(cat(u, 2u; dims = 3), dims)
        @test vector(-0.45, 1.9) ≈ [1, 2] .* scalar(-0.45, 1.9)
        @test vector(-0.45, 1.9; derivative_orders = (1, 0)) ≈
            [1, 2] .* scalar(-0.45, 1.9; derivative_orders = (1, 0))
        @test eval_grid(vector; derivative_orders = (1, 0))[:, :, 2] ≈
            2 .* eval_grid(scalar; derivative_orders = (1, 0))
    end
end

@testset "eval_unstructured outside the grid" begin
    t = [0.0, 0.3, 0.7, 1.0]
    u = [sin(i) + cos(j) for i in 1:4, j in 1:4]
    xs = [-0.45, 0.4, 2.15]
    ys = [1.6, -0.8, 0.1]
    for modes in ((ET.Linear, ET.Linear), (ET.Periodic, ET.Reflective))
        dims = (
            LinearInterpolationDimension(t; extrapolation = modes[1], t_eval = xs),
            LinearInterpolationDimension(t; extrapolation = modes[2], t_eval = ys),
        )
        interp = NDInterpolation(u, dims)
        for orders in ((0, 0), (1, 1))
            @test eval_unstructured(interp; derivative_orders = orders) ≈
                [interp(xs[i], ys[i]; derivative_orders = orders) for i in 1:3]
        end
    end
end

@testset "Periodic and reflective splines" begin
    t = [0.0, 0.5, 1.1]
    interp = NDInterpolation(
        [sin(i) + cos(j) + 0.1i * j for i in 1:5, j in 1:5],
        (
            BSplineInterpolationDimension(t, 3; extrapolation = ET.Periodic),
            BSplineInterpolationDimension(t, 3; extrapolation = ET.Reflective),
        )
    )
    x = (0.3, 0.2)
    @test interp(x[1] + 2.2, x[2]) ≈ interp(x...)
    @test interp(x[1] - 1.1, x[2]; derivative_orders = (1, 1)) ≈
        interp(x...; derivative_orders = (1, 1))
    # Mirrored copy along the reflective axis: odd derivatives change sign
    @test interp(x[1], 2.2 - x[2]; derivative_orders = (0, 1)) ≈
        -interp(x...; derivative_orders = (0, 1))
    @test interp(x[1], 2.2 - x[2]; derivative_orders = (0, 2)) ≈
        interp(x...; derivative_orders = (0, 2))
    # Far out queries keep their phase
    @test interp(1.0e16, x[2]) ≈ interp(mod(1.0e16, 1.1), x[2])
end

@testset "Allocation free" begin
    t = [0.0, 0.3, 0.7, 1.0]
    alloc(interp, x, orders) = @allocated interp(x; derivative_orders = orders)
    for mode in instances(ExtrapolationType.T)
        interp = NDInterpolation(
            [sin(i) + cos(j) for i in 1:4, j in 1:4],
            (LinearInterpolationDimension(t; extrapolation = mode), LinearInterpolationDimension(t))
        )
        x = mode == ET.None ? (0.5, -3.0) : (-1.3, 5.2)
        for orders in ((0, 0), (1, 1))
            alloc(interp, x, orders)
            @test alloc(interp, x, orders) == 0
        end
    end
end
