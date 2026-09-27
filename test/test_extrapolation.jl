using DataInterpolationsND
using ForwardDiff
using Adapt
using Test

@testset "Per-dimension extrapolation" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 2.0, 4.0]
    xs = [-1.0, 0.5, 3.0]
    ys = [-2.0, 1.0, 6.0]
    f(x, y) = 2 + 3x + 5y + 7x * y
    values = f.(x, y')
    constructors = (
        LinearInterpolationDimension,
        (t; kw...) -> BSplineInterpolationDimension(t, 1; max_derivative_order_eval = 2, kw...),
    )
    for constructor in constructors,
            xmode in (ExtrapolationType.Linear, ExtrapolationType.Constant),
            ymode in (ExtrapolationType.Linear, ExtrapolationType.Constant)
        dims = (
            constructor(x; t_eval = xs, extrapolation = xmode),
            constructor(y; t_eval = ys, extrapolation = ymode),
        )
        interp = NDInterpolation(values, dims)
        vector_interp = NDInterpolation(cat(values, 2values; dims = 3), dims)
        for orders in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
            expected = map(Iterators.product(xs, ys)) do (a, b)
                hold_x = xmode == ExtrapolationType.Constant && !(first(x) <= a <= last(x))
                hold_y = ymode == ExtrapolationType.Constant && !(first(y) <= b <= last(y))
                a = hold_x ? clamp(a, first(x), last(x)) : a
                b = hold_y ? clamp(b, first(y), last(y)) : b
                if (hold_x && orders[1] > 0) || (hold_y && orders[2] > 0) || any(>(1), orders)
                    0.0
                elseif orders == (1, 1)
                    7.0
                elseif orders == (1, 0)
                    3 + 7b
                elseif orders == (0, 1)
                    5 + 7a
                else
                    f(a, b)
                end
            end
            @test eval_grid(interp; derivative_orders = orders) ≈ expected
            @test eval_unstructured(interp; derivative_orders = orders) ≈ [expected[i, i] for i in eachindex(xs)]
            for (i, a) in pairs(xs), (j, b) in pairs(ys)
                @test interp(a, b; derivative_orders = orders) ≈ expected[i, j]
                out = fill(NaN, 2)
                vector_interp(out, (a, b); derivative_orders = orders)
                @test out ≈ [expected[i, j], 2expected[i, j]]
            end
        end
        for a in xs, b in ys
            @test ForwardDiff.derivative(z -> interp(z, b), a) ≈ interp(a, b; derivative_orders = (1, 0))
            @test ForwardDiff.derivative(z -> interp(a, z), b) ≈ interp(a, b; derivative_orders = (0, 1))
        end
        adapted = Adapt.adapt(Array, interp)
        @test adapted(-1.0, 6.0) == interp(-1.0, 6.0)
        @test adapted.interp_dims[1].extrapolation_left == xmode
        @test adapted.interp_dims[2].extrapolation_right == ymode
    end
end

@testset "Independent sides and boundary derivatives" begin
    for (left, right, expected) in (
            (ExtrapolationType.Constant, ExtrapolationType.Linear, (0.0, 3.0)),
            (ExtrapolationType.Linear, ExtrapolationType.Constant, (-1.0, 2.0)),
        )
        dim = LinearInterpolationDimension([0.0, 1.0, 2.0]; extrapolation_left = left, extrapolation_right = right)
        interp = NDInterpolation([0.0, 1.0, 2.0], dim)
        @test (interp(-1.0), interp(3.0)) == expected
        for boundary in (0.0, 2.0)
            @test interp(boundary; derivative_orders = (1,)) == 1.0
            @test ForwardDiff.derivative(interp, boundary) == 1.0
        end
    end
    dim = LinearInterpolationDimension([0.0, 1.0]; extrapolation = ExtrapolationType.Constant, extrapolation_right = ExtrapolationType.Linear)
    @test NDInterpolation([0.0, 1.0], dim)(2.0) == 1.0
    @test NDInterpolation([0.0, 1.0], LinearInterpolationDimension([0.0, 1.0]))(2.0) == 2.0
    for constructor in (LinearInterpolationDimension, (t; kw...) -> BSplineInterpolationDimension(t, 1; kw...))
        dim = constructor([0.5, 1.5]; extrapolation = ExtrapolationType.Constant)
        interp = NDInterpolation([2.0, 4.0], dim)
        @test interp(0) == 2.0
        @test interp(2) == 4.0
    end
end

@testset "Quadratic spline extrapolation" begin
    for mode in (ExtrapolationType.Constant, ExtrapolationType.Linear, ExtrapolationType.Extension)
        dim = BSplineInterpolationDimension([0.0, 1.0], 2; t_eval = [-1.0, 2.0], max_derivative_order_eval = 2, extrapolation = mode)
        interp = NDInterpolation([0.0, 0.0, 1.0], dim)
        expected = if mode == ExtrapolationType.Constant
            ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        elseif mode == ExtrapolationType.Linear
            ([0.0, 3.0], [0.0, 2.0], [0.0, 0.0])
        else
            ([1.0, 4.0], [-2.0, 4.0], [2.0, 2.0])
        end
        for order in 0:2
            @test eval_grid(interp; derivative_orders = (order,)) ≈ expected[order + 1]
            @test [interp(t; derivative_orders = (order,)) for t in dim.t_eval] ≈ expected[order + 1]
        end
    end
    interp = NDInterpolation([0.0, 0.0, 1.0], BSplineInterpolationDimension([0.0, 1.0], 2))
    @test interp(2.0) == 4.0
end

@testset "Constant dimensions and NURBS" begin
    for mode in (ExtrapolationType.Constant, ExtrapolationType.Linear, ExtrapolationType.Extension)
        dim = ConstantInterpolationDimension([0.0, 1.0, 2.0]; extrapolation = mode)
        interp = NDInterpolation([2.0, 4.0, 7.0], dim)
        @test interp(-1.0) == 2.0
        @test interp(3.0) == 7.0
        for t in (-1.0, 3.0)
            @test interp(t; derivative_orders = (1,)) == 0.0
        end
        two_dims = NDInterpolation([2.0 3.0; 4.0 5.0; 7.0 8.0], (dim, ConstantInterpolationDimension([0.0, 1.0]; extrapolation = mode)))
        @test two_dims(-1.0, 0.0; derivative_orders = (1, 0)) == 0.0
        @test two_dims(2.0, 2.0; derivative_orders = (0, 1)) == 0.0
        vector_interp = NDInterpolation(reshape([2.0, 4.0, 7.0], 3, 1), dim)
        out = [NaN]
        vector_interp(out, (-1.0,); derivative_orders = (1,))
        @test out == [0.0]
    end
    dim = BSplineInterpolationDimension([0.0, 1.0], 2; t_eval = [-1.0, 2.0], extrapolation = ExtrapolationType.Constant)
    interp = NDInterpolation([2.0, 4.0, 7.0], dim; cache = NURBSWeights([1.0, 2.0, 1.0]))
    @test eval_grid(interp) == [2.0, 7.0]
    @test [interp(-1.0), interp(2.0)] == [2.0, 7.0]
    @test ForwardDiff.derivative(interp, -1.0) == 0.0
    @test ForwardDiff.derivative(interp, 2.0) == 0.0
    linear_dim = BSplineInterpolationDimension([0.0, 1.0], 2; extrapolation = ExtrapolationType.Linear)
    @test_throws ArgumentError NDInterpolation([2.0, 4.0, 7.0], linear_dim; cache = NURBSWeights([1.0, 2.0, 1.0]))
end

@testset "Hold-edge reproducer" begin
    soc = [0.0, 0.5, 1.0]
    temperatures = [0.0, 25.0]
    vals = [3.0 3.1; 3.5 3.6; 3.9 4.0]
    dims = (LinearInterpolationDimension(soc; extrapolation = ExtrapolationType.Constant), LinearInterpolationDimension(temperatures; extrapolation = ExtrapolationType.Constant))
    nd = NDInterpolation(vals, dims)
    @test nd(1.2, 25.0) ≈ 4.0
    @test nd(-0.2, 0.0) ≈ 3.0
    @test nd(0.5, 40.0) ≈ 3.6
end
