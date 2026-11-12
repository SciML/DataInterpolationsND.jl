using DataInterpolationsND
using ForwardDiff
using Random

function test_deriv_10(itp, t)
    return @test itp(t; derivative_orders = (1, 0)) ≈
        ForwardDiff.derivative(t₁ -> itp(t₁, t[2]), t[1])
end

function test_deriv_01(itp, t)
    return @test itp(t; derivative_orders = (0, 1)) ≈
        ForwardDiff.derivative(t₂ -> itp(t[1], t₂), t[2])
end

function test_deriv_11(itp, t)
    return @test itp(t; derivative_orders = (1, 1)) ≈
        ForwardDiff.derivative(
        t₂ -> ForwardDiff.derivative(
            t₁ -> itp(t₁, t₂), t[1]
        ),
        t[2]
    )
end

function test_derivatives(itp::NDInterpolation{2})
    t1 = itp.interp_dims[1].t
    t2 = itp.interp_dims[2].t

    # Derivatives in data points
    for t in Iterators.product(t1, t2)
        test_deriv_10(itp, t)
        test_deriv_01(itp, t)
        test_deriv_11(itp, t)
    end

    # Derivatives between data points
    for t in Iterators.product(
            t1[1:(end - 1)] + diff(t1) / 2, t2[1:(end - 1)] + diff(t2) / 2
        )
        test_deriv_10(itp, t)
        test_deriv_01(itp, t)
        test_deriv_11(itp, t)
    end
    return
end

@testset "Linear Interpolation" begin
    Random.seed!(1)
    t1 = [1.0, 2.0, 3.5, 4.0, 10.0]
    t2 = [-1.5, 0.0, 2.5, 7.8, 12.9]
    u = rand(5, 5)
    itp_dims = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2),
    )
    itp = NDInterpolation(u, itp_dims)
    test_derivatives(itp)

    for t in Iterators.product(t1, t2)
        idx = DataInterpolationsND.get_idx(itp.interp_dims, t)
        @test idx ==
            DataInterpolationsND.get_idx(itp.interp_dims, (ForwardDiff.Dual(t[1], 1.0), t[2]))
        @test idx ==
            DataInterpolationsND.get_idx(itp.interp_dims, (t[1], ForwardDiff.Dual(t[2], 1.0)))
        @test idx == DataInterpolationsND.get_idx(
            itp.interp_dims, (ForwardDiff.Dual(t[1], -1.0), t[2])
        )
    end
end

@testset "BSpline interpolation" begin
    Random.seed!(1)
    t1 = [1.0, 2.0, 3.5, 4.0, 10.0]
    t2 = [-1.5, 0.0, 2.5, 7.8, 12.9]
    u = rand(6, 6)
    itp_dims = (
        BSplineInterpolationDimension(t1, 2),
        BSplineInterpolationDimension(t2, 2),
    )
    itp = NDInterpolation(u, itp_dims)
    test_derivatives(itp)

    u = rand(9, 9)
    itp_dims = (
        BSplineInterpolationDimension(t1, 5),
        BSplineInterpolationDimension(t2, 5),
    )
    itp = NDInterpolation(u, itp_dims)
    test_derivatives(itp)
end

@testset "BSpline higher order and mixed derivatives" begin
    # Repeated interior knots in the first dimension, extrapolation in the second
    t1 = [0.0, 0.7, 1.1, 2.0, 2.4, 3.5]
    multiplicities = [4, 1, 2, 3, 1, 4]
    t2 = [-1.0, 0.0, 0.3, 1.5, 2.0]
    t1_eval = [0.35, 0.9, 1.3, 2.2, 3.1]
    t2_eval = [-1.4, -0.6, 0.1, 1.7, 2.3]
    Random.seed!(2)
    u = rand(sum(multiplicities) - 4, length(t2) + 1, 2)
    nested_derivative(f, k) = iszero(k) ? f : nested_derivative(x -> ForwardDiff.derivative(f, x), k - 1)

    for extrapolation in (ExtrapolationType.Extension, ExtrapolationType.Linear)
        dims = (
            BSplineInterpolationDimension(
                t1, 3; multiplicities, t_eval = t1_eval, max_derivative_order_eval = 4
            ),
            BSplineInterpolationDimension(
                t2, 2; t_eval = t2_eval, max_derivative_order_eval = 2, extrapolation
            ),
        )
        itp = NDInterpolation(u[:, :, 1], dims)
        itp_vec = NDInterpolation(u, dims)
        for orders in ((3, 0), (4, 0), (0, 2), (1, 1), (2, 1))
            vals = [itp(a, b; derivative_orders = orders) for a in t1_eval, b in t2_eval]
            refs = [
                nested_derivative(s -> nested_derivative(r -> itp(r, s), orders[1])(a), orders[2])(b)
                    for a in t1_eval, b in t2_eval
            ]
            @test vals ≈ refs rtol = 1.0e-10
            @test eval_grid(itp; derivative_orders = orders) ≈ vals
            @test eval_unstructured(itp; derivative_orders = orders) ≈ [vals[i, i] for i in 1:5]
            @test eval_grid(itp_vec; derivative_orders = orders)[:, :, 1] ≈ vals
        end
    end

    # The basis function vectors used for plotting give the basis function derivatives
    dim = BSplineInterpolationDimension(
        t1, 3; multiplicities, t_eval = t1_eval, max_derivative_order_eval = 2
    )
    for i in 1:size(u, 1)
        itp = NDInterpolation([j == i ? 1.0 : 0.0 for j in 1:size(u, 1)], dim)
        @test DataInterpolationsND.BasisFunctionVector(dim, i, 2) ≈
            [itp(x; derivative_orders = (2,)) for x in t1_eval] atol = 1.0e-12
    end
end

@testset "BSpline derivative accuracy" begin
    # Derivatives are evaluated by differencing the control points (de Boor), so the rounding
    # error is relative to the size of the derivative rather than to |u| / hᵏ, which would be
    # hundreds to thousands of times eps * scale here because of the offset in the data.
    function max_scaled_error(derivative_orders::NTuple{N}; n = 12) where {N}
        ts = ntuple(d -> [i + 0.3 * sin(2.7 * i + d) for i in 0.0:(n - 1)] ./ n, N)
        u = [100 + prod(d -> sin(2 + 3d * I[d] / (n + 2)), 1:N) for I in CartesianIndices(ntuple(_ -> n + 2, N))]
        itp = NDInterpolation(u, map(t -> BSplineInterpolationDimension(t, 3), ts))
        itp_big = NDInterpolation(big.(u), map(t -> BSplineInterpolationDimension(big.(t), 3), ts))
        xs = Iterators.product(ntuple(d -> range(ts[d][1], ts[d][end]; length = 7)[2:(end - 1)] .+ 0.01, N)...)
        errs = [itp(x; derivative_orders) - itp_big(big.(x); derivative_orders) for x in xs]
        scale = maximum(x -> abs(itp_big(big.(x); derivative_orders)), xs)
        return Float64(maximum(abs, errs) / (eps() * scale))
    end

    for derivative_orders in ((2,), (3,), (1, 1), (0, 3), (1, 1, 1))
        @test max_scaled_error(derivative_orders) < 50
    end
end

@testset "BSpline derivatives of integer and Float32 control points" begin
    # The control points are differenced in the output type, so they must not wrap around or overflow
    t = [0.0, 0.3, 0.5, 0.8, 1.0]
    dims = (BSplineInterpolationDimension(t, 3), BSplineInterpolationDimension(t, 2))
    u_uint8 = UInt8[(i + j) % 2 == 0 ? 255 : 3 * i + j for i in 1:7, j in 1:6]
    u_float32 = Float32[(-1)^(i + j) * 3.0f38 for i in 1:7, j in 1:6]
    for u in (u_uint8, u_float32)
        itp = NDInterpolation(u, dims)
        itp_ref = NDInterpolation(Float64.(u), dims)
        for orders in ((1, 0), (2, 1), (3, 2))
            @test itp(0.4, 0.6; derivative_orders = orders) ≈
                itp_ref(0.4, 0.6; derivative_orders = orders)
        end
    end
end
