using DataInterpolationsND
using Random
using DataInterpolationsND: AbstractInterpolationDimension, EmptyCache, 
    keepat, removeat, insertat, remove, keep

@testset "tuple primitives" begin
    # These help with handling NoInterpolationDimension and its consequences
    @test insertat(Nothing, 1, (2,), (nothing, 4, nothing)) === (1, 2, 1)
    @test insertat(Float64, missing, (1,), (8, 9.0)) === (1, missing)
    @test keepat(Nothing, (1, 2, 3), (nothing, 4, nothing)) === (1, 3)
    @test keepat(Float64, (1, 2,), (8, 9.0)) === (2,)
    @test removeat(Nothing, (1, 2, 3), (nothing, 4, nothing)) === (2,)
    @test removeat(Float64, (1, 2,), (8, 9.0)) === (1,)
    @test remove(Nothing, (1, nothing, 3)) === (1, 3)
    @test keep(Int, (1, nothing, 3)) === (1, 3)
end

function test_globally_constant(ID::Type{<:AbstractInterpolationDimension}; 
    args1 = (), 
    args2 = (), 
    kwargs1 = (),
    kwargs2 = (), 
    cache = EmptyCache(), 
    test_derivatives = true
)
    t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
    t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]

    u = if ID == BSplineInterpolationDimension
        fill(2.0, 4 + args1[1], 4 + args2[1])
    else
        fill(2.0, 5, 5)
    end

    # Evaluation in data points
    itp_dims = (
        ID(t1, args1...; t_eval = t1, kwargs1...),
        ID(t2, args2...; t_eval = t2, kwargs2...)
    )

    itp = NDInterpolation(u, itp_dims; cache)

    grid = eval_grid(itp)
    @test all(x -> isapprox(x, 2.0; atol = 1e-10), grid)
    if test_derivatives
        @test all(
            x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (1, 0)))
        @test all(
            x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (0, 1)))
    end

    # Evaluation between data points
    itp_dims = (
        ID(t1, args1...; t_eval = t1[1:(end - 1)] + diff(t1) / 2, kwargs1...),
        ID(t2, args2...; t_eval = t2[1:(end - 1)] + diff(t2) / 2, kwargs2...)
    )
    itp = NDInterpolation(u, itp_dims; cache)
    @test all(x -> isapprox(x, 2.0; atol = 1e-10), eval_grid(itp))
    if test_derivatives
        @test all(
            x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (1, 0)))
        @test all(
            x -> isapprox(x, 0.0; atol = 1e-10), eval_grid(itp, derivative_orders = (0, 1)))
    end
end

function test_analytic(itp::NDInterpolation{<:Any,N_in}, f) where {N_in}
    used_interp_dims = remove(NoInterpolationDimension, itp.interp_dims)
    # Evaluation in data points
    ts = map(d -> d.t, used_interp_dims)
    for t in Iterators.product(ts...)
        @test itp(t) ≈ f(t...)
    end

    # Evaluation between data points
    ts_ = ntuple(dim_in -> ts[dim_in][1:(end - 1)] + diff(ts[dim_in]) / 2, N_in)
    t = first(Iterators.product(ts_...))
    for t in Iterators.product(ts_...)
        @test itp(t) ≈ f(t...)
    end
end

@testset "Constant Interpolation" begin
    test_globally_constant(ConstantInterpolationDimension)
end

@testset "Linear Interpolation" begin
    test_globally_constant(LinearInterpolationDimension)

    f(t1, t2) = 3.0 + 2.3t1 - 4.7t2
    Random.seed!(1)
    t1 = cumsum(rand(10))
    t2 = cumsum(rand(10))

    itp_dims = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2)
    )
    u = f.(t1, t2')
    itp = NDInterpolation(u, itp_dims)
    test_analytic(itp, f)


    itp_dims = (
        NoInterpolationDimension(),
        LinearInterpolationDimension(t2; t_eval = t2 ./ 2)
    )
    itp = NDInterpolation(u, itp_dims)
    eval_grid(itp)
end

@testset "BSpline Interpolation" begin
    ID = BSplineInterpolationDimension
    args1 = (2,)
    args2 = (3,)
    kwargs1 = (:max_derivative_order_eval => 1,)
    kwargs2 = (:max_derivative_order_eval => 1,)
    cache = EmptyCache() 
    test_derivatives = true
    test_globally_constant(ID; args1, args2, kwargs1, kwargs2, cache, test_derivatives)

    f(t1, t2, t3) = t1^2 + t2^2 + t3^2

    u = zeros(3, 3, 3)
    u[2, 2, 2] = -3
    for I in Iterators.product((1, 3), (1, 3), (1, 3))
        u[I...] = 3
    end

    itp_dim = BSplineInterpolationDimension([-1.0, 1.0], 2)
    itp = NDInterpolation(u, (itp_dim, itp_dim, itp_dim))
    test_analytic(itp, f)
end

@testset "NURBS Interpolation" begin
    Random.seed!(10)

    ID = BSplineInterpolationDimension
    args1 = (3,)
    args2 = (1,)
    kwargs1 = (:max_derivative_order_eval => 1,)
    kwargs2 = (:max_derivative_order_eval => 1,)
    cache = NURBSWeights(rand(7, 5))
    test_derivatives = false
    test_globally_constant(ID; args1, args2, kwargs1, kwargs2, cache, test_derivatives)

    ## Circle representation
    # Knots
    t = collect(0:(π / 2):(2π))

    t_eval = collect(range(0, 2π, length = 100))

    # Multiplicities
    multiplicities = [3, 2, 2, 2, 3]

    # Control points
    u = Float64[1 0;
                1 1;
                0 1;
                -1 1;
                -1 0;
                -1 -1;
                0 -1;
                1 -1;
                1 0]

    # Weights
    weights = ones(9)
    weights[2:2:end] ./= sqrt(2)
    cache = NURBSWeights(weights)

    itp_dim = BSplineInterpolationDimension(t, 2; multiplicities, t_eval)
    itp = NDInterpolation(u, itp_dim; cache)

    out = eval_grid(itp)
    points_on_circle = eachrow(out)
    @test allunique(points_on_circle[2:end])
    @test all(point -> point[1]^2 + point[2]^2 ≈ 1, points_on_circle)
end

@testset "Mixed Interpolation" begin
    t1 = cumsum(rand(3))
    t2 = cumsum(rand(4))
    t3 = collect(0:(π / 2):(2π))
    u = zeros(3, 4, 5)
    itp_dims = (
        NoInterpolationDimension(),
        LinearInterpolationDimension(t2; t_eval=t2),
        BSplineInterpolationDimension(t3, 1; t_eval=t3),
    )
    itp = NDInterpolation(u, itp_dims)
    eval_grid(itp)
end
