
@testitem "Linear Interpolation" begin
    include("utils.jl")
    test_globally_constant(LinearInterpolationDimension)

    f(t1, t2) = 3.0 + 2.3t1 - 4.7t2

    Random.seed!(1)
    t1 = cumsum(rand(10))
    t2 = cumsum(rand(10))

    itp_dims = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2),
    )
    u = f.(t1, t2')
    itp = NDInterpolation(u, itp_dims)
    test_analytic(itp, f)
end

@testitem "BSpline Interpolation" begin
    include("utils.jl")
    test_globally_constant(
        BSplineInterpolationDimension, args1=(2,), args2=(3,),
        kwargs1=(:max_derivative_order_eval => 1,),
        kwargs2=(:max_derivative_order_eval => 1,)
    )

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

@testitem "NURBS Interpolation" begin
    include("utils.jl")

    test_globally_constant(
        BSplineInterpolationDimension; args1=(3,), args2=(1,),
        kwargs1=(:max_derivative_order_eval => 1,),
        kwargs2=(:max_derivative_order_eval => 1,),
        cache=NURBSWeights(rand(7, 5)),
        test_derivatives=false
    )

    ## Circle representation
    # Knots
    t = collect(0:(π/2):(2π))

    t_eval = collect(range(0, 2π, length=100))

    # Multiplicities
    multiplicities = [3, 2, 2, 2, 3]

    # Control points
    u = Float64[
        1 0;
        1 1;
        0 1;
        -1 1;
        -1 0;
        -1 -1;
        0 -1;
        1 -1;
        1 0
    ]

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
