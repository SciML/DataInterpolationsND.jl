using DataInterpolationsND
using Test

# Interface tests to verify the package properly adheres to Julia's standard interfaces
# and SciML's array/number interfaces. Uses BigFloat to test numeric type genericity.

@testset "BigFloat Support" begin
    @testset "LinearInterpolationDimension" begin
        t = BigFloat[1.0, 2.0, 3.0, 4.0]
        t_eval = BigFloat[1.5, 2.5]

        itp_dim = LinearInterpolationDimension(t)
        @test eltype(itp_dim.t) == BigFloat

        itp_dim_eval = LinearInterpolationDimension(t; t_eval = t_eval)
        @test eltype(itp_dim_eval.t_eval) == BigFloat
    end

    @testset "ConstantInterpolationDimension" begin
        t = BigFloat[1.0, 2.0, 3.0, 4.0]
        t_eval = BigFloat[1.5, 2.5]

        itp_dim = ConstantInterpolationDimension(t)
        @test eltype(itp_dim.t) == BigFloat

        itp_dim_eval = ConstantInterpolationDimension(t; t_eval = t_eval)
        @test eltype(itp_dim_eval.t_eval) == BigFloat
    end

    @testset "BSplineInterpolationDimension" begin
        t = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        t_eval = BigFloat[1.5, 2.5, 3.5, 4.5]

        itp_dim = BSplineInterpolationDimension(t, 2)
        @test eltype(itp_dim.t) == BigFloat
        @test eltype(itp_dim.knots_all) == BigFloat

        itp_dim_eval = BSplineInterpolationDimension(
            t, 2; t_eval = t_eval, max_derivative_order_eval = 1
        )
        @test eltype(itp_dim_eval.t_eval) == BigFloat
        @test eltype(itp_dim_eval.basis_function_eval) <: BigFloat
    end

    @testset "NDInterpolation with Linear" begin
        t1 = BigFloat[1.0, 2.0, 3.0, 4.0]
        t2 = BigFloat[1.0, 2.0, 3.0, 4.0]
        u = rand(BigFloat, 4, 4)

        itp_dims = (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        )
        itp = NDInterpolation(u, itp_dims)

        # Single point evaluation
        result = itp((BigFloat(1.5), BigFloat(2.5)))
        @test result isa BigFloat

        # Evaluation with Float64 points (type promotion)
        result_promoted = itp((1.5, 2.5))
        @test result_promoted isa BigFloat
    end

    @testset "NDInterpolation with Constant" begin
        t1 = BigFloat[1.0, 2.0, 3.0, 4.0]
        t2 = BigFloat[1.0, 2.0, 3.0, 4.0]
        u = rand(BigFloat, 4, 4)

        itp_dims = (
            ConstantInterpolationDimension(t1),
            ConstantInterpolationDimension(t2),
        )
        itp = NDInterpolation(u, itp_dims)

        result = itp((BigFloat(1.5), BigFloat(2.5)))
        @test result isa BigFloat
    end

    @testset "NDInterpolation with BSpline" begin
        t = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        itp_dim = BSplineInterpolationDimension(t, 2)
        n_basis = DataInterpolationsND.get_n_basis_functions(itp_dim)
        u = rand(BigFloat, n_basis, n_basis)

        itp_dims = (
            BSplineInterpolationDimension(t, 2),
            BSplineInterpolationDimension(t, 2),
        )
        itp = NDInterpolation(u, itp_dims)

        result = itp((BigFloat(2.5), BigFloat(3.5)))
        @test result isa BigFloat
    end

    @testset "NURBS with BigFloat" begin
        t = BigFloat[0.0, 1.0, 2.0, 3.0]
        itp_dim = BSplineInterpolationDimension(t, 1)
        n_basis = DataInterpolationsND.get_n_basis_functions(itp_dim)
        u = rand(BigFloat, n_basis)
        weights = ones(BigFloat, n_basis)
        cache = NURBSWeights(weights)

        itp = NDInterpolation(u, itp_dim; cache)
        result = itp((BigFloat(0.5),))
        @test result isa BigFloat
    end

    @testset "Multi-point evaluation with BigFloat" begin
        t1 = BigFloat[1.0, 2.0, 3.0, 4.0]
        t2 = BigFloat[1.0, 2.0, 3.0, 4.0]
        t1_eval = BigFloat[1.5, 2.5]
        t2_eval = BigFloat[1.5, 2.5]
        u = rand(BigFloat, 4, 4)

        itp_dims = (
            LinearInterpolationDimension(t1; t_eval = t1_eval),
            LinearInterpolationDimension(t2; t_eval = t2_eval),
        )
        itp = NDInterpolation(u, itp_dims)

        result = eval_grid(itp)
        @test eltype(result) == BigFloat
        @test size(result) == (2, 2)
    end

    @testset "Multi-output with BigFloat" begin
        t1 = BigFloat[1.0, 2.0, 3.0, 4.0]
        t2 = BigFloat[1.0, 2.0, 3.0, 4.0]
        u = rand(BigFloat, 4, 4, 3)  # 3D output

        itp_dims = (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        )
        itp = NDInterpolation(u, itp_dims)

        result = itp((BigFloat(1.5), BigFloat(2.5)))
        @test eltype(result) == BigFloat
        @test size(result) == (3,)
    end
end

@testset "Float32 Support" begin
    @testset "NDInterpolation with Linear" begin
        t1 = Float32[1.0, 2.0, 3.0, 4.0]
        t2 = Float32[1.0, 2.0, 3.0, 4.0]
        u = rand(Float32, 4, 4)

        itp_dims = (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        )
        itp = NDInterpolation(u, itp_dims)

        # Single point evaluation with Float32
        result = itp((Float32(1.5), Float32(2.5)))
        @test result isa Float32

        # Evaluation with Float64 points (type promotion)
        result_promoted = itp((1.5, 2.5))
        @test result_promoted isa Float64
    end

    @testset "Multi-point evaluation with Float32" begin
        t1 = Float32[1.0, 2.0, 3.0, 4.0]
        t2 = Float32[1.0, 2.0, 3.0, 4.0]
        t1_eval = Float32[1.5, 2.5]
        t2_eval = Float32[1.5, 2.5]
        u = rand(Float32, 4, 4)

        itp_dims = (
            LinearInterpolationDimension(t1; t_eval = t1_eval),
            LinearInterpolationDimension(t2; t_eval = t2_eval),
        )
        itp = NDInterpolation(u, itp_dims)

        result = eval_grid(itp)
        @test eltype(result) == Float32
    end
end

@testset "Type Promotion" begin
    @testset "Mixed precision t and u" begin
        t1 = Float32[1.0, 2.0, 3.0, 4.0]
        t2 = Float32[1.0, 2.0, 3.0, 4.0]
        u = rand(Float64, 4, 4)  # Float64 u with Float32 t

        itp_dims = (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        )
        itp = NDInterpolation(u, itp_dims)

        result = itp((Float32(1.5), Float32(2.5)))
        @test result isa Float64  # Promoted to Float64 due to u
    end
end

@testset "eval_unstructured Multi-Dimensional BSpline" begin
    # Test eval_unstructured for multi-dimensional BSpline interpolation
    # This is a regression test for proper multi_point_index handling
    @testset "BSpline 2D eval_unstructured" begin
        t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
        t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]
        t1_eval = t1[1:(end - 1)] + diff(t1) / 2
        t2_eval = t2[1:(end - 1)] + diff(t2) / 2

        u_bspline = fill(2.0, 6, 7)
        itp_bspline = NDInterpolation(
            u_bspline,
            (BSplineInterpolationDimension(t1, 2; t_eval = t1_eval,
                    max_derivative_order_eval = 1),
                BSplineInterpolationDimension(t2, 3; t_eval = t2_eval,
                    max_derivative_order_eval = 1))
        )

        result = eval_unstructured(itp_bspline)
        @test size(result) == (4,)
        # For constant data, result should be approximately constant
        @test all(x -> isapprox(x, 2.0; atol = 1e-10), result)

        # Test with derivatives
        result_deriv = eval_unstructured(itp_bspline; derivative_orders = (1, 0))
        @test size(result_deriv) == (4,)
    end

    @testset "BSpline eval_unstructured with BigFloat" begin
        t = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        t_eval = BigFloat[1.5, 2.5, 3.5, 4.5]
        itp_dim = BSplineInterpolationDimension(t, 2; t_eval = t_eval)
        n_basis = DataInterpolationsND.get_n_basis_functions(itp_dim)
        u = fill(BigFloat(3.0), n_basis, n_basis)

        itp_dims = (
            BSplineInterpolationDimension(t, 2; t_eval = t_eval),
            BSplineInterpolationDimension(t, 2; t_eval = t_eval)
        )
        itp = NDInterpolation(u, itp_dims)

        result = eval_unstructured(itp)
        @test eltype(result) == BigFloat
        @test size(result) == (4,)
        @test all(x -> isapprox(x, BigFloat(3.0); atol = BigFloat(1e-10)), result)
    end

    @testset "NURBS eval_unstructured" begin
        t_nurbs = collect(0:(π / 2):(2π))
        t_eval_nurbs = collect(range(0, 2π, length = 100))
        multiplicities = [3, 2, 2, 2, 3]
        u_nurbs = Float64[1 0; 1 1; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1; 1 0]
        weights = ones(9)
        weights[2:2:end] ./= sqrt(2)
        itp_nurbs = NDInterpolation(u_nurbs,
            (BSplineInterpolationDimension(t_nurbs, 2;
                multiplicities = multiplicities, t_eval = t_eval_nurbs),);
            cache = NURBSWeights(weights))

        result = eval_unstructured(itp_nurbs)
        @test size(result) == (100, 2)
        # Points should be on unit circle
        @test all(row -> isapprox(row[1]^2 + row[2]^2, 1.0; atol = 1e-10), eachrow(result))
    end
end
