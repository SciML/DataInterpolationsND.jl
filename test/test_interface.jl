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
