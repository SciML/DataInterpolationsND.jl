using JET
using DataInterpolationsND
using Test

@testset "JET static analysis" begin
    # Linear interpolation setup
    t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
    t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]
    t1_eval = t1[1:(end - 1)] + diff(t1) / 2
    t2_eval = t2[1:(end - 1)] + diff(t2) / 2
    u_linear = fill(2.0, 5, 5)

    itp_linear = NDInterpolation(
        u_linear,
        (
            LinearInterpolationDimension(t1; t_eval = t1_eval),
            LinearInterpolationDimension(t2; t_eval = t2_eval),
        )
    )

    # B-spline setup
    u_bspline = fill(2.0, 6, 7)
    itp_bspline = NDInterpolation(
        u_bspline,
        (
            BSplineInterpolationDimension(t1, 2; t_eval = t1_eval, max_derivative_order_eval = 1),
            BSplineInterpolationDimension(t2, 3; t_eval = t2_eval, max_derivative_order_eval = 1),
        )
    )

    # Constant interpolation setup
    itp_const = NDInterpolation(
        u_linear,
        (
            ConstantInterpolationDimension(t1; t_eval = t1_eval),
            ConstantInterpolationDimension(t2; t_eval = t2_eval),
        )
    )

    # NURBS setup
    t_nurbs = collect(0:(π / 2):(2π))
    t_eval_nurbs = collect(range(0, 2π, length = 100))
    multiplicities = [3, 2, 2, 2, 3]
    u_nurbs = Float64[1 0; 1 1; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1; 1 0]
    weights = ones(9)
    weights[2:2:end] ./= sqrt(2)
    itp_nurbs = NDInterpolation(
        u_nurbs,
        (
            BSplineInterpolationDimension(
                t_nurbs, 2;
                multiplicities = multiplicities, t_eval = t_eval_nurbs
            ),
        );
        cache = NURBSWeights(weights)
    )

    @testset "Linear interpolation" begin
        rep = JET.report_call(
            eval_unstructured, (typeof(itp_linear),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(
            eval_grid, (typeof(itp_linear),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(
            itp_linear, (Tuple{Float64, Float64},);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "B-spline interpolation" begin
        rep = JET.report_call(
            eval_unstructured, (typeof(itp_bspline),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(
            eval_grid, (typeof(itp_bspline),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "Constant interpolation" begin
        rep = JET.report_call(
            eval_unstructured, (typeof(itp_const),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(
            eval_grid, (typeof(itp_const),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "NURBS interpolation" begin
        rep = JET.report_call(
            eval_unstructured, (typeof(itp_nurbs),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(
            eval_grid, (typeof(itp_nurbs),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end
end

@testset "JET type optimization" begin
    # Linear interpolation setup
    t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
    t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]
    t1_eval = t1[1:(end - 1)] + diff(t1) / 2
    t2_eval = t2[1:(end - 1)] + diff(t2) / 2
    u_linear = fill(2.0, 5, 5)

    itp_linear = NDInterpolation(
        u_linear,
        (
            LinearInterpolationDimension(t1; t_eval = t1_eval),
            LinearInterpolationDimension(t2; t_eval = t2_eval),
        )
    )

    # Constant interpolation setup
    itp_const = NDInterpolation(
        u_linear,
        (
            ConstantInterpolationDimension(t1; t_eval = t1_eval),
            ConstantInterpolationDimension(t2; t_eval = t2_eval),
        )
    )

    @testset "Linear interpolation optimization" begin
        rep = JET.report_opt(
            eval_unstructured, (typeof(itp_linear),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(
            itp_linear, (Tuple{Float64, Float64},);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "Constant interpolation optimization" begin
        rep = JET.report_opt(
            eval_unstructured, (typeof(itp_const),);
            target_modules = (DataInterpolationsND,)
        )
        @test length(JET.get_reports(rep)) == 0
    end
end
