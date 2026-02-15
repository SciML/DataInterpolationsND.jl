using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        # Linear interpolation dimensions
        t1 = collect(range(0.0, 1.0, 5))
        t2 = collect(range(0.0, 1.0, 4))
        u_2d = rand(5, 4)

        dim1 = LinearInterpolationDimension(t1)
        dim2 = LinearInterpolationDimension(t2)
        interp_linear = NDInterpolation(u_2d, (dim1, dim2))
        interp_linear(0.5, 0.5)

        # Linear derivatives
        interp_linear(0.5, 0.5; derivative_orders = (1, 0))
        interp_linear(0.5, 0.5; derivative_orders = (0, 1))
        interp_linear(0.5, 0.5; derivative_orders = (1, 1))

        # Constant interpolation dimensions
        cdim1 = ConstantInterpolationDimension(t1)
        cdim2 = ConstantInterpolationDimension(t2)
        interp_const = NDInterpolation(u_2d, (cdim1, cdim2))
        interp_const(0.5, 0.5)

        # BSpline interpolation dimensions (degree 2)
        t_bs = collect(range(0.0, 1.0, 4))
        u_bs = rand(5, 5)  # n_basis = n_knots + degree - 1 = 4 + 2 - 1 = 5
        bdim1 = BSplineInterpolationDimension(t_bs, 2)
        bdim2 = BSplineInterpolationDimension(t_bs, 2)
        interp_bspline = NDInterpolation(u_bs, (bdim1, bdim2))
        interp_bspline(0.5, 0.5)

        # BSpline derivatives
        interp_bspline(0.5, 0.5; derivative_orders = (1, 0))
        interp_bspline(0.5, 0.5; derivative_orders = (0, 1))

        # Multi-point evaluation (unstructured)
        t_eval = collect(range(0.1, 0.9, 3))
        dim1_mp = LinearInterpolationDimension(t1; t_eval = t_eval)
        dim2_mp = LinearInterpolationDimension(t2; t_eval = t_eval)
        interp_mp = NDInterpolation(u_2d, (dim1_mp, dim2_mp))
        eval_unstructured(interp_mp)

        # Multi-point evaluation (grid)
        eval_grid(interp_mp)

        # 1D interpolations (common use case)
        t_1d = collect(range(0.0, 1.0, 6))
        u_1d = rand(6)
        dim_1d = LinearInterpolationDimension(t_1d)
        interp_1d = NDInterpolation(u_1d, (dim_1d,))
        interp_1d(0.5)
        interp_1d(0.5; derivative_orders = (1,))

        # 1D constant
        cdim_1d = ConstantInterpolationDimension(t_1d)
        interp_1d_const = NDInterpolation(u_1d, (cdim_1d,))
        interp_1d_const(0.5)

        # 1D BSpline
        t_bs_1d = collect(range(0.0, 1.0, 4))
        u_bs_1d = rand(5)
        bdim_1d = BSplineInterpolationDimension(t_bs_1d, 2)
        interp_1d_bs = NDInterpolation(u_bs_1d, (bdim_1d,))
        interp_1d_bs(0.5)
        interp_1d_bs(0.5; derivative_orders = (1,))
    end
end
