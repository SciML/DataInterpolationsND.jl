@recipe function f(itp_dim::BSplineInterpolationDimension; derivative_order = 0)
    n_basis_functions = get_n_basis_functions(itp_dim)

    # Plot each basis function
    for i in 1:n_basis_functions
        bfv = BasisFunctionVector(itp_dim, i, derivative_order)
        @series begin
            seriestype := :line
            label := "Basis function $i"
            itp_dim.t_eval, bfv
        end
    end

    # Plot the knots
    @series begin
        seriestype := :scatter
        label := "Knots"
        itp_dim.t, zero(itp_dim.t)
    end
end
