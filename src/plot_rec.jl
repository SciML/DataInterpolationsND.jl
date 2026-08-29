
# BSpline basis visualization


@recipe function f(
    itp_dim::BSplineInterpolationDimension;
    derivative_order = 0,
)
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

    # Plot knots
    @series begin
        seriestype := :scatter
        label := "Knots"
        itp_dim.t, zero(itp_dim.t)
    end
end



# 1D NDInterpolation


@recipe function f(interp::NDInterpolation{1})
    dim = interp.interp_dims[1]
    t = dim.t_eval

    y = eval_unstructured(interp)

    if ndims(y) == 1
        @series begin
            seriestype := :line
            t, y
        end
    else
        for k in axes(y, 2)
            @series begin
                seriestype := :line
                label := "Output $k"
                t, y[:, k]
            end
        end
    end
end



# 2D NDInterpolation


@recipe function f(interp::NDInterpolation{2})
    dim1, dim2 = interp.interp_dims
    x = dim1.t_eval
    y = dim2.t_eval

    out = eval_grid(interp)

    if ndims(out) == 2
        @series begin
            seriestype := :surface
            x, y, out
        end
    else
        for k in axes(out, 3)
            @series begin
                seriestype := :surface
                label := "Output $k"
                x, y, out[:, :, k]
            end
        end
    end
end


# 3D NDInterpolation (SLICE VIEW)


@recipe function f(interp::NDInterpolation{3})
    dim1, dim2, dim3 = interp.interp_dims
    x = dim1.t_eval
    y = dim2.t_eval
    z = dim3.t_eval

    out = eval_grid(interp)

    # Middle slice in 3rd dimension
    k = length(z) ÷ 2

    if ndims(out) == 3
        @series begin
            seriestype := :surface
            title := "Slice at z = $(z[k])"
            x, y, out[:, :, k]
        end
    else
        for c in axes(out, 4)
            @series begin
                seriestype := :surface
                label := "Output $c (z=$(z[k]))"
                x, y, out[:, :, k, c]
            end
        end
    end
end
