module NDInterpolationsLinearMapsExt
using NDInterpolations
using NDInterpolations: validate_derivative_orders, get_output_size
using LinearMaps

# A linear map interp.u -> grid evaluation
function LinearMaps.LinearMap(
        interp::NDInterpolation{N_in}, ::Val{:grid};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp)

    T = NDInterpolations.output_type(interp)

    grid_size = map(itp_dim -> length(itp_dim.t_eval), interp.interp_dims)
    size_out = (grid_size..., get_output_size(interp)...)
    N_input = length(interp.u)
    N_output = prod(size_out)

    function map!(out_flat, u_flat)
        u_reshaped = reshape(u_flat, size(interp.u))
        out_reshaped = reshape(out_flat, size_out)

        interp_ = NDInterpolation(u_reshaped, interp.interp_dims, interp.cache)
        eval_grid!(out_reshaped, interp_; derivative_orders)
        return out_flat
    end

    function map_adjoint!(u_flat, out_flat)
        u_reshaped = reshape(u_flat, size(interp.u))
        out_reshaped = reshape(out_flat, size_out)

        interp_ = NDInterpolation(u_reshaped, interp.interp_dims, interp.cache)
        eval_grid!(out_reshaped, interp_; derivative_orders, adjoint = true)
        return u_flat
    end

    return FunctionMap{T}(map!, map_adjoint!, N_output, N_input)
end

# A linear map interp.u -> unstructured evaluation
function LinearMaps.LinearMap(
        interp::NDInterpolation{N_in}, ::Val{:unstructured};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp)

    T = NDInterpolations.output_type(interp)

    N_input = length(interp.u)
    size_out = (length(first(interp.interp_dims).t_eval), get_output_size(interp)...)
    N_output = prod(size_out)

    function map!(out_flat, u_flat)
        u_reshaped = reshape(u_flat, size(interp.u))
        out_reshaped = reshape(out_flat, size_out)

        interp_ = NDInterpolation(u_reshaped, interp.interp_dims, interp.cache)
        eval_unstructured!(out_reshaped, interp_; derivative_orders)
        return out_flat
    end

    # function map_adjoint!()
    # end

    return FunctionMap{T}(map!, N_output, N_input)
end

end # module NDInterpolationsLinearMapsExt