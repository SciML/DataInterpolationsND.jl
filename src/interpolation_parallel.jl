"""
    function eval_unstructured(
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the unstructured set of points defined by `t_eval`
in the interpolation dimensions out of place. That is, `t_eval` must have the same
length for each interpolation dimension and the interpolation is evaluated at the `zip` if these `t_eval`.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_unstructured(interp::NDInterpolation; kwargs...)
    n_points = length(first(interp.interp_dims).t_eval)
    out = similar(interp.u, (n_points, get_output_size(interp)...))
    eval_unstructured!(out, interp; kwargs...)
end

"""
    function eval_unstructured!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the unstructured set of points defined by `t_eval`
in the interpolation dimensions in place. That is, `t_eval` must have the same
length for each interpolation dimension and the interpolation is evaluated at the `zip` if these `t_eval`.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_unstructured!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N_in}
    validate_derivative_orders(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    @assert all(i -> length(interp.interp_dims[i].t_eval) == size(out, 1), N_in) "The t_eval of all interpolation dimensions must have the same length as the first dimension of out."
    @assert size(out)[2:end]==get_output_size(interp) "The size of the last N_out dimensions of out must be the same as the output size of the interpolation."
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        false,
        ndrange = size(out, 1)
    )
    synchronize(backend)
    return out
end

"""
    function eval_grid(
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the Cartesian product of the `t_eval` of the interpolation dimensions
out of place.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_grid(interp::NDInterpolation; kwargs...)
    # TODO: do we need to promote the type here, e.g. for eltype(u) <: Integer ?
    out = similar(interp.u, grid_size(interp))
    return eval_grid!(out, interp; kwargs...)
end

"""
    function eval_grid!(
        out::AbstractArray,
        interp::NDInterpolation{N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)) where {N_in}

Evaluate the interpolation in the Cartesian product of the `t_eval` of the interpolation dimensions
in place.

## Keyword arguments

  - `derivative_orders`: The partial derivative order for each interpolation dimension. Defaults to `0` for each.
"""
function eval_grid!(
        out::AbstractArray,
        interp::NDInterpolation{N,N_in};
        derivative_orders::NTuple{N, <:Integer} = ntuple(_ -> 0, N)
) where {N,N_in}
    validate_derivative_order(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    # used_interp_dims = remove(NoInterpolationDimension, interp.interp_dims)
    # @assert all(ntuple(i -> size(out, i) == length(used_interp_dims[i].t_eval), N_in)) "The length must match the t_eval of the corresponding interpolation dimension."
    # @assert size(out) == get_output_size(interp) "The size of the last N_out dimensions of out must be the same as the output size of the interpolation."
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        true,
        ndrange = get_ndrange(interp)
    )
    synchronize(backend)
    return out
end

@kernel function eval_kernel(
        out,
        A::NDInterpolation{N, N_in, N_out},
        derivative_orders,
        eval_grid,
) where {N, N_in, N_out}
    # This kernel is only over interpolated dimensions, we need 
    # to insert fillers to match the number of dimensions in the data
    I = @index(Global, NTuple)
    k = insertat(NoInterpolationDimension, Colon(), I, A.interp_dims) 

    t_eval = map(get_t_eval, A.interp_dims, k)
    idx_eval = map(get_idx_eval, A.interp_dims, k)

    if iszero(N_out)
        dest = make_out(A, t_eval)
        out[I...] = _interpolate!(dest, A, t_eval, idx_eval, derivative_orders, k)
    else
        dest = view(out, k...)
        _interpolate!(dest, A, t_eval, idx_eval, derivative_orders, k)
    end
end

get_t_eval(d, i) = d.t_eval[i]
get_t_eval(d::NoInterpolationDimension, i) = nothing

get_idx_eval(d, i) = d.idx_eval[i]
get_idx_eval(d::NoInterpolationDimension, i) = nothing
