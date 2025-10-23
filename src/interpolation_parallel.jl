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
    validate_output_size(out, interp) 
    backend = get_backend(out)
    # TODO this may be broken but it isn't tested
    @assert all(d -> length(d.t_eval) == size(out, 1), remove(NoInterpolationDimension, interp.interp_dims)) "The t_eval of all interpolation dimensions must have the same length as the first dimension of out."
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
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
    validate_t_eval_lengths(out, interp)
    validate_output_size(out, interp) 
    validate_derivative_order(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        ndrange = get_ndrange(interp)
    )
    synchronize(backend)
    return out
end

function validate_t_eval_lengths(out, interp)
    (; interp_dims) = interp
    interp_sizes = removeat(NoInterpolationDimension, size(out), interp_dims) 
    interp_t_eval_lengths = map(d -> length(d.t_eval), remove(NoInterpolationDimension, interp_dims))
    all(map(==, interp_sizes, interp_t_eval_lengths)) || 
        throw(ArgumentError("The length must match the t_eval of the corresponding interpolation dimension."))
end

function validate_output_size(out, interp) 
    keepat(NoInterpolationDimension, size(out), interp.interp_dims) == get_output_size(interp) ||
        throw(ArumentError("The size of the NoInterpolationDimension dimensions of `out` must be the same as the output size of the interpolation."))
end

@kernel function eval_kernel(
        out,
        A::NDInterpolation{N, N_in, N_out},
        derivative_orders,
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
