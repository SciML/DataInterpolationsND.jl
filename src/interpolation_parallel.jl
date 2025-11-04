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
        interp::NDInterpolation{N,N_in};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in)
) where {N,N_in}
    validate_derivative_order(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    no_interp_inds = map(_ -> Colon(), keep(NoInterpolationDimension, interp.interp_dims))
    @assert all(d -> length(d.t_eval) == size(out, 1),
        remove(NoInterpolationDimension, interp.interp_dims)) "The t_eval of all interpolation dimensions must have the same length as the first dimension of out."
    @assert size(out)[2:end]==get_output_size(interp) "The size of the last N_out dimensions of out must be the same as the output size of the interpolation."
    eval_grid = false
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        no_interp_inds,
        eval_grid,
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
        interp::NDInterpolation{N, N_in, N_out};
        derivative_orders::NTuple{N_in, <:Integer} = ntuple(_ -> 0, N_in),
        no_interp_inds = map(_ -> Colon(), N_out)
) where {N, N_in, N_out}
    validate_t_eval_lengths(out, interp)
    validate_output_size(out, interp)
    validate_derivative_order(derivative_orders, interp; multi_point = true)
    backend = get_backend(out)
    eval_grid = true
    eval_kernel(backend)(
        out,
        interp,
        derivative_orders,
        no_interp_inds,
        eval_grid,
        ndrange = get_ndrange(interp)
    )
    synchronize(backend)
    return out
end

function validate_t_eval_lengths(out, interp)
    (; interp_dims) = interp
    interp_sizes = removeat(NoInterpolationDimension, size(out), interp_dims)
    interp_t_eval_lengths = map(
        d -> length(d.t_eval), remove(NoInterpolationDimension, interp_dims))
    all(map(==, interp_sizes, interp_t_eval_lengths)) ||
        throw(ArgumentError("The length must match the t_eval of the corresponding interpolation dimension."))
end

function validate_output_size(out, interp)
    keepat(NoInterpolationDimension, size(out), interp.interp_dims) ==
    get_output_size(interp) ||
        throw(ArumentError("The size of the NoInterpolationDimension dimensions of `out` must be the same as the output size of the interpolation."))
end

@kernel function eval_kernel(
        out,
        A, # @Const(A), TODO: somehow this now hits a bug in KernelAbstractions where elsize is not defined for Const
        derivative_orders,
        no_interp_inds,
        eval_grid,
)
    (; interp_dims) = A
    N_out = length(keep(NoInterpolationDimension, interp_dims))
    I = @index(Global, NTuple)
    d_o = insertat(NoInterpolationDimension, 0, derivative_orders, interp_dims)

    # TODO eval_grid should be a static bool or Val so this branch can be deleted
    if eval_grid
        # Insert no_interp_inds in I
        k = insertat(NoInterpolationDimension, no_interp_inds, I, interp_dims)
    else # eval_unstructured
        # The same index is used for all interp dimensions in eval_unstructured
        I1 = map(_ -> only(I), remove(NoInterpolationDimension, interp_dims))
        # Insert no_interp_inds in I1
        k = insertat(NoInterpolationDimension, no_interp_inds, I1, interp_dims)
    end
    t_eval = map(get_t_eval, interp_dims, k)
    idx_eval = map(get_idx_eval, interp_dims, k)

    if iszero(N_out)
        @inbounds out[k...] = _interpolate!(
            make_out(A, t_eval), A, t_eval, idx_eval, d_o, k)
    else
        _interpolate!(
            view(out, k...),
            A, t_eval, idx_eval, d_o, k)
    end
end

get_t_eval(d::AbstractInterpolationDimension, i::Integer) = d.t_eval[i]
get_t_eval(d::AbstractInterpolationDimension, i::Colon) = d.t_eval
get_t_eval(d::AbstractInterpolationDimension, i::AbstractVector) = d.t_eval[i]
get_t_eval(::NoInterpolationDimension, i::Integer) = nothing
get_t_eval(::NoInterpolationDimension, i::Colon) = nothing
get_t_eval(::NoInterpolationDimension, i::AbstractVector) = nothing

get_idx_eval(d::AbstractInterpolationDimension, i::Integer) = d.idx_eval[i]
get_idx_eval(d::AbstractInterpolationDimension, i::AbstractVector) = d.idx_eval[i]
get_idx_eval(d::AbstractInterpolationDimension, i::Colon) = d.idx_eval
get_idx_eval(::NoInterpolationDimension, i::Integer) = i
get_idx_eval(::NoInterpolationDimension, i::Colon) = i
get_idx_eval(::NoInterpolationDimension, i::AbstractVector) = i
