module DataInterpolationsND
using KernelAbstractions # Keep as dependency or make extension?
using Adapt: @adapt_structure
using EllipsisNotation
using RecipesBase

abstract type AbstractInterpolationDimension end
abstract type AbstractInterpolationCache end

struct EmptyCache <: AbstractInterpolationCache end

"""
    NDInterpolation(interp_dims, u)

The interpolation object containing the interpolation dimensions and the data to interpolate `u`.
Given the number of interpolation dimensions `N_in`, for first `N_in` dimensions of `u`
the size of `u` along that dimension must match the length of `t` of the corresponding interpolation dimension.

## Arguments

  - `interp_dims`: A tuple of identically typed interpolation dimensions.
  - `u`: The array to be interpolated.
"""
struct NDInterpolation{
    N,
    N_in, 
    N_out,
    gType <: AbstractInterpolationCache,
    D,
    uType <: AbstractArray
}
    u::uType
    interp_dims::D
    cache::gType
    function NDInterpolation(u::AbstractArray{<:Any,N}, interp_dims, cache) where N
        interp_dims = _add_trailing_interp_dims(interp_dims, Val{N}())
        N_in = _count_interpolating_dims(interp_dims)
        N_out = _count_noninterpolating_dims(interp_dims)
        @assert N_out≥0 "The number of dimensions of u must be at least the number of interpolation dimensions."
        validate_size_u(interp_dims, u)
        validate_cache(cache, interp_dims, u)
        new{N, N_in, N_out, typeof(cache), typeof(interp_dims), typeof(u)}(
            u, interp_dims, cache
        )
    end
end

# TODO probably not type-stable (this needs to compile away completely)
_count_interpolating_dims(interp_dims) = count(map(d -> !(d isa NoInterpolationDimension), interp_dims))
_count_noninterpolating_dims(interp_dims) = count(map(d -> d isa NoInterpolationDimension, interp_dims))

_add_trailing_interp_dims(dim::AbstractInterpolationDimension, n) = 
    _add_trailing_interp_dims((dim,), n)
_add_trailing_interp_dims(dims::Tuple, ::Val{N}) where N = 
    (dims..., ntuple(_ -> NoInterpolationDimension(), Val{N-length(dims)}())...)

# Constructor with optional global cache
function NDInterpolation(u, interp_dims; cache = EmptyCache())
    NDInterpolation(u, interp_dims, cache)
end

@adapt_structure NDInterpolation

include("interpolation_dimensions.jl")
include("spline_utils.jl")
include("interpolation_utils.jl")
include("interpolation_methods.jl")
include("interpolation_parallel.jl")
include("plot_rec.jl")
include("interp_array.jl")

# Multiple `t` arguments to tuple (can these 2 be done in 1?)
function (interp::NDInterpolation)(t_args::Vararg{Number}; kwargs...)
    interp(t_args; kwargs...)
end

function (interp::NDInterpolation)(
        out::AbstractArray, t_args::Vararg{Number}; kwargs...)
    interp(out, t_args; kwargs...)
end

# In place single input evaluation
function (interp::NDInterpolation{N,N_in,N_out})(
        out::Union{Number, AbstractArray{<:Number, N_out}},
        t::Tuple{Vararg{Any, N_in}};
        derivative_orders::Tuple{Vararg{Integer,N_in}} = ntuple(_ -> 0, N_in)
) where {N,N_in,N_out}
    validate_size_u(interp, out)
    # We need to add the NoInterpolationDimensions for all arguments to _interpolate.
    # Currently interp() accepts only the interpolating dimensions.
    # We could switch this so it needs indices for NoInterpolationDimension, but thats a breaking change.
    t_all = insertat(NoInterpolationDimension, Colon(), t, interp.interp_dims)
    idx_all = get_idx(interp.interp_dims, t_all)
    d_o_all = insertat(NoInterpolationDimension, 0, derivative_orders, interp.interp_dims)
    validate_derivative_order(d_o_all, interp)
    multi_point_index = nothing
    return _interpolate!(out, interp, t_all, idx_all, d_o_all, multi_point_index)
end

# Out of place single input evaluation
function (interp::NDInterpolation)(t::Tuple{Vararg{Number}}; kwargs...)
    out = make_out(interp, t)
    interp(out, t; kwargs...)
end

export NDInterpolation, LinearInterpolationDimension, ConstantInterpolationDimension,
       BSplineInterpolationDimension, NURBSWeights, NoInterpolationDimension,
       eval_unstructured, eval_unstructured!, eval_grid, eval_grid!

end # module DataInterpolationsND
