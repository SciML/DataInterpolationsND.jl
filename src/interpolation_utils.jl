trivial_range(i::Integer) = i:i

Base.length(itp_dim::AbstractInterpolationDimension) = length(itp_dim.t)

function validate_derivative_order(
        derivative_orders::NTuple,
        A::NDInterpolation;
        multi_point::Bool = false
)
    map(derivative_orders, A.interp_dims) do d_o, d
        validate_derivative_order(d_o, d; multi_point, cache = A.cache)
    end
end
function validate_derivative_order(
        derivative_order::Integer,
        interp_dim::BSplineInterpolationDimension;
        multi_point::Bool,
        cache
)
    if multi_point
        @assert derivative_order≤interp_dim.max_derivative_order_eval """
For BSpline interpolation, when using multi-point evaluation the derivative orders cannot be
larger than the `max_derivative_order_eval` eval of of the `BSplineInterpolationDimension`. If you want
to compute higher order multi-point derivatives, pass a larger `max_derivative_order_eval` to the
`BSplineInterpolationDimension` constructor(s).
"""
    end
    validate_derivative_order_by_cache(cache, derivative_order)
end
function validate_derivative_order(
        derivative_order::Integer,
        interp_dim::AbstractInterpolationDimension;
        multi_point::Bool,
        cache
)
    validate_derivative_order_by_cache(cache, derivative_order)
end

function validate_derivative_order_by_cache(::NURBSWeights, derivative_order)
    @assert derivative_order==0 "Currently partial derivatives of NURBS are not supported."
end
function validate_derivative_order_by_cache(::Any, derivative_order)
    @assert derivative_order>=0 "Derivative orders must me non-negative."
end

function validate_t(t)
    @assert t isa AbstractVector{<:Number} "t must be an AbstractVector with number like elements."
    @assert all(>(0), diff(t)) "The elements of t must be sorted and unique."
end

validate_size_u(interp::NDInterpolation, u) = validate_size_u(interp.interp_dims, u)
validate_size_u(interp_dims::Tuple, u::Number) = nothing
function validate_size_u(interp_dims::Tuple, u::AbstractArray)
    validate_size_u(interp_dims, axes(u))
end
validate_size_u(interp_dims::Tuple, ax::Tuple) = map(validate_size_u, interp_dims, ax)
validate_size_u(interp_dim::NoInterpolationDimension, ax::AbstractRange) = nothing
function validate_size_u(interp_dim::AbstractInterpolationDimension, ax::AbstractRange)
    @assert length(interp_dim) == length(ax) "The size if `u` must match the t of the corresponding interpolation dimension. Got $interp_dim and $ax"
end

function validate_size_u(
        interp_dim::BSplineInterpolationDimension,
        ax::AbstractRange
)
    expected_size = get_n_basis_functions(interp_dim)
    @assert expected_size==length(ax) "Expected the size to be $expected_size based on the BSplineInterpolation properties, got $(length(ax))."
end

function validate_cache(
        cache::AbstractInterpolationCache, dims::Tuple, u::AbstractArray
)
    ntuple(length(dims)) do n
        validate_cache(cache, dims[n], u, n)
    end
end
function validate_cache(
        ::EmptyCache, ::AbstractInterpolationDimension, ::AbstractArray, ::Int)
    nothing
end
validate_cache(::EmptyCache, ::NoInterpolationDimension, ::AbstractArray, ::Int) = nothing
function validate_cache(
        ::AbstractInterpolationCache, ::NoInterpolationDimension, ::AbstractArray, ::Int)
    nothing
end
function validate_cache(
        nurbs_weights::NURBSWeights,
        ::BSplineInterpolationDimension,
        u::AbstractArray,
        n::Int
)
    size_expected = size(u, n)
    @assert size(nurbs_weights.weights, n)==size_expected "The size of the weights array must match the length of the first N_in dimensions of u ($size_expected), got $(size(nurbs_weights.weights, n))."
end

function validate_cache(
        ::gType, ::ID, ::AbstractArray, ::Int) where {
        gType, ID <: AbstractInterpolationDimension}
    @error("Interpolation dimension type $ID is not compatible with global cache type $gType.")
end

function get_output_size(interp::NDInterpolation)
    I = map(interp.interp_dims, axes(interp.u)) do d, ax
        d isa NoInterpolationDimension ? length(ax) : nothing
    end
    return remove(Nothing, I)
end

function grid_size(interp::NDInterpolation)
    (; interp_dims) = interp
    # Get the size of dims that are not NoInterpolationDimension
    interp_size = map(d -> length(d.t_eval), remove(NoInterpolationDimension, interp_dims))
    # Get the size of NoInterpolationDimension dims 
    nointerp_size = get_output_size(interp)
    # Insert the nointerp sizes back into the interp_size tuple
    return insertat(NoInterpolationDimension, nointerp_size, interp_size, interp_dims)
end

make_zero!!(::T) where {T <: Number} = zero(T)

function make_zero!!(v::AbstractArray)
    fill!(v, zero(eltype(v)))
    v
end

function make_out(
        interp::NDInterpolation{<:Any, N_in, 0},
        t::NTuple{N_in, >:Number}
) where {N_in}
    zero(eltype(interp.u))
end
function make_out(
        interp::NDInterpolation{<:Any, N_in},
        t::NTuple{N_in, >:Number}
) where {N_in}
    T = promote_type(eltype(interp.u), map(eltype, t)...)
    similar(interp.u, T, get_output_size(interp))
end

get_left(::AbstractInterpolationDimension) = false
get_left(::LinearInterpolationDimension) = true

get_idx_bounds(::AbstractInterpolationDimension) = (1, -1)
function get_idx_bounds(itp_dim::BSplineInterpolationDimension)
    (itp_dim.degree + 1, -itp_dim.degree - 1)
end

get_idx_shift(::AbstractInterpolationDimension) = 0
get_idx_shift(::LinearInterpolationDimension) = -1

# TODO: Implement a more efficient (GPU compatible) version
function get_idx(
        interp_dim::AbstractInterpolationDimension,
        t_eval::Number
)
    t = if interp_dim isa BSplineInterpolationDimension
        interp_dim.knots_all
    else
        interp_dim.t
    end
    left = get_left(interp_dim)
    lb, ub_shift = get_idx_bounds(interp_dim)
    idx_shift = get_idx_shift(interp_dim)
    ub = length(t) + ub_shift
    return if left
        clamp(searchsortedfirst(t, t_eval) + idx_shift, lb, ub)
    else
        clamp(searchsortedlast(t, t_eval) + idx_shift, lb, ub)
    end
end
# t_eval must already be an index for NoInterpolationDimension
get_idx(::NoInterpolationDimension, t_eval::Union{Colon, Int, AbstractArray{Int}}) = t_eval
function get_idx(interp_dims::Tuple{Vararg{Any, N}}, t::Tuple{Vararg{Any, N}}) where {N}
    map(get_idx, interp_dims, t)
end

function set_eval_idx!(
        interp_dim::AbstractInterpolationDimension,
)
    backend = get_backend(interp_dim.t)
    if !isempty(interp_dim.t_eval)
        set_idx_kernel(backend)(
            interp_dim,
            ndrange = length(interp_dim.t_eval)
        )
    end
    synchronize(backend)
end

@kernel function set_idx_kernel(
        interp_dim
)
    i = @index(Global, Linear)
    interp_dim.idx_eval[i] = get_idx(interp_dim, interp_dim.t_eval[i])
end

function typed_nan(x::AbstractArray{T}) where {T <: AbstractFloat}
    x .= NaN
end

function typed_nan(x::AbstractArray{T}) where {T <: Integer}
    x .= 0
end

typed_nan(::T) where {T <: Integer} = zero(T)
typed_nan(::T) where {T <: AbstractFloat} = T(NaN)

# Get the KernelAbstractions nd_range, over interpolated dimensions
function get_ndrange(interp::NDInterpolation)
    I = map(interp.interp_dims) do d
        d isa NoInterpolationDimension ? nothing : length(d.t_eval)
    end
    return remove(Nothing, I)
end

# Some tuple handling primitives
# TODO: make sure these compile away completely

# Remove objects of type `T` from `in` (reworked from DimensionalData.jl) 
remove(::Type{T}, in) where {T} = _remove(T, in...)
_remove(::Type{T}, x, xs...) where {T} = (x, _remove(T, xs...)...)
_remove(::Type{T}, x::T, xs...) where {T} = _remove(T, xs...)
_remove(::Type) = ()

# Keep only objects of type `T` from `in`
keep(::Type{T}, in) where {T} = _keep(T, in...)
_keep(::Type{T}, x, xs...) where {T} = _keep(T, xs...)
_keep(::Type{T}, x::T, xs...) where {T} = (x, _keep(T, xs...)...)
_keep(::Type) = ()

# Remove values from `in` where `matches` are of type `T`
function removeat(::Type{T}, in::Tuple, matches::Tuple) where {T}
    @assert length(in) == length(matches)
    _removeat(T, in, matches...)
end
# If `!(m isa T)` take from `in`
function _removeat(::Type{T}, in::Tuple{<:Any, Vararg}, m, ms...) where {T}
    (first(in), _removeat(T, Base.tail(in), ms...)...)
end
# If `m isa T` remove
function _removeat(::Type{T}, in::Tuple{<:Any, Vararg}, m::T, ms...) where {T}
    _removeat(T, Base.tail(in), ms...)
end
# `in` can be empty if there are no remaining `m`
_removeat(::Type, in::Tuple{}) = ()

# Keep only values from `in` where `matches` are of type `T`
function keepat(::Type{T}, in::Tuple, matches::Tuple) where {T}
    @assert length(in) == length(matches)
    _keepat(T, in, matches...)
end
# If `!(m isa T)` disgaurd
function _keepat(::Type{T}, in::Tuple{<:Any, Vararg}, m, ms...) where {T}
    _keepat(T, Base.tail(in), ms...)
end
# If `m isa T` keep
function _keepat(::Type{T}, in::Tuple{<:Any, Vararg}, m::T, ms...) where {T}
    (first(in), _keepat(T, Base.tail(in), ms...)...)
end
# `in` can be empty if there are no remaining `m`
_keepat(::Type, in::Tuple{}) = ()

# Insert x into `in` where `matches` are of type `T`, 
# otherwise output one of `in` for each `m`. 
function insertat(::Type{T}, x, in::Tuple, matches::Tuple) where {T}
    _insertat(T, x, in, matches...)
end
# If `!(m isa T)` take from `in`
function _insertat(::Type{T}, x, in::Tuple{<:Any, Vararg}, m, ms...) where {T}
    (first(in), _insertat(T, x, Base.tail(in), ms...)...)
end
# If `m isa T` insert `x`
function _insertat(::Type{T}, x, in::Tuple{<:Any, Vararg}, m::T, ms...) where {T}
    (x, _insertat(T, x, in, ms...)...)
end
# For Tuple x we insert the first
function _insertat(::Type{T}, xs::Tuple, in::Tuple{<:Any, Vararg}, m::T, ms...) where {T}
    (first(xs), _insertat(T, Base.tail(xs), in, ms...)...)
end
# `in` can be empty if there are no remaining `m`
_insertat(::Type, x, in::Tuple{}) = ()
# `in` can also be empty if all trailing ms are T
function _insertat(::Type{T}, x, in::Tuple{}, m::T, ms::T...) where {T}
    (x, _insertat(T, x, in, ms...)...)
end
function _insertat(::Type{T}, xs::Tuple, in::Tuple{}, m::T, ms::T...) where {T}
    (first(xs), _insertat(T, Base.tail(xs), in, ms...)...)
end
