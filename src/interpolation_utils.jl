trivial_range(i::Integer) = i:i

Base.length(itp_dim::AbstractInterpolationDimension) = length(itp_dim.t)

function validate_derivative_order(derivative_orders::NTuple, A::NDInterpolation;
    multi_point::Bool = false
)
    map(derivative_orders, A.interp_dims) do d_o, d
        validate_derivative_order(d_o, d; multi_point, cache=A.cache)
    end
end
function validate_derivative_order(
    derivative_order::Integer,
    interp_dim::BSplineInterpolationDimension;
    multi_point::Bool,
    cache,
)
    if multi_point
        @assert derivative_order ≤ interp_dim.max_derivative_order_eval """
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
    cache,
)
    validate_derivative_order_by_cache(cache, derivative_order)
end

validate_derivative_order_by_cache(::NURBSWeights, derivative_order)  =
    @assert derivative_order == 0 "Currently partial derivatives of NURBS are not supported."
validate_derivative_order_by_cache(::Any, derivative_order) =
    @assert derivative_order >= 0 "Derivative orders must me non-negative."


function validate_t(t)
    @assert t isa AbstractVector{<:Number} "t must be an AbstractVector with number like elements."
    @assert all(>(0), diff(t)) "The elements of t must be sorted and unique."
end

validate_size_u(interp_dims::NTuple, u) = map(validate_size_u, interp_dims, axes(u))
function validate_size_u(interp_dim::AbstractInterpolationDimension, ax)
    @assert length(interp_dim) == length(ax) "For the first N_in dimensions of u the length must match the t of the corresponding interpolation dimension."
end
function validate_size_u(interp_dim::BSplineInterpolationDimension, ax)
    expected_size = get_n_basis_functions(interp_dim)
    @assert expected_size == length(ax) "Expected the size of the first N_in dimensions of u to be $expected_size based on the BSplineInterpolation properties."
end

function validate_cache(cache, dims::NTuple, u)
    ntuple(length(dims)) do n
        validate_cache(cache, dims[n], u, n)
    end
end
validate_cache(::EmptyCache, ::AbstractInterpolationDimension, ::AbstractArray, ::Int) = nothing
function validate_cache(
        nurbs_weights::NURBSWeights,
        ::BSplineInterpolationDimension,
        u::AbstractArray,
        n::Int,
)
    size_expected = size(u, n)
    @assert size(nurbs_weights.weights, n) == size_expected "The size of the weights array must match the length of the first N_in dimensions of u ($size_expected), got $(size(nurbs_weights.weights, n))."
end
function validate_cache(::gType, ::ID, ::AbstractArray, ::Int) where {gType,ID<:AbstractInterpolationDimension}
    @error("Interpolation dimension type $ID is not compatible with global cache type $gType.")
end

function get_output_size(interp::NDInterpolation)
    I = map(interp.interp_dims, axes(interp.u)) do d, ax
        d isa NoInterpolationDimension ? Colon() : first(ax)
    end
    return size(view(interp.u, I...))
end

make_zero!!(::T) where {T <: Number} = zero(T)
function make_zero!!(v::T) where {T <: AbstractArray}
    v .= 0
    v
end

function make_out(
        interp::NDInterpolation{<:Any,N_in, 0},
        t::NTuple{N_in, >:Number}
) where {N_in}
    zero(promote_type(eltype(interp.u), map(typeof, t)...))
end
function make_out(
        interp::NDInterpolation{<:Any,N_in},
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
function get_idx(interp_dims::NTuple{N_in}, t::Tuple{Vararg{Number, N_in}}) where N_in
    used_interp_dims = _remove(NoInterpolationDimension, interp_dims...)
    map(get_idx, used_interp_dims, t)
end

function set_eval_idx!(interp_dim::AbstractInterpolationDimension)
    backend = get_backend(interp_dim.t)
    if !isempty(interp_dim.t_eval)
        set_idx_kernel(backend)(
            interp_dim,
            ndrange = length(interp_dim.t_eval)
        )
    end
    synchronize(backend)
end

@kernel function set_idx_kernel(interp_dim)
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
