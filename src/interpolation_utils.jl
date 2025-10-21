trivial_range(i::Integer) = i:i

Base.length(itp_dim::AbstractInterpolationDimension) = length(itp_dim.t)

function validate_derivative_orders(
        derivative_orders::NTuple{N, <:Integer},
        ::NDInterpolation{N};
        kwargs...
) where {N}
    @assert all(≥(0), derivative_orders) "Derivative orders must me non-negative."
end

function validate_derivative_orders(
        derivative_orders::NTuple{N, <:Integer},
        A::NDInterpolation{N, N_in, N_out, <:BSplineInterpolationDimension};
        multi_point::Bool = false
) where {N, N_in, N_out}
    @assert all(≥(0), derivative_orders) "Derivative orders must me non-negative."

    if multi_point
        @assert all(
            i -> derivative_orders[i] ≤ A.interp_dims[i].max_derivative_order_eval, 1:N_in
        ) "For BSpline interpolation, when using multi-point evaluation the derivative orders cannot be \
        larger than the `max_derivative_order_eval` eval of of the `BSplineInterpolationDimension`. If you want \
        to compute higher order multi-point derivatives, pass a larger `max_derivative_order_eval` to the \
        `BSplineInterpolationDimension` constructor(s)."
    end

    if A.cache isa NURBSWeights
        @assert all(==(0), derivative_orders) "Currently partial derivatives of NURBS are not supported."
    end
end

function validate_t(t)
    @assert t isa AbstractVector{<:Number} "t must be an AbstractVector with number like elements."
    @assert all(>(0), diff(t)) "The elements of t must be sorted and unique."
end

function validate_size_u(
        interp_dims::NTuple{N_in, <:AbstractInterpolationDimension},
        u::AbstractArray
) where {N_in}
    @assert ntuple(i -> length(interp_dims[i]), N_in)==size(u)[1:N_in] "For the first N_in dimensions of u the length must match the t of the corresponding interpolation dimension."
end
function validate_size_u(
        interp_dims::NTuple{N_in, <:BSplineInterpolationDimension},
        u::AbstractArray
) where {N_in}
    expected_size = ntuple(dim_in -> get_n_basis_functions(interp_dims[dim_in]), N_in)
    @assert expected_size==size(u)[1:N_in] "Expected the size of the first N_in dimensions of u to be $expected_size based on the BSplineInterpolation properties."
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
    # Replace non-interpolated dimensions with :, 
    # interpolated with their first index
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
