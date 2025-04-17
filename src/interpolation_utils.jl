trivial_range(i::Integer) = i:i

Base.length(itp_dim::AbstractInterpolationDimension) = length(itp_dim.t)

function validate_derivative_orders(derivative_orders::NTuple{N_in, <:Integer}) where {N_in}
    @assert all(≥(0), derivative_orders) "Derivative orders must me non-negative."
end

function get_ts(interp_dims::NTuple{
        N_in, AbstractInterpolationDimension}) where {N_in}
    ntuple(i -> interp_dims[i].t, N_in)
end

function get_output_size(interp::NDInterpolation{N_in}) where {N_in}
    size(interp.u)[(N_in + 1):end]
end

function make_out(
        interp::NDInterpolation{N_in, 0},
        t::NTuple{N_in, >:Number}
) where {N_in}
    zero(promote_type(eltype(interp.u), map(typeof, t)...))
end

function make_out(
        interp::NDInterpolation{N_in},
        t::NTuple{N_in, >:Number}
) where {N_in}
    similar(
        interp.u, promote_type(eltype(interp.u), map(eltype, t)...), get_output_size(interp))
end

get_left(::AbstractInterpolationDimension) = false
get_left(::LinearInterpolationDimension) = true

get_idx_bounds(::AbstractInterpolationDimension) = (1, -1)

get_idx_shift(::AbstractInterpolationDimension) = 0
get_idx_shift(::LinearInterpolationDimension) = -1

# TODO: Implement a more efficient (GPU compatible) version
function get_idx(
        interp_dim::AbstractInterpolationDimension,
        t_eval::Number
)
    (; t) = interp_dim
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

function get_idx(
        interp_dims::NTuple{N_in},
        t::Tuple{Vararg{Number, N_in}};
) where {N_in}
    ntuple(dim_in -> get_idx(interp_dims[dim_in], t[dim_in]), N_in)
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
