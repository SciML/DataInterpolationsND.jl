function _interpolate!(
        out::Union{Number, AbstractArray{<:Any, N_out}},
        A::NDInterpolation{N, N_in, N_out},
        ts::Tuple{Vararg{Any, N}},
        idx::Tuple{Vararg{Any ,N}},
        derivative_orders::Tuple{Vararg{Any, N}},
        multi_point_index
) where {N,N_in,N_out}
    (; interp_dims, cache, u) = A

    out, valid_derivative_orders = check_derivative_order(interp_dims, derivative_orders, ts, out)
    valid_derivative_orders || return out
    if isnothing(multi_point_index)
        multi_point_index = map(_ -> nothing, interp_dims)
    end
    out = make_zero!!(out)
    denom = zero(eltype(u))
    # Setup
    space = map(iteration_space, interp_dims)
    preparations = map(prepare, interp_dims, derivative_orders, multi_point_index, ts, idx)

    for I in Iterators.product(space...)
        scaling = map(scale, interp_dims, preparations, I)
        J = map(index, interp_dims, ts, idx, I)
        if cache isa EmptyCache
            product = prod(scaling)
        else
            K = removeat(NoInterpolationDimension, J, interp_dims)
            product = cache.weights[K...] * prod(scaling)
            denom += product
        end
        if iszero(N_out)
            out += product * u[J...]
        else
            out .+= product .* view(u, J...)
        end
    end

    if !(cache isa EmptyCache)
        if iszero(N_out)
            out /= denom
        else
            out ./= denom
        end
    end

    return out
end

function check_derivative_order(dims::Tuple, derivative_orders::Tuple, ts::Tuple, out)
    itr = map(tuple, dims, derivative_orders, ts)
    # Fold over itr for all dims, combining out and valid
    foldl(itr; init=(out, true)) do (acc_out, acc_valid), (d, d_o, t)
        dim_out, dim_valid = check_derivative_order(d, d_o, t, acc_out)
        dim_out, dim_valid & acc_valid
    end
end
check_derivative_order(::AbstractInterpolationDimension, d_o, t, out) = (out, true)
check_derivative_order(::LinearInterpolationDimension, d_o, t, out) = (out, d_o <= 1)
function check_derivative_order(d::ConstantInterpolationDimension, d_o, t, out)
    if d_o > 0
        # Check if t is on the boundary between constant steps and if so return nans
        return if isempty(searchsorted(d.t, t))
            (out, false)
        else
            (typed_nan(out), false)
        end
    else
        (out, true)
    end
end

function prepare(d::LinearInterpolationDimension, derivative_order, multi_point_index, t, i)
    t₁ = d.t[i]
    t₂ = d.t[i + 1]
    t_vol_inv = inv(t₂ - t₁)
    return (; t, t₁, t₂, t_vol_inv, derivative_order)
end
prepare(::ConstantInterpolationDimension, derivative_orders, multi_point_index, t, i) = (;)
prepare(::NoInterpolationDimension, derivative_orders, multi_point_index, t, i) = (;)
function prepare(d::BSplineInterpolationDimension, derivative_order, multi_point_index, t, i)
    # TODO the dim_in arg isn't really needed, so drop it. Currently just 0
    basis_function_values = get_basis_function_values(
        d, t, i, derivative_order, multi_point_index
    )
    return (; basis_function_values)
end

iteration_space(::LinearInterpolationDimension) = (false, true)
iteration_space(::ConstantInterpolationDimension) = 1
iteration_space(::NoInterpolationDimension) = 1
iteration_space(d::BSplineInterpolationDimension) = 1:d.degree + 1

function scale(::LinearInterpolationDimension, prep::NamedTuple, right_point::Bool)
    (; t, t₁, t₂, t_vol_inv, derivative_order) = prep
    if right_point
        iszero(derivative_order) ? t - t₁ : one(t)
    else
        iszero(derivative_order) ? t₂ - t : -one(t)
    end * t_vol_inv
end
scale(::ConstantInterpolationDimension, prep::NamedTuple, i) = 1
scale(::NoInterpolationDimension, prep::NamedTuple, i) = 1
scale(::BSplineInterpolationDimension, prep::NamedTuple, i) = prep.basis_function_values[i]

index(::LinearInterpolationDimension, t, idx, i) = idx + i
index(d::ConstantInterpolationDimension, t, idx, i) = t >= d.t[end] ? length(d.t) : idx[i]
index(::NoInterpolationDimension, t, idx, i) = Colon()
index(d::BSplineInterpolationDimension, t, idx, i) = idx + i - d.degree - 1
