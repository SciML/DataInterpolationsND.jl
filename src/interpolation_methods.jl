function _interpolate!(
        out,
        A::NDInterpolation{N_in, N_out, <:NURBSWeights},
        ts::Tuple{Vararg{Number, N_in}},
        idx::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, <:Integer},
        multi_point_index
) where {N_in, N_out}
    (; interp_dims, cache) = A
    check_derivative_orders(interp_dims, derivative_orders) || return
    if isnothing(multi_point_index)
        multi_point_index = ntuple(_ -> 1, N_in)
    end
    out = make_zero!!(out)
    denom = zero(eltype(t))
    # Setup
    space = map(iteration_space, interp_dims)
    preparations = map(prepare, interp_dims, derivative_orders, multi_point_index, ts, idx)

    for I in Iterators.product(space...)
        scaling = map(scale, A.interp_dims, preparations, I)
        J = map(index, A.interp_dims, ts, idx, I)
        product = if isnothing(cache)
            scaling
        else
            weight = cache.weights[J...]
            product = weight * scaling
            denom += product
        end
        if iszero(N_out)
            out += product * A.u[J...]
        else
            out .+= product * view(A.u, J..., ..)
        end
    end

    if !isnothing(cache)
        if iszero(N_out)
            out /= denom
        else
            out ./= denom
        end
    end

    return out
end

check_derivative_orders(dims, derivative_orders) = false
# TODO:
# any(>(1), derivative_orders) && return out
# if any(>(0), derivative_orders)
#     return if any(i -> !isempty(searchsorted(A.interp_dims[i].t, t[i])), 1:N_in)
#         typed_nan(out)
#     else
#         out
#     end
# end

function prepare(d::LinearInterpolationDimension, derivative_order, multi_point_index, t, i)
    t₁ = d.t[i]
    t₂ = d.t[i + 1]
    t_vol_inv = inv(t₂ - t₁)
    return (; t, t₁, t₂, t_vol_inv, derivative_order)
end
prepare(::ConstantInterpolationDimension, derivative_orders, multi_point_index, t, i) = nothing
function prepare(d::BSplineInterpolationDimension, derivative_order, multi_point_index, t::Number, i::Integer)
    # TODO the dim_in arg isn't really needed, so drop it. Currently just 0
    basis_function_values = get_basis_function_values(d, t, i, derivative_order, multi_point_index, 0)
    return (; basis_function_values)
end

iteration_space(::LinearInterpolationDimension) = (false, true)
iteration_space(::ConstantInterpolationDimension) = 1
iteration_space(d::BSplineInterpolationDimension) = 1:d.degree + 1

function scale(::LinearInterpolationDimension, prep::NamedTuple, right_point::Bool)
    (; t, t₁, t₂, t_vol_inv, derivative_order) = prep
    if right_point
        iszero(derivative_order) ? t - t₁ : one(t)
    else
        iszero(derivative_order) ? t₂ - t : -one(t)
    end * t_vol_inv
end
scale(::ConstantInterpolationDimension, prep, i) = 1
scale(::BSplineInterpolationDimension, prep::NamedTuple, i) = prep.basis_function_values[i]

index(::LinearInterpolationDimension, t, idx, i) = idx + i
index(d::ConstantInterpolationDimension, t, idx, i) = t >= d.t[end] ? length(d.t) : idx[i]
index(d::BSplineInterpolationDimension, t, idx, i) = idx + i - d.degree - 1
