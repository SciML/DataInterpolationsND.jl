function _interpolate!(
        out,
        A::NDInterpolation{N_in, N_out, ID},
        ts::Tuple{Vararg{Number, N_in}},
        idx::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, <:Integer},
        multi_point_index
) where {N_in, N_out, ID}
    if isnothing(multi_point_index)
        multi_point_index = ntuple(_ -> 1, N_in)
    end
    out = make_zero!!(out)
    # TODO:
    # any(>(1), derivative_orders) && return out
    # if any(>(0), derivative_orders)
    #     return if any(i -> !isempty(searchsorted(A.interp_dims[i].t, t[i])), 1:N_in)
    #         typed_nan(out)
    #     else
    #         out
    #     end
    # end
    # Setup
    space = map(iteration_space, A.interp_dims)
    preparations = map(prepare, A.interp_dims, derivative_orders, multi_point_index, ts, idx)
    # Loop over interpolation space
    for I in Iterators.product(space...)
        scaling = map(scale, A.interp_dims, preparations, I)
        c = prod(scaling) 
        J = map(index, A.interp_dims, ts, idx, I)
        if iszero(N_out)
            out += c * A.u[J...]
        else
            @. out += c * A.u[J...]
        end
    end
    return out
end

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

# NURBS evaluation
# TODO: generalise as above
function _interpolate!(
        out,
        A::NDInterpolation{N_in, N_out, ID, <:NURBSWeights},
        t::Tuple{Vararg{Number, N_in}},
        idx::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, <:Integer},
        multi_point_index
) where {N_in, N_out, ID <: BSplineInterpolationDimension}
    (; interp_dims, cache) = A

    out = make_zero!!(out)
    degrees = ntuple(dim_in -> interp_dims[dim_in].degree, N_in)
    basis_function_vals = get_basis_function_values_all(
        A, t, idx, derivative_orders, multi_point_index
    )

    denom = zero(eltype(t))

    for I in CartesianIndices(ntuple(dim_in -> 1:(degrees[dim_in] + 1), N_in))
        B_product = prod(dim_in -> basis_function_vals[dim_in][I[dim_in]], 1:N_in)
        cp_index = ntuple(
            dim_in -> idx[dim_in] + I[dim_in] - degrees[dim_in] - 1, N_in)
        weight = cache.weights[cp_index...]
        product = weight * B_product
        denom += product
        if iszero(N_out)
            out += product * A.u[cp_index...]
        else
            out .+= product * view(A.u, cp_index..., ..)
        end
    end

    if iszero(N_out)
        out /= denom
    else
        out ./= denom
    end

    return out
end
