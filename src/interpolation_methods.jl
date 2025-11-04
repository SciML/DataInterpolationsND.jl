Base.@propagate_inbounds function _interpolate!(
        out::Union{Number, AbstractArray},
        A::NDInterpolation{N},
        ts::Tuple{Vararg{Any, N}},
        idx::Tuple{Vararg{Any, N}},
        derivative_orders::Tuple{Vararg{Any, N}},
        multi_point_index
) where {N}
    (; interp_dims, cache, u) = A

    out,
    valid_derivative_orders = check_derivative_order(
        interp_dims, derivative_orders, ts, out)
    valid_derivative_orders || return out # Array was zeroed out in this case
    if isnothing(multi_point_index)
        multi_point_index = map(_ -> nothing, interp_dims)
    end

    # Setup
    out = make_zero!!(out) # TODO remove this
    stencils = map(stencil, interp_dims)
    coeffs = map(coefficients, interp_dims, derivative_orders, multi_point_index, ts, idx)
    denom = zero(eltype(u))

    # TODO this can be a single unrolled broadcast rather than a loop of .+=
    for I in Iterators.product(stencils...)
        J = map(index, interp_dims, ts, idx, I)
        product = prod(map(getindex, coeffs, I))

        if cache isa NURBSWeights
            K = removeat(NoInterpolationDimension, J, interp_dims)
            product *= cache.weights[K...]
            denom += product
        end

        if out isa AbstractArray
            out .+= product .* u[J...]
        else
            out += product * u[J...]
        end
    end

    if cache isa NURBSWeights
        if out isa AbstractArray
            out ./= denom
        else
            out /= denom
        end
    end

    return out
end

function check_derivative_order(dims::Tuple, derivative_orders::Tuple, ts::Tuple, out)
    itr = map(tuple, dims, derivative_orders, ts)
    # Fold over itr for all dims, combining out and valid
    foldl(itr; init = (out, true)) do (acc_out, acc_valid), (d, d_o, t)
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

stencil(::LinearInterpolationDimension) = (1, 2)
stencil(::ConstantInterpolationDimension) = 1
stencil(::NoInterpolationDimension) = 1
stencil(d::BSplineInterpolationDimension) = 1:(d.degree + 1)

# Precalculate coefficient/s
function coefficients(
        d::LinearInterpolationDimension, derivative_order, multi_point_index, t, i)
    t₁ = d.t[i]
    t₂ = d.t[i + 1]
    t_vol_inv = inv(t₂ - t₁)
    a = (iszero(derivative_order) ? t₂ - t : -one(t)) * t_vol_inv
    b = (iszero(derivative_order) ? t - t₁ : one(t)) * t_vol_inv
    return (a, b)
end
function coefficients(
        ::ConstantInterpolationDimension, derivative_order, multi_point_index, t, i)
    true
end
coefficients(::NoInterpolationDimension, derivative_order, multi_point_index, t, i) = true
function coefficients(
        d::BSplineInterpolationDimension, derivative_order, multi_point_index, t, i)
    get_basis_function_values(d, t, i, derivative_order, multi_point_index)
end

index(::LinearInterpolationDimension, t, idx, i) = idx + i - 1
# TODO: this should happen outside of the loop
index(d::ConstantInterpolationDimension, t, idx, i) = t >= d.t[end] ? length(d.t) : idx[i]
index(::NoInterpolationDimension, t, idx, i) = idx
index(d::BSplineInterpolationDimension, t, idx, i) = idx + i - d.degree - 1
