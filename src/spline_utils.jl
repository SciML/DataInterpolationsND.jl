@kernel function expand_knot_vector_kernel(
        knots_all,
        @Const(knot_values),
        @Const(multiplicities)
    )
    i = @index(Global, Linear)
    knot_value = knot_values[i]

    mult_sum = 0
    idx_end = i - 1
    for j in 1:idx_end
        mult_sum += multiplicities[j]
    end

    idx_start = mult_sum + 1
    idx_end = mult_sum + multiplicities[i]
    for k in idx_start:idx_end
        knots_all[k] = knot_value
    end
end

safe_div(a, b) = iszero(b) ? zero(promote_type(typeof(a), typeof(b))) : a / b

function cox_de_boor(
        basis_function_values, knots_all, t, idx, degree, d, k, deriv
    )
    T = eltype(basis_function_values)
    return if (k ≤ degree - d) || (k == degree + 2)
        zero(T)
    else
        i = idx + k - degree - 1
        tᵢ = knots_all[i]
        tᵢ₊₁ = knots_all[i + 1]
        tᵢ₊ₚ = knots_all[i + d]
        tᵢ₊ₚ₊₁ = knots_all[i + d + 1]
        if deriv
            T(
                d * (
                    safe_div(basis_function_values[k], tᵢ₊ₚ - tᵢ) -
                        safe_div(basis_function_values[k + 1], tᵢ₊ₚ₊₁ - tᵢ₊₁)
                )
            )
        else
            T(
                basis_function_values[k] * safe_div(t - tᵢ, tᵢ₊ₚ - tᵢ) +
                    basis_function_values[k + 1] * safe_div(tᵢ₊ₚ₊₁ - t, tᵢ₊ₚ₊₁ - tᵢ₊₁)
            )
        end
    end
end

# Basis function `Bᵢₚ` (Where `p` is the degree) has support on the interval `[tᵢ, tᵢ₊ₚ₊₁)`
# where these `tᵢ` come from `itp_dim.knots_all`. That means that an input `t ∈ [tⱼ, tⱼ₊₁)` (where `j = idx`)
# lies in the support of `Bᵢₚ` for `i = j - p, …, j`, i.e. `p + 1` of the basis functions.
#
# The basis functions are computed recursively using the Cox - de Boor formula in tuples of
# length `p + 2` as follows:
# (0, …, 0, Bⱼ₀, 0) = (0, … 0, 1, 0)
# (0, …, 0, Bⱼ₋₁ ₁, Bⱼ₁, 0)
#          ⋮
# (Bⱼ₋ₚ ₚ, Bⱼ₋ₚ₊₁ ₚ, …, Bⱼₚ, 0)
#
# The trailing zero is just a convenience for the algorithm and is removed in the output
#
# If `reduce_degree` is `true`, then instead of the derivatives of the degree `p` basis functions
# the values of the degree `p - derivative_order` basis functions are returned, in the last
# `p - derivative_order + 1` entries of the output. These are the weights of the differenced
# control points, see `differenced_contraction`.
function get_basis_function_values(
        itp_dim::BSplineInterpolationDimension,
        t::T_t,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::Nothing,
        dim_in::Integer,
        reduce_degree::Bool = false
    ) where {T_t <: Number}
    (; degree, knots_all) = itp_dim
    T_eval = eltype(itp_dim.basis_function_eval)
    # Use a concrete zero value for type stability
    _zero = zero(promote_type(T_t, T_eval))
    _one = one(promote_type(T_t, T_eval))
    degree_plus_1 = degree + 1

    if derivative_order > degree
        return ntuple(_ -> _zero, degree_plus_1)
    end
    iszero(degree) && return (_one,)

    degree_plus_2 = degree + 2

    # Degree 0 basis function values
    basis_function_values = ntuple(
        k -> (k == degree_plus_1) ? _one : _zero,
        degree_plus_2
    )

    boundary, mode, outside = extrapolation_boundary(itp_dim, t)
    t_eval = t
    if outside && mode != ExtrapolationType.Extension
        if mode == ExtrapolationType.Constant
            derivative_order > 0 && return ntuple(_ -> _zero, degree_plus_1)
            t_eval = boundary + zero(t)
        else
            derivative_order > 1 && return ntuple(_ -> _zero, degree_plus_1)
            edge = boundary + zero(t)
            if derivative_order == 1
                slope = _compute_basis_function_values(
                    basis_function_values, knots_all, edge, idx, degree, 1, degree_plus_2,
                    reduce_degree
                )
                return slope[1:degree_plus_1]
            end
            slope = _compute_basis_function_values(
                basis_function_values, knots_all, edge, idx, degree, 1, degree_plus_2, false
            )
            values = _compute_basis_function_values(
                basis_function_values, knots_all, edge, idx, degree, 0, degree_plus_2, false
            )
            return ntuple(i -> values[i] + (t - edge) * slope[i], degree_plus_1)
        end
    end

    # Higher order basis function values
    # Use a helper function to avoid capturing mutable variables in closures
    basis_function_values = _compute_basis_function_values(
        basis_function_values, knots_all, t_eval, idx, degree, derivative_order, degree_plus_2,
        reduce_degree
    )

    return basis_function_values[1:degree_plus_1]
end

# Helper function to avoid type instability from captured variables in closures
function _compute_basis_function_values(
        basis_function_values, knots_all, t, idx, degree, derivative_order, degree_plus_2,
        reduce_degree
    )
    # Stopping after `degree - derivative_order` steps gives the lower degree basis function values
    n_steps = reduce_degree ? degree - derivative_order : degree
    for d in 1:n_steps
        deriv = d > degree - derivative_order
        # Create a local copy for the closure to capture
        bfv = basis_function_values
        basis_function_values = ntuple(
            k -> cox_de_boor(bfv, knots_all, t, idx, degree, d, k, deriv),
            degree_plus_2
        )
    end
    return basis_function_values
end

# The weights of the (differenced) control points for evaluating the `derivative_order`-th
# derivative along this dimension: the degree `p - derivative_order` basis function values,
# see `get_basis_function_values`. For multi point evaluation these are precomputed.
function get_spline_weights(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::Nothing,
        dim_in::Integer
    )
    return get_basis_function_values(
        itp_dim, t, idx, derivative_order, multi_point_index, dim_in, true
    )
end

# Get the precomputed weights for one point in an
# unstructured multi point evaluation (given by the scalar multi point index)
function get_spline_weights(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::Number,
        dim_in::Integer
    )
    return view(
        itp_dim.basis_function_eval,
        multi_point_index, :, derivative_order + 1
    )
end

# Get the precomputed weights for one point in a
# grid evaluation (given by the tuple multi point index)
function get_spline_weights(
        itp_dim::BSplineInterpolationDimension,
        t::Number,
        idx::Integer,
        derivative_order::Integer,
        multi_point_index::NTuple{N_in, <:Integer},
        dim_in::Integer
    ) where {N_in}
    return view(
        itp_dim.basis_function_eval,
        multi_point_index[dim_in], :, derivative_order + 1
    )
end

# Get the weights for all dimensions to evaluate a BSpline interpolation in t
function get_spline_weights_all(
        A::NDInterpolation{N_in, N_out, <:BSplineInterpolationDimension},
        t::Tuple{Vararg{Number, N_in}},
        idx::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, <:Integer},
        multi_point_index
    ) where {N_in, N_out}
    return ntuple(
        dim_in -> get_spline_weights(
            A.interp_dims[dim_in], t[dim_in], idx[dim_in], derivative_orders[dim_in], multi_point_index, dim_in
        ),
        N_in
    )
end

function set_basis_function_eval!(itp_dim::BSplineInterpolationDimension)::Nothing
    backend = get_backend(itp_dim.t_eval)
    basis_function_eval_kernel(backend)(
        itp_dim,
        ndrange = (length(itp_dim.t_eval), itp_dim.max_derivative_order_eval + 1)
    )
    synchronize(backend)
    return nothing
end

@kernel function basis_function_eval_kernel(
        itp_dim
    )
    i, derivative_order_plus_1 = @index(Global, NTuple)

    itp_dim.basis_function_eval[
        i,
        :,
        derivative_order_plus_1,
    ] .= get_spline_weights(
        itp_dim,
        itp_dim.t_eval[i],
        itp_dim.idx_eval[i],
        derivative_order_plus_1 - 1,
        nothing,
        0
    )
end

# Wrapper for a BSplineInterpolationDimension which acts as a vector of the values
# of the i-th basis function in the `itp_dim.t_eval`.
struct BasisFunctionVector{ID <: BSplineInterpolationDimension, T} <: AbstractVector{T}
    itp_dim::ID
    i::Int
    derivative_order_eval::Int
    function BasisFunctionVector(itp_dim, i, derivative_order_eval)
        n_basis_functions = get_n_basis_functions(itp_dim)
        @assert 1 ≤ i ≤ n_basis_functions "The itp_dim has only $n_basis_functions basis functions, got $i."
        @assert 0 ≤ derivative_order_eval ≤ itp_dim.max_derivative_order_eval "The itp_dim has max_derivative_order_eval = $(itp_dim.max_derivative_order_eval), got $derivative_order_eval."
        return new{typeof(itp_dim), eltype(itp_dim.basis_function_eval)}(
            itp_dim, i, derivative_order_eval
        )
    end
end

Base.length(bfv::BasisFunctionVector) = length(bfv.itp_dim.t_eval)
Base.size(bfv::BasisFunctionVector) = (length(bfv),)

function Base.getindex(bfv::BasisFunctionVector, j)
    (; itp_dim, i, derivative_order_eval) = bfv
    (; basis_function_eval, idx_eval, degree) = itp_dim
    idx = idx_eval[j]
    T = eltype(basis_function_eval)
    return if !(i ≤ idx ≤ i + degree)
        zero(T)
    elseif iszero(derivative_order_eval)
        basis_function_eval[j, degree + i - idx + 1, 1]
    else
        # For derivatives `basis_function_eval` holds the lower degree basis function values
        T(
            get_basis_function_values(
                itp_dim, itp_dim.t_eval[j], idx, derivative_order_eval, nothing, 0
            )[degree + i - idx + 1]
        )
    end
end

function get_n_basis_functions(itp_dim::BSplineInterpolationDimension)
    return length(itp_dim.knots_all) - itp_dim.degree - 1
end

# The derivative of a B-spline is a B-spline of lower degree whose control points are scaled
# differences of the original control points (de Boor). Along a dimension of degree `p`, the
# control points for the `r`-th derivative are
#
#     Pʳᵢ = (p - r + 1) * (Pʳ⁻¹ᵢ - Pʳ⁻¹ᵢ₋₁) / (tᵢ₊ₚ₋ᵣ₊₁ - tᵢ),
#
# which are the coefficients of the degree `p - r` basis functions Bᵢ,ₚ₋ᵣ (with the same knots).
# Differencing the control points before contracting them with the basis functions keeps the
# rounding error relative to the size of the derivative, instead of relative to |P| / hᵏ as
# when contracting the control points with the derivatives of the basis functions.
#
# This evaluates the (mixed) partial derivative given by `derivative_orders` at the point in the
# knot span `idx`, contracted over the dimensions with a nonzero derivative order only, one
# dimension at a time: along dimension `d` the `p + 1` local control points are differenced
# `derivative_orders[d]` times, and the last `p - derivative_orders[d] + 1` of the results are
# contracted with the lower degree basis function values `weights[d]`. The local control points
# along dimension `d` are themselves such contractions over the dimensions below `d`. Along the
# dimensions with derivative order zero the control point index is fixed to `cp_index`.
# This is O(p * k) work per dimension, and no temporaries other than tuples of length p + 1.
# Values are converted to `T` before differencing, so that integer control points do not wrap
# around and small float types do not overflow.
@inline function differenced_contraction(
        A::NDInterpolation{N_in},
        idx::NTuple{N_in, <:Integer},
        cp_index::NTuple{N_in, <:Integer},
        derivative_orders::NTuple{N_in, Int},
        weights,
        out_index::Tuple,
        ::Type{T}
    ) where {N_in, T}
    return _differenced_contraction(
        A.u, A.interp_dims, idx, cp_index, derivative_orders, weights, out_index, T, Val(N_in), ()
    )
end

# Contract along dimension `d`, with the indices of the dimensions above `d` given by `i_tail`
@inline function _differenced_contraction(
        u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, ::Type{T},
        ::Val{d}, i_tail
    ) where {T, d}
    if iszero(derivative_orders[d])
        return local_control_point(
            u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, T, Val(d - 1),
            (cp_index[d], i_tail...)
        )
    end
    degree = interp_dims[d].degree
    i_first = idx[d] - degree
    control_points = ntuple(degree + 1) do j
        i = (i_first + j - 1, i_tail...)
        local_control_point(
            u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, T, Val(d - 1), i
        )
    end
    return contract_differenced(
        control_points, interp_dims[d].knots_all, i_first, degree, derivative_orders[d],
        weights[d], T
    )
end

# The local control point with index `i`: a control point of the interpolation for `d = 0`,
# otherwise the contraction over the dimensions up to `d`
@inline function local_control_point(
        u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, ::Type{T}, ::Val{0},
        i
    ) where {T}
    return convert(T, u[i..., out_index...])
end

@inline function local_control_point(
        u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, ::Type{T}, ::Val{d},
        i
    ) where {T, d}
    return _differenced_contraction(
        u, interp_dims, idx, cp_index, derivative_orders, weights, out_index, T, Val(d), i
    )
end

# Difference the local control points `derivative_order` times and contract them with `weights`
function contract_differenced(
        control_points::Tuple, knots, i_first, degree, derivative_order, weights, ::Type{T}
    ) where {T}
    for r in 1:derivative_order
        control_points = difference_control_points(control_points, knots, i_first, degree, r)
    end
    out = zero(T)
    for j in (derivative_order + 1):(degree + 1)
        out += weights[j] * control_points[j]
    end
    return out
end

# One step of the recursion for the control points of the `r`-th derivative. The entry `j`
# corresponds to the control point with index `i_first + j - 1`, the first `r` entries are unused.
function difference_control_points(control_points::Tuple, knots, i_first, degree, r)
    q = degree - r + 1
    return ntuple(length(control_points)) do j
        P = control_points[j]
        j ≤ r && return P
        i = i_first + j - 1
        Δt = knots[i + q] - knots[i]
        ΔP = P - control_points[j - 1]
        # Δt = 0 only for basis functions that are identically zero
        q * (iszero(Δt) ? zero(ΔP) / oneunit(Δt) : ΔP / Δt)
    end
end

"""
    NURBSWeights(weights::AbstractArray)

Weights associated with the control points to define a NURBS geometry.
"""
struct NURBSWeights{W <: AbstractArray} <: AbstractInterpolationCache
    weights::W
end
