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
# control points, see `differenced_control_point`.
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
# This computes the control point with index `i` of the (mixed) differenced control net for
# `derivative_orders` (which must not all be zero), applying the recursion per dimension.
# `knots` and `degrees` are the `knots_all` and `degree` of the interpolation dimensions.
# It recomputes the lower order differences it needs instead of storing them, so that it does not
# need temporaries whose size depends on the degree.
@inline function differenced_control_point(
        u::AbstractArray,
        knots::NTuple{N_in, AbstractVector},
        degrees::NTuple{N_in, Int},
        i::NTuple{N_in, Int},
        derivative_orders::NTuple{N_in, Int},
        out_index::Tuple
    ) where {N_in}
    order = sum(derivative_orders)
    # Unroll the recursion for low total derivative orders, recursive function calls are slow
    order == 1 && return _differenced_control_point(
        u, knots, degrees, i, derivative_orders, out_index, Val(1)
    )
    order == 2 && return _differenced_control_point(
        u, knots, degrees, i, derivative_orders, out_index, Val(2)
    )
    order == 3 && return _differenced_control_point(
        u, knots, degrees, i, derivative_orders, out_index, Val(3)
    )
    return _differenced_control_point(u, knots, degrees, i, derivative_orders, out_index, order)
end

# `order` is the sum of `derivative_orders`, either as an `Int` or as a `Val`
@inline function _differenced_control_point(
        u, knots, degrees, i, derivative_orders, out_index, order
    )
    dim_in, derivative_orders_lower, i_prev = lower_difference_order(derivative_orders, i)
    if is_one(order)
        P = u[i..., out_index...]
        P_prev = u[i_prev..., out_index...]
    else
        order_lower = minus_one(order)
        P = _differenced_control_point(
            u, knots, degrees, i, derivative_orders_lower, out_index, order_lower
        )
        P_prev = _differenced_control_point(
            u, knots, degrees, i_prev, derivative_orders_lower, out_index, order_lower
        )
    end
    # (p - r + 1) * (P - P_prev) / (tᵢ₊ₚ₋ᵣ₊₁ - tᵢ) for derivative order r
    knots_dim = knots[dim_in]
    i_dim = i[dim_in]
    q = degrees[dim_in] - derivative_orders[dim_in] + 1
    Δt = knots_dim[i_dim + q] - knots_dim[i_dim]
    ΔP = P - P_prev
    # Δt = 0 only for basis functions that are identically zero
    return q * (iszero(Δt) ? zero(ΔP) / oneunit(Δt) : ΔP / Δt)
end

is_one(order::Int) = isone(order)
is_one(::Val{order}) where {order} = isone(order)
minus_one(order::Int) = order - 1
minus_one(::Val{order}) where {order} = Val(order - 1)

# Reduce the first nonzero derivative order by one, and give the index of the control point
# that is subtracted in that dimension
@inline function lower_difference_order(
        derivative_orders::NTuple{N_in, Int}, i::NTuple{N_in, Int}
    ) where {N_in}
    dim_in = first_nonzero_index(derivative_orders)
    derivative_orders_lower = ntuple(d -> derivative_orders[d] - (d == dim_in), Val(N_in))
    i_prev = ntuple(d -> i[d] - (d == dim_in), Val(N_in))
    return dim_in, derivative_orders_lower, i_prev
end

# Separate function so that the closures above do not capture a reassigned variable
@inline function first_nonzero_index(x::Tuple)
    i = 1
    while iszero(x[i])
        i += 1
    end
    return i
end

"""
    NURBSWeights(weights::AbstractArray)

Weights associated with the control points to define a NURBS geometry.
"""
struct NURBSWeights{W <: AbstractArray} <: AbstractInterpolationCache
    weights::W
end
