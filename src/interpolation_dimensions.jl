"""
    LinearInterpolationDimension(t; t_eval = similar(t, 0))

Interpolation dimension for linear interpolation between the data points.
Both sides default to `ExtrapolationType.Linear`.

## Arguments

  - `t`: The time points for this interpolation dimension.

## Keyword arguments

  - `extrapolation`: An [`ExtrapolationType`](@ref) mode for both sides, overriding the side keywords.
  - `extrapolation_left`, `extrapolation_right`: Independent extrapolation modes for each side.
  - `t_eval`: A vector (like) of time evaluation points for efficient evaluation of multiple points,
    see `eval_unstructured` and `eval_grid`. Defaults to no points.
"""
struct LinearInterpolationDimension{
        tType <: AbstractVector{<:Number},
        t_evalType <: AbstractVector{<:Number},
        idxType <: AbstractVector{<:Integer},
    } <: AbstractInterpolationDimension
    t::tType
    t_eval::t_evalType
    idx_eval::idxType
    extrapolation_left::ExtrapolationType.T
    extrapolation_right::ExtrapolationType.T
    function LinearInterpolationDimension(
            t, t_eval, idx_eval,
            extrapolation_left::ExtrapolationType.T = ExtrapolationType.Linear,
            extrapolation_right::ExtrapolationType.T = ExtrapolationType.Linear
        )
        validate_t(t)
        return new{typeof(t), typeof(t_eval), typeof(idx_eval)}(t, t_eval, idx_eval, extrapolation_left, extrapolation_right)
    end
end

function adapt_structure(to, interp_dim::LinearInterpolationDimension)
    return LinearInterpolationDimension(
        adapt_structure(to, interp_dim.t),
        adapt_structure(to, interp_dim.t_eval),
        adapt_structure(to, interp_dim.idx_eval),
        interp_dim.extrapolation_left, interp_dim.extrapolation_right,
    )
end

function LinearInterpolationDimension(
        t; t_eval = similar(t, 0),
        extrapolation::Union{Nothing, ExtrapolationType.T} = nothing,
        extrapolation_left::ExtrapolationType.T = ExtrapolationType.Linear,
        extrapolation_right::ExtrapolationType.T = ExtrapolationType.Linear
    )
    extrapolation_left, extrapolation_right = extrapolation_modes(
        extrapolation, extrapolation_left, extrapolation_right
    )
    idx_eval = similar(t_eval, Int)
    itp_dim = LinearInterpolationDimension(
        t, t_eval, idx_eval, extrapolation_left, extrapolation_right
    )
    set_eval_idx!(itp_dim)
    return itp_dim
end

"""
    ConstantInterpolationDimension(t; t_eval = similar(t, 0), left = true)

Interpolation dimension for constant interpolation between the data points.
Both sides default to `ExtrapolationType.Constant`; all supported modes hold the edge value.

## Arguments

  - `t`: The time points for this interpolation dimension.

## Keyword arguments

  - `left`: Whether the interpolation looks to the left of the evaluation `t` for the value. Defaults to `true`.
  - `extrapolation`: An [`ExtrapolationType`](@ref) mode for both sides, overriding the side keywords.
  - `extrapolation_left`, `extrapolation_right`: Independent extrapolation modes for each side.
  - `t_eval`: A vector (like) of time evaluation points for efficient evaluation of multiple points,
    see `eval_unstructured` and `eval_grid`. Defaults to no points.
"""
struct ConstantInterpolationDimension{
        tType <: AbstractVector{<:Number},
        t_evalType <: AbstractVector{<:Number},
        idxType <: AbstractVector{<:Integer},
    } <: AbstractInterpolationDimension
    t::tType
    left::Bool
    t_eval::t_evalType
    idx_eval::idxType
    extrapolation_left::ExtrapolationType.T
    extrapolation_right::ExtrapolationType.T
    function ConstantInterpolationDimension(
            t, left, t_eval, idx_eval,
            extrapolation_left::ExtrapolationType.T = ExtrapolationType.Constant,
            extrapolation_right::ExtrapolationType.T = ExtrapolationType.Constant
        )
        validate_t(t)
        return new{typeof(t), typeof(t_eval), typeof(idx_eval)}(
            t, left, t_eval, idx_eval, extrapolation_left, extrapolation_right
        )
    end
end

function adapt_structure(to, interp_dim::ConstantInterpolationDimension)
    return ConstantInterpolationDimension(
        adapt_structure(to, interp_dim.t),
        adapt_structure(to, interp_dim.left),
        adapt_structure(to, interp_dim.t_eval),
        adapt_structure(to, interp_dim.idx_eval),
        interp_dim.extrapolation_left, interp_dim.extrapolation_right,
    )
end

function ConstantInterpolationDimension(
        t; left = true, t_eval = similar(t, 0),
        extrapolation::Union{Nothing, ExtrapolationType.T} = nothing,
        extrapolation_left::ExtrapolationType.T = ExtrapolationType.Constant,
        extrapolation_right::ExtrapolationType.T = ExtrapolationType.Constant
    )
    extrapolation_left, extrapolation_right = extrapolation_modes(
        extrapolation, extrapolation_left, extrapolation_right
    )
    idx_eval = similar(t_eval, Int)
    itp_dim = ConstantInterpolationDimension(
        t, left, t_eval, idx_eval, extrapolation_left, extrapolation_right
    )
    set_eval_idx!(itp_dim)
    return itp_dim
end

"""
    BSplineInterpolationDimension(
        t, degree;
        t_eval = similar(t, 0),
        max_derivative_order_eval::Integer = 0,
        multiplicities::Union{AbstractVector{<:Integer}, Nothing} = nothing)

Interpolation dimension for BSpline or NURBS interpolation between the data points, used for evaluating
the BSpline basis functions. Both sides default to `ExtrapolationType.Extension`.
`ExtrapolationType.Linear` extends the boundary tangent and is not supported with `NURBSWeights`.

## Arguments

  - `t`: The time points for this interpolation dimension, in this context also known as knots.
  - `degree`: The degree of the basis functions

## Keyword arguments

  - `extrapolation`: An [`ExtrapolationType`](@ref) mode for both sides, overriding the side keywords.
  - `extrapolation_left`, `extrapolation_right`: Independent extrapolation modes for each side.
  - `t_eval`: A vector (like) of time evaluation points for efficient evaluation of multiple points,
    see `eval_unstructured` and `eval_grid`. Defaults to no points.
  - `max_derivative_order_eval`: The maximum derivative order for which the basis functions will be precomputed
    for `eval_unstructured` and `eval_grid`.
  - `multiplicities`: The multiplicities of the knots `t`. Defaults to multiplicities for an open/clamped knot vector.
"""
struct BSplineInterpolationDimension{
        tType <: AbstractVector{<:Number},
        t_evalType <: AbstractVector{<:Number},
        idxType <: AbstractVector{<:Integer},
        evalType <: AbstractArray{<:Number, 3},
        mType <: AbstractVector{<:Integer},
    } <: AbstractInterpolationDimension
    t::tType
    knots_all::tType
    t_eval::t_evalType
    idx_eval::idxType
    degree::Int
    max_derivative_order_eval::Int
    basis_function_eval::evalType
    multiplicities::mType
    extrapolation_left::ExtrapolationType.T
    extrapolation_right::ExtrapolationType.T
    function BSplineInterpolationDimension(
            t, knots_all, t_eval, idx_eval, degree, max_derivative_order_eval,
            basis_function_eval, multiplicities,
            extrapolation_left::ExtrapolationType.T = ExtrapolationType.Extension,
            extrapolation_right::ExtrapolationType.T = ExtrapolationType.Extension
        )
        validate_t(t)
        return new{
            typeof(t), typeof(t_eval), typeof(idx_eval),
            typeof(basis_function_eval), typeof(multiplicities),
        }(
            t, knots_all, t_eval, idx_eval, degree, max_derivative_order_eval,
            basis_function_eval, multiplicities, extrapolation_left, extrapolation_right
        )
    end
end

function adapt_structure(to, interp_dim::BSplineInterpolationDimension)
    return BSplineInterpolationDimension(
        adapt_structure(to, interp_dim.t),
        adapt_structure(to, interp_dim.knots_all),
        adapt_structure(to, interp_dim.t_eval),
        adapt_structure(to, interp_dim.idx_eval),
        adapt_structure(to, interp_dim.degree),
        adapt_structure(to, interp_dim.max_derivative_order_eval),
        adapt_structure(to, interp_dim.basis_function_eval),
        adapt_structure(to, interp_dim.multiplicities),
        interp_dim.extrapolation_left, interp_dim.extrapolation_right,
    )
end

function BSplineInterpolationDimension(
        t, degree;
        t_eval = similar(t, 0),
        max_derivative_order_eval::Integer = 0,
        multiplicities::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        extrapolation::Union{Nothing, ExtrapolationType.T} = nothing,
        extrapolation_left::ExtrapolationType.T = ExtrapolationType.Extension,
        extrapolation_right::ExtrapolationType.T = ExtrapolationType.Extension
    )
    extrapolation_left, extrapolation_right = extrapolation_modes(
        extrapolation, extrapolation_left, extrapolation_right
    )
    if isnothing(multiplicities)
        # Multiplicities for open/clamped knot vector if no multiplicities are provided
        multiplicities = similar(t, Int)
        multiplicities .= 1
        multiplicities[[1, length(multiplicities)]] .= degree + 1
    end
    @assert length(multiplicities) == length(t) "There must be the same amount of knots (points in t) as multiplicities."
    @assert all(m -> 1 ≤ m ≤ (degree + 1), multiplicities) "All multiplicities must be between 1 and degree + 1."

    knots_all = similar(t, sum(multiplicities))
    backend = get_backend(t)
    expand_knot_vector_kernel(backend)(
        knots_all,
        t,
        multiplicities,
        ndrange = length(t)
    )
    synchronize(backend)

    idx_eval = similar(t_eval, Int)
    basis_function_eval = similar(
        t_eval,
        typeof(inv(one(eltype(t))) * inv(one(eltype(t_eval)))),
        (
            length(t_eval),
            degree + 1,
            max_derivative_order_eval + 1,
        )
    )
    itp_dim = BSplineInterpolationDimension(
        t, knots_all, t_eval, idx_eval, degree, max_derivative_order_eval,
        basis_function_eval, multiplicities, extrapolation_left, extrapolation_right
    )
    set_eval_idx!(itp_dim)
    set_basis_function_eval!(itp_dim)
    return itp_dim
end
