"""
    ExtrapolationType

Extrapolation modes for each interpolation dimension, using the naming convention of
DataInterpolations.jl. The `extrapolation::ExtrapolationType.T` keyword sets the mode
for both sides of a dimension. Different dimensions can use different modes; when several
dimensions are out of range at once, the per-dimension rules compose as a tensor product.

  - `ExtrapolationType.None`: throw an [`DataInterpolationsND.ExtrapolationError`](@ref)
    when evaluating outside the grid.
  - `ExtrapolationType.Constant`: hold the boundary value. Derivatives along an axis
    strictly outside its grid are zero; derivatives at the boundary use the interpolation.
  - `ExtrapolationType.Linear`: continue with the boundary value and the boundary slope,
    `f(x_b) + ∂f/∂xᵢ(x_b) (xᵢ - x_bᵢ)`. This is the default for linear dimensions.
    For constant dimensions the slope is zero. Not supported with `NURBSWeights`.
  - `ExtrapolationType.Extension`: extend the boundary interpolation polynomial.
    This is the default for spline dimensions. Constant dimensions default to `Constant`.
  - `ExtrapolationType.Periodic`: wrap the query point periodically onto the grid.
  - `ExtrapolationType.Reflective`: reflect the query point back onto the grid. Odd
    derivatives along the axis change sign on mirrored copies.

For `Periodic` and `Reflective`, the phase is lost once the period is below the float spacing at the query's magnitude.

The mode type is `ExtrapolationType.T`.
"""
module ExtrapolationType
    "Enum type for the extrapolation modes in [`ExtrapolationType`](@ref)."
    @enum T Constant Linear Extension None Periodic Reflective
    @doc "Hold the boundary value outside the grid, with zero derivatives along the held axis." Constant
    @doc "Continue with the boundary value and boundary slope outside the grid; unsupported with `NURBSWeights`." Linear
    @doc "Continue the boundary interpolation polynomial outside the grid." Extension
    @doc "Throw a `DataInterpolationsND.ExtrapolationError` when evaluating outside the grid." None
    @doc "Wrap the query point periodically onto the grid." Periodic
    @doc "Reflect the query point back onto the grid; odd derivatives change sign on mirrored copies." Reflective
    export T, Constant, Linear, Extension, None, Periodic, Reflective
end

"""
    DataInterpolationsND.ExtrapolationError(dim::Int, left::Bool)

Thrown when an interpolation is evaluated outside the grid of interpolation dimension
`dim` and that dimension uses `ExtrapolationType.None`. `left` is `true` if the query
point was below the first grid point and `false` if it was above the last one.
"""
struct ExtrapolationError <: Exception
    dim::Int
    left::Bool
end

function Base.showerror(io::IO, e::ExtrapolationError)
    return print(
        io, "ExtrapolationError: cannot extrapolate ",
        e.left ? "below the first" : "above the last",
        " point of interpolation dimension ", e.dim,
        " because its extrapolation is `ExtrapolationType.None`."
    )
end

# Boundary handling inside the evaluation methods. `None`, `Periodic` and `Reflective` are
# resolved before evaluation (by `check_extrapolation` and `fold_extrapolation`), so any
# remaining out-of-range query for those modes only comes from rounding and is treated as
# `Extension`.
function extrapolation_boundary(dim, t)
    value = search_value(t)
    mode = dim.extrapolation
    mode = mode == ExtrapolationType.Constant || mode == ExtrapolationType.Linear ? mode :
        ExtrapolationType.Extension
    return if value < first(dim.t)
        first(dim.t), mode, true
    elseif value > last(dim.t)
        last(dim.t), mode, true
    else
        value, ExtrapolationType.Extension, false
    end
end

function hold_edge(dim, t)
    boundary, mode, outside = extrapolation_boundary(dim, t)
    hold = outside && mode == ExtrapolationType.Constant
    return hold ? boundary + zero(t) : t, hold
end

"""
    fold_extrapolation(dim, t) -> (t_folded, keep_sign)

Map `t` onto the grid of `dim` for `Periodic` and `Reflective` extrapolation; other modes
return `t` unchanged (promoted with the grid element type). `keep_sign` is `false` on
mirrored copies of a reflective dimension, where odd derivatives change sign.
"""
function fold_extrapolation(dim::AbstractInterpolationDimension, t::Number)
    # Work in floating point so that unsigned grids do not wrap around when subtracting
    t0 = float(first(dim.t))
    t1 = float(last(dim.t))
    mode = dim.extrapolation
    value = search_value(t)
    if !(mode == ExtrapolationType.Periodic || mode == ExtrapolationType.Reflective) ||
            isone(length(dim.t)) || (t0 <= value <= t1)
        return t + zero(t0), true
    end
    # A single `mod` of the offset from the grid start is exact, so far out queries keep
    # their phase. The folded point is assembled as `(t - value) + folded_value` so that
    # dual numbers keep their partials (with flipped sign on mirrored copies).
    period = t1 - t0
    offset = value - t0
    if mode == ExtrapolationType.Periodic
        return (t - value) + (t0 + mod(offset, period)), true
    end
    r = mod(offset, 2 * period)
    return if r <= period
        (t - value) + (t0 + r), true
    else
        (value - t) + (t1 - (r - period)), false
    end
end

function fold_extrapolation(
        interp_dims::NTuple{N_in, AbstractInterpolationDimension},
        t::Tuple{Vararg{Number, N_in}}
    ) where {N_in}
    folded = ntuple(i -> fold_extrapolation(interp_dims[i], t[i]), N_in)
    return ntuple(i -> folded[i][1], N_in), ntuple(i -> folded[i][2], N_in)
end

# Whether the folded derivative changes sign: odd derivative orders along mirrored axes flip it.
function fold_flips_sign(
        keep_sign::NTuple{N_in, Bool}, derivative_orders::NTuple{N_in, <:Integer}
    ) where {N_in}
    flips = false
    for i in 1:N_in
        flips ⊻= !keep_sign[i] & isodd(derivative_orders[i])
    end
    return flips
end

negate!!(out::Number) = -out
negate!!(out::AbstractArray) = (out .= .-out)

# `ExtrapolationType.None`: throw for query points outside the grid.
function check_extrapolation(
        interp_dims::NTuple{N_in, AbstractInterpolationDimension},
        t::Tuple{Vararg{Number, N_in}}
    ) where {N_in}
    for i in 1:N_in
        dim = interp_dims[i]
        dim.extrapolation == ExtrapolationType.None || continue
        value = search_value(t[i])
        value < first(dim.t) && throw(ExtrapolationError(i, true))
        value > last(dim.t) && throw(ExtrapolationError(i, false))
    end
    return nothing
end

# Host side check of the `t_eval` points for multi point evaluation. Uses reductions instead
# of scalar indexing so that it also works for GPU arrays.
function check_extrapolation_eval(
        interp_dims::NTuple{N_in, AbstractInterpolationDimension}
    ) where {N_in}
    for i in 1:N_in
        dim = interp_dims[i]
        (dim.extrapolation == ExtrapolationType.None && !isempty(dim.t_eval)) || continue
        minimum(dim.t_eval) < minimum(dim.t) && throw(ExtrapolationError(i, true))
        maximum(dim.t_eval) > maximum(dim.t) && throw(ExtrapolationError(i, false))
    end
    return nothing
end
