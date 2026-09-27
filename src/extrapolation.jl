"""
    ExtrapolationType

Extrapolation modes for each interpolation dimension, using the naming convention of
DataInterpolations.jl. Set `extrapolation` for both sides, or set `extrapolation_left`
and `extrapolation_right` independently. An explicit `extrapolation` overrides both sides.

  - `ExtrapolationType.Constant`: hold the boundary value. Derivatives along an axis
    strictly outside its grid are zero; derivatives at the boundary use the interpolation.
  - `ExtrapolationType.Linear`: extend the boundary tangent. This is the default for
    linear dimensions. For constant dimensions the tangent is zero. Not supported with
    `NURBSWeights`.
  - `ExtrapolationType.Extension`: extend the boundary interpolation polynomial.
    This is the default for spline dimensions. Constant dimensions default to `Constant`.

The mode type is `ExtrapolationType.T`. Other DataInterpolations.jl extrapolation modes
are not supported.
"""
module ExtrapolationType
    "Enum type for the extrapolation modes in [`ExtrapolationType`](@ref)."
    @enum T Constant Linear Extension
    @doc "Hold the boundary value outside the grid, with zero derivatives along the held axis." Constant
    @doc "Extend the boundary tangent outside the grid; unsupported with `NURBSWeights`." Linear
    @doc "Continue the boundary interpolation polynomial outside the grid." Extension
    export T, Constant, Linear, Extension
end

function extrapolation_modes(extrapolation, left, right)
    return isnothing(extrapolation) ? (left, right) : (extrapolation, extrapolation)
end

function extrapolation_boundary(dim, t)
    value = search_value(t)
    return if value < first(dim.t)
        first(dim.t), dim.extrapolation_left, true
    elseif value > last(dim.t)
        last(dim.t), dim.extrapolation_right, true
    else
        value, ExtrapolationType.Extension, false
    end
end

function hold_edge(dim, t)
    boundary, mode, outside = extrapolation_boundary(dim, t)
    hold = outside && mode == ExtrapolationType.Constant
    return hold ? boundary + zero(t) : t, hold
end
