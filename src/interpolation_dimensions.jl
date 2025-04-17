"""
    LinearInterpolationDimension(t; t_eval = similar(t, 0))

Interpolation dimension for linear interpolation between the data points.

## Arguments

  - `t`: The time points for this interpolation dimension.

## Keyword arguments

  - `t_eval`: A vector (like) of time evaluation points for efficient evaluation of multiple points,
    see `eval_unstructured` and `eval_grid`. Defaults to no points.
"""
struct LinearInterpolationDimension{
    tType <: AbstractVector{<:Number},
    t_evalType <: AbstractVector{<:Number},
    idxType <: AbstractVector{<:Integer}
} <: AbstractInterpolationDimension
    t::tType
    t_eval::t_evalType
    idx_eval::idxType
    function LinearInterpolationDimension(t, t_eval, idx_eval)
        new{typeof(t), typeof(t_eval), typeof(idx_eval)}(t, t_eval, idx_eval)
    end
end

@adapt_structure LinearInterpolationDimension

function LinearInterpolationDimension(t; t_eval = similar(t, 0))
    idx_eval = similar(t_eval, Int)
    itp_dim = LinearInterpolationDimension(
        t, t_eval, idx_eval
    )
    set_eval_idx!(itp_dim)
    itp_dim
end

"""
    ConstantInterpolationDimension(t; t_eval = similar(t, 0), left = true)

Interpolation dimension for constant interpolation between the data points.

## Arguments

  - `t`: The time points for this interpolation dimension.

## Keyword arguments

  - `left`: Whether the interpolation looks to the left of the evaluation `t` for the value. Defaults to `true`.
  - `t_eval`: A vector (like) of time evaluation points for efficient evaluation of multiple points,
    see `eval_unstructured` and `eval_grid`. Defaults to no points.
"""
struct ConstantInterpolationDimension{
    tType <: AbstractVector{<:Number},
    t_evalType <: AbstractVector{<:Number},
    idxType <: AbstractVector{<:Integer}
} <: AbstractInterpolationDimension
    t::tType
    left::Bool
    t_eval::t_evalType
    idx_eval::idxType
    function ConstantInterpolationDimension(t, left, t_eval, idx_eval)
        new{typeof(t), typeof(t_eval), typeof(idx_eval)}(
            t, left, t_eval, idx_eval
        )
    end
end

@adapt_structure ConstantInterpolationDimension

function ConstantInterpolationDimension(t; left = true, t_eval = similar(t, 0))
    idx_eval = similar(t_eval, Int)
    itp_dim = ConstantInterpolationDimension(
        t, left, t_eval, idx_eval
    )
    set_eval_idx!(itp_dim)
    itp_dim
end
