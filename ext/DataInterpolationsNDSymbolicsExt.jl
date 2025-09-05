module DataInterpolationsNDSymbolicsExt

using DataInterpolationsND: NDInterpolation
using Symbolics
using Symbolics: Num, unwrap, SymbolicUtils

# Register just one symbolic function - the promote_symtype is handled by the macro
@register_symbolic (interp::NDInterpolation)(t::Real)

Base.nameof(interp::NDInterpolation) = :NDInterpolation

# Add method to handle multiple arguments symbolically  
function (interp::NDInterpolation)(args::Vararg{Num})
    unwrapped_args = unwrap.(args)
    Symbolics.wrap(SymbolicUtils.term(interp, unwrapped_args...))
end

# Handle direct differentiation of interpolation objects with respect to individual arguments
function Symbolics.derivative(interp::NDInterpolation, args::NTuple{N, Any}, ::Val{I}) where {N, I}
    # Create a symbolic term representing the partial derivative
    # The I-th argument gets differentiated (1-indexed)
    derivative_orders = ntuple(j -> j == I ? 1 : 0, N)
    
    # Create a symbolic function call that represents this partial derivative
    # We'll use a custom function name to distinguish it from the base interpolation
    symbolic_args = Symbolics.wrap.(args)
    Symbolics.unwrap(
        SymbolicUtils.term(
            PartialDerivative{I}(interp), 
            unwrap.(symbolic_args)...
        )
    )
end

# Define a partial derivative wrapper type to carry the differentiation information
struct PartialDerivative{I}
    interp::NDInterpolation
end

# Make the partial derivative callable
function (pd::PartialDerivative{I})(args...) where {I}
    derivative_orders = ntuple(j -> j == I ? 1 : 0, length(args))
    pd.interp(args...; derivative_orders = derivative_orders)
end

# Promote symtype for partial derivatives
SymbolicUtils.promote_symtype(::PartialDerivative, _...) = Real

# Name the partial derivative functions appropriately
Base.nameof(pd::PartialDerivative{I}) where {I} = Symbol("∂$(I)_NDInterpolation")

# Handle higher-order derivatives by chaining partial derivatives
function Symbolics.derivative(pd::PartialDerivative{J}, args::NTuple{N, Any}, ::Val{I}) where {J, N, I}
    # Create a new partial derivative that represents higher-order differentiation
    new_pd = MixedPartialDerivative(pd.interp, (J, I))
    symbolic_args = Symbolics.wrap.(args)
    Symbolics.unwrap(
        SymbolicUtils.term(
            new_pd,
            unwrap.(symbolic_args)...
        )
    )
end

# Define mixed partial derivatives for higher-order cases
struct MixedPartialDerivative
    interp::NDInterpolation
    orders::Tuple{Vararg{Int}}
end

# Make mixed partial derivatives callable
function (mpd::MixedPartialDerivative)(args...)
    derivative_orders = ntuple(length(args)) do j
        count(==(j), mpd.orders)
    end
    mpd.interp(args...; derivative_orders = derivative_orders)
end

# Promote symtype for mixed partial derivatives
SymbolicUtils.promote_symtype(::MixedPartialDerivative, _...) = Real

# Name mixed partial derivatives
function Base.nameof(mpd::MixedPartialDerivative)
    orders_str = join(mpd.orders, "_")
    Symbol("∂$(orders_str)_NDInterpolation")
end

# Handle further differentiation of mixed partial derivatives
function Symbolics.derivative(mpd::MixedPartialDerivative, args::NTuple{N, Any}, ::Val{I}) where {N, I}
    new_mpd = MixedPartialDerivative(mpd.interp, (mpd.orders..., I))
    symbolic_args = Symbolics.wrap.(args)
    Symbolics.unwrap(
        SymbolicUtils.term(
            new_mpd,
            unwrap.(symbolic_args)...
        )
    )
end

end # module