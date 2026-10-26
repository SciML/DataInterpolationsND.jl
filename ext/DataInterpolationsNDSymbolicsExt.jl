module DataInterpolationsNDSymbolicsExt

import DataInterpolationsND
using DataInterpolationsND: NDInterpolation
using Symbolics: Symbolics, Num, @register_derivative
import SymbolicUtils
using SymbolicUtils: unwrap

struct DifferentiatedNDInterpolation{N_in, N_out, I <: NDInterpolation{N_in, N_out}}
    interp::I
    derivative_orders::NTuple{N_in, Int}
end

function (interp::DifferentiatedNDInterpolation)(args...)
    return interp.interp(args; derivative_orders = interp.derivative_orders)
end

Base.nameof(::NDInterpolation) = :NDInterpolation
Base.nameof(::DifferentiatedNDInterpolation) = :DifferentiatedNDInterpolation

const SymbolicNDInterpolation = Union{
    NDInterpolation{N_in, N_out},
    DifferentiatedNDInterpolation{N_in, N_out},
} where {N_in, N_out}

base_interp(interp::NDInterpolation) = interp
base_interp(interp::DifferentiatedNDInterpolation) = interp.interp

function output_shape(interp, ::Val{N_out}) where {N_out}
    return if N_out == 0
        SymbolicUtils.ShapeVecT()
    else
        sz = DataInterpolationsND.get_output_size(base_interp(interp))
        SymbolicUtils.ShapeVecT(map(n -> 1:n, sz))
    end
end

for interpT in [NDInterpolation, DifferentiatedNDInterpolation],
        symT in [Num, Symbolics.SymbolicT]

    @eval function (interp::$interpT{N_in, N_out})(
            t::Vararg{
                $symT, N_in,
            }
        ) where {N_in, N_out}
        if $(symT === Num)
            t = unwrap.(t)
        end
        res = SymbolicUtils.term(
            interp, t...;
            type = N_out == 0 ? Real : Array{Real, N_out},
            shape = output_shape(interp, Val(N_out))
        )
        if $(symT === Num)
            if N_out == 0
                res = Num(res)
            else
                res = Symbolics.Arr{Num, N_out}(res)
            end
        end
        return res
    end
end

function SymbolicUtils.promote_symtype(
        ::SymbolicNDInterpolation{N_in, N_out}, ::Vararg
    ) where {N_in, N_out}
    return N_out == 0 ? Real : Array{Real, N_out}
end

function SymbolicUtils.promote_shape(
        interp::SymbolicNDInterpolation{N_in, N_out}, ::SymbolicUtils.ShapeT...
    ) where {N_in, N_out}
    return output_shape(interp, Val(N_out))
end

@register_derivative (interp::NDInterpolation)(args...) I begin
    orders = ntuple(Int ∘ isequal(I), Val{Nargs}())
    DifferentiatedNDInterpolation(interp, orders)(args...)
end

@register_derivative (interp::DifferentiatedNDInterpolation)(args...) I begin
    orders_offset = ntuple(Int ∘ isequal(I), Val{Nargs}())
    orders = interp.derivative_orders .+ orders_offset
    typeof(interp)(interp.interp, orders)(args...)
end

# Partial derivative of the `NDInterpolation` passed as the first argument, which lets
# the interpolation itself be symbolic. `output_shape` is the shape of the result.
struct NDPartialDerivative{N_in, S}
    derivative_orders::NTuple{N_in, Int}
    output_shape::S
end

function (d::NDPartialDerivative)(interp::NDInterpolation, args::Number...)
    return interp(args; derivative_orders = d.derivative_orders)
end

function (d::NDPartialDerivative)(interp, args...)
    return SymbolicUtils.term(
        d, interp, args...;
        type = SymbolicUtils.promote_symtype(d, SymbolicUtils.symtype(interp)),
        shape = SymbolicUtils.promote_shape(d)
    )
end

Base.nameof(::NDPartialDerivative) = :NDPartialDerivative

interp_output_shape(interp::NDInterpolation{N_in, N_out}) where {N_in, N_out} = output_shape(interp, Val(N_out))
interp_output_shape(interp) = SymbolicUtils.shape(interp)

function partial_derivative(derivative_orders, interp)
    sh = interp_output_shape(interp)
    return NDPartialDerivative(derivative_orders, sh isa SymbolicUtils.ShapeVecT ? Tuple(sh) : sh)
end

output_symtype(::Type{<:NDInterpolation{N_in, N_out}}) where {N_in, N_out} = N_out == 0 ? Real : Array{Real, N_out}
output_symtype(T::Type{<:SymbolicUtils.FnType}) = SymbolicUtils.fntype_ret_type(T)

function SymbolicUtils.promote_symtype(::NDPartialDerivative, T::SymbolicUtils.TypeT, ::Vararg)
    return output_symtype(T)
end

function SymbolicUtils.promote_shape(d::NDPartialDerivative, ::SymbolicUtils.ShapeT...)
    sh = d.output_shape
    return sh isa Tuple ? SymbolicUtils.ShapeVecT(sh) : sh
end

@register_derivative (interp::Symbolics.SymbolicCallable{<:NDInterpolation})(args...) I begin
    partial_derivative(ntuple(Int ∘ isequal(I), Val{Nargs}()), interp.f)(interp.f, args...)
end

@register_derivative (d::NDPartialDerivative)(args...) I begin
    if I == 1
        nothing
    else
        orders_offset = ntuple(Int ∘ isequal(I - 1), Val{Nargs - 1}())
        NDPartialDerivative(d.derivative_orders .+ orders_offset, d.output_shape)(args...)
    end
end

end # module
