module DataInterpolationsNDSymbolicsExt

import DataInterpolationsND
using DataInterpolationsND: NDInterpolation
using Symbolics
using Symbolics: Num, unwrap, @register_derivative
import SymbolicUtils

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

end # module
