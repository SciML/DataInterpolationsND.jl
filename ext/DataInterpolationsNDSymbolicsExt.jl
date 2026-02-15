module DataInterpolationsNDSymbolicsExt

import DataInterpolationsND
using DataInterpolationsND: NDInterpolation
using Symbolics
using Symbolics: Num, unwrap
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

for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
    @eval function (interp::NDInterpolation{N_in, N_out})(
            t::Vararg{
                $symT, N_in,
            }
        ) where {N_in, N_out}
        if $(symT === Num)
            t = unwrap.(t)
        end
        res = if N_out == 0
            SymbolicUtils.term(interp, t...; type = Real)
        else
            Symbolics.array_term(
                interp, t...; eltype = Real, container_type = Array, ndims = N_out,
                size = DataInterpolationsND.get_output_size(interp)
            )
        end
        if $(symT === Num)
            if N_out == 0
                res = Num(res)
            else
                res = Symbolics.Arr{Num, N_out}(res)
            end
        end
        return res
    end
    @eval function (interp::DifferentiatedNDInterpolation{N_in, N_out})(
            t::Vararg{
                $symT, N_in,
            }
        ) where {N_in, N_out}
        if $(symT === Num)
            t = unwrap.(t)
        end
        res = if N_out == 0
            SymbolicUtils.term(interp, t...; type = Real)
        else
            Symbolics.array_term(
                interp, t...; eltype = Real, container_type = Array, ndims = N_out,
                size = DataInterpolationsND.get_output_size(interp.interp)
            )
        end
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
function SymbolicUtils.promote_symtype(::NDInterpolation{N_in, N_out}, ::Vararg) where {
        N_in, N_out,
    }
    return N_out == 0 ? Real : Array{Real, N_out}
end

function Symbolics.derivative(
        interp::NDInterpolation{N_in, N_out},
        args::NTuple{N_in, Any}, ::Val{I}
    ) where {N_in, N_out, I}
    @assert I <= N_in
    orders = ntuple(Int ∘ isequal(I), Val{N_in}())
    dinterp = DifferentiatedNDInterpolation{N_in, N_out, typeof(interp)}(interp, orders)
    return dinterp(args...)
end

function Symbolics.derivative(
        interp::DifferentiatedNDInterpolation{N_in, N_out},
        args::NTuple{N_in, Any}, ::Val{I}
    ) where {N_in, N_out, I}
    @assert I <= N_in
    orders_offset = ntuple(Int ∘ isequal(I), Val{N_in}())
    orders = interp.derivative_orders .+ orders_offset
    return typeof(interp)(interp.interp, orders)(args...)
end

end # module
