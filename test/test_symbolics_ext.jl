using DataInterpolationsND
using Symbolics
import SymbolicUtils as SU
using Symbolics: unwrap
using Test

t1 = cumsum(rand(5))
t2 = cumsum(rand(7))

interpolation_dimensions = (
    LinearInterpolationDimension(t1),
    LinearInterpolationDimension(t2),
)

u = rand(5, 7, 2)

interp = NDInterpolation(u, interpolation_dimensions)
@variables x y

@testset "Basics" begin
    ex = interp(x, y)
    @test ex isa Symbolics.Arr
    @test size(ex) == (2,)
    @test SU.symtype(unwrap(ex)) == Vector{Real}

    res = eval(
        quote
            let x = 0.4, y = 0.8
                $(SU.Code.toexpr(ex))
            end
        end
    )
    @test res ≈ interp(0.4, 0.8)

    ex = interp(unwrap(x), unwrap(y))
    @test ex isa SU.BasicSymbolic{Vector{Real}}
end

@testset "Differentiation" begin
    ex = interp(x, y)
    der = Symbolics.derivative(ex[1], x)
    @test size(der) == ()
    @test SU.symtype(unwrap(der)) == Real
    res = eval(
        quote
            let x = 0.4, y = 0.8
                $(SU.Code.toexpr(der))
            end
        end
    )
    @test res ≈ interp(0.4, 0.8; derivative_orders = (1, 0))[1]

    der = Symbolics.derivative(ex[1], y)
    @test size(der) == ()
    @test SU.symtype(unwrap(der)) == Real
    res = eval(
        quote
            let x = 0.4, y = 0.8
                $(SU.Code.toexpr(der))
            end
        end
    )
    @test res ≈ interp(0.4, 0.8; derivative_orders = (0, 1))[1]
end
