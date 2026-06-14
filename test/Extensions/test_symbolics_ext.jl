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
    @test SU.symtype(ex) == Vector{Real}
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

# Scalar output (N_out == 0), e.g. a 2D table of scalars as used with
# ModelingToolkit callable parameters (issue #69)
u_scalar = rand(5, 7)
interp_scalar = NDInterpolation(u_scalar, interpolation_dimensions)

@testset "Scalar output basics" begin
    ex = interp_scalar(x, y)
    @test ex isa Num
    @test SU.symtype(unwrap(ex)) == Real
    @test !SU.is_array_shape(SU.shape(unwrap(ex)))

    res = eval(
        quote
            let x = 0.4, y = 0.8
                $(SU.Code.toexpr(unwrap(ex)))
            end
        end
    )
    @test res ≈ interp_scalar(0.4, 0.8)
end

@testset "Scalar output differentiation" begin
    ex = interp_scalar(x, y)

    for (var, orders) in ((x, (1, 0)), (y, (0, 1)))
        der = expand_derivatives(Differential(var)(ex))
        @test SU.symtype(unwrap(der)) == Real
        # the derivative must be fully expanded, with no `Differential` left over
        @test !Symbolics.hasderiv(unwrap(der))
        res = eval(
            quote
                let x = 0.4, y = 0.8
                    $(SU.Code.toexpr(unwrap(der)))
                end
            end
        )
        @test res ≈ interp_scalar(0.4, 0.8; derivative_orders = orders)
    end

    # chain rule through both arguments
    @variables s
    ex_s = interp_scalar(2.0 * s, s)
    der = expand_derivatives(Differential(s)(ex_s))
    res = eval(
        quote
            let s = 0.4
                $(SU.Code.toexpr(unwrap(der)))
            end
        end
    )
    @test res ≈ 2.0 * interp_scalar(0.8, 0.4; derivative_orders = (1, 0)) +
        interp_scalar(0.8, 0.4; derivative_orders = (0, 1))

    # second derivatives differentiate the `DifferentiatedNDInterpolation`
    der2 = expand_derivatives(Differential(y)(Differential(x)(ex)))
    res = eval(
        quote
            let x = 0.4, y = 0.8
                $(SU.Code.toexpr(unwrap(der2)))
            end
        end
    )
    @test res ≈ interp_scalar(0.4, 0.8; derivative_orders = (1, 1))
end
