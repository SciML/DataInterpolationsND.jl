using Random
using ForwardDiff
using DataInterpolationsND: AbstractInterpolationDimension, EmptyCache
using Symbolics
import SymbolicUtils as SU
using Symbolics: unwrap

###
#### Interpolations
###

function test_globally_constant(
    ID::Type{<:AbstractInterpolationDimension}; args1=(), args2=(), kwargs1=(),
    kwargs2=(), cache=EmptyCache(), test_derivatives=true
)
    t1 = [-3.14, 1.0, 3.0, 7.6, 12.8]
    t2 = [-2.71, 1.41, 12.76, 50.2, 120.0]

    u = if ID == BSplineInterpolationDimension
        fill(2.0, 4 + args1[1], 4 + args2[1])
    else
        fill(2.0, 5, 5)
    end

    # Evaluation in data points
    itp_dims = (
        ID(t1, args1...; t_eval=t1, kwargs1...),
        ID(t2, args2...; t_eval=t2, kwargs2...),
    )

    itp = NDInterpolation(u, itp_dims; cache)
    @test all(x -> isapprox(x, 2.0; atol=1.0e-10), eval_grid(itp))
    if test_derivatives
        @test all(
            x -> isapprox(x, 0.0; atol=1.0e-10), eval_grid(itp, derivative_orders=(1, 0))
        )
        @test all(
            x -> isapprox(x, 0.0; atol=1.0e-10), eval_grid(itp, derivative_orders=(0, 1))
        )
    end

    # Evaluation between data points
    itp_dims = (
        ID(t1, args1...; t_eval=t1[1:(end-1)] + diff(t1) / 2, kwargs1...),
        ID(t2, args2...; t_eval=t2[1:(end-1)] + diff(t2) / 2, kwargs2...),
    )
    itp = NDInterpolation(u, itp_dims)
    @test all(x -> isapprox(x, 2.0; atol=1.0e-10), eval_grid(itp))
    return if test_derivatives
        @test all(
            x -> isapprox(x, 0.0; atol=1.0e-10), eval_grid(itp, derivative_orders=(1, 0))
        )
        @test all(
            x -> isapprox(x, 0.0; atol=1.0e-10), eval_grid(itp, derivative_orders=(0, 1))
        )
    end
end

function test_analytic(itp::NDInterpolation{N_in}, f) where {N_in}
    # Evaluation in data points
    ts = ntuple(dim_in -> itp.interp_dims[dim_in].t, N_in)
    for t in Iterators.product(ts...)
        @test itp(t) ≈ f(t...)
    end

    # Evaluation between data points
    ts_ = ntuple(dim_in -> ts[dim_in][1:(end-1)] + diff(ts[dim_in]) / 2, N_in)
    for t in Iterators.product(ts_...)
        @test itp(t) ≈ f(t...)
    end
    return
end

###
#### Symbolics
###

function get_interp()
    t1 = cumsum(rand(5))
    t2 = cumsum(rand(7))

    interpolation_dimensions = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2),
    )

    u = rand(5, 7, 2)

    return NDInterpolation(u, interpolation_dimensions)
end