using DataInterpolationsND
using DataInterpolationsND: get_n_basis_functions
using Adapt
using CUDA
using Random
using Test

evaluate(itp, derivative_orders, t...) = itp(t...; derivative_orders)

# Values plus partial derivatives of order 1, mixed order 2 (order 2 in 1D) and order 3
const DERIVATIVE_ORDERS = (
    ((0,), (1,), (2,), (3,)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)),
)

if !CUDA.functional()
    @info "CUDA is not functional, skipping the GPU tests"
    @test_skip false
else
    CUDA.allowscalar(false)

    @testset "$(N_in)D cubic BSpline" for N_in in 1:3
        rng = Xoshiro(N_in)
        bounds = ((0.0, 1.0), (-2.0, 3.0), (1.0, 4.5))[1:N_in]
        dims = map(bounds) do (lo, hi)
            # Non-uniform knots on [lo, hi]
            t = sort!(lo .+ (hi - lo) .* rand(rng, 7))
            t[1], t[end] = lo, hi
            BSplineInterpolationDimension(t, 3)
        end
        itp = NDInterpolation(rand(rng, map(get_n_basis_functions, dims)...), dims)
        # Random points in range, every knot, and one out-of-range point
        ts = map(bounds, dims) do (lo, hi), dim
            vcat(lo .+ (hi - lo) .* rand(rng, 1000), dim.t, hi + (hi - lo) / 10)
        end

        itp_gpu = adapt(CuArray, itp)
        ts_gpu = map(CuArray, ts)
        for derivative_orders in DERIVATIVE_ORDERS[N_in]
            out_gpu = evaluate.(Ref(itp_gpu), Ref(derivative_orders), ts_gpu...)
            out_cpu = evaluate.(Ref(itp), Ref(derivative_orders), ts...)
            # The NVPTX back-end contracts multiply-add pairs into FMA instructions, which
            # the CPU does not, so the results can differ in the last bits.
            @test maximum(abs, Array(out_gpu) - out_cpu) ≤ 64 * eps(maximum(abs, out_cpu))
        end
    end
end
