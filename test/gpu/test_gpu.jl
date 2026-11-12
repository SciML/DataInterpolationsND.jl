using DataInterpolationsND
using DataInterpolationsND: get_n_basis_functions
using Adapt
using CUDA
using Random
using Test

evaluate(itp, derivative_orders, t...) = itp(t...; derivative_orders)

# Non-uniform knots on [lo, hi]
function make_knots(rng, lo, hi, n)
    t = sort!(lo .+ (hi - lo) .* rand(rng, n))
    t[1], t[end] = lo, hi
    return t
end

function cubic_bspline(rng, N_in)
    bounds = ((0.0, 1.0), (-2.0, 3.0), (1.0, 4.5))[1:N_in]
    dims = ntuple(i -> BSplineInterpolationDimension(make_knots(rng, bounds[i]..., 7), 3), N_in)
    u = rand(rng, map(get_n_basis_functions, dims)...)
    return NDInterpolation(u, dims), bounds
end

# Evaluation points: random points slightly beyond the knot range (to also cover
# extrapolation), plus every knot so that the knot span lookup is exercised at the boundaries
function eval_points(rng, itp, bounds, n)
    return ntuple(length(bounds)) do i
        lo, hi = bounds[i]
        knots = itp.interp_dims[i].t
        pad = (hi - lo) / 10
        vcat(lo - pad .+ (hi - lo + 2pad) .* rand(rng, n), knots)
    end
end

# All partial derivative orders with a total order of at most 3
derivative_orders_upto3(N_in) = filter(d -> sum(d) ≤ 3, vec(collect(Iterators.product(ntuple(_ -> 0:3, N_in)...))))

if !CUDA.functional()
    @info "CUDA is not functional, skipping the GPU tests"
    @test_skip false
else
    CUDA.allowscalar(false)

    @testset "Cubic BSpline, N_in = $N_in" for N_in in 1:3
        rng = Xoshiro(N_in)
        itp, bounds = cubic_bspline(rng, N_in)
        ts = eval_points(rng, itp, bounds, 1000)

        itp_gpu = adapt(CuArray, itp)
        @test itp_gpu.u isa CuArray{Float64}
        @test all(dim -> dim.knots_all isa CuArray{Float64}, itp_gpu.interp_dims)
        @test isbits(cudaconvert(itp_gpu))
        ts_gpu = map(CuArray, ts)

        @testset "derivative_orders = $derivative_orders" for derivative_orders in derivative_orders_upto3(N_in)
            out_gpu = evaluate.(Ref(itp_gpu), Ref(derivative_orders), ts_gpu...)
            @test out_gpu isa CuArray{Float64}
            out_cpu = evaluate.(Ref(itp), Ref(derivative_orders), ts...)
            # The NVPTX back-end contracts multiply-add pairs into FMA instructions, which
            # the CPU does not, so the results can differ in the last bits.
            @test maximum(abs, Array(out_gpu) - out_cpu) ≤ 64 * eps(maximum(abs, out_cpu))
        end
    end
end
