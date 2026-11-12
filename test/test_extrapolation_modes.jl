using DataInterpolationsND
using DataInterpolationsND: ExtrapolationError
import DataInterpolations
using ForwardDiff
using Test

const ET = ExtrapolationType
const DI_ET = DataInterpolations.ExtrapolationType

# Expected value of the tensor product extrapolation, built from evaluations of the
# interpolation inside (or on the boundary of) the grid. Per axis that is out of range:
#   - Constant: evaluate at the boundary, derivatives along the axis vanish;
#   - Linear: f(x_b) + ∂f/∂xᵢ(x_b) (xᵢ - x_bᵢ), so the derivative along the axis is the
#     boundary slope and higher derivatives vanish;
#   - Extension: evaluate at the point itself.
# Axes in range (and Extension axes) keep the point and derivative order unchanged.
function expected_extrapolation(interp, modes, x, orders)
    N = length(x)
    options = ntuple(N) do i
        t = interp.interp_dims[i].t
        lo, hi = first(t), last(t)
        xb = clamp(x[i], lo, hi)
        if xb == x[i] || modes[i] == ET.Extension
            [(x[i], orders[i], 1.0)]
        elseif modes[i] == ET.Constant
            iszero(orders[i]) ? [(xb, 0, 1.0)] : Tuple{Float64, Int, Float64}[]
        elseif modes[i] == ET.Linear
            if orders[i] == 0
                [(xb, 0, 1.0), (xb, 1, x[i] - xb)]
            elseif orders[i] == 1
                [(xb, 1, 1.0)]
            else
                Tuple{Float64, Int, Float64}[]
            end
        else
            error("unsupported mode in reference")
        end
    end
    result = 0.0
    for choice in Iterators.product(options...)
        point = map(first, choice)
        ords = map(c -> c[2], choice)
        coef = prod(c -> c[3], choice)
        result += coef * interp(point; derivative_orders = ords)
    end
    return result
end

# Points on each side and corner of the box `[lo, hi]^N`, plus one interior point
function side_and_corner_points(ranges)
    axes = map(ranges) do (lo, hi)
        w = hi - lo
        (lo - 0.37w, lo + 0.41w, hi + 0.53w)
    end
    return vec(collect(Iterators.product(axes...)))
end

function make_dims(kind, ts, modes)
    return ntuple(length(ts)) do i
        if kind == :linear
            LinearInterpolationDimension(ts[i]; extrapolation = modes[i])
        else
            BSplineInterpolationDimension(ts[i], kind; extrapolation = modes[i])
        end
    end
end

function n_coefficients(kind, t)
    return kind == :linear ? length(t) :
        DataInterpolationsND.get_n_basis_functions(BSplineInterpolationDimension(t, kind))
end

const TS = ([0.0, 0.3, 0.7, 1.0, 1.6], [-1.0, -0.2, 0.5, 2.0], [2.0, 2.5, 3.1, 3.4, 4.0])

coefficients(kind, ts) = [
    sin(1.3i) + cos(0.7j) * (1 + 0.1k) + 0.05i * j * k
        for i in 1:n_coefficients(kind, ts[1]),
        j in 1:n_coefficients(kind, ts[2]),
        k in 1:(length(ts) == 3 ? n_coefficients(kind, ts[3]) : 1)
]

function test_interp(kind, N, modes)
    ts = TS[1:N]
    u = coefficients(kind, ts)
    u = N == 2 ? u[:, :, 1] : u
    return NDInterpolation(u, make_dims(kind, ts, modes))
end

@testset "1D agreement with DataInterpolations" begin
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]
    # Points outside the grid on both sides, several periods away, avoiding knots after folding
    ts = [-2.35, -1.3, -0.4, -0.05, 1.17, 1.63, 2.5, 3.33]
    modes = (
        (ET.Constant, DI_ET.Constant),
        (ET.Linear, DI_ET.Linear),
        (ET.Extension, DI_ET.Extension),
        (ET.Periodic, DI_ET.Periodic),
        (ET.Reflective, DI_ET.Reflective),
    )
    for (dim_type, di_type) in (
                (LinearInterpolationDimension, DataInterpolations.LinearInterpolation),
                (ConstantInterpolationDimension, DataInterpolations.ConstantInterpolation),
            ),
            (mode, di_mode) in modes

        di = di_type(u, t; extrapolation = di_mode)
        nd = NDInterpolation(u, dim_type(t; extrapolation = mode, t_eval = ts))
        @test nd.(ts) ≈ di.(ts)
        @test eval_grid(nd) ≈ di.(ts)
        @test eval_unstructured(nd) ≈ di.(ts)
        # Derivatives of constant interpolation vanish away from the knots
        derivs = dim_type === ConstantInterpolationDimension ? zero(ts) :
            DataInterpolations.derivative.(Ref(di), ts)
        @test [nd(x; derivative_orders = (1,)) for x in ts] ≈ derivs
        @test eval_grid(nd; derivative_orders = (1,)) ≈ derivs
        @test [ForwardDiff.derivative(nd, x) for x in ts] ≈ derivs
    end

    # None throws outside the grid and evaluates normally on and inside it
    di = DataInterpolations.LinearInterpolation(u, t; extrapolation = DI_ET.None)
    nd = NDInterpolation(u, LinearInterpolationDimension(t; extrapolation = ET.None))
    @test nd.([0.0, 0.35, 1.1]) ≈ di.([0.0, 0.35, 1.1])
    @test_throws ExtrapolationError(1, true) nd(-0.1)
    @test_throws ExtrapolationError(1, false) nd(1.2)
    @test_throws ExtrapolationError nd(1.2; derivative_orders = (1,))
    @test_throws ExtrapolationError ForwardDiff.derivative(nd, 1.2)
end

@testset "Linear extrapolation sides and corners" begin
    for kind in (:linear, 1, 2, 3), N in (2, 3)
        interp = test_interp(kind, N, ntuple(_ -> ET.Linear, N))
        points = side_and_corner_points(map(t -> (first(t), last(t)), TS[1:N]))
        orders_list = N == 2 ?
            ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)) :
            ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (0, 2, 0))
        for x in points, orders in orders_list
            expected = expected_extrapolation(interp, ntuple(_ -> ET.Linear, N), x, orders)
            @test interp(x; derivative_orders = orders) ≈ expected atol = 1.0e-10
        end
    end

    # The corner of a cubic spline grows bilinearly, including the mixed term
    interp = test_interp(3, 2, (ET.Linear, ET.Linear))
    xb = (1.6, 2.0)
    f(o) = interp(xb; derivative_orders = o)
    @test !isapprox(f((1, 1)), 0; atol = 1.0e-6)
    for Δ in ((0.5, 0.25), (1.0, 2.0), (3.0, 0.1))
        x = xb .+ Δ
        @test interp(x) ≈ f((0, 0)) + f((1, 0)) * Δ[1] + f((0, 1)) * Δ[2] + f((1, 1)) * Δ[1] * Δ[2]
    end
end

@testset "Mixed modes sides and corners" begin
    mode_sets = (
        (ET.Linear, ET.Constant),
        (ET.Constant, ET.Extension),
        (ET.Extension, ET.Linear, ET.Constant),
        (ET.Linear, ET.Extension, ET.Linear),
    )
    for kind in (:linear, 2, 3), modes in mode_sets
        N = length(modes)
        interp = test_interp(kind, N, modes)
        points = side_and_corner_points(map(t -> (first(t), last(t)), TS[1:N]))
        for x in points, orders in Iterators.product(ntuple(_ -> 0:2, N)...)
            sum(orders) > 3 && continue
            expected = expected_extrapolation(interp, modes, x, orders)
            @test interp(x; derivative_orders = orders) ≈ expected atol = 1.0e-10
        end
    end
end

@testset "Multi point evaluation outside the grid" begin
    t_evals = ([-0.5, 0.4, 2.1], [-1.7, 0.1, 2.9], [1.5, 3.0, 4.6])
    for kind in (:linear, 2, 3),
            modes in ((ET.Linear, ET.Linear, ET.Linear), (ET.Periodic, ET.Linear, ET.Reflective))

        ts = TS
        u = coefficients(kind, ts)
        dims = ntuple(3) do i
            if kind == :linear
                LinearInterpolationDimension(ts[i]; extrapolation = modes[i], t_eval = t_evals[i])
            else
                BSplineInterpolationDimension(
                    ts[i], kind; extrapolation = modes[i], t_eval = t_evals[i],
                    max_derivative_order_eval = 2
                )
            end
        end
        interp = NDInterpolation(u, dims)
        vector_interp = NDInterpolation(cat(u, 2u; dims = 4), dims)
        for orders in ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 0, 1))
            grid = eval_grid(interp; derivative_orders = orders)
            vector_grid = eval_grid(vector_interp; derivative_orders = orders)
            for I in CartesianIndices(grid)
                x = ntuple(i -> t_evals[i][I[i]], 3)
                @test grid[I] ≈ interp(x; derivative_orders = orders) atol = 1.0e-12
                @test vector_grid[I, :] ≈ [grid[I], 2grid[I]] atol = 1.0e-12
            end
            @test eval_unstructured(interp; derivative_orders = orders) ≈
                [grid[i, i, i] for i in 1:3] atol = 1.0e-12
        end
    end
end

@testset "Derivative continuity at the boundary" begin
    ε = 1.0e-9
    for kind in (:linear, 2, 3), mode in (ET.Linear, ET.Extension)
        interp = test_interp(kind, 2, (mode, mode))
        for (axis, xb) in ((1, 0.0), (1, 1.6), (2, -1.0), (2, 2.0)), other in (0.45, -3.0, 5.0)
            point(s) = axis == 1 ? (xb + s, other) : (other, xb + s)
            e = axis == 1 ? (1, 0) : (0, 1)
            inner = xb == first(TS[axis]) ? ε : -ε
            # Value and first partials are continuous across the boundary
            @test interp(point(-inner)) ≈ interp(point(inner)) atol = 1.0e-5
            for orders in ((1, 0), (0, 1))
                @test interp(point(-inner); derivative_orders = orders) ≈
                    interp(point(inner); derivative_orders = orders) atol = 1.0e-4
                @test interp(point(0.0); derivative_orders = orders) ≈
                    interp(point(-inner); derivative_orders = orders) atol = 1.0e-4
            end
            # For Linear the slope outside is exactly the boundary slope
            if mode == ET.Linear
                @test interp(point(-1.0e3 * inner); derivative_orders = e) ≈
                    interp(point(0.0); derivative_orders = e)
                @test interp(point(-1.0e3 * inner); derivative_orders = 2 .* e) == 0
            end
        end
    end
end

@testset "Derivatives against finite differences outside the grid" begin
    h = 1.0e-5
    fd(f, x, i) = (f(Base.setindex(x, x[i] + h, i)) - f(Base.setindex(x, x[i] - h, i))) / (2h)
    modes_list = (
        (ET.Linear, ET.Linear, ET.Linear),
        (ET.Constant, ET.Linear, ET.Extension),
        (ET.Periodic, ET.Reflective, ET.Linear),
        (ET.Reflective, ET.Periodic, ET.Constant),
    )
    for kind in (:linear, 2, 3), modes in modes_list
        interp = test_interp(kind, 3, modes)
        points = filter(side_and_corner_points(map(t -> (first(t), last(t)), TS))) do x
            # Stay away from grid points (kinks of linear interpolation) after folding
            all(i -> all(t -> abs(t - x[i]) > 1.0e-3, TS[i]), 1:3)
        end
        for x in points, i in 1:3
            e = ntuple(j -> Int(j == i), 3)
            @test interp(x; derivative_orders = e) ≈ fd(interp, x, i) atol = 1.0e-5
            @test ForwardDiff.derivative(s -> interp(Base.setindex(x, s, i)), x[i]) ≈
                interp(x; derivative_orders = e) atol = 1.0e-10
            kind == :linear && continue
            for j in 1:3
                e2 = e .+ ntuple(k -> Int(k == j), 3)
                @test interp(x; derivative_orders = e2) ≈
                    fd(y -> interp(y; derivative_orders = e), x, j) atol = 1.0e-4
            end
        end
    end
end

@testset "Periodic and reflective N-D" begin
    for kind in (:linear, 2, 3)
        interp = test_interp(kind, 2, (ET.Periodic, ET.Reflective))
        P1 = last(TS[1]) - first(TS[1])
        P2 = last(TS[2]) - first(TS[2])
        for x in ((0.25, 0.3), (1.1, -0.9), (0.05, 1.7))
            for k in (-2, -1, 1, 3)
                @test interp(x[1] + k * P1, x[2]) ≈ interp(x...)
                @test interp(x[1] + k * P1, x[2]; derivative_orders = (1, 1)) ≈
                    interp(x...; derivative_orders = (1, 1))
            end
            mirrored = 2 * last(TS[2]) - x[2]
            @test interp(x[1], mirrored) ≈ interp(x...)
            @test interp(x[1], mirrored; derivative_orders = (0, 1)) ≈
                -interp(x...; derivative_orders = (0, 1))
            @test interp(x[1], mirrored; derivative_orders = (1, 1)) ≈
                -interp(x...; derivative_orders = (1, 1))
            @test interp(x[1], x[2] + 2P2; derivative_orders = (0, 1)) ≈
                interp(x...; derivative_orders = (0, 1))
            if kind != :linear
                @test interp(x[1], mirrored; derivative_orders = (0, 2)) ≈
                    interp(x...; derivative_orders = (0, 2))
            end
        end
    end
end

@testset "None" begin
    for kind in (:linear, 2)
        interp = test_interp(kind, 2, (ET.Linear, ET.None))
        @test interp(5.0, 0.5) isa Float64
        @test interp(5.0, 2.0) isa Float64
        @test_throws ExtrapolationError(2, true) interp(0.5, -1.5)
        @test_throws ExtrapolationError(2, false) interp(5.0, 2.5)
        @test_throws ExtrapolationError interp(0.5, 2.5; derivative_orders = (1, 0))
        @test sprint(showerror, ExtrapolationError(2, false)) ==
            "ExtrapolationError: cannot extrapolate above the last point of interpolation " *
            "dimension 2 because its extrapolation is `ExtrapolationType.None`."
    end
    # Multi point evaluation: constructing with points outside the grid is fine,
    # evaluating there throws
    dims = (
        LinearInterpolationDimension([0.0, 1.0]; t_eval = [0.5, 1.5]),
        LinearInterpolationDimension([0.0, 1.0]; t_eval = [-0.5, 0.5], extrapolation = ET.None),
    )
    interp = NDInterpolation([1.0 2.0; 3.0 4.0], dims)
    @test_throws ExtrapolationError(2, true) eval_grid(interp)
    @test_throws ExtrapolationError(2, true) eval_unstructured(interp)
    dims = (
        LinearInterpolationDimension([0.0, 1.0]; t_eval = [0.5, 1.5]),
        LinearInterpolationDimension([0.0, 1.0]; t_eval = [0.0, 1.0], extrapolation = ET.None),
    )
    interp = NDInterpolation([1.0 2.0; 3.0 4.0], dims)
    @test eval_grid(interp) ≈ [interp(a, b) for a in dims[1].t_eval, b in dims[2].t_eval]
    # A single breakpoint only admits the breakpoint itself
    interp = NDInterpolation([5.0], LinearInterpolationDimension([1.0]; extrapolation = ET.None))
    @test interp(1.0) == 5.0
    @test_throws ExtrapolationError interp(1.5)
end

@testset "Default extrapolation is unchanged" begin
    # Reference values computed with the default extrapolation before `None`, `Periodic`
    # and `Reflective` were added.
    x = [0.0, 0.4, 1.0, 1.5]
    y = [-1.0, 0.0, 2.0]
    u = [sin(3a) + cos(2b) + a * b for a in x, b in y]
    u_bspline = [sin(i) + cos(2j) + 0.3i * j for i in 1:5, j in 1:4]
    pts = [(-0.7, -1.6), (-0.7, 0.5), (0.7, 2.9), (2.2, 2.9), (2.2, -1.6), (0.7, 0.5)]
    reference = (
        (
            (LinearInterpolationDimension(x), LinearInterpolationDimension(y)), u, ET.Linear,
            [-1.776903338918074, -1.3944793056585487, 1.1687962967613097, 2.438576456067716, -7.329475232155476, 1.4731686417976437],
            [0.7300977149180659, 2.830097714918066, 1.581801536821068, 0.6626997485500721, -3.837300251449927, -0.8181984631789319],
            [1.0, 1.0000000000000004, 1.0, 1.0, 0.9999999999999996, 1.0],
        ),
        (
            (ConstantInterpolationDimension(x), ConstantInterpolationDimension(y)), u, ET.Constant,
            [-0.4161468365471424, 1.0, 1.0783954651036145, 1.3688262614712912, -2.893676954212239, 1.9320390859672263],
            zeros(6), zeros(6),
        ),
        (
            (BSplineInterpolationDimension(x, 2), BSplineInterpolationDimension(y, 2)), u_bspline, ET.Extension,
            [0.17589843696628493, -3.7558810776179987, 3.0019646126046506, 10.513739664431988, -2.1437708073234982, 2.827544001264887],
            [2.268234543424896, 11.8852970434249, 1.2605521236270993, 10.858288199489849, -0.58594816414654, -0.045266058191083036],
            [8.61000000000001, 2.3062499999999995, 0.6586363636363641, 2.4338181818181823, 5.9258181818182045, 0.42954545454545456],
        ),
    )
    for (dims, U, mode, vals, d10, d11) in reference
        @test all(dim -> dim.extrapolation == mode, dims)
        interp = NDInterpolation(U, dims)
        @test [interp(p...) for p in pts] ≈ vals rtol = 1.0e-12
        @test [interp(p...; derivative_orders = (1, 0)) for p in pts] ≈ d10 rtol = 1.0e-12
        @test [interp(p...; derivative_orders = (1, 1)) for p in pts] ≈ d11 rtol = 1.0e-12
    end
end

@testset "Allocation free and isbits modes" begin
    @test isbitstype(ExtrapolationType.T)
    @test isbitstype(ExtrapolationError)
    alloc(interp, x, orders) = @allocated interp(x; derivative_orders = orders)
    for mode in instances(ExtrapolationType.T),
            dim_type in (LinearInterpolationDimension, ConstantInterpolationDimension)

        dims = (
            dim_type(TS[1]; extrapolation = mode),
            dim_type(TS[2]; extrapolation = ET.Linear),
            dim_type(TS[3]; extrapolation = mode),
        )
        interp = NDInterpolation(coefficients(:linear, TS), dims)
        x = mode == ET.None ? (0.5, -3.0, 3.0) : (-1.3, -3.0, 5.2)
        for orders in ((0, 0, 0), (1, 0, 0), (1, 1, 1))
            alloc(interp, x, orders)
            @test alloc(interp, x, orders) == 0
        end
    end
end
