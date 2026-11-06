using DataInterpolationsND
using Test

@testset "Interpolation dimension show" begin
    t = [0.0, 1.0, 2.0, 3.0, 4.0]

    @testset "LinearInterpolationDimension" begin
        dim = LinearInterpolationDimension(t)
        @test sprint(show, dim) ==
            "LinearInterpolationDimension with 5 points, extrapolation = Linear"
        @test sprint(show, MIME"text/plain"(), dim) ==
            "LinearInterpolationDimension with 5 points, extrapolation = Linear"
        @test sprint(show, dim; context = :compact => true) ==
            "LinearInterpolationDimension"
    end

    @testset "ConstantInterpolationDimension" begin
        dim = ConstantInterpolationDimension(t)
        @test sprint(show, MIME"text/plain"(), dim) ==
            "ConstantInterpolationDimension with 5 points, in left direction, extrapolation = Constant"
        dim_right = ConstantInterpolationDimension(t; left = false)
        @test occursin("right direction", sprint(show, MIME"text/plain"(), dim_right))
    end

    @testset "BSplineInterpolationDimension" begin
        dim = BSplineInterpolationDimension(t, 2)
        @test sprint(show, MIME"text/plain"(), dim) ==
            "BSplineInterpolationDimension with 5 knots, degree 2, extrapolation = Extension"
        @test sprint(show, dim; context = :compact => true) ==
            "BSplineInterpolationDimension"
    end
end

@testset "Cache show" begin
    @test sprint(show, DataInterpolationsND.EmptyCache()) == "EmptyCache()"
    w = NURBSWeights(ones(3, 4))
    @test sprint(show, MIME"text/plain"(), w) == "NURBSWeights with weights size (3, 4)"
    @test sprint(show, w; context = :compact => true) == "NURBSWeights"
end

@testset "NDInterpolation show" begin
    t1 = [0.0, 1.0, 2.0, 3.0, 4.0]
    t2 = [0.0, 1.0, 2.0]
    u = reshape(1.0:15.0, 5, 3)
    itp = NDInterpolation(
        u,
        (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        ),
    )

    compact = sprint(show, itp; context = :compact => true)
    @test compact ==
        "NDInterpolation{2→0}(LinearInterpolationDimension, LinearInterpolationDimension)"

    plain = sprint(show, MIME"text/plain"(), itp)
    @test startswith(plain, "NDInterpolation with 2 input dimensions and scalar output\n")
    @test occursin("u: 5×3", plain)
    @test occursin(
        "1: LinearInterpolationDimension with 5 points, extrapolation = Linear", plain
    )
    @test occursin(
        "2: LinearInterpolationDimension with 3 points, extrapolation = Linear", plain
    )
    @test !occursin("cache:", plain)

    # Non-compact Base.show should use the short header (no full array dump)
    noncompact = sprint(show, itp)
    @test noncompact == "NDInterpolation with 2 input dimensions and scalar output"
    @test !occursin(string(u[1]), noncompact)

    # Vector-valued output
    u_vec = reshape(1.0:30.0, 5, 3, 2)
    itp_vec = NDInterpolation(
        u_vec,
        (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2),
        ),
    )
    plain_vec = sprint(show, MIME"text/plain"(), itp_vec)
    @test startswith(
        plain_vec, "NDInterpolation with 2 input dimensions and output size (2,)\n"
    )

    # 1D input
    itp1 = NDInterpolation(collect(1.0:5.0), LinearInterpolationDimension(t1))
    plain1 = sprint(show, MIME"text/plain"(), itp1)
    @test startswith(plain1, "NDInterpolation with 1 input dimension and scalar output\n")

    # With NURBSWeights cache
    t_bs = [0.0, 1.0, 2.0, 3.0]
    dim_bs = BSplineInterpolationDimension(t_bs, 1)
    n_basis = DataInterpolationsND.get_n_basis_functions(dim_bs)
    u_bs = ones(n_basis)
    weights = NURBSWeights(ones(n_basis))
    itp_nurbs = NDInterpolation(u_bs, dim_bs; cache = weights)
    plain_nurbs = sprint(show, MIME"text/plain"(), itp_nurbs)
    @test occursin("cache: NURBSWeights", plain_nurbs)
    @test occursin(
        "1: BSplineInterpolationDimension with 4 knots, degree 1, extrapolation = Extension",
        plain_nurbs,
    )
end
