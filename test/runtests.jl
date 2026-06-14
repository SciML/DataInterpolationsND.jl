using SafeTestsets, SciMLTesting

run_tests(;
    core = () -> begin
        @safetestset "Interpolations" include("test_interpolations.jl")
        @safetestset "Derivatives" include("test_derivatives.jl")
        @safetestset "DataInterpolations" include("test_datainterpolations_comparison.jl")
        @safetestset "Interface" include("test_interface.jl")
    end,
    groups = Dict(
        "Extensions" => joinpath(@__DIR__, "Extensions", "test_symbolics_ext.jl"),
        "GPU" => (; env = joinpath(@__DIR__, "gpu"), body = () -> nothing),
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "runtests.jl")),
    all = ["Core"],
)
