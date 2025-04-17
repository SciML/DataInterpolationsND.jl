using SafeTestsets

@safetestset "Aqua" include("aqua.jl")
@safetestset "Interpolations" include("test_interpolations.jl")
@safetestset "Derivatives" include("test_derivatives.jl")
@safetestset "DataInterpolations" include("test_datainterpolations_comparison.jl")
