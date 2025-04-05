using SafeTestsets

@safetestset "Aqua" include("aqua.jl")
@safetestset "DataInterpolations" include("test_datainterpolations_comparison.jl")