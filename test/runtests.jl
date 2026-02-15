using SafeTestsets, Pkg
const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

function activate_qa_env()
    Pkg.activate("qa")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Interpolations" include("test_interpolations.jl")
    @safetestset "Derivatives" include("test_derivatives.jl")
    @safetestset "DataInterpolations" include("test_datainterpolations_comparison.jl")
    @safetestset "Interface" include("test_interface.jl")
elseif GROUP == "Extensions"
    @safetestset "Symbolics Extension" include("test_symbolics_ext.jl")
elseif GROUP == "QA"
    activate_qa_env()
    include("qa/runtests.jl")
elseif GROUP == "GPU"
    activate_gpu_env()
    # TODO: Add GPU tests
end
