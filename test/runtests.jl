using TestItemRunner, Pkg
const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    return Pkg.instantiate()
end

function activate_qa_env()
    Pkg.activate("qa")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    core_files = ("test_interpolations.jl", "test_derivatives.jl", "test_datainterpolations_comparison.jl", "test_interface.jl")
    @run_package_tests filter = ti -> any(endswith(ti.filename, file) for file in core_files)

elseif GROUP == "Extensions"
    extension_files = ("test_symbolics_ext.jl",)
    @run_package_tests filter = ti -> any(endswith(ti.filename, file) for file in extension_files)
elseif GROUP == "QA"
    activate_qa_env()
    @run_package_tests filter = ti -> endswith(ti.filename, "qa/runtests.jl")
elseif GROUP == "GPU"
    activate_gpu_env()
    # TODO: Add GPU tests
end
