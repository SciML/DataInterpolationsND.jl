using SafeTestsets

@safetestset "Aqua" include("aqua.jl")
@safetestset "ExplicitImports" include("explicit_imports.jl")
@safetestset "JET" include("jet.jl")
