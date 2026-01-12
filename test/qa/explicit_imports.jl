using ExplicitImports
using DataInterpolationsND
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DataInterpolationsND) === nothing
    @test check_no_stale_explicit_imports(DataInterpolationsND) === nothing
end
