@testitem "ExplicitImports" begin
    using ExplicitImports
    using DataInterpolationsND
    @test check_no_implicit_imports(DataInterpolationsND) === nothing
    @test check_no_stale_explicit_imports(DataInterpolationsND) === nothing
end
