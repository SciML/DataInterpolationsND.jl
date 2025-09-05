using DataInterpolationsND
using SafeTestsets

# Only run these tests if Symbolics is available
@safetestset "Symbolics Extension" begin
    try
        using Symbolics
        
        # Create a simple 2D interpolation
        t1 = [1.0, 2.0, 3.0]
        t2 = [0.0, 1.0, 2.0]
        u = [i + j for i in t1, j in t2]  # 3x3 matrix

        itp_dims = (
            LinearInterpolationDimension(t1),
            LinearInterpolationDimension(t2)
        )
        itp = NDInterpolation(u, itp_dims)

        # Test symbolic variables
        @variables x y

        # Test symbolic evaluation
        println("Testing symbolic evaluation...")
        result = itp(x, y)
        @test result isa Symbolics.Num
        println("Symbolic result: ", result)

        # Test symbolic differentiation
        println("Testing symbolic differentiation...")
        ∂f_∂x = Symbolics.derivative(result, x)
        ∂f_∂y = Symbolics.derivative(result, y)

        @test ∂f_∂x isa Symbolics.Num
        @test ∂f_∂y isa Symbolics.Num

        println("∂f/∂x = ", ∂f_∂x)
        println("∂f/∂y = ", ∂f_∂y)

        # Test that we can substitute values
        substituted = Symbolics.substitute(result, Dict(x => 1.5, y => 0.5))
        println("Substituted result: ", substituted)
        
        # Compare with numerical evaluation
        numerical_result = itp(1.5, 0.5)
        @test Float64(substituted) ≈ numerical_result
        
        println("Symbolics extension test completed successfully!")
    catch e
        if e isa ArgumentError && contains(string(e), "Package Symbolics not found")
            @info "Symbolics not available, skipping symbolic tests"
        else
            rethrow(e)
        end
    end
end