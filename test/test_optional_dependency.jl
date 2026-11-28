using Test

if !any(id -> id.name == "KernelAbstractions", keys(Base.loaded_modules))
    using DataInterpolationsND
    dim = LinearInterpolationDimension([0.0, 1.0])
    interp = NDInterpolation([0.0 1.0; 1.0 2.0], (dim, dim))
    interp(0.25, 0.75)
    @test !any(id -> id.name == "KernelAbstractions", keys(Base.loaded_modules))
end
