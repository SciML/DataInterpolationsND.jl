using NDInterpolations
using DataInterpolations

isapprox_or_nan(v1, v2) = (v1 ≈ v2) || (all(isnan, v1) && all(isnan, v2))

function interpolation_comparison(
        itp_dim_type::Type{<:NDInterpolations.AbstractInterpolationDimension},
        itp_type::Type{<:DataInterpolations.AbstractInterpolation}
)
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]

    n = length(t)

    itp_DI = itp_type(u, t)
    itp_NDI = NDInterpolation(u, itp_dim_type(t))

    # Interior non data points
    for i in 1:(n - 1)
        t_eval = range(t[i], t[i + 1]; length = 25)[2:24]
        @test itp_DI(t_eval) ≈ itp_NDI.(t_eval)
    end

    # Interior data points
    t_eval = view(t, 2:(n - 1))
    @test itp_DI(t_eval) ≈ itp_NDI.(t_eval)

    # Boundary points
    t_eval = [first(t), last(t)]
    @test itp_DI(t_eval) ≈ itp_NDI.(t_eval)
end

function interpolation_derivative_comparison(
        itp_dim_type::Type{<:NDInterpolations.AbstractInterpolationDimension},
        itp_type::Type{<:DataInterpolations.AbstractInterpolation}
)
    t = [0.0, 0.2, 0.5, 0.7, 0.9, 1.1]
    u = [1.5, -5.3, 0.5, 3.3, 3.6, 1.0]

    n = length(t)

    itp_DI = itp_type(u, t)
    itp_NDI = NDInterpolation(u, itp_dim_type(t))

    # Interior non data points
    for i in 1:(n - 1)
        t_eval = range(t[i], t[i + 1]; length = 25)[2:24]
        @test DataInterpolations.derivative.(Ref(itp_DI), t_eval) ≈
              itp_NDI.(t_eval; derivative_orders = (1,))
    end

    # Interior data points
    t_eval = view(t, 2:(n - 1))
    @test isapprox_or_nan(
        DataInterpolations.derivative.(Ref(itp_DI), t_eval),
        itp_NDI.(t_eval; derivative_orders = (1,))
    )

    # Boundary points
    t_eval = [first(t), last(t)]
    @test isapprox_or_nan(
        DataInterpolations.derivative.(Ref(itp_DI), t_eval),
        itp_NDI.(t_eval; derivative_orders = (1,))
    )
end

for (itp_dim_type, itp_type) in (
    (LinearInterpolationDimension, LinearInterpolation),
    (ConstantInterpolationDimension, ConstantInterpolation)
)
    @testset "$itp_type" begin
        interpolation_comparison(itp_dim_type, itp_type)
        interpolation_derivative_comparison(itp_dim_type, itp_type)
    end
end