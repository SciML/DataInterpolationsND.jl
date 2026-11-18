###################### Interpolation dimensions ######################

function Base.show(io::IO, dim::AbstractInterpolationDimension)
    if get(io, :compact, false)
        print(io, nameof(typeof(dim)))
    else
        print(io, dim_show_summary(dim))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", dim::AbstractInterpolationDimension)
    print(io, dim_show_summary(dim))
    return nothing
end

function dim_show_summary(dim::LinearInterpolationDimension)
    return string(
        nameof(typeof(dim)),
        " with ",
        length(dim.t),
        " points, extrapolation = ",
        dim.extrapolation,
    )
end

function dim_show_summary(dim::ConstantInterpolationDimension)
    direction = dim.left ? "left" : "right"
    return string(
        nameof(typeof(dim)),
        " with ",
        length(dim.t),
        " points, in ",
        direction,
        " direction, extrapolation = ",
        dim.extrapolation,
    )
end

function dim_show_summary(dim::BSplineInterpolationDimension)
    return string(
        nameof(typeof(dim)),
        " with ",
        length(dim.t),
        " knots, degree ",
        dim.degree,
        ", extrapolation = ",
        dim.extrapolation,
    )
end

###################### Caches ######################

function Base.show(io::IO, cache::EmptyCache)
    print(io, "EmptyCache()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", cache::EmptyCache)
    print(io, "EmptyCache()")
    return nothing
end

function Base.show(io::IO, cache::NURBSWeights)
    if get(io, :compact, false)
        print(io, "NURBSWeights")
    else
        print(io, "NURBSWeights with weights size ", size(cache.weights))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", cache::NURBSWeights)
    print(io, "NURBSWeights with weights size ", size(cache.weights))
    return nothing
end

###################### NDInterpolation ######################

function Base.show(io::IO, interp::NDInterpolation{N_in, N_out}) where {N_in, N_out}
    if get(io, :compact, false)
        dim_names = join((nameof(typeof(dim)) for dim in interp.interp_dims), ", ")
        print(io, "NDInterpolation{", N_in, "→", N_out, "}(", dim_names, ")")
    else
        print(io, nd_show_header(interp))
    end
    return nothing
end

function Base.show(
        io::IO, ::MIME"text/plain", interp::NDInterpolation{N_in, N_out}
    ) where {N_in, N_out}
    println(io, nd_show_header(interp))
    println(io, "  u: ", summary(interp.u))
    if !(interp.cache isa EmptyCache)
        println(io, "  cache: ", sprint(show, interp.cache; context = :compact => true))
    end
    println(io, "  dims:")
    for (i, dim) in enumerate(interp.interp_dims)
        suffix = i == N_in ? "" : "\n"
        print(io, "    ", i, ": ", dim_show_summary(dim), suffix)
    end
    return nothing
end

function nd_show_header(interp::NDInterpolation{N_in, N_out}) where {N_in, N_out}
    out_desc = if N_out == 0
        "scalar output"
    else
        "output size $(size(interp.u)[(N_in + 1):end])"
    end
    return string(
        "NDInterpolation with ",
        N_in,
        " input dimension",
        N_in == 1 ? "" : "s",
        " and ",
        out_desc,
    )
end
