using Documenter, NDInterpolations

makedocs(
    sitename = "NDInterpolations.jl",
    clean = true,
    doctest = false,
    linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/NDInterpolations/stable/"
    ),
    pages = [
        "index.md",
        "Usage" => "usage.md",
        "Interpolation Types" => "interpolation_types.md",
        "API" => "api.md"
    ]
)

deploydocs(repo = "github.com/SciML/NDInterpolations.jl")