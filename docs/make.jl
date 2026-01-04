using Documenter, DataInterpolationsND

makedocs(
    sitename = "DataInterpolationsND.jl",
    clean = true,
    doctest = false,
    linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DataInterpolationsND/stable/"
    ),
    pages = [
        "index.md",
        "Usage" => "usage.md",
        "Interpolation Types" => "interpolation_types.md",
        "Splines" => "splines.md",
        "API" => "api.md",
    ]
)

deploydocs(repo = "github.com/SciML/DataInterpolationsND.jl")
