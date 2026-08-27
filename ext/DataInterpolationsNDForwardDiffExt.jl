module DataInterpolationsNDForwardDiffExt

using DataInterpolationsND
using ForwardDiff: ForwardDiff

function DataInterpolationsND.search_value(d::ForwardDiff.Dual)
    return DataInterpolationsND.search_value(ForwardDiff.value(d))
end

end
