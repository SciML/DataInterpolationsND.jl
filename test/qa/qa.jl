using SciMLTesting, DataInterpolationsND, Test

run_qa(DataInterpolationsND)

# JET is run as a targeted analysis (report_call / report_opt on the public
# entry points) rather than via run_qa's JET.test_package path: typo-mode
# package analysis surfaces a JuliaSyntax parentheses warning on the
# `using EllipsisNotation: EllipsisNotation, (..)` import in src as a toplevel
# error, which is a tooling artifact, not a defect.
include("jet.jl")
