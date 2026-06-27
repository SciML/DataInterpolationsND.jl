using SciMLTesting, DataInterpolationsND, Test

run_qa(
    DataInterpolationsND; explicit_imports = true,
    # `@adapt_structure` is a non-public (un-`public`-declared) macro of Adapt;
    # ignore until Adapt marks it public.
    ei_kwargs = (; all_explicit_imports_are_public = (; ignore = (Symbol("@adapt_structure"),))),
)

# JET is run as a targeted analysis (report_call / report_opt on the public
# entry points) rather than via run_qa's JET.test_package path: typo-mode
# package analysis surfaces a JuliaSyntax parentheses warning on the
# `using EllipsisNotation: EllipsisNotation, (..)` import in src as a toplevel
# error, which is a tooling artifact, not a defect.
include("jet.jl")
