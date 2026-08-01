using SciMLTesting, DataInterpolationsND, Test

# ExplicitImports only sees an extension module once its triggers are loaded, so the
# weakdeps have to be brought in here for DataInterpolationsNDSymbolicsExt to be checked.
using SymbolicUtils, Symbolics

run_qa(
    DataInterpolationsND;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # SymbolicUtils: the hooks a callable must implement to participate in
                # symbolic terms are themselves not public -- `promote_symtype` and
                # `promote_shape` are the extension points, and `ShapeT` / `ShapeVecT`
                # are the shape types their signatures and return values are built from.
                :ShapeT, :ShapeVecT, :promote_shape, :promote_symtype,
                # Symbolics: `SymbolicT` is the unwrapped symbolic type the generated
                # interpolation call methods dispatch on; there is no public spelling.
                :SymbolicT,
                # DataInterpolationsND's own internal helper, reached from its own
                # extension module.
                :get_output_size,
            ),
        ),
    ),
)

# JET is run as a targeted analysis (report_call / report_opt on the public
# entry points) rather than via run_qa's JET.test_package path: typo-mode
# package analysis surfaces a JuliaSyntax parentheses warning on the
# `using EllipsisNotation: EllipsisNotation, (..)` import in src as a toplevel
# error, which is a tooling artifact, not a defect.
include("jet.jl")
