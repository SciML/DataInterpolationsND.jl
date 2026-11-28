module DataInterpolationsNDKernelAbstractionsExt

using DataInterpolationsND: DataInterpolationsND, CPUBackend, evaluate_point!,
    expand_knot_vector_point!, set_eval_idx_point!, set_basis_function_eval_point!
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel, get_backend,
    synchronize

function DataInterpolationsND.execution_backend(array::AbstractArray)
    storage = parent(array)
    storage === array || return DataInterpolationsND.execution_backend(storage)
    backend = get_backend(array)
    return backend isa KernelAbstractions.CPU ? CPUBackend() : backend
end

function DataInterpolationsND.evaluate_batch!(backend::KernelAbstractions.Backend, out, interp, derivative_orders, grid)
    N_in = length(interp.interp_dims)
    dims = grid ? size(out)[1:N_in] : (size(out, 1),)
    eval_kernel(backend)(out, interp, derivative_orders, Val(grid); ndrange = dims)
    synchronize(backend)
    return out
end

function DataInterpolationsND.set_eval_idx!(backend::KernelAbstractions.Backend, interp_dim)
    if !isempty(interp_dim.t_eval)
        set_idx_kernel(backend)(interp_dim; ndrange = length(interp_dim.t_eval))
    end
    synchronize(backend)
    return nothing
end

function DataInterpolationsND.expand_knot_vector!(backend::KernelAbstractions.Backend, knots_all, knot_values, multiplicities)
    expand_knot_vector_kernel(backend)(
        knots_all, knot_values, multiplicities; ndrange = length(knot_values)
    )
    synchronize(backend)
    return nothing
end

function DataInterpolationsND.set_basis_function_eval!(backend::KernelAbstractions.Backend, interp_dim)
    basis_function_eval_kernel(backend)(
        interp_dim;
        ndrange = (length(interp_dim.t_eval), interp_dim.max_derivative_order_eval + 1)
    )
    synchronize(backend)
    return nothing
end

# Interpolation contains views used in broadcasting; KA CPU Const wrappers do not
# support the alias checks these views require.
@kernel function eval_kernel(out, interp, derivative_orders, grid)
    k = @index(Global, NTuple)
    evaluate_point!(out, interp, derivative_orders, grid, k)
end

@kernel function set_idx_kernel(interp_dim)
    i = @index(Global, Linear)
    set_eval_idx_point!(interp_dim, i)
end

@kernel function expand_knot_vector_kernel(knots_all, @Const(knot_values), @Const(multiplicities))
    i = @index(Global, Linear)
    expand_knot_vector_point!(knots_all, knot_values, multiplicities, i)
end

@kernel function basis_function_eval_kernel(interp_dim)
    i, derivative_order_plus_1 = @index(Global, NTuple)
    set_basis_function_eval_point!(interp_dim, i, derivative_order_plus_1)
end

end
