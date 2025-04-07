const DiffFloat = Union{AbstractFloat, ForwardDiff.Dual}

"""
    get_module_function_names(mod)

Returns the symbols of module properties that correspond to functions.
"""
function get_module_function_names(
    mod::Module
)::Vector{Symbol}
    syms = names(mod, all=true)
    functions = filter(sym::Symbol -> isa(getproperty(mod,sym), Function), syms)
    return functions
end

"""
    get_module_function(mod, func_name)

Returns the function corresponding to the given name within the given module.
"""
function get_module_function(
    mod::Module,
    func_name::Symbol
)::Function
    # Get all functions in the module
    names = get_module_function_names(mod)
    # Find the appropriate function or throw an error
    for name in names
        if name == func_name
            return getproperty(mod, func_name)
        end
    end
    error("The given function is not implemented in this module!")
end

"""
    TransitionTiming(k, transition)

Contains the time step for which the given hybrid transition occurs at the beginning of.
"""
struct TransitionTiming
    k::Int
    transition::Transition
end
