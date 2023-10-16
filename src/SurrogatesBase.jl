module SurrogatesBase

import Statistics: mean, var
import Base: rand
import StatsBase: mean_and_var

export AbstractSurrogate, add_points!
export update_hyperparameters!, hyperparameters
export mean, var, mean_and_var, rand

"""
    abstract type AbstractSurrogate end

An abstract type for surrogate functions.

    (s::AbstractSurrogate)(x)

Subtypes of `AbstractSurrogate` are callable with input points `x` such that the result
is an evaluation of the surrogate at `x`.

# Examples
```jldoctest
julia> struct ZeroSurrogate <: AbstractSurrogate end

julia> (::ZeroSurrogate)(x) = 0

julia> s = ZeroSurrogate()
ZeroSurrogate()

julia> s(4) == 0
true
```
"""
abstract type AbstractSurrogate <: Function end

"""
    add_points!(s::AbstractSurrogate, new_xs::AbstractVector, new_ys::AbstractVector)

Add evaluations `new_ys` at points `new_xs` to the surrogate.
"""
function add_points! end

"""
    update_hyperparameters!(s::AbstractSurrogate, prior)

Use prior on hyperparameters passed in `prior` to perform an update.

See also [`hyperparameters`](@ref).
"""
function update_hyperparameters! end

"""
    hyperparameters(s::AbstractSurrogate)

Return a `NamedTuple`, in which names are hyperparameters and values are currently used
values of hyperparameters in `s`.

See also [`update_hyperparameters!`](@ref).
"""
function hyperparameters end

"""
    mean(s::AbstractSurrogate, xs::AbstractVector)

Return mean at points `xs`.
"""
mean(::AbstractSurrogate, ::AbstractVector)

"""
    var(s::AbstractSurrogate, xs::AbstractVector)

Return variance at points `xs`.
"""
var(::AbstractSurrogate, ::AbstractVector)

"""
    mean_and_var(s::AbstractSurrogate, xs)

Return a Tuple of means and variances at points `xs`.
"""
function mean_and_var(s::AbstractSurrogate, xs::AbstractVector)
    mean(s, xs), var(s, xs)
end

"""
    rand(s::AbstractSurrogate, xs::AbstractVector)

Return a sample from the joint posterior at points `xs`.
"""
rand(::AbstractSurrogate, ::AbstractVector)

end
