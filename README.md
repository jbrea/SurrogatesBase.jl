# SurrogatesBase.jl

API for deterministic and stochastic surrogates.

Given data ``((x_1, y_1), \ldots, (x_N, y_N))`` obtained by evaluating a function ``y_i =
f(x_i)`` or sampling from a conditional probability density ``p_{Y|X}(Y = y_i|X = x_i)``,
a **deterministic surrogate** is a function ``s(x)`` (e.g. a [radial basis function
interpolator](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation)) that
uses the data to approximate ``f`` or some statistic of ``p_{Y|X}`` (e.g. the mean),
whereas a **stochastic surrogate** is a stochastic process (e.g. a [Gaussian process
approximation](https://en.wikipedia.org/wiki/Gaussian_process_approximations)) that uses
the data to approximate ``f`` or ``p_{Y|X}`` *and* quantify the uncertainty of
approximation.

## Deterministic Surrogates

Deterministic surrogates `s` are subtypes of `SurrogatesBase.AbstractDeterministicSurrogate`.
The method ``add_points!(s, xs, ys)`` **must** be implemented and the surrogate must be
[callable](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects)
``s(xs)``, where `xs` is a `Vector` of inputs and `ys` is a `Vector` of outputs.
For single points `x` and `y`, these functions are called as ``add_points!(s, [x], [y])``
and ``s([x])``.

If the surrogate `s` has tunable hyper-parameters, the methods
``update_hyperparameters!(s, prior)`` and ``hyperparameters(s)`` have to be implemented.

### Example

```julia
using SurrogatesBase

struct RBF{T} <: AbstractDeterministicSurrogate
    scale::T
    centers::Vector{T}
    weights::Vector{T}
end

(rbf::RBF)(xs) = [rbf.weights' * exp.(-rbf.scale * (x .- rbf.centers).^2)
                  for x in xs]

function add_points!(rbf::RBF, xs, ys)
    # update rbf.weights
    return rbf
end

hyperparameters(rbf::RBG) = (rbf.scale,)

function update_hyperparameters!(rbf::RBF, prior)
    # change rbf.scale and adapt rbf.weights, if necessary
    return rbf
end
```

## Stochastic Surrogates
