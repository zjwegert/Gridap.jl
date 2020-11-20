module IntegratedLegendreDofBasesTests

using Test
using Gridap.Helpers
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials
using Gridap.ReferenceFEs
# using BenchmarkTools

include("../../src/ReferenceFEs/IntegratedLegendreDofBases.jl")

# Scalar: 1D Linear
D = 1
order = 1
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nf_nodes, nf_moments = _IL_nodes_and_moments(SEGMENT,b)

# Scalar: 1D Cubic
D = 1
order = 3
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nf_nodes2, nf_moments2 = _IL_nodes_and_moments(SEGMENT,b)

# Scalar: 2D Quadratic
D = 2
order = 2
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nf_nodes3, nf_moments3 = _IL_nodes_and_moments(QUAD,b)

# Scalar: 3D Quadratic
D = 3
order = 2
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nf_nodes4, nf_moments4 = _IL_nodes_and_moments(HEX,b)

# cache = return_cache(b,[xi,])
# @btime evaluate!($cache,$b,$[xi,])
# cache = return_cache(∇b,[xi,])
# @btime evaluate!($cache,$∇b,$[xi,])
# cache = return_cache(∇∇b,[xi,])
# @btime evaluate!($cache,$∇∇b,$[xi,])

end # module
