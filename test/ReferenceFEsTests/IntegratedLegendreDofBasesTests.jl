module IntegratedLegendreDofBasesTests

using Test
using Gridap.Helpers
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials
using Gridap.ReferenceFEs
# using BenchmarkTools

import Gridap.Fields: Broadcasting
import Gridap.ReferenceFEs: Quadrature
import Gridap.ReferenceFEs: get_coordinates, get_weights
import Gridap.ReferenceFEs: _compute_nodes

include("../../src/ReferenceFEs/IntegratedLegendreDofBases.jl")

# Scalar: 1D Linear
D = 1
order = 1
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nodes, f_moments, f_nodes = _IL_nodes_and_moments(SEGMENT,b)

# Scalar: 1D Cubic
D = 1
order = 3
V = Float64
b = IntegratedLegendreBasis{D}(V,order)
nodes2, f_moments2, f_nodes2 = _IL_nodes_and_moments(SEGMENT,b)

nd, fnd = compute_nodes(QUAD,(3,3))

# # Scalar: 2D Quadratic
# D = 2
# order = 2
# V = Float64
# b = IntegratedLegendreBasis{D}(V,order)
# nodes3, f_moments3 = _IL_nodes_and_moments(QUAD,b)

# # Scalar: 3D Quadratic
# D = 3
# order = 2
# V = Float64
# b = IntegratedLegendreBasis{D}(V,order)
# nodes4, f_moments4 = _IL_nodes_and_moments(HEX,b)

# cache = return_cache(b,[xi,])
# @btime evaluate!($cache,$b,$[xi,])
# cache = return_cache(∇b,[xi,])
# @btime evaluate!($cache,$∇b,$[xi,])
# cache = return_cache(∇∇b,[xi,])
# @btime evaluate!($cache,$∇∇b,$[xi,])

end # module
