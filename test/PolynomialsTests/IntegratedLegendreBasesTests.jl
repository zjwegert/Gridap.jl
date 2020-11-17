module IntegratedLegendreBasesTests

using Test
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials

import Gridap.Polynomials: _legendre
# import Gridap.Polynomials: _evaluate_1d_il!, _gradient_1d_il!, _hessian_1d_il!

@test all(map(i->map(x->_legendre(x,Val{i}()),(-1,1))==(1-2*(i%2),1),0:4))

# D = 3
# order = 4
# n = order + 1

# x = Point(0,0.5,1)
# V = zeros(Float64,(D,n))
# G = zeros(Float64,(D,n))
# H = zeros(Float64,(D,n))

# for d = 1:D
#   _evaluate_1d_il!(V,x,order,d)
#   _gradient_1d_il!(G,x,order,d)
#   _hessian_1d_il!(H,x,order,d)
# end

# order 0 degenerated case

# xi = Point(2,3)
# np = 5
# x = fill(xi,np)

# order = 0
# V = Float64
# G = gradient_type(V,xi)
# H = gradient_type(G,xi)
# b = MonomialBasis{2}(V,order)
# @test get_order(b) == 0
# @test get_orders(b) == (0,0)

# v = V[1.0,]
# g = G[(0.0, 0.0),]
# h = H[(0.0, 0.0, 0.0, 0.0),]

# bx = repeat(permutedims(v),np)
# ∇bx = repeat(permutedims(g),np)
# Hbx = repeat(permutedims(h),np)
# test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# Real-valued Q space with isotropic order

xi = Point(0.5,0.5)
np = 3
x = fill(xi,np)

order = 1
V = Float64
G = gradient_type(V,xi)
H = gradient_type(G,xi)
b = IntegratedLegendreBasis{2}(V,order)

v = V[0.25, 0.25, 0.25, 0.25]
g = G[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
h = H[(0.0, 1.0, 1.0, 0.0), (0.0, -1.0, -1.0, 0.0), (0.0, -1.0, -1.0, 0.0), (0.0, 1.0, 1.0, 0.0)]

bx = repeat(permutedims(v),np)
∇bx = repeat(permutedims(g),np)
Hbx = repeat(permutedims(h),np)
test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# Real-valued Q space with anisotropic order

orders = (1,2)
V = Float64
G = gradient_type(V,xi)
b = IntegratedLegendreBasis{2}(V,orders)

v = V[0.25, 0.25, 0.25, 0.25, -sqrt(3)/8, -sqrt(3)/8]
# g = G[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 2.0), (0.0, 6.0), (9.0, 12.0)]

bx = repeat(permutedims(v),np)
# ∇bx = repeat(permutedims(g),np)
# test_field_array(b,x,bx,cmp=(.≈))

# Vector-valued Q space with isotropic order

order = 1
V = VectorValue{3,Float64}
G = gradient_type(V,xi)
H = gradient_type(G,xi)
b = IntegratedLegendreBasis{2}(V,order)

v = V[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]

g = G[[-0.5 0.0 0.0; -0.5 0.0 0.0], [0.0 -0.5 0.0; 0.0 -0.5 0.0],
      [0.0 0.0 -0.5; 0.0 0.0 -0.5], [0.5 0.0 0.0; -0.5 0.0 0.0],
      [0.0 0.5 0.0; 0.0 -0.5 0.0], [0.0 0.0 0.5; 0.0 0.0 -0.5],
      [-0.5 0.0 0.0; 0.5 0.0 0.0], [0.0 -0.5 0.0; 0.0 0.5 0.0],
      [0.0 0.0 -0.5; 0.0 0.0 0.5], [0.5 0.0 0.0; 0.5 0.0 0.0],
      [0.0 0.5 0.0; 0.0 0.5 0.0], [0.0 0.0 0.5; 0.0 0.0 0.5]]

h = H[
      (0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0),
      (0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0),
      (0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0),
      (0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0)]

bx = repeat(permutedims(v),np)
∇bx = repeat(permutedims(g),np)
Hbx = repeat(permutedims(h),np)
test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# order = 1
# b = MonomialBasis{1}(Float64,order)
# @test evaluate(b,Point{1,Float64}[(0,),(1,)]) == [1.0 0.0; 1.0 1.0]

# b = MonomialBasis{0}(VectorValue{2,Float64},order)
# @test evaluate(b,Point{0,Float64}[(),()]) == VectorValue{2,Float64}[(1.0, 0.0) (0.0, 1.0); (1.0, 0.0) (0.0, 1.0)]

end # module
