using Gridap,Gridap.Algebra,Gridap.Arrays,Gridap.Geometry,
  Gridap.CellData,Gridap.ReferenceFEs,Gridap.FESpaces,Gridap.MultiField

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
cell_to_entity = labels.d_to_dface_to_entity[end]
entity = maximum(cell_to_entity) + 1
labels.d_to_dface_to_entity[1][[9,15,21,27]] .= entity
labels.d_to_dface_to_entity[2][[21,32,43]] .= entity
add_tag!(labels,"Γ",[entity])
entity = maximum(cell_to_entity) + 2
labels.d_to_dface_to_entity[1][[9,27]] .= entity
add_tag!(labels,"∂Γ",[entity])
Γ = BoundaryTriangulation(model,tags=["Γ","∂Γ"])
Γ_i = Skeleton(model,tags=["Γ","∂Γ"])
dΓ = Measure(Γ,4)
dΓ_i = Measure(Γ_i,4)
Ω = Triangulation(model)
dΩ = Measure(Ω,4)
V1 = FESpace(model,ReferenceFE(lagrangian,Float64,2),vector_type=Vector{ComplexF64},conformity=:L2)
V2 = FESpace(Γ,ReferenceFE(lagrangian,Float64,2),vector_type=Vector{ComplexF64})
V1V2 = MultiFieldFESpace([V1,V2])

AffineFEOperator(((φ,w),(v,r)) -> ∫(jump(φ)*mean(r))dΓ_i,((v1,v2),)->0,V1V2,V1V2)
AffineFEOperator(((φ,w),(v,r)) -> ∫(mean(w)*jump(v) )dΓ_i,((v1,v2),)->0,V1V2,V1V2)



using Test
import Gridap: ∇

u(x) = x[1]^2 + x[2]
p(x) = x[1]^2 + x[2]^2
∇u(x) = VectorValue( 2*x[1], one(x[2]) )
∇p(x) = VectorValue( 2*x[1], 2*x[2] )
Δu(x) = 2
Δp(x) = 4
f(x) = - Δu(x) + p(x)
g(x) = - Δp(x) + p(x)
∇(::typeof(u)) = ∇u
∇(::typeof(p)) = ∇p

V = TestFESpace(model,ReferenceFE(lagrangian,Float64,2),conformity=:L2)
U = TrialFESpace(V)

Γ_bdry = BoundaryTriangulation(model,tags=["boundary"])
dΓ_bdry = Measure(Γ_bdry,2)
n_Γ_bdry = get_normal_vector(Γ_bdry)

Λ = Skeleton(model,tags=["interior","Γ"])
n_Λ = get_normal_vector(Λ)
dΛ = Measure(Λ,4)

hmin = mean(CellField(lazy_map(vol->(vol)^(1/2),get_cell_measure(Ω)),Ω))
γ = 10

a(u,v) =
  ∫( ∇(v)⋅∇(u) + v*u)dΩ +
  ∫( (γ/hmin)*jump(v*n_Λ)⋅jump(u*n_Λ) - jump(v*n_Λ)⋅mean(∇(u)) -  mean(∇(v))⋅jump(u*n_Λ) )*dΛ

l(v) =
  ∫( v*f )*dΩ + ∫(v*(n_Γ_bdry⋅∇u))dΓ_bdry

op = AffineFEOperator(a,l,U,V)

uh = solve(op)

e = u - uh

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7


using Gridap,Gridap.Algebra,Gridap.Arrays,Gridap.Geometry,
  Gridap.CellData,Gridap.ReferenceFEs,Gridap.FESpaces,Gridap.MultiField
using Test
import Gridap: ∇

domain = (0,1,0,1)
partition = (5,5)
order = 2
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
cell_to_entity = labels.d_to_dface_to_entity[end]
entity = maximum(cell_to_entity) + 1
labels.d_to_dface_to_entity[1][[15,21]] .= entity
labels.d_to_dface_to_entity[2][[21,32,43]] .= entity
add_tag!(labels,"Γ",[entity])
entity = maximum(cell_to_entity) + 2
labels.d_to_dface_to_entity[1][[9,27]] .= entity
add_tag!(labels,"∂Γ",[entity])
Γ = BoundaryTriangulation(model,tags=["Γ","∂Γ"])
dΓ = Measure(Γ,2order)
Ω = Triangulation(model)
dΩ = Measure(Ω,2order)

p(x) = x[1]^2 + x[2]^2
∇p(x) = VectorValue( 2*x[1], 2*x[2] )
Δp(x) = 2 # Surface laplacian
g(x) = - Δp(x) + p(x)
∇(::typeof(p)) = ∇p

V = TestFESpace(Γ,ReferenceFE(lagrangian,Float64,2))
U = TrialFESpace(V)

∂Γ = BoundaryTriangulation(Γ,tags=["∂Γ"])
d∂Γ = Measure(∂Γ,2order)
n_∂Γ = get_normal_vector(∂Γ)

a(p,q) = ∫( ∇(q)⋅∇(p) + q*p)dΓ
l(q) = ∫( q*g )dΓ + ∫(q*(n_∂Γ⋅∇p))d∂Γ

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

e = p - uh

l2(p) = sqrt(sum( ∫( p⊙p )*dΓ ))
# Subtract the squared normal component from the squared ambient gradient
n_Γ = get_normal_vector(Γ)
tangent_sq_grad(q) = ∇(q)⊙∇(q) - (∇(q)⋅n_Γ)*(∇(q)⋅n_Γ)
h1(p) = sqrt(sum( ∫( p⊙p + tangent_sq_grad(p) )*dΓ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7

############# NOTE:
# There is nothing wrong with the Gridap framework itself here, but you have run into a classic mathematical "gotcha" when solving PDEs on embedded manifolds (surfaces or curves embedded in a higher-dimensional space).

# Your tests are failing because of two geometrical mismatches between the ambient 2D space and the 1D manifold ($\Gamma$) you are solving on.

# ### 1. The Surface Laplacian (Laplace-Beltrami Operator)
# In your analytical setup, you defined the ambient 2D Laplacian for $p(x_1, x_2) = x_1^2 + x_2^2$:
# ```julia
# Δp(x) = 4
# ```
# However, you are solving the PDE $-\Delta_\Gamma p + p = g$ on a 1D boundary triangulation ($\Gamma$). The operator here is the **surface Laplacian** (Laplace-Beltrami), not the 2D ambient Laplacian.

# Because your Cartesian mesh generates $\Gamma$ purely from straight horizontal and vertical line segments, the 1D surface Laplacian is simply the second derivative along the straight segment.
# * On a horizontal segment ($x_2 = c$), $p(x_1, c) = x_1^2 + c^2$, so $\frac{\partial^2 p}{\partial x_1^2} = 2$.
# * On a vertical segment ($x_1 = c$), $p(c, x_2) = c^2 + x_2^2$, so $\frac{\partial^2 p}{\partial x_2^2} = 2$.

# By setting `Δp(x) = 4`, you are feeding the solver an inconsistent forcing term $g(x)$. The exact solution to your manifold PDE requires **`Δp(x) = 2`**.

# ### 2. The $H^1$ Error Norm on a Manifold
# Your $L^2$ test will pass once the Laplacian is fixed, but your $H^1$ test will still fail because of how the error gradient $\nabla(e) = \nabla(p) - \nabla(u_h)$ is evaluated.

# * $\nabla(u_h)$ is evaluated by Gridap as a strictly **tangential** vector to $\Gamma$.
# * $\nabla(p) = (2x_1, 2x_2)$ is an analytical function, so Gridap correctly evaluates it as the **ambient 2D gradient**.

# The ambient gradient $\nabla p$ has non-zero components *normal* to the curve (e.g., $2x_2$ on a horizontal edge). Because $\nabla(u_h)$ has no normal component to cancel it out, $\nabla(e)$ will continuously register a large normal error. To fix this, you must project the gradient to the tangent space by subtracting the normal component when calculating your error.

# *(Note: Your weak form `a(p,q)` is perfectly fine as is, because `∇(q)` is tangential, so the dot product `∇(q)⋅∇(p)` mathematically cancels out the normal component automatically.)*

# ---

# ### The Fix

# Here are the specific lines to correct in your script.

# **Fix the Source Term:**
# ```julia
# p(x) = x[1]^2 + x[2]^2
# ∇p(x) = VectorValue( 2*x[1], 2*x[2] )
# Δp(x) = 2  # Changed from 4 to the correct surface Laplacian
# g(x) = - Δp(x) + p(x)
# ```

# **Fix the $H^1$ Error Evaluation:**
# ```julia
# # Retrieve the normal vector field for your manifold
# n_Γ = get_normal_vector(Γ)

# l2(p) = sqrt(sum( ∫( p⊙p )*dΓ ))

# # Subtract the squared normal component from the squared ambient gradient
# tangent_sq_grad(q) = ∇(q)⊙∇(q) - (∇(q)⋅n_Γ)*(∇(q)⋅n_Γ)
# h1(p) = sqrt(sum( ∫( p⊙p + tangent_sq_grad(p) )*dΓ ))

# el2 = l2(e)
# eh1 = h1(e)
# ul2 = l2(uh)
# uh1 = h1(uh)

# @test el2/ul2 < 1.e-8
# @test eh1/uh1 < 1.e-7
# ```

# With a second-order polynomial exact solution and second-order Lagrangian elements, your tests should now easily clear the tolerances and reach near machine precision.