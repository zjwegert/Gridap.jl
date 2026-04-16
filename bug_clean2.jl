using Gridap,Gridap.Algebra,Gridap.Arrays,Gridap.Geometry,
  Gridap.CellData,Gridap.ReferenceFEs,Gridap.FESpaces,Gridap.MultiField

domain = (0,1,0,1)
partition = (5,5)
order = 2
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
dΓ = Measure(Γ,2order)
dΓ_i = Measure(Γ_i,2order)
Ω = Triangulation(model)
dΩ = Measure(Ω,2order)
V1 = FESpace(model,ReferenceFE(lagrangian,Float64,order),vector_type=Vector{ComplexF64},conformity=:L2)
V2 = FESpace(Γ,ReferenceFE(lagrangian,Float64,order),vector_type=Vector{ComplexF64})
V1V2 = MultiFieldFESpace([V1,V2])

AffineFEOperator(((φ,w),(v,r)) -> ∫(jump(φ)*mean(r))dΓ_i,((v1,v2),)->0,V1V2,V1V2)
AffineFEOperator(((φ,w),(v,r)) -> ∫(mean(w)*jump(v) )dΓ_i,((v1,v2),)->0,V1V2,V1V2)

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

u(x) = x[1]^2 + x[2]
p(x) = x[1]^2 + x[2]^2
∇u(x) = VectorValue( 2*x[1], one(x[2]) )
∇p(x) = VectorValue( 2*x[1], 2*x[2] )
Δu(x) = 2
Δp(x) = 2 # Surface laplacian
f(x) = - Δu(x) + u(x)
g(x) = - Δp(x) + u(x)
∇(::typeof(u)) = ∇u
∇(::typeof(p)) = ∇p

V1 = FESpace(model,ReferenceFE(lagrangian,Float64,order),conformity=:L2)
V2 = FESpace(Γ,ReferenceFE(lagrangian,Float64,order),dirichlet_tags="∂Γ")
V1V2 = MultiFieldFESpace([V1,V2])
U2 = TrialFESpace(V2,p)
U1U2 = MultiFieldFESpace([V1,U2])

∂Ω = BoundaryTriangulation(model,tags=["boundary"])
d∂Ω = Measure(∂Ω,2order)
n_∂Ω = get_normal_vector(∂Ω)
Λ = Skeleton(model,tags=["interior","Γ"])
n_Λ = get_normal_vector(Λ)
dΛ = Measure(Λ,2order)

hmin = mean(CellField(lazy_map(vol->(vol)^(1/2),get_cell_measure(Ω)),Ω))
γ = 10

a((u,p),(v,q)) =
  ∫( ∇(v)⋅∇(u) + u*v)dΩ +
  ∫( (γ/hmin)*jump(v*n_Λ)⋅jump(u*n_Λ) - jump(v*n_Λ)⋅mean(∇(u)) -  mean(∇(v))⋅jump(u*n_Λ) )*dΛ + 
  ∫( ∇(q)⋅∇(p) + q*u)dΓ

l((v,q)) =
  ∫( v*f )*dΩ + ∫(v*(n_∂Ω⋅∇u))d∂Ω + 
  ∫( q*g )dΓ

op = AffineFEOperator(a,l,U1U2,V1V2)

uh,ph = solve(op)

l2_u(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1_u(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))
l2_p(p) = sqrt(sum( ∫( p⊙p )*dΓ ))
# Subtract the squared normal component from the squared ambient gradient
n_Γ = get_normal_vector(Γ)
tangent_sq_grad(q) = ∇(q)⊙∇(q) - (∇(q)⋅n_Γ)*(∇(q)⋅n_Γ)
h1_p(p) = sqrt(sum( ∫( p⊙p + tangent_sq_grad(p) )*dΓ ))

el2_u = l2_u(u - uh)
eh1_u = h1_u(u - uh)
ul2_u = l2_u(uh)
uh1_u = h1_u(uh)
el2_p = l2_p(p - ph)
eh1_p = h1_p(p - ph)
ul2_p = l2_p(ph)
uh1_p = h1_p(ph)

@test el2_u/ul2_u < 1.e-8
@test eh1_u/uh1_u < 1.e-7
@test el2_p/ul2_p < 1.e-8
@test eh1_p/uh1_p < 1.e-7

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