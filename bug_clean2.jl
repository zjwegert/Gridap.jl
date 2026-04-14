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
Γ = BoundaryTriangulation(model,tags=["Γ"])
Γ_i = Skeleton(model,tags="Γ")
dΓ = Measure(Γ,2)
dΓ_i = Measure(Γ_i,2)
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V1 = FESpace(model,ReferenceFE(lagrangian,Float64,2),vector_type=Vector{ComplexF64},conformity=:L2)
V2 = FESpace(Γ,ReferenceFE(lagrangian,Float64,2),vector_type=Vector{ComplexF64})
V1V2 = MultiFieldFESpace([V1,V2])

AffineFEOperator(((φ,w),(v,r)) -> ∫(jump(φ)*mean(r))dΓ_i,((v1,v2),)->0,V1V2,V1V2)
AffineFEOperator(((φ,w),(v,r)) -> ∫(mean(w)*jump(v) )dΓ_i,((v1,v2),)->0,V1V2,V1V2)

f(x) = x[1]^2 + x[2]^2
g(x) = x[1]^2 + x[2]
op = AffineFEOperator(((φ,w),(v,r)) -> ∫(φ*v)dΩ + ∫(w*r)dΓ,((v,r),)->∫(f*v)dΩ + ∫(g*r)dΓ,V1V2,V1V2)
uh1, uh2 = solve(op)

L2(uh) = sqrt(sum(∫(uh*uh)dΩ))
L2_Γ(uh) = sqrt(sum(∫(uh*uh)dΓ))

L2(uh1-f)
L2_Γ(uh2-g)