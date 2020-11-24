struct Mode <: Dof end

struct IntegratedLegendreDofBasis{P,V,T} <: AbstractVector{Mode}
  lag_dof_basis::LagrangianDofBasis{P,V}
  change_of_basis::Matrix{T}
end

function IntegratedLegendreDofBasis(::Type{T},p::Polytope,b::IntegratedLegendreBasis) where T
  lag_nodes, _ = compute_nodes(p,b.orders)
  lag_dof_basis = LagrangianDofBasis(T,lag_nodes)
  change_of_basis = inv(evaluate(lag_dof_basis,b))
  IntegratedLegendreDofBasis(lag_dof_basis,change_of_basis)
end

@inline Base.size(a::IntegratedLegendreDofBasis) = size(a.lag_dof_basis)
@inline Base.axes(a::IntegratedLegendreDofBasis) = axes(a.lag_dof_basis)
@inline Base.getindex(a::IntegratedLegendreDofBasis,i::Integer) = Mode()
@inline Base.IndexStyle(::IntegratedLegendreDofBasis) = IndexLinear()

function return_cache(b::IntegratedLegendreDofBasis,field)
  return_cache(b.lag_dof_basis,field)
end

@inline function evaluate!(cache,b::IntegratedLegendreDofBasis,field)
  c, cf = cache
  vals = evaluate!(cf,field,b.lag_dof_basis.nodes)
  vals = evaluate!(c,*,b.change_of_basis,vals)
  ndofs = length(b.lag_dof_basis.dof_to_node)
  T = eltype(vals)
  ncomps = num_components(T)
  @check ncomps == num_components(eltype(b.lag_dof_basis.node_and_comp_to_dof)) """\n
  Unable to evaluate LagrangianDofBasis. The number of components of the
  given Field does not match with the LagrangianDofBasis.

  If you are trying to interpolate a function on a FESpace make sure that
  both objects have the same value type.

  For instance, trying to interpolate a vector-valued funciton on a scalar-valued FE space
  would raise this error.
  """
  _evaluate_lagr_dof!(c,vals,b.lag_dof_basis.node_and_comp_to_dof,ndofs,ncomps)
end

function _IL_nodes_and_moments!(p::Polytope, b::IntegratedLegendreBasis{D,T}) where {D,T}

  @notimplementedif ! is_n_cube(p)
  @assert D == num_dims(p)

  # Compute quadrature on reference cell
  orders = b.orders
  maxorder = maximum(orders)
  degree = 2*maxorder
  quad = Quadrature(p,degree)
  cips = get_coordinates(quad)
  wips = get_weights(quad)

  # Compute nodes
  pt = Point{D,T}
  ft = VectorValue{D,T}
  nodes = vcat(cips)

  # Compute face nodes
  nnodes = num_faces(p,0)
  nfaces = num_faces(p)
  f_nodes = fill(1:length(cips),nfaces-nnodes)

  # Compute moments
  f_moments = fill(zeros(ft,0,0),nfaces-nnodes)

  # # Permute basis functions

  # # Generate indices of n-faces and order s.t.
  # # (1) dimension-increasing (2) lexicographic
  bin_rang_nfaces = tfill(0:1,Val{D}())
  bin_ids_nfaces = collect(Iterators.product(bin_rang_nfaces...))
  sum_bin_ids_nfaces = [sum(bin_ids_nfaces[i]) for i in eachindex(bin_ids_nfaces)]
  bin_ids_nfaces = permute!(bin_ids_nfaces,sortperm(sum_bin_ids_nfaces))

  # # Generate LIs of shapefuns s.t. order by n-faces
  lids_b = LinearIndices(Tuple([orders[i]+1 for i=1:D]))

  eet = eltype(eltype(bin_ids_nfaces))
  f(x) = Tuple( x[i] == one(eet) ? (0:0) : (1:2) for i in 1:length(x) )
  g(x) = Tuple( x[i] == one(eet) ? (3:orders[i]+1) : (0:0) for i in 1:length(x) )
  rang_nfaces = map(f,bin_ids_nfaces)
  rang_own_dofs = map(g,bin_ids_nfaces)

  P = Int64[]
  for i = 1:length(bin_ids_nfaces)
    cis_nfaces = CartesianIndices(rang_nfaces[i])
    cis_own_dofs = CartesianIndices(rang_own_dofs[i])
    for ci in cis_nfaces
      ci = ci .+ cis_own_dofs
      P = vcat(P,reshape(lids_b[ci],length(ci)))
    end
  end

  permute!(b.terms,P)

  nodes
end

# function IntegratedLegendreRefFE(::Type{T},p::Polytope{D},orders) where {T,D}

#   prebasis = IntegratedLegendreBasis{D}(T,orders)
#   nodes, f_moments, f_nodes = _IL_nodes_and_moments!(p,prebasis)

#   dofs = MomentBasedDofBasis(nodes,f_moments,f_nodes)
#   ∇pb = Broadcasting(∇)(prebasis)
#   evaluate(dofs,∇pb)

# end
