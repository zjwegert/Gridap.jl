function _IL_nodes_and_moments(p::Polytope, b::IntegratedLegendreBasis{D,T}) where {D,T}

  @notimplementedif ! is_n_cube(p)
  @assert D == num_dims(p)

  # Compute quadrature on reference cell
  order = maximum(b.orders)
  degree = 2*order
  quad = Quadrature(p,degree)
  cips = get_coordinates(quad)
  wips = get_weights(quad)

  # Compute nodes
  pt = Point{D,T}
  vcips = get_vertex_coordinates(p)
  nodes = vcat(vcips,cips)

  # Compute face nodes
  f_nodes = vcat([i:i+1 for i = 1:num_faces(p,0)],
                 [num_faces(p,0)+1:num_faces(p,0)+length(cips)])

  # Compute moments
  f_moments = [ zeros(T,0,0) for face in 1:num_faces(p) ]

  # # Compute vertex (aka nodal) moments
  vrange = get_dimrange(p,0)
  f_moments[vrange] = fill(ones(T,1,1),length(vrange))

  # # Compute other non-nodal moments
  if order > 1

    ∇b = Broadcasting(∇)(b)
    vals = evaluate(∇b,cips)



  end

  nodes, f_moments, f_nodes
end
