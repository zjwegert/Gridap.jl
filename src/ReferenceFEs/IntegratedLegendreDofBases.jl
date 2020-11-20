function _IL_nodes_and_moments(p::Polytope, pb::IntegratedLegendreBasis{D,T}) where {D,T}

  @notimplementedif ! is_n_cube(p)
  @assert D == num_dims(p)

  pt = Point{D,T}
  nf_nodes = [ zeros(pt,0) for face in 1:num_faces(p)]
  nf_moments = [ zeros(T,0,0) for face in 1:num_faces(p)]

  # Compute vertex (aka nodal or external) moments
  d = 0
  vcips = get_vertex_coordinates(p)
  vrange = get_dimrange(p,d)
  nf_nodes[vrange] = [ [vcips[vertex]] for vertex in 1:length(vrange) ]
  nf_moments[vrange] = fill(ones(T,1,1),length(vrange))

  # ecips, emoments = _Nedelec_edge_values(p,T,order)
  # erange = get_dimrange(p,1)
  # nf_nodes[erange] = ecips
  # nf_moments[erange] = emoments

  # if ( num_dims(p) == 3 && order > 0)

  #   fcips, fmoments = _Nedelec_face_values(p,T,order)
  #   frange = get_dimrange(p,D-1)
  #   nf_nodes[frange] = fcips
  #   nf_moments[frange] = fmoments

  # end

  # if (order > 0)

  #   ccips, cmoments = _Nedelec_cell_values(p,T,order)
  #   crange = get_dimrange(p,D)
  #   nf_nodes[crange] = ccips
  #   nf_moments[crange] = cmoments

  # end

  nf_nodes, nf_moments
end

function _IL_vertex_values(p,T)

end
