module CellDataTests

using Test

@testset "CellPoints" begin include("CellPointsTests.jl") end

@testset "CellMaps" begin include("CellMapsTests.jl") end

@testset "CellFields (1/2)" begin include("CellFieldsTests_1.jl") end

@testset "CellFields (2/2)" begin include("CellFieldsTests_2.jl") end

@testset "CellQuadratures" begin include("CellQuadraturesTests.jl") end

@testset "QPointCellFields" begin include("QPointCellFieldsTests.jl") end

@testset "CellDofs" begin include("CellDofsTests.jl") end

@testset "AttachDirichlet" begin include("AttachDirichletTests.jl") end

@testset "AttachConstraints" begin include("AttachConstraintsTests.jl") end

@testset "Law" begin include("LawTests.jl") end

@testset "CellContributions" begin include("CellContributionsTests.jl") end

end # module