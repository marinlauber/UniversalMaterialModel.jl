using FerriteDistributed
using HYPRE, Metis
using UniversalMaterialModel

function assemble_element!(ke, ge, cell, cv, fv, mat, ue, ΓN)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = mat(C)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function distributed_assemble!(K, g, dh, cv, fv, mat, u, ΓN, ch)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    # start_assemble resets K and g
    assembler = start_assemble(K, g)
    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        local_dofs = celldofs(cell)
        global_dofs = dh.ldof_to_gdof[local_dofs]
        ue = u[local_dofs] # element dofs
        assemble_element!(ke, ge, cell, cv, fv, mat, ue, ΓN)
        # TODO this changes
        apply_local!(ke, ge, local_dofs, ch; apply_zero=true)
        assemble!(assembler, global_dofs, ke, ge)
    end
    end_assemble(assembler)
end

# init MPI
MPI.Init()
HYPRE.Init()

# Generate a grid
N = 16
L = 1.0
left = zero(Vec{3})
right = L * ones(Vec{3})
grid = generate_nod_grid(MPI.COMM_WORLD, Hexahedron, (N, N, N), left, right;
                         partitioning_alg=FerriteDistributed.PartitioningAlgorithm.Metis(:RECURSIVE));

# Material parameters
E = 10.0
ν = 0.3
μ = E / (2(1 + ν))
λ = (E * ν) / ((1 + ν) * (1 - 2ν))

# NeoHook model tab
terms = [(1,1,1,1,1.0,1.0,μ/2),
         (3,1,2,1,1.0,1.0,λ/2)]
mat = UniversalMaterialModel.build_material(terms)

# Finite element base
ip = Lagrange{RefHexahedron, 1}()^3
qr = QuadratureRule{RefHexahedron}(2)
qr_facet = FacetQuadratureRule{RefHexahedron}(2)
cv = CellValues(qr, ip)
fv = FacetValues(qr_facet, ip)

# DofHandler
dh = DofHandler(grid)
add!(dh, :u, ip) # Add a displacement field
close!(dh)

# rotation of the face
function rotation(X, t)
    θ = pi / 2.0 # 90°
    x, y, z = X
    return Vec{3}((-t,L/2-y+(y-L/2)*cos(θ*t)-(z-L/2)*sin(θ*t),L/2-z+(y-L/2)*sin(θ*t)+(z-L/2)*cos(θ*t)))
end

dbcs = ConstraintHandler(dh)
# Add a homogeneous boundary condition on the "clamped" edge
add!(dbcs, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
add!(dbcs, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> rotation(x, t), [1, 2, 3]))
close!(dbcs)

# Neumann part of the boundary
ΓN = union(
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
    getfacetset(grid, "front"),
    getfacetset(grid, "back"),
)

# --------------------- Distributed assembly --------------------
# The synchronization with the global sparse matrix is handled by
# an assembler again. You can choose from different backends, which
# are described in the docs and will be expanded over time. This call
# may trigger a large amount of communication.
dgrid = getglobalgrid(dh)
comm = global_comm(dgrid)
ldofrange = local_dof_range(dh)
K = HYPREMatrix(comm, first(ldofrange), last(ldofrange))
g = HYPREVector(comm, first(ldofrange), last(ldofrange))

# these can live only locally?
Ndofs = FerriteDistributed.num_local_dofs(dh)
un = zeros(Ndofs) # previous solution vector
u = zeros(Ndofs)
Δu = zeros(Ndofs)
ΔΔu = zeros(Ndofs)
g_local = zeros(Ndofs)
apply!(un, dbcs)

# Newton iterations criterion
tol = 1.0e-8
maxiter = 20

# initialize solver
precond = HYPRE.BoomerAMG()
solver = HYPRE.PCG(; Precond = precond)

# MPI helper
my_rank   = global_rank(dgrid)
owned     = dh.ldof_to_rank .== my_rank          # pre-compute once outside all loops
println("rank $my_rank")
master() = my_rank==1

# Newton Solve
let λᵢ=0; norm_res=0; @time for λ in 0.0:0.01:0.6
    # Newton solve for current displacement step
    λᵢ += 1; newton_itr = -1
    # update the boundary conditions for the current load step
    Ferrite.update!(dbcs, λ)
    while true
        newton_itr += 1
        # Construct the current guess and enforce BCs at current λ
        u .= un .+ Δu
        apply!(u, dbcs)
        # Compute residual and tangent for current guess
        distributed_assemble!(K, g, dh, cv, fv, mat, u, ΓN, dbcs)

        # Compute the residual norm and compare with tolerance via HYPRE inner product
        FerriteDistributed.extract_local_part!(g_local, g, dh)
        # normg = sqrt(MPI.Allreduce(sum(abs2, g_local), +, comm))
        normg = sqrt(MPI.Allreduce(sum(abs2, @view g_local[owned]), MPI.SUM, comm))
        master() && (@show normg)

        # check conv or exit
        normg < tol && (norm_res=normg; break)
        newton_itr > maxiter && error("Reached maximum Newton iterations, aborting at $normg")
        # Compute Newton increment via direct solve
        ΔΔu_h = HYPRE.solve(solver, K, g)
        FerriteDistributed.extract_local_part!(ΔΔu, ΔΔu_h, dh);
        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu
    end
    master() && println("Load step λ=$(round(λ; digits=2)) converged in $newton_itr iterations to $norm_res")
    # Commit converged solution and reset increment for next load step
    un .= u
    fill!(Δu, 0.0)
end;
end

PVTKGridFile("block_distributed", dh) do vtk
    write_solution(vtk, dh, u)
    # For debugging purposes it can be helpful to enrich
    # the visualization with some meta  information about
    # the grid and its partitioning
    vtk_shared_vertices(vtk, dgrid)
    vtk_shared_faces(vtk, dgrid)
    vtk_partitioning(vtk, dgrid)
end