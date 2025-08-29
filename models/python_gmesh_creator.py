import gmsh

def generate_cylinder_mesh(radius=1.0, height=2.0, mesh_size=0.1, filename="first_steps/models/cylinder_lighter.msh"):
    gmsh.initialize()
    gmsh.model.add("cylinder")

    # Create a cylinder volume
    cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
    gmsh.model.occ.synchronize()

    # Get surfaces of the cylinder
    surfaces = gmsh.model.getBoundary([(3, cyl)], oriented=False, recursive=False)

    inlet, outlet, wall = None, None, []

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x, y, z = com

        if abs(z - 0.0) < 1e-6:        # bottom
            inlet = tag
        elif abs(z - height) < 1e-6:   # top
            outlet = tag
        else:                          # cylindrical wall
            wall.append(tag)

    gmsh.model.occ.synchronize()

    # Add physical groups
    if inlet:
        gmsh.model.addPhysicalGroup(2, [inlet], name="inlet")
    if outlet:
        gmsh.model.addPhysicalGroup(2, [outlet], name="outlet")
    if wall:
        gmsh.model.addPhysicalGroup(2, wall, name="wall")

    # Add the volume as "fluid"
    gmsh.model.addPhysicalGroup(3, [cyl], name="fluid")

    # Mesh refinement
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)

    gmsh.finalize()

if __name__ == "__main__":
    generate_cylinder_mesh()
