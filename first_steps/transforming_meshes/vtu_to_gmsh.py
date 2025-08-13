import meshio
import numpy as np
import os
from collections import defaultdict

# --- Configuration ---
# Specify the path to your input VTU file.
INPUT_FILENAME= "Artery_meshes/vtu_meshes/C024_fine.vtu"

# Specify the desired name for the output GMSH file.
OUTPUT_FILENAME = "Artery_meshes/vtu_meshes/C024_fine.msh"



# IMPORTANT: Specify the name of the cell data array in your VTU file that
# contains the integer tags (0, 1, 2, 3). You may need to open your VTU file
# in a viewer like ParaView to find this name. As per your feedback, this is
# set to "CellEntityIds".
VTU_TAG_ARRAY_NAME = "CellEntityIds" 

def convert_vtu_to_gmsh(input_file, output_file, tag_name):
    """
    Reads a VTU mesh, separates cell blocks,
    and writes a GMSH file with corresponding physical groups.

   (It also consolidates all cells of the same type (e.g., all triangles)
    into a single cell block to improve compatibility with Gridap)
    """
    # --- 1. Read the input mesh file ---
    print(f"Reading mesh from '{input_file}'...")
    mesh = meshio.read(input_file)
    print("Successfully read mesh.")

    print(f"  Points: {len(mesh.points)}")
    print(f"  Cell blocks: {len(mesh.cells)}")
    for i, cell_block in enumerate(mesh.cells):
        print(f"    - Block {i}: {cell_block.type}, {len(cell_block.data)} cells")

    # --- 2. Check for the existence of the tag array ---
    if tag_name not in mesh.cell_data:
        print(f"\nError: Cell data array named '{tag_name}' not found in the VTU file.")
        print("Available cell data arrays are:", list(mesh.cell_data.keys()))
        print("\nPlease update the 'VTU_TAG_ARRAY_NAME' variable in this script and run again.")
        return


    # --- 3. Consolidate cells by type and gather physical group info ---
    print("\nConsolidating cell blocks by type (e.g., all triangles in one block)...")
    
    # {cell_type: [list of cell connectivities]}
    consolidated_connectivity = defaultdict(list)
    # {cell_type: [list of tags for each cell]}
    consolidated_tags = defaultdict(list) 
    
    # {tag: dimension} for each physical group
    physical_groups_info = {} 
    # Mapping from cell type to its geometric dimension
    dimension_map = {"tetra": 3, "triangle": 2, "line": 1}
    # Optional: map tags to human-readable names for the GMSH file
    physical_name_map = {
        1000: "volurme",
        101: "walls",
        102: "inlet",
        103: "outlet1",
        104: "outlet2",
        105: "outlet3",
    }

    # Offset tag IDs by dimension to avoid collision
    TAG_OFFSET = {
        "line": 0,
        "triangle": 100,
        "tetra": 1000,
    }


    # Iterate through the original cell blocks and their corresponding tag arrays
    for cell_block, tags_for_block in zip(mesh.cells, mesh.cell_data[tag_name]):
        cell_type = cell_block.type
        offset = TAG_OFFSET.get(cell_type, 0)

        for cell_conn, tag in zip(cell_block.data, tags_for_block):
            tag_with_offset = int(tag) + offset
            consolidated_connectivity[cell_type].append(cell_conn)
            consolidated_tags[cell_type].append(tag_with_offset)

            if tag_with_offset not in physical_groups_info:
                dim = dimension_map.get(cell_type)
                if dim is not None:
                    physical_groups_info[tag_with_offset] = dim

    # --- 4. Build the new mesh object for writing ---
    # The points (nodes) of the mesh remain the same
    points = mesh.points
    new_cells = []
    cell_data_list_for_gmsh = []

    # Sort by key (cell_type) for a consistent output order
    for cell_type in sorted(consolidated_connectivity.keys()):
        # Create one new cell block per cell type
        connectivity_array = np.vstack(consolidated_connectivity[cell_type])
        new_cells.append(meshio.CellBlock(cell_type, connectivity_array))
        
        # Create the corresponding flat array of tags for this new block
        tags_array = np.array(consolidated_tags[cell_type], dtype=int)
        cell_data_list_for_gmsh.append(tags_array)

    # Meshio requires cell data for GMSH physical groups to be under the key "gmsh:physical".
    # This is now a list of arrays, where each array corresponds to a consolidated cell block.
    cell_data_for_gmsh = {
        "gmsh:physical": cell_data_list_for_gmsh,
        "gmsh:geometrical": cell_data_list_for_gmsh,
    }

    # Create field data for the GMSH $PhysicalNames section.
    field_data_for_gmsh = {
        physical_name_map.get(tag, f"physical_{tag}"): np.array([dim, tag], dtype=int)
        for tag, dim in physical_groups_info.items()
    }
    #invert order of file data for GMSH ([a,b] -> [b,a])
    field_data_for_gmsh = {k: v[::-1] for k, v in field_data_for_gmsh.items()}

    print("Physical groups info for GMSH:", field_data_for_gmsh)

    # Point data is not strictly required by all parsers but can be included.
    point_data = {
        "gmsh:dim_tags": np.array([[0, 0]] * len(points), dtype=int)
    }

    output_mesh = meshio.Mesh(
        points=points,
        cells=new_cells,
        cell_data=cell_data_for_gmsh,
        point_data=point_data,
        field_data=field_data_for_gmsh
    )

    print("\nFinal mesh structure for output:")
    print(output_mesh)

    # --- 5. Write the final GMSH file ---
    print(f"\nWriting GMSH file to '{output_file}'...")
    # We specify file_format="gmsh22" to ensure it writes the v2.2 ASCII format.
    meshio.write(
        output_file,
        output_mesh,
        file_format="gmsh22",
    )


    print("\nConversion complete!")
    print(f"The file '{output_file}' has been created successfully.")


if __name__ == '__main__':
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    convert_vtu_to_gmsh(INPUT_FILENAME, OUTPUT_FILENAME, VTU_TAG_ARRAY_NAME)
