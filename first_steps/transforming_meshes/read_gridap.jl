using Gridap
using GridapGmsh

model=DiscreteModelFromFile("Artery_meshes/vtu_meshes/C024_fine.msh")

#write in vtk
writevtk(model,"Artery_meshes/gridap_outputs/C024_fine")