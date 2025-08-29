using Gridap
using GridapGmsh

model=DiscreteModelFromFile("first_steps/Artery_meshes/vtu_meshes/C024_fine.msh")

#write in vtk
writevtk(model,"first_steps/Artery_meshes/gridap_outputs/C024_fine")