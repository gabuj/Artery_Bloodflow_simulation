using Gridap
using GridapGmsh

inputfile = "models/cylinder_lighter.msh"
model=DiscreteModelFromFile(inputfile)

#write in vtk
outputfile = "models/cylinder_model"
writevtk(model,outputfile)