SetFactory("OpenCASCADE");


// Parameters
radius = 0.5;
height = 2.0;
lc = 0.0001; // Mesh size

// Create the cylinder
Cylinder(1) = {0, 0, 0, 0, 0, height, radius};

// Create physical groups
Physical Surface("inlet") = {3};   // bottom face z = 0
Physical Surface("outlet") = {1};  // top face z = height
Physical Surface("wall") = {2};    // cylindrical lateral surface

// Volume tag for the fluid domain
Physical Volume("fluid") = {1}; // the main volume

