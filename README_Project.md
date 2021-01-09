# Accelerated Mesh Implementation
We used an AABB tree structure for this assignment. In order to organize our meshes, we did the following: 

- Extracted all triangles from the list of meshes and kept this in a list 
- For each triangle, stored the mesh index + the index of the triangle within the mesh and kept this in a list with corresponding indices 
- Stored, for each node in the tree, a minimum + maximum value (bounding box) as well as either a start and end index within the larger list of triangles (for leaf nodes) or a left and a right node 

In order to build our tree, we pretty much followed the instructions in the writeup exactly. We passed in a start + end index, representing a slice of the larger triangle list. We then computed the bounding box for this slice. If the length of the start - end slice was less than n_triangles (set, for us, to 100), we created a leaf node with the start and end indices and the bounding box. Otherwise, we sorted the triangle slice (as well as its corresponding mesh/index slice, to keep things synchronous) by the longest axis in the bounding box, and split by the median of the list. (A note about this - we actually tried computing equal surface area splits, but the median ended up being faster, so we just used that.) Then, we sliced by the median again and recurisvely created left and right nodes. 

Everything else was pretty straightforward and didn't require much thought into the design. We played around with many designs, but this way of storing the triangles as-is worked the fastest for us. It does require more space than if we'd just kept a list of mesh indices/triangle indices, but the latter way was significantly slower for us, so we felt justified in making this decision. 
  
# Scene Design
For our scene, we decided to depict a Zen Garden using various meshes and textures. 
Firstly, our nature-like image consists of many colorful trees. Some are in a black, rectangular pot, while the remaining are on gray soil; there is also a blue cloud in the sky. To add on, the scene 
also includes a Buddha statue and some bunnies, which accentuate the nature-like and Zen aspect of the scene. 
We built the trees by loading the tree images and downloading a various flower-pattern images from Google Images, which we used as tree meshes. This resulted in the trees being extremely colorful and flowery. In order to construct the black, rectangular pot and the blue cloud in the sky, we simply used the box object with differing lighting settings. Lastly, we also used the bunny and the Buddha statue through the bunny and Buddha objects, respectively.
