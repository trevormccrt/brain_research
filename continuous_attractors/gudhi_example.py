import gudhi
points=[[1, 1, 4], [7, 0,5], [4, 6,12], [9, 6,0], [0, 14,8], [2, 19,2], [9, 17, 12]]
x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]
import matplotlib.pyplot as plt
proj_fig = plt.figure()
proj_axs = proj_fig.add_subplot(projection="3d")
proj_axs.scatter(x, y, z)
plt.show()

rips_complex = gudhi.RipsComplex(points=points,
                                 max_edge_length=50.0)

simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)
fmt = '%s -> %.2f'
for filtered_value in simplex_tree.get_filtration():
    print(fmt % tuple(filtered_value))
