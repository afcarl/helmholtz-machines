
import numpy as np

# D-Wave Proprietary and Confidential Information


def make_chimera_mask(num_unit_cell_rows=8, num_unit_cell_cols=8, unit_cell_size=8):
    """Create numpy array corresponding to J_mat, with 1's denoting active connections in the chimera graph, and 0's
    elsewhere.  unit_cell_size is the total number of qubits in a unit cell; not just one half of it.
    On the D-Wave machine, unit_cell_size=8.  The largest current chips have num_unit_cell_rows=num_unit_cell_cols=12"""
    assert isinstance(num_unit_cell_rows, int)
    assert isinstance(num_unit_cell_cols, int)
    assert isinstance(unit_cell_size, int)
    assert unit_cell_size % 2 == 0  # we need to be able to split the unit cell into two halves
    vars_per_side = num_unit_cell_rows * num_unit_cell_cols * unit_cell_size/2
    vars_per_row = num_unit_cell_cols * unit_cell_size/2
    vars_per_col = unit_cell_size/2

    def get_index(r, c, i):
        return r * vars_per_row + c * vars_per_col + i

    mask = np.zeros((vars_per_side, vars_per_side))
    for r in range(num_unit_cell_rows):
        for c in range(num_unit_cell_cols):
            # left qubits are connected either vertically or horizontally.  Left qubits in vertically or horizontally
            # adjacent unit cells have opposite connectivity.  The left qubit with index 0 has vertical connectivity
            vert_or_horiz = (r + c) % 2
            for i in range(unit_cell_size/2):
                left = get_index(r=r, c=c, i=i)
                for j in range(unit_cell_size/2):
                    right = get_index(r=r, c=c, i=j)
                    mask[left, right] = 1
                    # print "connect r:", r, "c:", c, "i:", i, "to r:", r, "c:", c, "j:", j, "indices:", left, right
                if (vert_or_horiz == 1) and (c > 0):
                    right = get_index(r=r, c=c-1, i=i)
                    mask[left, right] = 1
                    # print "connect r:", r, "c:", c, "i:", i, "to r:", r, "c:", c-1, "j:", i, "indices:", left, right
                if (vert_or_horiz == 1) and (c < num_unit_cell_cols-1):
                    right = get_index(r=r, c=c+1, i=i)
                    mask[left, right] = 1
                    # print "connect r:", r, "c:", c, "i:", i, "to r:", r, "c:", c+1, "j:", i, "indices:", left, right
                if (vert_or_horiz == 0) and (r > 0):
                    right = get_index(r=r-1, c=c, i=i)
                    mask[left, right] = 1
                    # print "connect r:", r, "c:", c, "i:", i, "to r:", r-1, "c:", c, "j:", i, "indices:", left, right
                if (vert_or_horiz == 0) and (r < num_unit_cell_rows-1):
                    right = get_index(r=r+1, c=c, i=i)
                    mask[left, right] = 1
                    # print "connect r:", r, "c:", c, "i:", i, "to r:", r+1, "c:", c, "j:", i, "indices:", left, right
    return mask
