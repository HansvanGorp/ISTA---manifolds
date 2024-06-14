"""
This script defines a class to find all occurences of data on a hyperplane embedded in a higher dimensional space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

class DataOnPlane:
    def __init__(self, A: torch.tensor, data_region_extend: list[float,float], y_anchors: torch.tensor, K: int = 2, consider_partials = False, verbose = False, only_positive = False):
        """
        plot the data regions in the 2D space, if they intersect with that space.
        To that end, we check for each sparse x-vector (up to K-sparse) where they end up in y-space, and if that region intersects with the 2D space, we plot it.

        inputs:
        - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - data_region_extend: how much to extend the data region in the 2D space, a minimum and a maximum value
        - y_anchors: the anchor points, of shape (3, M) which define a plane in y-space
        - K: the maximum sparsity of the x-vectors to consider
        - consider_partials: if True, we considere e.g. that a plane can cut through the hyperplane as a line. If false, we disregard these cases
        - only_positive: if True, we only consider the positive side of the hyperplane
        """
        # assertions
        assert K >= 0, "K should be non-negative"
        assert K <= 2, "K should be at most 2"

        # store the inputs
        self.A = A
        self.data_region_extend = data_region_extend
        self.y_anchors = y_anchors
        self.K = K
        self.consider_partials = consider_partials

        # create a dict to store the regions
        self.regions = {}

        # get the shape of A
        self.M, self.N = A.shape
        
        # find the defining equation of the plane in y-space
        ns = null_space(y_anchors)
        if ns.shape[1] == 0:
            self.normal = None
            self.bias = None
        else:
            self.normal = torch.tensor(ns[:,0])
            self.bias = torch.tensor(np.dot(self.normal, y_anchors[0,:]))

        self.direction_1 = y_anchors[1,:] - y_anchors[0,:]
        self.direction_2 = y_anchors[2,:] - y_anchors[0,:]
        
        # step 0, check where x = all zeroes ends up
        self.origin_check()

        # step 1, check where all 1-sparse x-vectors end up, if we want those
        if K >= 1:
            self.sparse_1_check(only_positive = only_positive)
        
        # step 2, check where all 2-sparse x-vectors end up, if we want those
        if K >= 2:
            self.sparse_2_check(only_positive = only_positive)

        if verbose:
            for key, value in self.regions.items():
                print(f"{key}: {value}")
                
    # %% functions
    # create a function that maps an x-vector to a y-vector, using the matrix A
    def x_to_y(self, x: torch.tensor):
        return self.A @ x
    
    # function that checks if a point is on the hyperplane
    def check_if_on_hyperplane(self, y: torch.tensor):
        # is self.normal is None, there is no null space, so we are always on the hyperplane
        if self.normal is None:
            return 0

        else:
            # else we need to check the normal equation
            outcome = torch.dot(self.normal, y) - self.bias
            if outcome > 1e-6:
                side = 1
            elif outcome < -1e-6:
                side = -1
            else:
                side = 0

            return side
    
    # create a function that returns the hyperplane coordinates of a y-vector
    def y_to_hyperplane_coordinates(self, y: torch.tensor):
        # first we check if we are on the hyperplane at all, if not return None
        if self.check_if_on_hyperplane(y) != 0:
            return None
        
        # we are looking for a and b in such a way that this is equal to y:
        #y_reconstructed = self.y_anchors[0,:] + a[0]*self.direction_1 + a[1]*self.direction_2

        # thus to find a, b we need to solve the following system of equations:
        A = torch.stack([self.direction_1, self.direction_2], dim = 1)
        b = y - self.y_anchors[0,:]
        
        # just get the primrose inverse
        a = A.pinverse() @ b

        return a
    

    # origin check
    def origin_check(self):
        x_0 = torch.zeros(self.N)
        y_0 = self.x_to_y(x_0)
        hyperplane_pos = self.y_to_hyperplane_coordinates(y_0)

        if hyperplane_pos is not None:
            self.regions["origin"] = (hyperplane_pos,)

    # sparse 1 check
    def sparse_1_check(self, only_positive:bool = False):
        signs = [1] if only_positive else [-1,1]

        for sparse_idx in range(self.N):
            for sign in signs:
                # create the x-vector
                x = torch.zeros(self.N)
                x[sparse_idx] = 1

                # then create the line from min to max
                x_min = x * self.data_region_extend[0] * sign
                x_max = x * self.data_region_extend[1] * sign

                # create the y-vectors
                y_min = self.x_to_y(x_min)
                y_max = self.x_to_y(x_max)

                # check if the line intersects with the 2D space
                hyperplane_pos = self.y_line_intersects_with_2d_space(y_min, y_max)

                if hyperplane_pos is not None:
                    self.regions[f"x_{sparse_idx}_sign_{sign}"] = hyperplane_pos

    # create a function that check if a line intersects with the 2D space
    def y_line_intersects_with_2d_space(self, y_min: torch.tensor, y_max: torch.tensor):
        # there are 3 cases, either the line is completely outside the 2D space, or it is completely inside the 2D space, or it intersects with the 2D space
        # we can figure this out, by checking if the two points are on the same side of the hyperplane, if they are, they are either both inside or both outside
        # if they are on different sides, they intersect
        side_min = self.check_if_on_hyperplane(y_min)
        side_max = self.check_if_on_hyperplane(y_max)

        if (side_min > 0 and side_max > 0) or (side_min < 0 and side_max < 0):
            # there is no intersection
            result = None
        elif side_min == 0 and side_max == 0:
            # the line is completely inside the 2D space
            result = (self.y_to_hyperplane_coordinates(y_min), self.y_to_hyperplane_coordinates(y_max))

        elif self.consider_partials:
            # there is a single point of intersection, one do this is we are considering partials
            
                # find the intersection point
                t = torch.dot(self.normal, self.y_anchors[0,:] - y_min) / torch.dot(self.normal, y_max - y_min)
                intersection = y_min + t*(y_max - y_min)
                result = (self.y_to_hyperplane_coordinates(intersection))

        else:
            # if we are here we have found nothing
            result = None
        
        return result
            
    # sparse 2 check
    def sparse_2_check(self, only_positive:bool = False): #NOSONAR
        signs = [1] if only_positive else [-1,1]

        for sparse_idx_1 in range(self.N):
            for sparse_idx_2 in range(sparse_idx_1+1, self.N):
                for sign_1 in signs:
                    for sign_2 in signs:
                        # we now have a parrellogram, we need to check the four lines
                        x_min_min = torch.zeros(self.N)
                        x_min_min[sparse_idx_1] = self.data_region_extend[0] * sign_1
                        x_min_min[sparse_idx_2] = self.data_region_extend[0] * sign_2

                        x_min_max = torch.zeros(self.N)
                        x_min_max[sparse_idx_1] = self.data_region_extend[0] * sign_1
                        x_min_max[sparse_idx_2] = self.data_region_extend[1] * sign_2

                        x_max_min = torch.zeros(self.N)
                        x_max_min[sparse_idx_1] = self.data_region_extend[1] * sign_1
                        x_max_min[sparse_idx_2] = self.data_region_extend[0] * sign_2

                        x_max_max = torch.zeros(self.N)
                        x_max_max[sparse_idx_1] = self.data_region_extend[1] * sign_1
                        x_max_max[sparse_idx_2] = self.data_region_extend[1] * sign_2

                        # create the y-vectors
                        y_min_min = self.x_to_y(x_min_min)
                        y_min_max = self.x_to_y(x_min_max)
                        y_max_min = self.x_to_y(x_max_min)
                        y_max_max = self.x_to_y(x_max_max)

                        # check if the parrellogram intersects with the 2D space
                        hyperplane_pos_min_min = self.y_parrellogram_intersects_with_2D_space(y_min_min, y_min_max, y_max_min, y_max_max)

                        if hyperplane_pos_min_min is not None:
                            self.regions[f"x1_{sparse_idx_1}_sign_{sign_1}_x2_{sparse_idx_2}_sign_{sign_2}"] = hyperplane_pos_min_min

    # create the function that checks if a parrellogram intersects with the 2D space     
    def y_parrellogram_intersects_with_2D_space(self, y_min_min: torch.tensor, y_min_max: torch.tensor, y_max_min: torch.tensor, y_max_max: torch.tensor):
        # there are 4 cases:
        # 1. the parrellogram is completely outside the 2D space
        # 2. the parrellogram is completely inside the 2D space
        # 3. the parrellogram intersects with the 2D space at a signle corner point
        # 4. the parrellogram intersects with the 2D space as a line

        side_min_min = self.check_if_on_hyperplane(y_min_min)
        side_min_max = self.check_if_on_hyperplane(y_min_max)
        side_max_min = self.check_if_on_hyperplane(y_max_min)
        side_max_max = self.check_if_on_hyperplane(y_max_max)
        sides = [side_min_min, side_min_max, side_max_min, side_max_max]

        # case 1, the parrellogram is completely outside the 2D space
        if all([side > 0 for side in sides]) or all([side < 0 for side in sides]):
            result = None
        
        # case 2, the parrellogram is completely inside the 2D space
        elif all([side == 0 for side in sides]):
            result = (self.y_to_hyperplane_coordinates(y_min_min), self.y_to_hyperplane_coordinates(y_min_max), self.y_to_hyperplane_coordinates(y_max_min), self.y_to_hyperplane_coordinates(y_max_max))
        
        # case 3, if one side is 0 and the other three are on the same side, we have a single corner point intersection
        elif sides.count(0) == 1 and self.consider_partials:
            # find the corner point
            if side_min_min == 0:
                corner = y_min_min
            elif side_min_max == 0:
                corner = y_min_max
            elif side_max_min == 0:
                corner = y_max_min
            else:
                corner = y_max_max

            result =  self.y_to_hyperplane_coordinates(corner)
        
        # case 4, the parrellogram intersects with the 2D space as a line
        elif self.consider_partials:
            # we can figure out the intersection by checking each of the 4 lines
            intersections = []
            for y1, y2 in [(y_min_min, y_min_max), (y_min_min, y_max_min), (y_min_max, y_max_max), (y_max_min, y_max_max)]:
                intersection = self.y_line_intersects_with_2d_space(y1, y2)
                if intersection is not None:
                    intersections.append(intersection)

            # there can be a strange case where we have more than 2 intersections, this can happen if the parrellogram is on the edge of the 2D space
            # to counteract this case, we need only unique values
            intersections = list(set(intersections))

            # if we have 2 intersections, we are good
            if len(intersections) !=2:
                print("Warning: we have more than 2 intersections, this should not happen")
            result = (intersections[0], intersections[1])

        else:
            result = None

        return result
        
    # %%
    # create a function that plots the data regions
    def plot_data_regions(self, show_legend = True, colors = ["red", "blue", "green"], ax = None):
        seen = [False,False,False]
        for key, value in self.regions.items():
            # check the key to know what color to use
            # the cases are:
            # key = "origin"-> red
            # key = "x_{}_sign_{}".format(sparse_idx, sign) -> blue
            # key = "x1_{}_sign_1_x2_{}_sign_{}".format(sparse_idx_1, sparse_idx_2, sign_1, sign_2) -> green
            if key == "origin":
                color = colors[0]
                if seen[0] == False and show_legend == True:
                    seen[0] = True
                    label = "origin"
                else:
                    label = None
            elif "x_" in key:
                color = colors[1]
                if seen[1] == False and show_legend == True:
                    seen[1] = True
                    label = "1-sparse"
                else:
                    label = None
            elif "x1_" in key:
                color = colors[2]
                if seen[2] == False and show_legend == True:
                    seen[2] = True
                    label = "2-sparse"
                else:
                    label = None

            # there are three cases, either we have a single point, a line, or a parrellogram
            lw = 8
            ms = 25
            if len(value) == 1:
                if ax is None:
                    plt.plot(value[0][0], value[0][1], 'o', color = color, label = label, markersize = ms)
                else:
                    ax.plot(value[0][0], value[0][1], 'o', color = color, label = label, markersize = ms)
            elif len(value) == 2:
                if ax is None:
                    plt.plot([value[0][0], value[1][0]], [value[0][1], value[1][1]], '-', color = color, label = label, linewidth = lw)
                else:
                    ax.plot([value[0][0], value[1][0]], [value[0][1], value[1][1]], '-', color = color, label = label, linewidth = lw)
            elif len(value) == 4:
                if ax is None:
                    plt.plot([value[0][0], value[1][0], value[3][0], value[2][0], value[0][0]], [value[0][1], value[1][1], value[3][1], value[2][1], value[0][1]], '-', color = color, label = label, linewidth = lw)
                else:
                    ax.plot([value[0][0], value[1][0], value[3][0], value[2][0], value[0][0]], [value[0][1], value[1][1], value[3][1], value[2][1], value[0][1]], '-', color = color, label = label, linewidth = lw)

        # make a legend that shows the colors saying "origin", "1-sparse", "2-sparse" and the corresponding colors
        if show_legend:
            plt.legend()
        

# %% debugging
# %% debugging
if __name__ == "__main__":
    torch.manual_seed(0)

    # create a random matrix A
    A = torch.randn(4, 8)

    # create the y-anchors
    from hyper_plane_analysis import create_anchors_from_x_indices
    y_anchors, _ = create_anchors_from_x_indices((None,0,1), A)

    # test the plot data regions function
    data_on_plane = DataOnPlane(A, [0.5, 1.5], y_anchors, K = 2, verbose=True, consider_partials=True)

    # plot the data regions
    plt.figure()
    data_on_plane.plot_data_regions()
    plt.grid()
    plt.show()