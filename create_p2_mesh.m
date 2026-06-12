function [p2_nodes, p2_elements, n_p2, n_midpoints, edge_midpoints_in_elem] = create_p2_mesh(nodes, tri)
% Create P2 mesh nodes (add edge midpoints) for quadratic finite elements
% Input:
%   nodes - Node coordinates [x, y] for circular (P1) mesh
%   tri   - Triangle connectivity matrix (n_elements x 3)
% Output:
%   p2_nodes                 - P2 node coordinates [x, y] (including P1 nodes and edge midpoints)
%   p2_elements              - P2 element connectivity (n_elements x 6) [v1,v2,v3, mid12,mid23,mid31]
%   n_p2                     - Total number of P2 nodes
%   n_midpoints              - Number of edge midpoints
%   edge_midpoints_in_elem   - Midpoint index for each edge in each element (n_elements x 3)

n_elements = size(tri, 1);
n_vertices = size(nodes, 1);

% Initialize
edge_map = containers.Map();
midpoints = [];
next_id = n_vertices + 1;

% Store midpoint index for each edge in each element
edge_midpoints_in_elem = zeros(n_elements, 3);
% Store midpoint index for each edge (for query)
edge_mid_map = containers.Map();

for e = 1:n_elements
    v1 = tri(e, 1);
    v2 = tri(e, 2);
    v3 = tri(e, 3);

    edges = [v1, v2; v2, v3; v3, v1];

    for k = 1:3
        a = edges(k, 1);
        b = edges(k, 2);
        key = sprintf('%d-%d', min(a,b), max(a,b));

        if ~edge_map.isKey(key)
            % If this edge doesn't exist yet, create a new midpoint
            mid = (nodes(a, :) + nodes(b, :)) / 2;
            edge_map(key) = next_id;
            midpoints = [midpoints; mid];
            edge_midpoints_in_elem(e, k) = next_id;
            edge_mid_map(key) = next_id;
            next_id = next_id + 1;
        else
            edge_midpoints_in_elem(e, k) = edge_map(key);
        end
    end
end

% All P2 nodes
p2_nodes = [nodes; midpoints];
n_p2 = size(p2_nodes, 1);
n_midpoints = size(midpoints, 1);

% Create P2 elements (6 nodes per triangle)
p2_elements = zeros(n_elements, 6);
for e = 1:n_elements
    p2_elements(e, 1:3) = tri(e, :);
    p2_elements(e, 4:6) = edge_midpoints_in_elem(e, :);
end

end