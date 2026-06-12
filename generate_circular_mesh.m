function [nodes, node_rings, node_count] = generate_circular_mesh(R, n_radial, n_angular)
% Generate polar grid nodes for a circular region
% Input:
%   R         - Radius of the circle
%   n_radial  - Number of radial layers
%   n_angular - Number of angular points at the outermost layer
% Output:
%   nodes      - Node coordinates [x, y]
%   node_rings - Cell array storing node indices for each ring
%   node_count - Total number of nodes

% Initialization
nodes = [];
node_rings = {};
node_count = 0;

% Center point (first layer)
nodes = [nodes; 0, 0];
node_count = node_count + 1;
node_rings{1} = node_count;

% Radial rings
for r = 1:n_radial
    radius = r * R / n_radial;
    
    % Number of angular points, increasing from inside out, at least 6 points per layer
    if r < n_radial
        n_points = max(6, round(n_angular * r / n_radial));
    else
        n_points = n_angular;  % Keep uniform at the outermost layer
    end
    
    % Polar coordinates
    theta = linspace(0, 2*pi, n_points+1)';
    theta = theta(1:end-1);
    
    x = radius * cos(theta);
    y = radius * sin(theta);
    
    % Store node coordinates and indices
    nodes = [nodes; x, y];
    node_rings{r+1} = (node_count+1):(node_count + n_points);
    node_count = node_count + n_points;
end

end