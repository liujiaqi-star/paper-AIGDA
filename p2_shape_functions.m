function [phi, dphi_dl1, dphi_dl2, dphi_dl3] = p2_shape_functions(l1, l2, l3)
% P2 quadratic shape functions and their derivatives on a triangle
% Input:
%   l1, l2, l3 - Barycentric coordinates (l1 + l2 + l3 = 1)
% Output:
%   phi        - Shape function values (6 x 1)
%   dphi_dl1   - Derivatives with respect to l1 (6 x 1)
%   dphi_dl2   - Derivatives with respect to l2 (6 x 1)
%   dphi_dl3   - Derivatives with respect to l3 (6 x 1)
%
% Node numbering:
%   1: vertex 1 (l1=1, l2=0, l3=0)
%   2: vertex 2 (l1=0, l2=1, l3=0)
%   3: vertex 3 (l1=0, l2=0, l3=1)
%   4: midpoint of edge 1-2
%   5: midpoint of edge 2-3
%   6: midpoint of edge 3-1

phi = zeros(6, 1);
dphi_dl1 = zeros(6, 1);
dphi_dl2 = zeros(6, 1);
dphi_dl3 = zeros(6, 1);

% Vertex 1: node 1
phi(1) = l1 * (2*l1 - 1);
dphi_dl1(1) = 4*l1 - 1;

% Vertex 2: node 2
phi(2) = l2 * (2*l2 - 1);
dphi_dl2(2) = 4*l2 - 1;

% Vertex 3: node 3
phi(3) = l3 * (2*l3 - 1);
dphi_dl3(3) = 4*l3 - 1;

% Midpoint of edge 1-2: node 4
phi(4) = 4 * l1 * l2;
dphi_dl1(4) = 4*l2;
dphi_dl2(4) = 4*l1;

% Midpoint of edge 2-3: node 5
phi(5) = 4 * l2 * l3;
dphi_dl2(5) = 4*l3;
dphi_dl3(5) = 4*l2;

% Midpoint of edge 3-1: node 6
phi(6) = 4 * l3 * l1;
dphi_dl3(6) = 4*l1;
dphi_dl1(6) = 4*l3;

end