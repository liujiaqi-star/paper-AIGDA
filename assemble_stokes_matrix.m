function [A, B, b, boundary_nodes] = assemble_stokes_matrix(p2_nodes, p2_elements, tri, nodes, R, nu, f)
% Assemble Stokes flow finite element matrices (P2-P1)
% Input:
%   p2_nodes    - P2 node coordinates [x, y]
%   p2_elements - P2 element connectivity (n_elements x 6)
%   tri         - Triangle connectivity for pressure (n_elements x 3)
%   nodes       - P1 node coordinates
%   R           - Radius of the circular domain
%   nu          - Viscosity coefficient
%   f           - Body force function handle: f(x, y) = [fx, fy]
% Output:
%   A           - Stiffness matrix
%   B           - Divergence matrix
%   b           - Right-hand side vector
%   boundary_nodes - List of boundary node indices

n_p2 = size(p2_nodes, 1);
n_elements = size(p2_elements, 1);
n_vertices = size(nodes, 1);

% Initialize matrices
n_u = 2 * n_p2;  % Each P2 node has ux, uy
n_p = n_vertices; % Pressure nodes use vertices
A = sparse(n_u, n_u);
B = sparse(n_p, n_u);
b = zeros(n_u, 1);

% Gauss quadrature parameters (3-point quadrature on triangle)
gauss_pts = [1/6, 1/6; 2/3, 1/6; 1/6, 2/3];
gauss_wts = [1/6, 1/6, 1/6];

% Loop over all elements
for e = 1:n_elements
    % Get element information
    vertices_idx = tri(e, :);
    p2_idx = p2_elements(e, :);
    vert_coords = nodes(vertices_idx, :);
    
    % Compute Jacobian matrix
    J = [vert_coords(2,1)-vert_coords(1,1), vert_coords(3,1)-vert_coords(1,1);
         vert_coords(2,2)-vert_coords(1,2), vert_coords(3,2)-vert_coords(1,2)];
    detJ = abs(det(J));
    Jinv = inv(J);
    
    % Element matrices
    A_elem = zeros(12, 12);
    B_elem = zeros(3, 12);
    b_elem = zeros(12, 1);
    
    % Gauss integration
    for gp = 1:3
        xi = gauss_pts(gp, 1);
        eta = gauss_pts(gp, 2);
        weight = gauss_wts(gp) * detJ;
        
        % Barycentric coordinates
        l1 = 1 - xi - eta;
        l2 = xi;
        l3 = eta;
        
        % P2 shape functions
        [phi, dphi_dl1, dphi_dl2, dphi_dl3] = p2_shape_functions(l1, l2, l3);
        
        % Transform derivatives from barycentric coordinates to physical coordinates
        % ∂φ/∂x = (∂φ/∂l1)*(∂l1/∂x) + (∂φ/∂l2)*(∂l2/∂x) + (∂φ/∂l3)*(∂l3/∂x)
        % ∂φ/∂y = (∂φ/∂l1)*(∂l1/∂y) + (∂φ/∂l2)*(∂l2/∂y) + (∂φ/∂l3)*(∂l3/∂y)
        % where [∂l1/∂x, ∂l1/∂y; ∂l2/∂x, ∂l2/∂y] = inv(J)
        dphi_dx = zeros(6, 1);
        dphi_dy = zeros(6, 1);
        for i = 1:6
            dphi_dx(i) = Jinv(1,1)*dphi_dl1(i) + Jinv(1,2)*dphi_dl2(i) + Jinv(1,2)*dphi_dl3(i);
            dphi_dy(i) = Jinv(2,1)*dphi_dl1(i) + Jinv(2,2)*dphi_dl2(i) + Jinv(2,2)*dphi_dl3(i);
        end
        
        % P1 pressure shape functions
        psi = [l1; l2; l3];
        
        % Assemble local stiffness matrix for viscous term
        % A_elem(2*i-1, 2*j-1) = nu * ∫_Ω (∂φ_i/∂x * ∂φ_j/∂x + ∂φ_i/∂y * ∂φ_j/∂y) dΩ  (ux-ux component)
        % A_elem(2*i, 2*j)     = nu * ∫_Ω (∂φ_i/∂x * ∂φ_j/∂x + ∂φ_i/∂y * ∂φ_j/∂y) dΩ  (uy-uy component)
        for i = 1:6
            for j = 1:6
                val = nu * (dphi_dx(i)*dphi_dx(j) + dphi_dy(i)*dphi_dy(j)) * weight;
                A_elem(2*i-1, 2*j-1) = A_elem(2*i-1, 2*j-1) + val;
                A_elem(2*i, 2*j) = A_elem(2*i, 2*j) + val;
            end
        end
        
        % Assemble local divergence matrix (divergence constraint)
        % B_elem(k, 2*i-1) = ∫_Ω ψ_k * (∂φ_i/∂x) dΩ  (ux contribution)
        % B_elem(k, 2*i)   = ∫_Ω ψ_k * (∂φ_i/∂y) dΩ  (uy contribution)
        for k = 1:3
            for i = 1:6
                B_elem(k, 2*i-1) = B_elem(k, 2*i-1) + psi(k) * dphi_dx(i) * weight;
                B_elem(k, 2*i) = B_elem(k, 2*i) + psi(k) * dphi_dy(i) * weight;
            end
        end
        
        % Assemble right-hand side (body force)
        % b_elem(2*i-1) = ∫_Ω f_x * φ_i dΩ  (ux component)
        % b_elem(2*i)   = ∫_Ω f_y * φ_i dΩ  (uy component)

        % Compute physical coordinates of Gauss point
        x_phys = l1 * vert_coords(1,1) + l2 * vert_coords(2,1) + l3 * vert_coords(3,1);
        y_phys = l1 * vert_coords(1,2) + l2 * vert_coords(2,2) + l3 * vert_coords(3,2);
        
        % Evaluate body force at Gauss point
        F = f(x_phys, y_phys);

        for i = 1:6
            b_elem(2*i-1) = b_elem(2*i-1) + F(1) * phi(i) * weight;
            b_elem(2*i)   = b_elem(2*i)   + F(2) * phi(i) * weight;
        end
    end
    
    % Assemble into global matrices
    % Add contributions from all elements to global matrices
    % For nodes shared by multiple triangles, their contributions are summed together

    % DOF mapping for velocity (2 DOFs per node: ux and uy):
    % Local node i (1-6) → global node number: global_i = p2_idx(i)
    % ux at node i → global DOF: dof_ix = 2*global_i - 1
    % uy at node i → global DOF: dof_iy = 2*global_i
    for i = 1:6
        global_i = p2_idx(i);
        dof_ix = 2*global_i - 1;
        dof_iy = 2*global_i;
        
        for j = 1:6
            global_j = p2_idx(j);
            dof_jx = 2*global_j - 1;
            dof_jy = 2*global_j;
            
            A(dof_ix, dof_jx) = A(dof_ix, dof_jx) + A_elem(2*i-1, 2*j-1);
            A(dof_ix, dof_jy) = A(dof_ix, dof_jy) + A_elem(2*i-1, 2*j);
            A(dof_iy, dof_jx) = A(dof_iy, dof_jx) + A_elem(2*i, 2*j-1);
            A(dof_iy, dof_jy) = A(dof_iy, dof_jy) + A_elem(2*i, 2*j);
        end
        
        % Pressure DOFs: use vertex nodes only (P1 element)
        % Local vertex k (1-3) → global pressure DOF: p_idx = vertices_idx(k)
        for k = 1:3
            p_idx = vertices_idx(k);
            B(p_idx, dof_ix) = B(p_idx, dof_ix) + B_elem(k, 2*i-1);
            B(p_idx, dof_iy) = B(p_idx, dof_iy) + B_elem(k, 2*i);
        end
        
        b(dof_ix) = b(dof_ix) + b_elem(2*i-1);
        b(dof_iy) = b(dof_iy) + b_elem(2*i);
    end
end

% Apply boundary conditions (no-slip u=0 on boundary)

% Find boundary nodes (radius close to R)
tol = 1e-8;
boundary_nodes = [];
for i = 1:n_p2
    r = sqrt(p2_nodes(i,1)^2 + p2_nodes(i,2)^2);
    if abs(r - R) < tol
        boundary_nodes = [boundary_nodes, i];
    end
end

% Apply boundary conditions (no-slip condition: u = 0 on boundary)
% For each boundary node, set both ux and uy components to zero
% Mathematical principle:
%   Original equation: A(j,:) * x = b(j)
%   After modification: x_j = 0 (since 1 * x_j = 0)
for i = 1:length(boundary_nodes)
    node_idx = boundary_nodes(i);
    dof_x = 2*node_idx - 1;
    dof_y = 2*node_idx;
    
    % Enforce ux = 0 at this node
    A(dof_x, :) = 0;     % Clear the row
    A(:, dof_x) = 0;     % Clear the column (maintain symmetry)
    A(dof_x, dof_x) = 1; % Set diagonal to 1
    b(dof_x) = 0;        % Set RHS to 0

    % Enforce uy = 0 at this node
    A(dof_y, :) = 0;
    A(:, dof_y) = 0;
    A(dof_y, dof_y) = 1;
    b(dof_y) = 0;
end

end