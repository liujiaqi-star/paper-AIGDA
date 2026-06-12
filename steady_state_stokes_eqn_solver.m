% Circular domain Stokes equation solver
% Using polar coordinate method to generate mesh and finite element method (P2-P1 element) to solve Stokes equations
% Solves: -nu * Laplacian(u) + grad(p) = f, div(u) = 0 in circular domain
% Boundary condition: u = 0 on boundary (no-slip)
clear;
clc;

%% 1. Parameter settings
R = 1.0;            % Radius of the circle
n_radial = 10;       % Number of radial layers (controls mesh density in radial direction)
n_angular = 32;     % Number of angular points at the outermost layer (controls mesh density circumferentially)
nu = 1.0;           % Viscosity coefficient (kinematic viscosity)

%% 2. Generate nodes using polar coordinate method
% Creates nodes with increasing density from center to boundary
[nodes, node_rings, node_count] = generate_circular_mesh(R, n_radial, n_angular);
n_vertices = size(nodes, 1);  % Number of P1 nodes (vertices)

%% 3. Perform Delaunay triangulation

% 3.1 Delaunay triangulation
% Returns a three-column matrix, each row represents the vertex indices of a triangle
tri = delaunay(nodes(:,1), nodes(:,2));

% 3.2 Remove triangles outside the circle
% Calculate the centroid of each triangle
center = (nodes(tri(:,1),:) + nodes(tri(:,2),:) + nodes(tri(:,3),:)) / 3;
radius_center = sqrt(sum(center.^2, 2));  % Distance from origin to triangle centroid
tri = tri(radius_center <= R, :);  % Keep only triangles whose centroid lies inside the circle
n_elements = size(tri, 1);  % Number of triangular elements

% 3.3 Visualize the mesh
figure('units','normalized','position',[0.00 0.00 0.6 0.8]);
theta_plot = linspace(0, 2*pi, 100);  % Points for plotting the circular boundary
triplot(tri, nodes(:,1), nodes(:,2), 'k-', 'LineWidth', 0.5);  % Plot triangular elements
hold on;
plot(nodes(:,1), nodes(:,2), 'ro', 'MarkerSize', 3);  % Plot nodes as red circles
plot(R*cos(theta_plot), R*sin(theta_plot), 'b-', 'LineWidth', 2);  % Plot circular boundary
hold off;
axis equal;  % Equal aspect ratio
xlim([-1.2*R, 1.2*R]);
ylim([-1.2*R, 1.2*R]);
set(gca,'xtick',-1.2*R:0.6*R:1.2*R,'xticklabel',-1.2*R:0.6*R:1.2*R);
set(gca,'ytick',-1.2*R:0.6*R:1.2*R,'yticklabel',-1.2*R:0.6*R:1.2*R);
set(gca,'fontsize',30);
% xlabel('x'); ylabel('y');
% title(sprintf('Triangulation: %d nodes, %d triangles', n_vertices, n_elements));

%% 4. Create P2 nodes (add edge midpoints) and create P2 elements (6 nodes per triangle)
% P2 elements have 6 nodes: 3 vertices + 3 edge midpoints for quadratic approximation
[p2_nodes, p2_elements, n_p2, n_midpoints, edge_midpoints_in_elem] = create_p2_mesh(nodes, tri);

%% 5. Define body force and visualize (vectorized version)
% Body force f = (fx, fy). 
f = @(x, y) [-y, x];  % Example rotational force (can be replaced with any fx, fy)
figure('units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);

% Evaluate body force on a regular grid for visualization
[X_grid, Y_grid] = meshgrid(linspace(-R, R, 20), linspace(-R, R, 20));

F_val = f(X_grid(:), Y_grid(:));  % Evaluate force at all grid points
Fx_grid = reshape(F_val(:, 1), size(X_grid));  % Reshape fx component to grid
Fy_grid = reshape(F_val(:, 2), size(X_grid));  % Reshape fy component to grid

% Only display points inside the circle (avoid displaying forces outside domain)
mask_grid = (X_grid.^2 + Y_grid.^2) <= R^2;
X_display = X_grid(mask_grid);
Y_display = Y_grid(mask_grid);
Fx_display = Fx_grid(mask_grid);
Fy_display = Fy_grid(mask_grid);

% Plot body force vectors as arrows
quiver(X_display, Y_display, Fx_display, Fy_display, 0.8, 'k', 'LineWidth', 1);
hold on;
plot(R*cos(theta_plot), R*sin(theta_plot), 'b-', 'LineWidth', 2);  % Plot domain boundary
hold off;
axis equal;
xlim([-1.2*R, 1.2*R]);
ylim([-1.2*R, 1.2*R]);
set(gca, 'xtick', -1.2*R:0.6*R:1.2*R, 'xticklabel', -1.2*R:0.6*R:1.2*R);
set(gca, 'ytick', -1.2*R:0.6*R:1.2*R, 'yticklabel', -1.2*R:0.6*R:1.2*R);
set(gca, 'fontsize', 30);

%% 6. Assemble Stokes matrices
% Assembles the stiffness matrix A (viscous term), divergence matrix B,
% and right-hand side vector b (body force)
[A, B, b, boundary_nodes] = assemble_stokes_matrix(p2_nodes, p2_elements, tri, nodes, R, nu, f);

%% 7. Initialize saddle point system
% The Stokes system leads to a saddle point problem:
% [A  B'] [u] = [b]
% [B  0 ] [p]   [0]
n_u = size(A, 1);  % Number of velocity degrees of freedom
n_p = size(B, 1);  % Number of pressure degrees of freedom 
O = sparse(n_p, n_p);  % Zero block in saddle point matrix

% Build the monotone operator F(z) = F_matrix * z + F_const
F_matrix = [A, B'; -B, O];
F_const = -[b; zeros(n_p, 1)];

% Solve the linear system directly to obtain exact solution z* = [x*; y*] (for comparison)
z_exact = F_matrix \ (-F_const);
x_exact = z_exact(1:n_u);  % Exact velocity solution
y_exact = z_exact(n_u+1:end);  % Exact pressure solution

% Define the Lagrangian function L(x,y) = 0.5*x'*A*x - b'*x + y'*(B*x)
L = @(x, y) 1/2 * x' * A * x - b' * x + y' * (B * x);
% Define the duality gap: L(x, y*) - L(x*, y)
dual_gap = @(x, y) L(x, y_exact) - L(x_exact, y);

% Initial guess for primal and dual variables
x_0 = zeros(n_u, 1);  % Initial velocity
y_0 = zeros(n_p, 1);  % Initial pressure
D = norm(z_exact)^2;  % the distance ||z_0-z*||^2 (for convergence rate)

%% 8. Test PALM (Proximal Augmented Lagrangian Method)
% PALM achieves an o(1/k^2) convergence rate on the squared gradient norm.
max_iter = 100;  % Maximum number of iterations
t = 0:max_iter;  % Iteration counter

% Run PALM algorithm
[err, F_norm_2, z_opt] = PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter);

% Plot results: squared gradient norm vs. iteration
figure('units','normalized','position',[0.00 0.00 0.6 0.8]);

% Theoretical convergence rate: O(1/k^2)
convergence_rate_2 = D ./ (t + 1).^2;

plot(t, F_norm_2, 'b-', 'LineWidth', 3); hold on;
plot(t, convergence_rate_2, 'k--', 'LineWidth', 3); hold on;

% Set log scale for y-axis to show convergence rate clearly
set(gca, 'YScale', 'log', 'YLim', [1e-7, 1e-1], 'YTick', 10.^(-7:2:-1));
set(gca,'fontsize',30);
set(gca,'xlim',([0, 100]),'xtick',(0 : 25 : 100));
legend('$\texttt{PALM}$','Location','northeast' ,'Interpreter', 'latex', 'FontSize', 45);
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
text(70, 10^(-5), '$O(1/k^2)$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold');

%% 9. Test A-PALM (Accelerated PALM with parameter r)
% A-PALM achieves an o(1/k^{r+1}) convergence rate on the primal-dual gap.
max_iter_A = 100;
t_A = 0:max_iter_A;

% Create new figure
figure('units','normalized','position',[0.00 0.00 0.6 0.8]);

% Plot PALM result for comparison
plot(t_A, err(t+1), 'b-', 'LineWidth', 3); hold on;

% Test A-PALM with different r values (0, 1, 2, 3)
for r = 0:3
    [err_A, F_norm_2_A, z_opt_A] = A_PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter_A, r);
    plot(t_A, err_A, '-', 'LineWidth', 3); hold on;
end

% Save optimal solution when r=3 (best acceleration)
U = z_opt_A(1:n_u, end);  % Optimal velocity solution
P = z_opt_A(n_u+1:end, end);  % Optimal pressure solution

% Theoretical convergence rate for r=0: O(1/k)
convergence_rate_1 = (D/2) ./ (t + 1);
plot(t, convergence_rate_1, 'k--', 'LineWidth', 3); hold on;

% Configure plot appearance
set(gca,'fontsize',30);
set(gca,'xlim',([0, 100]),'xtick',(0 : 25 : 100));
ylim([1e-16, 1e-1]);
set(gca, 'YScale', 'log', 'YLim', [1e-16, 1e-1], 'YTick', 10.^(-16:5:-1));
legend('$\texttt{PALM}$', '$\texttt{A-PALM}$: $r=0$','$\texttt{A-PALM}$: $r=1$','$\texttt{A-PALM}$: $r=2$','$\texttt{A-PALM}$: $r=3$','Location','southwest' ,'Interpreter', 'latex', 'FontSize', 45);
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$L(x_k,y^*)-L(x^*,y_k)$', 'Interpreter', 'latex', 'FontSize', 50);
text(70 , 1e-3, '$O(1/k)$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold');
%title('Solve Stokes Equation in a Circular Domain');

%% 10. Visualize results
% Plot streamlines of the computed velocity field

% Extract velocity at vertices (P1 nodes)
ux_vertices = zeros(n_vertices, 1);
uy_vertices = zeros(n_vertices, 1);
for i = 1:n_vertices
    ux_vertices(i) = U(2*i-1);  % x-velocity component at vertex i
    uy_vertices(i) = U(2*i);    % y-velocity component at vertex i
end

% Create interpolation grid for contour plot (finer grid for smooth visualization)
n_plot = 50;
[Xq, Yq] = meshgrid(linspace(-R, R, n_plot), linspace(-R, R, n_plot));

% Interpolate velocity components from vertices to regular grid
% Using linear interpolation, 'none' for extrapolation (points outside domain remain NaN)
F_ux = scatteredInterpolant(nodes(:,1), nodes(:,2), ux_vertices, 'linear', 'none');
F_uy = scatteredInterpolant(nodes(:,1), nodes(:,2), uy_vertices, 'linear', 'none');

Uxq = F_ux(Xq, Yq);  % Interpolated x-velocity on grid
Uyq = F_uy(Xq, Yq);  % Interpolated y-velocity on grid

% Streamline plot
% streamslice creates streamlines from 2D vector field data
figure('units','normalized','position',[0.00 0.00 0.6 0.8]);
h = streamslice(Xq, Yq, Uxq, Uyq, 1.5);  % 1.5 controls streamline density
set(h, 'Color', 'k', 'LineWidth', 1.5);  % Set streamline appearance
hold on;
plot(R*cos(theta_plot), R*sin(theta_plot), 'b-', 'LineWidth', 2);  % Plot domain boundary
hold off;
axis equal;
box on;
xlim([-1.2*R, 1.2*R]);
ylim([-1.2*R, 1.2*R]);
set(gca,'xtick',-1.2*R:0.6*R:1.2*R,'xticklabel',-1.2*R:0.6*R:1.2*R);
set(gca,'ytick',-1.2*R:0.6*R:1.2*R,'yticklabel',-1.2*R:0.6*R:1.2*R);
set(gca,'fontsize',30);