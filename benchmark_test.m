%% Solve min_x max_y 1/2 x^T H x - h^T x - y^T (A x - b)
% This script demonstrates the PALM and A-PALM algorithms for solving
% a convex-concave saddle-point problem arising from a positive semidefinite quadratic programming.

clear;
clc;

%% Problem Setup
% Define problem dimensions and data matrices
n = 200;                                    % dimension of primal variable

% Construct matrix A (a lower bidiagonal matrix with flipped columns)
A = diag(ones(n,1)) + diag(-ones(n-1,1), 1);
A = 1/4 * fliplr(A);

% Construct Hessian H = 2 * A' * A (positive semidefinite)
H = 2 * (A' * A);

% Define vectors b and h
b = 1/4 * ones(n, 1);
h = 1/4 * [zeros(n-1, 1); 1];

% Build the monotone operator F(z) = F_matrix * z + F_const
O = zeros(n, n);
F_matrix = [H, -A'; A, O];
F_const = [-h; -b];

% Compute exact solution z* = [x*; y*] by solving the linear system
z_exact = F_matrix \ (-F_const);
x_exact = z_exact(1:n);                     % exact primal solution
y_exact = z_exact(n+1:end);                 % exact dual solution

% Define the Lagrangian function L(x, y)
L = @(x, y) 1/2 * x' * H * x - h' * x - y' * (A * x - b);

% Define the primal-dual gap function
% gap(x, y) = L(x, y*) - L(x*, y)
dual_gap = @(x, y) L(x, y_exact) - L(x_exact, y);

% Initial guess
x_0 = zeros(n, 1);
y_0 = zeros(n, 1);
D = norm(z_exact)^2;                        % the distance ||z_0-z^*||^2 (for convergence rate)
%% Test PALM (Proximal Augmented Lagrangian Method)
% PALM achieves an o(1/k^2) convergence rate on the squared gradient norm.

max_iter = 5 * 10^3;                        % number of iterations
t = 0 : max_iter;                           % iteration index vector

% Run PALM algorithm
[err, F_norm_2, z_opt] = PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter);

% Plot results: squared gradient norm vs. iteration
figure('units', 'normalized', 'position', [0.00, 0.00, 0.6, 0.8]);

% Theoretical convergence rate: O(1/k^2)
convergence_rate_2 = D ./ (t + 1).^2;

plot(t, F_norm_2, 'b-', 'LineWidth', 3); hold on;
plot(t, convergence_rate_2, 'k--', 'LineWidth', 3); hold on;

% Formatting
set(gca, 'YScale', 'log');
set(gca, 'xlim', ([0, 5000]), 'xtick', (0 : 1000 : 5000));
set(gca, 'YScale', 'log', 'YLim', [1e-3, 1e7], 'YTick', 10.^(-3:2:7));
set(gca, 'fontsize', 30);
legend('$\texttt{PALM}$', 'Location', 'SouthWest', 'Interpreter', 'latex', 'FontSize', 45);
xlabel('Iteration: $k$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$\|F(z_k)\|^2$', 'Interpreter', 'latex', 'FontSize', 50);
text(3500, 10^0, '$O(1/k^2)$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold');

%% Test A-PALM (Accelerated PALM with parameter r)
% A-PALM achieves an o(1/k^{r+1}) convergence rate on the primal-dual gap.

max_iter_A = 100;                           % iterations for A-PALM (shorter run)
t_A = 0 : max_iter_A;

% Create new figure
fig = figure('units', 'normalized', 'position', [0.00, 0.00, 0.6, 0.8]);

% Plot PALM baseline (primal-dual gap)
plot(t_A, err(t_A + 1), 'b-', 'LineWidth', 3); hold on;

% Run A-PALM for different acceleration parameter r = 0,1,2,3
for r = 0 : 3
    [err_A, F_norm_2_A, z_opt_A] = A_PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter_A, r);
    plot(t_A, err_A, '-', 'LineWidth', 3); hold on;
end

box on;

% Theoretical convergence rate for r=0: O(1/k)
convergence_rate_1 = (D / 10^3) ./ (t + 1);
plot(t, convergence_rate_1, 'k--', 'LineWidth', 3); hold on;

% Formatting
set(gca, 'fontsize', 30);
legend('$\texttt{PALM}$', ...
       '$\texttt{A-PALM}$: $r=0$', ...
       '$\texttt{A-PALM}$: $r=1$', ...
       '$\texttt{A-PALM}$: $r=2$', ...
       '$\texttt{A-PALM}$: $r=3$', ...
       'Location', 'southwest', 'Interpreter', 'latex', 'FontSize', 45);

ylim([1e-15, 1e5]);
set(gca, 'xlim', ([0, 100]), 'xtick', (0 : 25 : 100));
set(gca, 'YScale', 'log', 'YLim', [1e-15, 1e5], 'YTick', 10.^(-15:5:5));

xlabel('Iteration: $k$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$L(x_k, y^*) - L(x^*, y_k)$', 'Interpreter', 'latex', 'FontSize', 50);

text(70, 10^2.5, '$O(1/k)$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold');