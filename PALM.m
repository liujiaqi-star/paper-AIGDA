function [err, F_norm_2, z_opt] = PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter)
% PALM - Proximal Augmented Lagrangian Method
%
% Input:
%   F_matrix, F_const - define the monotone operator F(z) = F_matrix * z + F_const
%   dual_gap         - function handle for the primal-dual gap: L(x, y^*) - L(x^*, y)
%   (x_0, y_0)       - initial primal-dual point
%   max_iter         - maximum number of iterations
%
% Output:
%   err              - primal-dual gap values recorded at each iteration (including initialization)
%   F_norm_2         - squared norms of F(z) recorded at each iteration
%   z_opt            - approximate solution at the maximum number of iterations

% Define the monotone operator F(z)
F = @(z) F_matrix * z + F_const;

n_x = length(x_0);                % dimension of the primal variable
err = zeros(1, max_iter + 1);     % preallocate error history
F_norm_2 = zeros(1, max_iter + 1);% preallocate gradient norm history

z_0 = [x_0; y_0];                 % initial primal-dual point
n_z=length(z_0);                  % dimension of the primal-dual variable
err(1) = dual_gap(x_0, y_0);      % initial primal-dual gap

g = F(z_0);
F_norm_2(1) = norm(g, 2)^2;       % initial squared norm of F(z)

z_old = z_0;
I=eye(n_z);
for k = 1:max_iter
    % PALM update:
    % z_k + (k+2)sF(z_k) = z_{k-1} + ks*F(z_{k-1}) with s=1
    % Equivalent to: z_k = z_{k-1} - (I + (k+2)*F_matrix) \ (2 * F(z_{k-1})) 
    z_new = z_old - (I + (k+2) * F_matrix) \ (2 * F(z_old));

    g = F(z_new);                                 % evaluate F at the new point
    F_norm_2(k+1) = norm(g, 2)^2;                % store squared norm

    x = z_new(1:n_x);                             % extract primal variable
    y = z_new(n_x+1:end);                         % extract dual variable

    err(k+1) = dual_gap(x, y);                    % compute and store primal-dual gap

    z_old = z_new;                                % move to next iteration
end
z_opt=z_new;
end