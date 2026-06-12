function [err,F_norm_2, z_opt]=A_PALM(F_matrix, F_const, dual_gap, x_0, y_0, max_iter,r)
% A_PALM - Accelerated Proximal Augmented Lagrangian Method
%
% Input:
%   F_matrix, F_const - define the monotone operator F(z) = F_matrix * z + F_const
%   dual_gap         - function handle for the primal-dual gap: L(x, y^*) - L(x^*, y)
%   (x_0, y_0)       - initial primal-dual point
%   max_iter         - maximum number of iterations
%   r                - convergence rate parameter
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

z_old = z_0;w_old=z_0;
I=eye(n_z);
for k=1:max_iter
    % A-PALM update:
    % z_k = w_{k-1} - k^r*s/(r+2)*F(z_k)
    % Equivalent to: z_k=(I+k^r/(r+2)*F_matrix)\(w_{k-1}-k^r/(r+2)*F_const) with s=1
    % w_k = z_k + k/(k+r+3)*(z_k-z_{k-1})-(k+1)^{r+1}/(k^r*(k+r+3)) *(z_k-w_{k-1})
    z_new=(I+k^r/(r+2)*F_matrix)\(w_old-k^r/(r+2)*F_const);
    w_new=z_new+k/(k+r+3)*(z_new-z_old)-(k+1)^(r+1)/((k^r)*(k+r+3))*(z_new-w_old);

    g = F(z_new);                                % evaluate F at the new point
    F_norm_2(k+1) = norm(g, 2)^2;                % store squared norm

    x=z_new(1:n_x);                              % extract primal variable
    y = z_new(n_x+1:end);                        % extract dual variable
    err(k+1)=dual_gap(x,y);                      % compute and store primal-dual gap

    z_old=z_new;                                 % move to next iteration
    w_old=w_new;
end
z_opt=z_new;
end
