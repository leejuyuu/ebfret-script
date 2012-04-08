function u_new = hstep_ml(w, u, weights)
% u_new = hstep_ml(w, u, varargin) 
%
% Hyper parameter updates for Hierarchical Model Inference (HMI)
% on a single-molecule FRET dataset.
%
% The input of this method is a set of  posteriors produced by running 
% Variational Bayes Expectation Maxmimization (VBEM) on a series
% of FRET traces and maximizes the total summed evidence by solving 
% the system of equations:
%
%   Sum_n  Grad_u L_n  =  0
%
% Where u is the set of hyperparameters that determine the form of
% the prior distribution on the parameters. L_n is the lower bound
% evidence for the n-th trace, defined by:
%
%   L  =  Integral d z d theta  q(z) q(theta)  
%         ln[ p(x, z | theta) p(theta) / (q(z) q(theta)) ]
%
% The approximate posteriors q(z) and q(theta), which have been
% optimised in the VBEM process, are now kept constant as Sum L_n is
% maximised wrt to u.
%
% 
% Inputs
% ------
%
%   w : struct (N x 1)
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w), for each of N traces 
%
%       .A (K x K)
%           Dirichlet prior for each row of transition matrix
%       .pi (K x 1)
%           Dirichlet prior for initial state probabilities
%       .mu (K x D)
%           Normal-Wishart prior - state means 
%       .beta (K x 1)
%           Normal-Wishart prior - state occupation count
%       .W (K x D x D)
%           Normal-Wishart prior - state precisions
%       .nu (K x 1)
%           Normal-Wishart prior - degrees of freedom
%           (must be equal to beta+1)
%
%   u : struct 
%       Hyperparameters for the prior distribution p(theta | u)
%       (same fields as w)
%
%   weights : (N x 1) optional
%       Weighting for each trace in updates.
%
%
% Outputs
% -------
%
%   u_new : struct 
%       Updated hyperparameters
%
% Jan-Willem van de Meent
% $Revision: 1.10$  $Date: 2011/08/04$

% Note: this currently does not work for 2D donor/acceptor inference
%
% Explanation of Updates
% ----------------------
%
% For a prior/posterior in Conjugate Exponential form 
%
%   p(theta)  =  f(nu, chi) g(theta)^nu exp[ eta(theta) . chi ]
%   q(theta)  =  f(nu', chi') g(theta)^nu' exp[ eta(theta) . chi' ]
%
% The derivatives of L w.r.t. to the parameters are given by
% 
%   Grad_nu L  =  (Grad_nu f(eta, nu)) / f(eta, nu)
%                 + <ln g(theta)>_q(theta)  
%
% Similarly the derivatives wrt to nu_i are given by                   
%
%   Grad_chi_i L =  (Grad_chi_i f(eta, nu)) / f(eta, nu)
%                   + <eta_i>_q(theta)
%
% We now define the averaged expectation value over all traces:
%
%   E[theta]  =  1/N  Sum_n  <theta>_q(theta | w(n))
%
% The system of equations Grad Sum_n L^n = 0 can now be solved
% from the above identities, resulting in the following updates
%
%   (mu, lambda) ~ NormalGamma(mu, lambda | u.mu, u.beta, u.nu, u.W)
%
%   u.beta  =  u.nu - 1
%   u.mu  =  E[mu lambda] / E[lambda]
%   u.W  =  E[lambda] / u.nu
%   -<log g> =  1/2 ( 1 / (u.nu -1)
%                     + E[mu lambda]^2 / E[lambda]
%                     + Log[pi / uW]
%                     - psi(u.nu/2) )
%       
% {A(k,:)} ~ Dir(A_k | u.A(k,:))
%
%   psi(Sum u.A(k,:)) - psi(u.A(k,:)) = -E[log A(k,:)]
%
% pi ~ Dir(pi | u.pi)
%
%   psi(Sum u.pi) - psi(u.pi) = -E[log pi] 


% TODO: make this a variable arg
threshold = 1e-6;

% Get dimensions
N = length(w);
K = length(w(1).pi);

% set small number > eps
EPS = 10 * eps;

% set optimization settings
opts = optimset('display', 'off', 'tolX', threshold, 'tolFun', eps);

% initialize dummy weights if not specified
if nargin < 3
    weights = ones(N,1);
end

% assign weights to w
w0 = num2cell(bsxfun(@rdivide, weights, sum(weights, 1)));
[w(:).wt] = deal(w0{:});

% initialize struct for updated params
u_old = u;

% (mu, lambda) ~ Normal-Gamma
% Expectation Value eta1: E[lambda] = w.nu * w.W
E_l = arrayfun(@(w) w.nu .* w.W .* w.wt, w, 'UniformOutput', false);
E_l = sum([E_l{:}], 2);

% (mu, lambda)
% Expectation Value eta2: E[mu * lambda] = w.nu * w.W * w.mu
E_ml = arrayfun(@(w) w.nu .* w.W .* w.mu .* w.wt, w, 'UniformOutput', false);
E_ml = sum([E_ml{:}], 2);

% (mu, lambda): For expectation values of E[log g] we use 
%
% Grad_nu'  Int  d theta  q(theta)  =  0 
%           =  [ (Grad_nu' f(nu', chi')) / f  +  <log g>_q(theta) ]
%
% So
%
% <log g>_q(theta | w(n)) = - (Grad_nu' f(nu', chi')) / f
%
% With the latter expression reducing to:
% 
% -(Grad_nu' f(nu', chi')) / f  =
% - 1/2 [ 1/w.beta +  w.nu w.W w.mu^2
%         + log(pi / w.W) - psi^0(w.nu/2) ]
E_log_g = arrayfun(@(w) -0.5 * w.wt .* (1 ./ w.beta ...
                               + w.nu .* w.W .* w.mu.^2 ...
                               + log(pi ./ w.W) ...
                               - psi(w.nu/2)), ...
                   w, 'UniformOutput', false);
E_log_g = sum([E_log_g{:}], 2);

% (mu, lambda): Solve for u.nu
%
%   -<log g> =  1/2 ( 1 / (u.nu -1)
%                     + E[mu lambda]^2 / E[lambda]
%                     + Log[pi u.nu / E[lambda]]
%                     - psi(u.nu/2) )
root_fun = @(nu) E_log_g ...
                 + 0.5 * ((1 ./ (nu - 1)) ...
                          + (E_ml.^2 ./ E_l) ...
                          + log(pi .* nu ./ E_l) ...
                          - psi(nu / 2));
u.nu = lsqnonlin(root_fun, ...
                 u_old.nu, ...
                 ones(size(u_old.nu)), ...
                 Inf + zeros(size(u_old.nu)), ...
                 opts);

% (mu, lambda): Solve u.mu, u.beta and u.W
u.mu = E_ml ./ E_l;
u.beta = u.nu - 1;
u.W  = E_l ./ u.nu;

% pi ~ Dirichlet
% 
% Expectation value E[log pi]: Use same trick as for E[log g]:
%
% E[log pi] = - (Grad_nu' f(nu', chi')) / f
%           = - psi(Sum w.pi) + psi(w.pi)
E_log_wpi = arrayfun(@(w) w.wt * (psi(w.pi + eps) - psi(sum(w.pi + eps))), ...
                    w, 'UniformOutput', false);
E_log_wpi = sum([E_log_wpi{:}], 2);
w_pi = exp(E_log_wpi);

%(pi): solve system of equations

% get norm of u.piin right ballpark first
root_fun = @(P) (E_log_wpi - (psi(P * u.pi) - psi(sum(P * u.pi)))) .* (w_pi + eps);
P = lsqnonlin(root_fun, 1, 0, Inf, opts);
u.pi = P * u.pi;

% now run interative updates on individual components
upi_old = eps * ones(size(u.pi));
while kl_dir(u.pi, upi_old) > threshold
    upi_old = u.pi;
    for k = 1:K
        upi0 = sum(u.pi .* (k ~= 1:K)');
        root_fun = @(upik) (E_log_wpi(k) - (psi(upik) - psi(upik + upi0))); 
        u.pi(k) = lsqnonlin(root_fun, u.pi(k), 0, Inf, opts);
    end
end

% A(k,:) ~ Dirichlet
% E[log A(k,:)]: Same as E[log pi]
E_log_wA = arrayfun(@(w) w.wt * bsxfun(@plus, psi(w.A + EPS), -psi(sum(w.A + EPS, 2))), ...
                    w, 'UniformOutput', false);
E_log_wA = sum(cat(3, E_log_wA{:}), 3);
w_A = exp(E_log_wA);

%A(k,:): solve system of equations
u.A = u_old.A;
for k = 1:K
    %get amplitude in right ballpark first
    root_fun = @(U) (E_log_wA(k,:) - (psi(U * u.A(k,:)) - psi(sum(U * u.A(k,:))))) .* (w_A(k,:) + eps);
    U = lsqnonlin(root_fun, 1, 0, Inf, opts);
    u.A(k,:) = U * u.A(k,:);

    % now do all components
    uAk_old = eps * ones(size(u.A(k,:)));
    while kl_dir(u.A(k,:), uAk_old) > threshold
        uAk_old = u.A(k,:);
        for l = 1:K
            uA0 = sum(u.A(k,:) .* (l ~= 1:K));
            root_fun = @(uAkl) (E_log_wA(k,l) - (psi(uAkl) - psi(uAkl + uA0))); 
            u.A(k,l) = lsqnonlin(root_fun, u.A(k,l), 0, Inf, opts);
        end
    end
end

u_new = u;
