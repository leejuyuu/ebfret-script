function [PostHPar PostPar] = hstep_ml(out, PriorPar)
% Hyper parameter updates for Hierarchical Model Inference (HMI)
% on a single-molecule FRET dataset.
%
% The input of this method is a set of  posteriors produced by running 
% Variational Bayes Expectation Maxmimization (VBEM) on a series
% of FRET traces and maximizes the total summed evidence by solving 
% the system of equations:
%
%   Sum_n  Grad_u L^n  =  0
%
% Where u is the set of hyperparameters that determine the form of
% the prior distributions in the parameters. The lower bound evidence
% of the n-th trace is defined by:
%
%   L^n  =  Int d z^n d theta^n  q(z^n) q(theta^n)  
%           ln[ p(x^n, z^n | theta^n) p(theta^n) / (q(z^n) q(theta^n)) ]
%
% The approximate posteriors q(z) and q(theta), which have been
% optimised in the VBEM process, are now kept constant as Sum L^n is
% maximised wrt to u.
%
% 
% Inputs
% ------
%
% out (Nx1 Cell)
%   VBEM output for each trace (see VBEM source for further info)
%
% PriorPar (Struct)
%   Hyperparameters that define priors, as passed to VBEM
%
%   .upi (1xK)
%       Dirichlet prior for initial state probabilities
%   .ua (KxK)
%       Dirichlet prior for each row of transition matrix
%   .mu (1xK)
%       Gaussian-Gamma/Wishart prior for state means 
%   .beta (Kx1)
%       Gaussian-Gamma/Wishart prior for state occupation count 
%   .W (DxDxK)
%       Gaussian-Gamma/Wishart prior for state precisions
%   .v (Kx1)
%       Gaussian-Gamma/Wishart prior for degrees of freedom
%
%
% Outputs
% -------
%
% PostHpar (Struct)
%   Updated hyperparameters (same fields as PriorPar)
%
% PostPar
%   Maximum likelihood parameters associated with updated priors.
%
%   .pi (1xK)
%       Initial state probabilities
%   .A (KxK)
%       Transition matrix
%   .m (1xK)
%       Emission FRET levels
%   .sigma (1xK)
%       Emission FRET standard deviation 
%
% TODO: this currently does not work for 2D donor/acceptor inference
%
% Jan-Willem van de Meent
% $Revision: 1.00$  $Date: 2011/05/21$


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
% {mu, lambda} ~ NormalGamma(mu, lambda | u.mu, u.beta, u.nu, u.W)
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

% Get dimensions
N = length(out);
K = length(out{1}.Wpi);

% Prior parameters to struct form
u_old = struct();
u_old.pi = PriorPar.upi(:);
u_old.A = PriorPar.ua;
u_old.mu = PriorPar.mu(:);
u_old.beta = PriorPar.beta(:);
u_old.nu = PriorPar.v(:);
u_old.W = PriorPar.W(:);

% Get posterior hyperparameters w = u' for q(theta)
w = struct('pi', cellfun(@(o) o.Wpi(:), out, 'UniformOutput', false), ...
           'A', cellfun(@(o) o.Wa, out, 'UniformOutput', false), ...
           'mu', cellfun(@(o) o.m(:), out, 'UniformOutput', false), ...
           'beta', cellfun(@(o) o.beta(:), out, 'UniformOutput', false), ...
           'W', cellfun(@(o) o.W(:), out, 'UniformOutput', false), ...
           'nu', cellfun(@(o) o.v(:), out, 'UniformOutput', false));

% Init struct for updated params
u = struct();

% (mu, lambda) ~ Normal-Gamma
% Expectation Value eta1: E[lambda] = w.nu * w.W
E_l = arrayfun(@(w) w.nu .* w.W, w, 'UniformOutput', false);
E_l = sum([E_l{:}], 2) / N;

% (mu, lambda)
% Expectation Value eta2: E[mu * lambda] = w.nu * w.W * w.mu
E_ml = arrayfun(@(w) w.nu .* w.W .* w.mu, w, 'UniformOutput', false);
E_ml = sum([E_ml{:}], 2) / N;

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
E_log_g = arrayfun(@(w) -0.5 * (1 ./ w.beta ...
                               + w.nu .* w.W .* w.mu.^2 ...
                               + log(pi ./ w.W) ...
                               - psi(w.nu/2)), ...
                   w, 'UniformOutput', false);
E_log_g = sum([E_log_g{:}], 2) / N;

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
u.nu = lsqnonlin(root_fun, u_old.nu, 1 + zeros(size(u_old.nu)));

% (mu, lambda): Solve u.mu, u.beta and u.W
u.mu = E_ml ./ E_l;
u.beta = u.nu - 1;
u.W  = E_l ./ u.nu;


% (pi) ~ Dirichlet
% 
% Expectation value E[log pi]: Use same trick as for E[log g]:
%
% E[log pi] = - (Grad_nu' f(nu', chi')) / f
%           = - psi(Sum w.pi) + psi(w.pi)
E_log_pi = arrayfun(@(w) -psi(sum(w.pi)) + psi(w.pi), ...
                    w, 'UniformOutput', false);
E_log_pi = sum([E_log_pi{:}], 2) / N;

%(pi): solve system of equations
root_fun = @(p) E_log_pi + psi(sum(p)) - psi(p);
u.pi = lsqnonlin(root_fun, u_old.pi, zeros(size(u_old.pi)) + eps);


% A(k,:) ~ Dirichlet
% E[log A(k,:)]: Same as E[log pi]
E_log_A = arrayfun(@(w) bsxfun(@plus, -psi(sum(w.A,2)), psi(w.A)), ...
                    w, 'UniformOutput', false);
E_log_A = sum(cat(3, E_log_A{:}), 3) / N;

%A(k,:): solve system of equations 
u.A = zeros(size(u_old.A));
for k = 1:K
  root_fun = @(Ak) E_log_A(k,:) + psi(sum(Ak)) - psi(Ak);
  u.A(k, :) = lsqnonlin(root_fun, u_old.A(k, :), zeros(size(u_old.A(k, :))) + eps);
end

% Pack results back into legacy form
PostHPar = struct();
PostHPar.upi = u.pi';
PostHPar.ua = u.A;
PostHPar.mu = u.mu';
PostHPar.beta = u.beta;
PostHPar.v = u.nu;
PostHPar.W = u.W';

% Get Max Posterior Parameters (based on updated priors)
% (this is a pain)

p = cellfun(@(o) o.Wpi(:) - u_old.pi(:), out, 'UniformOutput', false);
p = sum([p{:}], 2);
Nk = cellfun(@(o) o.Nk(:), out, 'UniformOutput', false);
Nk = sum([Nk{:}], 2);
xbar = cellfun(@(o) o.Nk(:) .* o.xbar(:), out, 'UniformOutput', false);
xbar = sum([xbar{:}], 2) ./ Nk;
S = cellfun(@(o) o.Nk(:) .* squeeze(o.S), out, 'UniformOutput', false);
S = sum([S{:}], 2) ./ Nk;
A = cellfun(@(o) o.Wa - u_old.A, out, 'UniformOutput', false);
A = sum(cat(3, A{:}), 3);

% get posterior over all data (based on updated priors)
W = struct('pi', u.pi + p, ... 
           'A', u.A + A, ...
           'mu', (u.beta .* u.mu + Nk .* xbar) ./ (u.beta + Nk), ...
           'beta', u.beta + Nk, ...
           'W', 1./ (1 ./ u.W + Nk .* S + u.beta .* Nk .* (xbar - u.mu).^2 ./ (u.beta + Nk)), ...
           'nu', u.nu + Nk);

% Pack MAP results
PostPar = struct();
PostPar.pi = normalise(W.pi);
PostPar.A = normalise(W.A, 2); 
PostPar.mu = W.mu;
PostPar.sigma = 1./sqrt(W.nu .* W.W);