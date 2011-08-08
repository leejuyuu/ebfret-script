function u_new = hstep_ml(w, u)
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
% Outputs
% -------
%
%   u_new : struct 
%       Updated hyperparameters
%
% Jan-Willem van de Meent
% $Revision: 1.10$  $Date: 2011/08/04$

% TODO: this currently does not work for 2D donor/acceptor inference
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

% Get dimensions
N = length(w);
K = length(w(1).pi);

% % get weights for updates
% if WeighTraces
%   % weigh by evidence of each trace
%   weights = normalise(cellfun(@(o) o.F(end), out));
% else
%   weights = ones(N,1) / N;
% end

% Dummy weights
for n = 1:N
  w(n).wt = 1;
end

% Init struct for updated params
u_old = u;

% (mu, lambda) ~ Normal-Gamma
% Expectation Value eta1: E[lambda] = w.nu * w.W
E_l = arrayfun(@(w) w.nu .* w.W .* w.wt, w, 'UniformOutput', false);
E_l = mean([E_l{:}], 2);

% (mu, lambda)
% Expectation Value eta2: E[mu * lambda] = w.nu * w.W * w.mu
E_ml = arrayfun(@(w) w.nu .* w.W .* w.mu .* w.wt, w, 'UniformOutput', false);
E_ml = mean([E_ml{:}], 2);

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
E_log_g = mean([E_log_g{:}], 2);

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
opts = optimset('display', 'off');
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
E_log_pi = arrayfun(@(w) -psi(sum(w.pi)) + psi(w.pi), ...
                    w, 'UniformOutput', false);
E_log_pi = mean([E_log_pi{:}], 2);

%(pi): solve system of equations
root_fun = @(p) E_log_pi + psi(sum(p)) - psi(p);
u.pi = lsqnonlin(root_fun, ...
                 u_old.pi, ...
                 zeros(size(u_old.pi)) + eps, ...
                 Inf + zeros(size(u_old.pi)), ...
                 opts);


% A(k,:) ~ Dirichlet
% E[log A(k,:)]: Same as E[log pi]
E_log_A = arrayfun(@(w) w.wt * bsxfun(@plus, -psi(sum(w.A,2)), psi(w.A)), ...
                    w, 'UniformOutput', false);
E_log_A = mean(cat(3, E_log_A{:}), 3);

%A(k,:): solve system of equations 
u.A = zeros(size(u_old.A));
for k = 1:K
  root_fun = @(Ak) E_log_A(k,:) + psi(sum(Ak)) - psi(Ak);
  u.A(k, :) = lsqnonlin(root_fun, ...
                        u_old.A(k, :), ...
                        zeros(size(u_old.A(k, :))) + eps, ...
                        Inf + zeros(size(u_old.A(k, :))), ...
                        opts);
end

u_new = u;

% u_new = struct('mu', u.mu, ...
%                'beta', u.beta, ...
%                'nu', u.nu, ...
%                'W', u.W, ...
%                'pi', u.pi, ...
%                'A', u.A);


% % Get Max Posterior Parameters (based on updated priors)
% % (this is a pain)

% p = cellfun(@(o) o.Wpi(:) - u_old.pi(:), out, 'UniformOutput', false);
% p = sum([p{:}], 2);
% Nk = cellfun(@(o) o.Nk(:), out, 'UniformOutput', false);
% Nk = sum([Nk{:}], 2);
% xbar = cellfun(@(o) o.Nk(:) .* o.xbar(:), out, 'UniformOutput', false);
% xbar = sum([xbar{:}], 2) ./ Nk;
% S = cellfun(@(o) o.Nk(:) .* squeeze(o.S), out, 'UniformOutput', false);
% S = sum([S{:}], 2) ./ Nk;
% A = cellfun(@(o) o.Wa - u_old.A, out, 'UniformOutput', false);
% A = sum(cat(3, A{:}), 3);

% % get posterior over all data (based on updated priors)
% W = struct('pi', u.pi + p, ... 
%            'A', u.A + A, ...
%            'mu', (u.beta .* u.mu + Nk .* xbar) ./ (u.beta + Nk), ...
%            'beta', u.beta + Nk, ...
%            'W', 1./ (1 ./ u.W + Nk .* S + u.beta .* Nk .* (xbar - u.mu).^2 ./ (u.beta + Nk)), ...
%            'nu', u.nu + Nk);

% % Pack MAP results
% w = struct();
% w.pi = normalise(W.pi);
% w.A = normalise(W.A, 2); 
% w.mu = W.mu;
% w.sigma = 1./sqrt(W.nu .* W.W);