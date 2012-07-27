function u = hstep_shmm(w, weights)
% u = hstep_shmm(w, weights)
%
% Hyper parameter updates for empirical Bayes inference (EB)
% using a Stepping Hidden Markov Model.
%
% 
% Inputs
% ------
%
%   w : struct (N x 1)
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w), for each of N traces 
%
%       .A : K x 2
%           Dirichlet prior on self and forward transitions
%       .dmu : scalar
%           Normal-Wishart prior - state offset 
%       .beta : scalar
%           Normal-Wishart prior - state occupation count
%       .W : scalar
%           Normal-Wishart prior - state precision
%       .nu : scalar 
%           Normal-Wishart prior - degrees of freedom
%
%   weights : (N x 1) optional
%       Weighting for each trace in updates.
%
%
% Outputs
% -------
%
%   u : struct  
%       Hyperparameters for the prior distribution p(theta | u)
%       (same fields as w)
%
% Jan-Willem van de Meent
% $Revision: 1.10$  $Date: 2011/08/04$

% intialize empty weights if unspecified
if nargin < 2
    weights = ones(size(w));
end
% run normal-wishart updates for emission model parameters
u = hstep_nw(w, weights, {'dmu', 'beta', 'W', 'nu'});
% add dirichlet updates for transition matrix
u.A = hstep_dir({w.A}, weights);
% ensure fields are aligned with w
u = orderfields(u, fieldnames(w));