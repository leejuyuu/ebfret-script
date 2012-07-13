function [w, L, stat] = vbem_hmm(x, w0, u, varargin)
% function [w, L, stat] = vbem_hmm(x, w0, u, varargin)
%
% Variational Bayes Expectation Maximization for a Hidden Markov Model
% with Gaussian emissions.
%
% Given a set of observations x(1:T), this algorithm returns an 
% approximation for the posterior distribution q(theta | w) over
% the model paramaters theta.
%
% Inputs
% ------
%
%   x : (T x D) float
%       Observation time series. May be one dimensional 
%       (e.g. a FRET signal), or higher dimensional 
%       (e.g. D=2 for a donor/acceptor signal)
%
%   u : struct 
%       Hyperparameters for the prior distribution p(theta | u)
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
%   w0 : struct 
%       Initial guess for the variational parameters of the 
%       approximating posterior q(theta | w). Same fields as u
%
% Variable Inputs
% ---------------
%
%   threshold : float (default: 1e-5)
%      Convergence threshold. Execution halts when the relative
%      increase in the lower bound evidence drop below threshold 
%
%   max_iter : int (default: 100)
%      Maximum number of iteration before execution is truncated
%
%   ignore : {'none', 'spike', 'intermediate', 'all'}
%      Ignore states with length 1 on viterbi path that either
%      collapse back to the previous state ('spike') or move
%      to a third state ('intermediate').
%
% Outputs
% -------
%
%   w : struct
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w). (same fields as u)
%
%   L : (I x 1) float
%       Lower bound estimate of evidence for each iteration
%
%   stat : struct
%       Statistics calculated under the approximate posterior for 
%       latent states q(z)
%
%       gamma : (T x K)
%           Posterior probability for z(t)
%               gamma(t, k) = E_q(z)[z(t, k)] 
%                           = p(z(t)=k | x(1:T))
%
%       xi : (T x K x K)
%           Joint posterior probilities for z(t) and z(t+1)
%               xi(t, k, l) = E_q(z)[z(t, k) z(t+1, l)]  
%                           = p(z(t)=k, z(t+1)=l | x(1:T))
%
%       ln_Z : float
%           Log normalization constant of q(z).
%           
%       G : (K x 1)
%           State occupation count Sum_t gamma(t, k)
%
%       xmean : (K x D)
%           Expectation of emission means under gamma(t, k)
%             xmean(k, d) = E_t[x(t, d)] 
%
%       xvar : (K x D x D)
%           Expectation of emission variances under gamma(t, k)
%             xvar(k, d, e) = E_t[(x(t, d) - xmean(k)) 
%                                 (x(t, e) - xmean(k))]
%
% Jan-Willem van de Meent (modified from Matthew Beal and Jonathan Bronson)
% $Revision: 1.00 $  $Date: 2011/08/03$

% Algorithm
% ---------
% 
% The VBEM algorithm approximates the evidence
%
%   p(x)  =  Int d theta  Sum_z  p(x, z, theta) 
%
% By optimization of a lower bound 
%
%   L  =  Int d theta  Sum_z  q(theta, z) ln[ p(x, z, theta) / q(theta, z) ]
%
% Here, q(theta, z) is a factorized approximating distribution for the 
% posterior in the parameters theta and the latent state sequence z
%
%   q(theta, z)  =  q(theta) q(z)
%                ~  p(theta, z | x)
%
% This model assumes a joint p(x, z, theta) of the following form
%
%   p(x, z, theta)    p(x | z, theta) p(z | theta) p(theta | u)
%
%   x(1:T)            Observation sequence
%   z(1:T)            Sequence of latent states
%   theta             Model parameters
%
%   p(x | z, theta)   Gaussian emission probabilities 
%
%                     p(x | z, theta)  =  Norm(x | theta(z).mu, theta(z).L) 
%
%   p(z | theta)      Markov chain probabilities for states z
%
%                     p(z)  =  p(z(1)) Prod_t p(z(t+1) | z(t))
%                              pi(z(1)) Prod_t A(z(t), z(t+1))
%
%   p(theta | u)      Prior on model parameters theta
%               
%                     p(pi)     =  Dir(pi | u.pi)
%                     p(A)      =  Prod_k Dir(A(k,:) | u.A(k,:))
%                     p(mu, L)  =  Norm(mu | u.mu, u.beta L)
%                                  Wish(L | u.W, u.nu)
%
% Credits
% -------
%
% This code is a rewrite of Jonathan Bronson's vbFRET project. The vbFRET
% project in turn borrows from Matthew Beal's VBEM code for HMM's
% with discrete emission probabilities. 
%
% References
% ----------
%
% [CB]  Chris Bishop
%       Pattern recognition and machine learning
%       Springer, New York, 2006
%
% [JEB] Jonathan E Bronson, Jingyi Fei, Jake M Hofman, 
%       Ruben L Gonzalez, Chris H Wiggins
%       Learning rates and states from biophysical time series: 
%       a Bayesian approach to model selection and single-molecule 
%       FRET data.
%       Biophysical Journal, 97(12), pp 3196-3205, 2009
%
% [JWM] TODO: INSERT REFERENCE TO WRITE-UP HERE
%
% [MJB] Matthew J Beal
%       Variational Algorithms for Approximate Bayesian Inference
%       PhD Thesis, University College London, 2003
%       http://www.cse.buffalo.edu/faculty/mbeal/

Debug = false;

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('x', @(x) isnumeric(x) & (ndims(x)==2));
ip.addRequired('w0', @isstruct);
ip.addRequired('u', @isstruct);
ip.addParamValue('threshold', 1e-5, @isscalar);
ip.addParamValue('max_iter', 100, @isscalar);
ip.addParamValue('ignore', 'none', ...
                 @(s) any(strcmpi(s, {'none', 'spike', 'intermediate', 'all'})));
ip.parse(x, w0, u, varargin{:});

% collect inputs
args = ip.Results;
x = args.x;
w0 = args.w0;
u = args.u;

% get dimensions
[T D] = size(x);
K = length(u.pi);

% set w to initial guess
w = w0;

if Debug
    iter = struct();
end

% Main loop of algorithm
for it = 1:args.max_iter
    % fprintf('[debug] vbem iteration: %d\n', it)

    % E-STEP: UPDATE Q(Z)
    %
    % q(z) = 1/Z_q(z) E_q(theta)[ ln p(x,z,theta) ]
    [E_ln_pi, E_ln_A, E_ln_px_z] = e_step_hmm(w, x);

    % Forward-back algorithm - computes expecation under q(z) of
    %
    %   g(t, k) = p(z(t)=k | x, theta) 
    %   xi(t, k, l) = p(z(t)=k, z(t+1)=l | x, theta)
    %   Z_(q(z)) = p*(x | theta) 
    %
    % using expectation values under q(theta)
    %
    %   pi* = exp(E[ln pi])
    %   A* = exp(E[ln A])
    %   p*(x | z, theta) = E[p(x | z, theta)]
    %
    % NOTE: technically this is part of the M-step, but we will
    % calculate the lower bound L to check for convergence before
    % updating the variational parameters w
    [g, xi, ln_Z] = forwback(exp(E_ln_px_z), exp(E_ln_A), exp(E_ln_pi));  

    % COMPUTE LOWER BOUND L
    %
    % L = ln(p*(x)) - D_kl(q(theta| w) || p(theta | u))
    %
    % D_kl(q(theta) || p(theta)) = D_kl(q(mu, l) || p(mu, l)) 
    %                              + D_kl(q(A) || p(A)) 
    %                              + D_kl(q(pi) || p(pi))
    L(it) = ln_Z - kl_hmm(w, u);

    if Debug
        iter(it).ln_Z = ln_Z;
        iter(it).D_kl_mu_L = D_kl_mu_L;
        iter(it).D_kl_A = -D_kl_A;
        iter(it).D_kl_pi = -D_kl_pi;
    end

    % print warning if lower bound decreases
    if it>2 && ((L(it) - L(it-1)) < -10 * args.threshold * abs(L(it))) 
        fprintf('Warning!!: Lower bound decreased by %e \n', ...
                L(it) - L(it-1));
    end

    % M STEP: UPDATE Q(THETA | W) BY CALCULATION OF VARIATIONAL 
    % PARAMETERS W FROM SUFFICIENT STATISTICS OF Q(Z)
    %
    % CB 10.60-10.63 and MJB 3.54 (JKC 25), 3.56 (JKC 21). 

    % check whether points need to be masked out
    if strcmp(args.ignore, 'none')
        w = m_step_hmm(u, x, g, xi);
    else
        z_hat = viterbi(E_ln_px_z, E_ln_A, E_ln_pi);
        [g_f, xi_f] = jitter_filter(z_hat, g, xi, args.ignore); 
        w = m_step_hmm(u, x, g_f, xi_f);
    end

    % check if the lower bound increase is less than threshold
    if (it>2)    
        if abs((L(it) - L(it-1)) / L(it-1)) < args.threshold || ~isfinite(L(it)) 
            L(it+1:end) = [];  
            break;
        end
    end
end

stat = struct();
stat.gamma = g;
stat.xi = xi;
stat.ln_Z = ln_Z;
%stat.G = G;
%stat.xmean = xmean;
%stat.xvar = xvar;

% print debugging output
if Debug
    fprintf(['\nRUN SUMMARY:\n', ...
             '  iterations   ', sprintf('%d', it), '\n', ...
             '  F:           ', sprintf('% 7.1e', L(end)), '\n', ...
             '    ln(Z):     ', sprintf('% 7.1e', ln_Z), '\n', ...
             '    D_kl_mu_l: ', sprintf('% 7.1e  ', D_kl_mu_L),'\n', ...
             '    D_kl_A:    ', sprintf('% 7.1e  ', D_kl_A),'\n', ...
             '    D_kl_pi:   ', sprintf('% 7.1e', D_kl_pi), '\n\n']);
end