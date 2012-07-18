function [w, L, stat] = vbem_shmm(x, w0, u, mu0, varargin)
% vbem_shmm(x, w0, u, mu0, varargin)
%
% Variational Bayes Expectation Maximization for a Stepping Hidden 
% Markov Model (SHMM). 
%
% This model allows only transitions to the next state and pauses 
% (i.e. self-transitions). The positions of the states are assumed 
% to be known, up to a offset dmu which is learned from the data.
%
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
%   mu0 : K x 1
%       Locations of states relative to offset dmu
%       mu(k) = dmu + mu0(k). (assumed constant)
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
%
% Outputs
% -------
%
%   w : struct
%       Variational parameters of approximate posterior distribution 
%       for parameters q(theta | w) 
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
%
% Jan-Willem van de Meent

% set debug flag
Debug = true;

% parse inputs
ip = inputParser();
ip.StructExpand = true;
ip.addRequired('x', @(x) isnumeric(x) & (ndims(x)==2));
ip.addRequired('w0', @isstruct);
ip.addRequired('u', @isstruct);
ip.addRequired('mu0', @isnumeric);
ip.addParamValue('threshold', 1e-5, @isscalar);
ip.addParamValue('max_iter', 100, @isscalar);
ip.addParamValue('ignore', 'none', ...
                 @(s) any(strcmpi(s, {'none', 'spike', 'intermediate', 'all'})));
ip.parse(x, w0, u, mu0, varargin{:});

% collect inputs
args = ip.Results;
x = args.x;
w0 = args.w0;
u = args.u;
mu0 = args.mu0;

% get dimensions
[T D] = size(x);
K = size(u.A, 1);

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
    [E_ln_A, E_ln_px_z] = e_step_shmm(w, x, mu0);

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
    [g, xi, ln_Z] = forwback_banded(exp(E_ln_px_z), exp(E_ln_A), [1 zeros(1, K-1)]', [0 1]);  

    % COMPUTE LOWER BOUND L
    %
    % L = ln(p*(x)) - D_kl(q(theta| w) || p(theta | u))
    %
    % D_kl(q(theta) || p(theta)) = D_kl(q(mu, l) || p(mu, l)) 
    %                              + D_kl(q(A) || p(A)) 
    %                              + D_kl(q(pi) || p(pi))
    D_kl = kl_shmm(w, u);
    L(it) = ln_Z - D_kl;

    if Debug
        iter(it).g = g;
        iter(it).xi = xi;
        iter(it).ln_Z = ln_Z;
        iter(it).D_kl = D_kl;
        iter(it).w = w;
        %fprintf('it: %d ln_Z: %.4e D_kl: %.4e dmu: %05.2f sigma: %05.2f \n', ... 
        %         it, ln_Z, D_kl, mean(w.mu - mu0), 1./sqrt(w.W .* w.nu))
    end

    % print warning if lower bound decreases
    if it>2 && ((L(it) - L(it-1)) < -10 * args.threshold * abs(L(it))) 
        fprintf('[it: %d] warning: lower bound decreased by %e \n', ...
                it, L(it) - L(it-1));
    end

    % M STEP: UPDATE Q(THETA | W) BY CALCULATION OF VARIATIONAL 
    % PARAMETERS W FROM SUFFICIENT STATISTICS OF Q(Z)
    %
    % CB 10.60-10.63 and MJB 3.54 (JKC 25), 3.56 (JKC 21). 

    % check whether points need to be masked out
    w = m_step_shmm(u, x, g, xi, mu0);

    % check if the lower bound increase is less than threshold
    if (it>2)    
        if abs((L(it) - L(it-1)) / L(it-1)) < args.threshold || ~isfinite(L(it)) 
            L(it+1:end) = [];  
            break;
        end
    end
end

% ensure field order is correct
w = orderfields(w, fieldnames(args.u));

% wrap sufficient stats in struct
stat = struct();
stat.gamma = g;
stat.xi = xi;
stat.ln_Z = ln_Z;

% print debugging output
% if Debug
%     fprintf(['\nRUN SUMMARY:\n', ...
%              '  iterations   ', sprintf('%d', it), '\n', ...
%              '  L:           ', sprintf('% 7.1e', L(end)), '\n', ...
%              '    ln(Z):     ', sprintf('% 7.1e', ln_Z), '\n', ...
%              '    D_kl:      ', sprintf('% 7.1e  ', D_kl),'\n']);
% end