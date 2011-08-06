function [w, L, stat] = vbem(x, w0, u, options)
% function [w, L, stat] = vbem(x, w0, u, options)
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
%       .nu (K x 1)
%           Normal-Wishart prior - degrees of freedom
%           (must be equal to beta+1)
%       .W (K x D x D)
%           Normal-Wishart prior - state precisions
%
%   w0 : struct 
%       Initial guess for the variational parameters of the 
%       approximating posterior q(theta | w). Same fields as u
%
%   options : struct
%       Algorithm options
%
%       .threshold : float (default: 1e-5)
%           Convergence threshold. Execution halts when the relative
%           increase in the lower bound evidence drop below threshold 
%
%       .maxIter : int (default: 100)
%           Maximum number of iteration before execution is truncated    
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
%       Z : float
%           Normalization constant of q(z).
%           
%       G : (K x 1)
%           State occupation count Sum_t gamma(t, k)
%
%       xmean : (K x D)
%           Expectation of emission means under gamma(t, k)
%             xmean(k, d) = E_t[x(t, d)] 
%
%       xsigma : (K x D x D)
%           Expectation of emission variances under gamma(t, k)
%             xsigma(k, d, e) = E_t[(x(t, d) - xmean(k)) 
%                                   (x(t, e) - xmean(k))]
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

% set defaults for options that are not supplied
if nargin < 4
    options = struct();
end
if ~isfield(options, 'threshold')    
    options.threshold = 1e-5;
end
if ~isfield(options, 'maxIter')    
    options.maxIter = 100;
end

% get dimensions
[T D] = size(x);
K = length(u.pi);

% move this down
Fold = -Inf;
w = w0;

if Debug
    iter = struct();
end

% Main loop of algorithm
for it = 1:options.maxIter
    % fprintf('[debug] vbem iteration: %d\n', it)

    % E-STEP: UPDATE Q(Z)
    %
    % ln q(z) = E[ln p(x, z | theta)]_q(theta) - ln(p(x))

    % Expectation of log intial state priors pi under q(pi | w.pi) 
    % (MJB 3.69, CB 10.66, JCK 41)
    %
    % E[ln(w.pi(k))]  =  Int d pi  Dir(pi | w.pi) ln(pi)
    %                 =  psi(w.pi(k)) - psi(Sum_l w.pi(l)))
    E_ln_pi = psi(w.pi) - psi(sum(w.pi)); 

    % Expectation of log transition matrix A under q(A | w.A) 
    % (MJB 3.70, JCK 42)
    %
    % E_ln_A(k, l)  =  psi(w.A(k,l)) - psi(Sum_l w.A(k,l))
    E_ln_A = bsxfun(@minus, psi(w.A), psi(sum(w.A, 2)));

    % Expectation of log emission precision |Lambda| 
    % under q(W | w.W) (CB 10.65, JKC 44)
    %
    % E_ln_det_L(k)  =  E[ln(|Lambda(k)|)]
    %                =  ln(|w.W|) + D ln(2) 
    %                   + Sum_d psi((w.nu(k) + 1 - d)/2)
    if D>1
        E_ln_det_L = zeros(K, 1);  
        for k=1:K
          E_ln_det_L(k) = log(det(w.W(k, :, :))) + D * log(2) + ...
                          sum(psi((w.nu(k) + 1 - (1:D)) / 2), 2);
        end
    else
        E_ln_det_L = log(w.W) + D * log(2) + ...
                     sum(psi(0.5 * bsxfun(@minus, w.nu + 1, (1:D))), 2);
    end

    % Expectation of Mahalanobis distance Delta^2 under q(theta | w)
    % (10.64, JKC 44)
    %
    % E_Delta2(t, k) 
    %   = E[(x(t,:) - mu(k,:))' * Lambda * (x(t,:) - mu(l,:))]
    %   = D / w.beta(k) 
    %    + w.nu(k) Sum_de dx(t, d, k) W(d, e) dx(t, e, k)
    if D>1
        % dx(d, t, k) = x(t, d) - mu(k, d)
        dx = bsxfun(@minus, x', reshape(w.mu', [D 1 K]));
        % W(d, e, k) = w.W(k, d, e)
        W = permute(w.W, [2 3 1]);
        % dxWdx(t, k) = Sum_de dx(d,t,k) * W(d, e, k) * dx(e, t, k)
        dxWdx = squeeze(mtimesx(reshape(dx, [1 D T K]), ...
                                mtimesx(reshape(W, [D D 1 K]), ...
                                        reshape(dx, [D 1 T K]))));
        % note, the mtimesx function applies matrix multiplication to
        % the first two dimensions of an N-dim array, while using singleton
        % expansion to the remaining dimensions. 
        % 
        % TODO: make mtimesx usage optional? (needs compile on Linux/MacOS)
    else
        % dx(t, k) = x(t) - mu(k)
        dx = bsxfun(@minus, x, w.mu');
        % dxWdx(t, k) = Sum_de dx(t,k) * W(k) * dx(t, k)
        dxWdx = bsxfun(@times, dx, bsxfun(@times, w.W', dx));
    end
    % E_md(t, k) = D / w.beta(k) + w.nu(k) * dxWdx(t,k)
    E_Delta2 = bsxfun(@plus, (D ./ w.beta)', bsxfun(@times, w.nu', dxWdx));

    % Expectation of p(x | z, theta) under q(theta | w)
    %
    % E_p_x_z(t, k)
    %   = (1 / 2 pi)^(D / 2)
    %     exp[E[ln(|Lambda(k,:,:|)]]^(1/2)
    %     exp(-0.5 * E[Delta(t,k)^2])
    E_p_x_z = (2 * pi)^(-D / 2) ...
              * bsxfun(@times, exp(0.5 * E_ln_det_L'), ...
                               exp(-0.5 * E_Delta2)) ...
              + eps;

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
    [g, xi, Z] = forwback(E_p_x_z, exp(E_ln_A), exp(E_ln_pi));  

    % COMPUTE LOWER BOUND L
    %
    % L = ln(p*(x)) - D_kl(q(theta| w) || p(theta | u))
    %
    % D_kl(q(theta) || p(theta)) = D_kl(q(mu, l) || p(mu, l)) 
    %                              + D_kl(q(A) || p(A)) 
    %                              + D_kl(q(pi) || p(pi))

    % D_kl(q(pi) || p(pi)) = sum_l (w.pi(l) - u.pi(l)) 
    %                              (psi(w.pi(l)) - psi(u.pi(l)))
    D_kl_pi = kl_dir(w.pi, u.pi);

    % D_KL(q(A) || p(A)) = sum_{k,l} (w.A(k,l) - u.A(k,l)) 
    %                                (psi(w.A(k,l)) - psi(u.A(k,l)))
    D_kl_A = kl_dir(w.A, u.A);

    
    % Calculate Dkl(q(mu, L | w) || p(mu, L | u)) 

    % pre-compute some terms so calculation can be vectorized
    % (this was done after profiling)
    if D > 1
        log_det_W_u = zeros(K, 1);
        log_det_W_w = zeros(K, 1);
        E_Tr_Winv_L = zeros(K, 1);
        dmWdm = zeros(K, 1);
        for k = 1:K
            log_det_W_u(k) = log(det(u.W(k, :, :)));
            log_det_W_w(k) = log(det(w.W(k, :, :)));
            E_Tr_Winv_L(k) = trace(inv(u.W(k, :, :) ...
                                       * w.W(k, :, :)));
            dmWdm = (w.mu(k,:) - u.mu(k,:))' ...
                    * w.W(k,:,:) ...
                    * (w.mu(k,:) - u.mu(k,:));
        end
    else
        % for the most common D=1 case we don't need 
        % calls to det, inv, mtimes
        log_det_W_u = log(u.W);
        log_det_W_w = log(w.W);
        E_Tr_Winv_L = w.W ./ u.W;
        dmWdm = w.W .* (w.mu-u.mu).^2;
    end 

    % Log norm const Log[B(W, nu)] for Wishart (CB B.79)
    log_B = @(log_det_W, nu) ...
            - (nu / 2) .* log_det_W ...
            - (nu * D / 2) * log(2) ...
            - (D * (D-1) / 4) * log(pi) ...
            - sum(gammaln(0.5 * bsxfun(@minus, nu + 1, (1:D))), 2);

    % E_q[q(mu, L | w)]
    % =
    % 1/2 E_q[log |L|]
    % + D log(w.beta / (2*pi)) 
    % - 1/2 D
    % + log(B(w.W, w.nu))
    % + 1/2 (w.nu - D - 1) * E_q[log |L|]
    % - 1/2 w.nu D
    E_log_NW_w = 0.5 * E_ln_det_L ...        
                 + 0.5 * D * log(w.beta ./ (2*pi)) ...
                 - 0.5 * D ...
                 + log_B(log_det_W_w, w.nu) ...
                 + 0.5 * (w.nu-D-1) .* E_ln_det_L ...
                 - 0.5 * w.nu * D;

    % E_q[log[Norm(mu | u.mu, u.beta L)]
    % =
    %   1/2 D log(u.beta / 2 pi) 
    %   + 1/2 E_q[log |L|]
    %   - 1/2 u.beta (D / w.beta + E_q[(mu-u.mu)^T L (mu-u.mu)])  
    E_log_Norm_u = 0.5 * (D * log(u.beta / (2*pi)) ...
                          + E_ln_det_L ... 
                          - D * u.beta ./ w.beta ...
                          - u.beta .* w.nu .* dmWdm);

    % E_q[log[Wish(L | u.W, u.nu)]
    % =
    % log(B(u.W, u.nu)) 
    % + 1/2 (u.nu - D - 1) E_q[log |L|]
    % - 1/2 w.nu Tr[Inv(u.W) * w.W]
    E_log_Wish_u = log_B(log_det_W_u, u.nu) ...
                   + 0.5 * (u.nu - D - 1) .* E_ln_det_L ...
                   - 0.5 * w.nu .* E_Tr_Winv_L;

    % E_q[p(mu, L | u)]
    % = 
    % E_q[log[Norm(mu | u.mu, u.beta L)]
    % + E_q[log[Wish(L | u.W, u.nu)]
    E_log_NW_u = E_log_Norm_u + E_log_Wish_u;

    % Dkl(q(mu, L | w) || p(mu, L | u)) 
    % =
    % E_q[log q(mu, L | w)] - E_q[log p(mu, L | u)]
    D_kl_mu_L = E_log_NW_w - E_log_NW_u;    

    % L = ln(Z) - D_kl(q(theta | w) || p(theta | u))
    L(it) = log(Z) - sum(D_kl_mu_L) - sum(D_kl_A) - sum(D_kl_pi);

    if Debug
        iter(it).ln_Z = log(Z);
        iter(it).D_kl_mu_L = D_kl_mu_L;
        iter(it).D_kl_A = -D_kl_A;
        iter(it).D_kl_pi = -D_kl_pi;
    end

    % print warning if lower bound decreses
    if it>2 && (L(it) < (1 - 10*options.threshold) * L(it-1)) 
        fprintf('Warning!!: Lower bound decreased by %e \n', ...
                L(it) - L(it-1));
    end

    % M STEP: UPDATE Q(THETA | W) BY CALCULATION OF VARIATIONAL 
    % PARAMETERS W FROM SUFFICIENT STATISTICS OF Q(Z)
    %
    % CB 10.60-10.63 and MJB 3.54 (JKC 25), 3.56 (JKC 21). 

    % Update for pi
    %
    % w.pi(k) = u.pi(k) + g(1, k) 
    w.pi = u.pi + g(1, :)'; 

    % Update fo A
    %
    % w.A(k, l) = u.A(k, l) + sum_t xi(t, k, l)
    w.A = u.A + squeeze(sum(xi, 1));

    % Calculate expectation of sufficient statistics under q(z)
    %
    %   G(k) = Sum_t gamma(t, k)
    %   xmean(k,d) = Sum_t gamma(t, k)/G(k) x(t,d) 
    %   xvar(k,d,e) = Sum_t gamma(t, k)/G(k) 
    %                           (x(t,d1) - xmean(k,d))
    %                           (x(t,d2) - xmean(k,e))
    % G(k) = Sum_t gamma(t, k)
    G = sum(g, 1)';
    G = G + 1e-10;
    % g0(t,k) = g(t,k) / G(k)
    g0 = bsxfun(@rdivide, g, reshape(G, [1 K]));
    % xmean(k,d) = Sum_t g0(t, k) x(t,d) 
    xmean = sum(bsxfun(@times, ...
                       reshape(g0', [K 1 T]), ...
                       reshape(x', [1 D T])), 3);
    % dx(k,d,t) = x(t,d) - xmean(k,d) 
    dx = bsxfun(@minus, reshape(x', [1 D T]), xmean);
    % xsigma(k, d, e) = Sum_t g0(t, k) dx(k, d, t) dx(k, e, t)
    xvar = squeeze(sum(bsxfun(@times, ...
                              reshape(g0', [K 1 1 T]), ... 
                              bsxfun(@times, ...
                                     reshape(dx, [K D 1 T]), ...
                                     reshape(dx, [K 1 D T]))), 4));

    % Updates for beta and nu: Add counts for each state
    w.beta = u.beta + G;
    w.nu = u.nu + G;

    % Update for mu: Weighted average of u.mu and xmean
    %
    % w.mu(k) = (u.beta(k) * u.mu(k) + G(k) * xmean(k))
    %           / (u.beta(k) + G(k))
    w.mu = (u.beta .* u.mu + G .* xmean) ./ w.beta;
                            
    % Update for W:
    %
    % Inv(w.W(k, d, e)) = Inv(u.W(k, d, e)
    %                     + G(k) * xsigma(k, d, e)
    %                     + (u.beta(k) * G(k)) / (u.beta(k) + G(k))  
    %                       * (xmean(k, d) - u.mu(k, e))
    %                       * (xmean(k, d) - u.mu(k, e)) 
    
    % dx0(k,d) = xmean(k,d) - u.mu(k,d)
    dx0 = xmean - u.mu;
    % xvar0(k, d, e) = dx0(k, d) dx0(k, e)
    xvar0 = bsxfun(@times, ...
                   reshape(dx0, [K D 1]), ...
                   reshape(dx0, [K 1 D]));
    % w.W = inv(inv(u.W)(k, d1, d2) + G(k) xsigma(k, d1, d2) 
    %           + ((u.beta(k) * G(k)) / w.beta(k)) xvar0(k, d1, d2))
    if D>1
        w.W = arrayfun(@(k) inv(inv(u.W(k,:,:)) ... 
                                + G(k) * xvar(k,:,:) ...
                                + (u.beta(k) * G(k))/w.beta(k) ...
                                   * xvar0(k,:,:)), ...
                       (1:K)');
    else
        w.W = 1 ./ (1 ./ u.W + G .* xvar ...
                    + (u.beta .* G) ./ w.beta .* xvar0);
    end
  
    % check if the lower bound increase is less than threshold
    if (it>2)    
        if abs((L(it) - L(it-1)) / L(it-1)) < options.threshold || ~isfinite(L(it)) 
            L(it+1:end) = [];  
            break;
        end
    end
end

stat = struct();
stat.gamma = g;
stat.xi = xi;
stat.Z = Z;
stat.G = G;
stat.xmean = xmean;
stat.xvar = xvar;

% print debugging output
if Debug
    fprintf(['\nRUN SUMMARY:\n', ...
             '  iterations   ', sprintf('%d', it), '\n', ...
             '  F:           ', sprintf('% 7.1e', L(end)), '\n', ...
             '    ln(Z):     ', sprintf('% 7.1e', log(Z)), '\n', ...
             '    D_kl_mu_l: ', sprintf('% 7.1e  ', D_kl_mu_L),'\n', ...
             '    D_kl_A:    ', sprintf('% 7.1e  ', D_kl_A),'\n', ...
             '    D_kl_pi:   ', sprintf('% 7.1e', D_kl_pi), '\n\n']);
end