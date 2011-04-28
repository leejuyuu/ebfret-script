function [out] = chmmVBEM(x, initM, PriorPar, options)
% dbstop if warning
% 
% function [out] = chmmVBEM(x, mix, PriorPar, options)
% function formerly called gmmVbemHmmFRET
% 
% Variational Bayes EM algorithm for a hidden Markov Gaussian Mixture
% Model. Much of the code is modified from Mohammad Emtiyaz Khan and Kevin
% Murphy's VBEMGMM program, which implements the Variational Bayesian
% treatment of a gaussian mixture model described in chapter 10.2 of
% Christopher Bishop's book 'Pattern recognition and machine learning'
% (2006). VBEMGMM is available for download at: 
% 
% http://www.cs.ubc.ca/~murphyk/Software/VBEMGMM/index.html
% 
% The forward-back algorithm is adapted from Matthew Beal's program to
% analyze discrete hidden Markov models using variational Bayes'. This
% code is available at http://www.cse.buffalo.edu/faculty/mbeal/
% 
% 
% Inputs:
%   x (DxN) - data set whose hidden states are to be inferred.
%   mix (structure) - gmm mixture model which is used as a starting
%       estimation for the chmm hidden states. The mix structure is
%       typically initialized with netlab's gmmem function. 
%   PriorPar(structure) - structure containing priors over the
%       hyperparameters. 
%   options (structure) - options for maxIter, threshold etc. etc.
% 
% Outputs:
%   out (structure) - the out structure contains the chmmVBEM estimates for
%   the posterior hyperparameters. 
% 
% Internal variables:
%   D - dimension of the data set.
%   T - the number of Data points in the data set.
%   K - number of gaussian clusters used to infer hidden states. 
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ref 1. IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE
% INTELLIGENCE,VOL. 28, NO. 4, APRIL 2006
%
% ref 2. Pattern Recognition and Machine Learning by Chris Bishop (CB),
% 2006, Springer 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dbstop if warning
% initialize variables
[D T] = size(x);
K = length(PriorPar.upi);
Fold = -Inf;
logLambdaTilde = zeros(1,K);
trW0invW = zeros(1,K);
lnZ = zeros(1,options.maxIter);
Fa = zeros(1,options.maxIter); 
Fpi = zeros(1,options.maxIter);
Fgw = zeros(1,options.maxIter);
F = zeros(1,options.maxIter); 

% hyperparameter priors
% prior over initial hidden state - MJB 3.6/3.54 (CB 10.39)
% upi = PriorPar.upi/K;
upi = PriorPar.upi;
% prior over transition matrix - MJB 3.4/3.56
ua = PriorPar.ua;
% gaussian mean prior variables  - CB 10.40
m0 = PriorPar.mu;
% beta0 = PriorPar.beta/K;
beta0 = PriorPar.beta;
% wishart prior variables - CB 10.40
W0 = PriorPar.W;
v0 = PriorPar.v;
% W0inv = inv(W0);
W0inv = W0.^-1;

beta = initM.beta;
v = initM.v;
m = initM.mu;
W = initM.W; 
Wpi = initM.upi;
Wa = initM.ua;

% Pre-calculate constant term used in lower bound estimation
% B(lambda0) CB B.79
logB0 = zeros(K,1);
for k=1:K
    logB0(k) = -(v0(k)/2)*log(det(W0(k))) - (v0(k)*D/2)*log(2) ...
          - (D*(D-1)/4)*log(pi) - sum(gammaln(   0.5*(v0(k)+1-(1:D))   ));
end

% Main loop of algorithm
for iter = 1:options.maxIter

    % E Step
 
    % <ln(Wa)> - MJB 3.70 (JCK 42)
    astar  = exp(  psi(Wa) - psi(sum(Wa,2))*ones(1,K)  );
    % <ln(Wpi)> - MJB 3.69 (CB 10.66 / JCK 41)
    pistar = exp(  psi(Wpi) - psi(sum(Wpi,2))  ); 
    % <ln(lambda)>  - CB 10.65 JKC 44.
    for k=1:K
%       logLambdaTilde(k) = sum(psi(  (v(k)*ones(1,D)+1-(1:D)) / 2  )) +...
%                     D*log(2) + log(det(W(:,:,k)));
      logLambdaTilde(k) = sum(psi(  (v(k)*ones(1,D)+1-(1:D)) / 2  )) +...
                    D*log(2) + log(det(W(k)));
    end
    
    % Calculate E - CB 10.64/JKC 44
    % OLD AND SLOW (but easier to understand)
    %%%% % <[(xn-muk)^T*lambda*(xn-muk)]> wrt. mu and lambda, equn 10.62
    %%%% % on p.478 of bishop.
    %%%% for t = 1:T
    %%%% for k=1:K
    %%%% diff = x{n}(:,t) - m(:,k);
    %%%% Eold{n}(t,k) = D/beta(k) + v(k)*diff'*W(:,:,k)*diff;
    %%%% end
    %%%% end
    % THE NEW HOTNESS (at least 10x faster)
    % logic: in general we expect D<K<<T
    % a T*K matrix:
    xWx=zeros(T,K);
    % a T*k matrix:
    xWm=zeros(T,K);
    % a vector of length K:
    mWm=zeros(1,K);
    for d1=1:D
        m1=m(d1,:);
        x1=x(d1,:);
        for d2=1:D
            m2=m(d2,:);
            x2=x(d2,:);
%             W12=reshape(W(d1,d2,:),[1 K]);
%             xWx=xWx+(x1.*x2)'*W12;
            xWx=xWx+(x1.*x2)'*W;
            xWm=xWm+x1'*(W.*m2);
            mWm=mWm+(m1.*W.*m2);
        end 
    end
    
    E =(xWx-2*xWm+ones(T,1)*mWm)*diag(v)+ones(T,1)*(D./beta)';
  
    % calculate <ln(P(Xn|Zn))>. TxK matrix - JKC 45, related to CB 10.67 
    % pXgivenZtilde = (pi^(-D/2))*exp ( repmat(0.5*logLambdaTilde, T,1) - 0.5*E ); 
    pXgivenZtilde = ((2*pi)^(-D/2))*exp ( 0.5*(logLambdaTilde(ones(1,T),:) - E) ) + 1e-100;
    
    % Forward-back algorithm
    % Probablities can be sub-normalized   
    
    [wa wpi xbar S Nk lnZ(iter) ] = forwbackFRET(astar,pXgivenZtilde,pistar,x);  
    
    % Compute F, straight after E Step.

    H =0;
    for k = 1:K
        % sum(H(q(Lamba(k)))), for Lt7 - CB B.82
%         logBk = -(v(k)/2)*log(det(W(:,:,k))) - (v(k)*D/2)*log(2)...
%                 - (D*(D-1)/4)*log(pi) - sum(gammaln(  0.5*(v(k)+1-(1:D))  ));
        logBk = -(v(k)/2)*log(det(W(k))) - (v(k)*D/2)*log(2)...
                - (D*(D-1)/4)*log(pi) - sum(gammaln(  0.5*(v(k)+1-(1:D))  ));
        H = H - logBk - 0.5*(v(k)-D-1)*logLambdaTilde(k) + 0.5*v(k)*D;
        % for Lt4 - Fourth term
%         diff = m(:,k) - m0;
%         mWm(k) = diff'*W(:,:,k)*diff; 
%         trW0invW(k) = trace(W0inv*W(:,:,k));
        diff = m(:,k) - m0(:,k);
        mWm(k) = diff'*W(k)*diff; 
        trW0invW(k) = trace(W0inv(k)*W(k));
    end
  
    % For <ln(p(mu,lambda))>, the Dkl of the Gaussian-Wishart
%     Lt41 = 0.5*sum(  D*log(beta0/(2*pi)) + logLambdaTilde' - D*beta0./beta...
%                       - beta0.*v.*mWm'  );
    Lt41 = 0.5*sum(  D*log(beta0/(2*pi)) + logLambdaTilde' - D*beta0./beta...
                      - beta0.*v.*mWm'  );
%     Lt42 = K*logB0 + 0.5*(v0-D-1)*sum(logLambdaTilde) - 0.5*sum(v.*trW0invW');
    Lt42 = sum(logB0) + 0.5*sum((v0-D-1).*logLambdaTilde') - 0.5*sum(v.*trW0invW');
    Lt4 = Lt41+Lt42;
    Lt7 = 0.5*sum(logLambdaTilde' + D*log(beta/(2*pi))) - 0.5*D*K - H;    
    Fgw(iter) = Lt4 - Lt7;    
    for kk = 1:K
        Fa(iter) = Fa(iter) - kldirichlet(Wa(kk,:),ua(kk,:));
    end;
%     Fpi(iter) = - kldirichlet(Wpi,upi_vec);
    Fpi(iter) = - kldirichlet(Wpi,upi);
    F(iter) = Fa(iter)+Fgw(iter)+Fpi(iter)+lnZ(iter); 
    % warning  if lower bound decreses
    if iter>2 && (F(iter)<F(iter-1)-1e-6) 
        disp(sprintf('Warning!!: Lower bound decreased by %f ', F(iter)-F(iter-1)));
    end

    % M Step: use sufficient statistics for M step update equations - CB
    % 10.60-10.63 and MJB 3.54 (JKC 25), 3.56 (JKC 21). beta and v are Kx1,
    % m is DxK, W is DxDxK. Wpi is 1xK and Wa is KxK.
    beta = beta0 + Nk;
    v = v0 + Nk;
%     m = ((beta0*m0)*ones(1,K) + (ones(D,1)*Nk').*xbar)./(ones(D,1)*beta');
    m = (beta0'.*m0 + Nk'.*xbar)./(beta');
    for k = 1:K
        % this is all modified
        mult1 = beta0(k)*Nk(k)/(beta0(k) + Nk(k));
        diff3 = xbar(:,k) - m0(:,k);
        W(k) = inv(W0inv(k) + Nk(k)*S(:,:,k) + mult1*diff3*diff3');
    end
    Wa = wa + ua;
    Wpi = wpi + upi; 
  
    % check if the lower bound increase is less than threshold
    if (iter>2)    
        if abs((F(iter)-F(iter-1))/F(iter-1))<options.threshold || ~isfinite(F(iter)) 
            lnZ(iter+1:end) = [];
            Fa(iter+1:end) = []; 
            Fpi(iter+1:end) = [];
            Fgw(iter+1:end) = [];
            F(iter+1:end) = [];  
            break;
        end
    end
end

% Assign all working variables to subparts of the "out" structure:
out.Wa = Wa;
out.Wpi = Wpi;
out.beta = beta;
out.m = m;
out.W = W;
out.v = v;
out.F = F;
out.S = S;
out.xbar = xbar;
out.Nk = Nk';
% out.scaledVar = zeros(D,D,K);
% % mode variance of the data analyzed (which might have been rescaled)
% for k=1:K
%     out.rescaledVar(:,:,k)=((inv(out.W(:,:,k))-inv(PriorPar.W))/(out.v(k)-D-1));
% end