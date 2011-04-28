function [PostHpar PostPar] = get_ML_par(out, PriorPar, LP)
% This function takes the posterior from a N element cell array, out, which
% contains posterior distributions for traces which were already fit. It
% uses thes posteriors to calculate a new set of hyperparameter priors,
% PostHpar,
% which reflect the mean and standard deviation of the N posteriors. Best
% fit parameters (i.e. most likely hidden state means, stdevs and
% transition frequencies) are also returned in the structure PostPar.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
% variable initialization   %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(out);
% K is the same for each out{n}, since 1 set of hparms should be used
K = length(out{1}.Wpi);

% Flag: if set to 1, when calculating sufficient statistics ignore:
% 1.    traces where the order of the states fit does not match the order
%       of the prior means 
%
% 2.    traces where FRET states are fit within 0.1 FRET of one another
IG_BAD = 1;


% fewest number of data points allowed to be used to learn about a state
MIN_DATA = 5;
% arrays to hold parameter information
muMtx = zeros(N,K);
lambdaMtx = zeros(N,K);
piMtx = zeros(N,K);
Arows = cell(1,K);
mu_mean = zeros(1,K);
mu_var = zeros(1,K);
lambda_mean = zeros(1,K);
lambda_var = zeros(1,K);
pi_mean = zeros(1,K);
pi_var = zeros(1,K);
for k = 1:K
    Arows{k} = zeros(N,K);
end
% if state is unpopulated in posterior, don't use it to calculate new
% parameters
not_empty = zeros(N,K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% inferred parameters states by LP and Nk?
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% array to hold weights of states (number of counts) so weighted averages
% can be used
wMtx = zeros(N,K);

if nargin < 3
    weight = false;
else
    weight = true;
    for n = 1:N
        LP(LP<0) = 0;
        wMtx(n,:) = LP(n) .* out{n}.Nk(:)' ./ sum(out{n}.Nk(:)) ;
    end
end    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% move posterior parameters from out into arrays
% and get means and variances of parameters 
% (except transition matrix - will be done later
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:N
    piMtx(n,:) = out{n}.Wpi - PriorPar.upi;
    muMtx(n,:) = out{n}.xbar;
    % take mode
    lambdaMtx(n,:) = squeeze(out{n}.S)'.^-1;
    
    % remember which states are populated in each trace
    not_empty(n,:) = (out{n}.beta - PriorPar.beta)' > MIN_DATA;
    for k = 1:K
        Arows{k}(n,:) = normalise(out{n}.Wa(k,:)-PriorPar.ua(k,:));
    end        
end

for k=1:K
    if weight
        w_vec = wMtx(not_empty(:,k)==1,k);
        mu_mean(k) = sum(muMtx(not_empty(:,k)==1,k).*normalise(w_vec));
        lambda_mean(k) = sum(lambdaMtx(not_empty(:,k)==1,k).*normalise(w_vec));
    else
        w_vec = 0;
        mu_mean(k) = mean(muMtx(not_empty(:,k)==1,k));
        lambda_mean(k) = mean(lambdaMtx(not_empty(:,k)==1,k));
    end
    mu_var(k) = var(muMtx(not_empty(:,k)==1,k),w_vec);
    lambda_var(k) = var(lambdaMtx(not_empty(:,k)==1,k),w_vec);
    % pi doesn't get weighted since its always based on 1 observation
    pi_mean(k) = mean(piMtx(not_empty(:,k)==1,k));
    pi_var(k) = var(piMtx(not_empty(:,k)==1,k));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compute hyperparameter values using sufficient
% stats for from the distribution of parameters
% observed distributions using means
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% for a wishart
% variance of lambda = 2v(W^2) 
% mean of lambda = vW
% v = mean/W --> variance = 2*mean*W
% W = variance / (2*mean)
PostHpar.W = lambda_var ./ (2*lambda_mean + eps);
PostHpar.v = (lambda_mean ./ (PostHpar.W + eps))';

% set beta such that mean 1/(beta*lambda_mean) = mu_var
% --> beta = 1 /(mu_var*lambda_mean)
PostHpar.beta = 1 ./ (mu_var .* lambda_mean + eps)';

% mu should just be the most probable mu
PostHpar.mu = mu_mean;

% for dirichlet, total counts should be given by V_k = M_k(1-M_k) / 
% (a*+1), where M_k is
% the mean of the Kth state, a* is total counts and V_k is the variance of
% the kth state --> a* =[M_K(1-M_K)/V_k]-1
% use average variance of populated states only
sum_alpha = [pi_mean .* (1-pi_mean) ./ (pi_var + eps)] - 1;
sum_alpha = mean(sum_alpha(isfinite(sum_alpha)));

%a_k = mean_k*sum_alpha
PostHpar.upi = pi_mean * sum_alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% now do same for transition matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PostHpar.ua = zeros(K);
for k = 1:K
    A_mean = zeros(1,K);
    A_var = zeros(1,K);
    if weight
        for kk=1:K
            w_vec = wMtx(not_empty(:,kk)==1,k);
            A_mean(kk) = sum(Arows{k}(not_empty(:,kk)==1,kk).*normalise(w_vec));
            A_var(kk) = var(Arows{k}(not_empty(:,kk)==1,kk),w_vec);
        end
    else
        for kk=1:K
            A_mean(kk) = mean(Arows{k}(not_empty(:,kk)==1,kk));
            A_var(kk) = var(Arows{k}(not_empty(:,kk)==1,kk));
        end
    end
    sum_alpha = [A_mean .* (1-A_mean) ./ (A_var + eps)] - 1;
    sum_alpha = mean(sum_alpha(isfinite(sum_alpha)));
    PostHpar.ua(k,:) = A_mean * sum_alpha;    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% add counts if hyperparameters are going to 0 or NaN
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set all NaNs in mu to 10 
PostHpar.mu(isnan(PostHpar.mu)) = 10;

% fix v/w too small
vw_issues = PostHpar.v < 5 | isnan(PostHpar.v) | PostHpar.v > 1e3;
if any(vw_issues)
    disp(sprintf('Warning: vw_issue. v:%s W:%s',num2str(PostHpar.v','%g'),num2str(PostHpar.W,'%g')))
    PostHpar.v(vw_issues) = 5;
    PostHpar.W(vw_issues) = 50;
end

% make sure beta is at least 0.1
PostHpar.beta(PostHpar.beta < 0.1 | isnan(PostHpar.beta)) = 0.1;
PostHpar.beta(PostHpar.beta > 1e3) = 1e3;

% make sure no NaNs in upi
PostHpar.upi(isnan(PostHpar.upi)) = 1e-10;
% make sure sum(upi) > 1 
if sum(PostHpar.upi) < 1
    PostHpar.upi = normalise(PostHpar.upi);
end
% every entry should be at least 0.001
PostHpar.upi(PostHpar.upi < 0.001) = 0.001;
if max(PostHpar.upi) > 100
    PostHpar.upi = PostHpar.upi / (max(PostHpar.upi)/100);
end
    
% same deal for rows of ua

% make sure no NaNs in ua
PostHpar.ua(isnan(PostHpar.ua)) = 1e-10;
PostHpar.ua(PostHpar.ua < 1e-10) = 1e-10;

% make sure sum(ua(k,:)) > 1 

for k = 1:K
    if sum(PostHpar.ua(k,:),2) < 1 || sum(PostHpar.ua(k,:),2)/k > 1e3
        disp(sprintf('Warning: ua issue. ua(%d,:) = %s',k,num2str(PostHpar.ua(k,:))))
        if sum(PostHpar.ua(k,:)) < 1
            PostHpar.ua(k,:) = normalise(PostHpar.ua(k,:),2);
        else
            PostHpar.ua(k,:) = PostHpar.ua(k,:) / (max(PostHpar.ua(k,:))/1e3);
        end
    end
end

% every entry should be at least 0.001
PostHpar.ua(PostHpar.ua < 0.001) = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% set most probable posterior estimate
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mlA = zeros(k);
for n=1:N
    mlA = mlA + out{n}.Wa - PriorPar.ua;
end
mlA = normalise(mlA,2);

PostPar.m = mu_mean;
PostPar.sigma = sqrt(1./lambda_mean);
PostPar.pi = normalise(PostHpar.upi); 
PostPar.A = mlA;
PostPar.Wa = PostHpar.ua;
PostPar.Wan = normalise(PostHpar.ua,2);



% mu_mean = mean(muMtx);
% mu_var = var(muMtx);
% lambda_mean = mean(lambdaMtx);
% lambda_var = var(lambdaMtx);
% pi_mean = mean(piMtx);
% pi_var = var(piMtx);
