function u_new = hstep_mm(w, u, stat, L)
% This function takes the posterior from a N element cell array, out, which
% contains posterior distributions for traces which were already fit. It
% uses thes posteriors to calculate a new set of hyperparameter priors,
% u_new,
% which reflect the mean and standard deviation of the N posteriors. Best
% fit parameters (i.e. most likely hidden state means, stdevs and
% transition frequencies) are also returned in the structure PostPar.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           %
% variable initialization   %
%                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(w);
% K is the same for each w(n), since 1 set of hparms should be used
K = length(w(1).pi);

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

if nargin < 4
    weight = false;
else
    weight = true;
    L(L<0) = 0;
    for n = 1:N
        wMtx(n,:) = L(n) .* stat(n).G(:)' ./ sum(stat(n).G(:)) ;
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
    piMtx(n,:) = w(n).pi - u.pi;
    muMtx(n,:) = stat(n).xmean;
    % take mode
    lambdaMtx(n,:) = squeeze(stat(n).xvar)'.^-1;
    
    % remember which states are populated in each trace
    not_empty(n,:) = (w(n).beta - u.beta)' > MIN_DATA;
    for k = 1:K
        Arows{k}(n,:) = normalize(w(n).A(k,:)-u.A(k,:));
    end        
end

for k=1:K
    if weight
        w_vec = wMtx(not_empty(:,k)==1,k);
        mu_mean(k) = sum(muMtx(not_empty(:,k)==1,k).*normalize(w_vec));
        lambda_mean(k) = sum(lambdaMtx(not_empty(:,k)==1,k).*normalize(w_vec));
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
u_new.W = lambda_var ./ (2*lambda_mean + eps);
u_new.nu = (lambda_mean ./ (u_new.W + eps))';

%% set beta such that mean 1/(beta*lambda_mean) = mu_var
%% --> beta = 1 /(mu_var*lambda_mean)
%
% above update wass wrong. mu_var should be defined in terms
% of hyperparams alone and not lambda_mean
% 
% mu_var = 1 / (beta * W * (v-2))
% --> beta = 1 / (mu_var * W * (v-2))
u_new.beta = 1 ./ (mu_var(:) .* u_new.W(:) .* (u_new.nu(:) - 2));

% mu should just be the most probable mu
u_new.mu = mu_mean;

% for dirichlet, total counts should be given by V_k = M_k(1-M_k) / 
% (a*+1), where M_k is
% the mean of the Kth state, a* is total counts and V_k is the variance of
% the kth state --> a* =[M_K(1-M_K)/V_k]-1
% use average variance of populated states only
sum_alpha = [pi_mean .* (1-pi_mean) ./ (pi_var + eps)] - 1;
sum_alpha = mean(sum_alpha(isfinite(sum_alpha)));

%a_k = mean_k*sum_alpha
u_new.pi = pi_mean * sum_alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% now do same for transition matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_new.A = zeros(K);
for k = 1:K
    A_mean = zeros(1,K);
    A_var = zeros(1,K);
    if weight
        for kk=1:K
            w_vec = wMtx(not_empty(:,kk)==1,k);
            A_mean(kk) = sum(Arows{k}(not_empty(:,kk)==1,kk).*normalize(w_vec));
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
    u_new.A(k,:) = A_mean * sum_alpha;    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% add counts if hyperparameters are going to 0 or NaN
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set all NaNs in mu to 10 
u_new.mu(isnan(u_new.mu)) = 10;

% fix v/w too small
vw_issues = u_new.nu < 5 | isnan(u_new.nu) | u_new.nu > 1e3;
if any(vw_issues)
    disp(sprintf('Warning: vw_issue. v:%s W:%s',num2str(u_new.nu','%g'),num2str(u_new.W,'%g')))
    u_new.nu(vw_issues) = 5;
    u_new.W(vw_issues) = 50;
end

% make sure beta is at least 0.1
u_new.beta(u_new.beta < 0.1 | isnan(u_new.beta)) = 0.1;
u_new.beta(u_new.beta > 1e3) = 1e3;

% make sure no NaNs in upi
u_new.pi(isnan(u_new.pi)) = 1e-10;
% make sure sum(upi) > 1 
if sum(u_new.pi) < 1
    u_new.pi = normalize(u_new.pi);
end
% every entry should be at least 0.001
u_new.pi(u_new.pi < 0.001) = 0.001;
if max(u_new.pi) > 100
    u_new.pi = u_new.pi / (max(u_new.pi)/100);
end
    
% same deal for rows of ua

% make sure no NaNs in ua
u_new.A(isnan(u_new.A)) = 1e-10;
u_new.A(u_new.A < 1e-10) = 1e-10;

% make sure sum(ua(k,:)) > 1 

for k = 1:K
    if sum(u_new.A(k,:),2) < 1 || sum(u_new.A(k,:),2)/k > 1e3
        disp(sprintf('Warning: ua issue. ua(%d,:) = %s',k,num2str(u_new.A(k,:))))
        if sum(u_new.A(k,:)) < 1
            u_new.A(k,:) = normalize(u_new.A(k,:),2);
        else
            u_new.A(k,:) = u_new.A(k,:) / (max(u_new.A(k,:))/1e3);
        end
    end
end

% every entry should be at least 0.001
u_new.A(u_new.A < 0.001) = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% set most probable posterior estimate
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mlA = zeros(k);
for n=1:N
    mlA = mlA + w(n).A - u.A;
end
mlA = normalize(mlA,2);

u_new = struct('mu', u_new.mu(:), ...
               'beta', u_new.beta(:), ...
               'nu', u_new.nu(:), ...
               'W', u_new.W(:), ...
               'pi', u_new.pi(:), ...
               'A', u_new.A);

% PostPar.m = mu_mean;
% PostPar.sigma = sqrt(1./lambda_mean);
% PostPar.pi = normalize(u_new.pi); 
% PostPar.A = mlA;
% PostPar.Wa = u_new.A;
% PostPar.Wan = normalize(u_new.A,2);

% mu_mean = mean(muMtx);
% mu_var = var(muMtx);
% lambda_mean = mean(lambdaMtx);
% lambda_var = var(lambdaMtx);
% pi_mean = mean(piMtx);
% pi_var = var(piMtx);
