function save_name = hFRET_double_decay(save_name,Ktru,FRET,tru_par)

% Parameter Setting 
addpath('./src')
K = 5      
H=1

vb_opts = get_hFRET_vbopts();


% VBHMM preprocessing
% get number of traces in data
N = length(FRET);   
out0 = cell(K,H,N);
LP0 = -inf*ones(K,H,N);
priors = cell(K,H);

for k=1:K
    if any(k==[1])
        continue
    end
    for h=1:H
        % make hyperparameters for the trial
        priors{k,h} = get_priors_double_decay(k,h);
        
        priors{k,h}.ua
                
        for n=1:N
            if size(FRET{n},1) > 1
                FRET{n} = FRET{n}';
            end
                disp(sprintf('k:%d h:%d n:%d',k,h,n))
                out = VBEM_eb(FRET{n}, priors{k,h}, priors{k,h},vb_opts);
                % not efficiently coded, but fine for now
                if out.F(end) >  LP0(k,h,n)
                    out0{k,h,n} = out;
                    LP0(k,h,n) = out0{k,h,n}.F(end);
                end        
        end
    end
end
save(save_name)

%%%%%%%%%%%%%%%%%%%%%%%% VBHMM postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
R = 20;
I = 20;
out = cell(R,K);
LP = cell(R,K);
sumc = cell(R,K);
sumv = cell(R,K);
z_hat = cell(R,K);
x_hat = cell(R,K);
u = cell(R,K);
theta = cell(R,K);
Hbest = zeros(1,K);

% sum evidence over traces
% note: could normalize evidence by trace length - one reason to not do
% this is that longer traces probably have more accurate inference and,
% consequently should be weighted more heavily
LPkh = sum(LP0,3);


% find hyperparameters which maximized evidence
% and consolidate posteriors from best prior
%%
for k=1:K
    if any(k==[1])
        continue
    end
    [ig Hbest(k)] = max(LPkh(k,:));
    out{1,k} = squeeze(out0(k,Hbest(k),:))';
    % compute posterior hyperparmaters and most probable posterior parameters
    [u{1,k} theta{1,k}] = get_ML_par(out{1,k},priors{k,Hbest(k)});
    z_hat{1,k} = cell(1,N); x_hat{1,k} = cell(1,N);
    for n = 1:N
        [z_hat{1,k}{n} x_hat{1,k}{n}] = chmmViterbi_eb(out{1,k}{n},FRET{n});
    end
    [sumc{1,k} sumv{1,k}] = bj_analysis(out{1,k}, z_hat{1,k},tru_par);
end
%%
save(save_name)

for r = 2:R
    for k = 1:K
        if any(k==[1])
            continue
        end
        out{r,k} = cell(1,N); LP{r,k} = -inf*ones(1,N);
        z_hat{r,k} = cell(1,N); x_hat{r,k} = cell(1,N);
        
        for n=1:N
                disp(sprintf('r: %d k: %d n:%d',r,k,n))
            for i = 1:I
                initM = get_M0(u{r-1,k},length(FRET{n}));
                temp_out = VBEM_eb(FRET{n}, initM, u{r-1,k},vb_opts);
                % Only save the iterations with the best out.F
                if temp_out.F(end) > LP{r,k}(n)
                    LP{r,k}(n) = temp_out.F(end);
                    out{r,k}{n} = temp_out;
                end
            end 
        end

        for n = 1:N
            [z_hat{r,k}{n} x_hat{r,k}{n}] = chmmViterbi_eb(out{r,k}{n},FRET{n});
        end
        % compute posterior hyperparmaters and most probable posterior parameters
        [u{r,k} theta{r,k}] = get_ML_par(out{r,k},u{r-1,k});

        % check results quality
        [sumc{r,k} sumv{r,k}] = bj_analysis(out{r,k},z_hat{r,k},tru_par);

        save(save_name)
    end
end

%priors{4,1}.ua
%for i = 1:size(sumc,1)
%lps= [-inf sumc{i,2}.LP -inf sumc{i,4}.LP];
%[ig kstar] = max(lps);
%disp([lps(2) lps(4) lps(4)-lps(2) kstar])
%end

%roundn(theta{end,2}.Wa,-2)
%roundn(theta{end,4}.Wa,-2)
 
