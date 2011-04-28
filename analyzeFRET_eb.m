%%%%%%%%%%%%%%%%%%%%%%%% Parameter Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
% analyzeFRET program settings
[load_name Kmax H I] = FRETparams_eb('analyzeFRETopts');

% set the vb_opts for VBEM
vb_opts=FRETparams_eb('set_vbem_vb_opts');


%%%%%%%%%%%%%%%%%%%%%%%% VBHMM preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load(load_name)

% get number of traces in data
N = length(FRET)
    
out = cell(N,Kmax,H);
LP = -inf*ones(N,Kmax,H);
priors = cell(Kmax,H);

for n=1:N
    for k=3
        for h=1:1
            % make hyperparameters for the trial
            priors{k,h} = FRETparams_eb('set_hparams',k,h);

            for i=1:I
                disp(sprintf('n:%d k:%d h:%d i:%d',n,k,h,i))
                initM = get_initM(priors{k,h},length(FRET{n}));
                out{n,k,h} = VBEM_eb(FRET{n}', initM, priors{k,h},vb_opts);
                % Only save the iterations with the best out.F
                if out{n,k,h}.F(end) > LP(n,k,h)
                    LP(n,k,h) = out{n,k,h}.F(end);
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%% VBHMM postprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%

% analyze accuracy and save analysis
disp('Analyzing results...')
d_t = clock;
save_name = sprintf('%s_out_D%02d%02d%02d_T%02d%02d',load_name,d_t(2),d_t(3),d_t(1)-2000,d_t(4),d_t(5));
save(save_name)

%%
% sum evidence over traces
% note: could normalize evidence by trace length - one reason to not do
% this is that longer traces probably have more accurate inference and,
% consequently should be weighted more heavily
LPkh = squeeze(sum(LP,1));

% find hyperparameters which maximized evidence
[ig bestLP] = max(LPkh(:));
[Kbest Hbest] = ind2sub(size(LPkh),bestLP);

% consolidate posteriors from best prior
bestOut = out(:,Kbest,Hbest);

% compute posterior hyperparmaters and most probable posterior parameters
[PostHpar PostPar] = get_norm_post_par(bestOut,priors{Kbest,Hbest});

z_hat1 = cell(1,N); x_hat1 = cell(1,N);
% get viterbi paths
for n = 1:N
    [z_hat1{n} x_hat1{n}] = chmmViterbi_eb(bestOut{n},FRET{n}');
end

% check results quality
[sumc sumv] = results_analysis_eb(bestOut,z_hat1,mun,zPath,xPath,Aobs);

% save(save_name)

out2 = cell(1,N);
LP2 = -inf*ones(1,N);

for n=1:N
    for i = 1:10
        disp(sprintf('n:%d i:%d',n,i))
        initM = get_initM(PostHpar,length(FRET{n}));
        temp_out = VBEM_eb(FRET{n}', initM, PostHpar,vb_opts);
        % Only save the iterations with the best out.F
        if temp_out.F(end) > LP2(n)
            LP2(n) = temp_out.F(end);
            out2{n} = temp_out;
        end
    end 
end

z_hat2 = cell(1,N); x_hat2 = cell(1,N);
for n = 1:N
    [z_hat2{n} x_hat2{n}] = chmmViterbi_eb(out2{n},FRET{n}');
end
% compute posterior hyperparmaters and most probable posterior parameters
[PostHpar2 PostPar2] = get_norm_post_par(out2,PostHpar);

% check results quality
[sumc2 sumv2] = results_analysis_eb(out2,z_hat2,mun,zPath,xPath,Aobs);




out3 = cell(1,N);
LP3 = -inf*ones(1,N);

for n=1:N
    for i = 1:10
        disp(sprintf('n:%d i:%d',n,i))
        initM = get_initM(PostHpar2,length(FRET{n}));
        temp_out = VBEM_eb(FRET{n}', initM, PostHpar2,vb_opts);
        % Only save the iterations with the best out.F
        if temp_out.F(end) > LP3(n)
            LP3(n) = temp_out.F(end);
            out3{n} = temp_out;
        end
    end 
end

z_hat3 = cell(1,N); x_hat3 = cell(1,N);
for n = 1:N
    [z_hat3{n} x_hat3{n}] = chmmViterbi_eb(out3{n},FRET{n}');
end
% compute posterior hyperparmaters and most probable posterior parameters
[PostHpar3 PostPar3] = get_norm_post_par(out3,PostHpar);

% check results quality
[sumc3 sumv3] = results_analysis_eb(out3,z_hat3,mun,zPath,xPath,Aobs);





out4 = cell(1,N);
LP4 = -inf*ones(1,N);

for n=1:N
    for i = 1:10
        disp(sprintf('n:%d i:%d',n,i))
        initM = get_initM(PostHpar3,length(FRET{n}));
        temp_out = VBEM_eb(FRET{n}', initM, PostHpar3,vb_opts);
        % Only save the iterations with the best out.F
        if temp_out.F(end) > LP4(n)
            LP4(n) = temp_out.F(end);
            out4{n} = temp_out;
        end
    end 
end

z_hat4 = cell(1,N); x_hat4 = cell(1,N);
for n = 1:N
    [z_hat4{n} x_hat4{n}] = chmmViterbi_eb(out4{n},FRET{n}');
end
% compute posterior hyperparmaters and most probable posterior parameters
[PostHpar4 PostPar4] = get_norm_post_par(out4,PostHpar);

% check results quality
[sumc4 sumv4] = results_analysis_eb(out4,z_hat4,mun,zPath,xPath,Aobs);






out5 = cell(1,N);
LP5 = -inf*ones(1,N);

for n=1:N
    for i = 1:10
        disp(sprintf('n:%d i:%d',n,i))
        initM = get_initM(PostHpar4,length(FRET{n}));
        temp_out = VBEM_eb(FRET{n}', initM, PostHpar4,vb_opts);
        % Only save the iterations with the best out.F
        if temp_out.F(end) > LP5(n)
            LP5(n) = temp_out.F(end);
            out5{n} = temp_out;
        end
    end 
end

z_hat5 = cell(1,N); x_hat5 = cell(1,N);
for n = 1:N
    [z_hat5{n} x_hat5{n}] = chmmViterbi_eb(out5{n},FRET{n}');
end
% compute posterior hyperparmaters and most probable posterior parameters
[PostHpar5 PostPar5] = get_norm_post_par(out5,PostHpar);

% check results quality
[sumc5 sumv5] = results_analysis_eb(out5,z_hat5,mun,zPath,xPath,Aobs);
%%


save(save_name)
