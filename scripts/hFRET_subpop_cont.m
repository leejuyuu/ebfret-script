% file to load/save results
save_name = './110427-06-hfret-jwm-ic12+puro+efg-gdpnp-1000nM-3-2S.mat';

% add path for dependencies
addpath(genpath('/proj/jv2403/code/matlab/jwm'));
addpath(genpath('/proj/jv2403/code/matlab/jonbron/vbFRET_June10'));

% prevent save name from being overwritten
save_name_ = save_name;
% load data
load(save_name)
% restore save name
save_name = save_name_;

% resume from last index
disp(sprintf('RESUMING FROM SAVED STATE'))

while any(cellfun(@isempty, u))
    [s, r] = find(cellfun(@isempty, u)', 1);

    out{r,s} = cell(1,N); 
    LP{r,s} = -inf*ones(1,N);
    z_hat{r,s} = cell(1,N); 
    x_hat{r,s} = cell(1,N);
    
    for n = 1:N
            disp(sprintf('r: %d s: %d n:%d',r,s,n))
            temp_out = cell(I,1);
            
            for i = 1:I
                if r == 1
                    initM = get_M0(u0{s}, length(FRET{n}));
                    temp_out{i} = VBEM_eb(FRET{n}, initM, u0{s}, vb_opts);
                else
                    initM = get_M0(u{r-1,s}, length(FRET{n}));
                    temp_out{i} = VBEM_eb(FRET{n}, initM, u{r-1,s}, vb_opts);
                end
            end 

            for i = 1:I
                % Only save the iterations with the best out.F
                if temp_out{i}.F(end) > LP{r,s}(n)
                    LP{r,s}(n) = temp_out{i}.F(end);
                    out{r,s}{n} = temp_out{i};
                end
            end 
    end

    for n = 1:N
        [z_hat{r,s}{n} x_hat{r,s}{n}] = chmmViterbi_eb(out{r,s}{n},FRET{n});
    end
    % compute posterior hyperparmaters and most probable posterior parameters
    if r ==1
        [u{r,s} theta{r,s}] = get_ML_par(out{r,s}, u0{s}, LP{r,s});
    else
        [u{r,s} theta{r,s}] = get_ML_par(out{r,s}, u{r-1,s}, LP{r,s});
    end
    save(save_name)
end

%matlabpool close force

