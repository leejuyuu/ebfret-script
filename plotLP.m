function plotLP(sumc,Tvec,l,save_name)

scheme = {'r-x' 'b-x' 'c-x' 'g-x' 'm-x' 'k-x'};
circlec = {'ro' 'bo' 'co' 'go' 'mo' 'ko'};

LP = zeros(size(sumc));
ltext = {};

if mod(l,2)
    figure
    subplot(1,2,1)
else
    subplot(1,2,2)
end

hold on

for i=1:length(sumc(:));
    if ~isempty(sumc{i})
        LP(i) = sumc{i}.LP; 
    end
end

for i=1:size(LP,2)
    if sum(LP(:,i))~=0
        plot(LP(:,i),scheme{i})
        ltext{length(ltext)+1} = sprintf('k=%d',i);
    end
end
legend(ltext,'Location','Southeast')
ylabel('Sum(log(P))')
xlabel('Program iteration')

[ig bestLP] = max(LP');

for i = 1:length(bestLP)
    plot(i,LP(i,bestLP(i)),circlec{bestLP(i)},'MarkerSize',9)
end
%     title(sprintf('\\Sigma(LP) for %s, <\\sigma> = %.02f',filtype,x_axis(l)),'interpreter','none')
title(sprintf('\\Sigma(LP), T = %g',Tvec(l)))

if ~mod(l,2) && ~isempty(save_name)
    % unless this is set, matlab converts from vector image to bitmat then back
    % to vector
    set(gcf, 'renderer', 'painters');
    filname = sprintf('%s_LP_%d_%d',save_name,l-1,l);

    % print(gcf, '-dpdf', 'my-figure.pdf');
    print(gcf, '-dpng', [filname '.png']);
    print(gcf, '-depsc2', [filname '.eps']);
%     printpreview
%     close all
end
