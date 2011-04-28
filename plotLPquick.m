LP = zeros(size(sumc));
scheme = {'r-x' 'b-x' 'y-x' 'g-x'};
ltext = {};
figure
hold on

for i=1:length(sumc(:));
    LP(i) = sumc{i}.LP;
end

for i=1:size(LP,2)
    plot(LP(:,i),scheme{i})
    ltext{length(ltext)+1} = sprintf('k=%d',i);
    
end

legend(ltext)

