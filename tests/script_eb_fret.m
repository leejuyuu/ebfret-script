x = load('test_traces.mat').traces;
max_val = zeros(size(x));
min_val = zeros(size(x));
for i = 1:length(x)
    max_val(i) = max(x{i});
    min_val(i) = min(x{i});
end
max_int = max(max_val);
min_int = min(min_val);
for i = 1:length(x)
    % Add 1e-10 to avoid 0 values
    x{i} = 0.9*(x{i} - min_int)/(max_int - min_int) + 0.1;
end

eb_fret(x', 1:4, 1);
