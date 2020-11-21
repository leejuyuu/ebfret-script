function make_test_data()
    N = 10;
    pi = ones(2, 1) * 0.5;
    A = ones(2) * 0.25;
    lengths = randi([500, 1500], N);
    traces = cell(N, 1);
    a = 2;
    b = 2/2000;
    beta = 10;
    m = [0, 5000];
    for i_trace = 1:N
        lambda = gamrnd(a, b, 2, 1);
        for i = 1:2
            mu(i) = normrnd(m(i), sqrt((beta*lambda(i))^(-1)));
        end
        trace = zeros(lengths(i_trace), 1);
        trace(1) = rand() > pi(1);
        for i_time = 2:lengths(i_trace)
            if rand() > 0.9
                trace(i_time) = ~trace(i_time - 1);
            else
                trace(i_time) = trace(i_time - 1);
            end
        end
        for i_time = 1:lengths(i_trace)
            state = trace(i_time) + 1;
            trace(i_time) = normrnd(mu(state), 1/lambda(state));
        end
        figure()
        plot(1:lengths(i_trace), trace)
        traces{i_trace} = trace;
    end
    keyboard
    save('test_traces.mat', 'traces');

end
