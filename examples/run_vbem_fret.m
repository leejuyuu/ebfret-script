% global options
save_name = 'save-name';
data_files = {'data-file.dat'};

K_values = [1, 2, 3, 4, 5];
restarts = 10;
num_cpu = 1;

% options for loading and stripping traces
opts.load_fret = load_fret_defaults();
opts.load_fret.min_length = 50;
opts.load_fret.max_outliers = 1;
opts.load_fret.strip_first = true;
opts.load_fret.remove_bleaching = true;

% options for vbem algorithm
opts.vbem = vbem_defaults();

% run script
vbem_fret(save_name, data_files, K_values, restarts, ...
          'load_fret', opts.load_fret, 'vbem', opts.vbem, 'display', 'states', 'num_cpu', num_cpu);
