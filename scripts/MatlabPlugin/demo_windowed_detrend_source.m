%% Windowed detrending: MATLAB demonstration
% Run all sections in order. This demonstration uses base MATLAB and imports
% the plugin from the same directory, without File Exchange installation.
% Pipeline: pulse handling -> bad-trial/channel removal -> epoching ->
% windowed detrending -> ICA -> feature extraction.
pluginDir = fileparts(mfilename('fullpath'));
addpath(pluginDir);

%% A. Toy signal parameters
% These values generate the input; they do not configure detrending.
toy.fs_hz = 10000;              % Sampling rate in Hz.
toy.epoch_start_s = -0.100;     % First epoch sample, relative to TMS.
toy.epoch_end_s = 0.250;        % Exclusive end; last sample is one step earlier.
toy.onset_s = 0.010;           % Synthetic artifact and oscillator onset.
toy.peak_s = 0.015;            % Charging-to-discharging transition.
toy.tau_rise_s = 0.003;        % Positive artifact charging time constant.
toy.tau_decay_s = 0.200;       % Positive artifact discharging time constant.
toy.artifact_gain = -100;      % Signed scale; positive values reverse polarity.
toy.neural_amplitude = 5;      % Oscillator amplitude scale, arbitrary units.
toy.neural_frequency_hz = 50;  % Oscillator frequency in Hz.
toy.neural_damping_per_s = 25; % Envelope exp(-rate * time).
toy.background_amplitude = 0.5; % Continuous synthetic background.
toy.background_frequency_hz = 11; % Background frequency in Hz.

%% B. Detrending parameters
% All 22 fields are explicit. No configuration is overwritten in later sections.
cfg = tmspath_config();
cfg.mode = 'windowed';        % Separate rise/decay; 'global' is external comparison only.
cfg.onset_s = 0.010;          % Correction start after pulse handling.
cfg.peak_s = [];              % Detect the peak within the user-chosen search window.
cfg.peak_search_end_s = 0.030; % Automatic search end, limited by correction end.
cfg.end_s = [];              % [] uses last sample; otherwise supply seconds.
cfg.rise_model = 'lagrange';  % Alternatives: 'poly', 'singlerise'.
cfg.decay_model = 'poly';     % Alternatives: 'singledecay', 'doubledecay'.
cfg.rise_degree = 3;         % Only for 'poly'; ignored for Lagrange (degree N-1).
cfg.decay_degree = 3;        % Polynomial degree; alternatives include 1 and 5.
cfg.pre_degree = [];        % [] preserves pre-onset data; 0 constant, 1 linear fit.
cfg.constrain_rise = false;  % Required for Lagrange; true anchors other rise fits to zero.
cfg.constrain_decay = true;  % Anchor to observed split amplitude; false fits freely.
cfg.fit_offset = false;     % Optional constant offset for exponential models.
cfg.tau_min_s = 0.0005;      % Lower fitted-tau bound, positive.
cfg.tau_max_s = 1.0;         % Upper fitted-tau bound, greater than lower bound.
cfg.grid_size = 21;         % Initial candidates per tau, integer >= 3.
cfg.refinements = 5;        % Default 5; integer >= 0. More levels improve tau resolution.
cfg.channels = [];          % [] selects all; 1 selects the toy channel (one-based).
cfg.global_model = 'biexp';  % Inactive in windowed mode; external comparator only.
cfg.fill_mode = 'none';      % Demo: off. Use 'gaussian' for experimental recordings.
cfg.fill_seed = 42;         % Reproducible local RNG; change to obtain another realization.
cfg.fill_margin_s = 0.005;   % Expand sampled rise by 5 ms on each side; 0 fills only rise.

% Evaluation and display settings do not control fitting.
evaluation_start_s = toy.onset_s;
evaluation_end_s = [];      % [] uses last sample; MSE interval for the polynomial fit.
plot_limits_ms = [-20 250]; % Display only.

%% Parameter options and interpretation
% Edit sections A and B, then Run All. Toy and fit parameters are independent.
% A changed toy onset/peak does not automatically change cfg.onset_s/peak_s.
% Peak: [] detects maximum absolute input amplitude; inspect automatic peaks.
% Rise: lagrange interpolates all nodes; poly fits a polynomial; singlerise
% models exponential charging. Decay: poly, singledecay or doubledecay.
% Polynomial degrees: nonnegative integers; 1 linear, 3 cubic, 5 quintic.
% More flexibility can absorb neural activity. Lagrange degree is N-1.
% Keep constrain_rise=false for Lagrange; subtraction zeroes its rise samples,
% including neural activity. Decay anchoring uses observed signal, not truth.
% Tau bounds, grid and refinements apply only to exponential fits. They are
% not the toy generating constants. Larger searches increase computation.
% fit_offset is inactive for polynomial and Lagrange models.
% Advanced core bases: alpha (alpha-shaped transient), biexp (multiplicative
% rise/decay). These are not recommended presets for individual windows.
% Not all model/anchor combinations are valid: Lagrange requires no anchor;
% zero-at-origin functions cannot match arbitrary nonzero anchors without an offset.
% Presets: lagrange/poly with cubic decay and unconstrained rise (default);
% poly/poly degrees 3/3 or 3/5; singlerise/singledecay with constrained rise/decay.
% Keep mode='windowed'. The global mode is an external-comparison utility.
% Times are seconds except plot_limits_ms. Tau values must be positive.
% Fit peak must be strictly inside the correction interval. Minimum samples:
% Lagrange 2; polynomial max(4,degree+2); exponential 4. Do not upsample to
% manufacture independent fitting information. The toy has one channel.

%% 1. Generate the toy signal using section A
assert(toy.fs_hz > 0 && toy.epoch_end_s > toy.epoch_start_s);
assert(toy.epoch_start_s <= toy.onset_s && toy.onset_s < toy.peak_s && toy.peak_s < toy.epoch_end_s);
assert(toy.tau_rise_s > 0 && toy.tau_decay_s > 0);
times = (round(toy.epoch_start_s*toy.fs_hz):round(toy.epoch_end_s*toy.fs_hz)-1)' / toy.fs_hz;
u = max(times-toy.onset_s,0);
neural = toy.neural_amplitude * exp(-toy.neural_damping_per_s*u) .* sin(2*pi*toy.neural_frequency_hz*u);
neural = neural + toy.background_amplitude * sin(2*pi*toy.background_frequency_hz*times);
artifact = toy.artifact_gain * (1-exp(-u/toy.tau_rise_s));
decay = times >= toy.peak_s;
artifact(decay) = toy.artifact_gain * (1-exp(-(toy.peak_s-toy.onset_s)/toy.tau_rise_s)) * exp(-(times(decay)-toy.peak_s)/toy.tau_decay_s);
data = reshape(neural+artifact,1,1,[]);
fprintf('Array shape (epochs, channels, samples): %d x %d x %d\n',size(data));

%% 2. Apply the configuration from section B
[corrected, trend, diagnostics] = tmspath_detrend(data,times,cfg);
detected_peak_s = diagnostics.fits{1}.peak_s;
fprintf('Detected/supplied split: %.3f ms; toy artifact split: %.3f ms\n',1000*detected_peak_s,1000*toy.peak_s);
assert(isequal(size(corrected),size(data)));
if isempty(cfg.pre_degree)
    outside = times < cfg.onset_s;
    if isfield(diagnostics.fits{1},'fill')
        fill = diagnostics.fits{1}.fill;
        outside(fill.target_first:fill.target_last) = false;
    end
    assert(isequal(corrected(:,:,outside),data(:,:,outside)));
end
disp('Output dimensions and applicable preservation checks passed');

%% 3. Evaluate recovery against the known oscillator
% The decay-only mask uses the toy peak independently of the fitted split.
recovered = reshape(corrected,[],1);
if isempty(evaluation_end_s), evaluation_end = times(end); else, evaluation_end = evaluation_end_s; end
mask = times >= evaluation_start_s & times <= evaluation_end;
decay_mask = mask & times >= toy.peak_s;
assert(any(mask) && any(decay_mask),'Evaluation windows must contain samples');
mse_full = mean((recovered(mask)-neural(mask)).^2);
mse_decay = mean((recovered(decay_mask)-neural(decay_mask)).^2);
fprintf('Evaluation interval: recovery MSE = %.6f a.u.^2\n',mse_full);
fprintf('Toy decay interval: recovery MSE = %.6f a.u.^2\n',mse_decay);

%% 4. Plot the input, fitted trend, recovery and error
fig = figure('Color','w','Position',[100 100 1000 850]);
layout = tiledlayout(fig,3,1,'TileSpacing','compact');
ax1 = nexttile(layout);
plot(times*1000, reshape(data,[],1), 'DisplayName','Observed: neural + artifact');
hold on;
plot(times*1000, reshape(trend,[],1), '--', 'DisplayName','Estimated trend');
ax2 = nexttile(layout);
plot(times*1000, neural, 'k', 'DisplayName','Known neural signal'); hold on;
plot(times*1000, recovered, 'DisplayName','Polynomial');
ax3 = nexttile(layout);
plot(times*1000, recovered-neural, 'r', 'DisplayName','Recovery error');
for ax = [ax1 ax2 ax3]
    xline(ax,toy.onset_s*1000,':','HandleVisibility','off');
    xline(ax,toy.peak_s*1000,':','HandleVisibility','off');
    xline(ax,detected_peak_s*1000,'--','Fitted split','HandleVisibility','off');
    ylabel(ax,'Amplitude (a.u.)');
    legend(ax,'Location','best'); xlim(ax,plot_limits_ms);
end
xlabel(ax3,'Time relative to TMS pulse (ms)');
title(layout,'Polynomial: Lagrange rise / cubic decay');

%% 5. Continue your existing pipeline
% Replace toy data with preprocessed epochs x channels x samples and times
% in seconds. Edit section B and inspect output before fitting subsequent ICA.
% For an existing EEGLAB dataset: tmspath_eeglab(EEG,cfg).
% A single-channel toy cannot demonstrate multichannel ICA.
data_for_ica = corrected;
