% windowedDetrend 0.1.0 - Post-epoching TMS-EEG detrending.
%
% Core
%   tmspath_config     - Default parameters (seconds; indices 1-based).
%   tmspath_detrend    - Clean epoched numeric data and return fitted trend.
%   tmspath_eeglab     - Adapter for EEGLAB epoched structures (milliseconds).
%
% Validation and packaging
%   tmspath_synthetic  - Damped oscillator and capacitive artifact generator.
%   run_benchmark      - Native grid or shared Python/MATLAB comparison.
%   run_scalp_benchmark - Shared 64-channel scalp benchmark, five methods.
%   test_plugin        - Core and adapter checks.
%   build_toolbox      - Build a local .mltbx installer.
