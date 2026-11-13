function info = tmspath_fill_offset(signal,times,onset_s,peak_s,margin_s,mode,seed,epoch,channel)
% Filling following pipeline apply_offset_correction: Gaussian or Fourier.
% Source: ORIGINAL pre-onset samples trimmed by margin_s (fallback: all pre).
% Target: first to last sampled rise time, expanded by margin_s, epoch-clipped.
% Returned sample indices are inclusive and one-based. The epoch grid is fixed.
t=double(times(:)); signal=double(signal(:));
if nargin<6, mode='resample'; end
if nargin<7, seed=42; end
if nargin<8, epoch=1; end
if nargin<9, channel=1; end
assert(isvector(times) && numel(t)==numel(signal) && numel(t)>=2 && ...
 all(isfinite(t)) && all(isfinite(signal)),'Fill requires finite signal and times');
spacing=diff(t);
assert(all(spacing>0),'Filling requires increasing sample times');
if strcmp(mode,'resample')
 assert(all(abs(spacing-spacing(1))<=1e-12+1e-7*abs(spacing(1))), ...
  'Fourier filling requires uniformly increasing sample times');
end
assert(isscalar(margin_s) && isfinite(margin_s) && margin_s>=0,'Invalid fill_margin_s');
pre=find(t<onset_s); rise=find(t>=onset_s & t<peak_s);
assert(~isempty(pre) && ~isempty(rise),'Filling requires pre-onset source samples and a nonempty rise window');
source=find(t>=t(pre(1)) & t<=t(pre(end))-margin_s);
fallback=isempty(source); if fallback, source=pre; end
target=find(t>=t(rise(1))-margin_s & t<=t(rise(end))+margin_s);
if strcmp(mode,'gaussian')
 mu=mean(signal(source)); sigma=max(std(signal(source),1)*0.5,1e-8);
 trace_seed=mod(double(seed)+104729*(double(epoch)-1)+130363*(double(channel)-1),2^32);
 stream=RandStream('mt19937ar','Seed',trace_seed);
 values=mu+sigma*randn(stream,numel(target),1); method='normal';
elseif strcmp(mode,'resample')
 values=tmspath_resample_fft(signal(source),numel(target)); method='fourier';
else
 error('Unsupported fill mode');
end
info=struct('mode',mode,'method',method,'margin_s',margin_s, ...
 'source_first',source(1),'source_last',source(end),'source_count',numel(source), ...
 'source_fallback',fallback,'target_first',target(1),'target_last',target(end), ...
 'target_count',numel(target),'source_start_s',t(source(1)),'source_end_s',t(source(end)), ...
 'target_start_s',t(target(1)),'target_end_s',t(target(end)),'values',values);
if strcmp(mode,'gaussian')
 info.mean=mu; info.std=sigma; info.seed=seed; info.trace_seed=trace_seed;
 info.rng='MATLAB mt19937ar / local RandStream';
end
end
