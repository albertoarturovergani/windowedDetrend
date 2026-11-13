function test_filling()
% Base-MATLAB checks for optional filling and Fourier/Nyquist conventions.
t=(-200:500)'/2000; b=.7*sin(2*pi*11*t)+.2*cos(2*pi*23*t);
data=zeros(1,2,numel(t)); data(1,1,:)=b; data(1,2,:)=2*b+1;
cfg=tmspath_config();cfg.fill_mode='none'; cfg.peak_s=.015; cfg.channels=1;
[raw,rawTrend]=tmspath_detrend(data,t,cfg);
cfg.fill_mode='resample'; [filled,trend,info]=tmspath_detrend(data,t,cfg);
f=info.fits{1}.fill; expected=raw;
expected(1,1,f.target_first:f.target_last)=reshape(f.values,1,1,[]);
assert(isequal(trend,rawTrend) && isequal(filled,expected));
assert(isequal(filled(:,2,:),data(:,2,:)));
assert(max(abs(filled(:)-raw(:)))>0);
cfg.pre_degree=1; [~,~,preInfo]=tmspath_detrend(data,t,cfg);
assert(isequal(preInfo.fits{1}.fill.values,f.values));
f=tmspath_fill_offset(b,t,.010,.0153,0);
ids=find(t>=.010 & t<.0153);
assert(f.target_first==ids(1) && f.target_last==ids(end));
f=tmspath_fill_offset(b,t,.010,.0153,.5);
assert(f.source_fallback && f.target_count==numel(t));
for n=[1 2 3 8 9 32]
 for m=[1 2 3 7 8 31 32]
  assert(max(abs(tmspath_resample_fft(2.5*ones(n,1),m)-2.5))<1e-13);
 end
end
assert(max(abs(tmspath_resample_fft((-1).^(0:7)',16)-cos(pi*(0:15)'/2)))<1e-13);
bad=t; bad(10)=bad(10)+.00001;
failed=false; try, tmspath_detrend(data,bad,cfg); catch, failed=true; end; assert(failed);
cfg.fill_margin_s=-1;
failed=false; try, tmspath_detrend(data,t,cfg); catch, failed=true; end; assert(failed);
cfg.fill_margin_s=.005; cfg.mode='global';
failed=false; try, tmspath_detrend(data,t,cfg); catch, failed=true; end; assert(failed);
fprintf('MATLAB optional filling tests passed.\n');
cfg=tmspath_config(); assert(strcmp(cfg.fill_mode,'gaussian')); cfg.peak_s=.015;
global_rng=rng;
[a,~,ai]=tmspath_detrend(data,t,cfg); [b,~,~]=tmspath_detrend(data,t,cfg);
assert(isequal(a,b) && isequal(rng,global_rng));
cfg.fill_seed=43; c=tmspath_detrend(data,t,cfg); assert(~isequal(a,c));
f=ai.fits{1}.fill; source=reshape(data(1,1,f.source_first:f.source_last),[],1);
assert(abs(f.mean-mean(source))<1e-14 && abs(f.std-max(.5*std(source,1),1e-8))<1e-14);
cfg.fill_seed=42; cfg.channels=2; [~,~,single]=tmspath_detrend(data,t,cfg);
assert(isequal(single.fits{1}.fill.values,ai.fits{2}.fill.values));
long_t=(-1000:50000)'/10000; signal=sin(2*pi*17*long_t);
f=tmspath_fill_offset(signal,long_t,0,5,0,'gaussian',42,1,1);
z=(f.values-f.mean)/f.std; assert(abs(mean(z))<.03 && abs(std(z,1)-1)<.03);
f=tmspath_fill_offset(zeros(size(signal)),long_t,0,5,0,'gaussian',42,1,1); assert(f.std==1e-8);
fprintf('MATLAB Gaussian defaults, seeded streams and distribution checks passed.\n');
end
