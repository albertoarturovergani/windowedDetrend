function [data,times,truth,artifact,metadata]=tmspath_demo_data(toy)
% Synthetic EEG for the demo; no toolbox and no global RNG changes.
% Same portable draw sequence as Python demo_multichannel_data.py.
ne=toy.n_epochs; nc=toy.n_channels; fs=toy.fs_hz;
assert(ne>=2 && nc>=8,'Use at least two epochs and eight channels');
assert(toy.seed>0 && toy.seed<2147483647 && toy.seed==fix(toy.seed),'Invalid demo seed');
assert(all([toy.stimulus_jitter_s toy.channel_jitter_s toy.amplitude_jitter toy.noise_sd]>=0),'Negative jitter/noise');
assert(toy.stimulus_jitter_s+toy.channel_jitter_s<.003 && toy.amplitude_jitter<1,'Jitter too large');
state=double(toy.seed);
times=(round(-.1*fs):round(.25*fs)-1)'/fs; nt=numel(times);
channel_latency=toy.channel_jitter_s*(2*uniform(nc)-1);
stimulus_latency=toy.stimulus_jitter_s*(2*uniform(ne)-1);
channel_gain=4+2*uniform(nc); channel_gain(1:2)=[80;65];
phase=2*pi*uniform(nc); neural_gain=.9+.2*uniform(nc);
data=zeros(ne,nc,nt); truth=data; artifact=data;
onsets=zeros(ne,nc); peaks=onsets; gains=onsets;
for e=1:ne
 for c=1:nc
  random=uniform(2);
  onset=.003+stimulus_latency(e)+channel_latency(c); peak=onset+.006;
  gain=channel_gain(c)*(1+toy.amplitude_jitter*(2*random(1)-1));
  u=max(times-onset,0); pulse=-gain*(-expm1(-u/.0025));
  tail=times>=peak;
  pulse(tail)=-gain*(-expm1(-.006/.0025))*exp(-(times(tail)-peak)/.12);
  neural_u=max(times-(.003+stimulus_latency(e)+.5*channel_latency(c)),0);
  response=3*neural_gain(c)*(1+.1*(2*random(2)-1))*exp(-25*neural_u).*sin(2*pi*40*neural_u);
  background=.3*sin(2*pi*11*times+phase(c)+2*pi*(e-1)/ne);
  pairs=ceil(nt/2); u1=uniform(pairs); u2=uniform(pairs);
  radius=sqrt(-2*log(u1)); angle=2*pi*u2;
  noise=reshape([radius.*cos(angle) radius.*sin(angle)]',[],1); noise=noise(1:nt);
  truth(e,c,:)=reshape(response+background+toy.noise_sd*noise,1,1,[]);
  artifact(e,c,:)=reshape(pulse,1,1,[]); data(e,c,:)=truth(e,c,:)+artifact(e,c,:);
  onsets(e,c)=onset; peaks(e,c)=peak; gains(e,c)=gain;
 end
end
labels=arrayfun(@(c)sprintf('EEG%02d',c),1:nc,'UniformOutput',false);
metadata=struct('channel_labels',{labels},'stimulus_jitter_s',stimulus_latency, ...
 'channel_jitter_s',channel_latency,'artifact_onsets_s',onsets,'artifact_peaks_s',peaks, ...
 'artifact_gains',gains,'seed',toy.seed,'fs_hz',fs);
 function values=uniform(n)
  values=zeros(n,1);
  for i=1:n
   state=mod(16807*state,2147483647); values(i)=state/2147483647;
  end
 end
end
