function [neural,artifact]=tmspath_synthetic(times,tauRise,tauDecay,family)
% Figure-notebook equations in seconds; amplitudes in arbitrary units.
if nargin<4, family='windowed'; end
assert(tauRise>0 && tauDecay>0,'Positive taus required');
t=times(:); onset=.010; peak=.015; u=max(t-onset,0);
neural=5*exp(-25*u).*sin(2*pi*50*u);
switch family
 case 'windowed'
  artifact=-expm1(-u/tauRise); mask=t>=peak;
  artifact(mask)=-expm1(-(peak-onset)/tauRise)*exp(-(t(mask)-peak)/tauDecay);
 case 'biexp'
  artifact=-expm1(-u/tauRise).*exp(-u/tauDecay);
 otherwise, error('Unknown family');
end
artifact=-100*artifact;
end
