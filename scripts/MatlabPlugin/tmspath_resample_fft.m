function output = tmspath_resample_fft(values,count)
% Real Fourier resampling equivalent to scipy.signal.resample defaults.
% Base MATLAB only; unlike Signal Processing Toolbox resample, this uses
% spectral truncation/padding with periodic boundaries, not a polyphase FIR.
assert(isvector(values) && ~isempty(values) && isreal(values) && all(isfinite(values(:))), ...
 'Expected a finite real vector');
assert(isscalar(count) && isfinite(count) && count>=1 && count==fix(count),'Invalid output count');
values=double(values(:)); count=double(count); source_count=numel(values);
spectrum=fft(values); positive=complex(zeros(floor(count/2)+1,1));
shared=min(source_count,count); stop=floor(shared/2)+1;
positive(1:stop)=spectrum(1:stop);
if mod(shared,2)==0
 if count<source_count, positive(shared/2+1)=2*positive(shared/2+1);
 elseif source_count<count, positive(shared/2+1)=positive(shared/2+1)/2;
 end
end
% The output's DC and (when present) Nyquist bin must be real.
positive(1)=real(positive(1));
if mod(count,2)==0
 positive(end)=real(positive(end));
 full_spectrum=[positive;conj(positive(end-1:-1:2))];
else
 full_spectrum=[positive;conj(positive(end:-1:2))];
end
output=real(ifft(full_spectrum))*(count/source_count);
end
