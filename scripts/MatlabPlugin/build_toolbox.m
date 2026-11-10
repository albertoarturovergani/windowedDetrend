function output=build_toolbox(outputDir)
% Build with MATLAB R2023a+; produces a local installer, does not publish.
root=fileparts(mfilename('fullpath'));
if nargin<1, outputDir=fullfile(fileparts(root),'dist'); end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
opts=matlab.addons.toolbox.ToolboxOptions(root,'98edbf31-5b8a-4e73-9bb9-67bc23173da5');
opts.ToolboxName='windowedDetrend';opts.ToolboxVersion='0.1.0';
opts.AuthorName='Alberto Arturo Vergani';
opts.Summary='Standalone windowed detrending of epoched TMS-EEG';
opts.MinimumMatlabRelease='R2023a';
opts.ToolboxMatlabPath={root};
output=fullfile(outputDir,'windowedDetrend-0.1.0.mltbx');
opts.OutputFile=output;
matlab.addons.toolbox.packageToolbox(opts);
fprintf('Built %s\n',output);
end
