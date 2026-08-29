clear all
% Wrapper around the GBVS model of Harel et al. (2006). The model itself is not
% distributed here - obtain it from https://github.com/Pinoshino/gbvs
%
% Run this from the repository root - every path below is relative to it.

% Dataset selection - specify which dataset to process
DATASET = 'P21_scegram';  % 'P21_scegram', 'HH25_indoor', 'HH25_outdoor', 'TEST'

inpath = ['data\' DATASET '\images\'];
outpath = ['data\' DATASET '\maps\gbvs_ncb\'];
p = makeGBVSParams();
% The paper's analyses are all ncb: GBVS carries its own image-independent
% centre bias, which is switched off here for a fair comparison.
p.unCenterBias = 1; % 1 for ncb (the paper), 0 for cb

% Create output directory if it doesn't exist
if ~exist(outpath, 'dir')
    mkdir(outpath);
end

% Get all image files
files = dir([inpath '*.png']);

% Process each image
for i = 1 :length(files)
    
    % Load image
    img = imread(fullfile(inpath, files(i).name));
    
    % Process with gbvs
    [out] = gbvs(img,p);
    map = out.master_map_resized;
    
    % Save as PNG with same filename
    [~, name, ~] = fileparts(files(i).name);
    imwrite(map, fullfile(outpath, [name '.png']));
    
%     % Save as MAT with same filename
%     save(fullfile(outpath, [name '.mat']),'map');
    
end