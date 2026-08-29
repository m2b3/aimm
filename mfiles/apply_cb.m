clear all
% Apply the GBVS center bias to a directory of PNG maps.
% Run this from the repository root - every path below is relative to it.
%
% The Python equivalent is src/patch_reconstruction25.apply_center_bias();
% both build the same weight map and must stay in step.

% Dataset selection - specify which dataset to process
DATASET = 'P21_scegram';  % 'P21_scegram', 'HH25_indoor', 'HH25_outdoor', 'TEST'

% select maps

%--- aws maps
inpath = ['data/' DATASET '/maps/aws_ncb/'];
outpath = ['data/' DATASET '/maps/aws_cb/'];

%--- meaning (mm) maps
% inpath = ['data/' DATASET '/maps/meaning_ncb/'];
% outpath = ['data/' DATASET '/maps/meaning_cb/'];

%--- result (aimm) maps
% inpath = ['results/results_maps/' DATASET '/gemini-2.5-flash_original/maps_combined_raw/'];
% outpath = ['results/results_maps/' DATASET '/gemini-2.5-flash_original/maps_combined_cb/'];

% Create output directory if it doesn't exist
if ~exist(outpath, 'dir')
    mkdir(outpath);
end

% Center-bias template, 32x32, from gbvs-master/util/invCenterBias.mat.
% The tracked .txt holds 0.5 * (1 - invCenterBias), so the weight map
%   0.5 + 0.5 * (1 - invCenterBias)
% that the original GBVS unCenterBias branch uses is just 0.5 + the file.
% Weights run from 0.5 at the corners to 1.0 at the centre.
CB_template = 0.5 + csvread('data/templates_cb/centerbias_gbvs.txt');

% Get all image files
files = dir([inpath '*.png']);

% Process each image
for i = 1 :length(files)
    % Load image
    master_map = double(imread(fullfile(inpath, files(i).name)));

    % apply CB - resize from the template every time, so that a directory
    % holding more than one image size still gets the right weights
    CB = imresize( CB_template , size( master_map ) );
    master_map = master_map .* CB;
    master_map = mat2gray(master_map);

    [~, name, ~] = fileparts(files(i).name);
    imwrite(master_map, fullfile(outpath, [name '.png']));
end
