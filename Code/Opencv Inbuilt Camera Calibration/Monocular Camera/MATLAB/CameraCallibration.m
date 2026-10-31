%% Camera Parameter Estimation

checkerboardPatterns = imageDatastore("..\CheckBoard_Pattern", "FileExtensions",'.jpg');
imageFileNames = checkerboardPatterns.Files;

for i= 1:size(imageFileNames)
    img = imread(imageFileNames{i});
%     %To Check the readed image
    if i == 1 
    figure, imshow(img), title("Original Image");
    end
end
%%
% class(checkerboardPatterns.Files)

% Detect checkerboards in images

% size(imageFileNames) is 32 so we are doing for 10 images only
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames(1:11));
imageFileNames = imageFileNames(imagesUsed);
%%
% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);
%%
% Generate world coordinates of the corners of the squares
squareSize = 10;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);
%%
% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);
%%
% % For example, you can use the calibration data to remove effects of lens distortion.
% undistortedImage = undistortImage(originalImage, cameraParams);

% Im = [originalImage undistortedImage]; 
% image(Im)   
% axis image off