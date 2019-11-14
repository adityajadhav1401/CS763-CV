%% Rigit Transform between 2 sets of 3D Points

%% Load Data
% load('../input/Q1data.mat');
image = imread('../input/wembley.jpeg');
imshow(image);
p2 = [1060, 719; 845, 677; 959, 534; 1126, 555;];
p1 = [18, 44; 0, 44; 0, 0; 18,0;];

H = homography(p2, p1);
p3 = [375, 433, 1; 1140, 517, 1; 1024, 808, 1;]';
% p3 = [311,455, 1; 1126, 555, 1;]'; 408, 466, 1; 702, 468,1]';
p4 = H*p3;
p4 = p4 ./ repmat(p4(3, :), 3, 1);
Length = p4(1,2)-p4(1,1) %%118.9513
Breadth = p4(2,3)-p4(2,2) %%74.4298
%% Your code here