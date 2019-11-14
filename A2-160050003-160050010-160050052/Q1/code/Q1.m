%% Load Dataset
clc; clear;
load('dataset.mat');
checkerboard = imread('../input/checkerboard.jpg');

%% Normalization
f2D_mean = mean(f2D,2);
f3D_mean = mean(f3D,2);
f2D_dist = sqrt(2) / mean(sqrt(sum((f2D(1:2,:)-repmat(f2D_mean(1:2),1,size(f2D,2))).^2,1)),2);
f3D_dist = sqrt(3) / mean(sqrt(sum((f3D(1:3,:)-repmat(f3D_mean(1:3),1,size(f3D,2))).^2,1)),2);
T = [f2D_dist,0,0;0,f2D_dist,0;0,0,1] * [1,0,-f2D_mean(1);0,1,-f2D_mean(2);0,0,1];
U = [f3D_dist,0,0,0;0,f3D_dist,0,0;0,0,f3D_dist,0;0,0,0,1] * [1,0,0,-f3D_mean(1);0,1,0,-f3D_mean(2);0,0,1,-f3D_mean(3);0,0,0,1];
f2D_trans = T * f2D;
f3D_trans = U * f3D;

%% Solve for P
N = size(f2D_trans,2);
A = zeros(2*N,12);

for i = 1:N
    X = f3D_trans(1,i); Y = f3D_trans(2,i); Z = f3D_trans(3,i);
    x = f2D_trans(1,i); y = f2D_trans(2,i);
    A(2*i-1,:) = [X, 0, -X*x, Y, 0, -Y*x, Z, 0, -Z*x, 1, 0, -x];
    A(2*i,:)   = [0, X, -X*y, 0, Y, -Y*y, 0, Z, -Z*y, 0, 1, -y];
end

[~,S,V] = svd(A);
m = V(:,12);
disp(['Smallest singular value for A = ', num2str(S(12,12))])
P_hat = reshape(m,3,4);
P = inv(T) * P_hat * U

%% Decomposition of P = K[R| − RX0]
M = P(:,1:size(P,2)-1);
[q,r] = qr(flipud(M)');
r = flipud(r');
r = fliplr(r);
q = q';   
q = flipud(q);

Xo = -inv(M) * P(:,size(P,2))
K = r
R = q

%% Reconstruct 2D cordinates
h3D = P * f3D;
rf2D = h3D./repmat(h3D(3,:),3,1); 
RMSE = mean(sum((rf2D-f2D).^2,1));
disp(['RMSE between the marked points and estimated 2D projections of the marked 3D points = ', num2str(RMSE)])

test_h3D = P * test_f3D;
test_rf2D = test_h3D./repmat(test_h3D(3,:),3,1); 

%% Plotting Points 
% Measured 2D points - Blue
% Calculated 2D points (2D and 3D measurements for estimating P) - Green
% Calculated 2D points (2D and 3D measurements NOT used for estimating P) - Yellow 
figure(1);
imshow(checkerboard);
hold on;
plot(f2D(1,:), f2D(2,:), 'bO', 'LineWidth', 2, 'MarkerSize', 15);
hold on;
plot(rf2D(1,:), rf2D(2,:), 'gx', 'LineWidth', 2, 'MarkerSize', 15);
hold on;
plot(test_rf2D(1,:), test_rf2D(2,:), 'y.', 'LineWidth', 2, 'MarkerSize', 15);

%% Reason for Normalizing Data
% The normalization is basically a preconditioning to decrease CONDITION NUMBER of the matrix P (the larger the condition number, the nearer the matrix is to the singular matrix).
% Putting it simply, the matrix P (un-normalized) consists of products of image coordinates which can have different scale. 
% The source and target coordinate data are usually noisy. Without normalization, the data from source can have multiple orders of magnitude larger variance than from target (or vice versa).
% The homography estimation usually finds parameters in a least-squares sense - hence the best statistical estimate is found only if variances of the parameters are the same (or known beforehand, but it is more practical just to normalize the input).
% So normalization is essential not only for numerical stability, but also for more accurate estimation in presence of noise and faster solution (in case of iterative solver).

