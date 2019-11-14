fixed = imread('../input/barbara.png');
moving = imread('../input/negative_barbara.png');
% fixed = rgb2gray(imread('../input/flash1.jpg'));
% moving = rgb2gray(imread('../input/noflash1.jpg'));
tic;

[rotated, mask] = rotate(moving, 23.5);%%Angle to be rotated
figure;
imshow(fixed);
trans_rotated = translate(rotated, -3);%%tx also change in line 11
trans_mask = translate(mask, -3);%%tx also change in line 10
noisy = (double(trans_rotated)+rand(size(trans_rotated))*8).*double(trans_mask./255);
noisy(noisy>255) = 255;
noisy(noisy<0) = 0;
noisy = uint8(noisy);
figure;
imshow(noisy);
jtable = zeros(121,25);
for theta = -60:60
    for tx = -12:12
        [temp_rot, ~] = rotate(noisy, theta);
        [temp_mask, ~] = rotate(trans_mask, theta);
        temp_trans = translate(temp_rot, tx);
        temp_mask = translate(temp_mask, tx);
        jtable(theta+61, tx+13) = joint_entropy(fixed, temp_trans, temp_mask);
    end
end
[x, y] = find(jtable==min(min(jtable)));
Theta = x-61%%moving image to be rotated by this angle for image alignment value should be near -23.5
Tx = y-13%%moving image to be translated by this number of pixels for image alignment value should be +3
figure;
colormap spring;
[X, Y] = meshgrid(-12:12, -60:60);
surf(X,Y,jtable)%%surf plot 
xlim([-12 12])
ylim([-60 60])
toc;