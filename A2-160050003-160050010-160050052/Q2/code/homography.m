function [ H ] = homography( p1, p2 )
%HOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    P = zeros(8,9);
    for i=1:4
        P(2*i-1,:) = [-p1(i,1), -p1(i,2), -1, 0, 0, 0, p2(i,1)*p1(i,1), p2(i,1)*p1(i,2), p2(i,1)];
        P(2*i, :)  = [0, 0, 0, -p1(i,1), -p1(i,2), -1, p2(i,2)*p1(i,1), p2(i,2)*p1(i,2), p2(i,2)];
    end
    [~, ~, V]  = svd(P);
    h = V(:,end);
    H = reshape(h, [3,3])';
end

