function [ H ] = ransacHomography( x1, x2, thresh )
%RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    maxIndex = zeros(size(x1,1),1);
    for i=1:1000
        R = randsample(size(x1,1), 4);
        p1 = x1(R,:);
        p2 = x2(R,:);
        H = homography(p1, p2);
        y2 = H*[x1,ones(size(x1,1),1)]';
        y2 = y2 ./ repmat(y2(3, :), 3, 1);
        y2 = y2(1:2, :)';
        index = sum((y2-x2).*(y2-x2),2)<=thresh;
        if(sum(index)>sum(maxIndex))
            maxIndex = index;
        end
    end
    p1 = x1(find(maxIndex==1),:);
    p2 = x2(find(maxIndex==1),:);
    H = homography(p1, p2);
end



