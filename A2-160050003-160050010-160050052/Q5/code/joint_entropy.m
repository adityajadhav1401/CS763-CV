function result = joint_entropy(fixed, moving, mask)
    result = zeros(26);
    fixed = double(fixed);
    moving = double(moving);
    for i = 1:size(fixed,1)
        for j = 1:size(fixed,2)
            if(mask(i,j) == 255)
                result(floor(fixed(i,j)./10)+1, floor(moving(i,j)./10)+1) = result(floor(fixed(i,j)./10)+1, floor(moving(i,j)./10)+1) + 1;
            end
        end
    end
    result = result./sum(sum(result));
    result = result.*log2(1./result);
    result(isnan(result)) = 0;
    result = sum(sum(result));
end