function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here 
    function children = findChildren(index, kinematic_chain)
      children = [];
      for k=1:size(index)
         a = find(kinematic_chain(:,2)==index(k));
         if a
            child_index = kinematic_chain(a,1);
            b = findChildren(child_index, kinematic_chain);
         else
            b = [];
         end
         children = union(children, b);
      end
      children = union(children, index);
    end
    
    result_pose = pose;
    for i = 1:size(rotations,1)
        children = findChildren([kinematic_chain(i,1)], kinematic_chain);
        parent = result_pose(kinematic_chain(i, 2), :);
        child_pose = result_pose(children, :);
        c = child_pose - repmat(parent, [numel(children), 1]);
        d = reshape(permute(rotations(i, :, :), [2,3,1]), size(rotations,2), size(rotations, 3)) * c';
        
        e = d' + repmat(parent, [numel(children), 1]);
        for j=1:numel(children)
            result_pose(children(j), :) = e(j, :);
        end
    end
end