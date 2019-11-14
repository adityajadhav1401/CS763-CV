function [rotated, mask] = rotate(image, theta)
   rotated = zeros(size(image));
   rot_matrix = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
   [x, y]=meshgrid(1:size(image, 2), 1:size(image,1));
   center = size(image)/2;
   x = x - center(2);
   y = y - center(1);
   x = x(:);
   y = y(:);
   z = [x,y];
   z_rot = rot_matrix * z';
   x_rot = z_rot(1,:)';
   y_rot = z_rot(2,:)';
   x_rot = round(reshape(x_rot, size(image)));
   y_rot = round(reshape(y_rot, size(image)));
   mask = (x_rot<=center(2) & x_rot>=1-center(2)).*(y_rot<=center(1) & y_rot>=1-center(1));
   for i = 1:size(image,1)
       for j = 1:size(image, 2)
           if(mask(i,j)==0)
               rotated(i,j) = 0;
           else
               rotated(i,j) = image(round(y_rot(i,j))+center(1), round(x_rot(i,j))+center(2));
           end
       end
   end
   rotated = uint8(rotated);
   mask = uint8(mask.*255);
end
