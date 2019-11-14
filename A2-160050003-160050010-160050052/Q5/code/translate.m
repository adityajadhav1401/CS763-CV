function translated = translate(image, tx)
    translated = zeros(size(image));
    translated(:, max(tx+1,1):min(size(image,2)+tx, size(image,2))) = image(:, max(1-tx,1):min(size(image,2)-tx, size(image,2)));
    translated = uint8(translated);
end