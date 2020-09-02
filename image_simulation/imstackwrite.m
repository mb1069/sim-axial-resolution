function imstackwrite(stack, filename)
	t = Tiff(filename, 'w');

	tagstruct.ImageLength = size(stack, 1);
	tagstruct.ImageWidth = size(stack, 2);
	tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
	tagstruct.BitsPerSample = 16;
	tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
	tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
	for k = 1:size(stack, 3)
		t.setTag(tagstruct)
        t.setTag('Compression', Tiff.Compression.LZW);
		t.write(uint16(stack(:, :, k)));
		t.writeDirectory();
	end

	t.close();
end