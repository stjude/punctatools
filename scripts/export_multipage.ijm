source_dir = getDirectory("Source Directory");
target_dir = getDirectory("Target Directory");

list = getFileList(source_dir);
for (i=0; i<list.length; i++) {
    path = source_dir + '/' + list[i];
    print(path);
    temp = split(list[i], '.');
    folder = target_dir + '/' +temp[0];
    File.makeDirectory(folder);
    run("Bio-Formats Importer", "open=[path] autoscale color_mode=Default open_files open_all_series view=Hyperstack stack_order=XYCZT");

	imglist = getList("image.titles");
	for (j=0; j<imglist.length; j++){
		title = getTitle();
		title = title.replace('/', '-');
		fn_out = folder + '/' + title + '.tif';
		saveAs('tiff', fn_out);
		close();
	}
}