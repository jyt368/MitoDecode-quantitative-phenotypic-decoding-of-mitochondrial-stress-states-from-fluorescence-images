// Options
Dialog.create("Background Removal Settings");
	Dialog.addCheckbox("Batch process", false);
	Dialog.addNumber("Median filter iterations:", 1);
	Dialog.addNumber("Threshold multiply factor:", 0.7);
	Dialog.addNumber("Analyze Particles lower size (pixel unit):", 50);
	Dialog.addNumber("Analyze Particles upper size (pixel unit):", 10000);
	Dialog.show();
	
	doBatch = Dialog.getCheckbox();
	nIter = Dialog.getNumber();
	thFactor = Dialog.getNumber();
	lowsize=Dialog.getNumber();
	upsize=Dialog.getNumber();
	
// Segment
function processImage(nIter, thFactor) {
	orig = getTitle();
	name = replace(orig, ".tif", "");
	//run("Delete Slice");
    //run("Stack to Images");
  
    //selectImage(name + "-0001");
    //close();
    //selectImage(name + "-0002");
    
	run("Split Channels");
    selectWindow("C1-" + orig);
    close();
    selectWindow("C3-" + orig);
    close();

    selectWindow("C2-" + orig);
	
	
	// 2. Background subtraction
	run("Subtract Background...", "rolling=10 disable");
	
	current = getTitle();
	
	for (i = 0; i < nIter; i++) {
	
	    selectImage(current);
	    run("Unsharp Mask...", "radius=2 mask=0.50");
	    run("Duplicate...", "title=dup");
	    run("Median...", "radius=10");
	
	    imageCalculator("Subtract create", current, "dup");
	    current = getTitle();
	
	    close("dup");
	}
	
	selectImage(current);
	
    run("32-bit"); 
	// 3. Thresholding 1
	setAutoThreshold("Huang dark");
	getThreshold(lower, upper);
	scaledLower = lower * thFactor;
	setThreshold(scaledLower, upper);
	run("NaN Background");
	
	// 4. Thresholding 2
	setAutoThreshold("Otsu dark no-reset");
	getThreshold(lower, upper);
	scaledLower = lower * thFactor;
	setThreshold(scaledLower, upper);
	run("Convert to Mask");
	
	// 5. Morphological operations
	run("Options...", "iterations=3 count=5 black do=Close");
	run("Options...", "iterations=3 count=5 black do=Dilate");
	
	// 6. Particle analysis
	run("Analyze Particles...", 
	    "size=" + lowsize + "-" + upsize+ " pixel show=[Count Masks] display summarize overlay composite");
}


// Batch process
if (doBatch) {

    inputDir = getDirectory("Choose Input Folder");
    outputDir = getDirectory("Choose Output Folder");

    if (inputDir == "" || outputDir == "")
        exit("Batch processing canceled.");

    fileList = getFileList(inputDir);

    setBatchMode(true);

    for (i = 0; i < fileList.length; i++) {

        if (endsWith(fileList[i], ".tif") || endsWith(fileList[i], ".png") || endsWith(fileList[i], ".jpg")) {

            open(inputDir + fileList[i]);
            processImage(nIter, thFactor);

            saveAs("Tiff", outputDir + fileList[i]);
            close("*");
        }
    }

    setBatchMode(false);
    print("Batch processing complete.");
    // Save final Summary table
	if (isOpen("Summary")) {
	    selectWindow("Summary");
	    saveAs("Results", outputDir + "Summary.csv");
	    run("Close");
	}
	
	if (isOpen("Results")) {
	    selectWindow("Results");
	    saveAs("Results", outputDir + "results.csv");
	    run("Close");
	}


} else {
    if (nImages == 0)
        exit("No image open!");

    processImage(nIter, thFactor);
}

