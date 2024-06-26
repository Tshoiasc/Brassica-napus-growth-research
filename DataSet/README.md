# Dataset
This dataset is from the reserach (Collection of side view and top view RGB images of Brassica napus from a large scale, high throughput experiment).

### Image Dataset Description:

Top and side view RGB png images as analysed for 'Integrated Phenomics and Genomics reveals genetic loci associated with inflorescence growth in Brassica napus'.
Image collection details and examples of analysis are provided in the body and appendices of the paper.
The images were collected over a six week period, usually on a daily basis.
71 genotypes, two treatments, three replicates; total 426 plants.
1×top view image, 3×side view images (000, 045 & 090° rotations).

Example:

<img width="440" alt="image" src="https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/51d4b5ed-56e3-427a-a530-984576fed75b">

This dataset is used to train the neural network model for simulating the growth of rapeseed plants. The data annotation adopts a hierarchical recursive method based on the biological growth characteristics of rapeseed:

Start annotation from the stem root, and mark the leaves as potential branch points.

The main trunk is marked as "stem", the first-level branch is marked as "branch_1", and so on.

If a leaf develops a branch, the corresponding level is marked at the top inflorescence of the branch.

Repeat this process for each branch until the top flower of the stem is marked.

Like the structure below:

<img width="617" alt="image" src="https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/8d375d49-d1d9-49b0-b8ef-70407033517b">

So the data structure like below:

　stem    
　　│── branch_1    
　　│　　　　│── branch_2    
　　│　　　　│　　　　│── branch_3 (if available)    
　　│　　　　│── branch_2    
　　│　　　　│── ...    
　　│── branch_1    
　　│　　　　│── branch_2    
　　│　　　　│── ...    
　　│── ...    

Finally, a tree structure is formed, branch_1 belongs to stem, branch_2 belongs to branch_1, and so on.
This annotation method can accurately reflect the branching structure of rapeseed plants and provide structured training data for subsequent neural network modeling. The annotation data is stored in JSON format, including image information, key point coordinates and their hierarchical relationships.

This method combines botanical knowledge with data science technology to more accurately simulate the growth process of rapeseed. The complete code implementation and more details will be in other sections.
<img width="503" alt="image" src="https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/a42461f4-2967-4441-924c-c931366c8f12">
![image](https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/20839ebf-0fb3-4185-8478-b1d1d8a6e615)


