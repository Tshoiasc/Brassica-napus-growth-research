# Brassica-napus-growth-research

**DATA ANNOTATION**

This project aims to use neural networks to model and simulate the growth of rapeseed plants. The data annotation adopts a hierarchical recursive method based on the biological growth characteristics of rapeseed:

Start annotation from the stem root, and mark the leaves as potential branch points.

The main trunk is marked as "stem", the first-level branch is marked as "branch_1", and so on.

If a leaf develops a branch, the corresponding level is marked at the top inflorescence of the branch.

Repeat this process for each branch until the top flower of the stem is marked.

Like the structure below:

stem
├── branch_1
│           ├── branch_2
│           │           └── branch_3 (if available)
│           ├── branch_2
│           └── ...
├── branch_1
│          ├── branch_2
│          └── ...
└── ...

Finally, a tree structure is formed, branch_1 belongs to stem, branch_2 belongs to branch_1, and so on.
This annotation method can accurately reflect the branching structure of rapeseed plants and provide structured training data for subsequent neural network modeling. The annotation data is stored in JSON format, including image information, key point coordinates and their hierarchical relationships.

This method combines botanical knowledge with data science technology to more accurately simulate the growth process of rapeseed. The complete code implementation and more details will be in other sections.
<img width="503" alt="image" src="https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/a42461f4-2967-4441-924c-c931366c8f12">
![image](https://github.com/Tshoiasc/Brassica-napus-growth-research/assets/30382941/20839ebf-0fb3-4185-8478-b1d1d8a6e615)
