The datasets used in this project are sourced from Kaggle.

**Main Dataset: COD10K — Camouflaged Object Detection Dataset**

Link: [https://www.kaggle.com/datasets/getcam/cod10k](https://www.kaggle.com/datasets/getcam/cod10k)

COD10K contains 10,000 images across 78 object categories, organized into 10 super-classes and divided into 5,066 camouflaged, 1,934 non-camouflaged, and 3,000 background images. The camouflaged super-categories span five major ecological environments — terrestrial, atmobios, aquatic, amphibian, and other — collected from photography websites using carefully designed search keywords. Each image is accompanied by rich annotations including object categories, bounding boxes, object-level masks, instance-level masks, edge annotations, and challenging attribute labels covering multiple objects, big/small objects, out-of-view objects, occlusions, and shape complexity. The dataset is split into 6,000 training images and 4,000 testing images, making it the largest and most densely annotated camouflaged object detection benchmark at the time of its release. COD10K is used as the primary training and evaluation dataset throughout this project, supporting segmentation, boundary supervision, and uncertainty estimation across all four backbone models.

**Cross-Validation Dataset: CAMO — Camouflaged Object Dataset**

Link: [https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset](https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset)

The CAMO dataset consists of 1,250 camouflaged images split into 1,000 training and 250 testing images, supplemented by 1,250 non-camouflaged images drawn from MS-COCO, each accompanied by objectness mask ground-truth annotations. The dataset covers two primary camouflage categories — naturally camouflaged objects and artificially camouflaged objects — where natural camouflage encompasses amphibians, birds, insects, mammals, reptiles, and underwater animals across diverse environments including ground, underwater, desert, forest, mountain, and snow, while artificial camouflage covers soldiers and human body art. CAMO is used exclusively for cross-dataset generalization evaluation to assess model robustness on a distribution distinct from the primary training data, following the same role as NSL-KDD in intrusion detection benchmarks.
