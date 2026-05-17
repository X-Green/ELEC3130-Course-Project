# Background
I am taking ELEC3130, a course on Digital Image Processing. This course introduces methods to process images on a computer. Topics include the formation and quantification of digital images, morphological image processing, image enhancement in the spatial and frequency domain, image restoration, color image processing, image compression, image segmentation, object recognition and face detection. Students are expected to obtain knowledge of digital image processing, machine learning algorithms, and hands‐on experience in programming and presentation skills through the final project. This course is mathematics oriented. It requires basic knowledge of linear algebra, calculus and linear filtering. Familiarity with the programming language MATLAB is needed (actually python is also ok in projects).

Syllabus: 

Week 1           Introduction

Week 2           Transforms

Week 3           Morphological Image Processing

Week 4           Image Enhancement in the Spatial Domain

Week 5           Image Enhancement in the Frequency Domain

Week 6           Image Restoration

Week 7           Motion Deblurring using Wiener Filter

Week 8           Color Image Processing

Week 9           Image Compression: JPEG as an Example

Week 10          Image Segmentation: Canny Edge Detector

Week 11          Object Recognition: Classical Approaches

Week 12          Face Recognition: Eigen Faces

Week 13          Final Review and Outlook: Deep Learning

The final project will be a group project, requiring 3-5 students to work together. Individual projects or projects with only two members or over 5 members will not be accepted.

Regarding the topic, each group can choose a topic based on their interests. If your group cannot find a suitable topic, you may opt for a default topic. A tutorial will be held two weeks(tentatively) before the last class to assist you in this process. 

It is essential that each group clearly defines the roles of each member for the chosen topic. There should be no overlap in responsibilities, and each member's role must connect to create a cohesive workflow focused on a specific image processing application. It is not acceptable for several members to work on the same type of processing using different methods.

After the last class, a Zoom presentation will take place during spring term examination period. Each group will have 9-15 minutes to present, averaging about 3 minutes per person. The presentation should tell a complete story, and it will account for 20% of your total grade.

IMPORTANT: The project should be completed using traditional image processing techniques, without the use of deep learning. The project should be mostly based on the content covered in the course, and it should not involve any deep learning methods.

# What we have came up with

Automated PCB defect detection (not using deep learning).

1. Boarder Alignment by Edge Detection   
2. Spatial Domain & Registration, outputing aligned and color-matched images 
3. Frequency Domain FIltering, isolating fine wiring patterns and supress background 
4. Morphological Subtraction to remain defects 
5. Classification and Detection: Use graph based method to segmentify, classify the remaining defects.

# What you need to do

Brainstorm and research on the topic, and find out what methods we can use for each step.