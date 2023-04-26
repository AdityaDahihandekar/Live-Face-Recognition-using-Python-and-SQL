# Live-Face-Recognition-using-Python-and-SQL

This project involves the use of Haar Cascade in python to screen and detect faces live using the inbuit camera on laptop/PC.
Here, MATLAB is used to convert and check RGB images to grayscale images.
MySQL Workbench is used to smoothly checking and managing the schema and database created

Pre-requisites -
1) It is advised to create a folder titled "Face Recognition" to keep all the files in one place.
2) In the folder "Face recognition", download the haar cascade file and make another folder titled "training_data" under which sub-folders titled "0", "1" ..etc will be used to save gray images of the faces of person which are to be used to detect later on.
3) Also, in the folder "Face Recognition", save the python script "Live_Face_Recognition".
4) All images used in training data are ".jpg" type.
5) Set-up of a MySQL server is recommended with schema titled "newdb" and database titled "database1" to be created. File for which hs been attached.
6) Inbuilt camera is required for the code to run and detect faces.

Things to be edited as per user -
1) PC name to be typed in-place of "Username" in directory change command "cd" in MATLAB scripts. If the line gives error, change manually to "Training_data" file created earlier.
2) In python script "Live_Face_Recognition", "name1"-"name2"-"name3" is to be swapped with person's name which is to be detected.
3) In the python script "Live_Face_Recognition", user,database and password for the MySQL.connect part of the code to be change accordingly.

How it works - 
1) Firstly the user makes sure that there are sufficient number of images present of the person who is to detected by the camera. It is advised that atleast 10 pictures are used. These images are stored in the "training_data" folder.
2) The MATLAB script "convert_x_number_of_images_to_grayscale.mlx" is run and images are converted to grayscale and stored in the same folder.
3) Now move the images of person 1 in sub-folder 0, person 2 in su-folder 1 and so on.
4) Inside folder 0, rename the images as "person1_01", "person1_02", "person3_03" and so on. Similarly, in folder 1 rename the images as "person2_01", "person2_02" and so on
5) Run the MATLAB script rgb_or_gray to check if all images are grayscale or not, for surety.
6) After setting up MySQL server, either in the command line or in workbench, create schema titled "newdb" and then run the SQL script attached.
7) Now run the python script "Live_Face_Recognition" and check the SQL database for results by re-running the "SELECT * FROM database1;" command.
