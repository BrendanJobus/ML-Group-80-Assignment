# ML-Group-Project-80
Group assignment for CSU-44061 Machine Learning based upon predicting housing prices depending on available input features.

## Download Project
To download the project run `git clone https://github.com/BrendanJobus/ML-Group-Project-80.git`.

## Running the Project
You can run the project through the dockerfile by doing `sudo docker compose up` if on linux, then by doing `sudo docker exec -it ml-group-project-80-app-1 bash` we are put into the image where we can run make scrape to start the scraping script. Remember to stop the script by running `sudo docker kill ml-group-project-80-app-1`.
