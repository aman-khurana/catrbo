## Objective Function Set Up 


Step 1: Create the docker file for your objective function

Step 2: Create a docker image using the docker file

> put any data required for the objective function under the data folder inside the root dir

`docker build -t objective_server .`

Step 3: Set the `PORT` variable in the objective_server.py file. Server will listen to this port inside the docker container.

Step 4: Create the container and map the docker `PORT` to the port number you want to listen to on the host.

`docker run -p 5000:8000 -d objective_server`

This will listen for requests on port 5000 on the host machine.
