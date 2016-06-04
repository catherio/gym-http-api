gym-http-api
============

This project provides a local REST API to the [gym](https://github.com/openai/gym) open-source library, allowing development in languages other than python.

A python client is included, to demonstrate how to interact with the server. Contributions of clients in other languages are welcomed!


Installation
============

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/catherio/gym-http-api
    cd gym-http-api
    pip install -r requirements.txt


Getting started
============

This code is intended to be run locally by a single user. The server runs in python. You can implement your own HTTP clients using any language; a demo client written in python is provided to demonstrate the idea.

To start the server from the command line, run this:

    python gym_server.py

In a separate terminal, you can then try running the example agent and see what happens:

    python example_agent.py  

You can also write code like this to create your own client, and test it out by creating a new environment:

    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)
    exists = client.env_check_exists(instance_id)


Testing
============

This repository contains tests that can be run using the `nose2` framework. From a shell (such as bash) you can run nose2 directly:

    cd gym-http-api
    nose2


API specification
============

  * POST `/v1/envs/`
      * Create an instance of the specified environment
      * param: `env_id` -- gym environment ID string, such as 'CartPole-v0'
      * returns: `instance_id` -- a short identifier (such as '3c657dbc')
	    for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated

  * GET `/v1/envs/`
      * List all environments running on the server
	  * returns: `envs` -- dict mapping `instance_id` to `env_id` 
	    (e.g. `{'3c657dbc': 'CartPole-v0'}`) for every env on the server

  * POST `/v1/envs/<instance_id>/check_exists/`
  	  * Determine whether the specified instance_id corresponds to
	    a valid environment instance that has been created.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
	  * returns: `exists` -- True or False, indicating whether the given
        instance exists

  * POST `/v1/envs/<instance_id>/reset/`
      * Reset the state of the environment and return an initial
        observation.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
      * returns: `observation` -- the initial observation of the space
    
  * POST `/v1/envs/<instance_id>/step/`
      * Reset the state of the environment and return an initial
        observation.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
	  * param: `action` -- an action to take in the environment
      * returns: `observation` -- agent's observation of the current
        environment
      * returns: `reward` -- amount of reward returned after previous action
      * returns: `done` -- whether the episode has ended
      * returns: `info` -- a dict containing auxiliary diagnostic information

  * GET `/v1/envs/<instance_id>/action_space/`
      * Get information (name and dimensions/bounds) of the env's
        `action_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * GET `/v1/envs/<instance_id>/observation_space/`
      * Get information (name and dimensions/bounds) of the env's
        `observation_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * POST `/v1/envs/<instance_id>/monitor/start/`
      * Start monitoring
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * param: force (default=False) -- Clear out existing training
        data from this directory (by deleting every file
        prefixed with "openaigym.")
      * param: resume (default=False) -- Retain the training data
        already in this directory, which will be merged with
        our new data
      * (NOTE: the 'video_callable' parameter from the native
    env.monitor.start function is NOT implemented)

  * POST `/v1/envs/<instance_id>/monitor/close/`
      * Flush all monitor data to disk
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance 

  * POST `/v1/upload/`
      * Flush all monitor data to disk
      * param: `training_dir` -- A directory containing the results of a
        training run.
      * param: `api_key` -- Your OpenAI API key
      * param: `algorithm_id` (default=None) -- An arbitrary string
        indicating the paricular version of the algorithm
        (including choices of parameters) you are running.
      * param: `writeup` (default=None) -- A Gist URL (of the form
        https://gist.github.com/<user>/<id>) containing your
        writeup for this evaluation.
   
  * POST `/v1/shutdown/`
      * Shut down the server