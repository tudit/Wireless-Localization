**************************
To Run Falcon REST Server:<br/>

*Follow these steps to setup gunicorn and falcon rest code:<br/>
https://falcon.readthedocs.io/en/stable/user/tutorial.html<br/>
*$ gunicorn falcon_server<br/>

Setup:<br/>
*Follow these steps to setup gunicorn and falcon rest code:<br/>
    https://falcon.readthedocs.io/en/stable/user/tutorial.html<br/>

Running Falcon:<br/>
*To run normally:<br/>
    $ gunicorn falcon_server<br/>
*Run with the below options, since localization takes some time<br/>
    $gunicorn -t 3600 --access-logfile - falcon_server<br/>
**************************
To run the client:<br/>
    $python3 client.py<br/>
******************
Notes:<br/>
*To change configuration for the project like K, Number of transmitters, Number of blocks, etc. refer utils.py script<br/>
******
