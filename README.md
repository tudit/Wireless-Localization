**************************
To Run Falcon REST Server:
**************************

*Follow these steps to setup gunicorn and falcon rest code:
https://falcon.readthedocs.io/en/stable/user/tutorial.html
*$ gunicorn falcon_server

Setup:
*****
*Follow these steps to setup gunicorn and falcon rest code:<br/>
    https://falcon.readthedocs.io/en/stable/user/tutorial.html<br/>

Running Falcon:
***************
*To run normally:
    $ gunicorn falcon_server
*Run with the below options, since localization takes some time
    $gunicorn -t 3600 --access-logfile - falcon_server

******************
To run the client:
******************
$python3 client.py

******
Notes:
******
To change configuration for the project like K, Number of transmitters, Number of blocks, etc. refer utils.py script
