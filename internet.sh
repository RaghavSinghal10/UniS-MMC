#!/bin/bash

ldapId="190020124"
accessToken="ebc22c2492724f3917ec9b631e3a0e5d"

/usr/bin/curl -s --location-trusted -u $ldapId:$accessToken "https://internet-sso.iitb.ac.in/login.php" | \
grep -q window.location.href && \
echo 'Logged in!' || \
echo 'Something is wrong or already logged in!'