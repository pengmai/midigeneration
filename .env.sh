# Set the username, password, and database name appropriately.
DB_USER=midgenuser
DB_PASS=devpassword
DB_NAME=midigendb

export DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost/${DB_NAME}
