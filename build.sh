#!/usr/bin/env bash

pip install -r requirements.txt
python ckd_web/manage.py collectstatic --noinput
python ckd_web/manage.py migrate
