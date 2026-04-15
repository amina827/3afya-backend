#!/usr/bin/env sh
set -e

echo "==> Running database migrations..."
python manage.py migrate --noinput

echo "==> Ensuring Django superuser exists..."
DJANGO_SUPERUSER_USERNAME="${DJANGO_SUPERUSER_USERNAME:-admin}" \
DJANGO_SUPERUSER_EMAIL="${DJANGO_SUPERUSER_EMAIL:-bdccoworking@gmail.com}" \
DJANGO_SUPERUSER_PASSWORD="${DJANGO_SUPERUSER_PASSWORD:-3afya@Admin2026!}" \
python manage.py shell -c "
import os
from django.contrib.auth import get_user_model
User = get_user_model()
u = os.environ['DJANGO_SUPERUSER_USERNAME']
e = os.environ['DJANGO_SUPERUSER_EMAIL']
p = os.environ['DJANGO_SUPERUSER_PASSWORD']
obj, created = User.objects.get_or_create(username=u, defaults={'email': e, 'is_staff': True, 'is_superuser': True})
obj.email = e
obj.is_staff = True
obj.is_superuser = True
obj.set_password(p)
obj.save()
print(('created' if created else 'updated') + ' superuser: ' + u)
"

echo "==> Collecting static files..."
python manage.py collectstatic --noinput

echo "==> Starting gunicorn on port ${PORT:-8000}..."
exec gunicorn core.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
