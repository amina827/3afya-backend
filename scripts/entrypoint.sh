#!/usr/bin/env sh
set -e

echo "==> Running database migrations..."
python manage.py migrate --noinput

echo "==> Ensuring Django superuser exists..."
python manage.py shell << 'EOF'
import os
from django.contrib.auth import get_user_model
User = get_user_model()
username = os.environ.get("DJANGO_SUPERUSER_USERNAME", "admin")
email = os.environ.get("DJANGO_SUPERUSER_EMAIL", "bdccoworking@gmail.com")
password = os.environ.get("DJANGO_SUPERUSER_PASSWORD", "3afya@Admin2026!")
obj, created = User.objects.get_or_create(username=username)
obj.email = email
obj.is_staff = True
obj.is_superuser = True
obj.set_password(password)
obj.save()
print(("Created" if created else "Updated") + " superuser:", username)
EOF

echo "==> Collecting static files..."
python manage.py collectstatic --noinput

echo "==> Starting gunicorn on port ${PORT:-8080}..."
exec gunicorn core.wsgi:application \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
