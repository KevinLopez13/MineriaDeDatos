# Generated by Django 4.1.3 on 2022-11-12 23:38

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0013_alter_project_datafile"),
    ]

    operations = [
        migrations.RemoveField(model_name="project", name="dataFile",),
    ]