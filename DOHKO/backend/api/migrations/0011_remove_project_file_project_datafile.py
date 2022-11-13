# Generated by Django 4.1.3 on 2022-11-12 23:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0010_remove_project_dataname"),
    ]

    operations = [
        migrations.RemoveField(model_name="project", name="file",),
        migrations.AddField(
            model_name="project",
            name="dataFile",
            field=models.FileField(default=None, upload_to="datasets/"),
        ),
    ]
