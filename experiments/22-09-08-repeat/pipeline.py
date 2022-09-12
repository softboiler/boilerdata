from boilerdata.pipeline import pipeline
from boilerdata.models.project import Project

proj = Project.get_project("experiments/22-09-08-repeat/config/project.yaml")
pipeline(proj)
...
