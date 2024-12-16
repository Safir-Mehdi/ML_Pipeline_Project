from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT = '-e .'

# Getting packages from requirement.txt 
def get_requirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [line.replace("\n", "") for line in requirements]

        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)
    
    return requirements

# Setting up the 
setup(
    name='ML_Pipeline_Project',
    version='0.0.1',
    description='A machine learning project',
    author='Safir Mehdi',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)