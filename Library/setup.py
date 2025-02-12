from setuptools import find_packages, setup

setup(
    name="LibMCMC",
    version="0.2",
    packages=find_packages(),  # This will find all packages
    package_data={
        'LibMCMC': ['*.py'],  # This ensures .py files are included
    },
    include_package_data=True,  # This ensures all data files are included
)
